//! Bounded, read-only access to the pinned Bonsai GGUF profile.
//!
//! The native Metal backend runs the checkpoint straight from this file:
//! `metal_tensor` wraps 16-KiB-aligned windows of the immutable mapping as
//! shared Metal buffers without copying. `--export-index` describes the same
//! checked tensor layout, Hadamard signs and tokenizer for the F64 reference.
//! The complete file is virtually mapped for
//! checked metadata and CPU views; this does not make it resident.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use local_metal::buffer::MetalBuffer;
use memmap2::{Mmap, MmapOptions};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::bonsai_tokenizer::TokenizerDefinition;

mod cursor;
mod hadamard;
mod metal_tensor;
mod profile;

pub use self::hadamard::HadamardMetadata;
pub use self::metal_tensor::BonsaiMetalTensor;
pub use self::profile::validate_profile;

use self::cursor::{
    Cursor, array_i32, array_strings, scalar_bool, scalar_string, scalar_u32, skip_value,
    value_cursor,
};
use self::hadamard::parse_hadamard;
use self::metal_tensor::metal_window;

/// The pinned checkpoint the runtime runs directly; also the source
/// `--export-index`/`--tokenize` read.
///
/// Shipped builds keep this repository-relative so a relocated install
/// resolves `./models` beside itself; `resources::discover_model` treats it as
/// the first candidate. Test fixtures need the opposite: `cargo test -p
/// local-engine` runs with `local-engine/` as the working directory, so the
/// same relative string cannot resolve. The test build therefore anchors the
/// identical file to the workspace root, where nothing joins it onto a working
/// directory before opening it.
#[cfg(not(test))]
pub const DEFAULT_BONSAI_GGUF: &str = "models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf";
#[cfg(test)]
pub const DEFAULT_BONSAI_GGUF: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
);

/// Metal shared buffers wrap whole 16-KiB pages of the mapping.
const METAL_PAGE: usize = 16 * 1024;

/// Pinned Bonsai 2 27B profile the GGUF checks enforce.
pub const LAYERS: usize = 64;
pub const WIDTH: usize = 5120;
pub const FFN: usize = 17_408;
pub const VOCAB: usize = 248_320;
pub const FULL_INTERVAL: usize = 4;
pub const TRAINING_CONTEXT: usize = 262_144;

/// `PTQ1_0` packs 128 signed trits plus one FP16 scale into 28 bytes.
pub const PTQ1_BLOCK_ELEMENTS: usize = 128;
pub const PTQ1_BLOCK_BYTES: usize = 28;

/// Which side of a PTQ1 matrix the checkpoint's signed Hadamard rotation is
/// on; the native kernels' own enum so directions flow to them unchanged.
pub use local_metal::bonsai::HadamardDirection;

const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_METADATA: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_DIMS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BonsaiTensorType {
    F32,
    Bf16,
    Ptq1,
}

impl BonsaiTensorType {
    fn from_wire(value: u32) -> crate::Result<Self> {
        match value {
            0 => Ok(Self::F32),
            30 => Ok(Self::Bf16),
            143 => Ok(Self::Ptq1),
            _ => Err(crate::Error::InvalidFormat(format!(
                "unsupported Bonsai tensor type {value}"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BonsaiTensor {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: BonsaiTensorType,
    offset: u64,
    bytes: u64,
}

struct Descriptor {
    name: String,
    dims: Vec<u64>,
    ty: BonsaiTensorType,
    offset: u64,
}

impl BonsaiTensor {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    #[must_use]
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    #[must_use]
    pub const fn tensor_type(&self) -> BonsaiTensorType {
        self.tensor_type
    }
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    #[must_use]
    pub(crate) const fn relative_offset(&self) -> u64 {
        self.offset
    }
}

#[derive(Clone)]
struct MetadataValue {
    ty: u32,
    payload: std::ops::Range<usize>,
    element_ty: Option<u32>,
    count: Option<u64>,
}

/// An inspected GGUF and its retained inode. The mapping has no mutable API.
pub struct BonsaiPackage {
    file: File,
    map: Arc<Mmap>,
    tensors: Vec<BonsaiTensor>,
    by_name: HashMap<String, usize>,
    data_start: u64,
    metadata_count: u64,
    metadata: HashMap<String, MetadataValue>,
    hadamard: HadamardMetadata,
}

impl BonsaiPackage {
    /// The inspected checkpoint inode must remain immutable for all returned views.
    /// Integrity hashes are verified separately by the pinned artifact installer.
    #[allow(unsafe_code, clippy::too_many_lines)]
    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < 24 {
            return invalid("truncated GGUF header");
        }
        // MAP_PRIVATE/read-only: callers must not mutate the retained checkpoint inode.
        let map = Arc::new(unsafe { MmapOptions::new().map_copy_read_only(&file)? });
        let mut cursor = Cursor::new(&map, MAX_HEADER_BYTES);
        if cursor.take(4)? != b"GGUF" {
            return invalid("bad GGUF magic");
        }
        if cursor.u32()? != 3 {
            return invalid("only little-endian GGUF v3 is supported");
        }
        let tensor_count = cursor.u64()?;
        let metadata_count = cursor.u64()?;
        if tensor_count == 0 || tensor_count > MAX_TENSORS || metadata_count > MAX_METADATA {
            return invalid("GGUF count exceeds inspection bound");
        }

        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = cursor.string()?.to_owned();
            if key.is_empty() {
                return invalid("empty GGUF metadata key");
            }
            let ty = cursor.u32()?;
            let start = cursor.pos;
            let (element_ty, count) = skip_value(&mut cursor, ty)?;
            if metadata
                .insert(
                    key.clone(),
                    MetadataValue {
                        ty,
                        payload: start..cursor.pos,
                        element_ty,
                        count,
                    },
                )
                .is_some()
            {
                return invalid(format!("duplicate GGUF metadata key {key}"));
            }
        }

        let alignment = match metadata.get("general.alignment") {
            None => 32_u32,
            Some(value) if value.ty == 4 => scalar_u32(&map, value)?,
            Some(_) => return invalid("general.alignment must be u32"),
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return invalid("GGUF alignment must be a nonzero power of two");
        }
        let architecture = metadata
            .get("general.architecture")
            .ok_or_else(|| crate::Error::InvalidFormat("missing general.architecture".into()))?;
        if scalar_string(&map, architecture)? != "qwen35" {
            return invalid("Bonsai reader requires general.architecture=qwen35");
        }

        // Grow only after actually reading descriptors, never reserve from an
        // untrusted count before proving that its bytes exist in the header.
        let mut descriptors = Vec::new();
        let mut names = HashSet::new();
        for _ in 0..tensor_count {
            let name = cursor.string()?.to_owned();
            if name.is_empty() || !names.insert(name.clone()) {
                return invalid(format!("duplicate GGUF tensor {name}"));
            }
            let n_dims = cursor.u32()?;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return invalid(format!("invalid dimension count for {name}"));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let dim = cursor.u64()?;
                if dim == 0 {
                    return invalid(format!("zero dimension in {name}"));
                }
                dims.push(dim);
            }
            let ty = BonsaiTensorType::from_wire(cursor.u32()?)?;
            let offset = cursor.u64()?;
            descriptors.push(Descriptor {
                name,
                dims,
                ty,
                offset,
            });
        }
        let data_start_usize = align_up(cursor.pos, alignment as usize)?;
        let data_start = data_start_usize as u64;
        if data_start > file_len {
            return invalid("GGUF tensor data starts beyond EOF");
        }

        let mut tensors = Vec::with_capacity(descriptors.len());
        let mut by_name = HashMap::with_capacity(descriptors.len());
        let mut expected_offset = 0_u64;
        for descriptor in descriptors {
            if descriptor.offset != expected_offset || descriptor.offset % u64::from(alignment) != 0
            {
                return invalid(format!(
                    "tensor {} has non-contiguous or unaligned offset",
                    descriptor.name
                ));
            }
            let elements = checked_product(&descriptor.dims)?;
            let bytes = match descriptor.ty {
                BonsaiTensorType::F32 => elements.checked_mul(4),
                BonsaiTensorType::Bf16 => elements.checked_mul(2),
                BonsaiTensorType::Ptq1 => {
                    if !descriptor.dims[0].is_multiple_of(PTQ1_BLOCK_ELEMENTS as u64) {
                        return invalid(format!(
                            "PTQ1 tensor {} has incompatible contiguous width",
                            descriptor.name
                        ));
                    }
                    elements
                        .checked_div(PTQ1_BLOCK_ELEMENTS as u64)
                        .and_then(|n| n.checked_mul(PTQ1_BLOCK_BYTES as u64))
                }
            }
            .ok_or_else(|| crate::Error::InvalidFormat("tensor extent overflow".into()))?;
            let end = data_start
                .checked_add(descriptor.offset)
                .and_then(|v| v.checked_add(bytes))
                .ok_or_else(|| crate::Error::InvalidFormat("tensor file range overflow".into()))?;
            if end > file_len {
                return invalid(format!("truncated payload for {}", descriptor.name));
            }
            let padded = align_up_u64(bytes, u64::from(alignment))?;
            expected_offset = descriptor.offset.checked_add(padded).ok_or_else(|| {
                crate::Error::InvalidFormat("tensor padded offset overflow".into())
            })?;
            let tensor = BonsaiTensor {
                name: descriptor.name,
                dimensions: descriptor.dims,
                tensor_type: descriptor.ty,
                offset: descriptor.offset,
                bytes,
            };
            by_name.insert(tensor.name.clone(), tensors.len());
            tensors.push(tensor);
        }
        let hadamard = parse_hadamard(&map, &metadata, &tensors, &by_name)?;
        Ok(Self {
            file,
            map,
            tensors,
            by_name,
            data_start,
            metadata_count,
            metadata,
            hadamard,
        })
    }

    fn metadata_value(&self, key: &str) -> crate::Result<&MetadataValue> {
        self.metadata
            .get(key)
            .ok_or_else(|| crate::Error::InvalidFormat(format!("missing metadata {key}")))
    }

    pub(crate) fn metadata_u32(&self, key: &str) -> crate::Result<u32> {
        scalar_u32(&self.map, self.metadata_value(key)?)
    }

    pub(crate) fn metadata_f32(&self, key: &str) -> crate::Result<f32> {
        let value = self.metadata_value(key)?;
        if value.ty != 6 {
            return invalid("metadata type mismatch: expected f32");
        }
        Ok(f32::from_bits(value_cursor(&self.map, value).u32()?))
    }

    pub(crate) fn metadata_bool(&self, key: &str) -> crate::Result<bool> {
        scalar_bool(&self.map, self.metadata_value(key)?)
    }

    pub(crate) fn metadata_string(&self, key: &str) -> crate::Result<&str> {
        scalar_string(&self.map, self.metadata_value(key)?)
    }

    pub(crate) fn metadata_i32_array(&self, key: &str) -> crate::Result<Vec<i32>> {
        array_i32(&self.map, self.metadata_value(key)?)
    }

    pub(crate) fn metadata_strings(&self, key: &str) -> crate::Result<Vec<String>> {
        array_strings(&self.map, self.metadata_value(key)?)
    }

    #[must_use]
    pub fn tensors(&self) -> &[BonsaiTensor] {
        &self.tensors
    }
    #[must_use]
    pub const fn metadata_count(&self) -> u64 {
        self.metadata_count
    }
    #[must_use]
    pub const fn data_start(&self) -> u64 {
        self.data_start
    }
    #[must_use]
    pub const fn hadamard(&self) -> &HadamardMetadata {
        &self.hadamard
    }

    /// Describe the checked package without copying or decoding its tensor payloads.
    #[doc(hidden)]
    pub fn export_index(&self, source: &Path) -> crate::Result<serde_json::Value> {
        let source = source.canonicalize()?;
        let source_bytes = self.file.metadata()?.len();
        let mut tensors = serde_json::Map::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let offset = self
                .data_start
                .checked_add(tensor.relative_offset())
                .ok_or_else(|| crate::Error::InvalidFormat("tensor offset overflow".into()))?;
            let mut shape = tensor.dimensions.clone();
            shape.reverse();
            let (dtype, encoding) = match tensor.tensor_type {
                BonsaiTensorType::F32 => ("f32", None),
                BonsaiTensorType::Bf16 => ("bf16", None),
                BonsaiTensorType::Ptq1 => ("u8", Some("ptq1_0")),
            };
            let mut record = serde_json::json!({
                "dtype": dtype, "shape": shape, "offset": offset, "bytes": tensor.bytes,
            });
            if let Some(encoding) = encoding {
                record["encoding"] = serde_json::json!(encoding);
            }
            tensors.insert(tensor.name.clone(), record);
        }
        let signs = self
            .hadamard
            .all_signs()
            .map(|(width, values)| (width.to_string(), serde_json::json!(values)))
            .collect::<serde_json::Map<_, _>>();
        // The complete tokenizer definition, so a consumer of the index never
        // has to parse GGUF metadata itself.
        let tokenizer = serde_json::to_value(TokenizerDefinition::from_package(self)?)?;
        Ok(serde_json::json!({
            "schema_version": 1,
            "source": {"path": source, "bytes": source_bytes},
            "tokenizer": tokenizer,
            "profile": {
                "layers": 64, "width": 5120, "ffn": 17_408, "vocab": 248_320,
                "full_attention_interval": 4, "training_context": 262_144,
                "epsilon": 1e-6, "rope_base": 10_000_000, "rope_dimensions": 64,
            },
            "tensors": tensors,
            "hadamard_signs": signs,
        }))
    }
    pub fn tensor(&self, name: &str) -> crate::Result<&BonsaiTensor> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| crate::Error::MissingTensor(name.into()))
    }
    pub fn bytes(&self, name: &str) -> crate::Result<&[u8]> {
        let tensor = self.tensor(name)?;
        let start = usize::try_from(
            self.data_start
                .checked_add(tensor.offset)
                .ok_or_else(|| crate::Error::InvalidFormat("tensor offset overflow".into()))?,
        )
        .map_err(|_| crate::Error::InvalidFormat("tensor offset exceeds address space".into()))?;
        let bytes = usize::try_from(tensor.bytes)
            .map_err(|_| crate::Error::InvalidFormat("tensor size exceeds address space".into()))?;
        Ok(&self.map[start..start + bytes])
    }

    /// Bind one tensor for the native backend without copying when the
    /// page-aligned window fits the device's buffer limit and the file.
    #[allow(unsafe_code)]
    pub fn metal_tensor(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        name: &str,
    ) -> crate::Result<BonsaiMetalTensor> {
        let tensor = self.tensor(name)?.clone();
        let absolute = usize::try_from(
            self.data_start
                .checked_add(tensor.offset)
                .ok_or_else(|| crate::Error::InvalidFormat("tensor offset overflow".into()))?,
        )
        .map_err(|_| crate::Error::InvalidFormat("tensor offset exceeds address space".into()))?;
        let bytes = usize::try_from(tensor.bytes)
            .map_err(|_| crate::Error::InvalidFormat("tensor size exceeds address space".into()))?;
        let natural_alignment = match tensor.tensor_type {
            BonsaiTensorType::F32 => 4,
            BonsaiTensorType::Bf16 | BonsaiTensorType::Ptq1 => 2,
        };
        if let Some(window) =
            metal_window(absolute, bytes, self.map.len(), device.maxBufferLength())?
            && absolute.is_multiple_of(natural_alignment)
        {
            let pointer = NonNull::new(self.map[window.start..].as_ptr().cast_mut())
                .ok_or_else(|| crate::Error::InvalidFormat("empty Bonsai mapping".into()))?;
            let owner: Arc<dyn Send + Sync> = self.map.clone();
            // SAFETY: the validated window contains whole readable pages. The
            // native buffer's deallocator retains this immutable mapping owner.
            let buffer =
                unsafe { MetalBuffer::from_bytes_no_copy(device, pointer, window.len(), owner) }?;
            Ok(BonsaiMetalTensor {
                buffer,
                offset: absolute - window.start,
                tensor,
                copied: false,
            })
        } else {
            let buffer = MetalBuffer::from_slice(device, &self.map[absolute..absolute + bytes])?;
            Ok(BonsaiMetalTensor {
                buffer,
                offset: 0,
                tensor,
                copied: true,
            })
        }
    }
}

fn checked_product(dims: &[u64]) -> crate::Result<u64> {
    dims.iter().try_fold(1_u64, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| crate::Error::InvalidFormat("tensor element count overflow".into()))
    })
}
fn align_up(value: usize, alignment: usize) -> crate::Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or_else(|| crate::Error::InvalidFormat("alignment overflow".into()))
}
fn align_up_u64(value: u64, alignment: u64) -> crate::Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or_else(|| crate::Error::InvalidFormat("alignment overflow".into()))
}

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidFormat(message.into()))
}

#[cfg(test)]
mod tests;
