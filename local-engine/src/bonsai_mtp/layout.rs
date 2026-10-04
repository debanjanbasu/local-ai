use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;

use super::{ATTENTION, FFN, HEAD_DIM, KV, QUERY_GATE, WIDTH, invalid};

pub(super) const SAFETENSORS_MAX_HEADER: u64 = 1 << 20;

/// Pinned head layout: `(name, shape)`, every tensor BF16.
pub(super) const TENSORS: [(&str, &[usize]); 15] = [
    ("mtp.fc.weight", &[WIDTH, 2 * WIDTH]),
    ("mtp.pre_fc_norm_embedding.weight", &[WIDTH]),
    ("mtp.pre_fc_norm_hidden.weight", &[WIDTH]),
    ("mtp.layers.0.input_layernorm.weight", &[WIDTH]),
    ("mtp.layers.0.post_attention_layernorm.weight", &[WIDTH]),
    ("mtp.layers.0.self_attn.q_proj.weight", &[QUERY_GATE, WIDTH]),
    ("mtp.layers.0.self_attn.k_proj.weight", &[KV, WIDTH]),
    ("mtp.layers.0.self_attn.v_proj.weight", &[KV, WIDTH]),
    ("mtp.layers.0.self_attn.o_proj.weight", &[WIDTH, ATTENTION]),
    ("mtp.layers.0.self_attn.q_norm.weight", &[HEAD_DIM]),
    ("mtp.layers.0.self_attn.k_norm.weight", &[HEAD_DIM]),
    ("mtp.layers.0.mlp.gate_proj.weight", &[FFN, WIDTH]),
    ("mtp.layers.0.mlp.up_proj.weight", &[FFN, WIDTH]),
    ("mtp.layers.0.mlp.down_proj.weight", &[WIDTH, FFN]),
    ("mtp.norm.weight", &[WIDTH]),
];

/// Byte range of one tensor inside the safetensors data region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRange {
    pub name: &'static str,
    pub start: usize,
    pub end: usize,
}

/// Checked safetensors layout; independent of any GPU so it is unit-testable.
#[derive(Debug)]
pub struct SafetensorsLayout {
    pub data_start: usize,
    pub tensors: Vec<TensorRange>,
}

impl SafetensorsLayout {
    /// `header` is the JSON header, `data_len` the byte count after it.
    pub fn parse(header: &[u8], data_len: usize) -> crate::Result<Self> {
        #[derive(Deserialize)]
        struct Entry {
            dtype: String,
            shape: Vec<usize>,
            data_offsets: [usize; 2],
        }
        let mut entries: HashMap<String, serde_json::Value> = serde_json::from_slice(header)?;
        entries.remove("__metadata__");
        if entries.len() != TENSORS.len() {
            return invalid(format!(
                "MTP head must contain exactly {} tensors, found {}",
                TENSORS.len(),
                entries.len()
            ));
        }
        let mut tensors = Vec::with_capacity(TENSORS.len());
        for (name, shape) in TENSORS {
            let value = entries
                .remove(name)
                .ok_or_else(|| crate::Error::MissingTensor(name.to_owned()))?;
            let entry: Entry = serde_json::from_value(value)?;
            let elements = shape.iter().product::<usize>();
            if entry.dtype != "BF16" || entry.shape != shape {
                return invalid(format!("MTP tensor {name} must be BF16 {shape:?}"));
            }
            let [start, end] = entry.data_offsets;
            if end <= start
                || end - start != elements * 2
                || end > data_len
                || !start.is_multiple_of(2)
            {
                return invalid(format!("MTP tensor {name} has an invalid byte range"));
            }
            tensors.push(TensorRange { name, start, end });
        }
        let mut sorted: Vec<&TensorRange> = tensors.iter().collect();
        sorted.sort_by_key(|range| range.start);
        let contiguous = sorted.windows(2).all(|pair| pair[0].end <= pair[1].start);
        if !contiguous || sorted.last().is_none_or(|last| last.end != data_len) {
            return invalid("MTP tensor ranges overlap or leave unused bytes");
        }
        Ok(Self {
            data_start: 0,
            tensors,
        })
    }

    pub fn range(&self, name: &str) -> crate::Result<&TensorRange> {
        self.tensors
            .iter()
            .find(|range| range.name == name)
            .ok_or_else(|| crate::Error::MissingTensor(name.to_owned()))
    }
}

/// Read and validate the safetensors framing of an MTP head file.
pub fn open_layout(path: &Path) -> crate::Result<(Mmap, SafetensorsLayout)> {
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    if length < 8 {
        return invalid("MTP head file is too short for a safetensors header");
    }
    // SAFETY: read-only mapping of a file we never write. Concurrent external
    // modification is out of scope, as for the GGUF mapping.
    #[allow(unsafe_code)]
    let map = unsafe { MmapOptions::new().map(&file)? };
    let header_len = u64::from_le_bytes(
        map[..8]
            .try_into()
            .map_err(|_| crate::Error::InvalidFormat("safetensors header".into()))?,
    );
    if header_len == 0 || header_len > SAFETENSORS_MAX_HEADER || header_len + 8 > length {
        return invalid("MTP head safetensors header length is invalid");
    }
    let data_start = 8 + header_len as usize;
    let data_len = usize::try_from(length - 8 - header_len)
        .map_err(|_| crate::Error::InvalidFormat("MTP head exceeds address space".into()))?;
    let mut layout = SafetensorsLayout::parse(&map[8..data_start], data_len)?;
    layout.data_start = data_start;
    Ok((map, layout))
}
