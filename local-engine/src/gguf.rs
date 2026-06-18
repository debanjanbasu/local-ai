use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GGUFType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ty: GGUFType,
    pub shape: Vec<u64>,
    pub offset: usize,
    pub bytes: usize,
}

impl TensorInfo {
    #[must_use]
    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape.iter().map(|&d| d as usize).collect()
    }
    #[must_use]
    pub const fn elem_bytes(&self) -> usize {
        match self.ty {
            GGUFType::F32 => 4,
            GGUFType::F16 => 2,
            _ => 1,
        }
    }
    #[must_use]
    pub fn compute_bytes(&self) -> usize {
        let total: u64 = self.shape.iter().product();
        if self.ty == GGUFType::F32 || self.ty == GGUFType::F16 {
            (total * self.elem_bytes() as u64) as usize
        } else {
            self.ty.tensor_bytes(total as usize)
        }
    }
}

#[derive(Debug, Clone)]
pub struct GGUFModel {
    pub version: u32,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub data_offset: usize,
    pub path: std::path::PathBuf,
}

impl GGUFModel {
    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref();
        let mut raw = Vec::new();
        File::open(path)
            .map_err(|e| crate::Error::Io(e.to_string()))?
            .read_to_end(&mut raw)
            .map_err(|e| crate::Error::Io(e.to_string()))?;
        Self::parse(&raw, path)
    }
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }
    #[must_use]
    pub fn tensor_data<T: bytemuck::Pod>(&self, name: &str) -> Option<&[T]> {
        let info = self.tensor_info(name)?;
        // Tensor offsets in the GGUF header are relative to the aligned data
        // section, so add the data-section base.
        let start = self.data_offset + info.offset;
        let end = start + info.bytes;
        let slice = raw_slice(&self.path, start, end)?;
        Some(bytemuck::cast_slice(slice))
    }
    fn parse(raw: &[u8], path: &Path) -> crate::Result<Self> {
        if raw.len() < 24 {
            return Err(crate::Error::InvalidFormat("GGUF too short".into()));
        }
        if &raw[0..4] != b"GGUF" {
            return Err(crate::Error::InvalidFormat("bad magic".into()));
        }
        let version = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let tensor_count = u64::from_le_bytes([
            raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
        ]);
        let kv_count = u64::from_le_bytes([
            raw[16], raw[17], raw[18], raw[19], raw[20], raw[21], raw[22], raw[23],
        ]);
        let mut off = 24usize;
        for _ in 0..kv_count {
            off = skip_kv(raw, off)?;
        }
        let mut tensors: BTreeMap<String, TensorInfo> = BTreeMap::new();
        for _t in 0..tensor_count {
            let name = read_string(raw, &mut off)?;
            let n_dims =
                u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
            off += 4;
            let mut shape: Vec<u64> = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u64::from_le_bytes([
                    raw[off],
                    raw[off + 1],
                    raw[off + 2],
                    raw[off + 3],
                    raw[off + 4],
                    raw[off + 5],
                    raw[off + 6],
                    raw[off + 7],
                ]));
                off += 8;
            }
            let ty_raw = u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            off += 4;
            let ty = GGUFType::from_u32(ty_raw).unwrap_or_else(|| {
                eprintln!("warning: unknown GGUF tensor type {ty_raw} for tensor '{name}', treating as F16");
                GGUFType::F16
            });
            let file_offset = u64::from_le_bytes([
                raw[off],
                raw[off + 1],
                raw[off + 2],
                raw[off + 3],
                raw[off + 4],
                raw[off + 5],
                raw[off + 6],
                raw[off + 7],
            ]) as usize;
            off += 8;
            let total_elems: u64 = shape.iter().product();
            let bytes = ty.tensor_bytes(total_elems as usize);
            let name = name.trim_end_matches('\0').to_string();
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    ty,
                    shape,
                    offset: file_offset,
                    bytes,
                },
            );
        }
        // The data section is aligned to `general.alignment` (default 32).
        let data_offset = (off + 31) & !31;
        Ok(Self {
            version,
            tensors,
            data_offset,
            path: path.to_path_buf(),
        })
    }
}

impl GGUFType {
    /// Map a GGUF tensor-type discriminant to its [`GGUFType`].
    ///
    /// Returns `None` for unrecognized discriminants.
    #[must_use]
    pub const fn from_u32(raw: u32) -> Option<Self> {
        let ty = match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            31 => Self::Q4_0_4_4,
            32 => Self::Q4_0_4_8,
            33 => Self::Q4_0_8_8,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            40 => Self::NVFP4,
            41 => Self::Q1_0,
            _ => return None,
        };
        Some(ty)
    }

    #[must_use]
    pub const fn elem_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            _ => 1,
        }
    }

    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub const fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
            Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ4_NL
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M => 256,
            Self::TQ1_0 | Self::TQ2_0 => 256,
            Self::Q1_0 => 128,
            Self::Q4_0_4_4 | Self::Q4_0_4_8 | Self::Q4_0_8_8 => 32,
            Self::MXFP4 => 32,
            Self::NVFP4 => 64,
            _ => 1,
        }
    }

    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub const fn tensor_bytes(&self, num_elements: usize) -> usize {
        match self {
            // Q4_0: f16 d + 16 packed nibbles (32 vals) = 18 bytes.
            Self::Q4_0 => num_elements.div_ceil(32) * (2 + 16),
            // Q4_1: f16 d + f16 m + 16 packed nibbles = 20 bytes.
            Self::Q4_1 => num_elements.div_ceil(32) * (2 + 2 + 16),
            // Q8_0: f16 d + 32 int8 = 34 bytes.
            Self::Q8_0 => num_elements.div_ceil(32) * (2 + 32),
            // Q8_1: f16 d + f16 s + 32 int8 = 36 bytes.
            Self::Q8_1 => num_elements.div_ceil(32) * (2 + 2 + 32),
            Self::Q2_K => num_elements.div_ceil(256) * (16 * 2 + 2 * 4 + 16 + 4),
            Self::Q3_K => num_elements.div_ceil(256) * (16 * 3 + 2 * 6 + 2),
            Self::Q4_K => num_elements.div_ceil(256) * (16 * 4 + 2 * 6 + 2),
            Self::Q5_K => num_elements.div_ceil(256) * (16 * 5 + 2 * 6 + 2),
            Self::Q6_K => num_elements.div_ceil(256) * (16 * 12 + 2 * 2),
            Self::Q8_K => num_elements.div_ceil(256) * (32 * 8 + 16 * 2 + 8 + 4),
            Self::IQ2_XXS | Self::IQ2_XS | Self::IQ2_S => {
                num_elements.div_ceil(256) * (256 / 4 * 2 + 256 / (4 * 4) * 2)
            }
            Self::IQ3_XXS | Self::IQ3_S => {
                num_elements.div_ceil(256) * (256 / 3 * 3 + 256 / (2 * 4) * 2)
            }
            Self::IQ1_S | Self::IQ1_M => num_elements.div_ceil(256) * 256,
            Self::IQ4_NL | Self::IQ4_XS => {
                num_elements.div_ceil(256) * (256 / 2 + 256 / (4 * 4) * 2)
            }
            Self::I8 => num_elements,
            Self::I16 => num_elements * 2,
            Self::I32 => num_elements * 4,
            Self::I64 => num_elements * 8,
            Self::F64 => num_elements * 8,
            Self::BF16 => num_elements * 2,
            Self::TQ1_0 => num_elements.div_ceil(256) * (2 + 4 * 13),
            Self::TQ2_0 => num_elements.div_ceil(256) * (2 + 64),
            _ => num_elements * self.elem_size(),
        }
    }
}

fn raw_slice(path: &Path, start: usize, end: usize) -> Option<&[u8]> {
    use std::io::Seek;
    let mut f = File::open(path).ok()?;
    let mut buf = vec![0u8; end - start];
    f.seek(std::io::SeekFrom::Start(start as u64)).ok()?;
    f.read_exact(&mut buf).ok()?;
    Some(buf.leak())
}

fn read_string(buf: &[u8], off: &mut usize) -> crate::Result<String> {
    let len = u64::from_le_bytes([
        buf[*off],
        buf[*off + 1],
        buf[*off + 2],
        buf[*off + 3],
        buf[*off + 4],
        buf[*off + 5],
        buf[*off + 6],
        buf[*off + 7],
    ]) as usize;
    *off += 8;
    let s = std::str::from_utf8(&buf[*off..*off + len])
        .map_err(|e| crate::Error::InvalidFormat(format!("utf8 {e}")))?;
    *off += len;
    Ok(s.to_string())
}

fn skip_kv(buf: &[u8], mut off: usize) -> crate::Result<usize> {
    let _key = read_string(buf, &mut off)?;
    if off >= buf.len() {
        return Ok(off);
    }
    let tag = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    off += 4;
    match tag {
        0 | 1 | 7 => {
            off += 1;
        } // u8, i8, bool
        2 | 3 => {
            off += 2;
        } // u16, i16
        4..=6 => {
            off += 4;
        } // u32, i32, f32
        10..=12 => {
            off += 8;
        } // u64, i64, f64
        8 => {
            // string
            let len = u64::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
                buf[off + 4],
                buf[off + 5],
                buf[off + 6],
                buf[off + 7],
            ]) as usize;
            off += 8 + len;
        }
        9 => {
            // array
            let elem_type =
                u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) as usize;
            off += 4;
            let len = u64::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
                buf[off + 4],
                buf[off + 5],
                buf[off + 6],
                buf[off + 7],
            ]) as usize;
            off += 8;
            match elem_type {
                0 | 1 | 7 => off += len,   // u8/i8/bool array
                2 | 3 => off += len * 2,   // u16/i16 array
                4..=6 => off += len * 4,   // u32/i32/f32 array
                10..=12 => off += len * 8, // u64/i64/f64 array
                8 => {
                    // string array
                    for _ in 0..len {
                        let slen = u64::from_le_bytes([
                            buf[off],
                            buf[off + 1],
                            buf[off + 2],
                            buf[off + 3],
                            buf[off + 4],
                            buf[off + 5],
                            buf[off + 6],
                            buf[off + 7],
                        ]) as usize;
                        off += 8 + slen;
                    }
                }
                _ => {
                    return Err(crate::Error::InvalidFormat(format!(
                        "unsupported array element type {elem_type}"
                    )));
                }
            }
        }
        _ => return Err(crate::Error::InvalidFormat(format!("unknown KV tag {tag}"))),
    }
    Ok(off)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
    use super::*;

    struct TensorHeaderPos {
        offset_pos: usize,
        data: Vec<u8>,
    }

    fn write_tensor_info(
        buf: &mut Vec<u8>,
        name: &str,
        n_dims: u32,
        dims: &[u64],
        ty: u32,
    ) -> TensorHeaderPos {
        let name_bytes = format!("{name}\0");
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_bytes.as_bytes());
        buf.extend_from_slice(&n_dims.to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&ty.to_le_bytes());
        let offset_pos = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder
        let num_elems: u64 = dims.iter().product();
        let data_size = match ty {
            0 => num_elems as usize * 4, // F32
            1 => num_elems as usize * 2, // F16
            _ => num_elems as usize,     // quantized (approximate)
        };
        TensorHeaderPos {
            offset_pos,
            data: vec![0u8; data_size],
        }
    }

    fn finalize_gguf(buf: &mut Vec<u8>, header_positions: &[TensorHeaderPos]) {
        const ALIGN: usize = 32;
        let headers_end = buf.len();
        let aligned_data_start = (headers_end + ALIGN - 1) & !(ALIGN - 1);
        let pad = vec![0u8; aligned_data_start - headers_end];
        buf.extend_from_slice(&pad);
        // Offsets are stored relative to the aligned data section.
        let mut rel_offset = 0usize;
        for pos in header_positions {
            buf[pos.offset_pos..pos.offset_pos + 8]
                .copy_from_slice(&(rel_offset as u64).to_le_bytes());
            rel_offset += pos.data.len();
        }
        for pos in header_positions {
            buf.extend_from_slice(&pos.data);
        }
    }

    #[test]
    fn read_minimal_gguf() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let positions = vec![write_tensor_info(&mut buf, "embed", 2, &[256, 1000], 0)];
        finalize_gguf(&mut buf, &positions);
        let path = std::env::temp_dir().join("test_minimal.gguf");
        std::fs::write(&path, &buf).unwrap();
        let model = GGUFModel::open(&path).expect("open");
        let info = model.tensor_info("embed").expect("tensor");
        assert_eq!(info.shape, &[256u64, 1000]);
        assert_eq!(info.ty, GGUFType::F32);
        let data = model.tensor_data::<u8>("embed").expect("data");
        assert_eq!(data.len(), 256 * 1000 * 4);
    }

    #[test]
    fn read_multiple_tensors() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let positions = vec![
            write_tensor_info(&mut buf, "x", 1, &[10], 0),
            write_tensor_info(&mut buf, "y", 1, &[20], 0),
        ];
        finalize_gguf(&mut buf, &positions);
        let path = std::env::temp_dir().join("test_multi.gguf");
        std::fs::write(&path, &buf).unwrap();
        let model = GGUFModel::open(&path).expect("open");
        assert_eq!(model.tensor_info("x").unwrap().shape, &[10u64]);
        assert_eq!(model.tensor_info("y").unwrap().shape, &[20u64]);
    }

    #[test]
    fn tq2_0_tensor_byte_size() {
        assert_eq!(GGUFType::Q2_K.tensor_bytes(1024), 4 * 60);
        assert_eq!(GGUFType::Q2_K.tensor_bytes(256), 60);
        assert_eq!(GGUFType::Q2_K.tensor_bytes(512), 120);
    }

    #[test]
    fn q4_0_tensor_byte_size() {
        // 1024 elements = 32 blocks of 32. Q4_0 = 18 B/block, Q8_0 = 34 B/block.
        assert_eq!(GGUFType::Q4_0.tensor_bytes(1024), 32 * 18);
        assert_eq!(GGUFType::Q8_0.tensor_bytes(1024), 32 * 34);
    }
}
