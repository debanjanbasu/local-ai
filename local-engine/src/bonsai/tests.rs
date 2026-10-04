#![allow(clippy::expect_used, clippy::too_many_lines)]
use super::*;

fn wire_string(value: &str) -> Vec<u8> {
    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(value.as_bytes());
    bytes
}

fn wire_i32s(values: &[i32]) -> Vec<u8> {
    let mut bytes = 5_u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn wire_strings(values: &[&str]) -> Vec<u8> {
    let mut bytes = 8_u32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend(wire_string(value));
    }
    bytes
}

#[derive(Clone)]
struct FixtureTensor {
    name: String,
    dims: Vec<u64>,
    ty: u32,
    data: Vec<u8>,
    offset: Option<u64>,
}

impl FixtureTensor {
    fn packed(name: &str, columns: usize, rows: usize, salt: usize) -> Self {
        let mut data = Vec::new();
        for block in 0..columns / 128 * rows {
            data.extend((0..26).map(|i| ((i * 47 + block * 19 + salt) % 256) as u8));
            // FP16 scale bits; nothing here decodes them.
            let scale = 0x3C00_u16 + ((block + salt) % 13) as u16;
            data.extend_from_slice(&scale.to_le_bytes());
        }
        Self {
            name: name.into(),
            dims: vec![columns as u64, rows as u64],
            ty: 143,
            data,
            offset: None,
        }
    }
}

#[derive(Clone)]
struct Fixture {
    metadata: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<FixtureTensor>,
}

impl Fixture {
    fn new() -> Self {
        let signs = (0..3072)
            .map(|i| if (i * 17 + i / 29) % 5 < 2 { -1 } else { 1 })
            .collect::<Vec<_>>();
        let metadata = vec![
            ("general.architecture", 8, wire_string("qwen35")),
            ("prism.hadamard.version", 4, 1_u32.to_le_bytes().to_vec()),
            (
                "prism.hadamard.block_size",
                4,
                1024_u32.to_le_bytes().to_vec(),
            ),
            (
                "prism.hadamard.transform",
                8,
                wire_string("normalized-sylvester-walsh-hadamard"),
            ),
            (
                "prism.hadamard.axis",
                8,
                wire_string("input-last-dimension"),
            ),
            ("prism.hadamard.sign_mode", 8, wire_string("explicit")),
            ("prism.hadamard.sign_widths", 9, wire_i32s(&[1024, 2048])),
            ("prism.hadamard.sign_values", 9, wire_i32s(&signs)),
            (
                "prism.hadamard.weight_names",
                9,
                wire_strings(&["output.weight", "blk.0.ffn_gate.weight"]),
            ),
            (
                "prism.hadamard.inverse_weight_names",
                9,
                wire_strings(&["token_embd.weight"]),
            ),
            ("prism.hadamard.gdn_v_grouped", 7, vec![1]),
            ("tokenizer.ggml.model", 8, wire_string("gpt2")),
            ("tokenizer.ggml.pre", 8, wire_string("qwen35")),
            ("tokenizer.ggml.add_bos_token", 7, vec![0]),
            ("tokenizer.ggml.tokens", 9, wire_strings(&["a", "b", "ab"])),
            ("tokenizer.ggml.token_type", 9, wire_i32s(&[1, 1, 1])),
            ("tokenizer.ggml.merges", 9, wire_strings(&["a b"])),
            (
                "tokenizer.ggml.bos_token_id",
                4,
                0_u32.to_le_bytes().to_vec(),
            ),
            (
                "tokenizer.ggml.eos_token_id",
                4,
                1_u32.to_le_bytes().to_vec(),
            ),
            (
                "tokenizer.ggml.padding_token_id",
                4,
                2_u32.to_le_bytes().to_vec(),
            ),
            ("tokenizer.chat_template", 8, wire_string("{{ messages }}")),
        ]
        .into_iter()
        .map(|(name, ty, data)| (name.into(), ty, data))
        .collect();
        let tensors = vec![
            FixtureTensor {
                name: "output_norm.weight".into(),
                dims: vec![3],
                ty: 0,
                data: [0.25_f32, -7.125, 0.8125]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                offset: None,
            },
            FixtureTensor {
                name: "blk.0.ssm_alpha.weight".into(),
                dims: vec![3, 2],
                ty: 30,
                data: [0xbf80_u16, 0x3f95, 0xc027, 0x3e51, 0x0080, 0x8000]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                offset: None,
            },
            FixtureTensor::packed("token_embd.weight", 1024, 11, 13),
            FixtureTensor::packed("blk.0.ffn_gate.weight", 2048, 5, 67),
            FixtureTensor::packed("output.weight", 1024, 7, 131),
        ];
        Self { metadata, tensors }
    }

    fn set_metadata(&mut self, name: &str, ty: u32, data: Vec<u8>) {
        self.metadata.retain(|(key, _, _)| key != name);
        self.metadata.push((name.into(), ty, data));
    }

    fn bytes(&self, page_padding: bool) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (name, ty, data) in &self.metadata {
            bytes.extend(wire_string(name));
            bytes.extend_from_slice(&ty.to_le_bytes());
            bytes.extend_from_slice(data);
        }
        let mut payload = Vec::new();
        for tensor in &self.tensors {
            payload.resize(payload.len().div_ceil(32) * 32, 0xab);
            bytes.extend(wire_string(&tensor.name));
            bytes.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for dimension in &tensor.dims {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&tensor.ty.to_le_bytes());
            bytes.extend_from_slice(&tensor.offset.unwrap_or(payload.len() as u64).to_le_bytes());
            payload.extend_from_slice(&tensor.data);
        }
        bytes.resize(bytes.len().div_ceil(32) * 32, 0xcd);
        bytes.extend(payload);
        if page_padding {
            bytes.resize(bytes.len().div_ceil(16384) * 16384, 0xef);
        }
        bytes
    }
}

fn reject(bytes: &[u8], expected_error: &str) {
    let file = tempfile::NamedTempFile::new().expect("invalid fixture");
    std::fs::write(file.path(), bytes).expect("fixture bytes");
    let error = BonsaiPackage::open(file.path())
        .err()
        .expect("fixture must fail");
    assert!(
        error.to_string().contains(expected_error),
        "expected {expected_error:?}, got {error}"
    );
}

#[test]
fn export_index_preserves_checked_offsets_types_shapes_and_signs() {
    let file = tempfile::NamedTempFile::new().expect("index fixture");
    std::fs::write(file.path(), Fixture::new().bytes(false)).expect("fixture bytes");
    let package = BonsaiPackage::open(file.path()).expect("checked package");
    let index = package.export_index(file.path()).expect("index");

    assert_eq!(index["schema_version"], 1);
    assert_eq!(
        index["source"]["path"],
        file.path()
            .canonicalize()
            .expect("canonical fixture path")
            .to_str()
            .expect("utf-8 fixture path")
    );
    assert_eq!(
        index["source"]["bytes"],
        file.as_file().metadata().expect("fixture metadata").len()
    );
    // The tokenizer block is a verbatim copy of the checkpoint metadata in
    // the sidecar's schema; the export copies it without reinterpretation.
    let tokenizer: TokenizerDefinition =
        serde_json::from_value(index["tokenizer"].clone()).expect("tokenizer block");
    assert_eq!(
        tokenizer,
        TokenizerDefinition {
            schema_version: 1,
            model: "gpt2".into(),
            pre: "qwen35".into(),
            add_bos_token: false,
            tokens: vec!["a".into(), "b".into(), "ab".into()],
            token_type: vec![1, 1, 1],
            merges: vec!["a b".into()],
            bos_token_id: 0,
            eos_token_id: 1,
            padding_token_id: 2,
            chat_template: "{{ messages }}".into(),
        }
    );
    let f32_record = &index["tensors"]["output_norm.weight"];
    assert_eq!(f32_record["dtype"], "f32");
    assert_eq!(f32_record["shape"], serde_json::json!([3]));
    assert_eq!(f32_record["bytes"], 12);
    assert_eq!(f32_record["offset"], package.data_start());
    let bf16_record = &index["tensors"]["blk.0.ssm_alpha.weight"];
    assert_eq!(bf16_record["dtype"], "bf16");
    assert_eq!(bf16_record["shape"], serde_json::json!([2, 3]));
    assert_eq!(bf16_record["bytes"], 12);
    assert_eq!(bf16_record["offset"], package.data_start() + 32);
    let packed = &index["tensors"]["token_embd.weight"];
    assert_eq!(packed["dtype"], "u8");
    assert_eq!(packed["encoding"], "ptq1_0");
    assert_eq!(packed["shape"], serde_json::json!([11, 1024]));
    assert_eq!(packed["bytes"], 1024 / 128 * 11 * 28);
    assert_ne!(packed["bytes"], 11 * 1024);
    assert_eq!(
        index["hadamard_signs"]["1024"]
            .as_array()
            .expect("1024 signs")
            .len(),
        1024
    );
    assert_eq!(
        index["hadamard_signs"]["2048"]
            .as_array()
            .expect("2048 signs")
            .len(),
        2048
    );
    assert!(
        index["hadamard_signs"]["1024"]
            .as_array()
            .expect("1024 signs")
            .iter()
            .all(|value| matches!(value.as_f64(), Some(-1.0 | 1.0)))
    );
}

#[test]
fn checked_arithmetic_rejects_overflow() {
    assert!(checked_product(&[u64::MAX, 2]).is_err());
    assert!(align_up(usize::MAX, 32).is_err());
    assert!(align_up_u64(u64::MAX, 32).is_err());
    assert_eq!(align_up(33, 32).expect("aligned"), 64);
}

#[test]
fn metal_window_boundaries() {
    // A two-byte tensor straddling a page boundary needs both pages.
    assert_eq!(
        metal_window(16383, 2, 32768, 32768).expect("two pages"),
        Some(0..32768)
    );
    // The second page runs past EOF: copy instead of mapping.
    assert_eq!(
        metal_window(16383, 2, 32767, 32768).expect("EOF fallback"),
        None
    );
    // The bare tensor fits the device limit but the page window does not.
    assert_eq!(
        metal_window(16383, 2, 32768, 2).expect("fits as copy"),
        None
    );
    assert!(metal_window(16383, 2, 32768, 1).is_err());
    assert!(metal_window(usize::MAX, 2, usize::MAX, usize::MAX).is_err());
    assert!(metal_window(20, 3, 22, 4096).is_err());
    assert!(metal_window(0, 0, 22, 4096).is_err());
}

#[test]
fn header_bounds_types_utf8_and_bool_are_checked_even_for_unused_metadata() {
    let good = Fixture::new().bytes(false);
    for end in [0, 3, 4, 7, 8, 15, 16, 23, 24, 31, 47, good.len() - 1] {
        reject(&good[..end], "");
    }
    for (offset, data, error) in [
        (0, b"FGUG".to_vec(), "magic"),
        (4, 3_u32.to_be_bytes().to_vec(), "little-endian"),
        (8, u64::MAX.to_le_bytes().to_vec(), "count"),
        (16, u64::MAX.to_le_bytes().to_vec(), "count"),
        (24, u64::MAX.to_le_bytes().to_vec(), "overflow"),
        (
            24,
            (MAX_HEADER_BYTES as u64 + 1).to_le_bytes().to_vec(),
            "header",
        ),
    ] {
        let mut bad = good.clone();
        bad[offset..offset + data.len()].copy_from_slice(&data);
        reject(&bad, error);
    }
    let mut oversized_array = 0_u32.to_le_bytes().to_vec();
    oversized_array.extend_from_slice(&u64::MAX.to_le_bytes());
    let mut invalid_utf8 = 1_u64.to_le_bytes().to_vec();
    invalid_utf8.push(0xff);
    for (ty, data, error) in [
        (99, vec![], "unknown"),
        (7, vec![2], "bool"),
        (8, invalid_utf8, "UTF-8"),
        (
            9,
            [99_u32.to_le_bytes().as_slice(), &0_u64.to_le_bytes()].concat(),
            "unknown",
        ),
        (
            9,
            [9_u32.to_le_bytes().as_slice(), &0_u64.to_le_bytes()].concat(),
            "nested",
        ),
        (9, oversized_array, "count"),
    ] {
        let mut fixture = Fixture::new();
        fixture.set_metadata("unused", ty, data);
        reject(&fixture.bytes(false), error);
    }
    let mut cursor = Cursor::new(&good, 7);
    assert!(cursor.u64().is_err(), "header budget must constrain reads");
}

#[test]
fn rejects_bad_descriptors_duplicate_keys_and_payload_extents() {
    let base = Fixture::new();
    let mut duplicate = base.clone();
    duplicate.metadata.push(duplicate.metadata[0].clone());
    reject(&duplicate.bytes(false), "duplicate GGUF metadata");
    duplicate = base.clone();
    duplicate.tensors[1].name = duplicate.tensors[0].name.clone();
    reject(&duplicate.bytes(false), "duplicate GGUF tensor");
    for alignment in [0_u32, 3] {
        let mut fixture = base.clone();
        fixture.set_metadata("general.alignment", 4, alignment.to_le_bytes().to_vec());
        reject(&fixture.bytes(false), "power of two");
    }
    for (dims, ty, offset, error) in [
        (vec![1024, 7], 1, None, "unsupported Bonsai tensor type"),
        (vec![], 143, None, "dimension count"),
        (vec![0, 7], 143, None, "zero dimension"),
        (vec![1; 5], 143, None, "dimension count"),
        (vec![1025, 7], 143, None, "contiguous width"),
        (vec![128, u64::MAX], 143, None, "count overflow"),
        (vec![1024, 8], 143, None, "truncated payload"),
        (vec![1024, 7], 143, Some(0), "offset"),
        (vec![1024, 7], 143, Some(u64::MAX), "offset"),
    ] {
        let mut fixture = base.clone();
        let tensor = fixture.tensors.last_mut().expect("last tensor");
        tensor.dims = dims;
        tensor.ty = ty;
        tensor.offset = offset;
        reject(&fixture.bytes(false), error);
    }
}

#[test]
fn validates_sign_partition_rotation_coverage_and_direction() {
    let base = Fixture::new();
    for (name, ty, data) in [
        ("prism.hadamard.version", 4, 2_u32.to_le_bytes().to_vec()),
        ("prism.hadamard.version", 5, 1_i32.to_le_bytes().to_vec()),
        (
            "prism.hadamard.block_size",
            4,
            512_u32.to_le_bytes().to_vec(),
        ),
        ("prism.hadamard.sign_mode", 8, wire_string("identity")),
        ("prism.hadamard.sign_widths", 9, wire_i32s(&[1024, 1024])),
        ("prism.hadamard.sign_widths", 9, wire_i32s(&[-1024, 2048])),
        ("prism.hadamard.sign_widths", 9, wire_i32s(&[512, 2048])),
        ("prism.hadamard.sign_widths", 9, wire_i32s(&[3072])),
        ("prism.hadamard.sign_values", 9, wire_i32s(&vec![1; 3071])),
        ("prism.hadamard.sign_values", 9, wire_i32s(&vec![1; 3073])),
        ("prism.hadamard.sign_values", 9, wire_i32s(&vec![0; 3072])),
        (
            "prism.hadamard.weight_names",
            9,
            wire_strings(&["output.weight"]),
        ),
        (
            "prism.hadamard.weight_names",
            9,
            wire_strings(&["output.weight", "output.weight", "blk.0.ffn_gate.weight"]),
        ),
        (
            "prism.hadamard.weight_names",
            9,
            wire_strings(&["blk.99.ffn_gate.weight"]),
        ),
        (
            "prism.hadamard.weight_names",
            9,
            wire_strings(&[
                "token_embd.weight",
                "output.weight",
                "blk.0.ffn_gate.weight",
            ]),
        ),
        (
            "prism.hadamard.inverse_weight_names",
            9,
            wire_strings(&["output.weight"]),
        ),
        ("prism.hadamard.gdn_v_grouped", 7, vec![0]),
    ] {
        let mut fixture = base.clone();
        fixture.set_metadata(name, ty, data);
        reject(&fixture.bytes(false), "");
    }
    // Architecture and Hadamard keys are mandatory at open; tokenizer keys
    // are only read when the tokenizer (or the index) is built.
    for (name, _, _) in base
        .metadata
        .iter()
        .filter(|(key, _, _)| !key.starts_with("tokenizer."))
    {
        let mut fixture = base.clone();
        fixture.metadata.retain(|(key, _, _)| key != name);
        reject(&fixture.bytes(false), "missing");
    }
}

#[test]
fn tensor_bytes_are_exact_views_of_the_mapped_payload() {
    let fixture = Fixture::new();
    for padded in [false, true] {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("fixture.gguf");
        let bytes = fixture.bytes(padded);
        std::fs::write(&path, &bytes).expect("write fixture");
        let package = BonsaiPackage::open(&path).expect("open fixture");
        let data_start = package.data_start() as usize;
        for tensor in package.tensors() {
            let start = data_start + tensor.relative_offset() as usize;
            let expected = &bytes[start..start + tensor.bytes() as usize];
            assert_eq!(package.bytes(tensor.name()).expect("bytes"), expected);
        }
        assert!(package.bytes("missing.weight").is_err());
    }
}

#[test]
#[ignore = "requires BONSAI_GGUF"]
fn real_bonsai_profile_is_pinned() {
    let path = std::env::var("BONSAI_GGUF").expect("BONSAI_GGUF must name the verified GGUF");
    let package = BonsaiPackage::open(path).expect("open real Bonsai");
    assert_eq!(
        (
            package.metadata_count(),
            package.tensors().len(),
            package.data_start()
        ),
        (49, 851, 11_120_992)
    );
    let counts = package.tensors().iter().fold([0; 3], |mut c, t| {
        c[match t.tensor_type() {
            BonsaiTensorType::F32 => 0,
            BonsaiTensorType::Bf16 => 1,
            BonsaiTensorType::Ptq1 => 2,
        }] += 1;
        c
    });
    assert_eq!(counts, [353, 96, 402]);
    eprintln!("Bonsai verified metadata49 tensors851 F32=353 BF16=96 PTQ1=402 data_start=11120992");
    validate_profile(&package).expect("pinned profile");
}
