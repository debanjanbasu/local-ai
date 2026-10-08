use std::collections::HashMap;
use std::path::Path;

use super::*;

#[test]
fn ternary_source_framing_rejects_short_and_oversized_headers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(MTP_TERNARY_SOURCE);
    std::fs::write(&path, [1u8, 0, 0, 0]).expect("write");
    assert!(head_sections(&path).is_err(), "short length prefix");
    let mut framed = (SAFETENSORS_MAX_HEADER + 1).to_le_bytes().to_vec();
    framed.extend_from_slice(b"{}");
    std::fs::write(&path, &framed).expect("write");
    assert!(head_sections(&path).is_err(), "oversized header");
    let json = br#"{"mtp.norm.weight":{"dtype":"BF16","shape":[5120],"data_offsets":[0,10240]}}"#;
    let mut framed = (json.len() as u64).to_le_bytes().to_vec();
    framed.extend_from_slice(json);
    framed.extend_from_slice(&[0u8; 16]);
    std::fs::write(&path, &framed).expect("write");
    // The header parses, but it is missing nearly every pinned tensor.
    assert!(head_sections(&path).is_err(), "missing tensors");
}

#[test]
fn norm_folding_adds_one_and_rejects_non_finite() {
    let zero = half::bf16::from_f32(0.0).to_bits().to_le_bytes();
    let minus_quarter = half::bf16::from_f32(-0.25).to_bits().to_le_bytes();
    let mut bytes = zero.to_vec();
    bytes.extend_from_slice(&minus_quarter);
    assert_eq!(fold_norm(&bytes).expect("finite"), vec![1.0, 0.75]);
    let nan = half::bf16::NAN.to_bits().to_le_bytes();
    assert!(fold_norm(&nan).is_err());
}

#[test]
fn head_file_round_trips_and_rejects_damage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sections: Vec<Vec<u8>> = vec![
        (0..3000_usize).map(|at| (at * 31) as u8).collect(),
        (0..20_000_usize).map(|at| (at * 7 + 3) as u8).collect(),
    ];
    let spec: Vec<Section> = vec![
        ("mtp.fc.weight.embedding.ptq1".to_owned(), sections[0].len()),
        ("mtp.norm.weight".to_owned(), sections[1].len()),
    ];
    let path = dir.path().join(MTP_HEAD_ARTIFACT);
    let head = write_artifact(&path, &spec, &sections).expect("write artifact");
    assert_eq!(&magic_of(&path), b"MTPT1\0\0\0");
    assert_eq!(head.payload_sha256.len(), 64);
    // The payload skips the padding, and the file skips nothing: one page holds
    // the header and every section is padded up to a page of its own.
    assert_eq!(head.payload_bytes, sections[0].len() + sections[1].len());
    assert_eq!(
        head.file_bytes,
        align_page(HEADER_BYTES).unwrap_or(0)
            + sections
                .iter()
                .map(|section| align_page(section.len()).unwrap_or(0))
                .sum::<usize>()
    );

    let read = read_head(&path, &spec).expect("read artifact");
    // The header records the digest the writer computed; reading re-derives it.
    assert_eq!(
        std::fs::read(&path).expect("read bytes")[32..96],
        *head.payload_sha256.as_bytes()
    );
    for ((offset, length), expected) in read.sections.iter().zip(&sections) {
        assert_eq!(offset % PAGE, 0);
        assert_eq!(&read.map[*offset..*offset + *length], expected);
    }
    // Identical payloads produce identical files.
    let again = dir.path().join("again.bin");
    write_artifact(&again, &spec, &sections).expect("rewrite");
    assert_eq!(
        std::fs::read(&again).expect("read again"),
        std::fs::read(&path).expect("read first")
    );
    // A section count or size the spec does not bind is refused at write time.
    assert!(write_artifact(&again, &spec[..1], &sections).is_err());

    // Sizes alone cannot bind sections; the names are what say which is which.
    let renamed = vec![
        ("mtp.fc.weight.hidden.ptq1".to_owned(), spec[0].1),
        spec[1].clone(),
    ];
    let unbound = read_head(&path, &renamed).err().expect("renamed section");
    assert!(unbound.contains("section 0 is"), "{unbound}");
    let resized = vec![spec[0].clone(), (spec[1].0.clone(), spec[1].1 + 1)];
    assert!(read_head(&path, &resized).is_err(), "resized section");

    let pristine = std::fs::read(&path).expect("read bytes");
    let reason = |bytes: &[u8]| -> String {
        std::fs::write(&path, bytes).expect("write damage");
        read_head(&path, &spec).err().expect("damaged artifact")
    };
    let mut corrupt = pristine.clone();
    corrupt[0] ^= 1;
    assert!(reason(&corrupt).contains("not an MTP head artifact"));
    // A different version is a different format, not this one misread.
    let mut v2 = pristine.clone();
    v2[8] = 2;
    assert!(reason(&v2).contains(&format!("reads version {PTQ1_VERSION}")));
    // Flipping a payload byte changes the content identity, which is what the
    // header digest exists to catch.
    let mut flipped = pristine.clone();
    flipped[read.sections[1].0 + 1] ^= 0xff;
    assert!(reason(&flipped).contains("payload hashes to"));
    let mut appended = pristine.clone();
    appended.extend_from_slice(&[0; 8]);
    assert!(reason(&appended).contains("header records"));
    assert!(reason(&pristine[..HEADER_BYTES / 2]).contains("shorter than one"));
    assert!(reason(b"").contains("shorter than one"));
    std::fs::write(&path, &pristine).expect("restore");
    assert!(read_head(&path, &spec).is_ok());
}

fn magic_of(path: &Path) -> [u8; 8] {
    let mut magic = [0_u8; 8];
    std::fs::File::open(path)
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
        .expect("read magic");
    magic
}

/// The installed head validates against this build's spec and is the trained
/// artifact the docs pin.
#[test]
#[ignore = "requires the installed MTP head artifact"]
fn installed_head_validates_and_matches_the_pinned_digest() {
    let path = Path::new(DEFAULT_BONSAI_MTP_ARTIFACT);
    // Reading re-derives the payload digest and checks it against the header,
    // through the loader's own binder: the installed head is mixed.
    let bind = |names: &[&str]| formats_from_names(names).map(|formats| head_spec(&formats));
    let head = read_head_with(path, &bind).expect("installed MTP head");
    // fc (both halves), k, v and o are int8; q, gate, up and down ternary.
    let mut formats = [MatrixFormat::Ptq1; 9];
    for index in [0, 1, 3, 4, 5] {
        formats[index] = MatrixFormat::Int8;
    }
    assert_eq!(head.spec, head_spec(&formats));
    assert_eq!(head.map.len(), 166_969_344);
    assert_eq!(
        head.map[32..96],
        *b"fb05507f87f54432c782b0fcb35e5cf1da1b4997610ab35a3d98502e536351cd"
    );
}

#[test]
fn settings_bound_depth() {
    assert!(MtpSettings::new(PathBuf::from("head"), 0).is_err());
    assert!(MtpSettings::new(PathBuf::from("head"), MAX_MTP_DEPTH + 1).is_err());
    // With a quantized head, gated depth 3 beat depth 2 on every measured
    // case (arithmetic 27.5 vs 26.8 decode tok/s, identical tokens).
    assert_eq!(DEFAULT_MTP_DEPTH, 3);
    const { assert!(DEFAULT_MTP_DEPTH <= MAX_MTP_DEPTH) }
    assert_eq!(
        MtpSettings::new(PathBuf::from("head"), DEFAULT_MTP_DEPTH)
            .expect("default")
            .depth,
        DEFAULT_MTP_DEPTH
    );
}

#[test]
fn auto_speculates_only_with_an_installed_head() {
    let dir = tempfile::tempdir().expect("tempdir");
    let installed = dir.path().join(MTP_HEAD_ARTIFACT);
    let missing = dir.path().join("absent.bin");
    std::fs::write(&installed, b"stub").expect("write head");
    let auto = MtpMode::Auto { depth: 3 };

    assert_eq!(
        auto.resolve_with_default(&installed, true)
            .expect("installed"),
        MtpResolution::Native(MtpSettings::new(installed.clone(), 3).expect("settings"))
    );
    let MtpResolution::Disabled(reason) = auto
        .resolve_with_default(&missing, true)
        .expect("missing head decodes plainly")
    else {
        unreachable!("a missing default head must not enable speculation");
    };
    assert!(reason.contains("absent.bin"), "{reason}");
    // A directory at the default path is not a head.
    assert!(matches!(
        auto.resolve_with_default(dir.path(), true)
            .expect("directory"),
        MtpResolution::Disabled(_)
    ));
    // The build default can turn an installed head off.
    assert_eq!(
        auto.resolve_with_default(&installed, false)
            .expect("auto off"),
        MtpResolution::Disabled(MTP_DEFAULT_OFF_REASON.into())
    );
    assert!(
        MtpMode::Auto { depth: 0 }
            .resolve_with_default(&installed, true)
            .is_err()
    );
    assert_eq!(
        MtpMode::default(),
        MtpMode::Auto {
            depth: DEFAULT_MTP_DEPTH
        }
    );
    assert!(matches!(
        MtpMode::Off(None)
            .resolve_with_default(&installed, true)
            .expect("off"),
        MtpResolution::Disabled(reason) if reason == MTP_OFF_REASON
    ));

    // The default is the shipped ternary artifact.
    assert_eq!(
        Path::new(DEFAULT_BONSAI_MTP_ARTIFACT).file_name(),
        Some(Path::new(MTP_HEAD_ARTIFACT).as_os_str())
    );
    assert!(
        MTP_HEAD_ARTIFACT.ends_with(&format!("-v{PTQ1_VERSION}.bin")),
        "the artifact filename must track the format version"
    );
}

#[test]
fn explicit_head_overrides_the_default() {
    let dir = tempfile::tempdir().expect("tempdir");
    let installed = dir.path().join(MTP_HEAD_ARTIFACT);
    let default_head = dir.path().join("default.bin");
    let absent_path = dir.path().join("absent.bin");
    std::fs::write(&installed, b"stub").expect("write head");
    std::fs::write(&default_head, b"stub").expect("write default head");
    let explicit = MtpMode::Head(MtpSettings::new(installed.clone(), 1).expect("settings"));

    // The explicit head wins over an installed default, whatever the policy.
    for auto_enabled in [true, false] {
        assert_eq!(
            explicit
                .resolve_with_default(&default_head, auto_enabled)
                .expect("explicit"),
            MtpResolution::Native(MtpSettings::new(installed.clone(), 1).expect("settings"))
        );
    }
    // An explicit head must exist.
    let absent = MtpMode::Head(MtpSettings::new(absent_path, 1).expect("settings"));
    assert!(absent.resolve_with_default(&default_head, true).is_err());
    // Off never errors and ignores every head.
    assert_eq!(
        MtpMode::Off(None)
            .resolve_with_default(&default_head, true)
            .expect("off"),
        MtpResolution::Disabled(MTP_OFF_REASON.into())
    );
}

/// `Off` carries the reason its own constructor recorded, so the record is
/// carried rather than re-derived.
///
/// A discoverer that finds no head builds `Off` exactly as an explicit opt-out
/// does; a mode with no payload makes the two indistinguishable, and every
/// report then names an opt-out the user never asked for.
#[test]
fn off_reports_the_reason_it_carried() {
    let dir = tempfile::tempdir().expect("tempdir");
    let installed = dir.path().join(MTP_HEAD_ARTIFACT);
    std::fs::write(&installed, b"stub").expect("write head");
    let absent = "no MTP head artifact installed beside the model";

    // A discoverer that found nothing beside the model says so, verbatim.
    assert_eq!(
        MtpMode::Off(Some(absent.into()))
            .resolve_with_default(&installed, true)
            .expect("off"),
        MtpResolution::Disabled(absent.into())
    );
    // ... which is not the request's opt-out text, whatever head is installed.
    assert_ne!(
        MtpMode::Off(Some(absent.into()))
            .resolve_with_default(&installed, true)
            .expect("off"),
        MtpResolution::Disabled(MTP_OFF_REASON.into())
    );
    // An `Off` that recorded nothing still falls back to the generic text.
    assert_eq!(
        MtpMode::Off(None)
            .resolve_with_default(&installed, true)
            .expect("off"),
        MtpResolution::Disabled(MTP_OFF_REASON.into())
    );
}

/// Deterministic xorshift stream for the ternary tests.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_codes(state: &mut u64, count: usize) -> Vec<i8> {
    (0..count)
        .map(|_| (xorshift(state) % 3) as i8 - 1)
        .collect()
}

#[test]
fn ptq1_packer_round_trips_through_the_engine_decoder() {
    // Every base-3 digit tuple at every byte position: block `k` puts the digits
    // of `(k + group) % 243` (or `% 81` for the four-trit bytes) in each group.
    let mut blocks = Vec::new();
    let mut expected = Vec::new();
    for k in 0..243_usize {
        let mut codes = [0_i8; PTQ1_BLOCK_ELEMENTS];
        let digits = |value: usize, count: usize| -> Vec<i8> {
            (0..count)
                .map(|n| ((value / 3_usize.pow((count - 1 - n) as u32)) % 3) as i8 - 1)
                .collect()
        };
        for m in 0..16 {
            for (n, digit) in digits((k + m) % 243, 5).into_iter().enumerate() {
                codes[m + n * 16] = digit;
            }
        }
        for m in 0..8 {
            for (n, digit) in digits((k + 7 * m) % 243, 5).into_iter().enumerate() {
                codes[80 + m + n * 8] = digit;
            }
        }
        for j in 0..2 {
            for (m, digit) in digits((k + 40 * j) % 81, 4).into_iter().enumerate() {
                codes[120 + j + m * 2] = digit;
            }
        }
        let scale = half::f16::from_f32(0.001 * (k as f32 + 1.0));
        blocks.extend_from_slice(&pack_ptq1_block(&codes, scale.to_bits()));
        expected.extend(codes.iter().map(|&code| f32::from(code) * scale.to_f32()));
    }
    let mut decoded = vec![0.0_f32; expected.len()];
    local_metal::bonsai::decode_ptq1_row(&blocks, &mut decoded).expect("decode");
    assert_eq!(decoded, expected);

    // Known bytes: all-zero codes are digits 1 everywhere, 121 -> ceil(121 * 256 / 243).
    let zero = pack_ptq1_block(&[0; PTQ1_BLOCK_ELEMENTS], 0x3c00);
    assert!(zero[..24].iter().all(|&byte| byte == 128));
    // 4-trit bytes: 1111 base 3 = 40, shifted one trit = 120 -> ceil(120 * 256 / 243) = 127.
    assert_eq!(zero[24..26], [127, 127]);
    assert_eq!(zero[26..], [0x00, 0x3c]);
    assert_eq!(zero.len(), PTQ1_BLOCK_BYTES);

    // The matrix packer: row-major codes, one F16 scale per 128 columns.
    let (rows, columns) = (3, 384);
    let mut state = 0x9e37_79b9_7f4a_7c15;
    let codes = random_codes(&mut state, rows * columns);
    let scales: Vec<half::f16> = (0..rows * columns / 128)
        .map(|index| half::f16::from_f32(0.003f32.mul_add(index as f32, 0.01)))
        .collect();
    let code_bytes: Vec<u8> = codes.iter().map(|&code| code as u8).collect();
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let packed = pack_ptq1_matrix(&code_bytes, &scale_bytes, rows, columns).expect("pack");
    let mut decoded = vec![0.0_f32; columns];
    for row in 0..rows {
        let row_bytes = columns / 128 * PTQ1_BLOCK_BYTES;
        local_metal::bonsai::decode_ptq1_row(
            &packed[row * row_bytes..(row + 1) * row_bytes],
            &mut decoded,
        )
        .expect("decode row");
        for (column, value) in decoded.iter().enumerate() {
            let index = row * columns + column;
            assert_eq!(
                *value,
                f32::from(codes[index]) * scales[index / 128].to_f32()
            );
        }
    }
    let mut bad = code_bytes.clone();
    bad[5] = 2;
    assert!(pack_ptq1_matrix(&bad, &scale_bytes, rows, columns).is_err());
    assert!(pack_ptq1_matrix(&code_bytes[1..], &scale_bytes, rows, columns).is_err());
    let infinite = half::f16::INFINITY.to_le_bytes();
    let mut bad_scales = scale_bytes;
    bad_scales[..2].copy_from_slice(&infinite);
    assert!(pack_ptq1_matrix(&code_bytes, &bad_scales, rows, columns).is_err());
}

#[test]
fn ternary_spec_packs_every_matrix_and_keeps_the_norms() {
    let spec = ternary_spec();
    assert_eq!(spec.len(), 16);
    assert_eq!(spec[0].0, "mtp.fc.weight.embedding.ptq1");
    assert_eq!(spec[8].0, "mtp.layers.0.mlp.down_proj.weight.ptq1");
    assert_eq!(spec[8].1, WIDTH * FFN / 128 * 28);
    assert_eq!(spec[9].0, "mtp.pre_fc_norm_embedding.weight");
    let matrices: usize = spec[..9].iter().map(|(_, bytes)| bytes).sum();
    // 424,673,280 ternary weights at 28 bytes per 128.
    assert_eq!(matrices, 92_897_280);
}

#[test]
fn mixed_spec_names_select_each_matrix_format() {
    let mut formats = [MatrixFormat::Ptq1; 9];
    // q (2) and down (8) int8: each takes two sections, weights then scales.
    formats[2] = MatrixFormat::Int8;
    formats[8] = MatrixFormat::Int8;
    let spec = head_spec(&formats);
    assert_eq!(spec.len(), 18);
    assert_eq!(spec[2].0, "mtp.layers.0.self_attn.q_proj.weight.int8");
    assert_eq!(spec[2].1, 12288 * WIDTH);
    assert_eq!(spec[3].0, "mtp.layers.0.self_attn.q_proj.weight.row_scales");
    assert_eq!(spec[3].1, 12288 * 4);
    assert_eq!(spec[4].0, "mtp.layers.0.self_attn.k_proj.weight.ptq1");
    assert_eq!(spec[9].0, "mtp.layers.0.mlp.down_proj.weight.int8");
    assert_eq!(spec[10].0, "mtp.layers.0.mlp.down_proj.weight.row_scales");
    assert_eq!(spec[11].0, "mtp.pre_fc_norm_embedding.weight");
    let names: Vec<&str> = spec.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(formats_from_names(&names), Ok(formats));
    // An all-ternary head is exactly the ternary spec, so its file is too.
    assert_eq!(head_spec(&[MatrixFormat::Ptq1; 9]), ternary_spec());

    let mut foreign = names.clone();
    foreign[2] = "mtp.layers.0.self_attn.q_proj.weight.bf16";
    let reason = formats_from_names(&foreign).expect_err("unknown format");
    assert!(reason.contains("section 2 is"), "{reason}");
    let mut swapped = names.clone();
    swapped.swap(4, 5);
    assert!(formats_from_names(&swapped).is_err(), "k and v swapped");
    assert!(formats_from_names(&names[..9]).is_err(), "truncated");
}

/// A head file whose names select a different spec than its sizes or count
/// fit is refused, through the same binder the loader uses.
#[test]
fn head_reader_binds_the_spec_its_names_select() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(MTP_HEAD_ARTIFACT);
    let int8_spec: Vec<Section> = vec![
        ("w.int8".to_owned(), 64),
        ("w.row_scales".to_owned(), 8),
        ("norm".to_owned(), 16),
    ];
    let ptq1_spec: Vec<Section> = vec![("w.ptq1".to_owned(), 56), ("norm".to_owned(), 16)];
    let bind = |names: &[&str]| -> Result<Vec<Section>, String> {
        match names.first() {
            Some(&"w.int8") => Ok(int8_spec.clone()),
            Some(&"w.ptq1") => Ok(ptq1_spec.clone()),
            other => Err(format!("unknown first section {other:?}")),
        }
    };
    let sections: Vec<Vec<u8>> = int8_spec
        .iter()
        .map(|(_, bytes)| (0..*bytes).map(|at| at as u8).collect())
        .collect();
    write_artifact(&path, &int8_spec, &sections).expect("write mixed");
    let read = read_head_with(&path, &bind).expect("read mixed");
    assert_eq!(read.spec, int8_spec);
    assert_eq!(read.sections.len(), 3);
    let (offset, length) = read.sections[1];
    assert_eq!(&read.map[offset..offset + length], &sections[1][..]);

    // The same bytes read against the ternary spec are refused by count.
    let reason = read_head(&path, &ptq1_spec).err().expect("wrong spec");
    assert!(reason.contains("holds 3 sections"), "{reason}");
    let refused = read_head_with(&path, &|_| Err("no".to_owned()))
        .err()
        .expect("binder refusal");
    assert!(refused.ends_with(": no"), "{refused}");
}

/// Build a full-size head source as a sparse file: every tensor at its pinned
/// shape, all bytes zero (valid in every dtype) unless poked afterwards.
/// Returns each tensor's data start.
fn sparse_source(path: &Path, formats: &[MatrixFormat; 9]) -> HashMap<String, u64> {
    let mut tensors: Vec<(String, &str, Vec<usize>, usize)> = Vec::new();
    for (matrix, format) in weights::MATRIX_SPECS.iter().zip(formats) {
        let (rows, columns) = (matrix.rows, matrix.columns);
        match format {
            MatrixFormat::Ptq1 => {
                tensors.push((
                    format!("{}.codes", matrix.section),
                    "I8",
                    vec![rows, columns],
                    1,
                ));
                tensors.push((
                    format!("{}.scales", matrix.section),
                    "F16",
                    vec![rows, columns / 128],
                    2,
                ));
            }
            MatrixFormat::Int8 => {
                tensors.push((
                    format!("{}.int8", matrix.section),
                    "I8",
                    vec![rows, columns],
                    1,
                ));
                tensors.push((
                    format!("{}.row_scales", matrix.section),
                    "F32",
                    vec![rows],
                    4,
                ));
            }
        }
    }
    for norm in &weights::NORM_SPECS {
        tensors.push((norm.section.to_owned(), "BF16", vec![norm.elements], 2));
    }
    let mut header = serde_json::Map::new();
    let mut starts = HashMap::new();
    let mut at = 0_usize;
    for (name, dtype, shape, element) in tensors {
        let bytes = shape.iter().product::<usize>() * element;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + bytes]}),
        );
        starts.insert(name, at as u64);
        at += bytes;
    }
    let json = serde_json::to_vec(&header).expect("header json");
    let data_start = 8 + json.len() as u64;
    let mut file = std::fs::File::create(path).expect("create source");
    std::io::Write::write_all(&mut file, &(json.len() as u64).to_le_bytes()).expect("length");
    std::io::Write::write_all(&mut file, &json).expect("header");
    file.set_len(data_start + at as u64).expect("sparse data");
    starts
        .into_iter()
        .map(|(name, start)| (name, data_start + start))
        .collect()
}

fn poke(path: &Path, offset: u64, bytes: &[u8]) {
    use std::os::unix::fs::FileExt;
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open source")
        .write_all_at(bytes, offset)
        .expect("poke");
}

/// A mixed source exports to an artifact whose names say which matrices are
/// int8, whose int8 sections are the source bytes verbatim, and which the
/// loader's binder accepts; invalid int8 values, non-finite row scales and a
/// matrix given in both formats are refused.
#[test]
fn mixed_source_exports_verbatim_and_reloads() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source = dir.path().join(MTP_TERNARY_SOURCE);
    let mut formats = [MatrixFormat::Ptq1; 9];
    formats[3] = MatrixFormat::Int8; // k_proj
    formats[4] = MatrixFormat::Int8; // v_proj
    let starts = sparse_source(&source, &formats);
    let k = "mtp.layers.0.self_attn.k_proj.weight";
    poke(&source, starts[&format!("{k}.int8")] + 7, &[127, 0x81]);
    poke(
        &source,
        starts[&format!("{k}.row_scales")] + 4,
        &0.25_f32.to_le_bytes(),
    );
    let exported = export_head(dir.path()).expect("export mixed");
    assert_eq!(
        exported.int8_matrices,
        vec![
            k.to_owned(),
            "mtp.layers.0.self_attn.v_proj.weight".to_owned()
        ]
    );
    assert_eq!(exported.sections, 18);
    let bind = |names: &[&str]| formats_from_names(names).map(|formats| head_spec(&formats));
    let read = read_head_with(&exported.path, &bind).expect("reload mixed");
    assert_eq!(read.spec, head_spec(&formats));
    let section = |index: usize| {
        let (offset, length) = read.sections[index];
        &read.map[offset..offset + length]
    };
    assert_eq!(read.spec[3].0, format!("{k}.int8"));
    assert_eq!(&section(3)[7..9], &[127, 0x81]);
    assert!(
        section(3)
            .iter()
            .enumerate()
            .all(|(at, &byte)| byte == 0 || (7..9).contains(&at))
    );
    assert_eq!(&section(4)[4..8], &0.25_f32.to_le_bytes());
    assert_eq!(section(4).len(), KV * 4);
    // Zero codes pack to the PTQ1 byte for all-zero trits.
    assert_eq!(section(0)[0], pack_ptq1_block(&[0; 128], 0)[0]);
    // Folded norms: zero-centered 0 becomes 1.
    assert_eq!(&section(11)[..4], &1.0_f32.to_le_bytes());

    // -128 has no positive twin; the contract's range is [-127, 127].
    poke(&source, starts[&format!("{k}.int8")] + 9, &[0x80]);
    let refused = export_head(dir.path()).expect_err("-128 refused");
    assert!(refused.to_string().contains("[-127, 127]"), "{refused}");
    poke(&source, starts[&format!("{k}.int8")] + 9, &[0]);
    poke(
        &source,
        starts[&format!("{k}.row_scales")],
        &f32::NAN.to_le_bytes(),
    );
    let refused = export_head(dir.path()).expect_err("NaN scale refused");
    assert!(refused.to_string().contains("row scale"), "{refused}");

    // Both formats for one matrix is ambiguous, not a choice to guess.
    let mut both = formats;
    both[3] = MatrixFormat::Ptq1;
    sparse_source(&source, &both);
    let mut header_json: serde_json::Value = {
        let bytes = std::fs::read(&source).expect("read source");
        let length = u64::from_le_bytes(bytes[..8].try_into().expect("length")) as usize;
        serde_json::from_slice(&bytes[8..8 + length]).expect("json")
    };
    header_json[format!("{k}.int8")] = serde_json::json!({
        "dtype": "I8", "shape": [KV, WIDTH], "data_offsets": [0, KV * WIDTH]
    });
    let json = serde_json::to_vec(&header_json).expect("json");
    let mut framed = (json.len() as u64).to_le_bytes().to_vec();
    framed.extend_from_slice(&json);
    std::fs::write(&source, &framed).expect("write ambiguous header");
    let refused = head_sections(&source).err().expect("both formats refused");
    assert!(
        refused.to_string().contains("both as ternary and as int8"),
        "{refused}"
    );
}

/// CPU forward transform: per 1024 block, `H (s ⊙ x) / 32` in natural order.
fn forward_rotation(input: &[f32], signs: &[f32]) -> Vec<f64> {
    let mut output: Vec<f64> = input
        .iter()
        .zip(signs)
        .map(|(&x, &s)| f64::from(x) * f64::from(s) / 32.0)
        .collect();
    for block in output.as_chunks_mut::<1024>().0 {
        let mut half = 1;
        while half < 1024 {
            for start in (0..1024).step_by(2 * half) {
                for i in start..start + half {
                    let (a, b) = (block[i], block[i + half]);
                    block[i] = a + b;
                    block[i + half] = a - b;
                }
            }
            half *= 2;
        }
    }
    output
}

/// Real checkpoint blocks repack byte for byte: the packer is Prism's encoder,
/// not merely something the decoder accepts.
#[test]
#[ignore = "requires the Bonsai GGUF"]
fn ptq1_packer_reproduces_checkpoint_blocks() {
    let package = crate::bonsai::BonsaiPackage::open(crate::bonsai::DEFAULT_BONSAI_GGUF)
        .expect("open Bonsai GGUF");
    for name in [
        "blk.3.attn_q.weight",
        "blk.0.ffn_down.weight",
        "output.weight",
    ] {
        let bytes = package.bytes(name).expect("tensor bytes");
        let blocks = &bytes[..(bytes.len() / PTQ1_BLOCK_BYTES).min(200_000) * PTQ1_BLOCK_BYTES];
        let mut checked = 0;
        for block in blocks.as_chunks::<PTQ1_BLOCK_BYTES>().0 {
            let scale_bits = u16::from_le_bytes([block[26], block[27]]);
            let mut values = [0.0_f32; PTQ1_BLOCK_ELEMENTS];
            local_metal::bonsai::decode_ptq1_row(block, &mut values).expect("decode");
            let scale = half::f16::from_bits(scale_bits).to_f32();
            if scale == 0.0 {
                continue;
            }
            let codes = values.map(|value| (value / scale).round() as i8);
            assert_eq!(&pack_ptq1_block(&codes, scale_bits), block, "{name}");
            checked += 1;
        }
        assert!(checked > 1000, "{name}: {checked}");
    }
}

/// A ternary head projection on Metal — the target's forward rotation then its
/// PTQ1 kernels, as `head/step.rs` runs them — equals the contract
/// `y = (codes · scale) · (R x)` computed on the CPU, at every input width the
/// head uses and at the row counts drafts, verify blocks and prefill chunks use.
#[test]
#[ignore = "requires the Bonsai GGUF and a Metal device"]
#[allow(clippy::too_many_lines)]
fn ternary_projection_matches_cpu_reference() {
    use local_metal::batch::CommandBatch;
    use local_metal::bonsai::{BonsaiKernels, HadamardDirection, Ptq1Matrix};
    use local_metal::buffer::MetalBuffer;
    use local_metal::context::MetalContext;
    use local_metal::shaders::ShaderLibrary;

    let package = crate::bonsai::BonsaiPackage::open(crate::bonsai::DEFAULT_BONSAI_GGUF)
        .expect("open Bonsai GGUF");
    let context = MetalContext::new().expect("Metal");
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let kernels = BonsaiKernels::new(&context, &shaders).expect("kernels");
    let rows = 72;
    let mut state = 0x2545_f491_4f6c_dd1d;
    let mut worst = 0.0_f64;
    for (columns, tokens) in [
        (WIDTH, 1),
        (WIDTH, 2),
        (WIDTH, 5),
        (WIDTH, 128),
        (ATTENTION, 1),
        (ATTENTION, 4),
        (FFN, 1),
        (FFN, 3),
        (FFN, 40),
    ] {
        let signs = package.hadamard().signs(columns as u32).expect("signs");
        let rotation = package
            .hadamard()
            .upload(&context, columns as u32)
            .expect("upload signs");
        let mut matrices = Vec::new();
        for _ in 0..2 {
            let codes = random_codes(&mut state, rows * columns);
            let scales: Vec<half::f16> = (0..rows * columns / 128)
                .map(|_| {
                    half::f16::from_f32(((xorshift(&mut state) % 1000) as f32).mul_add(4e-5, 0.002))
                })
                .collect();
            let packed = pack_ptq1_matrix(
                &codes.iter().map(|&code| code as u8).collect::<Vec<_>>(),
                &scales
                    .iter()
                    .flat_map(|s| s.to_le_bytes())
                    .collect::<Vec<_>>(),
                rows,
                columns,
            )
            .expect("pack");
            let buffer = MetalBuffer::from_slice(context.device(), &packed).expect("upload");
            matrices.push((codes, scales, buffer));
        }
        let input: Vec<f32> = (0..tokens * columns)
            .map(|_| (xorshift(&mut state) % 20_001) as f32 / 10_000.0 - 1.0)
            .collect();
        let input_buffer = MetalBuffer::from_slice(context.device(), &input).expect("input");
        let rotated = MetalBuffer::empty(context.device(), input.len() * 4).expect("rotated");
        let output = MetalBuffer::empty(context.device(), tokens * rows * 4).expect("output");
        let fused = MetalBuffer::empty(context.device(), rows * 4).expect("fused");
        let matrix = |index: usize| {
            Ptq1Matrix::new(&matrices[index].2, 0, rows as u32, columns as u32).expect("view")
        };
        let mut batch = CommandBatch::new(&context).expect("batch");
        kernels
            .transform(
                &mut batch,
                &rotation,
                &input_buffer,
                &rotated,
                tokens as u32,
                HadamardDirection::Forward,
            )
            .expect("rotate");
        kernels
            .matmul(&mut batch, matrix(0), &rotated, &output, tokens as u32)
            .expect("matmul");
        if tokens == 1 {
            kernels
                .matvec_swiglu(&mut batch, matrix(0), matrix(1), &rotated, &fused)
                .expect("swiglu");
        }
        batch.commit_and_wait().expect("run");

        let project = |index: usize, rotated: &[f64], row: usize| -> f64 {
            let (codes, scales, _) = &matrices[index];
            (0..columns)
                .map(|c| {
                    f64::from(codes[row * columns + c])
                        * f64::from(scales[(row * columns + c) / 128].to_f32())
                        * rotated[c]
                })
                .sum()
        };
        let gpu = output.as_slice::<f32>();
        for token in 0..tokens {
            let reference = forward_rotation(&input[token * columns..][..columns], signs);
            let expected: Vec<f64> = (0..rows).map(|row| project(0, &reference, row)).collect();
            let peak = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for (row, value) in expected.iter().enumerate() {
                let error = (f64::from(gpu[token * rows + row]) - value).abs() / peak;
                worst = worst.max(error);
                assert!(
                    error < 1e-4,
                    "width {columns} tokens {tokens} row {row}: {} vs {value}",
                    gpu[token * rows + row]
                );
            }
            if tokens == 1 {
                let fused = fused.as_slice::<f32>();
                let values: Vec<f64> = expected
                    .iter()
                    .enumerate()
                    .map(|(row, gate)| gate / (1.0 + (-gate).exp()) * project(1, &reference, row))
                    .collect();
                let peak = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                for (row, value) in values.iter().enumerate() {
                    let error = (f64::from(fused[row]) - value).abs() / peak;
                    assert!(error < 1e-4, "fused width {columns} row {row}");
                }
            }
        }
    }
    eprintln!("worst relative projection error {worst:e}");
}
