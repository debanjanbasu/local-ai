use std::path::Path;

use super::*;

fn header(overrides: &[(&str, serde_json::Value)]) -> (Vec<u8>, usize) {
    let mut offset = 0usize;
    let mut entries = serde_json::Map::new();
    for (name, shape) in TENSORS {
        let bytes = shape.iter().product::<usize>() * 2;
        entries.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + bytes]
            }),
        );
        offset += bytes;
    }
    entries.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    for (name, value) in overrides {
        if value.is_null() {
            entries.remove(*name);
        } else {
            entries.insert((*name).to_owned(), value.clone());
        }
    }
    (
        serde_json::to_vec(&serde_json::Value::Object(entries)).expect("json"),
        offset,
    )
}

#[test]
fn pinned_layout_accepts_exact_head_and_reports_sizes() {
    let (json, data_len) = header(&[]);
    let layout = SafetensorsLayout::parse(&json, data_len).expect("pinned layout");
    assert_eq!(layout.tensors.len(), 15);
    let fc = layout.range("mtp.fc.weight").expect("fc");
    assert_eq!(fc.end - fc.start, WIDTH * 2 * WIDTH * 2);
    let total: usize = layout
        .tensors
        .iter()
        .map(|range| range.end - range.start)
        .sum();
    // 424,699,392 BF16 parameters. The published file is 849,400,392 bytes,
    // leaving 1,608 bytes for the 8-byte length prefix and JSON header.
    assert_eq!(total, 849_398_784);
    assert!(849_400_392 - 8 - total < SAFETENSORS_MAX_HEADER as usize);

    // The artifact's section table: nine int8 matrices, their nine f32 scale
    // vectors, then seven folded norms.
    let spec = spec();
    assert_eq!(spec.len(), CACHE_SECTIONS);
    assert_eq!(spec[6].0, "mtp.layers.0.self_attn.k_proj.weight.i8");
    assert_eq!(spec[8].0, "mtp.layers.0.self_attn.v_proj.weight.i8");
    // `k_proj` and `v_proj` are both `[KV, WIDTH]`, so no pair of sizes can tell
    // them apart. That is why the header records names: swapping them in the
    // specs would otherwise validate and bind the wrong weights.
    assert_eq!(spec[6].1, spec[8].1);
    assert_eq!(spec[6].1, KV * WIDTH);
    assert_eq!(
        spec.iter().map(|&(_, size)| size).sum::<usize>(),
        425_056_256
    );
}

#[test]
fn layout_rejects_wrong_dtype_shape_missing_extra_and_gaps() {
    let (json, data_len) = header(&[]);
    assert!(
        SafetensorsLayout::parse(&json, data_len + 2).is_err(),
        "unused bytes"
    );
    assert!(
        SafetensorsLayout::parse(&json, data_len - 2).is_err(),
        "short file"
    );
    let f16 = header(&[(
        "mtp.norm.weight",
        serde_json::json!({"dtype": "F16", "shape": [WIDTH], "data_offsets": [data_len - WIDTH * 2, data_len]}),
    )]);
    assert!(SafetensorsLayout::parse(&f16.0, f16.1).is_err(), "dtype");
    let transposed = header(&[(
        "mtp.layers.0.self_attn.o_proj.weight",
        serde_json::json!({"dtype": "BF16", "shape": [ATTENTION, WIDTH], "data_offsets": [0, WIDTH * ATTENTION * 2]}),
    )]);
    assert!(
        SafetensorsLayout::parse(&transposed.0, transposed.1).is_err(),
        "shape order"
    );
    let missing = header(&[("mtp.pre_fc_norm_hidden.weight", serde_json::Value::Null)]);
    assert!(
        SafetensorsLayout::parse(&missing.0, missing.1).is_err(),
        "missing"
    );
    let extra = header(&[(
        "mtp.layers.1.input_layernorm.weight",
        serde_json::json!({"dtype": "BF16", "shape": [WIDTH], "data_offsets": [0, WIDTH * 2]}),
    )]);
    assert!(
        SafetensorsLayout::parse(&extra.0, extra.1).is_err(),
        "extra layer"
    );
    let overlap = header(&[(
        "mtp.pre_fc_norm_embedding.weight",
        serde_json::json!({"dtype": "BF16", "shape": [WIDTH], "data_offsets": [0, WIDTH * 2]}),
    )]);
    assert!(
        SafetensorsLayout::parse(&overlap.0, overlap.1).is_err(),
        "overlap"
    );
}

#[test]
fn framing_rejects_short_and_oversized_headers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mtp.safetensors");
    std::fs::write(&path, [1u8, 0, 0, 0]).expect("write");
    assert!(open_layout(&path).is_err());
    let mut framed = (SAFETENSORS_MAX_HEADER + 1).to_le_bytes().to_vec();
    framed.extend_from_slice(b"{}");
    std::fs::write(&path, &framed).expect("write");
    assert!(open_layout(&path).is_err());
    let (json, _) = header(&[]);
    let mut framed = (json.len() as u64).to_le_bytes().to_vec();
    framed.extend_from_slice(&json);
    framed.extend_from_slice(&[0u8; 16]);
    std::fs::write(&path, &framed).expect("write");
    // Header parses, but the data region is far shorter than the pinned tensors.
    assert!(open_layout(&path).is_err());
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
#[allow(clippy::too_many_lines)]
fn head_file_round_trips_and_rejects_damage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bf16: Vec<u8> = [-2.0_f32, -0.5, 0.0, 1.0, 3.0, 0.25]
        .into_iter()
        .flat_map(|value| half::bf16::from_f32(value).to_bits().to_le_bytes())
        .collect();
    let (weights, scales) = quantize_rows(&bf16, 2, 3).expect("quantize");
    let sections: Vec<Vec<u8>> = vec![
        weights.into_iter().map(|value| value as u8).collect(),
        scales.into_iter().flat_map(f32::to_le_bytes).collect(),
    ];
    let spec = vec![
        ("mtp.fc.weight.embedding.i8".to_owned(), sections[0].len()),
        (
            "mtp.fc.weight.embedding.scales".to_owned(),
            sections[1].len(),
        ),
    ];
    let head = write_cache(dir.path(), &spec, &sections).expect("write cache");
    // The name is the payload digest, so it means the same thing on every machine
    // and no mtime takes part in it.
    assert_eq!(
        head.path.file_name().and_then(|name| name.to_str()),
        Some(format!("mtp-head-v{CACHE_VERSION}-{}.bin", head.payload_sha256).as_str())
    );
    assert_eq!(head.payload_sha256.len(), 64);
    // The payload skips the padding, and the file skips nothing: one page holds
    // the header and every section is padded up to a page of its own.
    assert_eq!(head.payload_bytes, sections[0].len() + sections[1].len());
    assert_eq!(
        head.file_bytes,
        align_cache(CACHE_HEADER_BYTES).unwrap_or(0)
            + sections
                .iter()
                .map(|section| align_cache(section.len()).unwrap_or(0))
                .sum::<usize>()
    );
    assert_eq!(
        find_cache(dir.path()),
        Some(head.path.clone()),
        "the content address is the only candidate in the directory"
    );
    assert!(matches!(
        head_kind(&head.path).expect("classify"),
        HeadKind::Artifact
    ));

    let cached = read_cache(&head.path, &spec).expect("read cache");
    for ((offset, length), expected) in cached.sections.iter().zip(&sections) {
        assert_eq!(&cached.map[*offset..*offset + *length], expected);
    }
    assert_eq!(cached.payload_sha256, head.payload_sha256);

    // The shipped artifact is the same file under a fixed name, so it validates
    // without a digest in its filename and is not itself a cache entry.
    let artifact = dir.path().join(MTP_HEAD_ARTIFACT);
    std::fs::copy(&head.path, &artifact).expect("copy artifact");
    assert!(matches!(
        head_kind(&artifact).expect("classify"),
        HeadKind::Artifact
    ));
    assert!(read_head(&artifact, &spec).is_ok());
    assert!(
        read_cache(&artifact, &spec).is_err(),
        "a fixed artifact name is not a content address"
    );

    // A file too short to hold either format's discriminator is neither, and is
    // named as such rather than guessed at behind a warm cache.
    let stub = dir.path().join("stub.safetensors");
    std::fs::write(&stub, 1024_u64.to_le_bytes()).expect("write stub");
    assert!(matches!(
        head_kind(&stub).expect("classify"),
        HeadKind::Safetensors
    ));
    std::fs::write(&stub, b"stub").expect("write short stub");
    assert!(matches!(
        head_kind(&stub).expect("classify"),
        HeadKind::Unrecognizable(4)
    ));
    std::fs::write(&stub, b"").expect("write empty stub");
    assert!(matches!(
        head_kind(&stub).expect("classify"),
        HeadKind::Unrecognizable(0)
    ));

    // Sizes alone cannot bind sections; the names are what say which is which.
    let swapped = vec![
        (spec[1].0.clone(), spec[0].1),
        (spec[0].0.clone(), spec[1].1),
    ];
    let unbound = read_head(&head.path, &swapped)
        .err()
        .expect("swapped names");
    assert!(unbound.contains("section 0 is"), "{unbound}");

    let pristine = std::fs::read(&head.path).expect("read bytes");
    let damaged = |at: usize, delta: u8| -> Vec<u8> {
        let mut bytes = pristine.clone();
        bytes[at] ^= delta;
        bytes
    };
    let reason = |bytes: &[u8]| -> String {
        std::fs::write(&head.path, bytes).expect("write damage");
        read_cache(&head.path, &spec).err().expect("damaged cache")
    };
    let corrupt = damaged(0, 1);
    assert!(reason(&corrupt).contains("not an int8 MTP head"));
    // A v1 file is a different format, not a v2 file with a different name.
    let mut v1 = pristine.clone();
    v1[8] = 1;
    assert!(reason(&v1).contains("reads version 2"));
    // Flipping a payload byte changes the content identity, which is what the
    // header digest exists to catch.
    let payload_at = cached.sections[0].0 + 1;
    assert!(reason(&damaged(payload_at, 0xff)).contains("payload hashes to"));
    std::fs::write(&head.path, &pristine).expect("restore");
    assert!(read_cache(&head.path, &spec).is_ok());
    std::fs::write(&head.path, &pristine[..CACHE_HEADER_BYTES / 2]).expect("truncate");
    assert!(read_cache(&head.path, &spec).is_err());

    // The writer reclaims older generations, mtime-keyed v1 files included.
    let legacy = dir
        .path()
        .join("mtp-head-v1-849400392-1790964268527986261.bin");
    std::fs::write(&legacy, b"stale").expect("write legacy");
    write_cache(dir.path(), &spec, &sections).expect("rebuild cache");
    assert!(!legacy.exists(), "v1 files must be reclaimed");
    assert!(read_cache(&head.path, &spec).is_ok());
}

#[test]
#[allow(clippy::too_many_lines)]
fn compressed_head_inflates_to_the_stored_head_byte_for_byte() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Several sections with incompressible content, so the frame is a real frame
    // carrying the padding between them: one repeated byte compresses to nothing
    // and would not exercise that.
    let sections: Vec<Vec<u8>> = (0..4)
        .map(|index| {
            (0..(3000 + index * 997))
                .map(|at: usize| u8::try_from(at.wrapping_mul(31) ^ (index << 5)).unwrap_or(0))
                .collect()
        })
        .collect();
    let spec: Vec<Section> = sections
        .iter()
        .enumerate()
        .map(|(index, section)| (format!("mtp.layer.0.proj{index}.i8"), section.len()))
        .collect();

    let stored_path = dir.path().join("stored.bin");
    let zstd_path = dir.path().join("compressed.bin");
    let stored = write_artifact(&stored_path, &spec, &sections).expect("write stored");
    let zstd = write_zstd_artifact(&zstd_path, &spec, &sections).expect("write zstd");

    // Both describe one head, and each says which encoding it is in its own first
    // eight bytes: no filename convention, nothing outside the file takes part.
    assert!(matches!(
        head_kind(&stored_path).expect("classify stored"),
        HeadKind::Artifact
    ));
    assert!(matches!(
        head_kind(&zstd_path).expect("classify zstd"),
        HeadKind::ZstdArtifact
    ));
    assert_eq!(
        Codec::from_magic(magic_of(&stored_path)).map(Codec::name),
        Some("stored")
    );
    assert_eq!(
        Codec::from_magic(magic_of(&zstd_path)).map(Codec::name),
        Some("zstd")
    );

    // Same head, smaller file: the frame carries the inter-section padding, so
    // what it decodes to is the whole canonical image and not just the payload.
    assert_eq!(stored.canonical_bytes, zstd.canonical_bytes);
    assert_eq!(stored.file_bytes, stored.canonical_bytes);
    assert!(
        zstd.file_bytes < zstd.canonical_bytes,
        "a frame must shrink the file"
    );
    assert_eq!(stored.payload_bytes, zstd.payload_bytes);
    assert_eq!(stored.payload_sha256, zstd.payload_sha256);
    assert_eq!(
        zstd.canonical_bytes,
        CACHE_PAGE
            + sections
                .iter()
                .map(|section| align_cache(section.len()).unwrap_or(0))
                .sum::<usize>()
    );

    // The point of the format: inflating gives the same sections at the same
    // offsets, and the digest both headers carry still describes them.
    let inflated = read_head(&zstd_path, &spec).expect("read zstd head");
    let mapped = read_head(&stored_path, &spec).expect("read stored head");
    assert_eq!(inflated.sections, mapped.sections, "offsets must not move");
    assert_eq!(inflated.payload_sha256, mapped.payload_sha256);
    assert_eq!(inflated.payload_sha256, zstd.payload_sha256);
    for ((offset, length), expected) in inflated.sections.iter().zip(&sections) {
        assert_eq!(&inflated.map[*offset..*offset + *length], expected);
        assert_eq!(&mapped.map[*offset..*offset + *length], expected);
    }

    // Damage is a hard error rather than a silently shorter head: the frame no
    // longer decodes to the length its own header describes.
    let corrupt = |bytes: Vec<u8>| {
        let mut bytes = bytes;
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&zstd_path, &bytes).expect("write damage");
        read_head(&zstd_path, &spec).err().expect("damaged frame")
    };
    let damaged = corrupt(std::fs::read(&zstd_path).expect("read zstd"));
    assert!(damaged.contains("zstd frame"), "{damaged}");
    // So is a file whose length is not the one it recorded.
    let frame = std::fs::read(&zstd_path).expect("read zstd");
    std::fs::write(&zstd_path, &frame[..frame.len() - 8]).expect("truncate");
    let truncated = read_head(&zstd_path, &spec).err().expect("truncated frame");
    assert!(truncated.contains("header records"), "{truncated}");
    // And the magic is not decorative. Byte 3 is the codec letter, so claiming
    // the compressed framing over stored sections means inflating bytes that are
    // not a frame at all.
    let mut mislabelled = std::fs::read(&stored_path).expect("read stored");
    mislabelled[3] = b'Z';
    std::fs::write(&zstd_path, &mislabelled).expect("write");
    assert!(matches!(
        head_kind(&zstd_path).expect("classify mislabelled"),
        HeadKind::ZstdArtifact
    ));
    let mislabelled = read_head(&zstd_path, &spec)
        .err()
        .expect("stored sections as a frame");
    assert!(
        mislabelled.contains("zstd frame is unusable"),
        "{mislabelled}"
    );
    // The other way round is caught by the framing instead: a frame is far
    // shorter than the sections its own header says it holds.
    let mut mislabelled = frame;
    mislabelled[3] = b'Q';
    std::fs::write(&zstd_path, &mislabelled).expect("write");
    assert!(matches!(
        head_kind(&zstd_path).expect("classify mislabelled"),
        HeadKind::Artifact
    ));
    let mislabelled = read_head(&zstd_path, &spec).err().expect("frame as stored");
    assert!(
        mislabelled.contains("sections end at byte"),
        "{mislabelled}"
    );
}

fn magic_of(path: &Path) -> [u8; 8] {
    let mut magic = [0_u8; 8];
    std::fs::File::open(path)
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
        .expect("read magic");
    magic
}

#[test]
fn fc_split_keeps_embedding_columns_first() {
    let row_bytes = 2 * WIDTH * 2;
    let mut fused = vec![0u8; WIDTH * row_bytes];
    // Row 3: embedding half filled with 0x11, hidden half with 0x22.
    fused[3 * row_bytes..3 * row_bytes + WIDTH * 2].fill(0x11);
    fused[3 * row_bytes + WIDTH * 2..4 * row_bytes].fill(0x22);
    let (embedding, hidden) = split_fc(&fused).expect("split");
    assert_eq!(embedding.len(), WIDTH * WIDTH * 2);
    assert_eq!(embedding[3 * WIDTH * 2], 0x11);
    assert_eq!(hidden[3 * WIDTH * 2], 0x22);
    assert_eq!(embedding[4 * WIDTH * 2 - 1], 0x11);
    assert_eq!(hidden[2 * WIDTH * 2], 0);
    assert!(split_fc(&fused[..fused.len() - 2]).is_err());
}

#[test]
fn settings_bound_depth() {
    assert!(MtpSettings::new(PathBuf::from("head"), 0).is_err());
    assert!(MtpSettings::new(PathBuf::from("head"), MAX_MTP_DEPTH + 1).is_err());
    // With the int8 head, gated depth 3 beat depth 2 on every measured
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
    let installed = dir.path().join("model_mtp.safetensors");
    let missing = dir.path().join("absent.safetensors");
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
    assert!(reason.contains("absent.safetensors"), "{reason}");
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
        MtpMode::Off
            .resolve_with_default(&installed, true)
            .expect("off"),
        MtpResolution::Disabled(reason) if reason == MTP_OFF_REASON
    ));

    // An artifact beside the default head replaces the 849 MB source entirely,
    // which is what makes an artifact-only install work.
    let artifact = dir.path().join(MTP_HEAD_ARTIFACT);
    std::fs::write(&artifact, b"stub").expect("write artifact");
    let MtpResolution::Native(settings) = auto
        .resolve_with_default(&missing, true)
        .expect("artifact alone speculates")
    else {
        unreachable!("an installed artifact must enable speculation without the source");
    };
    assert_eq!(settings.path, artifact);
    // It also wins when the source is present, so discovery names one winner.
    let MtpResolution::Native(both) = auto
        .resolve_with_default(&installed, true)
        .expect("artifact preferred")
    else {
        unreachable!("the artifact must be preferred over the source");
    };
    assert_eq!(both.path, artifact);
    // The default artifact path is pinned to the default head's directory, under
    // the one artifact filename the cache format version names.
    assert_eq!(
        Path::new(DEFAULT_BONSAI_MTP_ARTIFACT).file_name(),
        Some(Path::new(MTP_HEAD_ARTIFACT).as_os_str())
    );
    assert_eq!(
        Path::new(DEFAULT_BONSAI_MTP_ARTIFACT).parent(),
        Path::new(DEFAULT_BONSAI_MTP_HEAD).parent()
    );
    assert!(
        MTP_HEAD_ARTIFACT.ends_with(&format!("-v{CACHE_VERSION}.bin")),
        "the artifact filename must track the cache format version"
    );
}

#[test]
fn explicit_head_overrides_the_default() {
    let dir = tempfile::tempdir().expect("tempdir");
    let installed = dir.path().join("model_mtp.safetensors");
    let default_head = dir.path().join("default.safetensors");
    let absent_path = dir.path().join("absent.safetensors");
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
        MtpMode::Off
            .resolve_with_default(&default_head, true)
            .expect("off"),
        MtpResolution::Disabled(MTP_OFF_REASON.into())
    );
}
