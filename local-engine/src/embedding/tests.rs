#![allow(clippy::expect_used)]

use super::*;

#[test]
fn prompts_follow_the_model_card() {
    assert_eq!(
        EmbeddingPrompt::Query {
            task: "search result"
        }
        .format("rain"),
        "task: search result | query: rain"
    );
    assert_eq!(
        EmbeddingPrompt::Document { title: None }.format("body"),
        "title: none | text: body"
    );
    assert_eq!(
        EmbeddingPrompt::Document { title: Some("  ") }.format("body"),
        "title: none | text: body"
    );
    assert_eq!(
        EmbeddingPrompt::Document {
            title: Some("Notes")
        }
        .format("body"),
        "title: Notes | text: body"
    );
}

#[test]
fn matryoshka_truncation_renormalizes() {
    let full = [3.0_f32, 4.0, 12.0];
    let short = truncate_normalized(&full, 2);
    assert!((short[0] - 0.6).abs() < 1e-6 && (short[1] - 0.8).abs() < 1e-6);
    assert_eq!(truncate_normalized(&full, 10).len(), 3);
}

#[test]
fn ids_are_validated() {
    assert!(EmbeddingModel::validate_ids(&[]).is_err());
    assert!(EmbeddingModel::validate_ids(&[2, 300, 1]).is_ok());
    assert!(EmbeddingModel::validate_ids(&[2, 258_880, 1]).is_err());
    assert!(EmbeddingModel::validate_ids(&[2, VOCAB as u32, 1]).is_err());
    assert!(matches!(
        EmbeddingModel::validate_ids(&vec![5; MAX_INPUT_TOKENS + 1]),
        Err(Error::ContextOverflow(_))
    ));
}

fn base64_decode(text: &str) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut buffer, mut bits) = (0_u32, 0);
    for byte in text.bytes().filter(|b| *b != b'=') {
        let value = ALPHABET.iter().position(|a| *a == byte).expect("base64") as u32;
        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
        }
    }
    out
}

/// The reference fixture: sentence-transformers 6.1.0 / transformers 5.19.0
/// FP32 on Kaggle (97 of the 1,000 parity texts: every edge case — empty,
/// whitespace, scripts, emoji, code, special-token text, inputs truncated at
/// 512 tokens — and a sample of `SciFact`, `NFCorpus` and `CosQA` queries and
/// documents). Token ids must be identical and the native embedding of the
/// reference ids within the parity gate (mean cosine ≥ 0.9995, min ≥ 0.999).
#[test]
#[ignore = "requires the installed embedding model"]
#[allow(clippy::cast_precision_loss)]
fn native_embeddings_match_the_reference_fixture() {
    #[derive(serde::Deserialize)]
    struct Row {
        text: String,
        ids: Vec<u32>,
        embedding_f32le_b64: String,
    }
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let fixture = format!("{root}/local-engine/tests/fixtures/embeddinggemma2-reference.jsonl");
    let rows: Vec<Row> = std::fs::read_to_string(fixture)
        .expect("fixture")
        .lines()
        .map(|line| serde_json::from_str(line).expect("row"))
        .collect();
    let dir = installed_model(&Path::new(root).join("models")).expect("installed model");
    let mut model = EmbeddingModel::open(&dir).expect("open");
    for row in &rows {
        let mut ids = model.tokenize(&row.text).expect("tokenize");
        if ids.len() > 512 {
            // The reference ran with max_seq_length 512: <bos>, 510 tokens, <eos>.
            ids.truncate(511);
            ids.push(1);
        }
        assert_eq!(ids, row.ids, "token ids of {:?}", row.text);
    }
    let inputs: Vec<Vec<u32>> = rows.iter().map(|row| row.ids.clone()).collect();
    let embeddings = model.embed_ids(&inputs, &mut || Ok(())).expect("embed");
    let cosines: Vec<f32> = embeddings
        .iter()
        .zip(&rows)
        .map(|(native, row)| {
            let reference: Vec<f32> = base64_decode(&row.embedding_f32le_b64)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(reference.len(), EMBEDDING_DIMENSIONS);
            native.iter().zip(&reference).map(|(a, b)| a * b).sum()
        })
        .collect();
    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(mean >= 0.9995 && min >= 0.999, "mean {mean}, min {min}");
}

/// One piece of a reference input's text: text, or a media placeholder.
#[derive(Debug, PartialEq, Eq)]
enum Piece<'a> {
    Text(&'a str),
    Image,
    Audio,
}

/// `text` split at its `<|image|>` and `<|audio|>` placeholders, in order.
fn placeholder_pieces(mut text: &str) -> Vec<Piece<'_>> {
    let mut pieces = Vec::new();
    loop {
        let image = text.split_once("<|image|>");
        let audio = text.split_once("<|audio|>");
        let (before, after, media) = match (image, audio) {
            (Some(i), Some(a)) if a.0.len() < i.0.len() => (a.0, a.1, Piece::Audio),
            (Some(i), _) => (i.0, i.1, Piece::Image),
            (None, Some(a)) => (a.0, a.1, Piece::Audio),
            (None, None) => break,
        };
        if !before.is_empty() {
            pieces.push(Piece::Text(before));
        }
        pieces.push(media);
        text = after;
    }
    if !text.is_empty() {
        pieces.push(Piece::Text(text));
    }
    pieces
}

#[test]
fn placeholders_split_text_in_order() {
    assert_eq!(
        placeholder_pieces("Picture <|image|> and sound <|audio|> together."),
        [
            Piece::Text("Picture "),
            Piece::Image,
            Piece::Text(" and sound "),
            Piece::Audio,
            Piece::Text(" together.")
        ]
    );
    assert_eq!(placeholder_pieces("<|audio|>"), [Piece::Audio]);
}

/// Mixed-modality inputs (text with an image, text with an audio clip, and
/// both) against the FP32 reference: sentence-transformers 6.1.0 /
/// transformers 5.19.0 on Kaggle (`eg2-mixedref`), each input one sequence
/// with the media placed by the model's `<|image|>` / `<|audio|>`
/// placeholders. The native layout must give the reference's token ids
/// (soft-token slots aside), and the embeddings must pass the parity gate
/// (mean cosine ≥ 0.9995, min ≥ 0.999).
#[test]
#[ignore = "requires the installed embedding model with the vision and audio artifacts"]
#[allow(clippy::cast_precision_loss)]
fn native_mixed_modality_embeddings_match_the_reference_fixture() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        text: String,
        image: Option<String>,
        audio: Option<String>,
        ids: Vec<u32>,
        embedding: Vec<f32>,
    }
    #[derive(serde::Deserialize)]
    struct AudioRow {
        name: String,
        wave_f32le_b64: Option<String>,
    }
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let fixtures = format!("{root}/local-engine/tests/fixtures");
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(format!("{fixtures}/embeddinggemma2-mixed-reference.json"))
            .expect("fixture"),
    )
    .expect("fixture json");
    // The clips: a generated 1 s 1 kHz tone and an AudioCaps crop whose
    // waveform the audio fixture holds.
    let tone: Vec<f32> = (0..16_000)
        .map(|i| {
            let t = f64::from(i) / 16_000.0;
            (0.5 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()) as f32
        })
        .collect();
    let crop: Vec<f32> =
        std::fs::read_to_string(format!("{fixtures}/embeddinggemma2-audio-reference.jsonl"))
            .expect("audio fixture")
            .lines()
            .map(|line| serde_json::from_str::<AudioRow>(line).expect("row"))
            .find(|row| row.name == "clip70-crop4000")
            .and_then(|row| row.wave_f32le_b64)
            .map(|wave| {
                base64_decode(&wave)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            })
            .expect("crop waveform");
    let dir = installed_model(&Path::new(root).join("models")).expect("installed model");
    let mut model = EmbeddingModel::open(&dir).expect("open");
    let mut cosines = Vec::new();
    for case in &fixture.cases {
        let image = case.image.as_ref().map(|file| {
            let bytes = std::fs::read(format!("{fixtures}/vision/{file}")).expect("image");
            prepare_image(&bytes).expect("prepare")
        });
        let clip: Option<&[f32]> = case.audio.as_deref().map(|name| {
            match name {
                "tone1k-1s" => Some(tone.as_slice()),
                "clip70-crop4000" => Some(crop.as_slice()),
                _ => None,
            }
            .expect("a known clip")
        });
        let parts: Vec<MediaPart<'_>> = placeholder_pieces(&case.text)
            .into_iter()
            .map(|piece| match piece {
                Piece::Text(text) => MediaPart::Text(text),
                Piece::Image => MediaPart::Image(image.as_ref().expect("case image")),
                Piece::Audio => MediaPart::Audio(clip.expect("case clip")),
            })
            .collect();
        let layout = model.multimodal_layout(&parts).expect("layout");
        let expected: Vec<Option<u32>> = case
            .ids
            .iter()
            .map(|id| (!MEDIA_PLACEHOLDERS.contains(id)).then_some(*id))
            .collect();
        assert_eq!(layout, expected, "token layout of {}", case.name);
        let native = model
            .embed_multimodal(&[parts], &mut || Ok(()))
            .expect("embed")
            .pop()
            .expect("one embedding");
        assert_eq!(case.embedding.len(), EMBEDDING_DIMENSIONS);
        let norm = case.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine: f32 = native
            .iter()
            .zip(&case.embedding)
            .map(|(a, b)| a * b / norm)
            .sum();
        eprintln!("{}: cosine {cosine:.6}", case.name);
        cosines.push(cosine);
    }
    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(mean >= 0.9995 && min >= 0.999, "mean {mean}, min {min}");
}
