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
