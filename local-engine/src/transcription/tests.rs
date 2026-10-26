#![allow(clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::path::Path;

use super::*;

#[test]
fn installation_needs_the_artifact_and_the_tokenizer() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(installed_model(dir.path()).is_none());
    let model = dir.path().join(TRANSCRIPTION_MODEL_DIR);
    std::fs::create_dir_all(&model).expect("mkdir");
    std::fs::write(model.join(TRANSCRIPTION_ARTIFACT), b"x").expect("write");
    assert!(installed_model(dir.path()).is_none());
    std::fs::write(model.join("tokenizer.json"), b"{}").expect("write");
    assert_eq!(installed_model(dir.path()), Some(model));
}

#[derive(serde::Deserialize)]
struct Fixture {
    tokenizer: TokenizerFacts,
    case: Case,
}

#[derive(serde::Deserialize)]
struct TokenizerFacts {
    encode: BTreeMap<String, Vec<u32>>,
    suppressed: Vec<u32>,
    language_codes: Vec<String>,
}

#[derive(serde::Deserialize)]
struct Case {
    audio: String,
    language: String,
    sequence: Vec<u32>,
    sample_begin: usize,
    decoded: Vec<u32>,
    steps: Vec<Step>,
    language_top: Vec<(String, f64)>,
}

#[derive(serde::Deserialize)]
struct Step {
    top_ids: Vec<u32>,
    top_logits: Vec<f32>,
}

/// The reference fixture: openai-whisper 20250625, FP32 on a Kaggle CPU, on
/// `LibriSpeech` test-clean 237-134493-0010 (CC BY 4.0) and the tokenizer
/// facts of the original package. The tokenizer must agree exactly, the
/// greedy decode must be token-identical, and the teacher-forced logits must
/// keep the reference's argmax at every step and its top logits within 0.05.
#[test]
#[ignore = "requires the installed Whisper model"]
fn native_whisper_matches_the_reference_fixture() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let fixtures = format!("{root}/local-engine/tests/fixtures");
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(format!("{fixtures}/whisper-large-v3-reference.json"))
            .expect("fixture"),
    )
    .expect("parse");
    let dir = installed_model(&Path::new(root).join("models")).expect("installed model");
    let mut model = TranscriptionModel::open(&dir).expect("open");
    for (text, ids) in &fixture.tokenizer.encode {
        assert_eq!(&model.encode_text(text).expect("encode"), ids, "{text:?}");
        assert_eq!(&model.decode_text(ids), text);
    }
    assert_eq!(
        model.tokenizer.special.suppressed(),
        fixture.tokenizer.suppressed
    );
    let codes: Vec<String> = model.tokenizer.special.languages.keys().cloned().collect();
    let mut expected = fixture.tokenizer.language_codes.clone();
    expected.sort();
    assert_eq!(codes, expected);

    let case = &fixture.case;
    let bytes = std::fs::read(format!("{fixtures}/{}", case.audio)).expect("audio");
    let samples = audio::decode(bytes, Some("flac"), 60).expect("decode");
    let mel = model.log_mel(&samples, mel::WINDOW_SAMPLES);
    let window = mel.window(0, (mel.frames - WINDOW_FRAMES).min(WINDOW_FRAMES));
    let busy = || false;
    let mut pause = || Ok(());
    let mut pace = Pace {
        busy: &busy,
        pause: &mut pause,
    };
    model.encoder_output(&window, &mut pace).expect("encode");
    let languages = model.window_languages(&mut pace).expect("languages");
    assert_eq!(languages[0].0, case.language_top[0].0);
    assert!((languages[0].1 - case.language_top[0].1).abs() < 1e-3);
    let decoded = model
        .decode_window(Task::Transcribe, &case.language, &[], 0.0, &mut pace)
        .expect("decode");
    assert_eq!(decoded, case.decoded);
    let logits = model
        .sequence_logits(&case.sequence, &mut pace)
        .expect("logits");
    for (i, step) in case.steps.iter().enumerate() {
        let row = &logits[case.sample_begin - 1 + i];
        let argmax = row
            .iter()
            .enumerate()
            .fold(0, |best, (j, v)| if *v > row[best] { j } else { best });
        assert_eq!(argmax as u32, step.top_ids[0], "step {i}");
        for (id, value) in step.top_ids.iter().zip(&step.top_logits) {
            let diff = (row[*id as usize] - value).abs();
            assert!(
                diff < 0.05,
                "step {i} token {id}: {} vs {value}",
                row[*id as usize]
            );
        }
    }
}
