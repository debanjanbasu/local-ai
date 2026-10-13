#![allow(clippy::expect_used)]

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
