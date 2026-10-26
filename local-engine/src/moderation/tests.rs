use serde_json::{Value, json};

use super::*;

const PAYLOAD: &str = "a7c15de94419f566ba448f59168778b1aa4c7f348fd944283f8b49a85d4296c5";

fn sha(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes).as_slice())
}

/// A head whose logit is `scale * x[0] + bias`.
fn head(bias: f64, threshold: f64) -> Value {
    let mut weight = vec![0.0_f32; EMBEDDING_DIMENSIONS];
    weight[0] = 4.0;
    json!({"weight":weight,"bias":bias,"platt_scale":1.0,"platt_offset":0.0,"threshold":threshold})
}

fn artifact() -> Value {
    let mut categories = serde_json::Map::new();
    for name in CATEGORIES {
        categories.insert(name.into(), json!({"unsupported":"no data"}));
    }
    categories.insert("hate".into(), head(0.0, 0.5));
    categories.insert("sexual".into(), head(-2.0, 0.9));
    json!({
        "format":FORMAT,"version":1,"model":MODERATION_MODEL_ID,"embedding_model":EMBEDDING_MODEL_ID,
        "embedding_payload_sha256":PAYLOAD,"prompt":PROMPT,"window_tokens":WINDOW_TOKENS,
        "dimensions":EMBEDDING_DIMENSIONS,"categories":categories,"provenance":{},"evaluation":{},
    })
}

fn load(value: &Value) -> crate::Result<ModerationHeads> {
    let bytes = serde_json::to_vec(value).expect("json");
    ModerationHeads::from_bytes(&bytes, &sha(&bytes), PAYLOAD)
}

fn unit(x0: f32) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBEDDING_DIMENSIONS];
    v[0] = x0;
    v
}

#[test]
fn a_valid_artifact_loads_with_its_supported_categories() {
    let heads = load(&artifact()).expect("valid");
    assert_eq!(heads.supported().collect::<Vec<_>>(), ["hate", "sexual"]);
    assert_eq!(heads.unsupported_reason("violence"), Some("no data"));
    assert!(heads.unsupported_reason("hate").is_none());
    assert!(heads.head("violence").is_none());
}

#[test]
fn hashes_are_validated() {
    let bytes = serde_json::to_vec(&artifact()).expect("json");
    let error = ModerationHeads::from_bytes(&bytes, &"0".repeat(64), PAYLOAD).expect_err("pin");
    assert!(error.to_string().contains("pinned"), "{error}");
    let error = ModerationHeads::from_bytes(&bytes, &sha(&bytes), &"1".repeat(64))
        .expect_err("encoder payload");
    assert!(error.to_string().contains("embedding payload"), "{error}");
    let mut tampered = bytes.clone();
    let at = tampered.len() - 2;
    tampered[at] = b' ';
    assert!(ModerationHeads::from_bytes(&tampered, &sha(&bytes), PAYLOAD).is_err());
}

#[test]
fn malformed_artifacts_are_refused() {
    let mut cases = Vec::new();
    let mut short = artifact();
    short["categories"]["hate"]["weight"] = json!([1.0, 2.0]);
    cases.push(short);
    for (field, value) in [
        ("threshold", json!(1.0)),
        ("threshold", json!(0.0)),
        ("platt_scale", json!(0.0)),
        ("platt_scale", json!(-1.0)),
    ] {
        let mut bad = artifact();
        bad["categories"]["hate"][field] = value;
        cases.push(bad);
    }
    let mut missing = artifact();
    missing["categories"]
        .as_object_mut()
        .expect("map")
        .remove("violence/graphic");
    cases.push(missing);
    let mut extra = artifact();
    extra["categories"]["toxicity"] = json!({"unsupported":"x"});
    cases.push(extra);
    let mut empty_reason = artifact();
    empty_reason["categories"]["violence"] = json!({"unsupported":" "});
    cases.push(empty_reason);
    let mut none = artifact();
    none["categories"]["hate"] = json!({"unsupported":"x"});
    none["categories"]["sexual"] = json!({"unsupported":"x"});
    cases.push(none);
    for (field, value) in [
        ("prompt", json!("task: clustering | query: ")),
        ("window_tokens", json!(256)),
        ("dimensions", json!(512)),
        ("version", json!(2)),
        ("model", json!("omni-moderation-latest")),
    ] {
        let mut bad = artifact();
        bad[field] = value;
        cases.push(bad);
    }
    let mut unknown = artifact();
    unknown["extra"] = json!(1);
    cases.push(unknown);
    for case in cases {
        assert!(load(&case).is_err(), "{case}");
    }
}

#[test]
fn flags_follow_the_thresholds_and_windows_take_the_maximum() {
    let heads = load(&artifact()).expect("valid");
    let hate = heads.head("hate").expect("hate");
    // logit 4 * x0: x0 = 0 is exactly p = 0.5 = the threshold, which flags.
    assert_eq!(hate.probability(&unit(0.0)), 0.5);
    let at = heads.score(&[unit(0.0)]);
    assert_eq!(at.flags[0], Some(true));
    let below = heads.score(&[unit(-0.01)]);
    assert_eq!(below.flags[0], Some(false));
    assert!(!below.flagged());
    // sexual: p = sigmoid(4 x0 - 2) >= 0.9 needs x0 >= ~1.049.
    let s = CATEGORIES
        .iter()
        .position(|c| *c == "sexual")
        .expect("sexual");
    let scores = heads.score(&[unit(-0.5), unit(1.0), unit(0.2)]);
    assert!((scores.scores[s].expect("score") - 1.0 / (1.0 + (-2.0_f64).exp())).abs() < 1e-9);
    assert_eq!(scores.flags[s], Some(false));
    assert_eq!(scores.scores[0], Some(1.0 / (1.0 + (-4.0_f64).exp())));
    assert!(scores.flagged());
    for (index, name) in CATEGORIES.iter().enumerate() {
        if !["hate", "sexual"].contains(name) {
            assert_eq!((scores.scores[index], scores.flags[index]), (None, None));
        }
    }
}

#[test]
fn long_inputs_are_split_into_prompted_windows() {
    let short: Vec<u32> = (0..WINDOW_TOKENS as u32).collect();
    assert_eq!(window_ids(&short, &[0, 1, 2, 1]), vec![short.clone()]);
    // <bos>=2, prompt ids 10 11 12, <eos>=1; the text's first token differs
    // from the prompt's last (merged space), so only 3 ids are shared.
    let prompt = [2, 10, 11, 12, 1];
    let mut full = vec![2, 10, 11, 99];
    full.extend(1000..2100_u32);
    full.push(1);
    let windows = window_ids(&full, &prompt);
    let body_per = WINDOW_TOKENS - 3 - 1;
    assert_eq!(windows.len(), (full.len() - 4).div_ceil(body_per));
    for window in &windows {
        assert!(window.len() <= WINDOW_TOKENS);
        assert_eq!(&window[..3], &[2, 10, 11]);
        assert_eq!(window.last(), Some(&1));
    }
    let body: Vec<u32> = windows
        .iter()
        .flat_map(|w| w[3..w.len() - 1].to_vec())
        .collect();
    assert_eq!(body, full[3..full.len() - 1]);
}

#[test]
fn the_committed_artifact_matches_the_pin_and_ships_only_passing_categories() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/moderation/moderation-heads.json");
    let bytes = std::fs::read(&path).expect("committed artifact");
    let heads = ModerationHeads::from_bytes(&bytes, MODERATION_HEADS_SHA256, PAYLOAD)
        .expect("the committed artifact validates against the pin");
    assert_eq!(
        heads.supported().collect::<Vec<_>>(),
        [
            "hate",
            "harassment",
            "harassment/threatening",
            "illicit",
            "self-harm",
            "sexual"
        ]
    );
    let file: Value = serde_json::from_slice(&bytes).expect("json");
    for name in CATEGORIES {
        let evaluated = &file["evaluation"]["categories"][name];
        assert_eq!(
            evaluated["ships"].as_bool(),
            Some(heads.head(name).is_some()),
            "{name}: the artifact ships exactly the categories that passed"
        );
        if heads.head(name).is_none() {
            assert!(
                heads
                    .unsupported_reason(name)
                    .is_some_and(|r| !r.is_empty())
            );
        }
    }
}
