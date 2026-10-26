use serde_json::{Map, Value, json};

use local_engine::embedding::EMBEDDING_DIMENSIONS;
use local_engine::moderation::{
    CATEGORIES, MODERATION_MODEL_ID, ModerationHeads, PROMPT, WINDOW_TOKENS,
};

use super::{ALIASES, parse, response_json, result_json, unscored};

/// `CreateModerationRequest`/`Response` (and what they reference), extracted
/// from the pinned `OpenAI` `OpenAPI` document.
const SCHEMAS: &str = include_str!("../moderation_openapi_schemas.json");
const PAYLOAD: &str = "a7c15de94419f566ba448f59168778b1aa4c7f348fd944283f8b49a85d4296c5";

fn conforms(name: &str, value: &Value) {
    let spec: Value = serde_json::from_str(SCHEMAS).expect("schema fixture");
    let mut root = Map::new();
    root.insert("$ref".into(), json!(format!("#/components/schemas/{name}")));
    root.insert("components".into(), spec["components"].clone());
    let validator = jsonschema::options()
        .build(&Value::Object(root))
        .expect("the pinned schema compiles");
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:#?}\n{value:#}");
}

/// Heads for `hate` (logit `4 x0`, threshold 0.5) and `illicit` (logit
/// `4 x1`, threshold 0.9); every other category unsupported.
fn heads() -> ModerationHeads {
    let head = |axis: usize, threshold: f64| {
        let mut weight = vec![0.0_f32; EMBEDDING_DIMENSIONS];
        weight[axis] = 4.0;
        json!({"weight":weight,"bias":0.0,"platt_scale":1.0,"platt_offset":0.0,"threshold":threshold})
    };
    let mut categories = Map::new();
    for name in CATEGORIES {
        categories.insert(name.into(), json!({"unsupported":"no data"}));
    }
    categories.insert("hate".into(), head(0, 0.5));
    categories.insert("illicit".into(), head(1, 0.9));
    let artifact = json!({
        "format":"local-ai-moderation-heads","version":1,"model":MODERATION_MODEL_ID,
        "embedding_model":"embeddinggemma-2","embedding_payload_sha256":PAYLOAD,"prompt":PROMPT,
        "window_tokens":WINDOW_TOKENS,"dimensions":EMBEDDING_DIMENSIONS,"categories":categories,
        "provenance":{},"evaluation":{},
    });
    let bytes = serde_json::to_vec(&artifact).expect("json");
    let digest = ring::digest::digest(&ring::digest::SHA256, &bytes);
    let sha = digest.as_ref().iter().fold(String::new(), |mut out, b| {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
        out
    });
    ModerationHeads::from_bytes(&bytes, &sha, PAYLOAD).expect("valid heads")
}

fn vector(x0: f32, x1: f32) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBEDDING_DIMENSIONS];
    v[0] = x0;
    v[1] = x1;
    v
}

fn body(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).expect("json")
}

#[test]
fn requests_follow_create_moderation_request() {
    for request in [
        json!({"input":"I want to kill them."}),
        json!({"input":["a","b"],"model":"omni-moderation-latest"}),
        json!({"input":[{"type":"text","text":"a"}],"model":MODERATION_MODEL_ID}),
    ] {
        conforms("CreateModerationRequest", &request);
        parse(&body(&request)).expect("accepted");
    }
    assert_eq!(
        parse(&body(&json!({"input":""}))).expect("empty string"),
        vec![String::new()]
    );
    assert_eq!(
        parse(&body(&json!({"input":["a","b"]})))
            .expect("strings")
            .len(),
        2
    );
    let parts = json!({"input":[{"type":"text","text":"a"},{"type":"text","text":"b"}]});
    assert_eq!(
        parse(&body(&parts)).expect("one result"),
        vec!["a\nb".to_owned()]
    );
    for alias in ALIASES {
        parse(&body(&json!({"input":"x","model":alias}))).expect("alias");
    }
}

#[test]
fn images_and_invalid_requests_are_refused() {
    let image =
        json!({"input":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]});
    conforms("CreateModerationRequest", &image);
    let error = parse(&body(&image)).expect_err("images are refused");
    assert!(error.contains("image inputs are not supported"), "{error}");
    let mixed = json!({"input":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]});
    assert!(parse(&body(&mixed)).is_err());
    for request in [
        json!({"input":[]}),
        json!({"input":["a",1]}),
        json!({"input":["a",{"type":"text","text":"b"}]}),
        json!({"input":[{"type":"text"}]}),
        json!({"input":[{"type":"text","text":"a","extra":1}]}),
        json!({"input":[{"type":"audio","audio":"x"}]}),
        json!({"input":7}),
        json!({"input":"a","model":"gpt-4o"}),
        json!({"input":"a","extra":true}),
        json!({"model":"omni-moderation-latest"}),
    ] {
        assert!(parse(&body(&request)).is_err(), "{request}");
    }
    let many = vec!["x"; super::MAX_INPUTS + 1];
    assert!(parse(&body(&json!({"input":many}))).is_err());
}

#[test]
fn responses_have_the_full_spec_shape_and_mark_unscored_categories() {
    let heads = heads();
    let results = [
        heads.score(&[vector(0.5, 0.0)]),
        heads.score(&[vector(-0.5, 0.0)]),
    ];
    let value = response_json(&heads, &results);
    conforms("CreateModerationResponse", &value);
    assert!(value["id"].as_str().expect("id").starts_with("modr-"));
    assert_eq!(value["model"], MODERATION_MODEL_ID);
    let first = &value["results"][0];
    assert_eq!(first["flagged"], true);
    assert_eq!(first["categories"]["hate"], true);
    assert_eq!(
        first["category_applied_input_types"]["hate"],
        json!(["text"])
    );
    // illicit is scored: p = sigmoid(0) = 0.5, below its 0.9 threshold.
    assert_eq!(first["categories"]["illicit"], false);
    assert_eq!(first["category_scores"]["illicit"], 0.5);
    assert_eq!(
        first["category_applied_input_types"]["illicit"],
        json!(["text"])
    );
    // Unscored: false (null where the spec allows it), score 0, applied to nothing.
    assert_eq!(first["categories"]["violence"], false);
    assert_eq!(first["categories"]["illicit/violent"], Value::Null);
    assert_eq!(first["category_scores"]["violence"], 0.0);
    assert_eq!(first["category_applied_input_types"]["violence"], json!([]));
    for name in CATEGORIES {
        assert!(first["categories"].get(name).is_some(), "{name}");
    }
    assert_eq!(value["results"][1]["flagged"], false);
    assert_eq!(
        unscored(&heads),
        CATEGORIES
            .iter()
            .filter(|c| !["hate", "illicit"].contains(c))
            .copied()
            .collect::<Vec<_>>()
            .join(",")
    );
}

#[test]
fn flagged_follows_each_threshold() {
    let heads = heads();
    // hate at exactly its threshold flags; illicit just under 0.9 does not.
    let at = result_json(&heads, &heads.score(&[vector(0.0, 0.0)]));
    assert_eq!(
        (at["categories"]["hate"].clone(), at["flagged"].clone()),
        (json!(true), json!(true))
    );
    let x1 = (0.9_f64 / 0.1).ln() / 4.0;
    let below = result_json(&heads, &heads.score(&[vector(-1.0, (x1 - 1e-3) as f32)]));
    assert_eq!(below["categories"]["illicit"], false);
    assert_eq!(below["flagged"], false);
    let above = result_json(&heads, &heads.score(&[vector(-1.0, (x1 + 1e-3) as f32)]));
    assert_eq!(above["categories"]["illicit"], true);
    assert_eq!(above["flagged"], true);
    // Windows: the maximum counts.
    let windows = result_json(&heads, &heads.score(&[vector(-1.0, 0.0), vector(1.0, 0.0)]));
    assert_eq!(windows["categories"]["hate"], true);
}
