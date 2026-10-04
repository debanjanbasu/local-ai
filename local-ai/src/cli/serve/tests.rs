use super::*;
fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}
#[test]
fn defaults_and_bounds() {
    let parsed = parse(&args(&["--no-thinking"])).expect("options");
    assert!(!parsed.thinking);
    for invalid in [
        vec!["--context", "4096"],
        vec!["--prefill-chunk", "128"],
        vec!["--max-queue", "8"],
        vec!["--http3"],
    ] {
        assert!(parse(&args(&invalid)).is_err());
    }
}
#[test]
fn rejects_media_message_parts() {
    assert!(message_text(&json!([{"type":"image_url"}])).is_err());
    assert_eq!(
        message_text(&json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]))
            .expect("text"),
        "ab"
    );
}
#[test]
fn zstd_json_and_sse_framing() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::ACCEPT_ENCODING,
        header::HeaderValue::from_static("gzip, zstd"),
    );
    assert!(wants_zstd(&headers));
}

#[test]
#[ignore = "requires BONSAI_GGUF; validates server requests before generation"]
fn serve_real_bonsai_validates_before_streaming_and_preserves_history() {
    let path = std::env::var("BONSAI_GGUF").expect("BONSAI_GGUF");
    let handle = Engine::open_model(path).expect("open engine").into_handle();
    for body in [
        json!({"prompt":"", "stream":true, "max_tokens":0}),
        json!({"prompt":"hello", "stream":true, "temperature":-1}),
        json!({"prompt":"hello", "stream":true, "max_tokens":usize::MAX}),
    ] {
        let prepared =
            prepare_generation(body.to_string().as_bytes(), false, true).expect("parse request");
        let GenerationRequest::Completion(request) = prepared.request else {
            unreachable!("completion request")
        };
        assert!(matches!(
            handle.complete(request).expect("queue").next(),
            Some(Event::Error(_))
        ));
    }
    let body = json!({"messages":[
            {"role":"user","content":"First"},
            {"role":"assistant","content":"Answer","reasoning_content":"prior reasoning"},
            {"role":"user","content":"Second"}
        ], "max_tokens":0});
    let prepared =
        prepare_generation(body.to_string().as_bytes(), true, true).expect("prepare history");
    let GenerationRequest::Chat(request) = prepared.request else {
        unreachable!("chat request")
    };
    assert!(
        handle
            .chat(request)
            .expect("queue")
            .all(|event| !matches!(event, Event::Error(_)))
    );
}
