use std::ops::ControlFlow;
use std::sync::Arc;

use serde_json::{Value, json};

use super::{ToolCall, ToolCallParser, ToolDefinition, ToolSet, Turn, render};
use crate::api::Event;

/// Rendered by the pinned template itself (Jinja2 3.1.6, `trim_blocks` and
/// `lstrip_blocks` as in `transformers`, `tojson` as `json.dumps(...,
/// ensure_ascii=False)`), not by this renderer.
const TOOLS_THINKING: &str = include_str!("fixtures/tools_thinking.txt");
const TOOLS_NOTHINKING: &str = include_str!("fixtures/tools_nothinking.txt");

fn weather() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".into(),
        description: Some("Get the weather.".into()),
        parameters: json!({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}),
        strict: false,
    }
}

fn search() -> ToolDefinition {
    ToolDefinition {
        name: "search".into(),
        description: None,
        parameters: json!({"type":"object","properties":{"query":{"type":"string"},"filters":{"type":"object"}}}),
        strict: false,
    }
}

fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    }
}

fn turn<'a>(role: &'a str, content: &'a str) -> Turn<'a> {
    Turn {
        role,
        content,
        reasoning_content: None,
        tool_calls: &[],
        tool_call_id: None,
    }
}

fn result<'a>(id: &'a str, content: &'a str) -> Turn<'a> {
    Turn {
        tool_call_id: Some(id),
        ..turn("tool", content)
    }
}

fn tools(definitions: &[ToolDefinition]) -> ToolSet {
    ToolSet::new(definitions).expect("valid tools")
}

#[test]
fn renders_tools_calls_and_grouped_results_like_the_pinned_template() {
    let calls = [
        call("a", "get_weather", json!({"city":"Paris\nFrance","days":3})),
        call(
            "b",
            "search",
            json!({"filters":{"lang":"fr","n":[1,2]},"query":"café"}),
        ),
    ];
    let turns = [
        turn("system", "Be exact."),
        turn("user", "Weather in Paris and café news?"),
        Turn {
            reasoning_content: Some("Need tools."),
            tool_calls: &calls,
            ..turn("assistant", "Checking.")
        },
        result("a", " sunny "),
        result("b", "news"),
    ];
    let rendered = render(&turns, true, &tools(&[weather(), search()])).expect("render");
    assert_eq!(rendered, TOOLS_THINKING);
}

#[test]
fn replayed_results_render_in_call_order_whatever_order_they_arrive_in() {
    // The prompt has no IDs, so position is the only pairing the model sees.
    let calls = [
        call("a", "get_weather", json!({"city":"Paris\nFrance","days":3})),
        call(
            "b",
            "search",
            json!({"filters":{"lang":"fr","n":[1,2]},"query":"café"}),
        ),
    ];
    let turns = [
        turn("system", "Be exact."),
        turn("user", "Weather in Paris and café news?"),
        Turn {
            reasoning_content: Some("Need tools."),
            tool_calls: &calls,
            ..turn("assistant", "Checking.")
        },
        result("b", "news"),
        result("a", " sunny "),
    ];
    let rendered = render(&turns, true, &tools(&[weather(), search()])).expect("render");
    assert_eq!(rendered, TOOLS_THINKING);
}

#[test]
fn renders_schema_and_argument_keys_sorted_whatever_order_the_client_sent() {
    // Written in reverse order so that `serde_json/preserve_order` (enabled by
    // any dependency) cannot change the prompt unnoticed.
    let definition = tool(json!({
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "filters": {
                "type": "object",
                "properties": {"z": {"type": "integer"}, "a": {"type": "object"}}
            }
        }
    }));
    let calls = [call(
        "a",
        &definition.name,
        json!({"query": "café", "filters": {"z": 1, "a": {"y": [{"b": 2, "a": 1}], "x": null}}}),
    )];
    let turns = [
        turn("user", "go"),
        Turn {
            tool_calls: &calls,
            ..turn("assistant", "")
        },
        result("a", "ok"),
    ];
    let rendered = render(&turns, false, &tools(&[definition])).expect("render");
    assert!(rendered.contains(
        "\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"f\", \"parameters\": {\"properties\": {\"filters\": {\"properties\": {\"a\": {\"type\": \"object\"}, \"z\": {\"type\": \"integer\"}}, \"type\": \"object\"}, \"query\": {\"type\": \"string\"}}, \"required\": [\"query\"], \"type\": \"object\"}}}\n</tools>\n"
    ), "{rendered}");
    assert!(rendered.contains(
        "<tool_call>\n<function=f>\n<parameter=filters>\n{\"a\": {\"x\": null, \"y\": [{\"a\": 1, \"b\": 2}]}, \"z\": 1}\n</parameter>\n<parameter=query>\ncafé\n</parameter>\n</function>\n</tool_call>"
    ), "{rendered}");
}

#[test]
fn renders_an_empty_answer_call_and_a_later_user_turn_without_thinking() {
    let calls = [call("a", "search", json!({}))];
    let turns = [
        turn("user", "Weather?"),
        Turn {
            tool_calls: &calls,
            ..turn("assistant", "")
        },
        result("a", "done"),
        turn("user", "Thanks"),
    ];
    let rendered = render(&turns, false, &tools(&[search()])).expect("render");
    assert_eq!(rendered, TOOLS_NOTHINKING);
}

#[test]
fn developer_takes_the_system_slot() {
    let developer = render(
        &[turn("developer", "Be exact."), turn("user", "Hi")],
        true,
        &ToolSet::default(),
    )
    .expect("developer");
    let system = render(
        &[turn("system", "Be exact."), turn("user", "Hi")],
        true,
        &ToolSet::default(),
    )
    .expect("system");
    assert_eq!(developer, system);
    assert!(
        render(
            &[turn("user", "Hi"), turn("developer", "late")],
            true,
            &ToolSet::default()
        )
        .is_err()
    );
}

#[test]
fn rejects_invalid_tool_definitions() {
    let bad = |parameters: Value| ToolDefinition {
        name: "f".into(),
        description: None,
        parameters,
        strict: false,
    };
    for parameters in [
        json!("object"),
        json!({"type":"string"}),
        json!({"type":"object","required":"x"}),
        json!({"type":"object","properties":{"bad name":{"type":"string"}}}),
        json!({"type":"object","properties":{"x":{"type":"text"}}}),
        json!({"type":"object","properties":{"x":{"$ref":"#/$defs/missing"}}}),
        json!({"type":"object","properties":{"x":{"type":"string","pattern":"("}}}),
        json!({"type":"object","properties":{"x":{"minLength":-1}}}),
        // Nothing is fetched: external references and dialects are refused.
        json!({"type":"object","properties":{"x":{"$ref":"https://example.com/s.json"}}}),
        json!({"type":"object","properties":{"x":{"$ref":"file:///etc/hosts"}}}),
        json!({"$schema":"https://example.com/dialect","type":"object"}),
    ] {
        assert!(
            ToolSet::new(&[bad(parameters.clone())]).is_err(),
            "{parameters}"
        );
    }
    assert!(ToolSet::new(&[weather(), weather()]).is_err());
    let mut unnamed = weather();
    unnamed.name = "has space".into();
    assert!(ToolSet::new(&[unnamed]).is_err());
    assert!(ToolSet::new(&[bad(Value::Null)]).is_ok());
    // Annotations are allowed.
    assert!(
        ToolSet::new(&[bad(
            json!({"type":"object","title":"T","properties":{"x":{"type":"string","default":"a","examples":["b"],"x-hint":1}}})
        )])
        .is_ok()
    );
}

fn tool(parameters: Value) -> ToolDefinition {
    ToolDefinition {
        name: "f".into(),
        description: None,
        parameters,
        strict: false,
    }
}

fn single_call(definition: ToolDefinition, parameters: &str) -> Result<Value, String> {
    let block = format!("<tool_call>\n<function=f>\n{parameters}</function>\n</tool_call>");
    match parse(&[definition], &[&block]) {
        (events, None) => match events.as_slice() {
            [Event::ToolCall(call)] => Ok(call.arguments.clone()),
            other => panic!("unexpected events {other:?}"),
        },
        (_, Some(failure)) => Err(failure),
    }
}

#[test]
fn enforces_pattern_unique_items_and_dependent_required() {
    let definition = tool(json!({
        "type":"object",
        "properties":{
            "code":{"type":"string","pattern":"^[A-Z]{3}$"},
            "tags":{"type":"array","uniqueItems":true},
            "card":{"type":"string"},
            "zip":{"type":"string"}
        },
        "dependentRequired":{"card":["zip"]}
    }));
    let accepted = single_call(
        definition.clone(),
        "<parameter=code>\nABC\n</parameter>\n<parameter=tags>\n[1, 2]\n</parameter>\n",
    );
    assert_eq!(accepted, Ok(json!({"code":"ABC","tags":[1,2]})));
    for parameters in [
        "<parameter=code>\nabc\n</parameter>\n",
        "<parameter=tags>\n[1, 1]\n</parameter>\n",
        "<parameter=card>\n4111\n</parameter>\n",
    ] {
        assert!(
            single_call(definition.clone(), parameters).is_err(),
            "{parameters:?} was accepted"
        );
    }
    // Replayed history is held to the same schema.
    let set = tools(&[definition]);
    let calls = [call("a", "f", json!({"code":"abc"}))];
    let turns = [
        turn("user", "go"),
        Turn {
            tool_calls: &calls,
            ..turn("assistant", "")
        },
        result("a", "x"),
    ];
    assert!(render(&turns, true, &set).is_err());
}

#[test]
fn raw_text_stays_a_string_when_the_parameter_schema_accepts_text() {
    let definition = tool(json!({
        "type":"object",
        "$defs":{"Text":{"type":"string"},"Count":{"type":"integer","minimum":1}},
        "properties":{
            "flag":{"$ref":"#/$defs/Text"},
            "note":{"type":["string","null"]},
            "on":{"type":"boolean"},
            "count":{"$ref":"#/$defs/Count"},
            "options":{"type":"object"},
            "any":{}
        },
        "additionalProperties":false
    }));
    let arguments = single_call(
        definition.clone(),
        "<parameter=flag>\ntrue\n</parameter>\n<parameter=note>\nnull\n</parameter>\n<parameter=on>\ntrue\n</parameter>\n<parameter=count>\n2\n</parameter>\n<parameter=options>\n{\"a\": 1}\n</parameter>\n<parameter=any>\n[1]\n</parameter>\n",
    );
    assert_eq!(
        arguments,
        Ok(json!({"flag":"true","note":"null","on":true,"count":2,"options":{"a":1},"any":[1]}))
    );
    assert!(single_call(definition, "<parameter=count>\n0\n</parameter>\n").is_err());
}

#[test]
fn rejects_unpaired_duplicate_and_invalid_tool_history() {
    let set = tools(&[weather()]);
    let one = [call("a", "get_weather", json!({"city":"Paris"}))];
    let two = [
        call("a", "get_weather", json!({"city":"Paris"})),
        call("a", "get_weather", json!({"city":"Rome"})),
    ];
    let unknown = [call("a", "search", json!({}))];
    let bad_arguments = [call("a", "get_weather", json!({"days":2}))];
    let not_object = [call("a", "get_weather", json!("city"))];
    let assistant = |calls| Turn {
        tool_calls: calls,
        ..turn("assistant", "")
    };
    let user = turn("user", "go");
    let cases: Vec<(&str, Vec<Turn<'_>>)> = vec![
        ("missing result", vec![user, assistant(&one)]),
        (
            "missing result before user",
            vec![user, assistant(&one), turn("user", "next")],
        ),
        (
            "duplicate id",
            vec![user, assistant(&two), result("a", "x")],
        ),
        (
            "unmatched result",
            vec![user, assistant(&one), result("b", "x")],
        ),
        (
            "answered twice",
            vec![user, assistant(&one), result("a", "x"), result("a", "y")],
        ),
        ("result without call", vec![user, result("a", "x")]),
        (
            "result without id",
            vec![user, assistant(&one), turn("tool", "x")],
        ),
        (
            "unknown tool",
            vec![user, assistant(&unknown), result("a", "x")],
        ),
        (
            "bad arguments",
            vec![user, assistant(&bad_arguments), result("a", "x")],
        ),
        (
            "non-object arguments",
            vec![user, assistant(&not_object), result("a", "x")],
        ),
        (
            "id on user",
            vec![Turn {
                tool_call_id: Some("a"),
                ..user
            }],
        ),
        (
            "calls on user",
            vec![Turn {
                tool_calls: &one,
                ..user
            }],
        ),
    ];
    for (name, turns) in cases {
        assert!(render(&turns, true, &set).is_err(), "{name} was accepted");
    }
    // An ID reused in a later turn is still a duplicate.
    let reused = [
        user,
        assistant(&one),
        result("a", "x"),
        turn("user", "again"),
        assistant(&one),
        result("a", "y"),
    ];
    assert!(render(&reused, true, &set).is_err());
}

fn parse(tools: &[ToolDefinition], pieces: &[&str]) -> (Vec<Event>, Option<String>) {
    let mut parser = ToolCallParser::new(Arc::new(ToolSet::new(tools).expect("tools")));
    let mut events = Vec::new();
    let mut sink = |event| {
        events.push(event);
        ControlFlow::Continue(())
    };
    for piece in pieces {
        if parser.feed(piece, &mut sink).is_break() {
            return (events, parser.take_failure());
        }
    }
    let _ = parser.finish(crate::bonsai_model::StopReason::Eos, &mut sink);
    (events, parser.take_failure())
}

fn chars(text: &str) -> Vec<String> {
    text.chars().map(String::from).collect()
}

const TWO_CALLS: &str = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\nFrance\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=search>\n<parameter=query>\n42\n</parameter>\n<parameter=filters>\n{\"lang\": \"fr\"}\n</parameter>\n</function>\n</tool_call>";

#[test]
fn streams_text_and_emits_validated_calls_one_character_at_a_time() {
    let text = TWO_CALLS.replace(r#"{"lang": "fr"}"#, r#"{"z":{"b":2,"a":1},"lang":"fr"}"#);
    let pieces = chars(&text);
    let pieces = pieces.iter().map(String::as_str).collect::<Vec<_>>();
    let (events, failure) = parse(&[weather(), search()], &pieces);
    assert_eq!(failure, None);
    let content = events
        .iter()
        .filter_map(|event| match event {
            Event::Content(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(content, "Let me check.");
    // Text streams before the call is complete.
    assert!(matches!(events.first(), Some(Event::Content(_))));
    let calls = events
        .iter()
        .filter_map(|event| match event {
            Event::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, json!({"city":"Paris\nFrance","days":3}));
    // A declared string stays raw text even when it looks like JSON.
    assert_eq!(
        calls[1].arguments,
        json!({"query":"42","filters":{"z":{"b":2,"a":1},"lang":"fr"}})
    );
    // HTTP argument strings keep their original canonical ordering, even
    // when another dependency enables serde_json's preserve_order feature.
    assert_eq!(
        calls[1].arguments.to_string(),
        r#"{"filters":{"lang":"fr","z":{"a":1,"b":2}},"query":"42"}"#
    );
    assert_ne!(calls[0].id, calls[1].id);
    assert!(calls[0].id.starts_with("call_"));
}

#[test]
fn ids_are_unique_across_generations() {
    let block = "<tool_call>\n<function=search>\n</function>\n</tool_call>";
    let (first, _) = parse(&[search()], &[block]);
    let (second, _) = parse(&[search()], &[block]);
    let id = |events: &[Event]| match events {
        [Event::ToolCall(call)] => call.id.clone(),
        other => panic!("unexpected events {other:?}"),
    };
    assert_ne!(id(&first), id(&second));
}

#[test]
fn parsed_calls_render_back_to_the_generated_text() {
    let generated = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\nFrance\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
    let (events, failure) = parse(&[weather()], &[generated]);
    assert_eq!(failure, None);
    let [Event::ToolCall(parsed)] = events.as_slice() else {
        panic!("unexpected events {events:?}");
    };
    let calls = [parsed.clone()];
    let turns = [
        turn("user", "go"),
        Turn {
            tool_calls: &calls,
            ..turn("assistant", "")
        },
        result(&parsed.id, "ok"),
    ];
    let rendered = render(&turns, false, &tools(&[weather()])).expect("render");
    assert!(rendered.contains(&format!("\n</think>\n\n{generated}<|im_end|>")));
}

#[test]
fn malformed_invalid_and_truncated_calls_never_become_events() {
    let cases = [
        "<tool_call>\n<function=unknown>\n</function>\n</tool_call>",
        "<tool_call>\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</tool_call>",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
    ];
    for case in cases {
        let (events, failure) = parse(&[weather()], &[case, "trailing text"]);
        assert!(failure.is_some(), "{case:?} did not fail");
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, Event::ToolCall(_))),
            "{case:?} produced {events:?}"
        );
    }
}

#[test]
fn partial_tags_and_text_after_calls_stay_content() {
    let (events, failure) = parse(&[weather()], &["a <tool", "s> b\n"]);
    assert_eq!(failure, None);
    let text = events
        .iter()
        .map(|event| match event {
            Event::Content(text) => text.as_str(),
            other => panic!("unexpected {other:?}"),
        })
        .collect::<String>();
    assert_eq!(text, "a <tools> b\n");
}
