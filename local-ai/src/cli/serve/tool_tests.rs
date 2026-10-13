//! CPU-only tests of Responses tool types and the hosted tool loop, driven
//! by synthetic engine streams so no model or GPU is touched.
//!
//! Safety: hosted calls here only read and write a temporary directory, or
//! run one `sleep` that is cancelled within seconds. Refusals are checked on
//! inert arguments that are never executed.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use local_engine::bonsai_model::{PromptCacheSource, StopReason};
use local_engine::{
    ChatRequest, Event, EventStream, GenerationStats, Signal, Stats, ToolCall, ToolChoice,
};
use local_services::tools::{Limits, Permissions, Toolbox};

use super::super::hosted::{Hosted, Kind};
use super::super::response::{Protocol, Reply};
use super::super::responses::{ResponsesState, prepare_responses};
use super::{Bounds, Loop, Tools, check, start};

fn stats(stop_reason: StopReason, generated: usize) -> Box<Stats> {
    Box::new(Stats {
        stop_reason,
        cache_source: PromptCacheSource::None,
        reasoning_tokens: 1,
        generation: GenerationStats {
            prompt_tokens: 10,
            reused_prompt_tokens: 0,
            generated_tokens: generated,
            ..GenerationStats::default()
        },
    })
}

fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    }
}

/// A finished synthetic generation of `events`, queued from a plain thread
/// (the relay blocks, which an async test's own thread may not).
fn generation(events: Vec<Event>) -> EventStream {
    let (relay, stream) = EventStream::relay();
    std::thread::spawn(move || {
        for event in events {
            assert!(relay.blocking_send(Signal::Event(event)));
        }
    })
    .join()
    .expect("queued");
    stream
}

const TOOLS: &str = r#"[{"type":"shell"},{"type":"custom","name":"grep_it","format":{"type":"grammar","syntax":"regex","definition":"[a-z]+"}},{"type":"local_shell"},{"type":"apply_patch"},{"type":"function","name":"local_fs_read"},{"type":"function","name":"local_fs_write"},{"type":"function","name":"weather","parameters":{"type":"object"}}]"#;

fn prepare(tools: &str, extra: &Value) -> super::super::responses::PreparedResponses {
    let mut body =
        json!({"input":"hi","tools":serde_json::from_str::<Value>(tools).expect("tools")});
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    prepare_responses(body.to_string().as_bytes(), true).expect("prepared")
}

#[test]
fn tool_types_become_strict_native_functions() {
    let prepared = prepare(
        TOOLS,
        &json!({"tool_choice":{"type":"custom","name":"grep_it"}}),
    );
    let request = &prepared.request;
    let names: Vec<&str> = request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "shell",
            "grep_it",
            "local_shell",
            "apply_patch",
            "local_fs_read",
            "local_fs_write",
            "weather"
        ]
    );
    assert!(request.tools.iter().take(6).all(|tool| tool.strict));
    assert_eq!(
        request.tools[1].parameters["properties"]["input"]["pattern"],
        "^(?:[a-z]+)$"
    );
    assert_eq!(request.tool_choice, ToolChoice::Function("grep_it".into()));
    let kinds = prepared.echo.kinds();
    assert_eq!(kinds.kind("shell"), Kind::Hosted(Hosted::Shell));
    assert_eq!(kinds.kind("grep_it"), Kind::Custom);
    assert_eq!(kinds.kind("local_shell"), Kind::LocalShell);
    assert_eq!(kinds.kind("apply_patch"), Kind::ApplyPatch);
    assert_eq!(kinds.kind("local_fs_read"), Kind::Hosted(Hosted::Read));
    assert_eq!(kinds.kind("weather"), Kind::Function);
}

#[test]
fn malformed_tool_declarations_are_refused() {
    for (tools, expected) in [
        (
            r#"[{"type":"custom","name":"x","format":{"type":"grammar","syntax":"lark","definition":"start: \"(\" start \")\" | \"a\""}}]"#,
            "custom tool \"x\" lark grammar: start is recursive",
        ),
        (
            r#"[{"type":"custom","name":"x","format":{"type":"grammar","syntax":"lark","definition":"start: WORD\n%import common.WORD\n%ignore \" \""}}]"#,
            "%ignore is not supported",
        ),
        (
            r#"[{"type":"namespace","name":"ns","description":"d","tools":[{"type":"local_shell"}]}]"#,
            "only hold function and custom tools",
        ),
        (
            r#"[{"type":"namespace","name":"a.b","description":"d","tools":[{"type":"function","name":"f"}]}]"#,
            "namespace tool requires a name",
        ),
        (
            r#"[{"type":"namespace","name":"ns","description":"d","tools":[{"type":"function","name":"local_fs_read"}]}]"#,
            "cannot be namespaced",
        ),
        (
            r#"[{"type":"namespace","name":"ns","tools":[]}]"#,
            "non-empty",
        ),
        (
            r#"[{"type":"web_search","external_web_access":false,"engine":"x"}]"#,
            "member \"engine\" is not supported",
        ),
        (
            r#"[{"type":"custom","name":"apply_patch"},{"type":"apply_patch"}]"#,
            "duplicate tool name \"apply_patch\"",
        ),
        (
            r#"[{"type":"function","name":"local_fs_read","parameters":{"type":"object"}}]"#,
            "not supported",
        ),
        (
            r#"[{"type":"shell","environment":{"type":"container_auto"}}]"#,
            "local",
        ),
        (r#"[{"type":"custom","name":"local_fs_read"}]"#, "reserved"),
        (r#"[{"type":"custom"}]"#, "name"),
    ] {
        let body =
            json!({"input":"hi","tools":serde_json::from_str::<Value>(tools).expect("tools")});
        let error = prepare_responses(body.to_string().as_bytes(), true)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains(expected), "{tools}: {error}");
    }
}

fn toolbox(root: &std::path::Path, write: bool, shell: bool) -> Toolbox {
    Toolbox::new(
        Permissions {
            root: Some(root.to_owned()),
            write,
            shell,
            fetch_domains: Vec::new(),
        },
        Limits::default(),
    )
    .expect("toolbox")
}

#[test]
fn hosted_tools_need_server_permission() {
    let prepared = prepare(TOOLS, &json!({}));
    let kinds = prepared.echo.kinds();
    let error = check(None, kinds).expect_err("no tools configured");
    assert!(error.contains("--tools-root"), "{error}");
    let root = tempfile::tempdir().expect("root");
    let read_only = Tools {
        toolbox: toolbox(root.path(), false, false),
        bounds: bounds(4),
    };
    let error = check(Some(&read_only), kinds).expect_err("shell and write refused");
    assert!(error.contains("not permitted"), "{error}");
    let everything = Tools {
        toolbox: toolbox(root.path(), true, true),
        bounds: bounds(4),
    };
    check(Some(&everything), kinds).expect("permitted");
    // A request without hosted tools needs nothing.
    let plain = prepare(r#"[{"type":"local_shell"}]"#, &json!({}));
    check(None, plain.echo.kinds()).expect("client tools only");
}

#[test]
fn client_tool_calls_render_as_their_spec_items_and_replay() {
    let prepared = prepare(TOOLS, &json!({}));
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), Arc::from("bonsai"));
    let mut state = ResponsesState::new(&reply, echo, true);
    state.start();
    state.event(Event::ToolCall(call(
        "c1",
        "grep_it",
        json!({"input":"abc"}),
    )));
    state.event(Event::ToolCall(call(
        "c2",
        "local_shell",
        json!({"command":["ls","-l"],"working_directory":"src"}),
    )));
    state.event(Event::ToolCall(call(
        "c3",
        "apply_patch",
        json!({"type":"update_file","path":"a.txt","diff":"@@\n-a\n+b"}),
    )));
    let response = state
        .event(Event::Finished(stats(StopReason::Eos, 3)))
        .expect("finished");
    let output = response["output"].as_array().expect("output");
    assert_eq!(output[0]["type"], "custom_tool_call");
    assert_eq!(output[0]["input"], "abc");
    assert_eq!(output[1]["type"], "local_shell_call");
    assert_eq!(output[1]["action"]["command"], json!(["ls", "-l"]));
    assert_eq!(output[1]["action"]["type"], "exec");
    assert_eq!(output[2]["type"], "apply_patch_call");
    assert_eq!(output[2]["operation"]["path"], "a.txt");
    assert!(
        state
            .take_events()
            .iter()
            .any(|event| event["type"] == "response.custom_tool_call_input.done")
    );

    // The client answers each call; the history replays as calls and tool
    // results the model can read.
    let mut input = output.clone();
    input.push(json!({"type":"custom_tool_call_output","call_id":"c1","output":"3 matches"}));
    input.push(json!({"type":"local_shell_call_output","call_id":"c2","output":"total 0"}));
    input.push(json!({"type":"apply_patch_call_output","call_id":"c3","status":"completed","output":"done"}));
    let body = json!({"input":input,"tools":serde_json::from_str::<Value>(TOOLS).expect("tools")});
    let next = prepare_responses(body.to_string().as_bytes(), true).expect("replayed");
    let messages = &next.request.messages;
    let calls = &messages[0].tool_calls;
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].arguments, json!({"input":"abc"}));
    assert_eq!(calls[1].name, "local_shell");
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(messages[2].content, "total 0");
    assert_eq!(messages[3].content, "[apply_patch completed] done");
}

fn bounds(max_iterations: usize) -> Bounds {
    Bounds {
        max_iterations,
        budget: Duration::from_secs(60),
        call_timeout: Duration::from_secs(20),
    }
}

/// A loop whose generations are `rounds`, in order, recording each request
/// it submitted after the first.
fn fake_loop(
    tools: &str,
    toolbox: Toolbox,
    bounds: Bounds,
    rounds: Vec<Vec<Event>>,
) -> (Loop, EventStream, Arc<Mutex<Vec<ChatRequest>>>) {
    let prepared = prepare(tools, &json!({}));
    let submitted = Arc::new(Mutex::new(Vec::new()));
    let mut rounds = rounds.into_iter();
    let first = generation(rounds.next().expect("a first round"));
    let record = Arc::clone(&submitted);
    let tool_loop = Loop {
        submit: Box::new(move |request| {
            record.lock().expect("lock").push(request);
            Ok(generation(rounds.next().unwrap_or_default()))
        }),
        request: prepared.request,
        toolbox,
        kinds: prepared.echo.kinds().clone(),
        bounds,
        max_calls: None,
        before_tools: None,
    };
    (tool_loop, first, submitted)
}

/// Every event of `stream` until it ends.
async fn drain(stream: EventStream) -> Vec<Event> {
    tokio::task::spawn_blocking(move || stream.collect())
        .await
        .expect("drain")
}

fn outputs(events: &[Event]) -> Vec<&local_engine::ToolOutput> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::ToolOutput(output) => Some(output),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn the_loop_runs_hosted_calls_and_continues_generation() {
    let root = tempfile::tempdir().expect("root");
    std::fs::write(root.path().join("notes.txt"), "the answer is 42\n").expect("write");
    let (tool_loop, first, submitted) = fake_loop(
        TOOLS,
        toolbox(root.path(), false, false),
        bounds(4),
        vec![
            vec![
                Event::Reasoning("look".into()),
                Event::ToolCall(call("r1", "local_fs_read", json!({"path":"notes.txt"}))),
                // Refused: this server grants no writes. Never run.
                Event::ToolCall(call(
                    "w1",
                    "local_fs_write",
                    json!({"path":"notes.txt","content":"overwritten"}),
                )),
                // Malformed for the tool even though the schema admits it.
                Event::ToolCall(call(
                    "r2",
                    "local_fs_read",
                    json!({"path":"../outside.txt"}),
                )),
                Event::Finished(stats(StopReason::Eos, 7)),
            ],
            vec![
                Event::Content("It is 42.".into()),
                Event::Finished(stats(StopReason::Eos, 4)),
            ],
        ],
    );
    let events = drain(start(tool_loop, first)).await;
    let outputs = outputs(&events);
    assert_eq!(outputs.len(), 3);
    assert!(
        outputs[0].content.contains("1: the answer is 42"),
        "{}",
        outputs[0].content
    );
    assert_eq!(outputs[0].detail["status"], "completed");
    assert!(
        outputs[1].content.starts_with("error: refused"),
        "{}",
        outputs[1].content
    );
    assert_eq!(outputs[1].detail["status"], "incomplete");
    assert!(
        outputs[2].content.starts_with("error:"),
        "{}",
        outputs[2].content
    );
    assert_eq!(
        std::fs::read_to_string(root.path().join("notes.txt")).expect("read"),
        "the answer is 42\n"
    );
    let Some(Event::Finished(total)) = events.last() else {
        panic!("the loop must end with the summed usage: {events:?}");
    };
    assert_eq!(total.generation.generated_tokens, 11);
    assert_eq!(total.generation.prompt_tokens, 20);
    let submitted = submitted.lock().expect("lock").clone();
    assert_eq!(submitted.len(), 1);
    let next = &submitted[0];
    let tool_messages: Vec<_> = next
        .messages
        .iter()
        .filter(|message| message.role == "tool")
        .collect();
    assert_eq!(tool_messages.len(), 3);
    assert_eq!(tool_messages[0].tool_call_id.as_deref(), Some("r1"));
    assert_eq!(
        next.max_tokens,
        prepare(TOOLS, &json!({})).request.max_tokens - 7
    );
    assert_eq!(next.tool_choice, ToolChoice::Auto);
}

#[tokio::test]
async fn the_loop_renders_hosted_items_in_the_response() {
    let root = tempfile::tempdir().expect("root");
    std::fs::write(root.path().join("a.txt"), "alpha\n").expect("write");
    let (tool_loop, first, _) = fake_loop(
        TOOLS,
        toolbox(root.path(), false, false),
        bounds(4),
        vec![
            vec![
                Event::ToolCall(call("r1", "local_fs_read", json!({"path":"a.txt"}))),
                Event::Finished(stats(StopReason::Eos, 2)),
            ],
            vec![
                Event::Content("done".into()),
                Event::Finished(stats(StopReason::Eos, 1)),
            ],
        ],
    );
    let events = drain(start(tool_loop, first)).await;
    let prepared = prepare(TOOLS, &json!({}));
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), Arc::from("bonsai"));
    let mut state = ResponsesState::new(&reply, echo, false);
    let mut response = None;
    for event in events {
        response = state.event(event).or(response);
    }
    let response = response.expect("finished");
    assert_eq!(response["status"], "completed");
    let output = response["output"].as_array().expect("output");
    let types: Vec<&str> = output
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect();
    assert_eq!(types, ["function_call", "function_call_output", "message"]);
    assert_eq!(output[1]["call_id"], "r1");
    assert!(
        output[1]["output"]
            .as_str()
            .expect("text")
            .contains("alpha")
    );
}

#[tokio::test]
async fn iteration_bound_forces_a_final_answer() {
    let root = tempfile::tempdir().expect("root");
    std::fs::write(root.path().join("a.txt"), "alpha\n").expect("write");
    let read = || Event::ToolCall(call("r", "local_fs_read", json!({"path":"a.txt"})));
    let (tool_loop, first, submitted) = fake_loop(
        TOOLS,
        toolbox(root.path(), false, false),
        bounds(1),
        vec![
            vec![read(), Event::Finished(stats(StopReason::Eos, 1))],
            vec![
                Event::Content("final".into()),
                Event::Finished(stats(StopReason::Eos, 1)),
            ],
        ],
    );
    let events = drain(start(tool_loop, first)).await;
    assert!(matches!(events.last(), Some(Event::Finished(_))));
    let submitted = submitted.lock().expect("lock").clone();
    assert_eq!(submitted.len(), 1);
    assert_eq!(submitted[0].tool_choice, ToolChoice::None);
}

#[tokio::test]
async fn a_turn_with_a_client_call_ends_after_its_hosted_calls() {
    let root = tempfile::tempdir().expect("root");
    std::fs::write(root.path().join("a.txt"), "alpha\n").expect("write");
    let (tool_loop, first, submitted) = fake_loop(
        TOOLS,
        toolbox(root.path(), false, false),
        bounds(4),
        vec![vec![
            Event::ToolCall(call("r", "local_fs_read", json!({"path":"a.txt"}))),
            Event::ToolCall(call("w", "weather", json!({}))),
            Event::Finished(stats(StopReason::Eos, 1)),
        ]],
    );
    let events = drain(start(tool_loop, first)).await;
    assert_eq!(outputs(&events).len(), 1);
    assert!(matches!(events.last(), Some(Event::Finished(_))));
    assert!(submitted.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn a_token_limited_turn_runs_nothing() {
    let root = tempfile::tempdir().expect("root");
    let (tool_loop, first, submitted) = fake_loop(
        TOOLS,
        toolbox(root.path(), true, false),
        bounds(4),
        vec![vec![
            Event::ToolCall(call(
                "w",
                "local_fs_write",
                json!({"path":"never.txt","content":"x"}),
            )),
            Event::Finished(stats(StopReason::TokenLimit, 1)),
        ]],
    );
    let events = drain(start(tool_loop, first)).await;
    assert_eq!(outputs(&events).len(), 0);
    assert!(!root.path().join("never.txt").exists());
    assert!(submitted.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn cancelling_the_response_stops_a_running_tool() {
    let root = tempfile::tempdir().expect("root");
    let (tool_loop, first, submitted) = fake_loop(
        TOOLS,
        toolbox(root.path(), false, true),
        bounds(4),
        vec![vec![
            Event::ToolCall(call("s", "shell", json!({"commands":["sleep 30"]}))),
            Event::Finished(stats(StopReason::Eos, 1)),
        ]],
    );
    let started = Instant::now();
    let stream = start(tool_loop, first);
    let cancel = stream.cancel_handle();
    let received = tokio::task::spawn_blocking(move || {
        let mut stream = stream;
        let first = stream.next();
        cancel.cancel();
        let rest: Vec<Event> = stream.collect();
        (first, rest)
    })
    .await
    .expect("consumer");
    assert!(matches!(received.0, Some(Event::ToolCall(_))));
    assert_eq!(outputs(&received.1).len(), 0);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    assert!(submitted.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn hosted_shell_calls_render_as_shell_items() {
    let root = tempfile::tempdir().expect("root");
    std::fs::write(root.path().join("a.txt"), "hello\n").expect("write");
    let (tool_loop, first, _) = fake_loop(
        TOOLS,
        toolbox(root.path(), false, true),
        bounds(4),
        vec![
            vec![
                Event::ToolCall(call("s1", "shell", json!({"commands":["wc -c a.txt"]}))),
                Event::Finished(stats(StopReason::Eos, 2)),
            ],
            vec![
                Event::Content("6 bytes".into()),
                Event::Finished(stats(StopReason::Eos, 1)),
            ],
        ],
    );
    let events = drain(start(tool_loop, first)).await;
    let prepared = prepare(TOOLS, &json!({}));
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), Arc::from("bonsai"));
    let mut state = ResponsesState::new(&reply, echo, true);
    let mut response = None;
    for event in events {
        response = state.event(event).or(response);
    }
    let response = response.expect("finished");
    let output = response["output"].as_array().expect("output");
    assert_eq!(output[0]["type"], "shell_call");
    assert_eq!(output[0]["action"]["commands"], json!(["wc -c a.txt"]));
    assert_eq!(output[0]["environment"], json!({"type":"local"}));
    assert_eq!(output[1]["type"], "shell_call_output");
    assert_eq!(output[1]["call_id"], "s1");
    let content = &output[1]["output"][0];
    assert!(
        content["stdout"]
            .as_str()
            .expect("stdout")
            .contains("6 a.txt")
    );
    assert_eq!(content["outcome"], json!({"type":"exit","exit_code":0}));
    let kinds: Vec<String> = state
        .take_events()
        .iter()
        .filter_map(|event| event["type"].as_str().map(str::to_owned))
        .collect();
    for expected in [
        "response.shell_call_command.done",
        "response.shell_call_output_content.done",
        "response.completed",
    ] {
        assert!(
            kinds.iter().any(|kind| kind == expected),
            "{expected}: {kinds:?}"
        );
    }
}

#[test]
fn hosted_tools_are_opt_in_server_flags() {
    use super::super::options::parse;
    let args =
        |list: &[&str]| -> Vec<String> { list.iter().map(|arg| (*arg).to_owned()).collect() };
    let parsed = parse(&args(&[])).expect("defaults");
    assert!(!parsed.tools.any());
    assert!(!parsed.tools.write && !parsed.tools.shell);
    let error = parse(&args(&["--tools-shell"])).expect_err("shell needs a root");
    assert!(error.contains("--tools-root"), "{error}");
    let error = parse(&args(&[
        "--tools-root",
        "/tmp",
        "--tools-max-iterations",
        "0",
    ]))
    .expect_err("bounded");
    assert!(error.contains("from 1 to 64"), "{error}");
    let parsed = parse(&args(&[
        "--tools-root",
        "/tmp",
        "--tools-shell",
        "--tools-fetch-domain",
        "example.com",
        "--tools-call-timeout",
        "5",
    ]))
    .expect("granted");
    assert!(parsed.tools.any() && parsed.tools.shell && !parsed.tools.write);
    assert_eq!(parsed.tools.fetch_domains, ["example.com"]);
    assert_eq!(parsed.tools.call_timeout, Duration::from_secs(5));
}

/// Codex 0.162.1's freeform `apply_patch` grammar, verbatim from the CLI.
const CODEX_APPLY_PATCH: &str = r#"start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"#;

/// The tool shapes Codex 0.162.1 sends with its default configuration (as
/// recorded by `tools/client_compat.py`), trimmed to one tool of each kind,
/// plus its freeform `apply_patch`.
fn codex_default_tools() -> Value {
    json!([
        {"type":"function","name":"exec_command","description":"Runs a command in a PTY.","strict":false,"parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"],"additionalProperties":false}},
        {"type":"namespace","name":"multi_agent_v1","description":"Tools for spawning and managing sub-agents.","tools":[
            {"type":"function","name":"close_agent","description":"Close an agent.","strict":false,"parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"],"additionalProperties":false}},
            {"type":"function","name":"spawn_agent","description":"Spawn a sub-agent.","strict":false,"parameters":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"],"additionalProperties":false}}
        ]},
        {"type":"web_search","external_web_access":false},
        {"type":"custom","name":"apply_patch","description":"Use the `apply_patch` tool to edit files.","format":{"type":"grammar","syntax":"lark","definition":CODEX_APPLY_PATCH}}
    ])
}

#[test]
fn codex_default_tools_are_accepted_as_the_client_tools_they_are() {
    let body = json!({"input":"hi","tools":codex_default_tools(),"tool_choice":"auto","parallel_tool_calls":true});
    let prepared = prepare_responses(body.to_string().as_bytes(), true).expect("prepared");
    let request = &prepared.request;
    let names: Vec<&str> = request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    // Namespaced tools are flattened under qualified names; web search is
    // never offered to the model, which cannot search here.
    assert_eq!(
        names,
        [
            "exec_command",
            "multi_agent_v1.close_agent",
            "multi_agent_v1.spawn_agent",
            "apply_patch"
        ]
    );
    assert_eq!(
        request.tools[2].description.as_deref(),
        Some("Tools for spawning and managing sub-agents.\n\nSpawn a sub-agent.")
    );
    // The custom tool's Lark grammar reaches the engine's constrained decoding.
    let patch = &request.tools[3];
    assert!(patch.strict);
    assert_eq!(
        patch.parameters["properties"]["input"][local_engine::LARK_KEYWORD],
        CODEX_APPLY_PATCH
    );
    let kinds = prepared.echo.kinds();
    assert_eq!(kinds.kind("apply_patch"), Kind::Custom);
    assert_eq!(kinds.kind("multi_agent_v1.spawn_agent"), Kind::Function);
    // The echo keeps the declarations as the client sent them.
    let echo = Arc::new(prepared.echo);
    let reply = Reply::new(Protocol::Responses(Arc::clone(&echo)), Arc::from("bonsai"));
    let mut state = ResponsesState::new(&reply, Arc::clone(&echo), true);
    state.start();
    let patch_text = "*** Begin Patch\n*** Update File: calc.py\n@@\n-a\n+b\n*** End Patch";
    state.event(Event::ToolCall(call(
        "c1",
        "multi_agent_v1.spawn_agent",
        json!({"message":"look"}),
    )));
    state.event(Event::ToolCall(call(
        "c2",
        "apply_patch",
        json!({"input":patch_text}),
    )));
    let response = state
        .event(Event::Finished(stats(StopReason::Eos, 3)))
        .expect("finished");
    let tools = response["tools"].as_array().expect("tools");
    assert_eq!(tools[1]["type"], "namespace");
    assert_eq!(tools[1]["tools"][1]["name"], "spawn_agent");
    assert_eq!(
        tools[2],
        json!({"type":"web_search","external_web_access":false})
    );
    let output = response["output"].as_array().expect("output");
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["namespace"], "multi_agent_v1");
    assert_eq!(output[0]["name"], "spawn_agent");
    assert_eq!(output[1]["type"], "custom_tool_call");
    assert_eq!(output[1]["name"], "apply_patch");
    assert_eq!(output[1]["input"], patch_text);
    assert!(output[1].get("namespace").is_none());
    let events = state.take_events();
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added")
        .expect("added");
    assert_eq!(added["item"]["namespace"], "multi_agent_v1");
    assert_eq!(added["item"]["name"], "spawn_agent");
    assert!(output.iter().all(|item| item["type"] != "web_search_call"));

    // Codex replays the calls with their namespace; the model reads them
    // under the names it called.
    let mut input = output.clone();
    input.push(json!({"type":"function_call_output","call_id":"c1","output":"agent 7"}));
    input.push(json!({"type":"custom_tool_call_output","call_id":"c2","output":"Done!"}));
    let body = json!({"input":input,"tools":codex_default_tools()});
    let next = prepare_responses(body.to_string().as_bytes(), true).expect("replayed");
    let calls = &next.request.messages[0].tool_calls;
    assert_eq!(calls[0].name, "multi_agent_v1.spawn_agent");
    assert_eq!(calls[1].name, "apply_patch");
    assert_eq!(calls[1].arguments, json!({"input":patch_text}));
}

#[test]
fn web_search_is_never_forced_or_reported_as_searched() {
    for (extra, needle) in [
        (
            json!({"tool_choice":{"type":"web_search"}}),
            "tool_choice type \"web_search\" is not supported",
        ),
        (json!({"tool_choice":"required"}), "needs at least one tool"),
        (
            json!({"include":["web_search_call.action.sources"]}),
            "include",
        ),
    ] {
        let mut body = json!({"input":"hi","tools":[{"type":"web_search"}]});
        if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            body.extend(extra.clone());
        }
        let error = prepare_responses(body.to_string().as_bytes(), true)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains(needle), "{extra}: {error}");
    }
    let prepared = prepare(
        r#"[{"type":"web_search_preview","search_context_size":"low"}]"#,
        &json!({}),
    );
    assert_eq!(prepared.request.tools.len(), 0);
}
