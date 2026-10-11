//! Responses tool types beyond plain functions, and the hosted tools this
//! server may run itself.
//!
//! Every tool reaches the model as a native function definition (strict, so
//! its arguments are constrained while they are generated); what differs is
//! who executes a call and which spec item reports it:
//!
//! | request tool                          | executed by | output items                               |
//! |---------------------------------------|-------------|--------------------------------------------|
//! | `function`                            | client      | `function_call`                            |
//! | `custom` (text or regex grammar)      | client      | `custom_tool_call`                         |
//! | `local_shell`                         | client      | `local_shell_call`                         |
//! | `apply_patch`                         | client      | `apply_patch_call`                         |
//! | `shell` (local environment)           | this server | `shell_call` + `shell_call_output`         |
//! | `function` named `local_fs_*`/`local_web_fetch`, no parameters | this server | `function_call` + `function_call_output` |
//!
//! The spec has no hosted filesystem or fetch tool (`file_search` is vector
//! store retrieval and `web_search` is search, neither of which these are),
//! so those are reported as what they are: a function call and the output
//! the server produced for it. Hosted tools are only offered when the request
//! asks for them *and* the server was started with the matching permission;
//! otherwise the request is refused before generation.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Duration;

use serde_json::{Map, Value, json};

use local_engine::{ToolCall, ToolChoice, ToolDefinition, ToolOutput};
use local_services::tools::shell::{CommandRun, Outcome};
use local_services::tools::{Capability, Output, Request, SearchMode, ToolError};

/// Exit status reported for a shell call that was refused or failed before
/// any command ran (the shell's own "cannot execute").
const NOT_EXECUTED: i32 = 126;

/// Who runs a tool's calls and how they are reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A client-executed function, the default.
    Function,
    /// A client-executed `custom` tool, called with freeform `input`.
    Custom,
    /// A client-executed `local_shell` tool.
    LocalShell,
    /// A client-executed `apply_patch` tool.
    ApplyPatch,
    /// A tool this server runs.
    Hosted(Hosted),
}

/// The hosted operations, each one native function to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Hosted {
    List,
    Read,
    Search,
    Write,
    Edit,
    Fetch,
    Shell,
}

impl Hosted {
    const FUNCTIONS: [(&'static str, Self); 6] = [
        ("local_fs_list", Self::List),
        ("local_fs_read", Self::Read),
        ("local_fs_search", Self::Search),
        ("local_fs_write", Self::Write),
        ("local_fs_edit", Self::Edit),
        ("local_web_fetch", Self::Fetch),
    ];

    fn from_function(name: &str) -> Option<Self> {
        Self::FUNCTIONS
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, hosted)| *hosted)
    }

    /// The permission this operation needs.
    pub(super) const fn capability(self) -> Capability {
        match self {
            Self::List | Self::Read | Self::Search => Capability::Read,
            Self::Write | Self::Edit => Capability::Write,
            Self::Fetch => Capability::Fetch,
            Self::Shell => Capability::Shell,
        }
    }

    fn definition(self, name: &str) -> ToolDefinition {
        let (description, parameters) = match self {
            Self::List => (
                "List the files under a directory of the workspace (relative path; omit for the \
                 whole workspace). Hidden, ignored and .git entries are not listed.",
                closed(json!({"path":{"type":"string"}}), &[]),
            ),
            Self::Read => (
                "Read lines of a workspace file by relative path. Lines are numbered from 1.",
                closed(
                    json!({"path":{"type":"string"},"start_line":{"type":"integer","minimum":1},"end_line":{"type":"integer","minimum":1}}),
                    &["path"],
                ),
            ),
            Self::Search => (
                "Search workspace files: literal (default) or regex lines matching pattern, or \
                 ranked files for its words.",
                closed(
                    json!({"pattern":{"type":"string"},"path":{"type":"string"},"mode":{"type":"string","enum":["literal","regex","ranked"]}}),
                    &["pattern"],
                ),
            ),
            Self::Write => (
                "Create or replace a workspace file (relative path) with content.",
                closed(
                    json!({"path":{"type":"string"},"content":{"type":"string"}}),
                    &["content", "path"],
                ),
            ),
            Self::Edit => (
                "Replace the single occurrence of old_text in a workspace file with new_text.",
                closed(
                    json!({"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}}),
                    &["new_text", "old_text", "path"],
                ),
            ),
            Self::Fetch => (
                "Fetch one http(s) URL on the server's allowlist and return its status and text.",
                closed(json!({"url":{"type":"string"}}), &["url"]),
            ),
            Self::Shell => (
                "Run shell commands, in order, in the workspace directory under a sandbox with no \
                 network access. Each command's output and exit status is returned.",
                closed(
                    json!({"commands":{"type":"array","items":{"type":"string"},"minItems":1},"timeout_ms":{"type":"integer","minimum":1}}),
                    &["commands"],
                ),
            ),
        };
        strict(name, description, parameters)
    }
}

/// A closed object schema, which strict tools require.
fn closed(properties: Value, required: &[&str]) -> Value {
    let mut schema = json!({"type":"object","required":required,"additionalProperties":false});
    schema["properties"] = properties;
    schema
}

fn strict(name: &str, description: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition {
        name: name.to_owned(),
        description: Some(description.to_owned()),
        parameters,
        strict: true,
    }
}

/// The kind of every declared tool that is not a plain client function.
#[derive(Clone, Debug, Default)]
pub(super) struct Kinds(HashMap<String, Kind>);

impl Kinds {
    pub(super) fn insert(&mut self, name: &str, kind: Kind) {
        if kind != Kind::Function {
            self.0.insert(name.to_owned(), kind);
        }
    }

    pub(super) fn kind(&self, name: &str) -> Kind {
        self.0.get(name).copied().unwrap_or(Kind::Function)
    }

    /// The hosted operation `name` runs, if it is one.
    pub(super) fn hosted(&self, name: &str) -> Option<Hosted> {
        match self.kind(name) {
            Kind::Hosted(hosted) => Some(hosted),
            _ => None,
        }
    }

    /// Every hosted operation declared.
    pub(super) fn all_hosted(&self) -> impl Iterator<Item = Hosted> + '_ {
        self.0.values().filter_map(|kind| match kind {
            Kind::Hosted(hosted) => Some(*hosted),
            _ => None,
        })
    }
}

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidArgument(message.into())
}

/// Refuse members of `tool` outside `allowed`.
fn only(tool: &Map<String, Value>, kind: &str, allowed: &[&str]) -> crate::Result<()> {
    tool.keys()
        .find(|key| !allowed.contains(&key.as_str()))
        .map_or(Ok(()), |key| {
            Err(invalid(format!(
                "{kind} tool member {key:?} is not supported"
            )))
        })
}

/// A declared tool this module handles: its native definition, kind and the
/// form the Response echoes. `None` for a plain function tool (or a type
/// nobody here handles), which the caller parses itself.
pub(super) fn tool(tool: &Value) -> crate::Result<Option<(ToolDefinition, Kind, Value)>> {
    let Some(object) = tool.as_object() else {
        return Ok(None);
    };
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match kind {
        "function" => {
            let Some(name) = object.get("name").and_then(Value::as_str) else {
                return Ok(None);
            };
            let Some(hosted) = Hosted::from_function(name) else {
                return Ok(None);
            };
            only(
                object,
                "hosted function",
                &["type", "name", "description", "strict"],
            )?;
            let definition = hosted.definition(name);
            let echo = json!({"type":"function","name":name,"description":definition.description,"parameters":definition.parameters,"strict":true});
            Ok(Some((definition, Kind::Hosted(hosted), echo)))
        }
        "shell" => {
            only(object, "shell", &["type", "environment"])?;
            match object.get("environment") {
                None | Some(Value::Null) => {}
                Some(Value::Object(environment))
                    if environment.len() == 1
                        && environment.get("type").and_then(Value::as_str) == Some("local") => {}
                Some(_) => {
                    return Err(invalid(
                        "shell environment must be {\"type\":\"local\"}: commands run in this \
                         server's sandbox, never a container",
                    ));
                }
            }
            Ok(Some((
                Hosted::Shell.definition("shell"),
                Kind::Hosted(Hosted::Shell),
                json!({"type":"shell","environment":{"type":"local"}}),
            )))
        }
        "local_shell" => {
            only(object, "local_shell", &["type"])?;
            let definition = strict(
                "local_shell",
                "Run a command on the user's machine. The client runs it and returns the output.",
                object_schema_local_shell(),
            );
            Ok(Some((
                definition,
                Kind::LocalShell,
                json!({"type":"local_shell"}),
            )))
        }
        "apply_patch" => {
            only(object, "apply_patch", &["type"])?;
            let definition = strict(
                "apply_patch",
                "Create, update or delete one file. create_file and update_file take a diff in \
                 the V4A format; the client applies it and returns the result.",
                closed(
                    json!({"type":{"type":"string","enum":["create_file","update_file","delete_file"]},"path":{"type":"string"},"diff":{"type":"string"}}),
                    &["path", "type"],
                ),
            );
            Ok(Some((
                definition,
                Kind::ApplyPatch,
                json!({"type":"apply_patch"}),
            )))
        }
        "custom" => custom(object).map(Some),
        _ => Ok(None),
    }
}

fn object_schema_local_shell() -> Value {
    closed(
        json!({"command":{"type":"array","items":{"type":"string"},"minItems":1},"working_directory":{"type":"string"},"timeout_ms":{"type":"integer","minimum":1}}),
        &["command"],
    )
}

/// A `custom` tool: one freeform string `input`, constrained by a `regex`
/// grammar when one is given. Lark grammars are refused.
fn custom(object: &Map<String, Value>) -> crate::Result<(ToolDefinition, Kind, Value)> {
    only(object, "custom", &["type", "name", "description", "format"])?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("a custom tool requires a name"))?;
    if Hosted::from_function(name).is_some()
        || matches!(name, "shell" | "local_shell" | "apply_patch")
    {
        return Err(invalid(format!("tool name {name:?} is reserved")));
    }
    let description = match object.get("description") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => return Err(invalid("custom tool description must be a string")),
    };
    let mut input = json!({"type":"string"});
    match object.get("format") {
        None | Some(Value::Null) => {}
        Some(Value::Object(format)) => match format.get("type").and_then(Value::as_str) {
            Some("text") if format.len() == 1 => {}
            Some("grammar") => {
                only(format, "custom format", &["type", "syntax", "definition"])?;
                let definition = format
                    .get("definition")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("a grammar format requires a definition string"))?;
                match format.get("syntax").and_then(Value::as_str) {
                    Some("regex") => {
                        // The whole input must match, as a grammar does.
                        input["pattern"] = json!(format!("^(?:{definition})$"));
                    }
                    Some("lark") => {
                        return Err(crate::Error::InvalidArgument(
                            "custom tool grammar syntax \"lark\" is not supported: only regex \
                             grammars are translated into the engine's constrained decoding"
                                .into(),
                        ));
                    }
                    _ => return Err(invalid("grammar syntax must be \"regex\" or \"lark\"")),
                }
            }
            _ => return Err(invalid("custom tool format must be text or grammar")),
        },
        Some(_) => return Err(invalid("custom tool format must be an object")),
    }
    let definition =
        ToolDefinition {
            name: name.to_owned(),
            description: Some(description.clone().unwrap_or_else(|| {
                "A custom tool; pass its freeform input as `input`.".to_owned()
            })),
            parameters: closed(json!({"input":input}), &["input"]),
            strict: true,
        };
    let mut echo = Map::new();
    echo.insert("type".into(), json!("custom"));
    echo.insert("name".into(), json!(name));
    if let Some(description) = description {
        echo.insert("description".into(), json!(description));
    }
    echo.insert(
        "format".into(),
        object
            .get("format")
            .cloned()
            .unwrap_or_else(|| json!({"type":"text"})),
    );
    Ok((definition, Kind::Custom, Value::Object(echo)))
}

/// A `tool_choice` naming a tool type this module defines, or `None` for
/// the caller to parse.
pub(super) fn tool_choice(choice: Option<&Value>) -> Option<crate::Result<(ToolChoice, Value)>> {
    let object = choice?.as_object()?;
    let name = match object.get("type").and_then(Value::as_str)? {
        "custom" => match object.get("name").and_then(Value::as_str) {
            Some(name) if object.len() == 2 && !name.is_empty() => name.to_owned(),
            _ => {
                return Some(Err(invalid(
                    "a custom tool_choice must be {\"type\":\"custom\",\"name\":...}",
                )));
            }
        },
        kind @ ("shell" | "apply_patch") if object.len() == 1 => kind.to_owned(),
        _ => return None,
    };
    Some(Ok((
        ToolChoice::Function(name),
        Value::Object(object.clone()),
    )))
}

/// An output item and the streaming events announcing it.
pub(super) struct Rendered {
    pub(super) item: Value,
    pub(super) events: Vec<(&'static str, Value)>,
}

/// The ID prefix of a call item of `kind`.
pub(super) const fn call_prefix(kind: Kind) -> &'static str {
    match kind {
        Kind::Function | Kind::Hosted(_) => "fc",
        Kind::Custom => "ctc",
        Kind::LocalShell => "lsh",
        Kind::ApplyPatch => "apc",
    }
}

/// The ID prefix of an output item answering a call of `kind`.
pub(super) const fn output_prefix(kind: Kind) -> &'static str {
    match kind {
        Kind::Hosted(Hosted::Shell) => "sho",
        _ => "fco",
    }
}

fn string_field<'a>(arguments: &'a Value, name: &str) -> Option<&'a str> {
    arguments.get(name).and_then(Value::as_str)
}

/// The call item for a call of a non-function `kind` (`None`: the caller
/// renders a `function_call`).
pub(super) fn call_item(kind: Kind, id: &str, index: usize, call: &ToolCall) -> Option<Rendered> {
    let arguments = &call.arguments;
    let (item, mut events) = match kind {
        Kind::Function
        | Kind::Hosted(
            Hosted::List
            | Hosted::Read
            | Hosted::Search
            | Hosted::Write
            | Hosted::Edit
            | Hosted::Fetch,
        ) => return None,
        Kind::Hosted(Hosted::Shell) => {
            let commands: Vec<String> = arguments
                .get("commands")
                .and_then(Value::as_array)
                .map(|commands| {
                    commands
                        .iter()
                        .filter_map(|command| command.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default();
            let mut events = Vec::new();
            for (command_index, command) in commands.iter().enumerate() {
                events.push((
                    "response.shell_call_command.added",
                    json!({"output_index":index,"command_index":command_index,"command":""}),
                ));
                events.push((
                    "response.shell_call_command.delta",
                    json!({"output_index":index,"command_index":command_index,"delta":command}),
                ));
                events.push((
                    "response.shell_call_command.done",
                    json!({"output_index":index,"command_index":command_index,"command":command}),
                ));
            }
            let item = json!({"id":id,"type":"shell_call","call_id":call.id,"action":{"commands":commands,"timeout_ms":arguments.get("timeout_ms"),"max_output_length":null},"status":"completed","environment":{"type":"local"}});
            (item, events)
        }
        Kind::Custom => {
            let input = string_field(arguments, "input").unwrap_or_default();
            let events = vec![
                (
                    "response.custom_tool_call_input.delta",
                    json!({"item_id":id,"output_index":index,"delta":input}),
                ),
                (
                    "response.custom_tool_call_input.done",
                    json!({"item_id":id,"output_index":index,"input":input}),
                ),
            ];
            let item = json!({"id":id,"type":"custom_tool_call","call_id":call.id,"name":call.name,"input":input});
            (item, events)
        }
        Kind::LocalShell => {
            let item = json!({"id":id,"type":"local_shell_call","call_id":call.id,"action":{"type":"exec","command":arguments.get("command").cloned().unwrap_or_else(|| json!([])),"timeout_ms":arguments.get("timeout_ms"),"working_directory":arguments.get("working_directory"),"env":{},"user":null},"status":"completed"});
            (item, Vec::new())
        }
        Kind::ApplyPatch => {
            let mut operation = Map::new();
            for field in ["type", "path", "diff"] {
                if let Some(value) = arguments.get(field) {
                    operation.insert(field.into(), value.clone());
                }
            }
            let item = json!({"id":id,"type":"apply_patch_call","call_id":call.id,"status":"completed","operation":operation});
            (item, Vec::new())
        }
    };
    let mut added = item.clone();
    if let Some(added) = added.as_object_mut() {
        if added.contains_key("status") {
            added.insert("status".into(), json!("in_progress"));
        }
        if added.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
            added.insert("input".into(), json!(""));
        }
    }
    events.insert(
        0,
        (
            "response.output_item.added",
            json!({"output_index":index,"item":added}),
        ),
    );
    events.push((
        "response.output_item.done",
        json!({"output_index":index,"item":item}),
    ));
    Some(Rendered { item, events })
}

/// The output item for a hosted call's result.
pub(super) fn output_item(kind: Kind, id: &str, index: usize, output: &ToolOutput) -> Rendered {
    let status = output
        .detail
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed");
    let mut events = Vec::new();
    let item = if kind == Kind::Hosted(Hosted::Shell) {
        let contents = output
            .detail
            .get("output")
            .cloned()
            .unwrap_or_else(|| json!([]));
        events.push((
            "response.output_item.added",
            json!({"output_index":index,"item":{"id":id,"type":"shell_call_output","call_id":output.call_id,"status":"in_progress","output":[],"max_output_length":null}}),
        ));
        if let Some(contents) = contents.as_array() {
            for (command_index, content) in contents.iter().enumerate() {
                events.push((
                    "response.shell_call_output_content.done",
                    json!({"item_id":id,"output_index":index,"command_index":command_index,"output":[content]}),
                ));
            }
        }
        json!({"id":id,"type":"shell_call_output","call_id":output.call_id,"status":status,"output":contents,"max_output_length":null})
    } else {
        events.push((
            "response.output_item.added",
            json!({"output_index":index,"item":{"id":id,"type":"function_call_output","call_id":output.call_id,"output":"","status":"in_progress"}}),
        ));
        json!({"id":id,"type":"function_call_output","call_id":output.call_id,"output":output.content,"status":status})
    };
    events.push((
        "response.output_item.done",
        json!({"output_index":index,"item":item}),
    ));
    Rendered { item, events }
}

/// A history item this module converts for the model: a call, or a call's
/// output.
pub(super) enum Input {
    Call(ToolCall),
    Output { call_id: String, content: String },
}

/// Convert an input item of type `kind`, or `None` when it is not one of
/// this module's item types.
pub(super) fn input_item(kind: &str, item: &Value) -> Option<crate::Result<Input>> {
    let field = |name: &str| {
        item.get(name)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| invalid(format!("a {kind} item requires {name}")))
    };
    let call = |name: &str, arguments: Value| -> crate::Result<Input> {
        Ok(Input::Call(ToolCall {
            id: field("call_id")?,
            name: name.to_owned(),
            arguments,
        }))
    };
    let output = |content: String| -> crate::Result<Input> {
        let call_id = field("call_id").or_else(|_| field("id"))?;
        Ok(Input::Output { call_id, content })
    };
    let action = item.get("action").cloned().unwrap_or(Value::Null);
    Some(match kind {
        "shell_call" => {
            let mut arguments =
                json!({"commands":action.get("commands").cloned().unwrap_or_else(|| json!([]))});
            if let Some(timeout) = action.get("timeout_ms").filter(|value| !value.is_null()) {
                arguments["timeout_ms"] = timeout.clone();
            }
            call("shell", arguments)
        }
        "shell_call_output" => output(shell_output_text(item.get("output"))),
        "local_shell_call" => {
            let mut arguments =
                json!({"command":action.get("command").cloned().unwrap_or_else(|| json!([]))});
            for name in ["working_directory", "timeout_ms"] {
                if let Some(value) = action.get(name).filter(|value| !value.is_null()) {
                    arguments[name] = value.clone();
                }
            }
            call("local_shell", arguments)
        }
        "local_shell_call_output" => output(
            item.get("output")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        ),
        "apply_patch_call" => call(
            "apply_patch",
            item.get("operation").cloned().unwrap_or_else(|| json!({})),
        ),
        "apply_patch_call_output" => {
            let status = item
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("completed");
            let text = item
                .get("output")
                .and_then(Value::as_str)
                .unwrap_or_default();
            output(
                format!("[apply_patch {status}] {text}")
                    .trim_end()
                    .to_owned(),
            )
        }
        "custom_tool_call" => {
            let name = match field("name") {
                Ok(name) => name,
                Err(error) => return Some(Err(error)),
            };
            let input = item
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default();
            call(&name, json!({"input":input}))
        }
        "custom_tool_call_output" => match item.get("output") {
            Some(Value::String(text)) => output(text.clone()),
            Some(Value::Array(parts)) => output(
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<String>(),
            ),
            _ => Err(invalid("custom_tool_call_output output must be text")),
        },
        _ => return None,
    })
}

/// The model's text for a replayed `shell_call_output`'s contents.
fn shell_output_text(contents: Option<&Value>) -> String {
    let mut text = String::new();
    for content in contents.and_then(Value::as_array).into_iter().flatten() {
        let stdout = content
            .get("stdout")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let stderr = content
            .get("stderr")
            .and_then(Value::as_str)
            .unwrap_or_default();
        text.push_str(stdout);
        if !stderr.is_empty() {
            let _ = write!(text, "\n[stderr]\n{stderr}");
        }
        let outcome = content.get("outcome");
        if outcome
            .and_then(|outcome| outcome.get("type"))
            .and_then(Value::as_str)
            == Some("timeout")
        {
            text.push_str("\n[timed out; the command was killed]\n");
        } else {
            let code = outcome
                .and_then(|outcome| outcome.get("exit_code"))
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let _ = writeln!(text, "\n[exit code {code}]");
        }
    }
    text
}

/// The native request for a hosted call, or why its arguments are unusable.
pub(super) fn request(hosted: Hosted, arguments: &Value) -> Result<Request, String> {
    let text = |name: &str| string_field(arguments, name).map(str::to_owned);
    let required = |name: &str| text(name).ok_or_else(|| format!("{name} is required"));
    let line = |name: &str| -> Result<Option<u64>, String> {
        match arguments.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_u64()
                .filter(|line| *line >= 1)
                .map(Some)
                .ok_or_else(|| format!("{name} must be a whole number of at least 1")),
        }
    };
    Ok(match hosted {
        Hosted::List => Request::List { path: text("path") },
        Hosted::Read => Request::Read {
            path: required("path")?,
            start_line: line("start_line")?,
            end_line: line("end_line")?,
        },
        Hosted::Search => Request::Search {
            pattern: required("pattern")?,
            path: text("path"),
            mode: match text("mode").as_deref() {
                None | Some("literal") => SearchMode::Literal,
                Some("regex") => SearchMode::Regex,
                Some("ranked") => SearchMode::Ranked,
                Some(other) => return Err(format!("unknown search mode {other:?}")),
            },
        },
        Hosted::Write => Request::Write {
            path: required("path")?,
            content: required("content")?,
        },
        Hosted::Edit => Request::Edit {
            path: required("path")?,
            old: required("old_text")?,
            new: required("new_text")?,
        },
        Hosted::Fetch => Request::Fetch {
            url: required("url")?,
        },
        Hosted::Shell => {
            let commands = arguments
                .get("commands")
                .and_then(Value::as_array)
                .ok_or("commands must be an array of strings")?
                .iter()
                .map(|command| {
                    command
                        .as_str()
                        .map(str::to_owned)
                        .ok_or("commands must be an array of strings")
                })
                .collect::<Result<Vec<_>, _>>()?;
            Request::Shell {
                commands,
                timeout: line("timeout_ms")?.map(Duration::from_millis),
                max_output_bytes: None,
            }
        }
    })
}

/// The event a hosted call's result is reported with.
pub(super) fn tool_output(
    call: &ToolCall,
    hosted: Hosted,
    result: &Result<Output, ToolError>,
) -> ToolOutput {
    let (content, detail) = match (hosted, result) {
        (Hosted::Shell, Ok(Output::Shell(runs))) => (
            output_text(result),
            json!({"status":"completed","output":runs.iter().map(shell_content).collect::<Vec<_>>()}),
        ),
        (Hosted::Shell, Err(error)) => (
            output_text(result),
            json!({"status":"incomplete","output":[{"stdout":"","stderr":error.to_string(),"outcome":{"type":"exit","exit_code":NOT_EXECUTED}}]}),
        ),
        (_, Ok(_)) => (output_text(result), json!({"status":"completed"})),
        (_, Err(_)) => (output_text(result), json!({"status":"incomplete"})),
    };
    ToolOutput {
        call_id: call.id.clone(),
        name: call.name.clone(),
        content,
        detail,
    }
}

/// The text the model reads for a hosted result; a failure says it is one.
fn output_text(result: &Result<Output, ToolError>) -> String {
    match result {
        Ok(output) => output.model_text(),
        Err(error) => format!("error: {error}"),
    }
}

/// A `shell_call_output` content entry. A command ended by a signal is
/// reported as the shell does, exit status 128 plus the signal.
fn shell_content(run: &CommandRun) -> Value {
    let outcome = match run.outcome {
        Outcome::Exit(code) => json!({"type":"exit","exit_code":code}),
        Outcome::Signal(signal) => json!({"type":"exit","exit_code":128 + signal}),
        Outcome::TimedOut => json!({"type":"timeout"}),
    };
    json!({"stdout":run.stdout,"stderr":run.stderr,"outcome":outcome})
}
