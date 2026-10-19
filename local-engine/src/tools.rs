//! Native structured tool calling in the pinned checkpoint's own format.
//!
//! The pinned Ternary Bonsai 2 chat template (revision
//! `3f926b415992eaa2ae9dd7b573706494d6bbf787`) lists tool schemas in the system
//! turn as one JSON object per line inside `<tools>`, and expects calls as
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=ARG>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! where string values are raw (possibly multiline) text and every other value
//! is JSON. Tool results are `tool` messages; consecutive results share one
//! user turn as `<tool_response>` blocks. Call IDs are never shown to the model.
//!
//! This module renders that format, validates requests and replayed history,
//! and turns generated text back into validated calls. It never executes a
//! tool: a [`ToolCall`] is only a complete, schema-checked request that the
//! harness may choose to run.
//!
//! [`ToolChoice`], `parallel_tool_calls` and [`ToolDefinition::strict`] opt a
//! request into native constrained calling (see [`grammar`]): the call layout
//! and, for strict tools, the arguments are enforced on every sampled token.
//! The defaults leave generation unconstrained.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher as _, Hash as _, Hasher as _};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::Event;
use crate::bonsai_model::StopReason;
use crate::bonsai_tokenizer::ReasoningEffort;

pub mod grammar;

use grammar::{
    FUNCTION_OPEN, PARAMETER_CLOSE, PARAMETER_OPEN, STRICT_CALL_END, StrictTool, ValueKind,
};

const CALL_START: &str = "<tool_call>";
const CALL_END: &str = "</tool_call>";
const TOOL_RESPONSE_START: &str = "<tool_response>";
const TOOL_RESPONSE_END: &str = "</tool_response>";
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
/// Local, never-fetched URI under which a tool's parameter schema is
/// registered so per-parameter subschemas can reference into it.
const SCHEMA_URI: &str = "urn:local-ai:tool-parameters";

/// A function the model may call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Function name: 1-64 ASCII letters, digits, `_`, `-` or `.`.
    pub name: String,
    /// Optional description shown to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments object. `null` means no parameters.
    #[serde(default)]
    pub parameters: Value,
    /// Enforce `parameters` while the call is generated (`OpenAI` `strict`).
    ///
    /// The arguments are then constrained token by token to the declared
    /// parameters in the template's layout, each value to its own schema.
    /// The schema must be a closed object (`"additionalProperties": false`)
    /// whose parameters this engine can enforce exactly: a raw string with
    /// `enum`/`const`/`pattern`/`minLength`/`maxLength`, or any JSON value
    /// whose schema the grammar engine supports. Anything else fails the
    /// request before generation with the reason, rather than being enforced
    /// partially. Not shown to the model. Defaults to `false`: the schema
    /// guides the model but is not enforced, and a call is returned with its
    /// arguments as generated for the client to validate (a malformed call
    /// layout still fails).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
}

/// Whether and which tool the model must call (`OpenAI` `tool_choice`).
///
/// Anything but [`Self::Auto`] is enforced by the sampler, not requested in
/// the prompt: the prompt is the same for every choice.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides (the default). Unconstrained unless another setting
    /// (`parallel_tool_calls: false`, a strict tool or a `response_format`)
    /// needs the grammar, in which case free text never contains
    /// `<tool_call>` outside a call.
    #[default]
    Auto,
    /// No call: the answer can never contain `<tool_call>`.
    None,
    /// At least one call, before which the answer may hold only blank lines;
    /// generation cannot end before a complete call.
    Required,
    /// One call to the named tool, as for [`Self::Required`].
    Function(String),
    /// Calls only to the named tools (`OpenAI` `allowed_tools`): every tool
    /// stays in the prompt, so its cached prefix is shared, but the grammar
    /// admits only these. `required` makes a call mandatory as for
    /// [`Self::Required`]; otherwise the model may also answer in text.
    Allowed { tools: Vec<String>, required: bool },
}

/// A complete, validated call the model requested. The engine never runs it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call ID: generated by the engine for model output, and matched against
    /// `tool_call_id` when the call is replayed in history. Not model-visible.
    pub id: String,
    /// Name of a configured [`ToolDefinition`].
    pub name: String,
    /// Arguments as a JSON object that satisfies the tool's schema.
    pub arguments: Value,
}

/// Part of a call's arguments, reported while the call is still being
/// generated.
///
/// In order, a call's deltas spell a prefix of its final arguments, serialized
/// as compact JSON in the order the model wrote them (that is the
/// [`ToolCall::arguments`] the closing [`Event::ToolCall`] carries, with the
/// same ID). String parameters whose schema accepts every string stream as
/// they are written; any other value arrives whole once it ends. Nothing
/// streams before the call names a configured tool, and a call that then
/// fails validation or is cut short gets no [`Event::ToolCall`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// The call's ID.
    pub id: String,
    /// The called tool.
    pub name: String,
    /// The next piece of the arguments' JSON text.
    pub arguments: String,
    /// When the piece continues one string parameter's value: that
    /// parameter and the raw text the piece adds to it.
    pub text: Option<(String, String)>,
}

/// A validated set of tool definitions with their schemas compiled once.
#[derive(Debug, Default)]
pub struct ToolSet {
    tools: Vec<Tool>,
}

#[derive(Debug)]
struct Tool {
    definition: ToolDefinition,
    /// `None` when the tool takes no parameters.
    schema: Option<ArgumentSchema>,
    /// For a strict tool, its native spelling, or why it cannot be enforced.
    /// The error surfaces only when a grammar is built, so counting and
    /// rendering never depend on it.
    strict: Option<Result<StrictTool, String>>,
}

#[derive(Debug)]
struct ArgumentSchema {
    /// The whole arguments object.
    arguments: jsonschema::Validator,
    /// Per-parameter subschemas, used only to decide how raw text converts.
    properties: HashMap<String, Parameter>,
    additional: Option<Parameter>,
}

#[derive(Debug)]
struct Parameter {
    validator: jsonschema::Validator,
    /// Accepts every JSON type, so the schema says nothing about raw text.
    unconstrained: bool,
    /// Accepts every string, so its raw text is always the value and can be
    /// streamed before the value ends.
    any_string: bool,
}

impl ToolSet {
    pub fn new(tools: &[ToolDefinition]) -> crate::Result<Self> {
        let mut names = HashSet::new();
        let mut compiled = Vec::with_capacity(tools.len());
        for tool in tools {
            check_tool_name(&tool.name).map_err(crate::Error::InvalidArgument)?;
            if !names.insert(tool.name.as_str()) {
                return invalid(format!("duplicate tool name {:?}", tool.name));
            }
            let schema = compile_parameters(&tool.parameters).map_err(|error| {
                crate::Error::InvalidArgument(format!("tool {:?} parameters: {error}", tool.name))
            })?;
            let strict = tool.strict.then(|| grammar::classify(&tool.parameters));
            compiled.push(Tool {
                definition: tool.clone(),
                schema,
                strict,
            });
        }
        Ok(Self { tools: compiled })
    }

    pub const fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|tool| tool.definition.name == name)
    }

    /// Whether any tool is strict, which always comes with a grammar.
    pub fn any_strict(&self) -> bool {
        self.tools.iter().any(|tool| tool.strict.is_some())
    }
}

/// Check that `choice` can apply to `tools`.
pub fn check_choice(tools: &[ToolDefinition], choice: &ToolChoice) -> crate::Result<()> {
    match choice {
        ToolChoice::Required if tools.is_empty() => {
            invalid("tool_choice \"required\" needs at least one tool")
        }
        ToolChoice::Auto | ToolChoice::None | ToolChoice::Required => Ok(()),
        ToolChoice::Function(name) if tools.iter().any(|tool| tool.name == *name) => Ok(()),
        ToolChoice::Function(name) => invalid(format!("tool_choice names unknown tool {name:?}")),
        ToolChoice::Allowed { tools: allowed, .. } if allowed.is_empty() => {
            invalid("allowed_tools needs at least one tool")
        }
        ToolChoice::Allowed { tools: allowed, .. } => allowed
            .iter()
            .find(|name| !tools.iter().any(|tool| tool.name == **name))
            .map_or(Ok(()), |name| {
                invalid(format!("allowed_tools names unknown tool {name:?}"))
            }),
    }
}

/// One message as the renderer sees it.
#[derive(Clone, Copy)]
pub struct Turn<'a> {
    pub role: &'a str,
    pub content: &'a str,
    pub reasoning_content: Option<&'a str>,
    pub tool_calls: &'a [ToolCall],
    pub tool_call_id: Option<&'a str>,
}

/// Render a conversation exactly as the pinned checkpoint template does with
/// `add_generation_prompt=True`, `enable_thinking` and `reasoning_effort` from
/// `thinking` (`None` closes the thought block empty), and
/// `preserve_thinking`.
///
/// Without `preserve_thinking`, assistant turns up to the last user query
/// render with no thought block at all, as the template does; later ones
/// (the current tool loop) keep theirs.
///
/// `developer` is accepted as a leading message and rendered in the `system`
/// slot: the template has no developer role and only one leading instruction
/// turn, so consecutive leading `system`/`developer` messages (Codex sends
/// several) are joined there in order, separated by a blank line. One leading
/// message renders exactly as before.
pub fn render(
    turns: &[Turn<'_>],
    thinking: Option<ReasoningEffort>,
    preserve_thinking: bool,
    tools: &ToolSet,
) -> crate::Result<String> {
    if turns.is_empty() {
        return invalid("no messages provided");
    }
    if turns.len() == 1 && turns[0].role == "user" && turns[0].content.trim().is_empty() {
        return invalid("message content is empty");
    }
    validate_history(turns, tools)?;
    let is_query = |turn: &Turn<'_>| {
        let content = turn.content.trim();
        turn.role == "user"
            && !(content.starts_with(TOOL_RESPONSE_START) && content.ends_with(TOOL_RESPONSE_END))
    };
    if !turns.iter().any(is_query) {
        return invalid("no user query found in messages");
    }
    let turns = &order_results(turns)?;
    let last_query = turns.iter().rposition(is_query).unwrap_or_default();
    let leading = turns
        .iter()
        .take_while(|turn| is_instruction(turn.role))
        .count();
    let system = leading_instructions(turns.get(..leading).unwrap_or_default());
    let instruction = thinking.and_then(ReasoningEffort::instruction);
    let mut rendered = render_system(&system, instruction, tools);
    for (index, turn) in turns.iter().enumerate() {
        let content = turn.content.trim();
        match turn.role {
            role if is_instruction(role) => {
                if index >= leading {
                    return invalid(format!(
                        "{role} message must be first: instructions may only lead the \
                         conversation"
                    ));
                }
            }
            "user" => {
                rendered.push_str("<|im_start|>user\n");
                rendered.push_str(content);
                rendered.push_str("<|im_end|>\n");
            }
            "assistant" => {
                rendered.push_str("<|im_start|>assistant\n");
                if preserve_thinking || index > last_query {
                    rendered.push_str("<think>\n");
                    rendered.push_str(turn.reasoning_content.unwrap_or_default().trim());
                    rendered.push_str("\n</think>\n\n");
                }
                rendered.push_str(content);
                for (position, call) in turn.tool_calls.iter().enumerate() {
                    if position > 0 {
                        rendered.push('\n');
                    } else if !content.is_empty() {
                        rendered.push_str("\n\n");
                    }
                    render_call(call, &mut rendered);
                }
                rendered.push_str("<|im_end|>\n");
            }
            "tool" => {
                if index > 0 && turns[index - 1].role != "tool" {
                    rendered.push_str("<|im_start|>user");
                }
                rendered.push('\n');
                rendered.push_str(TOOL_RESPONSE_START);
                rendered.push('\n');
                rendered.push_str(content);
                rendered.push('\n');
                rendered.push_str(TOOL_RESPONSE_END);
                if turns.get(index + 1).is_none_or(|next| next.role != "tool") {
                    rendered.push_str("<|im_end|>\n");
                }
            }
            role => return invalid(format!("unsupported message role {role:?}")),
        }
    }
    rendered.push_str("<|im_start|>assistant\n");
    if thinking.is_some() {
        rendered.push_str("<think>\n");
    } else {
        rendered.push_str("<think>\n\n</think>\n\n");
    }
    Ok(rendered)
}

/// The text of the leading `system`/`developer` messages, which share the
/// template's one system slot: each trimmed, empty ones skipped, joined in
/// order by a blank line.
fn leading_instructions(turns: &[Turn<'_>]) -> String {
    let mut system = String::new();
    for content in turns
        .iter()
        .map(|turn| turn.content.trim())
        .filter(|content| !content.is_empty())
    {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(content);
    }
    system
}

/// The leading system turn: reasoning instructions, tools and the leading
/// `system`/`developer` text `system`, in the template's order.
fn render_system(system: &str, instruction: Option<&str>, tools: &ToolSet) -> String {
    let mut rendered = String::new();
    if !tools.is_empty() {
        rendered.push_str("<|im_start|>system\n");
        if let Some(instruction) = instruction {
            rendered.push_str(instruction);
            rendered.push_str("\n\n");
        }
        rendered.push_str(TOOLS_HEADER);
        for tool in &tools.tools {
            rendered.push('\n');
            render_tool(&tool.definition, &mut rendered);
        }
        rendered.push_str(TOOLS_FOOTER);
        if !system.is_empty() {
            rendered.push_str("\n\n");
            rendered.push_str(system);
        }
        rendered.push_str("<|im_end|>\n");
    } else if instruction.is_some() || !system.is_empty() {
        rendered.push_str("<|im_start|>system\n");
        if let Some(instruction) = instruction {
            rendered.push_str(instruction);
            if !system.is_empty() {
                rendered.push_str("\n\n");
            }
        }
        rendered.push_str(system);
        rendered.push_str("<|im_end|>\n");
    }
    rendered
}

fn is_instruction(role: &str) -> bool {
    matches!(role, "system" | "developer")
}

/// Check that replayed calls and results pair up one-to-one.
///
/// Every assistant call must be answered by exactly one `tool` message before
/// the next non-tool message, IDs are unique across the conversation, and a
/// call to a configured strict tool must satisfy its schema.
fn validate_history(turns: &[Turn<'_>], tools: &ToolSet) -> crate::Result<()> {
    let mut seen = HashSet::new();
    let mut pending: Vec<&str> = Vec::new();
    for turn in turns {
        if turn.role != "tool"
            && let Some(id) = pending.first()
        {
            return invalid(format!("tool call {id:?} has no tool result"));
        }
        if turn.role != "assistant" && !turn.tool_calls.is_empty() {
            return invalid(format!("{} messages cannot contain tool calls", turn.role));
        }
        if turn.role != "tool" && turn.tool_call_id.is_some() {
            return invalid(format!("{} messages cannot set tool_call_id", turn.role));
        }
        match turn.role {
            "assistant" => {
                for call in turn.tool_calls {
                    if call.id.trim().is_empty() {
                        return invalid("tool call id is empty");
                    }
                    if !seen.insert(call.id.as_str()) {
                        return invalid(format!("duplicate tool call id {:?}", call.id));
                    }
                    check_tool_name(&call.name).map_err(crate::Error::InvalidArgument)?;
                    let Value::Object(arguments) = &call.arguments else {
                        return invalid(format!(
                            "tool call {:?} arguments must be a JSON object",
                            call.id
                        ));
                    };
                    for key in arguments.keys() {
                        check_parameter_name(key).map_err(crate::Error::InvalidArgument)?;
                    }
                    // A non-strict call is replayed as it was generated,
                    // even when it missed the schema and its result is the
                    // client's error report, and so is a call to a tool the
                    // request no longer (or never) declared: harnesses change
                    // their tool sets between turns.
                    if let Some(tool) = tools.get(&call.name).filter(|tool| tool.strict.is_some()) {
                        validate_arguments(tool, &call.arguments).map_err(|error| {
                            crate::Error::InvalidArgument(format!(
                                "tool call {:?}: {error}",
                                call.id
                            ))
                        })?;
                    }
                    pending.push(&call.id);
                }
            }
            "tool" => {
                let Some(id) = turn.tool_call_id else {
                    return invalid("tool message requires tool_call_id");
                };
                if let Some(position) = pending.iter().position(|pending| *pending == id) {
                    pending.remove(position);
                } else {
                    return invalid(format!(
                        "tool result {id:?} does not answer an unanswered call in the preceding assistant message"
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some(id) = pending.first() {
        return invalid(format!("tool call {id:?} has no tool result"));
    }
    Ok(())
}

/// Reorder each run of `tool` results into the call order of the assistant
/// message they answer.
///
/// The rendered prompt carries no call IDs, so the model can only pair a
/// result with a call by position; results replayed out of order would
/// otherwise be silently attributed to the wrong call. Expects history that
/// [`validate_history`] accepted, and still rejects anything that does not
/// pair up one-to-one.
fn order_results<'a>(turns: &[Turn<'a>]) -> crate::Result<Vec<Turn<'a>>> {
    let mut ordered = Vec::with_capacity(turns.len());
    let mut index = 0;
    while let Some(turn) = turns.get(index) {
        if turn.role != "tool" {
            ordered.push(*turn);
            index += 1;
            continue;
        }
        let end = turns
            .iter()
            .skip(index)
            .position(|turn| turn.role != "tool")
            .map_or(turns.len(), |length| index + length);
        let results = turns.get(index..end).unwrap_or_default();
        let calls = index
            .checked_sub(1)
            .and_then(|previous| turns.get(previous))
            .map_or(&[][..], |previous| previous.tool_calls);
        if calls.len() != results.len() {
            return invalid("tool results do not match the preceding assistant tool calls");
        }
        for call in calls {
            let mut matching = results
                .iter()
                .filter(|result| result.tool_call_id == Some(call.id.as_str()));
            match (matching.next(), matching.next()) {
                (Some(result), None) => ordered.push(*result),
                _ => {
                    return invalid(format!(
                        "tool call {:?} needs exactly one tool result",
                        call.id
                    ));
                }
            }
        }
        index = end;
    }
    Ok(ordered)
}

/// `{"type": "function", "function": {...}}`, as the template's `tojson` prints
/// an `OpenAI` tool object.
///
/// Schemas and replayed arguments keep the client's key order (the build
/// enables `serde_json/preserve_order`), as `transformers`, vLLM and llama.cpp
/// render the template.
fn render_tool(tool: &ToolDefinition, out: &mut String) {
    out.push_str("{\"type\": \"function\", \"function\": {\"name\": ");
    push_json_string(&tool.name, out);
    if let Some(description) = &tool.description {
        out.push_str(", \"description\": ");
        push_json_string(description, out);
    }
    if !tool.parameters.is_null() {
        out.push_str(", \"parameters\": ");
        push_python_json(&tool.parameters, out);
    }
    out.push_str("}}");
}

fn render_call(call: &ToolCall, out: &mut String) {
    out.push_str("<tool_call>\n<function=");
    out.push_str(&call.name);
    out.push_str(">\n");
    if let Value::Object(arguments) = &call.arguments {
        for (name, value) in arguments {
            out.push_str("<parameter=");
            out.push_str(name);
            out.push_str(">\n");
            match value {
                Value::String(text) => out.push_str(text),
                other => push_python_json(other, out),
            }
            out.push_str("\n</parameter>\n");
        }
    }
    out.push_str("</function>\n</tool_call>");
}

/// JSON with Python's default `json.dumps` separators (`", "` and `": "`) and
/// unescaped non-ASCII, which is what the template's `tojson` produces.
fn push_python_json(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                push_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                push_json_string(key, out);
                out.push_str(": ");
                push_python_json(item, out);
            }
            out.push('}');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn push_json_string(text: &str, out: &mut String) {
    out.push_str(&Value::from(text).to_string());
}

fn check_tool_name(name: &str) -> Result<(), String> {
    let valid = (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid tool name {name:?}: use 1-64 ASCII letters, digits, '_', '-' or '.'"
        ))
    }
}

/// A parameter name must survive the round trip through `<parameter=NAME>`.
fn check_parameter_name(name: &str) -> Result<(), String> {
    let valid = !name.is_empty()
        && !name.chars().any(|character| {
            matches!(character, '<' | '>') || character.is_whitespace() || character.is_control()
        });
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid parameter name {name:?}: it must be non-empty without '<', '>', whitespace or control characters"
        ))
    }
}

/// Compile a tool's parameter schema once, for the whole [`ToolSet`].
///
/// Validation is delegated to the `jsonschema` crate, which implements every
/// assertion of the supported drafts (2020-12 when `$schema` is absent).
/// Retrieval is disabled: a `$ref`, `$schema` or required `$vocabulary` that
/// does not resolve inside the schema itself or the crate's bundled
/// meta-schemas makes the schema invalid instead of being fetched or skipped.
/// Known `format`s are asserted; unknown ones are annotations, as are `title`,
/// `description`, `default`, `examples` and other unknown keywords.
fn compile_parameters(schema: &Value) -> Result<Option<ArgumentSchema>, String> {
    let map = match schema {
        Value::Null => return Ok(None),
        Value::Object(map) => map,
        _ => return Err("must be a JSON Schema object or null".into()),
    };
    if map.get("type").is_some_and(|kind| kind != "object") {
        return Err("top-level type must be \"object\"".into());
    }
    let arguments = schema_options()
        .build(schema)
        .map_err(|error| format!("invalid schema: {error}"))?;
    // Parameter values are converted against their own subschema, which may
    // `$ref` anything in the document, so subschemas are compiled as
    // references into a local registry that holds only this schema.
    let registry = jsonschema::Registry::new()
        .retriever(NoRetrieval)
        .add(SCHEMA_URI, schema.clone())
        .and_then(jsonschema::RegistryBuilder::prepare)
        .map_err(|error| format!("invalid schema: {error}"))?;
    let parameter = |pointer: String, declared: &Value| -> Result<Parameter, String> {
        let mut reference = Map::new();
        if let Some(dialect) = map.get("$schema") {
            reference.insert("$schema".into(), dialect.clone());
        }
        reference.insert(
            "$ref".into(),
            Value::from(format!("{SCHEMA_URI}#{pointer}")),
        );
        let validator = schema_options()
            .with_registry(&registry)
            .build(&Value::Object(reference))
            .map_err(|error| format!("invalid schema at {pointer}: {error}"))?;
        let unconstrained = [
            Value::Null,
            Value::Bool(false),
            Value::from(0),
            Value::from(""),
            Value::Array(Vec::new()),
            Value::Object(Map::new()),
        ]
        .iter()
        .all(|probe| validator.is_valid(probe));
        Ok(Parameter {
            validator,
            unconstrained,
            any_string: accepts_every_string(declared),
        })
    };
    let mut properties = HashMap::new();
    if let Some(Value::Object(declared)) = map.get("properties") {
        for (name, schema) in declared {
            check_parameter_name(name)?;
            let pointer = format!("/properties/{}", fragment_segment(name));
            properties.insert(name.clone(), parameter(pointer, schema)?);
        }
    }
    let additional = match map.get("additionalProperties") {
        Some(schema @ Value::Object(_)) => Some(parameter("/additionalProperties".into(), schema)?),
        _ => None,
    };
    Ok(Some(ArgumentSchema {
        arguments,
        properties,
        additional,
    }))
}

/// Whether `schema` is a string type with nothing but annotations, which
/// every string satisfies (formats are validated, so `format` constrains).
fn accepts_every_string(schema: &Value) -> bool {
    let Value::Object(map) = schema else {
        return false;
    };
    let string = match map.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind == "string"),
        _ => false,
    };
    string
        && map.keys().all(|key| {
            matches!(
                key.as_str(),
                "type"
                    | "description"
                    | "title"
                    | "examples"
                    | "default"
                    | "$comment"
                    | "deprecated"
                    | "readOnly"
                    | "writeOnly"
            )
        })
}

fn schema_options<'i>() -> jsonschema::ValidationOptions<'i> {
    jsonschema::options()
        .offline()
        .should_validate_formats(true)
}

/// Refuses every retrieval, so the registry never touches the network or disk.
struct NoRetrieval;

impl jsonschema::Retrieve for NoRetrieval {
    fn retrieve(
        &self,
        uri: &jsonschema::Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err(format!("external schema reference {uri} is not supported").into())
    }
}

/// One JSON Pointer segment, escaped for use in a URI fragment.
fn fragment_segment(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for byte in name.replace('~', "~0").replace('/', "~1").bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            out.push('%');
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 0xF)]));
        }
    }
    out
}

fn validate_arguments(tool: &Tool, arguments: &Value) -> Result<(), String> {
    let Some(schema) = &tool.schema else {
        return match arguments {
            Value::Object(map) if map.is_empty() => Ok(()),
            _ => Err(format!(
                "tool {:?} takes no arguments",
                tool.definition.name
            )),
        };
    };
    schema
        .arguments
        .validate(arguments)
        .map_err(|error| format!("arguments{}: {error}", error.instance_path()))
}

/// Process-unique call IDs that are also unlikely to repeat across processes:
/// a randomly keyed per-process prefix and a monotonic counter.
fn next_call_id() -> String {
    static PREFIX: OnceLock<u64> = OnceLock::new();
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let prefix = *PREFIX.get_or_init(|| {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        std::process::id().hash(&mut hasher);
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos())
            .hash(&mut hasher);
        hasher.finish()
    });
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{prefix:016x}{count:08x}")
}

/// Streaming parser for the visible answer when tools are configured.
///
/// Ordinary text streams as [`Event::Content`]; only a possible `<tool_call>`
/// prefix and trailing whitespace (which the template puts before a call) are
/// held back. While a call is generated, the part of its arguments that can
/// no longer change streams as [`Event::ToolCallDelta`]s; at `</tool_call>`
/// the call is parsed and validated whole and becomes one [`Event::ToolCall`]
/// or a recorded failure.
pub struct ToolCallParser {
    tools: Arc<ToolSet>,
    pending: String,
    in_call: bool,
    after_call: bool,
    failure: Option<String>,
    /// The open call's streamed deltas, once it named a configured tool.
    streamed: Option<Streamed>,
}

/// How much of the open call has been streamed: whole pieces, then bytes of
/// the current piece's JSON and of its raw text.
struct Streamed {
    id: String,
    name: String,
    piece: usize,
    json: usize,
    text: usize,
}

/// One certain part of a call's arguments: the opening brace, or one
/// parameter as far as it has been written.
struct Piece {
    json: String,
    text: Option<(String, String)>,
}

impl ToolCallParser {
    pub const fn new(tools: Arc<ToolSet>) -> Self {
        Self {
            tools,
            pending: String::new(),
            in_call: false,
            after_call: false,
            failure: None,
            streamed: None,
        }
    }

    pub const fn failed(&self) -> bool {
        self.failure.is_some()
    }

    /// The parse failure that stopped this stream, if any.
    pub const fn take_failure(&mut self) -> Option<String> {
        self.failure.take()
    }

    pub fn feed(
        &mut self,
        text: &str,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if self.failure.is_some() {
            return ControlFlow::Break(());
        }
        self.pending.push_str(text);
        loop {
            if self.in_call {
                let tools = Arc::clone(&self.tools);
                let Some((end, marker, strict)) = call_end(&self.pending, &tools) else {
                    return self.stream(callback);
                };
                let rest = self.pending.split_off(end + marker.len());
                let mut block = std::mem::replace(&mut self.pending, rest);
                block.truncate(end);
                self.in_call = false;
                self.after_call = true;
                let parsed = strict.map_or_else(
                    || parse_call(&block, &tools),
                    |tool| parse_strict(&block, tool),
                );
                let streamed = self.streamed.take();
                match parsed {
                    Ok(mut call) => {
                        if let Some(streamed) = streamed {
                            call.id = streamed.id;
                        }
                        if callback(Event::ToolCall(call)).is_break() {
                            return ControlFlow::Break(());
                        }
                    }
                    Err(error) => {
                        self.failure = Some(format!("invalid tool call: {error}"));
                        return ControlFlow::Break(());
                    }
                }
                continue;
            }
            if let Some(start) = self.pending.find(CALL_START) {
                let rest = self.pending.split_off(start + CALL_START.len());
                let mut text = std::mem::replace(&mut self.pending, rest);
                text.truncate(start);
                self.in_call = true;
                if self.text(text.trim_end(), callback).is_break() {
                    return ControlFlow::Break(());
                }
                continue;
            }
            let split = self.pending.len() - held_suffix(&self.pending);
            if split == 0 {
                return ControlFlow::Continue(());
            }
            let rest = self.pending.split_off(split);
            let text = std::mem::replace(&mut self.pending, rest);
            return self.text(&text, callback);
        }
    }

    /// Close the stream for generation that stopped for `reason`.
    ///
    /// A call still open when the model ends its turn is malformed and
    /// fails the stream. A call cut off by the token budget or cancellation
    /// is no call at all: its buffered text is dropped, never emitted, and
    /// the events already delivered stand.
    pub fn finish(
        &mut self,
        reason: StopReason,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if self.failure.is_some() {
            return ControlFlow::Break(());
        }
        if self.in_call {
            match reason {
                StopReason::Eos => {
                    self.failure =
                        Some("invalid tool call: generation ended before </tool_call>".into());
                    return ControlFlow::Break(());
                }
                StopReason::TokenLimit | StopReason::Cancelled | StopReason::Stopped => {
                    self.pending.clear();
                    self.in_call = false;
                    self.streamed = None;
                    return ControlFlow::Continue(());
                }
            }
        }
        let text = std::mem::take(&mut self.pending);
        self.text(&text, callback)
    }

    /// Report whatever more of the open call's arguments is now certain.
    fn stream(&mut self, callback: &mut impl FnMut(Event) -> ControlFlow<()>) -> ControlFlow<()> {
        let Some((name, pieces)) = pieces(&self.pending, &self.tools) else {
            return ControlFlow::Continue(());
        };
        let streamed = self.streamed.get_or_insert_with(|| Streamed {
            id: next_call_id(),
            name: name.to_owned(),
            piece: 0,
            json: 0,
            text: 0,
        });
        let last = pieces.len() - 1;
        for (index, piece) in pieces.iter().enumerate().skip(streamed.piece) {
            let arguments = piece.json.get(streamed.json..).unwrap_or_default();
            let text = piece.text.as_ref().map(|(key, text)| {
                (
                    key.clone(),
                    text.get(streamed.text..).unwrap_or_default().to_owned(),
                )
            });
            let delta = ToolCallDelta {
                id: streamed.id.clone(),
                name: streamed.name.clone(),
                arguments: arguments.to_owned(),
                text,
            };
            if index < last {
                (streamed.piece, streamed.json, streamed.text) = (index + 1, 0, 0);
            } else {
                streamed.json = piece.json.len();
                streamed.text = piece.text.as_ref().map_or(0, |(_, text)| text.len());
            }
            if !delta.arguments.is_empty() && callback(Event::ToolCallDelta(delta)).is_break() {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }

    fn text(
        &mut self,
        text: &str,
        callback: &mut impl FnMut(Event) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let text = if self.after_call {
            text.trim_start()
        } else {
            text
        };
        if text.is_empty() {
            return ControlFlow::Continue(());
        }
        self.after_call = false;
        callback(Event::Content(text.to_owned()))
    }
}

/// Where the open call ends: the block's end, the marker after it, and
/// the strict tool whose exact layout the grammar enforced, if any.
///
/// A strict call ends at `\n</function>\n</tool_call>`, which neither a
/// raw value (no `</tool_call>`) nor JSON (no raw newline before `<`) can
/// contain, so a JSON string holding `</tool_call>` does not cut it short.
/// Every other call ends at the first `</tool_call>`, as it always has.
fn call_end<'t>(
    pending: &str,
    tools: &'t ToolSet,
) -> Option<(usize, &'static str, Option<&'t Tool>)> {
    let legacy = || pending.find(CALL_END).map(|end| (end, CALL_END, None));
    if !tools.any_strict() {
        return legacy();
    }
    // Until the header names a strict tool, only a legacy end can apply.
    let Some((name, _)) = pending
        .strip_prefix(FUNCTION_OPEN)
        .and_then(|rest| rest.split_once('>'))
    else {
        return legacy();
    };
    match tools.get(name) {
        Some(tool) if tool.strict.is_some() => pending
            .find(STRICT_CALL_END)
            .map(|end| (end, STRICT_CALL_END, Some(tool))),
        _ => legacy(),
    }
}

/// Bytes at the end of `text` that might still become `<tool_call>` or the
/// whitespace the template puts before one.
fn held_suffix(text: &str) -> usize {
    let tag = CALL_START.as_bytes();
    let partial = (1..tag.len())
        .rev()
        .find(|&length| text.as_bytes().ends_with(&tag[..length]))
        .unwrap_or(0);
    let (head, _) = text.split_at(text.len() - partial);
    partial + head.len() - head.trim_end().len()
}

/// The certain pieces of the arguments of the call `block` (everything after
/// `<tool_call>` so far) is writing, once it names a configured tool.
///
/// The text is read as [`parse_call`] and [`parse_strict`] read it, and
/// each finished value is converted exactly as they convert it, so the
/// pieces always spell a prefix of the arguments those produce. A raw string
/// value streams only when it is certain to stay a string; it holds back any
/// tail that could still turn out to be its closing `\n</parameter>`.
/// Streaming stops at anything those parsers would refuse, and the call
/// then fails when it ends.
fn pieces<'b>(block: &'b str, tools: &ToolSet) -> Option<(&'b str, Vec<Piece>)> {
    let (name, mut rest) = block
        .trim_start()
        .strip_prefix("<function=")?
        .split_once('>')?;
    let tool = tools.get(name)?;
    let mut pieces = vec![Piece {
        json: "{".into(),
        text: None,
    }];
    let mut seen = HashSet::new();
    loop {
        rest = rest.trim_start();
        let Some((key, after)) = rest
            .strip_prefix("<parameter=")
            .and_then(|after| after.split_once('>'))
        else {
            break;
        };
        let text_mode = match streams_text(tool, key) {
            Some(text_mode) if check_parameter_name(key).is_ok() && seen.insert(key) => text_mode,
            _ => break,
        };
        let mut json = if pieces.len() > 1 { "," } else { "" }.to_owned();
        json.push_str(&Value::from(key).to_string());
        json.push(':');
        if after.is_empty() {
            pieces.push(Piece { json, text: None });
            break;
        }
        let value = after.strip_prefix('\n').unwrap_or(after);
        if let Some((raw, next)) = value.split_once("</parameter>") {
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            let Some(converted) = finished_value(tool, key, raw) else {
                break;
            };
            json.push_str(&converted.to_string());
            let text = text_mode.then(|| (key.to_owned(), raw.to_owned()));
            pieces.push(Piece { json, text });
            rest = next;
            continue;
        }
        let text = text_mode.then(|| {
            let certain = value
                .get(..value.len() - held_close(value))
                .unwrap_or_default();
            let quoted = Value::from(certain).to_string();
            json.push_str(quoted.strip_suffix('"').unwrap_or_default());
            (key.to_owned(), certain.to_owned())
        });
        pieces.push(Piece { json, text });
        break;
    }
    Some((name, pieces))
}

/// Whether parameter `key` of `tool` is a string whatever its text (so it
/// can stream raw), `Some(false)` when its value must be seen whole, and
/// `None` when the call cannot be valid.
fn streams_text(tool: &Tool, key: &str) -> Option<bool> {
    match &tool.strict {
        Some(Ok(strict)) => Some(matches!(strict.parameter(key)?.value, ValueKind::Raw(_))),
        Some(Err(_)) => None,
        None => Some(
            tool.schema
                .as_ref()
                .and_then(|schema| schema.properties.get(key).or(schema.additional.as_ref()))
                .is_some_and(|parameter| parameter.any_string),
        ),
    }
}

/// The value of parameter `key` with text `raw`, as the call's parser will
/// convert it.
fn finished_value(tool: &Tool, key: &str, raw: &str) -> Option<Value> {
    match &tool.strict {
        Some(Ok(strict)) => match &strict.parameter(key)?.value {
            ValueKind::Raw(_) => Some(Value::from(raw)),
            ValueKind::Json(_) => serde_json::from_str(raw).ok(),
        },
        Some(Err(_)) => None,
        None => Some(convert(
            raw,
            tool.schema
                .as_ref()
                .and_then(|schema| schema.properties.get(key).or(schema.additional.as_ref())),
        )),
    }
}

/// Bytes at the end of a raw value that might still be the start of its
/// closing `\n</parameter>`.
fn held_close(text: &str) -> usize {
    ["\n</parameter>", "</parameter>"]
        .iter()
        .filter_map(|tag| {
            (1..tag.len())
                .rev()
                .find(|&length| tag.get(..length).is_some_and(|start| text.ends_with(start)))
        })
        .max()
        .unwrap_or(0)
}

fn parse_call(block: &str, tools: &ToolSet) -> Result<ToolCall, String> {
    let body = block
        .trim()
        .strip_prefix("<function=")
        .ok_or("expected <function=NAME>")?;
    let (name, body) = body.split_once('>').ok_or("unterminated <function= tag")?;
    let mut rest = body
        .trim_end()
        .strip_suffix("</function>")
        .ok_or("missing </function>")?;
    // A call to a tool this request did not declare goes to the client like
    // any other: harnesses answer it with an error result the model can act
    // on (Codex's default tools have no `apply_patch`, yet its instructions
    // name it), where failing the response leaves nothing to correct.
    check_tool_name(name)?;
    let tool = tools.get(name);
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let parameter = rest
            .strip_prefix("<parameter=")
            .ok_or("unexpected text outside <parameter=...> blocks")?;
        let (key, parameter) = parameter
            .split_once('>')
            .ok_or("unterminated <parameter= tag")?;
        check_parameter_name(key)?;
        let parameter = parameter.strip_prefix('\n').unwrap_or(parameter);
        let (raw, after) = parameter
            .split_once("</parameter>")
            .ok_or_else(|| format!("parameter {key:?} is missing </parameter>"))?;
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        if arguments.contains_key(key) {
            return Err(format!("duplicate parameter {key:?}"));
        }
        let parameter = tool
            .and_then(|tool| tool.schema.as_ref())
            .and_then(|schema| schema.properties.get(key).or(schema.additional.as_ref()));
        let value = convert(raw, parameter);
        arguments.insert(key.to_owned(), value);
        rest = after;
    }
    // The order the model wrote, which streamed deltas already reported
    // (`llguidance` enables serde_json's `preserve_order`).
    let arguments = Value::Object(arguments);
    // A non-strict tool's schema guides the model but binds nobody: the call
    // goes to the client as generated, and the client validates it and
    // reports any error back as the call's result. Failing the whole
    // response instead leaves a client nothing to correct (Oh My Pi retried
    // the identical request until it gave up). Only a strict tool promises
    // arguments that satisfy its schema.
    if let Some(tool) = tool.filter(|tool| tool.strict.is_some()) {
        validate_arguments(tool, &arguments).map_err(|error| format!("{name}: {error}"))?;
    }
    Ok(ToolCall {
        id: next_call_id(),
        name: name.to_owned(),
        arguments,
    })
}

/// Parse a strict tool's call exactly as its grammar spelled it.
///
/// `block` runs from just after `<tool_call>` to just before
/// [`STRICT_CALL_END`]: `\n<function=NAME>` and then, per parameter,
/// `\n<parameter=KEY>\nVALUE\n</parameter>`. A raw value is the string
/// itself; any other value is JSON. The arguments are then validated against
/// the tool's whole schema, like every call.
fn parse_strict(block: &str, tool: &Tool) -> Result<ToolCall, String> {
    let name = &tool.definition.name;
    let Some(Ok(strict)) = &tool.strict else {
        return Err(format!("{name}: strict tool has no enforced layout"));
    };
    let mut rest = block
        .strip_prefix(FUNCTION_OPEN)
        .and_then(|rest| rest.strip_prefix(name.as_str()))
        .and_then(|rest| rest.strip_prefix('>'))
        .ok_or("expected <function=NAME>")?;
    let mut arguments = Map::new();
    while !rest.is_empty() {
        let parameter = rest
            .strip_prefix(PARAMETER_OPEN)
            .ok_or("expected <parameter=NAME>")?;
        let (key, parameter) = parameter
            .split_once(">\n")
            .ok_or("unterminated <parameter= tag")?;
        let declared = strict
            .parameter(key)
            .ok_or_else(|| format!("unknown parameter {key:?}"))?;
        let (raw, after) = parameter
            .split_once(PARAMETER_CLOSE)
            .ok_or_else(|| format!("parameter {key:?} is missing </parameter>"))?;
        if arguments.contains_key(key) {
            return Err(format!("duplicate parameter {key:?}"));
        }
        let value = match &declared.value {
            ValueKind::Raw(_) => Value::from(raw),
            ValueKind::Json(_) => serde_json::from_str(raw)
                .map_err(|error| format!("parameter {key:?} is not JSON: {error}"))?,
        };
        arguments.insert(key.to_owned(), value);
        rest = after;
    }
    let arguments = Value::Object(arguments);
    validate_arguments(tool, &arguments).map_err(|error| format!("{name}: {error}"))?;
    Ok(ToolCall {
        id: next_call_id(),
        name: name.clone(),
        arguments,
    })
}

/// Strings are raw text and everything else JSON. Raw text wins whenever the
/// parameter's schema accepts it (so `true` for a string, a nullable string
/// or a `$ref` to one stays text); otherwise JSON that satisfies the schema
/// wins, and anything else stays text for validation to judge. Without a
/// constraining schema the text carries no type, so valid JSON is used.
fn convert(raw: &str, parameter: Option<&Parameter>) -> Value {
    let text = Value::from(raw);
    let parsed = serde_json::from_str::<Value>(raw).ok();
    match parameter {
        Some(parameter) if !parameter.unconstrained => {
            if parameter.validator.is_valid(&text) {
                return text;
            }
            match parsed {
                Some(value) if parameter.validator.is_valid(&value) => value,
                _ => text,
            }
        }
        _ => parsed.unwrap_or(text),
    }
}

fn invalid<T>(message: impl Into<String>) -> crate::Result<T> {
    Err(crate::Error::InvalidArgument(message.into()))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
