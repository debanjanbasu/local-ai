//! Native constrained tool calling: the request's tool policy compiled into
//! one llguidance grammar over the checkpoint's own call format.
//!
//! The grammar is a Lark root that spells the Bonsai XML call layout with
//! quoted literal tags, plus one named JSON-schema subgrammar per JSON-valued
//! strict parameter (referenced as `@name`) and, with a response format, one
//! for the final answer. Every target selection is masked by it, so a call
//! that the grammar admits is a call the model actually produced; completed
//! calls are still parsed and checked against the tool's full schema by
//! [`super::ToolSet`].
//!
//! A strict tool's call is spelled exactly as the chat template renders one:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! with the declared parameters in sorted order, each required one present and
//! each optional one present at most once. A string parameter's VALUE is raw
//! text, constrained by a regular expression translated from its schema; any
//! other parameter's VALUE is JSON from its own schema, in the template's
//! `", "`/`": "` style. A schema the translation cannot enforce exactly is an
//! error, never a weaker constraint.
//!
//! `<tool_call>` and `</tool_call>` are single special tokens in the pinned
//! checkpoint (ids 248058 and 248059), which is also how the prompt and
//! replayed history encode them. Literal text can never produce a special
//! token, so where the tokenizer has them, the grammar accepts the tag either
//! as that one token or as quoted literal text; no other special token is
//! ever allowed, and free text can never contain the opening tag in either
//! form.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use llguidance::derivre::{RegexAst, RegexBuilder};
use serde_json::{Map, Value, json};

use super::{ToolChoice, ToolSet};

/// Closes a strict parameter value.
pub(super) const PARAMETER_CLOSE: &str = "\n</parameter>";
/// Opens a strict parameter after the header or a previous parameter.
pub(super) const PARAMETER_OPEN: &str = "\n<parameter=";
/// Ends a strict call: what follows the block parsed by `parse_strict`.
pub(super) const STRICT_CALL_END: &str = "\n</function>\n</tool_call>";
/// Opens every call body right after `<tool_call>`.
pub(super) const FUNCTION_OPEN: &str = "\n<function=";

/// Free text: anything that does not contain the opening tag. Special tokens
/// are never text.
const TEXT: &str = r"/(?s:.*)/ & ~/(?s:.*)<tool_call>(?s:.*)/";
/// The base of every raw string value: no delimiter the parser splits on.
const RAW_BASE: &str =
    r"/(?s:.*)/ & ~/(?s:.*)<\/parameter>(?s:.*)/ & ~/(?s:.*)<\/tool_call>(?s:.*)/";
/// A non-strict call body: unconstrained arguments that end `</function>`,
/// with no closing tag inside. The completed call is validated afterwards.
const GENERIC_BODY: &str =
    r"/(?s:.*)/ & ~/(?s:.*)<\/tool_call>(?s:.*)/ & /(?s:.*)<\/function>[ \t\r\n]*/";

/// ECMA-262 `\s`: `WhiteSpace` and `LineTerminator`, unlike Rust's Unicode
/// `White_Space` (which adds U+0085 and drops U+FEFF).
const ECMA_SPACE: &str = r"\t\n\x{0B}\x{0C}\r \x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";

/// Bounds keeping the translated regular expressions small.
const MAX_LENGTH_BOUND: u64 = 16_384;
const MAX_ENUM: usize = 256;
const MAX_PATTERN: usize = 1_024;
const MAX_PARAMETERS: usize = 128;
const MAX_REF_DEPTH: usize = 16;
const REGEX_FUEL: u64 = 1_000_000;

/// Keywords that only annotate a schema.
const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "$comment",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// How a strict tool's arguments are spelled and read back.
#[derive(Debug)]
pub struct StrictTool {
    /// Declared parameters in sorted (rendered) order.
    pub(super) parameters: Vec<StrictParameter>,
}

#[derive(Debug)]
pub(super) struct StrictParameter {
    pub(super) name: String,
    pub(super) required: bool,
    pub(super) value: ValueKind,
}

#[derive(Debug)]
pub(super) enum ValueKind {
    /// Raw text matching every one of these Lark regular expressions.
    Raw(Vec<String>),
    /// JSON valid under this self-contained schema.
    Json(Value),
}

impl StrictTool {
    pub(super) fn parameter(&self, name: &str) -> Option<&StrictParameter> {
        self.parameters
            .iter()
            .find(|parameter| parameter.name == name)
    }
}

/// The checkpoint's special-token ids for the call tags, when it has them.
#[derive(Clone, Copy, Debug, Default)]
pub struct CallTags {
    pub open: Option<u32>,
    pub close: Option<u32>,
}

/// A compiled tool policy: the grammar and whether the answer is mandatory
/// (so even the reasoning phase may not end the request).
pub struct ToolGrammar {
    pub grammar: TopLevelGrammar,
    pub call_required: bool,
    /// Each JSON-valued strict parameter's own schema and its tool and
    /// parameter, to say which one the grammar engine cannot enforce.
    pub parameters: Vec<(String, Value)>,
}

/// Whether a request with these settings needs a native grammar. The
/// default (`auto`, every tool non-strict, parallel calls, text answer)
/// keeps the unconstrained path exactly.
pub fn needs_grammar(
    tools: &ToolSet,
    choice: &ToolChoice,
    parallel: bool,
    format_is_text: bool,
) -> bool {
    !tools.is_empty()
        && (*choice != ToolChoice::Auto || !parallel || tools.any_strict() || !format_is_text)
}

/// Build the grammar for `tools` under `choice`, `parallel` and an optional
/// final-answer `format` schema.
pub fn build(
    tools: &ToolSet,
    choice: &ToolChoice,
    parallel: bool,
    format: Option<Value>,
    tags: CallTags,
) -> Result<ToolGrammar, String> {
    let callable: Vec<usize> = match choice {
        ToolChoice::Function(name) => vec![
            tools
                .tools
                .iter()
                .position(|tool| tool.definition.name == *name)
                .ok_or_else(|| format!("tool_choice names unknown tool {name:?}"))?,
        ],
        _ => (0..tools.tools.len()).collect(),
    };
    let mut lark = String::from("%llguidance {}\n");
    let mut grammars = Vec::new();
    let mut parameters = Vec::new();
    let open = alternatives("\"<tool_call>\"", tags.open);
    let close = alternatives("\"</tool_call>\"", tags.close);
    let more = if parallel {
        "(GAP? open call)* GAP?"
    } else {
        "GAP?"
    };
    let final_answer = format.is_some();
    if let Some(schema) = format {
        grammars.push(GrammarWithLexer {
            name: Some("final".into()),
            json_schema: Some(schema),
            lark_grammar: None,
        });
    }
    let call_required = matches!(choice, ToolChoice::Required | ToolChoice::Function(_));
    let start = match (choice, final_answer) {
        (ToolChoice::None, false) => "TEXT".to_owned(),
        (ToolChoice::None, true) => "LEAD? @final".to_owned(),
        (ToolChoice::Auto, false) => format!("TEXT | pre call {more}"),
        (ToolChoice::Auto, true) => format!("LEAD? @final | LEAD? open call {more}"),
        (ToolChoice::Required | ToolChoice::Function(_), _) => format!("LEAD? open call {more}"),
    };
    let _ = writeln!(lark, "start: {start}");
    let pre = tags.open.map_or_else(
        || "hd".to_owned(),
        |id| format!("hd | TEXT <[{id}]> | <[{id}]>"),
    );
    let _ = writeln!(lark, "pre: {pre}");
    lark.push_str("hd[lazy]: TEXT \"<tool_call>\"\n");
    let _ = writeln!(lark, "open: {open}");
    let _ = writeln!(lark, "close: {close}");
    let _ = writeln!(lark, "TEXT: {TEXT}");
    lark.push_str("LEAD: /\\n{1,4}/\n");
    lark.push_str("GAP: /[ \\t\\r\\n]{1,8}/\n");
    let _ = writeln!(lark, "RAW_BASE: {RAW_BASE}");
    let _ = writeln!(lark, "GENERIC: {GENERIC_BODY}");
    lark.push_str("GENERIC_CLOSE: GENERIC \"</tool_call>\"\n");
    let generic = tags.close.map_or_else(
        || "GENERIC_CLOSE".to_owned(),
        |id| format!("GENERIC_CLOSE | GENERIC <[{id}]>"),
    );
    let _ = writeln!(lark, "generic: {generic}");
    let calls = callable
        .iter()
        .map(|index| format!("fn_{index}"))
        .collect::<Vec<_>>()
        .join(" | ");
    let _ = writeln!(lark, "call: {calls}");
    for &index in &callable {
        let tool = &tools.tools[index];
        let name = &tool.definition.name;
        let header = lark_string(&format!("\n<function={name}>\n"));
        let Some(strict) = &tool.strict else {
            let _ = writeln!(lark, "fn_{index}: {header} generic");
            continue;
        };
        let strict = strict
            .as_ref()
            .map_err(|error| format!("strict tool {name:?}: {error}"))?;
        let rule = strict_rule(
            index,
            name,
            strict,
            &mut lark,
            &mut grammars,
            &mut parameters,
        );
        lark.push_str(&rule);
    }
    grammars.insert(0, GrammarWithLexer::from_lark(lark));
    Ok(ToolGrammar {
        grammar: TopLevelGrammar {
            grammars,
            max_tokens: None,
        },
        call_required,
        parameters,
    })
}

/// The rules for strict tool `index`, adding its parameter terminals to
/// `lark` and its JSON subgrammars to `grammars`; returns its `fn_` rule.
fn strict_rule(
    index: usize,
    name: &str,
    strict: &StrictTool,
    lark: &mut String,
    grammars: &mut Vec<GrammarWithLexer>,
    parameters: &mut Vec<(String, Value)>,
) -> String {
    let mut body = Vec::new();
    for (position, parameter) in strict.parameters.iter().enumerate() {
        let rule = format!("p_{index}_{position}");
        let key = lark_string(&format!("<parameter={}>\n", parameter.name));
        match &parameter.value {
            ValueKind::Raw(constraints) => {
                let terminal = format!("V_{index}_{position}");
                let mut pattern = String::from("RAW_BASE");
                for constraint in constraints {
                    let _ = write!(pattern, " & /{constraint}/");
                }
                let _ = writeln!(lark, "{terminal}: ({pattern}) \"\\n</parameter>\\n\"");
                let _ = writeln!(lark, "{rule}: {key} {terminal}");
            }
            ValueKind::Json(schema) => {
                let reference = format!("json_{index}_{position}");
                let mut schema = schema.clone();
                if let Value::Object(map) = &mut schema {
                    map.insert(
                        "x-guidance".into(),
                        json!({
                            "whitespace_flexible": false,
                            "item_separator": ", ",
                            "key_separator": ": ",
                            "lenient": false,
                            "coerce_one_of": false,
                        }),
                    );
                }
                parameters.push((
                    format!("strict tool {name:?} parameter {:?}", parameter.name),
                    schema.clone(),
                ));
                grammars.push(GrammarWithLexer {
                    name: Some(reference.clone()),
                    json_schema: Some(schema),
                    lark_grammar: None,
                });
                let _ = writeln!(lark, "{rule}: {key} @{reference} \"\\n</parameter>\\n\"");
            }
        }
        body.push(if parameter.required {
            rule
        } else {
            format!("{rule}?")
        });
    }
    let header = lark_string(&format!("\n<function={name}>\n"));
    format!(
        "fn_{index}: {header} {} \"</function>\\n\" close\n",
        body.join(" ")
    )
}

fn alternatives(literal: &str, special: Option<u32>) -> String {
    special.map_or_else(|| literal.to_owned(), |id| format!("{literal} | <[{id}]>"))
}

/// A Lark string literal (JSON string syntax, which Lark reads back with
/// `serde_json`; DEL must be escaped too).
fn lark_string(text: &str) -> String {
    Value::from(text).to_string().replace('\u{7f}', "\\u007f")
}

/// Translate one strict tool's parameter schema, or say exactly why not.
#[allow(clippy::too_many_lines)] // One linear check per root keyword.
pub(super) fn classify(parameters: &Value) -> Result<StrictTool, String> {
    let root = match parameters {
        Value::Null => {
            return Ok(StrictTool {
                parameters: Vec::new(),
            });
        }
        Value::Object(root) => root,
        _ => return Err("parameters must be a JSON Schema object or null".into()),
    };
    for (key, value) in root {
        match key.as_str() {
            "type" if value == "object" => {}
            "type" => return Err("top-level type must be \"object\"".into()),
            "properties" | "required" | "$defs" | "definitions" | "minProperties"
            | "maxProperties" => {}
            "additionalProperties" if *value == Value::Bool(false) => {}
            "additionalProperties" => {
                return Err("strict parameters must set \"additionalProperties\": false".into());
            }
            "$schema" => {
                const DIALECTS: &[&str] = &[
                    "https://json-schema.org/draft/2020-12/schema",
                    "https://json-schema.org/draft/2019-09/schema",
                    "http://json-schema.org/draft-07/schema#",
                    "http://json-schema.org/draft-07/schema",
                ];
                if !value.as_str().is_some_and(|uri| DIALECTS.contains(&uri)) {
                    return Err(format!("unsupported $schema {value}"));
                }
            }
            key if ANNOTATIONS.contains(&key) => {}
            key => {
                return Err(format!(
                    "top-level keyword {key:?} is not supported in strict mode"
                ));
            }
        }
    }
    if !root.contains_key("additionalProperties") {
        return Err("strict parameters must set \"additionalProperties\": false".into());
    }
    let declared = match root.get("properties") {
        None => Map::new(),
        Some(Value::Object(declared)) => declared.clone(),
        Some(_) => return Err("\"properties\" must be an object".into()),
    };
    if declared.len() > MAX_PARAMETERS {
        return Err(format!("more than {MAX_PARAMETERS} parameters"));
    }
    let mut required = HashSet::new();
    match root.get("required") {
        None => {}
        Some(Value::Array(names)) => {
            for name in names {
                let Some(name) = name.as_str() else {
                    return Err("\"required\" must list strings".into());
                };
                if !declared.contains_key(name) {
                    return Err(format!(
                        "required parameter {name:?} is not declared, so no call can satisfy the schema"
                    ));
                }
                if !required.insert(name.to_owned()) {
                    return Err(format!("\"required\" lists {name:?} twice"));
                }
            }
        }
        Some(_) => return Err("\"required\" must be an array".into()),
    }
    // Only bounds the fixed layout provably meets: at least the required
    // parameters, at most the declared ones.
    if let Some(minimum) = root.get("minProperties") {
        let minimum = minimum
            .as_u64()
            .ok_or("\"minProperties\" must be a non-negative integer")?;
        if minimum > required.len() as u64 {
            return Err(format!(
                "\"minProperties\": {minimum} exceeds the {} required parameters; only a bound the required parameters meet is enforced",
                required.len()
            ));
        }
    }
    if let Some(maximum) = root.get("maxProperties") {
        let maximum = maximum
            .as_u64()
            .ok_or("\"maxProperties\" must be a non-negative integer")?;
        if maximum < declared.len() as u64 {
            return Err(format!(
                "\"maxProperties\": {maximum} is below the {} declared parameters; only a bound every call meets is enforced",
                declared.len()
            ));
        }
    }
    let definitions = root
        .iter()
        .filter(|(key, _)| matches!(key.as_str(), "$defs" | "definitions"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<_, _>>();
    let sorted: BTreeMap<&String, &Value> = declared.iter().collect();
    let mut spelled = Vec::with_capacity(sorted.len());
    for (name, schema) in sorted {
        let value = classify_parameter(parameters, schema, &definitions)
            .map_err(|error| format!("parameter {name:?}: {error}"))?;
        spelled.push(StrictParameter {
            name: name.clone(),
            required: required.contains(name.as_str()),
            value,
        });
    }
    Ok(StrictTool {
        parameters: spelled,
    })
}

/// Follow a property's own `$ref` chain inside the parameter schema.
fn resolve<'a>(root: &'a Value, mut schema: &'a Value) -> Result<&'a Value, String> {
    for _ in 0..MAX_REF_DEPTH {
        let Some(map) = schema.as_object() else {
            return Ok(schema);
        };
        let Some(reference) = map.get("$ref") else {
            return Ok(schema);
        };
        if let Some(key) = map
            .keys()
            .find(|key| *key != "$ref" && !ANNOTATIONS.contains(&key.as_str()))
        {
            return Err(format!("keyword {key:?} beside \"$ref\" is not supported"));
        }
        let pointer = reference
            .as_str()
            .and_then(|reference| reference.strip_prefix('#'))
            .filter(|pointer| local_definition(pointer))
            .ok_or_else(|| {
                format!("$ref {reference} must point into the schema's own $defs or definitions")
            })?;
        schema = root
            .pointer(pointer)
            .ok_or_else(|| format!("$ref {reference} does not resolve"))?;
    }
    Err("$ref chain is too deep".into())
}

fn local_definition(pointer: &str) -> bool {
    pointer.starts_with("/$defs/") || pointer.starts_with("/definitions/")
}

fn classify_parameter(
    root: &Value,
    schema: &Value,
    definitions: &Map<String, Value>,
) -> Result<ValueKind, String> {
    let schema = resolve(root, schema)?;
    let Some(map) = schema.as_object() else {
        return Err("schema must be an object (true/false schemas are not supported)".into());
    };
    let types: Option<Vec<&str>> = match map.get("type") {
        None => None,
        Some(Value::String(kind)) => Some(vec![kind.as_str()]),
        Some(Value::Array(kinds)) => Some(
            kinds
                .iter()
                .map(|kind| kind.as_str().ok_or("\"type\" must name types"))
                .collect::<Result<_, _>>()?,
        ),
        Some(_) => return Err("\"type\" must be a string or an array".into()),
    };
    let values: Option<Vec<&Value>> = match (map.get("enum"), map.get("const")) {
        (Some(_), Some(_)) => return Err("\"enum\" beside \"const\" is not supported".into()),
        (Some(Value::Array(values)), None) => Some(values.iter().collect()),
        (Some(_), None) => return Err("\"enum\" must be an array".into()),
        (None, Some(value)) => Some(vec![value]),
        (None, None) => None,
    };
    let raw = match &types {
        Some(types) if types.contains(&"string") => {
            if types.iter().any(|kind| !matches!(*kind, "string" | "null")) {
                return Err(
                    "a type list mixing \"string\" with non-null types is ambiguous in the raw-text format"
                        .into(),
                );
            }
            true
        }
        Some(_) => false,
        None => match &values {
            Some(values) if values.iter().all(|value| value.is_string()) => true,
            Some(values) if values.iter().all(|value| !value.is_string()) => false,
            Some(_) => {
                return Err("an enum mixing strings and other values is not supported".into());
            }
            None => return Err("strict parameters must declare \"type\" (or enum/const)".into()),
        },
    };
    if raw {
        raw_constraints(map, values).map(ValueKind::Raw)
    } else {
        json_schema(schema, definitions).map(ValueKind::Json)
    }
}

/// The regular expressions a raw string value must match.
///
/// A raw value can only ever be read back as a string, so a nullable string
/// schema is enforced as its string part: every value produced is valid.
fn raw_constraints(
    map: &Map<String, Value>,
    values: Option<Vec<&Value>>,
) -> Result<Vec<String>, String> {
    for key in map.keys() {
        if !matches!(
            key.as_str(),
            "type" | "enum" | "const" | "pattern" | "minLength" | "maxLength"
        ) && !ANNOTATIONS.contains(&key.as_str())
        {
            return Err(format!(
                "keyword {key:?} is not supported for strict raw strings"
            ));
        }
    }
    let mut constraints = Vec::new();
    if let Some(values) = values {
        let strings: Vec<&str> = values.iter().filter_map(|value| value.as_str()).collect();
        if values
            .iter()
            .any(|value| !value.is_string() && !value.is_null())
        {
            return Err("enum/const of a string must hold strings".into());
        }
        if strings.is_empty() {
            return Err("enum/const allows no string, so no value is valid".into());
        }
        if strings.len() > MAX_ENUM {
            return Err(format!("more than {MAX_ENUM} enum values"));
        }
        let alternatives = strings
            .iter()
            .map(|text| regex_literal(text))
            .collect::<Vec<_>>()
            .join("|");
        constraints.push(format!("(?:{alternatives})"));
    }
    if let Some(pattern) = map.get("pattern") {
        let pattern = pattern.as_str().ok_or("\"pattern\" must be a string")?;
        constraints.push(search_pattern(pattern)?);
    }
    let bound = |key: &str| -> Result<Option<u64>, String> {
        match map.get(key) {
            None => Ok(None),
            Some(value) => {
                let value = value
                    .as_u64()
                    .ok_or_else(|| format!("{key:?} must be a non-negative integer"))?;
                if value > MAX_LENGTH_BOUND {
                    return Err(format!("{key:?} above {MAX_LENGTH_BOUND} is not supported"));
                }
                Ok(Some(value))
            }
        }
    };
    match (bound("minLength")?, bound("maxLength")?) {
        (None, None) => {}
        (Some(minimum), Some(maximum)) if minimum > maximum => {
            return Err("minLength exceeds maxLength".into());
        }
        (minimum, maximum) => {
            let minimum = minimum.unwrap_or(0);
            let maximum = maximum.map_or_else(String::new, |maximum| maximum.to_string());
            constraints.push(format!("(?s:.{{{minimum},{maximum}}})"));
        }
    }
    check_nonempty(&constraints)?;
    Ok(constraints)
}

/// Prove the raw value's language is not empty, so a required parameter
/// cannot strand the grammar.
fn check_nonempty(constraints: &[String]) -> Result<(), String> {
    let mut builder = RegexBuilder::new();
    let mut parts = vec![
        RegexAst::Regex("(?s:.*)".into()),
        RegexAst::Not(Box::new(RegexAst::Regex(
            r"(?s:.*)<\/parameter>(?s:.*)".into(),
        ))),
        RegexAst::Not(Box::new(RegexAst::Regex(
            r"(?s:.*)<\/tool_call>(?s:.*)".into(),
        ))),
    ];
    parts.extend(
        constraints
            .iter()
            .map(|constraint| RegexAst::Regex(constraint.clone())),
    );
    let expression = builder
        .mk(&RegexAst::And(parts))
        .map_err(|error| format!("unsupported regular expression: {error}"))?;
    let mut regex = builder
        .to_regex_limited(expression, REGEX_FUEL)
        .map_err(|error| format!("regular expression is too complex: {error}"))?;
    if regex.always_empty() {
        return Err(
            "no raw value satisfies the schema (values may not contain </parameter> or </tool_call>)"
                .into(),
        );
    }
    Ok(())
}

/// Escape `text` as a literal inside a Lark `/.../` regular expression.
fn regex_literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
            | '#' | '&' | '-' | '~' | '/' => {
                out.push('\\');
                out.push(character);
            }
            character if character.is_control() || character.is_whitespace() => {
                let _ = write!(out, "\\x{{{:x}}}", u32::from(character));
            }
            character => out.push(character),
        }
    }
    out
}

/// A JSON Schema `pattern` (ECMA-262, unanchored search) as an anchored
/// Lark regular expression.
///
/// `\d` and `\w` are narrowed to ASCII as in ECMA-262, `.` excludes the four
/// ECMA line terminators, and a leading `^` / trailing `$` anchor the match;
/// otherwise the match may start and end anywhere. Anchors anywhere else,
/// lookarounds, backreferences, inline flags and other syntax without an
/// exact translation are rejected.
#[allow(clippy::too_many_lines)] // One scanner over the pattern syntax.
fn search_pattern(pattern: &str) -> Result<String, String> {
    if pattern.len() > MAX_PATTERN {
        return Err(format!("pattern longer than {MAX_PATTERN} bytes"));
    }
    let mut body = String::with_capacity(pattern.len());
    let mut characters = pattern.chars().peekable();
    let mut class = false;
    let mut depth = 0usize;
    let mut alternation = false;
    let mut anchored_start = false;
    let mut anchored_end = false;
    let mut first = true;
    while let Some(character) = characters.next() {
        let at_start = std::mem::take(&mut first);
        match character {
            '\\' => {
                let Some(next) = characters.next() else {
                    return Err("pattern ends with a lone backslash".into());
                };
                if next.is_ascii_digit() && next != '0' {
                    return Err("backreferences are not supported in patterns".into());
                }
                if matches!(next, 'b' | 'B') && !class {
                    return Err("word boundaries are not supported in patterns".into());
                }
                if matches!(next, 'c' | 'k') {
                    return Err(format!("\\{next} escapes are not supported in patterns"));
                }
                match (next, class) {
                    ('s', true) => body.push_str(ECMA_SPACE),
                    ('s', false) => {
                        body.push('[');
                        body.push_str(ECMA_SPACE);
                        body.push(']');
                    }
                    ('S', true) => {
                        return Err("\\S inside a character class is not supported".into());
                    }
                    ('S', false) => {
                        body.push_str("[^");
                        body.push_str(ECMA_SPACE);
                        body.push(']');
                    }
                    _ => {
                        body.push('\\');
                        body.push(next);
                    }
                }
            }
            '[' if !class => {
                class = true;
                body.push('[');
                if characters.peek() == Some(&'^') {
                    body.push('^');
                    characters.next();
                }
                if characters.peek() == Some(&']') {
                    return Err("an empty or ']'-leading character class is not supported".into());
                }
            }
            ']' if class => {
                class = false;
                body.push(']');
            }
            _ if class => {
                // Rust classes nest and have set operators; ECMA's do not.
                if character == '-' && characters.peek() == Some(&'-') {
                    return Err("'--' inside a character class is not supported".into());
                }
                if matches!(character, '[' | '&' | '~') {
                    body.push('\\');
                }
                body.push(character);
            }
            '(' => {
                if characters.peek() == Some(&'?') {
                    characters.next();
                    match characters.next() {
                        Some(':') => body.push_str("(?:"),
                        _ => {
                            return Err(
                                "only (?:...) groups are supported (no lookarounds, named groups or flags)"
                                    .into(),
                            );
                        }
                    }
                } else {
                    body.push('(');
                }
                depth += 1;
            }
            ')' => {
                depth = depth.checked_sub(1).ok_or("unbalanced ')' in pattern")?;
                body.push(')');
            }
            '|' => {
                if depth == 0 {
                    alternation = true;
                }
                body.push('|');
            }
            '^' if at_start => anchored_start = true,
            '$' if characters.peek().is_none() && depth == 0 => anchored_end = true,
            '^' | '$' => {
                return Err("anchors are only supported at the start and end of a pattern".into());
            }
            '.' => body.push_str(r"[^\n\r\x{2028}\x{2029}]"),
            character => body.push(character),
        }
    }
    if class || depth != 0 {
        return Err("unbalanced pattern".into());
    }
    if alternation && (anchored_start || anchored_end) {
        return Err("anchors around a top-level alternation are not supported".into());
    }
    let body = llguidance::regex_to_lark(&body, "dw");
    let lead = if anchored_start { "" } else { "(?s:.*)" };
    let tail = if anchored_end { "" } else { "(?s:.*)" };
    let translated = format!("{lead}(?:{body}){tail}");
    RegexBuilder::new()
        .mk_regex(&translated)
        .map_err(|error| format!("unsupported pattern: {error}"))?;
    Ok(translated)
}

/// A JSON-valued parameter's schema, made self-contained: the parameter
/// schema's definitions are copied in, and every reference must point into
/// them so it resolves to the same subschema it did in the whole document.
fn json_schema(schema: &Value, definitions: &Map<String, Value>) -> Result<Value, String> {
    check_references(schema)?;
    for definition in definitions.values() {
        check_references(definition)?;
    }
    let Value::Object(map) = schema else {
        return Err("schema must be an object".into());
    };
    if map.contains_key("x-guidance") {
        return Err("\"x-guidance\" is not supported".into());
    }
    if map.contains_key("$defs") || map.contains_key("definitions") {
        return Err(
            "nested $defs/definitions are not supported; declare them at the top level".into(),
        );
    }
    let mut map = map.clone();
    for (key, value) in definitions {
        map.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(map))
}

/// Reject identifiers and references that would resolve differently once a
/// subschema is lifted out of its document.
fn check_references(schema: &Value) -> Result<(), String> {
    match schema {
        Value::Object(map) => {
            for (key, value) in map {
                match key.as_str() {
                    "$ref" => {
                        let local = value
                            .as_str()
                            .and_then(|reference| reference.strip_prefix('#'))
                            .is_some_and(local_definition);
                        if !local {
                            return Err(format!(
                                "$ref {value} must point into the schema's own $defs or definitions"
                            ));
                        }
                    }
                    "$id" | "$anchor" | "$dynamicRef" | "$dynamicAnchor" | "$recursiveRef"
                    | "$recursiveAnchor" | "x-guidance" => {
                        return Err(format!("keyword {key:?} is not supported in strict mode"));
                    }
                    "enum" | "const" | "default" | "examples" => {}
                    // Maps from names to schemas: the names are data.
                    "properties" | "patternProperties" | "$defs" | "definitions"
                    | "dependentSchemas" => match value {
                        Value::Object(schemas) => {
                            schemas.values().try_for_each(check_references)?;
                        }
                        other => check_references(other)?,
                    },
                    _ => check_references(value)?,
                }
            }
            Ok(())
        }
        Value::Array(items) => items.iter().try_for_each(check_references),
        _ => Ok(()),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
