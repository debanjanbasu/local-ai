//! Reading generations back from batch results, and grading one row.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::definitions::{Check, Criterion, Params, RenderedMessage};
use super::graders::{similarity, string_check};
use super::template::{self, Namespaces};
use crate::batches::Endpoint;

/// An error as the Evals API reports it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalApiError {
    pub code: String,
    pub message: String,
}

impl EvalApiError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_owned(),
            message: message.into(),
        }
    }
}

/// Token usage of one generation (the fields are the spec's names).
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
}

impl Usage {
    pub const fn add(&mut self, other: Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        self.cached_tokens += other.cached_tokens;
    }

    fn json(self) -> Value {
        json!({
            "total_tokens": self.total_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
        })
    }
}

/// What one generation request produced: the model's answer, or why there
/// is none. `invoked` is whether the model answered at all (its usage then
/// counts).
#[derive(Clone, Debug, Default)]
struct Generated {
    text: String,
    tools: Vec<Value>,
    finish_reason: String,
    usage: Usage,
    invoked: bool,
    error: Option<EvalApiError>,
}

fn number(value: Option<&Value>) -> u64 {
    value.and_then(Value::as_u64).unwrap_or(0)
}

/// Decode a batch result record (`{custom_id, response, error}`) of `endpoint`.
fn generated(record: Option<&Value>, endpoint: Endpoint) -> Generated {
    let Some(record) = record else {
        return Generated {
            error: Some(EvalApiError::new(
                "not_generated",
                "the request was not run before its batch ended",
            )),
            ..Generated::default()
        };
    };
    if let Some(error) = record.get("error").filter(|error| !error.is_null()) {
        return Generated {
            error: Some(EvalApiError::new(
                error
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or("request_failed"),
                error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )),
            ..Generated::default()
        };
    }
    let status = number(record.pointer("/response/status_code"));
    let body = record.pointer("/response/body").unwrap_or(&Value::Null);
    if !(200..300).contains(&status) {
        let error = body.get("error");
        let code = error
            .and_then(|error| error.get("code").and_then(Value::as_str))
            .or_else(|| error.and_then(|error| error.get("type").and_then(Value::as_str)))
            .unwrap_or("request_failed");
        let message = error
            .and_then(|error| error.get("message").and_then(Value::as_str))
            .map_or_else(
                || format!("the request failed with status {status}"),
                str::to_owned,
            );
        return Generated {
            invoked: true,
            error: Some(EvalApiError::new(code, message)),
            ..Generated::default()
        };
    }
    match endpoint {
        Endpoint::Responses => responses_answer(body),
        Endpoint::ChatCompletions | Endpoint::RetiredCompletions => chat_answer(body),
    }
}

/// The answer a successful `/v1/responses` body holds.
fn responses_answer(body: &Value) -> Generated {
    let usage = body.get("usage");
    let field = |name: &str| number(usage.and_then(|usage| usage.get(name)));
    let output = body.get("output").and_then(Value::as_array);
    let mut text = String::new();
    let mut tools = Vec::new();
    for item in output.into_iter().flatten() {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for part in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if part.get("type").and_then(Value::as_str) == Some("output_text") {
                        text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
                    }
                }
            }
            Some("function_call") => tools.push(item.clone()),
            _ => {}
        }
    }
    let finish_reason = match body.get("status").and_then(Value::as_str) {
        Some("completed") => "stop".to_owned(),
        Some("incomplete") => match body
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => "length".to_owned(),
            Some(reason) => reason.to_owned(),
            None => "incomplete".to_owned(),
        },
        Some(other) => other.to_owned(),
        None => String::new(),
    };
    Generated {
        text,
        tools,
        finish_reason,
        usage: Usage {
            prompt_tokens: field("input_tokens"),
            completion_tokens: field("output_tokens"),
            total_tokens: field("total_tokens"),
            cached_tokens: number(
                usage.and_then(|usage| usage.pointer("/input_tokens_details/cached_tokens")),
            ),
        },
        invoked: true,
        error: None,
    }
}

/// The answer a successful `/v1/chat/completions` body holds.
fn chat_answer(body: &Value) -> Generated {
    let usage = body.get("usage");
    let field = |name: &str| number(usage.and_then(|usage| usage.get(name)));
    let choice = body.pointer("/choices/0");
    let message = choice.and_then(|choice| choice.get("message"));
    Generated {
        text: message
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        tools: message
            .and_then(|message| message.get("tool_calls"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        finish_reason: choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        usage: Usage {
            prompt_tokens: field("prompt_tokens"),
            completion_tokens: field("completion_tokens"),
            total_tokens: field("total_tokens"),
            cached_tokens: number(
                usage.and_then(|usage| usage.pointer("/prompt_tokens_details/cached_tokens")),
            ),
        },
        invoked: true,
        error: None,
    }
}

/// A row's sample, as stored with the row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredSample {
    /// The `sample` template namespace; `None` when the row has none.
    pub namespace: Option<Value>,
    /// The output item's `sample` object.
    pub record: Value,
    /// Why the row has no usable sample.
    pub error: Option<EvalApiError>,
    /// Whether a generation answered (its usage counts).
    pub invoked: bool,
    pub usage: Usage,
}

fn sample_record(
    input: &[RenderedMessage],
    output: &str,
    generated: &Generated,
    model: &str,
    params: Params,
) -> Value {
    let output = if generated.error.is_some() {
        json!([])
    } else {
        json!([{"role": "assistant", "content": output}])
    };
    json!({
        "input": input,
        "output": output,
        "finish_reason": generated.finish_reason,
        "model": model,
        "usage": generated.usage.json(),
        "error": generated.error,
        "temperature": params.temperature,
        "max_completion_tokens": params.max_tokens,
        "top_p": params.top_p,
        "seed": params.seed,
    })
}

/// The sample namespace of an answer: its text, the text parsed as JSON
/// when it is JSON, its tool calls, finish reason, model and usage.
fn namespace(generated: &Generated, model: &str) -> Value {
    let mut namespace = Map::new();
    namespace.insert("model".into(), Value::from(model));
    namespace.insert("output_text".into(), Value::from(generated.text.as_str()));
    if let Ok(parsed) = serde_json::from_str::<Value>(&generated.text) {
        namespace.insert("output_json".into(), parsed);
    }
    namespace.insert("output_tools".into(), Value::Array(generated.tools.clone()));
    namespace.insert(
        "finish_reason".into(),
        Value::from(generated.finish_reason.as_str()),
    );
    namespace.insert("usage".into(), generated.usage.json());
    Value::Object(namespace)
}

/// The sample a batch result record holds for one row.
pub fn generated_sample(
    record: Option<&Value>,
    endpoint: Endpoint,
    model: &str,
    input: &[RenderedMessage],
    params: Params,
) -> StoredSample {
    let generated = generated(record, endpoint);
    StoredSample {
        namespace: generated
            .error
            .is_none()
            .then(|| namespace(&generated, model)),
        record: sample_record(input, &generated.text, &generated, model, params),
        error: generated.error.clone(),
        invoked: generated.invoked,
        usage: generated.usage,
    }
}

/// A row whose sampling request could not even be built.
pub fn unbuilt_sample(message: String, model: &str, params: Params) -> StoredSample {
    let generated = Generated {
        error: Some(EvalApiError::new("invalid_input", message)),
        ..Generated::default()
    };
    StoredSample {
        namespace: None,
        record: sample_record(&[], "", &generated, model, params),
        error: generated.error,
        invoked: false,
        usage: Usage::default(),
    }
}

/// The sample a data source row brought along (or its absence). Fields the
/// output item requires that the row does not give are empty or zero.
pub fn provided_sample(sample: Option<&Map<String, Value>>) -> StoredSample {
    let get = |key: &str| sample.and_then(|sample| sample.get(key));
    let text = get("output_text").and_then(Value::as_str);
    let output = text.map_or_else(
        || json!([]),
        |text| json!([{"role": "assistant", "content": text}]),
    );
    let usage = get("usage");
    let field = |name: &str| number(usage.and_then(|usage| usage.get(name)));
    StoredSample {
        namespace: sample.map(|sample| Value::Object(sample.clone())),
        record: json!({
            "input": get("input").filter(|input| input.is_array()).cloned().unwrap_or_else(|| json!([])),
            "output": output,
            "finish_reason": get("finish_reason").and_then(Value::as_str).unwrap_or_default(),
            "model": get("model").and_then(Value::as_str).unwrap_or_default(),
            "usage": Usage {
                prompt_tokens: field("prompt_tokens"),
                completion_tokens: field("completion_tokens"),
                total_tokens: field("total_tokens"),
                cached_tokens: field("cached_tokens"),
            }.json(),
            "error": null,
            "temperature": get("temperature").and_then(Value::as_f64).unwrap_or(0.0),
            "max_completion_tokens": number(get("max_completion_tokens")),
            "top_p": get("top_p").and_then(Value::as_f64).unwrap_or(0.0),
            "seed": get("seed").and_then(Value::as_i64).unwrap_or(0),
        }),
        error: None,
        invoked: false,
        usage: Usage::default(),
    }
}

/// One row's grading.
pub struct Graded {
    /// The output item's `results`, one per criterion.
    pub results: Vec<Value>,
    /// Per criterion: whether it passed.
    pub passed: Vec<bool>,
    /// Whether the sample or any grader failed to produce a result.
    pub errored: bool,
    /// Usage of the model graders that answered, and how many did.
    pub usage: Usage,
    pub invocations: u64,
}

fn errored(criterion: &Criterion, sample: &Value, error: &EvalApiError) -> Value {
    json!({
        "name": criterion.name(),
        "type": criterion.kind(),
        "score": 0.0,
        "passed": false,
        "sample": sample,
        "error": error,
    })
}

/// Grade one row against every criterion. `records` holds, by criterion
/// index, the batch result record of each model grader request.
pub fn grade_row(
    criteria: &[Criterion],
    item: &Value,
    sample: &StoredSample,
    records: &HashMap<usize, Value>,
) -> Graded {
    let mut graded = Graded {
        results: Vec::with_capacity(criteria.len()),
        passed: Vec::with_capacity(criteria.len()),
        errored: sample.error.is_some(),
        usage: Usage::default(),
        invocations: 0,
    };
    let namespaces = Namespaces {
        item,
        sample: sample.namespace.as_ref(),
    };
    for (index, criterion) in criteria.iter().enumerate() {
        if let Some(error) = &sample.error {
            let error = EvalApiError::new(
                "sample_error",
                format!("the row has no sample to grade: {}", error.message),
            );
            graded
                .results
                .push(errored(criterion, &Value::Null, &error));
            graded.passed.push(false);
            continue;
        }
        let (result, passed) = if let Some(check) = criterion.deterministic() {
            grade_check(criterion, &check, namespaces)
        } else {
            let (result, passed, usage, invoked) =
                grade_model(criterion, namespaces, records.get(&index));
            graded.usage.add(usage);
            graded.invocations += u64::from(invoked);
            (result, passed)
        };
        if result.get("error").is_some() {
            graded.errored = true;
        }
        graded.results.push(result);
        graded.passed.push(passed);
    }
    graded
}

fn grade_check(
    criterion: &Criterion,
    check: &Check<'_>,
    namespaces: Namespaces<'_>,
) -> (Value, bool) {
    let render = |input: &str, reference: &str| -> Result<(String, String), EvalApiError> {
        let render = |text: &str| {
            template::render(text, namespaces)
                .map_err(|message| EvalApiError::new("invalid_template", message))
        };
        Ok((render(input)?, render(reference)?))
    };
    let (score, passed) = match check {
        Check::String {
            input,
            reference,
            operation,
        } => match render(input, reference) {
            Ok((input, reference)) => {
                let passed = string_check(*operation, &input, &reference);
                (if passed { 1.0 } else { 0.0 }, passed)
            }
            Err(error) => return (errored(criterion, &Value::Null, &error), false),
        },
        Check::Similarity {
            input,
            reference,
            metric,
            pass_threshold,
        } => match render(input, reference) {
            Ok((input, reference)) => {
                let score = similarity(*metric, &input, &reference);
                (score, score >= *pass_threshold)
            }
            Err(error) => return (errored(criterion, &Value::Null, &error), false),
        },
    };
    (
        json!({
            "name": criterion.name(),
            "type": criterion.kind(),
            "score": score,
            "passed": passed,
            "sample": null,
        }),
        passed,
    )
}

/// A model grader's result for one row: (result, passed, usage, answered).
fn grade_model(
    criterion: &Criterion,
    namespaces: Namespaces<'_>,
    record: Option<&Value>,
) -> (Value, bool, Usage, bool) {
    let request = match criterion.grader_request(namespaces) {
        Some(Ok((_, messages))) => messages,
        Some(Err(message)) => {
            let error = EvalApiError::new("invalid_template", message);
            return (
                errored(criterion, &Value::Null, &error),
                false,
                Usage::default(),
                false,
            );
        }
        None => return (Value::Null, false, Usage::default(), false),
    };
    let generated = generated(record, Endpoint::ChatCompletions);
    let model = criterion.model().unwrap_or_default();
    let sample = json!({
        "input": request,
        "output": if generated.error.is_some() { json!([]) } else {
            json!([{"role": "assistant", "content": generated.text}])
        },
        "finish_reason": generated.finish_reason,
        "model": model,
        "usage": generated.usage.json(),
        "error": generated.error,
    });
    let (usage, invoked) = (generated.usage, generated.invoked);
    if let Some(error) = &generated.error {
        return (errored(criterion, &sample, error), false, usage, invoked);
    }
    let answer = serde_json::from_str::<Value>(&generated.text)
        .ok()
        .and_then(|answer| answer.get("result").cloned());
    let outcome = match (criterion, answer) {
        (
            Criterion::LabelModel {
                labels,
                passing_labels,
                ..
            },
            Some(Value::String(label)),
        ) if labels.contains(&label) => {
            let passed = passing_labels.contains(&label);
            Ok((if passed { 1.0 } else { 0.0 }, passed))
        }
        (Criterion::ScoreModel { pass_threshold, .. }, Some(Value::Number(score))) => {
            let (low, high) = criterion.range();
            match score.as_f64() {
                Some(score) if (low..=high).contains(&score) => {
                    Ok((score, score >= *pass_threshold))
                }
                _ => Err(format!(
                    "the grader's score {score} is outside its range [{low}, {high}]"
                )),
            }
        }
        _ => Err(format!(
            "the grader's answer is not a valid result (finish reason {:?})",
            generated.finish_reason
        )),
    };
    match outcome {
        Ok((score, passed)) => (
            json!({
                "name": criterion.name(),
                "type": criterion.kind(),
                "score": score,
                "passed": passed,
                "sample": sample,
            }),
            passed,
            usage,
            invoked,
        ),
        Err(message) => (
            errored(
                criterion,
                &sample,
                &EvalApiError::new("invalid_grader_output", message),
            ),
            false,
            usage,
            invoked,
        ),
    }
}
