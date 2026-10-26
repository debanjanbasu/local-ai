//! The Response state machine: engine events into output items and the
//! streaming events and document a client receives.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Map, Value, json};

use local_engine::bonsai_model::StopReason;
use local_engine::compaction::{SummaryDetail, reasoning_summary_request};
use local_engine::{Event, Stats, ToolCall, ToolCallDelta};
use local_services::compaction::COMPACTION;

use super::super::background::Temporary;
use super::super::hosted::{Hosted, Kind};
use super::super::response::{Reply, arguments_text, logprob_json, unix_now};
use super::super::store::ResponseStore;
use super::parse::{Echo, Spent, Summarizer};

/// Delivers streaming events while a Response is still inside one engine
/// event (the summary generation runs after the main one's last event), and
/// answers `false` once the client is gone.
pub(in crate::cli::serve) type Flush = Box<dyn FnMut(Vec<Value>) -> bool + Send>;

/// A borrowed [`Flush`], for one call (see [`ResponsesState::event_with`]).
pub(in crate::cli::serve) type Sink<'a> = dyn FnMut(Vec<Value>) -> bool + 'a;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    Reasoning,
    Message,
}

/// An output item a journal opened and a lost generation never finished,
/// with what the journal says of it (see [`ResponsesState::resume_from`]).
#[derive(Default)]
struct Unfinished {
    item: Value,
    text: String,
    part_added: bool,
    text_done: bool,
    part_done: bool,
    /// A reasoning item's summary part, once the journal opened it.
    summary: Option<SummaryPart>,
    /// A message's citations the journal announced.
    annotations: Vec<Value>,
}

/// The reasoning summary part a lost generation was streaming.
#[derive(Default)]
struct SummaryPart {
    text: String,
    text_done: bool,
    part_done: bool,
}

/// The output item text is currently being written into.
struct Open {
    kind: ItemKind,
    id: String,
    index: usize,
    text: String,
    /// Unicode scalar values in `text`, where the next citation points.
    chars: usize,
    /// A message's `file_citation` annotations so far.
    annotations: Vec<Value>,
    /// A message's token log-probabilities so far, when the request asked.
    logprobs: Vec<Value>,
}

/// A `function_call` or `custom_tool_call` item whose arguments (or input)
/// are streaming from [`ToolCallDelta`]s.
struct OpenCall {
    call_id: String,
    id: String,
    index: usize,
    custom: bool,
    /// The item as `output_item.added` announced it.
    item: Value,
    /// The arguments streamed so far, or a custom tool's input.
    streamed: String,
}

/// One Response being built from engine events.
///
/// The same state machine serves the streaming and the buffered body, so the
/// document a non-streaming client receives is exactly the `response` of the
/// terminal event a streaming client would have. Events are only materialised
/// when streaming.
pub(in crate::cli::serve) struct ResponsesState {
    id: String,
    /// Hex shared by every item ID of this response.
    stem: String,
    created: u64,
    model: Arc<str>,
    echo: Arc<Echo>,
    streaming: bool,
    sequence: u64,
    events: Vec<Value>,
    output: Vec<Value>,
    open: Option<Open>,
    /// The call whose arguments are streaming, until it is complete.
    calling: Option<OpenCall>,
    encryption_error: Option<String>,
    /// The output index of the reasoning item whose `output_item.done` waits
    /// for its summary.
    deferred: Option<usize>,
    /// Where summary events go as they are generated, when streaming.
    flush: Option<Flush>,
    /// The extra generations run for this Response so far (see [`Spent`]).
    spent: Spent,
    /// `file_search_call` items searching, by call ID: their output index.
    searching: HashMap<String, usize>,
    /// Citation markers of this Response's file search results.
    citations: super::super::file_search::Citations,
    /// Log-probabilities of the tokens behind the next message text.
    logprobs: Vec<Value>,
}

impl ResponsesState {
    pub(in crate::cli::serve) fn new(reply: &Reply, echo: Arc<Echo>, streaming: bool) -> Self {
        let stem = reply
            .id
            .strip_prefix("resp_")
            .unwrap_or(&reply.id)
            .to_owned();
        let output = echo.compaction.iter().cloned().collect();
        let spent = echo.spent;
        Self {
            id: reply.id.clone(),
            stem,
            created: reply.created,
            model: Arc::clone(&reply.model),
            echo,
            streaming,
            sequence: 0,
            events: Vec::new(),
            output,
            open: None,
            calling: None,
            encryption_error: None,
            deferred: None,
            flush: None,
            spent,
            searching: HashMap::new(),
            citations: super::super::file_search::Citations::default(),
            logprobs: Vec::new(),
        }
    }

    /// Deliver the reasoning summary's events through `flush` as they are
    /// generated, instead of all at once when the summary ends.
    pub(in crate::cli::serve) fn set_flush(&mut self, flush: Flush) {
        self.flush = Some(flush);
    }

    /// Whether this Response materialises streaming events.
    pub(in crate::cli::serve) const fn streaming(&self) -> bool {
        self.streaming
    }

    /// Queue one streaming event, numbering it.
    fn emit(&mut self, kind: &str, fields: Value) {
        if !self.streaming {
            return;
        }
        let mut event = Map::new();
        event.insert("type".into(), json!(kind));
        if let Value::Object(fields) = fields {
            event.extend(fields);
        }
        event.insert("sequence_number".into(), json!(self.sequence));
        self.sequence += 1;
        self.events.push(Value::Object(event));
    }

    /// The events produced since the last call, in order.
    pub(in crate::cli::serve) fn take_events(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.events)
    }

    /// A background stream's opening: `response.created` and
    /// `response.queued`, both carrying the `queued` Response, since nothing
    /// has run yet.
    pub(in crate::cli::serve) fn start_queued(&mut self) {
        let response = self.snapshot("queued");
        self.emit("response.created", json!({"response":response}));
        self.emit("response.queued", json!({"response":response}));
    }

    /// `response.in_progress`, once the engine has started the job.
    pub(in crate::cli::serve) fn started(&mut self) {
        let response = self.snapshot("in_progress");
        self.emit("response.in_progress", json!({"response":response}));
        self.announce_compaction();
    }

    /// The compaction item, which ran before generation and so is complete,
    /// as output item 0.
    fn announce_compaction(&mut self) {
        if let Some(item) = self.echo.compaction.clone() {
            self.emit(
                "response.output_item.added",
                json!({"output_index":0,"item":item}),
            );
            self.emit(
                "response.output_item.done",
                json!({"output_index":0,"item":item}),
            );
        }
    }

    /// Take over the stream `journaled` began before a restart lost its
    /// generation: the restart boundary of a resumed background Response.
    ///
    /// Items the journal finished are kept as they were. Each one it left
    /// open is finished here with exactly the events it lacks: `incomplete`,
    /// with the text it had, or `completed` when its text had already been
    /// finished. The regenerated output then continues after them, at the
    /// next output index, so no index or item ID is reused and every event a
    /// client has seen stays true. Returns whether the journal had announced
    /// `response.in_progress`.
    pub(in crate::cli::serve) fn resume_from(&mut self, journaled: &[Value]) -> bool {
        let mut started = false;
        let mut open = std::collections::BTreeMap::<usize, Unfinished>::new();
        for event in journaled {
            let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
            if kind == "response.in_progress" {
                started = true;
            }
            let Some(index) = event
                .get("output_index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
            else {
                continue;
            };
            if kind == "response.output_item.added" {
                if self.output.len() <= index {
                    self.output.resize(index + 1, Value::Null);
                }
                let Some(item) = event.get("item").filter(|item| item.is_object()).cloned() else {
                    continue;
                };
                open.insert(
                    index,
                    Unfinished {
                        item,
                        ..Unfinished::default()
                    },
                );
                continue;
            }
            if kind == "response.output_item.done" {
                if let Some(slot) = self.output.get_mut(index) {
                    *slot = event.get("item").cloned().unwrap_or(Value::Null);
                }
                open.remove(&index);
                continue;
            }
            let Some(unfinished) = open.get_mut(&index) else {
                continue;
            };
            match kind {
                "response.content_part.added" => unfinished.part_added = true,
                "response.reasoning_text.delta"
                | "response.output_text.delta"
                | "response.function_call_arguments.delta"
                | "response.custom_tool_call_input.delta" => {
                    if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                        unfinished.text.push_str(delta);
                    }
                }
                "response.reasoning_text.done"
                | "response.output_text.done"
                | "response.function_call_arguments.done"
                | "response.custom_tool_call_input.done" => unfinished.text_done = true,
                "response.content_part.done" => unfinished.part_done = true,
                "response.output_text.annotation.added" => {
                    if let Some(annotation) = event.get("annotation") {
                        unfinished.annotations.push(annotation.clone());
                    }
                }
                "response.reasoning_summary_part.added" => {
                    unfinished.summary = Some(SummaryPart::default());
                }
                "response.reasoning_summary_text.delta" => {
                    if let (Some(summary), Some(delta)) = (
                        unfinished.summary.as_mut(),
                        event.get("delta").and_then(Value::as_str),
                    ) {
                        summary.text.push_str(delta);
                    }
                }
                "response.reasoning_summary_text.done" => {
                    if let Some(summary) = unfinished.summary.as_mut() {
                        summary.text_done = true;
                    }
                }
                "response.reasoning_summary_part.done" => {
                    if let Some(summary) = unfinished.summary.as_mut() {
                        summary.part_done = true;
                    }
                }
                _ => {}
            }
        }
        // `response.in_progress` and the compaction item are one batch, but
        // a crash may have cut it short: an item the record holds is never
        // left unannounced.
        let announced = journaled.iter().any(|event| {
            event.get("type").and_then(Value::as_str) == Some("response.output_item.added")
                && event.get("output_index").and_then(Value::as_u64) == Some(0)
        });
        if started && !announced {
            self.announce_compaction();
        }
        for (index, unfinished) in open {
            self.finish_interrupted(index, unfinished);
        }
        started
    }

    /// Finish the item at `index` that a lost generation left `unfinished`,
    /// emitting only the events its journal lacks.
    fn finish_interrupted(&mut self, index: usize, unfinished: Unfinished) {
        let mut item = unfinished.item;
        if item.get("type").and_then(Value::as_str) == Some(COMPACTION) {
            // Made whole before generation; only its `done` was lost.
            self.emit(
                "response.output_item.done",
                json!({"output_index":index,"item":item}),
            );
            if let Some(slot) = self.output.get_mut(index) {
                *slot = item;
            }
            return;
        }
        let id = item.get("id").cloned().unwrap_or(Value::Null);
        let reasoning = item.get("type").and_then(Value::as_str) == Some("reasoning");
        // A reasoning item waiting for its summary is complete only once
        // the summary is.
        let summarized = !reasoning
            || self.echo.summary.is_none()
            || unfinished
                .summary
                .as_ref()
                .is_some_and(|summary| summary.part_done);
        let status = if unfinished.text_done && summarized {
            "completed"
        } else {
            "incomplete"
        };
        let (text, marks) = (unfinished.text, unfinished.annotations);
        let (part, done) = match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => (
                Some(json!({"type":"reasoning_text","text":text})),
                "response.reasoning_text.done",
            ),
            Some("message") => (
                Some(json!({"type":"output_text","text":text,"annotations":marks,"logprobs":[]})),
                "response.output_text.done",
            ),
            Some("function_call") => (None, "response.function_call_arguments.done"),
            Some("custom_tool_call") => (None, "response.custom_tool_call_input.done"),
            _ => (None, ""),
        };
        if let Some(part) = &part {
            if !unfinished.part_added {
                let mut empty = part.clone();
                empty["text"] = json!("");
                self.emit(
                    "response.content_part.added",
                    json!({"item_id":id,"output_index":index,"content_index":0,"part":empty}),
                );
            }
            if !unfinished.text_done {
                let mut fields =
                    json!({"item_id":id,"output_index":index,"content_index":0,"text":text});
                if done == "response.output_text.done" {
                    fields["logprobs"] = json!([]);
                }
                self.emit(done, fields);
            }
            if !unfinished.part_done {
                self.emit(
                    "response.content_part.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
                );
            }
            item["content"] = json!([part]);
        } else if done == "response.function_call_arguments.done" {
            if !unfinished.text_done {
                self.emit(
                    done,
                    json!({"item_id":id,"output_index":index,"arguments":text}),
                );
            }
            item["arguments"] = json!(text);
        } else if done == "response.custom_tool_call_input.done" {
            if !unfinished.text_done {
                self.emit(
                    done,
                    json!({"item_id":id,"output_index":index,"input":text}),
                );
            }
            item["input"] = json!(text);
        }
        if let Some(summary) = unfinished.summary.filter(|_| reasoning) {
            item["summary"] = json!([self.close_summary(index, &id, &summary)]);
        }
        item["status"] = json!(status);
        if part.is_some()
            && item.get("type").and_then(Value::as_str) == Some("reasoning")
            && let (Some(cipher), Some(id)) = (&self.echo.cipher, id.as_str())
        {
            match cipher.seal(&self.model, id, &text) {
                Ok(envelope) => item["encrypted_content"] = json!(envelope),
                Err(error) => self.encryption_error = Some(error.to_string()),
            }
        }
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        );
        if let Some(slot) = self.output.get_mut(index) {
            *slot = item;
        }
    }

    /// Close the summary part of the reasoning item `id` at `index` that a
    /// lost generation was streaming, emitting only the events its journal
    /// lacks, and return the part, with the text it had.
    fn close_summary(&mut self, index: usize, id: &Value, summary: &SummaryPart) -> Value {
        let part = json!({"type":"summary_text","text":summary.text});
        if !summary.text_done {
            self.emit(
                "response.reasoning_summary_text.done",
                json!({"item_id":id,"output_index":index,"summary_index":0,"text":summary.text}),
            );
        }
        if !summary.part_done {
            self.emit(
                "response.reasoning_summary_part.done",
                json!({"item_id":id,"output_index":index,"summary_index":0,"part":part}),
            );
        }
        part
    }

    /// Withdraw the events announcing the end that was just folded in —
    /// `response.completed`, `response.incomplete`, or `response.failed` and
    /// the `error` before it — leaving the output events before them.
    ///
    /// A background stream journals its end from the stored record instead,
    /// once that record is durable (see [`super::super::journal::terminal_events`]).
    pub(in crate::cli::serve) fn retract_end(&mut self) {
        let kind = |event: Option<&Value>| {
            event
                .and_then(|event| event.get("type"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        };
        let last = kind(self.events.last());
        if matches!(
            last.as_deref(),
            Some("response.completed" | "response.incomplete" | "response.failed")
        ) {
            self.events.pop();
            if last.as_deref() == Some("response.failed")
                && kind(self.events.last()).as_deref() == Some("error")
            {
                self.events.pop();
            }
        }
    }

    /// `response.created` and `response.in_progress`.
    pub(in crate::cli::serve) fn start(&mut self) {
        let response = self.response("in_progress", None, None, None);
        self.emit("response.created", json!({"response":response}));
        let response = self.response("in_progress", None, None, None);
        self.emit("response.in_progress", json!({"response":response}));
        self.announce_compaction();
    }

    fn item_id(&self, prefix: &str) -> String {
        format!("{prefix}_{}{:02}", self.stem, self.output.len())
    }

    /// Append text to the item of `kind`, opening it (and closing any other).
    fn text(&mut self, kind: ItemKind, piece: &str) {
        if self.open.as_ref().is_none_or(|open| open.kind != kind) {
            self.close("completed");
            let index = self.output.len();
            let (id, item, part) = match kind {
                ItemKind::Reasoning => {
                    let id = self.item_id("rs");
                    (
                        id.clone(),
                        json!({"id":id,"type":"reasoning","summary":[],"content":[],"status":"in_progress"}),
                        json!({"type":"reasoning_text","text":""}),
                    )
                }
                ItemKind::Message => {
                    let id = self.item_id("msg");
                    (
                        id.clone(),
                        json!({"id":id,"type":"message","status":"in_progress","role":"assistant","content":[]}),
                        json!({"type":"output_text","text":"","annotations":[],"logprobs":[]}),
                    )
                }
            };
            self.emit(
                "response.output_item.added",
                json!({"output_index":index,"item":item}),
            );
            self.emit(
                "response.content_part.added",
                json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
            );
            // The slot is reserved now so the next item's index follows it.
            self.output.push(Value::Null);
            self.open = Some(Open {
                kind,
                id,
                index,
                text: String::new(),
                chars: 0,
                annotations: Vec::new(),
                logprobs: Vec::new(),
            });
        }
        if kind == ItemKind::Message && self.citations.active() {
            for piece in self.citations.filter(piece) {
                match piece {
                    super::super::file_search::Piece::Text(text) => self.append(kind, &text),
                    super::super::file_search::Piece::Cite(cited) => self.cite(cited),
                }
            }
            return;
        }
        self.append(kind, piece);
    }

    /// Append `piece` to the open text item of `kind`.
    fn append(&mut self, kind: ItemKind, piece: &str) {
        let Some(open) = self.open.as_mut() else {
            return;
        };
        if piece.is_empty() {
            return;
        }
        open.text.push_str(piece);
        open.chars += piece.chars().count();
        let (id, index) = (open.id.clone(), open.index);
        match kind {
            ItemKind::Reasoning => self.emit(
                "response.reasoning_text.delta",
                json!({"item_id":id,"output_index":index,"content_index":0,"delta":piece}),
            ),
            ItemKind::Message => {
                let tokens = std::mem::take(&mut self.logprobs);
                open.logprobs.extend(tokens.iter().cloned());
                self.emit(
                    "response.output_text.delta",
                    json!({"item_id":id,"output_index":index,"content_index":0,"delta":piece,"logprobs":tokens}),
                );
            }
        }
    }

    /// Annotate the open message, at the end of its text so far, with
    /// citations of `cited` results (each file once per position).
    fn cite(&mut self, cited: Vec<(String, String)>) {
        let Some(open) = self.open.as_mut() else {
            return;
        };
        let mut added = Vec::new();
        for (file_id, filename) in cited {
            let annotation = json!({"type":"file_citation","file_id":file_id,"filename":filename,"index":open.chars});
            if open.annotations.contains(&annotation) {
                continue;
            }
            added.push((open.annotations.len(), annotation.clone()));
            open.annotations.push(annotation);
        }
        let (id, index) = (open.id.clone(), open.index);
        for (annotation_index, annotation) in added {
            self.emit(
                "response.output_text.annotation.added",
                json!({"item_id":id,"output_index":index,"content_index":0,"annotation_index":annotation_index,"annotation":annotation}),
            );
        }
    }

    /// Finish the open text item, if any, with `status`.
    fn close(&mut self, status: &str) {
        // A call the engine never completed (cut short) keeps what streamed.
        if let Some(call) = self.calling.take() {
            self.finish_streamed(call, None, "incomplete");
        }
        if self
            .open
            .as_ref()
            .is_some_and(|open| open.kind == ItemKind::Message)
        {
            // Whitespace held back in case a citation marker followed.
            let held = self.citations.flush();
            self.append(ItemKind::Message, &held);
        }
        let Some(open) = self.open.take() else {
            return;
        };
        let Open {
            kind,
            id,
            index,
            text,
            annotations,
            logprobs,
            ..
        } = open;
        let item = match kind {
            ItemKind::Reasoning => {
                let part = json!({"type":"reasoning_text","text":text});
                self.emit(
                    "response.reasoning_text.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"text":text}),
                );
                self.emit(
                    "response.content_part.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
                );
                let mut item = json!({"id":id,"type":"reasoning","summary":[],"content":[part],"status":status});
                if let Some(cipher) = &self.echo.cipher {
                    match cipher.seal(&self.model, &id, &text) {
                        Ok(envelope) => item["encrypted_content"] = json!(envelope),
                        Err(error) => self.encryption_error = Some(error.to_string()),
                    }
                }
                if self.echo.summary.is_some() {
                    // Done only once its summary is, after generation ends.
                    if let Some(slot) = self.output.get_mut(index) {
                        *slot = item;
                    }
                    self.deferred = Some(index);
                    return;
                }
                item
            }
            ItemKind::Message => {
                let part = json!({"type":"output_text","text":text,"annotations":annotations,"logprobs":logprobs});
                self.emit(
                    "response.output_text.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"text":text,"logprobs":logprobs}),
                );
                self.emit(
                    "response.content_part.done",
                    json!({"item_id":id,"output_index":index,"content_index":0,"part":part}),
                );
                json!({"id":id,"type":"message","status":status,"role":"assistant","content":[part]})
            }
        };
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        );
        if let Some(slot) = self.output.get_mut(index) {
            *slot = item;
        }
    }

    /// Summarize the deferred reasoning item's raw reasoning with a second
    /// generation, streaming it as its `summary`, then finish the item.
    ///
    /// Runs on the pump's blocking thread. The raw reasoning stays the
    /// item's `reasoning_text` content; only the generated text is summary.
    fn summarize(&mut self, mut sink: Option<&mut Sink<'_>>) -> Result<(), String> {
        let (Some(index), Some(detail)) = (self.deferred, self.echo.summary) else {
            return Ok(());
        };
        let Some(item) = self.output.get(index) else {
            return Ok(());
        };
        let id = item["id"].as_str().unwrap_or_default().to_owned();
        let reasoning = item["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let Some(Summarizer(generate)) = self.echo.summarizer.clone() else {
            self.release(Vec::new());
            return Err("reasoning.summary has no summarizer on this server".into());
        };
        let added = json!({"item_id":id,"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":""}});
        // The part opens with the first piece, so a summary that fails
        // before producing anything leaves no part to contradict.
        let mut opened = false;
        let mut text = String::new();
        let generated = generate(
            reasoning_summary_request(&reasoning, detail),
            &mut |piece: &str| {
                if !opened {
                    opened = true;
                    self.emit("response.reasoning_summary_part.added", added.clone());
                }
                text.push_str(piece);
                self.emit(
                    "response.reasoning_summary_text.delta",
                    json!({"item_id":id,"output_index":index,"summary_index":0,"delta":piece}),
                );
                let events = std::mem::take(&mut self.events);
                if let Some(sink) = sink.as_mut() {
                    sink(events)
                } else if let Some(flush) = self.flush.as_mut() {
                    flush(events)
                } else {
                    self.events = events;
                    true
                }
            },
        );
        let failed = match generated {
            Ok(stats) => {
                self.spent.add(&stats);
                None
            }
            Err(error) => Some(format!(
                "the reasoning summary could not be generated: {error}"
            )),
        };
        if failed.is_some() && !opened {
            self.release(Vec::new());
            return failed.map_or(Ok(()), Err);
        }
        if !opened {
            self.emit("response.reasoning_summary_part.added", added);
        }
        // A summary cut short keeps, and closes, the part it streamed; its
        // item is then incomplete.
        self.emit(
            "response.reasoning_summary_text.done",
            json!({"item_id":id,"output_index":index,"summary_index":0,"text":text}),
        );
        let part = json!({"type":"summary_text","text":text});
        self.emit(
            "response.reasoning_summary_part.done",
            json!({"item_id":id,"output_index":index,"summary_index":0,"part":part}),
        );
        if failed.is_some()
            && let Some(slot) = self.output.get_mut(index)
        {
            slot["status"] = json!("incomplete");
        }
        self.release(vec![part]);
        failed.map_or(Ok(()), Err)
    }

    /// Finish the deferred reasoning item, if any, with `summary`.
    fn release(&mut self, summary: Vec<Value>) {
        let Some(index) = self.deferred.take() else {
            return;
        };
        let Some(slot) = self.output.get_mut(index) else {
            return;
        };
        slot["summary"] = Value::Array(summary);
        let item = slot.clone();
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        );
    }

    /// The next part of a call's arguments while the engine generates it.
    ///
    /// A `function_call` (including hosted file and fetch calls) streams its
    /// arguments and a `custom_tool_call` its raw input. Shell, local shell,
    /// `apply_patch` and file search calls are rendered whole when they
    /// complete.
    fn call_delta(&mut self, delta: &ToolCallDelta) {
        let kind = self.echo.kinds.kind(&delta.name);
        let custom = kind == Kind::Custom;
        if matches!(
            kind,
            Kind::Hosted(Hosted::Shell | Hosted::FileSearch) | Kind::LocalShell | Kind::ApplyPatch
        ) {
            return;
        }
        if self
            .calling
            .as_ref()
            .is_none_or(|open| open.call_id != delta.id)
        {
            self.close("completed");
            let index = self.output.len();
            let id = self.item_id(super::super::hosted::call_prefix(kind));
            let mut item = if custom {
                json!({"id":id,"type":"custom_tool_call","call_id":delta.id,"name":delta.name,"input":""})
            } else {
                json!({"id":id,"type":"function_call","status":"in_progress","call_id":delta.id,"name":delta.name,"arguments":""})
            };
            self.echo.kinds.qualify(&mut item);
            self.emit(
                "response.output_item.added",
                json!({"output_index":index,"item":item}),
            );
            // The slot is reserved now so the next item's index follows it.
            self.output.push(Value::Null);
            self.calling = Some(OpenCall {
                call_id: delta.id.clone(),
                id,
                index,
                custom,
                item,
                streamed: String::new(),
            });
        }
        let Some(open) = self.calling.as_mut() else {
            return;
        };
        let piece = if open.custom {
            match &delta.text {
                Some((parameter, text)) if parameter == "input" => text.clone(),
                _ => return,
            }
        } else {
            delta.arguments.clone()
        };
        if piece.is_empty() {
            return;
        }
        open.streamed.push_str(&piece);
        let (id, index, custom) = (open.id.clone(), open.index, open.custom);
        let kind = if custom {
            "response.custom_tool_call_input.delta"
        } else {
            "response.function_call_arguments.delta"
        };
        self.emit(
            kind,
            json!({"item_id":id,"output_index":index,"delta":piece}),
        );
    }

    /// Finish the streaming call: emit whatever its validated `call` adds to
    /// what streamed, then `done` with `status`.
    fn finish_streamed(&mut self, open: OpenCall, call: Option<&ToolCall>, status: &str) {
        let OpenCall {
            id,
            index,
            custom,
            mut item,
            streamed,
            ..
        } = open;
        let whole = match call {
            Some(call) if custom => call
                .arguments
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            Some(call) => arguments_text(call),
            None => streamed.clone(),
        };
        debug_assert!(whole.starts_with(&streamed), "streamed past the call");
        let rest = whole.get(streamed.len()..).unwrap_or_default();
        let (delta, done, field) = if custom {
            (
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "input",
            )
        } else {
            (
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "arguments",
            )
        };
        if !rest.is_empty() {
            self.emit(
                delta,
                json!({"item_id":id,"output_index":index,"delta":rest}),
            );
        }
        self.emit(done, json!({"item_id":id,"output_index":index,field:whole}));
        item[field] = json!(whole);
        if !custom {
            item["status"] = json!(status);
        }
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        );
        if let Some(slot) = self.output.get_mut(index) {
            *slot = item;
        }
    }

    /// A complete, validated call from the engine: the end of the call its
    /// deltas streamed, or, for a call the engine reported only whole, one
    /// arguments delta followed at once by `done`.
    fn call(&mut self, call: &ToolCall) {
        if let Some(open) = self.calling.take_if(|open| open.call_id == call.id) {
            self.finish_streamed(open, Some(call), "completed");
            return;
        }
        self.close("completed");
        let index = self.output.len();
        let kind = self.echo.kinds.kind(&call.name);
        let id = self.item_id(super::super::hosted::call_prefix(kind));
        if kind == Kind::Hosted(Hosted::FileSearch) {
            // Done once the search has run (see `tool_output`).
            self.searching.insert(call.id.clone(), index);
            self.push_rendered(super::super::file_search::started(&id, index, call));
            return;
        }
        if let Some(mut rendered) = super::super::hosted::call_item(kind, &id, index, call) {
            self.echo.kinds.qualify_rendered(&mut rendered);
            self.push_rendered(rendered);
            return;
        }
        let arguments = arguments_text(call);
        let mut added = json!({"id":id,"type":"function_call","status":"in_progress","call_id":call.id,"name":call.name,"arguments":""});
        self.echo.kinds.qualify(&mut added);
        self.emit(
            "response.output_item.added",
            json!({"output_index":index,"item":added}),
        );
        self.emit(
            "response.function_call_arguments.delta",
            json!({"item_id":id,"output_index":index,"delta":arguments}),
        );
        self.emit(
            "response.function_call_arguments.done",
            json!({"item_id":id,"output_index":index,"arguments":arguments}),
        );
        let mut item = json!({"id":id,"type":"function_call","status":"completed","call_id":call.id,"name":call.name,"arguments":arguments});
        self.echo.kinds.qualify(&mut item);
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        );
        self.output.push(item);
    }

    /// A hosted call's result, produced by this server between generations.
    fn tool_output(&mut self, output: &local_engine::ToolOutput) {
        self.close("completed");
        if let Some(index) = self.searching.remove(&output.call_id) {
            let include = self.echo.file_search_results;
            let Some(item) = self.output.get_mut(index) else {
                return;
            };
            let events = super::super::file_search::finished(item, index, output, include);
            for (kind, fields) in events {
                self.emit(kind, fields);
            }
            self.citations.add(output);
            return;
        }
        let index = self.output.len();
        let kind = self.echo.kinds.kind(&output.name);
        let id = self.item_id(super::super::hosted::output_prefix(kind));
        let rendered = super::super::hosted::output_item(kind, &id, index, output);
        self.push_rendered(rendered);
    }

    /// End the file searches still running (the Response is ending without
    /// their results) as `incomplete`.
    fn abandon_searches(&mut self) {
        let mut searching: Vec<usize> = self.searching.drain().map(|(_, index)| index).collect();
        searching.sort_unstable();
        for index in searching {
            let Some(item) = self.output.get_mut(index) else {
                continue;
            };
            item["status"] = json!("incomplete");
            let item = item.clone();
            self.emit(
                "response.output_item.done",
                json!({"output_index":index,"item":item}),
            );
        }
    }

    fn push_rendered(&mut self, rendered: super::super::hosted::Rendered) {
        for (kind, fields) in rendered.events {
            self.emit(kind, fields);
        }
        self.output.push(rendered.item);
    }

    /// Fold one engine event in. Returns the final Response on a terminal
    /// event: `Finished`, or `Error` once generation had started.
    pub(in crate::cli::serve) fn event(&mut self, event: Event) -> Option<Value> {
        self.event_with(event, None)
    }

    /// [`Self::event`], handing the reasoning summary's events to `sink` as
    /// they are generated instead of to the [`Flush`] the state was given.
    /// A background job journals them through it; `sink` answering `false`
    /// stops the summary, which then fails the Response.
    pub(in crate::cli::serve) fn event_with(
        &mut self,
        event: Event,
        sink: Option<&mut Sink<'_>>,
    ) -> Option<Value> {
        match event {
            Event::Reasoning(piece) => self.text(ItemKind::Reasoning, &piece),
            Event::Content(piece) => self.text(ItemKind::Message, &piece),
            Event::ToolCallDelta(delta) => self.call_delta(&delta),
            Event::ToolCall(call) => self.call(&call),
            Event::ToolOutput(output) => self.tool_output(&output),
            Event::Logprobs(tokens) => self.logprobs.extend(tokens.iter().map(logprob_json)),
            Event::TokenIds(_) => {}
            Event::Finished(stats) => return Some(self.finish(&stats, sink)),
            Event::Error(message) => return Some(self.fail(&message)),
        }
        None
    }

    fn finish(&mut self, stats: &Stats, sink: Option<&mut Sink<'_>>) -> Value {
        // An end of turn the format did not complete (during reasoning, so
        // there is no answer at all) is no completed structured answer, and
        // `incomplete_details` has no reason for it: the Response fails.
        if let Some(message) = super::super::response::unfinished_format(stats) {
            return self.fail(message);
        }
        let (kind, overall, item_status, incomplete, error) = match stats.stop_reason {
            // `Stopped` (a stop sequence) needs the `stop` Responses lacks.
            StopReason::Eos | StopReason::Stopped => {
                ("response.completed", "completed", "completed", None, None)
            }
            StopReason::TokenLimit => (
                "response.incomplete",
                "incomplete",
                "incomplete",
                Some(json!({"reason":"max_output_tokens"})),
                None,
            ),
            StopReason::Cancelled => (
                "response.failed",
                "failed",
                "incomplete",
                None,
                Some(json!({"code":"server_error","message":"generation was cancelled"})),
            ),
        };
        self.close(item_status);
        self.abandon_searches();
        if let Some(message) = self.encryption_error.take() {
            return self.fail(&message);
        }
        if stats.stop_reason == StopReason::Cancelled {
            self.release(Vec::new());
        } else if let Err(message) = self.summarize(sink) {
            return self.fail(&message);
        }
        let response = self.response(overall, incomplete.as_ref(), error.as_ref(), Some(stats));
        // A conversation's new items are appended, and a stored response is
        // on disk, before any client is told it finished, so what a client
        // has seen complete can always be continued. A response whose items
        // could not be kept fails rather than claim them. The conversation
        // goes first: its append is the step that can conflict, and when it
        // fails nothing at all has been written.
        // A background response is persisted by its pump, under the lock that
        // also serialises cancellation and deletion, never from here (see
        // [`Self::save`]).
        if error.is_none() && !self.echo.background {
            if let Some(conversing) = &self.echo.conversation
                && let Err(message) = conversing.append(&response)
            {
                eprintln!("response {}: {message}", self.id);
                return self.not_kept(&message, stats);
            }
            if let Some(persist) = &self.echo.persist
                && let Err(failure) = persist.store.save_with(
                    &response,
                    &persist.input_items,
                    None,
                    persist.project.as_deref(),
                )
            {
                eprintln!("response {} could not be stored: {failure}", self.id);
                let mut message = format!("the response could not be stored: {failure}");
                if let Some(conversing) = &self.echo.conversation {
                    message.push_str(&already_appended(&conversing.id));
                }
                return self.not_kept(&message, stats);
            }
        }
        self.emit(kind, json!({"response":response}));
        response
    }

    /// End a Response whose items could not be kept as `failed`, with `message`.
    fn not_kept(&mut self, message: &str, stats: &Stats) -> Value {
        self.emit(
            "error",
            json!({"code":"server_error","message":message,"param":null}),
        );
        let error = json!({"code":"server_error","message":message});
        let failed = self.response("failed", None, Some(&error), Some(stats));
        self.emit("response.failed", json!({"response":failed}));
        failed
    }

    /// The Response as it stands, before any terminal state: `queued` or
    /// `in_progress`. Usage is only known at the end, so it is `null`.
    pub(in crate::cli::serve) fn snapshot(&self, status: &str) -> Value {
        self.response(status, None, None, None)
    }

    /// End the Response as `cancelled`, keeping the output produced so far
    /// with any unfinished item marked `incomplete`.
    pub(in crate::cli::serve) fn cancelled(&mut self) -> Value {
        self.close("incomplete");
        self.release(Vec::new());
        self.response("cancelled", None, None, None)
    }

    /// Persist `response` with this request's resolved input, durably. Fails
    /// when the response is not to be stored at all.
    ///
    /// A completed or incomplete `response` continuing a conversation first
    /// appends its items there, exactly once: saving the same response again
    /// replays that append rather than repeating it. When the append fails
    /// nothing is saved and the error says why.
    pub(in crate::cli::serve) fn save(&self, response: &Value) -> std::io::Result<()> {
        let persist = self.echo.persist.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "response is not stored")
        })?;
        let status = response.get("status").and_then(Value::as_str);
        let ended = matches!(status, Some("completed" | "incomplete"));
        // A pending record keeps the request it can be regenerated from.
        let request = persist
            .request
            .as_ref()
            .filter(|_| matches!(status, Some("queued" | "in_progress")))
            .map(|request| (request, persist.creator.as_ref()));
        let Some(conversing) = self.echo.conversation.as_ref().filter(|_| ended) else {
            return persist.store.save_with(
                response,
                &persist.input_items,
                request,
                persist.project.as_deref(),
            );
        };
        conversing.append(response).map_err(std::io::Error::other)?;
        persist
            .store
            .save_with(
                response,
                &persist.input_items,
                None,
                persist.project.as_deref(),
            )
            .map_err(|error| {
                std::io::Error::new(
                    error.kind(),
                    format!("{error}{}", already_appended(&conversing.id)),
                )
            })
    }

    /// The store this Response is persisted in, when it is stored.
    pub(in crate::cli::serve) fn store(&self) -> Option<&Arc<ResponseStore>> {
        self.echo.persist.as_ref().map(|persist| &persist.store)
    }

    /// The temporary store that expires this Response, when it is kept only
    /// temporarily (`background` with `store: false`).
    pub(in crate::cli::serve) fn temporary(&self) -> Option<&Arc<Temporary>> {
        self.echo
            .persist
            .as_ref()
            .and_then(|persist| persist.temporary.as_ref())
    }

    pub(in crate::cli::serve) fn id(&self) -> &str {
        &self.id
    }

    pub(in crate::cli::serve) fn fail(&mut self, message: &str) -> Value {
        self.close("incomplete");
        self.abandon_searches();
        self.release(Vec::new());
        self.emit(
            "error",
            json!({"code":"server_error","message":message,"param":null}),
        );
        let error = json!({"code":"server_error","message":message});
        let response = self.response("failed", None, Some(&error), None);
        self.emit("response.failed", json!({"response":response}));
        response
    }

    /// The Response object in its current state.
    fn response(
        &self,
        status: &str,
        incomplete: Option<&Value>,
        error: Option<&Value>,
        measured: Option<&Stats>,
    ) -> Value {
        let echo = &self.echo;
        let usage = measured.map(|stats| {
            let generation = &stats.generation;
            let extra = self.spent;
            let input = generation.prompt_tokens + extra.prompt;
            let cached = generation.reused_prompt_tokens + extra.reused;
            let output = generation.generated_tokens + extra.generated;
            // No separately accounted cache-creation tokens: cache writes are
            // automatic local state, not OpenAI's charged cache-write tier.
            json!({"input_tokens":input,"input_tokens_details":{"cached_tokens":cached,"cache_write_tokens":0},"output_tokens":output,"output_tokens_details":{"reasoning_tokens":stats.reasoning_tokens},"total_tokens":input+output})
        });
        let completed_at = (status == "completed").then(unix_now);
        let output: Vec<&Value> = self.output.iter().filter(|item| !item.is_null()).collect();
        let mut response = json!({
            "id":self.id,
            "object":"response",
            "created_at":self.created,
            "status":status,
            "completed_at":completed_at,
            "error":error,
            "incomplete_details":incomplete,
            "instructions":echo.instructions,
            "max_output_tokens":echo.max_output_tokens,
            "model":self.model.as_ref(),
            "output":output,
            "parallel_tool_calls":echo.parallel_tool_calls,
            "previous_response_id":echo.previous_response_id,
            "reasoning":{"effort":echo.effort,"summary":echo.summary.map(SummaryDetail::name),"context":echo.context},
            // Only a response that reached the store says so. A failed
            // foreground one, including one whose write failed, was never
            // stored; a background one is stored in every state it reports.
            // A temporary background one is kept only to be retrieved, so it
            // says `false`, as the client asked.
            "store":echo.persist.as_ref().is_some_and(|persist| persist.temporary.is_none())
                && (echo.background || status != "failed"),
            "background":echo.background,
            "access_programs":null,
            "service_tier":"default",
            "temperature":echo.temperature,
            "text":{"format":echo.text_format,"verbosity":echo.verbosity},
            "tool_choice":echo.tool_choice,
            "tools":echo.tools,
            "top_p":echo.top_p,
            "truncation":"disabled",
            "usage":usage,
            "prompt_cache_key":echo.prompt_cache_key,
            "safety_identifier":echo.safety_identifier,
            "metadata":echo.metadata,
        });
        // Only a Response that continues a conversation names one, so every
        // other Response keeps exactly its existing shape.
        if let Some(conversing) = &echo.conversation {
            response["conversation"] = json!({"id":conversing.id});
        }
        response
    }
}

/// Said of a Response that failed to be stored after its items were appended
/// to `conversation`, which keeps them: the append is not undone.
fn already_appended(conversation: &str) -> String {
    format!("; its items were already added to conversation {conversation}")
}

/// One Responses SSE frame: the event type is repeated in the `event:` line,
/// as `OpenAI` sends it.
pub(in crate::cli::serve) fn sse_frame(event: &Value) -> String {
    let kind = event.get("type").and_then(Value::as_str).unwrap_or("error");
    format!("event: {kind}\ndata: {event}\n\n")
}
