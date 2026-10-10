//! Streaming a background Response: the stream of `POST /v1/responses` with
//! `background` and `stream`, and its resumption by
//! `GET /v1/responses/{id}?stream=true&starting_after=N`.
//!
//! Both read the response's event journal (see [`super::journal`]) and send
//! every event whose `sequence_number` is greater than the cursor, exactly as
//! first written. While this server is generating the response, a subscriber
//! follows the job's [`Published`] watch and parks between batches; otherwise
//! the journal is complete and is replayed to its end. Either way the stream
//! ends with the events of the stored terminal record, derived from it when
//! the journal does not hold them yet. It stops early when the response is
//! deleted, or when a failed journal write could not be cut back off, so the
//! live stream cannot know the next sequence number; a replay once the job is
//! gone then ends after the lines actually on disk.
//!
//! Everything that decides the status code happens before the stream starts:
//! a malformed cursor is 400, an unknown or deleted response 404, a response
//! that is not a streamed background one 400, and one still being generated
//! by another server sharing the store 409, because following it would mean
//! polling a file this process is not told about. A cursor at or past the end
//! of a finished stream is an empty 200 stream.
//!
//! A subscriber holds one open journal file, one line and one channel slot,
//! never a copy of the log. One that disconnects or stops reading for the
//! stall budget is dropped without touching the job, which runs on.

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use bytes::Bytes;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};

use super::background::{Background, Failure, Published, pending};
use super::backpressure::{FrameOutcome, park_until, send_frame};
use super::journal::{JournalReader, terminal_events};
use super::responses::sse_frame;
use super::store::{ItemPage, ResponseStore, query_pairs};

/// What a stored-response request asks for.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Retrieval {
    /// The Response object, or a deletion or cancellation.
    Document,
    /// A page of `input_items`.
    Items(ItemPage),
    /// The events after this cursor, from the first when `None`.
    Stream(Option<u64>),
}

impl Retrieval {
    pub(super) fn into_page(self) -> Option<ItemPage> {
        match self {
            Self::Items(page) => Some(page),
            Self::Document | Self::Stream(_) => None,
        }
    }
}

/// The parameters of `GET /v1/responses/{id}`: the Response object, or a
/// stream of its events after a cursor.
///
/// Unknown parameters are ignored, as client libraries add their own.
pub(super) fn retrieval(query: Option<&str>) -> Result<Retrieval, String> {
    let mut stream = false;
    let mut after = None;
    for (key, value) in query_pairs(query) {
        match key.as_str() {
            "stream" => {
                stream = match value.as_str() {
                    "true" => true,
                    "false" => false,
                    other => return Err(format!("stream must be true or false, not {other:?}")),
                };
            }
            "starting_after" => {
                let cursor = Some(value.as_str())
                    .filter(|value| !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()))
                    .and_then(|value| value.parse::<u64>().ok())
                    .ok_or_else(|| {
                        format!(
                            "starting_after must be a non-negative integer sequence number, \
                             not {value:?}"
                        )
                    })?;
                after = Some(cursor);
            }
            "include_obfuscation" => match value.as_str() {
                "false" => {}
                "true" => {
                    return Err(
                        "include_obfuscation is not supported: stream events carry no \
                                obfuscation field"
                            .into(),
                    );
                }
                other => {
                    return Err(format!(
                        "include_obfuscation must be true or false, not {other:?}"
                    ));
                }
            },
            "include" | "include[]" if !value.is_empty() => {
                return Err(format!(
                    "include {value:?} is not supported: there are no logprobs, encrypted \
                     reasoning or tool outputs to add"
                ));
            }
            _ => {}
        }
    }
    if !stream && after.is_some() {
        return Err("starting_after resumes an event stream, so it requires stream=true".into());
    }
    Ok(if stream {
        Retrieval::Stream(after)
    } else {
        Retrieval::Document
    })
}

/// One client's stream of a background Response's events.
pub(super) struct Subscription {
    store: Arc<ResponseStore>,
    id: String,
    reader: JournalReader,
    /// The job's watch while this server generates it; `None` replays.
    live: Option<watch::Receiver<Published>>,
    after: Option<u64>,
}

/// Subscribe to response `id` after `after`; `Ok(None)` is a 404.
///
/// A pending response no server is generating any longer is recovered first,
/// as opening a store would, so its stream ends instead of waiting forever.
/// Blocks on the store.
pub(super) fn subscribe(
    background: &Background,
    store: &Arc<ResponseStore>,
    id: &str,
    after: Option<u64>,
) -> Result<Option<Subscription>, Failure> {
    let failed = |error: std::io::Error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("response store failed: {error}"),
        )
    };
    // Watched before the record is read, so a job that settles in between is
    // still followed to its end rather than missed.
    let live = background.watch(id);
    let Some(stored) = store.load(id).map_err(failed)? else {
        return Ok(None);
    };
    if stored.response.get("background").and_then(Value::as_bool) != Some(true) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "response {id} was not created with background=true; only background responses \
                 created with stream=true can be streamed"
            ),
        ));
    }
    if live.as_ref().is_some_and(|live| live.borrow().deleted) {
        return Ok(None);
    }
    let Some(reader) = store.open_journal(id).map_err(failed)? else {
        if store.load(id).map_err(failed)?.is_none() {
            return Ok(None);
        }
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "response {id} was not created with stream=true, so it has no events to stream; \
                 retrieve it without stream instead"
            ),
        ));
    };
    if live.is_none() && pending(&stored.response) && !store.recover(id).map_err(failed)? {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "response {id} is still being generated by another server sharing this response \
                 store; stream it from that server, or retrieve it without stream"
            ),
        ));
    }
    Ok(Some(Subscription {
        store: Arc::clone(store),
        id: id.to_owned(),
        reader,
        live,
        after,
    }))
}

/// Start sending `subscription` and hand the frame receiver to axum.
pub(super) fn start(
    subscription: Subscription,
    stall: Duration,
) -> mpsc::Receiver<Result<Bytes, std::io::Error>> {
    // Capacity 1, as for every streamed body: a subscriber's memory is one
    // frame, and a slow one waits here, not in the job.
    let (sender, receiver) = mpsc::channel(1);
    tokio::task::spawn_blocking(move || subscription.run(&sender, stall));
    receiver
}

impl Subscription {
    /// Send every event after the cursor until the stream ends, the response
    /// is deleted, or the client goes away or stalls. Blocks the thread.
    pub(super) fn run(
        mut self,
        sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
        stall: Duration,
    ) {
        let mut ended = false;
        // The last event read, while it may open a failed end whose writing
        // stopped after it.
        let mut opened_end = None;
        loop {
            let (limit, settled, torn) = match self.live.as_mut() {
                Some(live) => {
                    let published = *live.borrow_and_update();
                    if published.deleted {
                        return;
                    }
                    (published.bytes, published.settled, published.torn)
                }
                None => (u64::MAX, true, false),
            };
            loop {
                let line = match self.reader.next_line(limit) {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(error) => {
                        eprintln!("response {}: event journal unreadable: {error}", self.id);
                        return;
                    }
                };
                ended = line.terminal;
                if self.wanted(line.sequence) && !deliver(sender, line.frame(), stall) {
                    return;
                }
                opened_end = (line.kind == "error" && !line.terminal).then_some(line);
            }
            if torn {
                // A failed write left lines past the published ones that may
                // already carry the next sequence numbers, so no end can be
                // numbered here. The stream closes on what was sent; a replay
                // once the job is gone ends after the lines on disk.
                return;
            }
            if settled {
                break;
            }
            let Some(live) = self.live.as_mut() else {
                break;
            };
            let woke = park_until(async {
                tokio::select! {
                    biased;
                    () = sender.closed() => None,
                    changed = live.changed() => Some(changed.is_ok()),
                }
            });
            match woke {
                None => return,
                Some(true) => {}
                // The job is gone; what it wrote is all there is.
                Some(false) => self.live = None,
            }
        }
        if ended {
            return;
        }
        // The journal stops short of the end: it was cut by a crash or a
        // failed write after the record settled, or the record settled
        // elsewhere. The end follows from the record, numbered after the
        // lines read: every line on disk once the writer is gone, or, while
        // it runs, every line it published, its failed batch having been cut
        // back off (otherwise the stream stopped above).
        let Ok(Some(stored)) = self.store.load(&self.id) else {
            return;
        };
        if pending(&stored.response) {
            return;
        }
        let mut events = terminal_events(&stored.response);
        // A failed write may have kept only the first event of a two-event
        // end; it is not repeated.
        if let (Some(line), Some(first)) = (&opened_end, events.first()) {
            let mut first = first.clone();
            first["sequence_number"] = json!(line.sequence);
            if serde_json::from_str::<Value>(&line.text).is_ok_and(|text| text == first) {
                events.remove(0);
            }
        }
        for (sequence, mut event) in (self.reader.next_sequence()..).zip(events) {
            event["sequence_number"] = json!(sequence);
            if self.wanted(sequence) && !deliver(sender, sse_frame(&event), stall) {
                return;
            }
        }
    }

    fn wanted(&self, sequence: u64) -> bool {
        self.after.is_none_or(|after| sequence > after)
    }
}

/// Hand one frame to the client: `false` once it has gone or stalled. A
/// stalled subscriber is only disconnected; the job it watched runs on.
fn deliver(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    frame: String,
    stall: Duration,
) -> bool {
    match send_frame(sender, Bytes::from(frame), stall) {
        FrameOutcome::Sent => true,
        FrameOutcome::Closed => false,
        FrameOutcome::Stalled => {
            eprintln!(
                "dropping a background response stream: the client stopped accepting frames for \
                 {}s; the response keeps running and the stream can be resumed",
                stall.as_secs()
            );
            false
        }
    }
}
