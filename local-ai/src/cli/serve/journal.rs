//! The numbered event journal of a streamed background Response.
//!
//! A background Response created with `stream: true` keeps every event it
//! publishes in an append-only, owner-only sidecar, `.{id}.events`, one
//! compact JSON event per line with `sequence_number` equal to its line
//! number. `GET /v1/responses/{id}?stream=true&starting_after=N` replays it,
//! from this server or another sharing the directory, so a reconnecting
//! client receives byte-for-byte the frames the first client did, with their
//! original timestamps and IDs, never a regenerated document.
//!
//! The stored JSON record stays the source of truth for status; the journal
//! is the record's event history. They agree by one rule: the events that end
//! a stream are a pure function of the terminal record ([`terminal_events`]),
//! and they are appended only after that record is durable. Everything before
//! them is appended before the record is written. So:
//!
//! - a crash after the terminal record but before its events leaves a journal
//!   whose missing tail any reader can reproduce exactly, at exactly the
//!   sequence numbers the writer would have used; readers do so in memory and
//!   recovery appends it;
//! - a journal never announces an end the store does not hold;
//! - a write cut short leaves at most an incomplete last line, which readers
//!   ignore and recovery truncates, so it can never fail a stored response;
//! - a failed write that cannot be cut back off may leave whole lines past
//!   what live subscribers were shown. Their streams then close without an
//!   end, since its sequence numbers are not known; once the writer is gone a
//!   reader numbers the end after the lines actually on disk, so every
//!   sequence number a client has seen keeps naming the same event.
//!
//! Each batch of events is written with one `write` and flushed with
//! `fsync` before any subscriber is told it exists, so a frame a client has
//! seen survives the server process. On Apple platforms plain `fsync` does
//! not flush the drive's own cache (`F_FULLFSYNC` does, at a much higher
//! price per token), so only the first and terminal batches pay for a full
//! flush; a power cut may lose deltas a client saw, never the terminal record.

use std::fs;
use std::io::{BufRead as _, BufReader, ErrorKind, Read as _, Seek as _, SeekFrom, Write as _};
use std::path::Path;

use serde::Deserialize;
use serde_json::{Value, json};

use super::store::{Access, open_private};

/// `code` of the `error` event that ends a cancelled stream.
///
/// The published stream events have no cancellation event and a cancelled
/// response is neither completed, incomplete nor failed, so the stream ends
/// with the one valid event that says why, rather than a false terminal.
pub(super) const CANCELLED_CODE: &str = "response_cancelled";

/// Event types that carry a final Response.
const FINAL_KINDS: [&str; 3] = [
    "response.completed",
    "response.incomplete",
    "response.failed",
];

/// The events that end the stream of `response`, without sequence numbers;
/// none while it is still `queued` or `in_progress`.
pub(super) fn terminal_events(response: &Value) -> Vec<Value> {
    match response.get("status").and_then(Value::as_str) {
        Some("completed") => vec![json!({"type":"response.completed","response":response})],
        Some("incomplete") => vec![json!({"type":"response.incomplete","response":response})],
        Some("failed") => {
            let error = response.get("error").unwrap_or(&Value::Null);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("the response failed");
            vec![
                json!({"type":"error","code":error.get("code").cloned().unwrap_or(Value::Null),"message":message,"param":null}),
                json!({"type":"response.failed","response":response}),
            ]
        }
        Some("cancelled") => vec![
            json!({"type":"error","code":CANCELLED_CODE,"message":"the response was cancelled","param":null}),
        ],
        _ => Vec::new(),
    }
}

/// The Response a journal's terminal `line` ended with, for a record that
/// still claims `pending` to adopt.
pub(super) fn ended_with(pending: &Value, line: &Line) -> Option<Value> {
    if line.kind == "error" {
        let mut cancelled = pending.clone();
        cancelled["status"] = json!("cancelled");
        return Some(cancelled);
    }
    let mut event: Value = serde_json::from_str(&line.text).ok()?;
    let response = event.get_mut("response").map(Value::take)?;
    (response.get("id") == pending.get("id")).then_some(response)
}

/// One complete, well-formed journal line.
#[derive(Debug)]
pub(super) struct Line {
    pub(super) sequence: u64,
    pub(super) kind: String,
    /// The event exactly as written, without its newline.
    pub(super) text: String,
    /// Whether this event ends the stream.
    pub(super) terminal: bool,
}

impl Line {
    /// The SSE frame for this event, identical to the one first sent.
    pub(super) fn frame(&self) -> String {
        format!("event: {}\ndata: {}\n\n", self.kind, self.text)
    }
}

#[derive(Deserialize)]
struct Head {
    #[serde(rename = "type")]
    kind: String,
    sequence_number: u64,
    #[serde(default)]
    code: Option<String>,
}

/// Reads journal lines in order, stopping before anything incomplete.
pub(super) struct JournalReader {
    reader: BufReader<fs::File>,
    /// Bytes of complete, valid lines read so far.
    offset: u64,
    /// The sequence number the next line must carry.
    next: u64,
    /// A malformed or misnumbered line was met; nothing after it counts.
    stopped: bool,
    line: Vec<u8>,
}

impl JournalReader {
    /// Open the journal at `path`; `None` when there is none.
    pub(super) fn open(path: &Path) -> std::io::Result<Option<Self>> {
        match open_private(path, Access::Read) {
            Ok(file) => Ok(Some(Self {
                reader: BufReader::new(file),
                offset: 0,
                next: 0,
                stopped: false,
                line: Vec::new(),
            })),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// The next line wholly within the first `limit` bytes, or `None`.
    ///
    /// An incomplete last line is left unread, so it is read whole once it
    /// is finished, or never.
    pub(super) fn next_line(&mut self, limit: u64) -> std::io::Result<Option<Line>> {
        if self.stopped || self.offset >= limit {
            return Ok(None);
        }
        self.line.clear();
        let read = (&mut self.reader)
            .take(limit - self.offset)
            .read_until(b'\n', &mut self.line)?;
        if read == 0 {
            return Ok(None);
        }
        let Some((b'\n', body)) = self.line.split_last() else {
            self.reader.seek(SeekFrom::Start(self.offset))?;
            return Ok(None);
        };
        let parsed = std::str::from_utf8(body).ok().and_then(|text| {
            let head: Head = serde_json::from_str(text).ok()?;
            (head.sequence_number == self.next).then(|| Line {
                sequence: head.sequence_number,
                terminal: FINAL_KINDS.contains(&head.kind.as_str())
                    || (head.kind == "error" && head.code.as_deref() == Some(CANCELLED_CODE)),
                kind: head.kind,
                text: text.to_owned(),
            })
        });
        let Some(line) = parsed else {
            self.stopped = true;
            return Ok(None);
        };
        self.offset += read as u64;
        self.next += 1;
        Ok(Some(line))
    }

    /// Read every remaining complete line, returning the last.
    pub(super) fn last_line(&mut self) -> std::io::Result<Option<Line>> {
        let mut last = None;
        while let Some(line) = self.next_line(u64::MAX)? {
            last = Some(line);
        }
        Ok(last)
    }

    /// The sequence number the next event will carry.
    pub(super) const fn next_sequence(&self) -> u64 {
        self.next
    }
}

/// Appends numbered events to one journal. Not shared: its job's lock, or the
/// response's lease, serialises every writer.
#[derive(Debug)]
pub(super) struct JournalWriter {
    file: fs::File,
    /// Bytes of whole batches on disk; what subscribers may read.
    bytes: u64,
    next: u64,
    /// A write failed; nothing more is appended, so no gap is ever written.
    broken: bool,
    /// A failed write could not be rolled back to `bytes`, so the file may
    /// hold complete lines of that batch past what subscribers were shown.
    torn: bool,
    /// Make the next append write all but a fragment of its batch, then fail
    /// in a way that cannot be rolled back.
    #[cfg(test)]
    tear: Option<std::path::PathBuf>,
}

impl JournalWriter {
    /// Create a new journal at `path`; an existing one is an error.
    pub(super) fn create(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            file: open_private(path, Access::CreateNew)?,
            bytes: 0,
            next: 0,
            broken: false,
            torn: false,
            #[cfg(test)]
            tear: None,
        })
    }

    /// Continue the journal `reader` has read to its end, dropping any
    /// incomplete or invalid tail it stopped before.
    pub(super) fn resume(path: &Path, reader: &JournalReader) -> std::io::Result<Self> {
        let file = open_private(path, Access::Append)?;
        if file.metadata()?.len() != reader.offset {
            file.set_len(reader.offset)?;
        }
        Ok(Self {
            file,
            bytes: reader.offset,
            next: reader.next,
            broken: false,
            torn: false,
            #[cfg(test)]
            tear: None,
        })
    }

    /// Number `events` from the next sequence and append them as one write,
    /// flushed before returning. On failure the journal is cut back to its
    /// last whole batch, if it can be, and refuses further appends; when it
    /// cannot be, [`Self::torn`] says so.
    pub(super) fn append(&mut self, events: Vec<Value>) -> std::io::Result<()> {
        if self.broken {
            return Err(std::io::Error::other("an earlier journal write failed"));
        }
        if events.is_empty() {
            return Ok(());
        }
        let mut batch = String::new();
        let mut next = self.next;
        for mut event in events {
            if let Some(fields) = event.as_object_mut() {
                fields.insert("sequence_number".into(), json!(next));
            }
            batch.push_str(&serde_json::to_string(&event).map_err(std::io::Error::other)?);
            batch.push('\n');
            next += 1;
        }
        if let Err(error) = self.write_batch(batch.as_bytes()) {
            self.broken = true;
            self.torn = !self.roll_back();
            return Err(error);
        }
        self.bytes += batch.len() as u64;
        self.next = next;
        Ok(())
    }

    fn write_batch(&mut self, batch: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if let Some(path) = self.tear.take() {
            // Every line but the last lands whole, the last only in part.
            let last = batch[..batch.len() - 1]
                .iter()
                .rposition(|&byte| byte == b'\n')
                .map_or(0, |newline| newline + 1);
            let cut = last + (batch.len() - last) / 2;
            self.file.write_all(&batch[..cut])?;
            self.file.flush()?;
            // Read-only: the truncation that would roll it back fails too.
            self.file = fs::File::open(path)?;
            return Err(std::io::Error::other("injected torn journal write"));
        }
        self.file
            .write_all(batch)
            .and_then(|()| flush_to_device(&self.file))
    }

    /// Cut a failed batch back off, durably; `true` when the file is known to
    /// hold exactly the whole batches written before it.
    fn roll_back(&self) -> bool {
        match self.file.set_len(self.bytes) {
            Ok(()) => flush_to_device(&self.file).is_ok(),
            // A write that failed before reaching the file needs no cut.
            Err(_) => self
                .file
                .metadata()
                .is_ok_and(|metadata| metadata.len() == self.bytes),
        }
    }

    /// A failed write may have left complete lines past [`Self::bytes`] that
    /// could not be removed: what follows the published prefix is unknown
    /// until the writer is gone and a reader sees the file as it is.
    pub(super) const fn torn(&self) -> bool {
        self.torn
    }

    /// Flush everything appended through the drive's own cache.
    pub(super) fn sync(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }

    pub(super) const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Make every later append fail, as a full or failing disk would.
    #[cfg(test)]
    pub(super) fn break_for_test(&mut self, path: &Path) -> std::io::Result<()> {
        self.file = fs::File::open(path)?;
        Ok(())
    }

    /// Make the next append leave whole lines of its batch behind and fail
    /// to roll them back, as a disk failing mid-write and on truncate would.
    #[cfg(test)]
    pub(super) fn tear_for_test(&mut self, path: &Path) {
        self.tear = Some(path.to_owned());
    }
}

#[cfg(unix)]
fn flush_to_device(file: &fs::File) -> std::io::Result<()> {
    rustix::fs::fsync(file).map_err(std::io::Error::from)
}

#[cfg(not(unix))]
fn flush_to_device(file: &fs::File) -> std::io::Result<()> {
    file.sync_data()
}
