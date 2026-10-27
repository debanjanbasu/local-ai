//! `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` on the
//! native Whisper large-v3 model ([`local_engine::transcription`]).
//!
//! The artifact beside the Bonsai checkpoint (`models/whisper-large-v3/`, see
//! `docs/TRANSCRIPTION.md`) is required: the server does not start without
//! it, and there is no flag.
//!
//! Requests follow the pinned `CreateTranscriptionRequest` and
//! `CreateTranslationRequest` (multipart): `file` (flac, mp3, mp4, mpeg,
//! mpga, m4a, ogg, wav or webm, at most [`MAX_FILE_BYTES`] and
//! [`MAX_AUDIO_SECONDS`] of audio; Opus is not decodable here), `model`
//! ([`TRANSCRIPTION_MODEL_ID`] or `whisper-1`), `prompt`, `temperature` (0 to 1; 0
//! means the reference's fallback ladder), `response_format` (`json`,
//! `text`, `srt`, `verbose_json`, `vtt`). Transcriptions also take
//! `language` (ISO-639-1), `timestamp_granularities[]` (`segment` and
//! `word`, with `verbose_json`), `include[]` (`logprobs`, with `json` or streaming)
//! and `stream` (`transcript.text.delta` events per finished 30-second
//! window, then `transcript.text.done`). Diarization, `chunking_strategy`,
//! `known_speaker_*`, `languages` and `keywords` are refused with `400`.
//!
//! # Scheduling
//!
//! Speech runs on its own Metal queue, one request at a time, at low
//! priority: while Bonsai generations are running it submits about 2 ms of
//! GPU work at a time and waits for the engine worker to finish a decode
//! step between submissions ([`EngineActivity::yield_to_decode`]). At most
//! [`MAX_PENDING`] requests are admitted (running or waiting); more are
//! answered `503` with `Retry-After`. A client that goes away cancels its
//! request at the next submission boundary.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use axum::body::Body;
use axum::extract::multipart::MultipartError;
use axum::extract::{DefaultBodyLimit, FromRequest as _, Multipart, Request};
use axum::http::request::Parts;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use bytes::Bytes;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc};

use local_engine::activity::EngineActivity;
use local_engine::transcription::tokens::{language_code, language_name};
use local_engine::transcription::{
    Pace, Segment, TRANSCRIPTION_MODEL_ID, Task, Temperature, TranscribeOptions, Transcript,
    TranscriptionModel, audio, installed_model,
};

use super::admin::{self, Caller, Meter};
use super::response::{add_alt_svc, error_response, json_response};
use super::{AppState, event_stream};

pub(super) const TRANSCRIPTIONS: &str = "/v1/audio/transcriptions";
pub(super) const TRANSLATIONS: &str = "/v1/audio/translations";
/// Speech requests admitted at once (one runs, the rest wait in order).
pub(super) const MAX_PENDING: usize = 4;
/// Upload limit, the hosted API's own for these endpoints.
pub(super) const MAX_FILE_BYTES: u64 = 25 * 1024 * 1024;
/// Longest audio accepted.
pub(super) const MAX_AUDIO_SECONDS: u64 = 3600;
const FORM_OVERHEAD_BYTES: u64 = 64 * 1024;
const MAX_FORM_FIELDS: usize = 16;
const MAX_TEXT_FIELD_BYTES: usize = 16 * 1024;
const DECODE_YIELD_LIMIT: Duration = Duration::from_millis(500);
const RETRY_AFTER_SECONDS: u32 = 1;
const EXTENSIONS: [&str; 9] = [
    "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm",
];
/// Audio encoder frames per 30-second window, for token usage.
const AUDIO_TOKENS_PER_WINDOW: usize = 1500;

pub(super) struct Transcriber {
    model: Arc<Mutex<TranscriptionModel>>,
    /// Token bytes for `logprobs`, readable while the model is busy.
    token_bytes: Arc<Vec<Vec<u8>>>,
    eot: u32,
    slots: Arc<Semaphore>,
    activity: Arc<EngineActivity>,
}

impl Transcriber {
    /// The service, from the required install in `models_dir`.
    pub(super) fn start(
        models_dir: &Path,
        activity: Arc<EngineActivity>,
    ) -> crate::Result<Arc<Self>> {
        let dir = installed_model(models_dir)?;
        let model = TranscriptionModel::open(&dir).map_err(|error| {
            super::artifact_error(format!(
                "cannot load the transcription model in {}: {error}",
                dir.display()
            ))
        })?;
        eprintln!(
            "transcription at POST {TRANSCRIPTIONS} and {TRANSLATIONS}: {TRANSCRIPTION_MODEL_ID} from {} (F16, \
             payload sha256 {})",
            dir.display(),
            model.payload_sha256()
        );
        Ok(Arc::new(Self {
            token_bytes: model.token_byte_table(),
            eot: model.eot(),
            model: Arc::new(Mutex::new(model)),
            slots: Arc::new(Semaphore::new(MAX_PENDING)),
            activity,
        }))
    }
}

/// `/v1/models` entry of the speech model.
pub(super) fn model_json(created: u64) -> Value {
    json!({"id":TRANSCRIPTION_MODEL_ID,"object":"model","created":created,"owned_by":"local"})
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

#[derive(Debug)]
#[expect(clippy::struct_excessive_bools)]
struct Form {
    file: Vec<u8>,
    extension: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    format: Format,
    temperature: f32,
    stream: bool,
    logprobs: bool,
    /// `timestamp_granularities[]` includes `word` / `segment` (segments
    /// also when none is given).
    words: bool,
    segments: bool,
}

type Failure = (StatusCode, String);

fn bad(message: impl Into<String>) -> Failure {
    (StatusCode::BAD_REQUEST, message.into())
}

fn multipart_failure(error: &MultipartError) -> Failure {
    (
        error.status(),
        format!("invalid multipart body: {}", error.body_text()),
    )
}

/// The form fields of either endpoint, validated against its schema.
#[derive(Default)]
struct RawForm {
    file: Option<(Vec<u8>, Option<String>)>,
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    temperature: Option<String>,
    stream: Option<String>,
    granularities: Vec<String>,
    include: Vec<String>,
}

fn set_once(slot: &mut Option<String>, name: &str, value: String) -> Result<(), Failure> {
    if slot.is_some() {
        return Err(bad(format!("field {name:?} is repeated")));
    }
    *slot = Some(value);
    Ok(())
}

/// [`read_form`], failing with `408` unless the whole form arrives within
/// `deadline`.
async fn read_form_within(
    parts: &Parts,
    body: Body,
    task: Task,
    deadline: Duration,
) -> Result<RawForm, Failure> {
    tokio::time::timeout(deadline, read_form(parts, body, task))
        .await
        .unwrap_or_else(|_| {
            Err((
                StatusCode::REQUEST_TIMEOUT,
                format!(
                    "the form did not arrive within {} s (--stall-timeout)",
                    deadline.as_secs()
                ),
            ))
        })
}

#[expect(clippy::too_many_lines)]
async fn read_form(parts: &Parts, body: Body, task: Task) -> Result<RawForm, Failure> {
    let content_type = parts
        .headers
        .get(header::CONTENT_TYPE)
        .cloned()
        .ok_or_else(|| bad("a multipart/form-data body is required"))?;
    let mut request = Request::new(body);
    request
        .headers_mut()
        .insert(header::CONTENT_TYPE, content_type);
    let bound = usize::try_from(MAX_FILE_BYTES + FORM_OVERHEAD_BYTES).unwrap_or(usize::MAX);
    DefaultBodyLimit::max(bound).apply(&mut request);
    let mut form = Multipart::from_request(request, &())
        .await
        .map_err(|rejection| (rejection.status(), rejection.body_text()))?;
    let mut raw = RawForm::default();
    let mut count = 0;
    while let Some(mut field) = form
        .next_field()
        .await
        .map_err(|error| multipart_failure(&error))?
    {
        count += 1;
        if count > MAX_FORM_FIELDS {
            return Err(bad(format!(
                "a form may carry at most {MAX_FORM_FIELDS} fields"
            )));
        }
        let name = field.name().unwrap_or_default().to_owned();
        if name == "file" {
            if raw.file.is_some() {
                return Err(bad("field \"file\" is repeated"));
            }
            let filename = field.file_name().map(str::to_owned);
            let mut data = Vec::new();
            while let Some(chunk) = field
                .chunk()
                .await
                .map_err(|error| multipart_failure(&error))?
            {
                if (data.len() + chunk.len()) as u64 > MAX_FILE_BYTES {
                    return Err((
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!("the audio file exceeds the {MAX_FILE_BYTES}-byte limit"),
                    ));
                }
                data.extend_from_slice(&chunk);
            }
            raw.file = Some((data, filename));
            continue;
        }
        let mut text = Vec::new();
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|error| multipart_failure(&error))?
        {
            if text.len() + chunk.len() > MAX_TEXT_FIELD_BYTES {
                return Err(bad(format!(
                    "{name} may be at most {MAX_TEXT_FIELD_BYTES} bytes"
                )));
            }
            text.extend_from_slice(&chunk);
        }
        let text =
            String::from_utf8(text).map_err(|_| bad(format!("{name} must be UTF-8 text")))?;
        let transcription = task == Task::Transcribe;
        match name.as_str() {
            "model" => set_once(&mut raw.model, &name, text)?,
            "prompt" => set_once(&mut raw.prompt, &name, text)?,
            "response_format" => set_once(&mut raw.response_format, &name, text)?,
            "temperature" => set_once(&mut raw.temperature, &name, text)?,
            "language" if transcription => set_once(&mut raw.language, &name, text)?,
            "stream" if transcription => set_once(&mut raw.stream, &name, text)?,
            "timestamp_granularities[]" | "timestamp_granularities" if transcription => {
                raw.granularities.push(text);
            }
            "include[]" | "include" if transcription => raw.include.push(text),
            "chunking_strategy" if transcription => {
                return Err(bad(
                    "chunking_strategy is not supported: Whisper always transcribes in its own \
                     sequential 30-second windows",
                ));
            }
            "known_speaker_names[]"
            | "known_speaker_names"
            | "known_speaker_references[]"
            | "known_speaker_references"
                if transcription =>
            {
                return Err(bad("speaker diarization is not supported by this model"));
            }
            "languages[]" | "languages" | "keywords[]" | "keywords" if transcription => {
                return Err(bad(format!(
                    "{name} is not supported by this model (use language and prompt)"
                )));
            }
            other => return Err(bad(format!("unknown field {other:?}"))),
        }
    }
    Ok(raw)
}

#[expect(clippy::too_many_lines)]
fn validate(raw: RawForm, task: Task) -> Result<Form, Failure> {
    let (file, filename) = raw.file.ok_or_else(|| bad("field \"file\" is required"))?;
    let model = raw
        .model
        .ok_or_else(|| bad("field \"model\" is required"))?;
    if super::models::resolve(&model, super::models::BONSAI_MODEL_ID)
        != Some(super::models::Served::Transcription)
    {
        return Err(bad(format!(
            "model {model:?} is not served here; use {TRANSCRIPTION_MODEL_ID:?}"
        )));
    }
    let extension = filename
        .as_deref()
        .and_then(|name| Path::new(name).extension())
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);
    if let Some(ext) = &extension
        && !EXTENSIONS.contains(&ext.as_str())
    {
        return Err(bad(format!(
            "file format {ext:?} is not supported; use one of {}",
            EXTENSIONS.join(", ")
        )));
    }
    let format = match raw.response_format.as_deref() {
        None | Some("json") => Format::Json,
        Some("text") => Format::Text,
        Some("srt") => Format::Srt,
        Some("verbose_json") => Format::VerboseJson,
        Some("vtt") => Format::Vtt,
        Some("diarized_json") if task == Task::Transcribe => {
            return Err(bad("diarized_json is not supported by this model"));
        }
        Some(other) => {
            return Err(bad(format!(
                "response_format must be json, text, srt, verbose_json or vtt, not {other:?}"
            )));
        }
    };
    let temperature = match raw.temperature.as_deref() {
        None => 0.0,
        Some(text) => text
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|t| (0.0..=1.0).contains(t))
            .ok_or_else(|| bad("temperature must be a number between 0 and 1"))?,
    };
    let language = match raw.language {
        None => None,
        Some(language) if language.trim().is_empty() => None,
        Some(language) => Some(
            language_code(&language)
                .ok_or_else(|| bad(format!("language {language:?} is not supported by Whisper")))?
                .to_owned(),
        ),
    };
    let stream = match raw.stream.as_deref().map(str::trim) {
        None | Some("false") => false,
        Some("true") => true,
        Some(other) => return Err(bad(format!("stream must be true or false, not {other:?}"))),
    };
    if stream && !matches!(format, Format::Json | Format::Text) {
        return Err(bad("streaming supports response_format json or text"));
    }
    let (mut words, mut segments) = (false, raw.granularities.is_empty());
    for granularity in &raw.granularities {
        match granularity.as_str() {
            "segment" => segments = true,
            "word" => words = true,
            other => {
                return Err(bad(format!(
                    "timestamp_granularities must be word or segment, not {other:?}"
                )));
            }
        }
    }
    if !raw.granularities.is_empty() && format != Format::VerboseJson {
        return Err(bad(
            "timestamp_granularities requires response_format verbose_json",
        ));
    }
    let mut logprobs = false;
    for item in &raw.include {
        if item != "logprobs" {
            return Err(bad(format!("include must be logprobs, not {item:?}")));
        }
        logprobs = true;
    }
    if logprobs && !stream && format != Format::Json {
        return Err(bad("include[]=logprobs requires response_format json"));
    }
    Ok(Form {
        file,
        extension,
        language,
        prompt: raw.prompt.filter(|p| !p.trim().is_empty()),
        format,
        temperature,
        stream,
        logprobs,
        words,
        segments,
    })
}

/// Sets the flag when dropped: the handler future went away (client gone).
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

enum Outcome {
    Invalid(String),
    Internal(String),
    Cancelled,
}

fn options(form: &Form, task: Task) -> TranscribeOptions {
    TranscribeOptions {
        task,
        language: form.language.clone(),
        prompt: form.prompt.clone(),
        temperature: if form.temperature == 0.0 {
            Temperature::Fallback
        } else {
            Temperature::Fixed(form.temperature)
        },
        condition_on_previous_text: true,
        seed: u64::from(std::process::id()) ^ local_seed(),
        word_timestamps: form.words,
    }
}

fn local_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64)
}

/// Decode the upload and run the model, sending segments to `on_segments`.
fn run(
    service: &Transcriber,
    form: &Form,
    file: Vec<u8>,
    task: Task,
    cancelled: &AtomicBool,
    on_segments: &mut dyn FnMut(&[Segment]) -> local_engine::Result<()>,
) -> Result<Transcript, Outcome> {
    let samples = audio::decode(file, form.extension.as_deref(), MAX_AUDIO_SECONDS)
        .map_err(|error| Outcome::Invalid(error.to_string()))?;
    let mut model = service.model.lock().unwrap_or_else(PoisonError::into_inner);
    // The client may have gone while the file decoded or the model was busy.
    if cancelled.load(Ordering::SeqCst) {
        return Err(Outcome::Cancelled);
    }
    let activity = Arc::clone(&service.activity);
    let busy = || activity.busy();
    let mut pause = || {
        if cancelled.load(Ordering::SeqCst) {
            return Err(local_engine::Error::Generation(
                "transcription request cancelled".into(),
            ));
        }
        activity.yield_to_decode(DECODE_YIELD_LIMIT);
        Ok(())
    };
    let mut pace = Pace {
        busy: &busy,
        pause: &mut pause,
    };
    let result = model.transcribe(&samples, &options(form, task), &mut pace, on_segments);
    drop(model);
    result.map_err(|error| {
        if cancelled.load(Ordering::SeqCst) {
            Outcome::Cancelled
        } else if matches!(error, local_engine::Error::InvalidArgument(_)) {
            Outcome::Invalid(error.to_string())
        } else {
            Outcome::Internal(error.to_string())
        }
    })
}

fn logprobs_json(service: &Transcriber, segments: &[Segment]) -> Vec<Value> {
    segments
        .iter()
        .flat_map(|s| s.tokens.iter().zip(&s.logprobs))
        .filter(|(token, _)| **token < service.eot)
        .map(|(token, logprob)| {
            let bytes = service
                .token_bytes
                .get(*token as usize)
                .map_or(&[][..], Vec::as_slice);
            json!({"token":String::from_utf8_lossy(bytes),"logprob":logprob,"bytes":bytes})
        })
        .collect()
}

/// `whisper.utils.format_timestamp`.
fn timestamp(seconds: f64, always_hours: bool, decimal: char) -> String {
    let total = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = total / 3_600_000;
    let minutes = total % 3_600_000 / 60_000;
    let secs = total % 60_000 / 1000;
    let millis = total % 1000;
    let mut out = String::new();
    if always_hours || hours > 0 {
        let _ = write!(out, "{hours:02}:");
    }
    let _ = write!(out, "{minutes:02}:{secs:02}{decimal}{millis:03}");
    out
}

/// SRT or `WebVTT` captions of the non-empty segments.
fn captions(segments: &[Segment], vtt: bool) -> String {
    let mut out = String::new();
    if vtt {
        out.push_str("WEBVTT\n\n");
    }
    let mut index = 0;
    for segment in segments.iter().filter(|s| !s.text.trim().is_empty()) {
        index += 1;
        let text = segment.text.trim().replace("-->", "->");
        if vtt {
            let _ = write!(
                out,
                "{} --> {}\n{text}\n\n",
                timestamp(segment.start, false, '.'),
                timestamp(segment.end, false, '.')
            );
        } else {
            let _ = write!(
                out,
                "{index}\n{} --> {}\n{text}\n\n",
                timestamp(segment.start, true, ','),
                timestamp(segment.end, true, ',')
            );
        }
    }
    out
}

fn duration_usage(transcript: &Transcript) -> Value {
    json!({"type":"duration","seconds":(transcript.duration * 1000.0).round() / 1000.0})
}

fn verbose_json(transcript: &Transcript, task: Task, words: bool, segments: bool) -> Value {
    let segment_values: Vec<Value> = transcript
        .segments
        .iter()
        .map(|s| {
            json!({
                "id":s.id,"seek":s.seek,"start":s.start,"end":s.end,"text":s.text,
                "tokens":s.tokens,"temperature":s.temperature,"avg_logprob":s.avg_logprob,
                "compression_ratio":s.compression_ratio,"no_speech_prob":s.no_speech_prob,
            })
        })
        .collect();
    let (task_name, language) = match task {
        Task::Transcribe => (
            "transcribe",
            language_name(&transcript.language).unwrap_or("unknown"),
        ),
        Task::Translate => ("translate", "english"),
    };
    let mut value = json!({
        "task":task_name,
        "language":language,
        "duration":transcript.duration,
        "text":transcript.text.trim(),
    });
    if segments {
        value["segments"] = Value::Array(segment_values);
    }
    if words {
        let words: Vec<Value> = transcript
            .segments
            .iter()
            .flat_map(|s| &s.words)
            .map(|w| json!({"word":w.word.trim(),"start":w.start,"end":w.end}))
            .collect();
        value["words"] = Value::Array(words);
    }
    if task == Task::Transcribe {
        value["usage"] = duration_usage(transcript);
    }
    value
}

fn text_response(body: String, content_type: &'static str, state: &AppState) -> Response {
    let mut response = Response::new(Body::from(body));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    add_alt_svc(&mut response, state);
    response
}

async fn busy(state: &AppState, headers: &HeaderMap) -> Response {
    let mut response = json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"error":{
            "message":format!(
                "{MAX_PENDING} transcription requests are already running or waiting; retry after \
                 {RETRY_AFTER_SECONDS}s with your own backoff"
            ),
            "type":"server_error",
            "code":null,
            "param":null,
        }}),
        state,
        headers,
    )
    .await;
    response.headers_mut().insert(
        header::RETRY_AFTER,
        header::HeaderValue::from(RETRY_AFTER_SECONDS),
    );
    response
}

fn sse(value: &Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

#[expect(clippy::too_many_lines)]
pub(super) async fn handle(
    state: &AppState,
    parts: &Parts,
    body: Body,
    caller: &Caller,
) -> Response {
    let headers = &parts.headers;
    let service = Arc::clone(&state.transcription);
    let task = if parts.uri.path() == TRANSLATIONS {
        Task::Translate
    } else {
        Task::Transcribe
    };
    // The slot bounds the forms held in memory (each up to 25 MiB), so it is
    // taken first, and the form must then arrive within the stall budget: a
    // client trickling its upload loses the slot instead of holding one of
    // the few there are.
    let Ok(permit) = Arc::clone(&service.slots).try_acquire_owned() else {
        return busy(state, headers).await;
    };
    let form = match read_form_within(parts, body, task, state.stall)
        .await
        .and_then(|raw| validate(raw, task))
    {
        Ok(form) => form,
        Err((status, message)) => return error_response(status, &message, state, headers).await,
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    if form.stream {
        let meter = Meter::into_store(
            state.auth.admin(),
            admin::transcription_usage(caller, TRANSCRIPTION_MODEL_ID, 0.0),
        );
        return stream(state, service, form, task, permit, cancelled, meter);
    }
    let guard = CancelOnDrop(Arc::clone(&cancelled));
    let worker = Arc::clone(&service);
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut form = form;
        let file = std::mem::take(&mut form.file);
        let transcript = run(&worker, &form, file, task, &cancelled, &mut |_| Ok(()));
        let logprobs = match (&transcript, form.logprobs) {
            (Ok(t), true) => Some(logprobs_json(&worker, &t.segments)),
            _ => None,
        };
        (
            transcript,
            (form.format, form.words, form.segments),
            logprobs,
        )
    })
    .await;
    drop(guard);
    let (transcript, (format, words, segments), logprobs) = match result {
        Ok(value) => value,
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &error.to_string(),
                state,
                headers,
            )
            .await;
        }
    };
    let transcript = match transcript {
        Ok(transcript) => transcript,
        Err(Outcome::Invalid(message)) => {
            return error_response(StatusCode::BAD_REQUEST, &message, state, headers).await;
        }
        Err(Outcome::Internal(message)) => {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, &message, state, headers)
                .await;
        }
        Err(Outcome::Cancelled) => {
            return error_response(StatusCode::BAD_REQUEST, "request cancelled", state, headers)
                .await;
        }
    };
    // Translations are counted in the same family, as the hosted API does.
    admin::record(
        state,
        admin::transcription_usage(caller, TRANSCRIPTION_MODEL_ID, transcript.duration),
    );
    match format {
        Format::Json => {
            let mut value = json!({"text":transcript.text.trim()});
            if task == Task::Transcribe {
                value["usage"] = duration_usage(&transcript);
            }
            if let Some(logprobs) = logprobs {
                value["logprobs"] = Value::Array(logprobs);
            }
            json_response(StatusCode::OK, value, state, headers).await
        }
        Format::VerboseJson => {
            json_response(
                StatusCode::OK,
                verbose_json(&transcript, task, words, segments),
                state,
                headers,
            )
            .await
        }
        Format::Text => text_response(
            format!("{}\n", transcript.text.trim()),
            "text/plain; charset=utf-8",
            state,
        ),
        Format::Srt => text_response(
            captions(&transcript.segments, false),
            "text/plain; charset=utf-8",
            state,
        ),
        Format::Vtt => text_response(captions(&transcript.segments, true), "text/vtt", state),
    }
}

/// `stream: true`: deltas per finished window, then the done event.
fn stream(
    state: &AppState,
    service: Arc<Transcriber>,
    mut form: Form,
    task: Task,
    permit: tokio::sync::OwnedSemaphorePermit,
    cancelled: Arc<AtomicBool>,
    meter: Meter,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let file = std::mem::take(&mut form.file);
        let mut first = true;
        let worker = Arc::clone(&service);
        let mut on_segments = |segments: &[Segment]| -> local_engine::Result<()> {
            let mut delta: String = segments.iter().map(|s| s.text.as_str()).collect();
            if first {
                delta = delta.trim_start().to_owned();
            }
            if delta.is_empty() {
                return Ok(());
            }
            first = false;
            let mut event = json!({"type":"transcript.text.delta","delta":delta});
            if form.logprobs {
                event["logprobs"] = Value::Array(logprobs_json(&service, segments));
            }
            if sender.blocking_send(Ok(sse(&event))).is_err() {
                cancelled.store(true, Ordering::SeqCst);
                return Err(local_engine::Error::Generation("client went away".into()));
            }
            Ok(())
        };
        let result = run(&worker, &form, file, task, &cancelled, &mut on_segments);
        let frame = match result {
            Ok(transcript) => {
                meter.write_with(|record| record.units = admin::audio_seconds(transcript.duration));
                let text_tokens = transcript.usage.prompt_tokens;
                let audio_tokens = transcript.usage.audio_windows * AUDIO_TOKENS_PER_WINDOW;
                let mut event = json!({
                    "type":"transcript.text.done",
                    "text":transcript.text.trim(),
                    "usage":{
                        "type":"tokens",
                        "input_tokens":text_tokens + audio_tokens,
                        "input_token_details":{"text_tokens":text_tokens,"audio_tokens":audio_tokens},
                        "output_tokens":transcript.usage.output_tokens,
                        "total_tokens":text_tokens + audio_tokens + transcript.usage.output_tokens,
                    },
                });
                if form.logprobs {
                    event["logprobs"] = Value::Array(logprobs_json(&worker, &transcript.segments));
                }
                event
            }
            Err(Outcome::Cancelled) => return,
            Err(Outcome::Invalid(message)) => json!({"type":"error","error":{
                "message":message,"type":"invalid_request_error","code":null,"param":null}}),
            Err(Outcome::Internal(message)) => json!({"type":"error","error":{
                "message":message,"type":"server_error","code":null,"param":null}}),
        };
        let _ = sender.blocking_send(Ok(sse(&frame)));
    });
    event_stream(receiver, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(fields: &[(&str, &str)]) -> RawForm {
        let mut form = RawForm {
            file: Some((b"audio".to_vec(), Some("clip.wav".into()))),
            model: Some(TRANSCRIPTION_MODEL_ID.into()),
            ..RawForm::default()
        };
        for (name, value) in fields {
            let value = (*value).to_owned();
            match *name {
                "response_format" => form.response_format = Some(value),
                "temperature" => form.temperature = Some(value),
                "language" => form.language = Some(value),
                "stream" => form.stream = Some(value),
                "granularity" => form.granularities.push(value),
                "include" => form.include.push(value),
                "model" => form.model = Some(value),
                "file" => form.file = Some((b"a".to_vec(), Some(value))),
                _ => unreachable!(),
            }
        }
        form
    }

    #[tokio::test]
    async fn a_trickling_form_gives_its_slot_back_at_the_deadline() {
        // The form's first bytes, then nothing: the request holds a slot.
        let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
        sender
            .send(Ok(Bytes::from_static(
                b"--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhis",
            )))
            .await
            .expect("sent");
        let request = axum::http::Request::builder()
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=b")
            .body(())
            .expect("request");
        let (parts, ()) = request.into_parts();
        let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(receiver));
        let deadline = Duration::from_millis(200);
        let read = tokio::time::timeout(
            Duration::from_secs(5),
            read_form_within(&parts, body, Task::Transcribe, deadline),
        )
        .await
        .expect("the read ends at its deadline");
        let Err((status, message)) = read else {
            unreachable!("a form that never ends was read");
        };
        assert_eq!(status, StatusCode::REQUEST_TIMEOUT, "{message}");
        drop(sender);
    }

    #[test]
    fn valid_forms_follow_the_spec() {
        let form = validate(raw(&[]), Task::Transcribe).expect("defaults");
        assert_eq!(form.format, Format::Json);
        assert_eq!(form.temperature, 0.0);
        assert_eq!(form.extension.as_deref(), Some("wav"));
        let form = validate(
            raw(&[
                ("response_format", "verbose_json"),
                ("granularity", "segment"),
                ("language", "de"),
                ("temperature", "0.4"),
                ("model", "whisper-1"),
            ]),
            Task::Transcribe,
        )
        .expect("verbose");
        assert_eq!(form.language.as_deref(), Some("de"));
        assert_eq!(form.format, Format::VerboseJson);
        let form = validate(
            raw(&[("stream", "true"), ("include", "logprobs")]),
            Task::Transcribe,
        )
        .expect("stream");
        assert!(form.stream && form.logprobs);
        assert!(validate(raw(&[("response_format", "srt")]), Task::Translate).is_ok());
    }

    #[test]
    fn unsupported_requests_are_refused() {
        for fields in [
            vec![("granularity", "segment")],
            vec![("response_format", "diarized_json")],
            vec![("response_format", "xml")],
            vec![("temperature", "1.5")],
            vec![("temperature", "hot")],
            vec![("language", "klingon")],
            vec![("stream", "true"), ("response_format", "srt")],
            vec![("include", "logprobs"), ("response_format", "text")],
            vec![("model", "tts-1")],
            vec![("file", "clip.aiff")],
        ] {
            assert!(
                validate(raw(&fields), Task::Transcribe).is_err(),
                "{fields:?}"
            );
        }
        let mut missing = raw(&[]);
        missing.file = None;
        assert!(validate(missing, Task::Transcribe).is_err());
    }

    fn segment(id: usize, start: f64, end: f64, text: &str) -> Segment {
        Segment {
            id,
            seek: 0,
            start,
            end,
            text: text.into(),
            tokens: vec![1, 2],
            logprobs: vec![-0.1, -0.2],
            temperature: 0.0,
            avg_logprob: -0.1,
            compression_ratio: 1.0,
            no_speech_prob: 0.01,
            words: vec![local_engine::transcription::Word {
                word: format!(" w{id}"),
                start,
                end,
                probability: 0.9,
            }],
        }
    }

    #[test]
    fn captions_match_the_reference_writers() {
        let segments = vec![
            segment(0, 0.0, 2.5, " Hello there."),
            segment(1, 2.5, 2.5, ""),
            segment(2, 3661.004, 3662.0, " A --> B"),
        ];
        assert_eq!(
            captions(&segments, false),
            "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n\
             2\n01:01:01,004 --> 01:01:02,000\nA -> B\n\n"
        );
        assert_eq!(
            captions(&segments, true),
            "WEBVTT\n\n00:00.000 --> 00:02.500\nHello there.\n\n\
             01:01:01.004 --> 01:01:02.000\nA -> B\n\n"
        );
    }

    #[test]
    fn verbose_json_reports_language_names_and_segments() {
        let transcript = Transcript {
            text: " Hallo.".into(),
            language: "de".into(),
            duration: 2.0,
            segments: vec![segment(0, 0.0, 2.0, " Hallo.")],
            usage: local_engine::transcription::Usage::default(),
        };
        let value = verbose_json(&transcript, Task::Transcribe, true, false);
        assert_eq!(value["words"][0]["word"], "w0");
        assert!(value.get("segments").is_none());
        let value = verbose_json(&transcript, Task::Transcribe, false, true);
        assert!(value.get("words").is_none());
        assert_eq!(value["language"], "german");
        assert_eq!(value["text"], "Hallo.");
        assert_eq!(value["task"], "transcribe");
        assert_eq!(value["segments"][0]["tokens"], json!([1, 2]));
        assert_eq!(value["usage"]["type"], "duration");
        let value = verbose_json(&transcript, Task::Translate, false, true);
        assert_eq!(value["language"], "english");
        assert!(value.get("usage").is_none());
    }
}
