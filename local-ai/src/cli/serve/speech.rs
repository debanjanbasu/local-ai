//! `POST /v1/audio/speech` on the native Kokoro-82M synthesizer
//! ([`local_engine::speech`]).
//!
//! The artifact beside the Bonsai checkpoint (`models/kokoro-82m/`, see
//! `docs/SPEECH.md`) is required: the server does not start without it, and
//! there is no flag.
//!
//! The request follows the pinned `CreateSpeechRequest`:
//!
//! - `model` must be [`SPEECH_MODEL_ID`];
//! - `input`: 1 to 4,096 characters of English text;
//! - `voice`: a Kokoro voice id (`af_heart`, `bm_george`, …), one of the
//!   pinned built-in names (`alloy`, `ash`, … `cedar`, each mapped to a
//!   distinct Kokoro voice, see [`VOICE_ALIASES`]; they are not `OpenAI`'s
//!   voices), or a custom voice object `{"id": …}` naming either;
//! - `instructions` is refused when non-empty: Kokoro has no instruction or
//!   style-prompt input, so it could not honor one;
//! - `response_format`: `mp3` (default), `opus`, `flac`, `wav` or `pcm`
//!   (16-bit little-endian mono at 24 kHz); `aac` is refused (no
//!   permissively licensed encoder);
//! - `speed`: 0.25 to 4.0 (durations are divided by it, as Kokoro does);
//! - `stream_format`: `audio` (default; the encoded bytes, chunked as each
//!   text chunk is synthesized) or `sse` (`speech.audio.delta` events with
//!   base64 bytes of the same encoded stream, then `speech.audio.done` with
//!   usage: phonemes in, 24 kHz frames of 600 samples out).
//!
//! # Scheduling
//!
//! Synthesis runs on its own Metal queue, one request at a time, at low
//! priority: it submits bounded command buffers and, while Bonsai generations
//! are running, waits for the engine worker to finish a decode step between
//! them ([`EngineActivity::yield_to_decode`]). At most [`MAX_PENDING`]
//! requests are admitted (validating, running or waiting); more are answered
//! `503` with `Retry-After` before their input is phonemized. A client that goes away cancels its request at the next
//! submission boundary.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use base64::Engine as _;
use bytes::Bytes;
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use local_engine::activity::EngineActivity;
use local_engine::speech::encode::{Format, StreamEncoder};
use local_engine::speech::{
    Catalog, MAX_INPUT_CHARS, Noise, SAMPLE_RATE, SPEECH_MODEL_ID, SPEED_RANGE, SpeechModel,
    VOICE_ALIASES, installed_model,
};

use super::admin::{self, Caller, Meter};
use super::response::{add_alt_svc, error_response, json_response};
use super::{AppState, MAX_REQUEST_BYTES};

pub(super) const ROUTE: &str = "/v1/audio/speech";

/// Speech requests admitted at once (one runs, the rest wait in order).
pub(super) const MAX_PENDING: usize = 4;
/// Longest wait for one decode step before the next speech submission goes
/// ahead anyway, so speech is slowed, never starved.
const DECODE_YIELD_LIMIT: Duration = Duration::from_millis(500);
const RETRY_AFTER_SECONDS: u32 = 1;
/// Samples per acoustic frame (the usage's output unit).
const FRAME_SAMPLES: usize = 600;

pub(super) struct Speech {
    model: Arc<Mutex<SpeechModel>>,
    catalog: Catalog,
    slots: Arc<Semaphore>,
    activity: Arc<EngineActivity>,
    requests: AtomicU64,
}

impl Speech {
    /// The service, from the required install in `models_dir`.
    pub(super) fn start(
        models_dir: &Path,
        activity: Arc<EngineActivity>,
    ) -> crate::Result<Arc<Self>> {
        let dir = installed_model(models_dir)?;
        let model = SpeechModel::open(&dir).map_err(|error| {
            super::artifact_error(format!(
                "cannot load the speech model in {}: {error}",
                dir.display()
            ))
        })?;
        eprintln!(
            "speech at POST {ROUTE}: {SPEECH_MODEL_ID} from {} (F32, payload sha256 {})",
            dir.display(),
            model.payload_sha256()
        );
        Ok(Arc::new(Self {
            catalog: model.catalog(),
            model: Arc::new(Mutex::new(model)),
            slots: Arc::new(Semaphore::new(MAX_PENDING)),
            activity,
            requests: AtomicU64::new(0),
        }))
    }
}

/// `/v1/models` entry of the speech model.
pub(super) fn model_json(created: u64) -> Value {
    json!({"id":SPEECH_MODEL_ID,"object":"model","created":created,"owned_by":"local"})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    model: String,
    input: String,
    voice: Value,
    instructions: Option<String>,
    response_format: Option<String>,
    speed: Option<f64>,
    stream_format: Option<String>,
}

#[derive(Debug)]
struct Parsed {
    input: String,
    voice: String,
    format: Format,
    speed: f32,
    sse: bool,
}

fn parse(body: &[u8]) -> Result<Parsed, String> {
    let request: RequestBody =
        serde_json::from_slice(body).map_err(|error| format!("invalid speech request: {error}"))?;
    if super::models::resolve(&request.model, super::models::BONSAI_MODEL_ID)
        != Some(super::models::Served::Speech)
    {
        return Err(format!(
            "model {:?} is not served here; use {SPEECH_MODEL_ID:?}",
            request.model
        ));
    }
    if request.input.trim().is_empty() {
        return Err("input must not be empty".into());
    }
    if request.input.chars().count() > MAX_INPUT_CHARS {
        return Err(format!(
            "input holds more than {MAX_INPUT_CHARS} characters"
        ));
    }
    if request
        .instructions
        .as_deref()
        .is_some_and(|i| !i.trim().is_empty())
    {
        return Err(format!(
            "instructions are not supported: {SPEECH_MODEL_ID} has no instruction or style-prompt \
             input, so it cannot honor them; choose a voice and speed instead"
        ));
    }
    let voice = match &request.voice {
        Value::String(name) => name.clone(),
        Value::Object(map) if map.len() == 1 => match map.get("id") {
            Some(Value::String(id)) => id.clone(),
            _ => return Err("a custom voice object must be {\"id\": string}".into()),
        },
        _ => return Err("voice must be a voice name or {\"id\": string}".into()),
    };
    let format_name = request.response_format.as_deref().unwrap_or("mp3");
    let format = Format::parse(format_name).ok_or_else(|| {
        format!("response_format must be mp3, opus, aac, flac, wav or pcm, not {format_name:?}")
    })?;
    if !format.supported() {
        return Err(
            "response_format aac is not supported: no permissively licensed AAC encoder is \
             available; use mp3, opus, flac, wav or pcm"
                .into(),
        );
    }
    let speed = request.speed.unwrap_or(1.0);
    if !(f64::from(SPEED_RANGE.0)..=f64::from(SPEED_RANGE.1)).contains(&speed) {
        return Err(format!(
            "speed must be between {} and {}",
            SPEED_RANGE.0, SPEED_RANGE.1
        ));
    }
    let sse = match request.stream_format.as_deref() {
        None | Some("audio") => false,
        Some("sse") => true,
        Some(other) => return Err(format!("stream_format must be audio or sse, not {other:?}")),
    };
    Ok(Parsed {
        input: request.input,
        voice,
        format,
        speed: speed as f32,
        sse,
    })
}

/// Sets the flag when dropped: the handler or its body went away.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn sse_event(value: &Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

/// Synthesize on the blocking pool, sending encoded bytes (or SSE events)
/// as they are ready. Returns an error message for the first failure.
fn synthesize(
    service: &Speech,
    parsed: &Parsed,
    voice: &str,
    seed: u64,
    cancelled: &AtomicBool,
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    let model = service.model.lock().unwrap_or_else(PoisonError::into_inner);
    let activity = Arc::clone(&service.activity);
    let mut pace = || {
        if cancelled.load(Ordering::SeqCst) {
            return Err(local_engine::Error::Generation(
                "speech request cancelled".into(),
            ));
        }
        activity.yield_to_decode(DECODE_YIELD_LIMIT);
        Ok(())
    };
    let mut encoder = StreamEncoder::new(parsed.format, SAMPLE_RATE).map_err(|e| e.to_string())?;
    let emit = |bytes: Vec<u8>| -> local_engine::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let frame = if parsed.sse {
            sse_event(&json!({
                "type":"speech.audio.delta",
                "audio":base64::engine::general_purpose::STANDARD.encode(&bytes),
            }))
        } else {
            Bytes::from(bytes)
        };
        sender.blocking_send(Ok(frame)).map_err(|_| {
            cancelled.store(true, Ordering::SeqCst);
            local_engine::Error::Generation("speech client went away".into())
        })
    };
    let usage = {
        let mut sink =
            |samples: &[f32]| -> local_engine::Result<()> { emit(encoder.push(samples)?) };
        model.synthesize(
            &parsed.input,
            voice,
            parsed.speed,
            Noise::Seeded(seed),
            &mut pace,
            &mut sink,
        )
    }
    .map_err(|e| e.to_string())?;
    drop(model);
    emit(encoder.finish().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if parsed.sse {
        let output = usage.samples / FRAME_SAMPLES;
        let done = sse_event(&json!({
            "type":"speech.audio.done",
            "usage":{"input_tokens":usage.phonemes,"output_tokens":output,
                     "total_tokens":usage.phonemes + output},
        }));
        sender
            .blocking_send(Ok(done))
            .map_err(|_| "speech client went away".to_owned())?;
    }
    Ok(())
}

async fn busy(state: &AppState, headers: &HeaderMap) -> Response {
    let mut response = json_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"error":{
            "message":format!(
                "{MAX_PENDING} speech requests are already running or waiting; retry after \
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

pub(super) async fn handle(
    state: &AppState,
    headers: &HeaderMap,
    body: Body,
    caller: &Caller,
) -> Response {
    let service = Arc::clone(&state.speech);
    let body = match http_body_util::Limited::new(body, MAX_REQUEST_BYTES)
        .collect()
        .await
    {
        Ok(collected) => collected.to_bytes(),
        Err(error) => {
            let status = if error.is::<http_body_util::LengthLimitError>() {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            return error_response(
                status,
                &format!("invalid HTTP body: {error}"),
                state,
                headers,
            )
            .await;
        }
    };
    let parsed = match parse(&body) {
        Ok(parsed) => parsed,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, state, headers).await;
        }
    };
    // Admitted first: phonemizing the input to validate it is CPU work
    // (bounded per request, but not in number without the slot).
    let Ok(permit) = Arc::clone(&service.slots).try_acquire_owned() else {
        return busy(state, headers).await;
    };
    // Validate on the blocking pool without the model lock (a running
    // synthesis holds it), so bad requests fail fast with 400.
    let catalog = service.catalog.clone();
    let (input, requested) = (parsed.input.clone(), parsed.voice.clone());
    let voice = tokio::task::spawn_blocking(move || match catalog.resolve_voice(&requested) {
        Some(voice) if catalog.chunks(&input, voice, 510).is_empty() => {
            Err("the input has nothing to pronounce".to_owned())
        }
        Some(voice) => Ok(voice.to_owned()),
        None => Err(format!(
            "unknown voice {requested:?}: use a Kokoro voice id ({}) or one of {}",
            catalog.voices().join(", "),
            VOICE_ALIASES
                .iter()
                .map(|(a, _)| *a)
                .collect::<Vec<_>>()
                .join(", ")
        )),
    })
    .await
    .unwrap_or_else(|error| Err(error.to_string()));
    let voice = match voice {
        Ok(voice) => voice,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, state, headers).await;
        }
    };
    let seed = service.requests.fetch_add(1, Ordering::Relaxed);
    let cancelled = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
    let content_type = if parsed.sse {
        "text/event-stream"
    } else {
        parsed.format.content_type()
    };
    let worker_cancelled = Arc::clone(&cancelled);
    // Recorded once the whole input has been synthesized.
    let meter = Meter::into_store(
        state.auth.admin(),
        admin::speech_usage(caller, SPEECH_MODEL_ID, &parsed.input),
    );
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _guard = CancelOnDrop(Arc::clone(&worker_cancelled));
        match synthesize(&service, &parsed, &voice, seed, &worker_cancelled, &sender) {
            Ok(()) => meter.write(),
            Err(message) if !worker_cancelled.load(Ordering::SeqCst) => {
                eprintln!("speech request failed: {message}");
                let _ = sender.blocking_send(Err(std::io::Error::other(message)));
            }
            Err(_) => {}
        }
    });
    let mut response = Response::new(Body::from_stream(CancelStream {
        inner: ReceiverStream::new(receiver),
        _guard: CancelOnDrop(cancelled),
    }));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static(content_type),
    );
    if parsed_is_sse(content_type) {
        response.headers_mut().insert(
            header::CACHE_CONTROL,
            header::HeaderValue::from_static("no-cache"),
        );
    }
    add_alt_svc(&mut response, state);
    response
}

fn parsed_is_sse(content_type: &str) -> bool {
    content_type == "text/event-stream"
}

/// The response body: cancels the synthesis when dropped (client gone).
struct CancelStream {
    inner: ReceiverStream<Result<Bytes, std::io::Error>>,
    _guard: CancelOnDrop,
}

impl tokio_stream::Stream for CancelStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(extra: &str) -> Vec<u8> {
        format!(r#"{{"model":"kokoro-82m","input":"Hello there.","voice":"af_heart"{extra}}}"#)
            .into_bytes()
    }

    #[test]
    fn defaults_follow_the_pinned_request() {
        let parsed = parse(&request("")).expect("valid");
        assert_eq!(parsed.format, Format::Mp3);
        assert!((parsed.speed - 1.0).abs() < f32::EPSILON);
        assert!(!parsed.sse);
        assert_eq!(parsed.voice, "af_heart");
    }

    #[test]
    fn fields_are_parsed_and_checked() {
        let parsed = parse(&request(
            r#","response_format":"wav","speed":1.5,"stream_format":"sse""#,
        ))
        .expect("valid");
        assert_eq!(parsed.format, Format::Wav);
        assert!(parsed.sse);
        let custom = parse(br#"{"model":"kokoro-82m","input":"Hi.","voice":{"id":"bm_george"}}"#)
            .expect("custom voice object");
        assert_eq!(custom.voice, "bm_george");
        for bad in [
            r#","response_format":"aac""#,
            r#","response_format":"ogg""#,
            r#","speed":5"#,
            r#","speed":0.1"#,
            r#","stream_format":"chunks""#,
            r#","instructions":"Speak cheerfully.""#,
            r#","unknown":1"#,
        ] {
            assert!(parse(&request(bad)).is_err(), "{bad}");
        }
        assert!(parse(&request(r#","instructions":"""#)).is_ok());
        // OpenAI's speech model names are aliases of the local model.
        assert!(parse(br#"{"model":"tts-1","input":"Hi.","voice":"alloy"}"#).is_ok());
        assert!(parse(br#"{"model":"whisper-1","input":"Hi.","voice":"alloy"}"#).is_err());
        assert!(parse(br#"{"model":"kokoro-82m","input":" ","voice":"alloy"}"#).is_err());
        let long = format!(
            r#"{{"model":"kokoro-82m","input":"{}","voice":"alloy"}}"#,
            "a".repeat(MAX_INPUT_CHARS + 1)
        );
        assert!(parse(long.as_bytes()).is_err());
    }
}
