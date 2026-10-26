//! Image and video parts of Chat Completions and Responses messages.
//!
//! A part arrives as a `data:` URL (base64), an `http`/`https` URL the
//! server fetches, or an uploaded file named by `file_id` (the Files API
//! store). Each part keeps its place among the text parts; an image's
//! `OpenAI` `detail` (`low`, `auto`, `high`) sets its pixel budget
//! ([`ImageDetail`]). The engine decodes, resizes and encodes them; it needs
//! the vision projector installed.
//!
//! Remote URLs go through the hosted fetch tool's transport
//! ([`local_services::tools::fetch::download`]) without its allowlist: any
//! public host, never a loopback, private, link-local or other non-public
//! address (the address connected to is the one checked), at most five
//! redirects (each checked anew), a 60-second deadline, at most 64 MiB (the
//! generation body limit), and an `image/*` or `video/*` content type (or
//! none, or `application/octet-stream`). Fetched bytes are cached by URL for
//! ten minutes, so a conversation that replays a URL in its history fetches
//! it once.
//!
//! One request's parts are bounded together ([`request_scope`]): at most
//! [`MAX_REQUEST_PARTS`] parts and [`MAX_REQUEST_MEDIA_BYTES`], each
//! `file_id` and URL read once, and all remote fetches done within
//! [`REQUEST_FETCH_TIMEOUT`].

use std::cell::RefCell;
use std::future::Future;
use std::io::Read as _;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use base64::Engine as _;
use local_engine::bonsai_vision::{MAX_REQUEST_MEDIA_BYTES, MAX_REQUEST_PARTS};
use local_engine::{ChatImage, ImageDetail};
use local_services::Store;
use serde_json::Value;

/// Largest image or video accepted, encoded: the generation body limit.
const MAX_MEDIA_BYTES: usize = 64 << 20;
/// Deadline of one remote fetch, redirects included.
const FETCH_TIMEOUT: Duration = Duration::from_secs(60);
/// Deadline of all of one request's remote fetches together.
const REQUEST_FETCH_TIMEOUT: Duration = Duration::from_secs(120);
/// How long a fetched URL's bytes are reused.
const FETCH_FRESH: Duration = Duration::from_secs(600);
/// Fetched bytes kept for reuse.
const FETCH_CACHE_BYTES: usize = 256 << 20;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidArgument(message.into())
}

/// What a part holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Image,
    Video,
}

impl Kind {
    /// The kind with its article.
    const fn noun(self) -> &'static str {
        match self {
            Self::Image => "an image",
            Self::Video => "a video",
        }
    }
}

/// Message text and the images and videos placed in it.
#[derive(Default)]
pub(super) struct Parts {
    pub(super) text: String,
    pub(super) images: Vec<ChatImage>,
}

impl Parts {
    pub(super) fn push_text(&mut self, text: &str) {
        self.text.push_str(text);
    }

    pub(super) fn push_image(&mut self, data: Arc<[u8]>, detail: ImageDetail) {
        self.images.push(ChatImage {
            offset: self.text.len(),
            data,
            detail,
            video: false,
        });
    }

    pub(super) fn push_video(&mut self, data: Arc<[u8]>) {
        self.images.push(ChatImage {
            offset: self.text.len(),
            data,
            detail: ImageDetail::Auto,
            video: true,
        });
    }

    /// An uploaded or inline file: a video when it is one, else an image.
    pub(super) fn push_file(&mut self, data: Arc<[u8]>, detail: ImageDetail) {
        if local_engine::bonsai_vision::video::is_video(&data) {
            self.push_video(data);
        } else {
            self.push_image(data, detail);
        }
    }
}

/// What one request's parts have read so far.
#[derive(Default)]
struct Scope {
    /// Each `file_id` and URL read, by `file:<id>` or the URL, with the
    /// response's content type.
    sources: Vec<(String, Option<String>, Arc<[u8]>)>,
    parts: usize,
    bytes: usize,
    /// When the request's remote fetches must be done, from its first.
    deadline: Option<Instant>,
}

impl Scope {
    /// Count one more part of `bytes`.
    fn charge(&mut self, bytes: usize) -> crate::Result<()> {
        self.parts += 1;
        self.bytes += bytes;
        if self.parts > MAX_REQUEST_PARTS {
            return Err(invalid(format!(
                "a request may hold at most {MAX_REQUEST_PARTS} images and videos"
            )));
        }
        if self.bytes > MAX_REQUEST_MEDIA_BYTES {
            return Err(invalid(format!(
                "a request's images and videos may hold at most {MAX_REQUEST_MEDIA_BYTES} bytes \
                 together"
            )));
        }
        Ok(())
    }

    fn source(&self, key: &str) -> Option<(Option<String>, Arc<[u8]>)> {
        self.sources
            .iter()
            .find(|(source, ..)| source == key)
            .map(|(_, content_type, body)| (content_type.clone(), Arc::clone(body)))
    }
}

thread_local! {
    /// The scope of the request this thread is parsing, if any.
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

/// Parse one request's messages with `parse`, its parts bounded together.
///
/// Request parsing is synchronous on one thread, so the scope is that
/// thread's for the call; a nested call joins the outer scope. Parts read
/// outside any scope are bounded one at a time only.
pub(super) fn request_scope<T>(parse: impl FnOnce() -> T) -> T {
    struct Close;
    impl Drop for Close {
        fn drop(&mut self) {
            SCOPE.with(|scope| scope.borrow_mut().take());
        }
    }
    if SCOPE.with(|scope| scope.borrow().is_some()) {
        return parse();
    }
    SCOPE.with(|scope| *scope.borrow_mut() = Some(Scope::default()));
    let _close = Close;
    parse()
}

/// Run `f` on the current request's scope (a scope of its own outside one).
fn with_scope<T>(f: impl FnOnce(&mut Scope) -> T) -> T {
    SCOPE.with(|scope| match scope.borrow_mut().as_mut() {
        Some(scope) => f(scope),
        None => f(&mut Scope::default()),
    })
}

/// An image `detail`; absent (or `null`) is `auto`.
pub(super) fn detail(value: Option<&Value>) -> crate::Result<ImageDetail> {
    match value {
        None | Some(Value::Null) => Ok(ImageDetail::Auto),
        Some(Value::String(detail)) => ImageDetail::parse(detail),
        Some(_) => Err(invalid("image detail must be low, auto or high")),
    }
}

/// The bytes of a `data:` URL (`data:[<type>][;base64],<data>`; base64
/// only), or of bare base64 when `bare` is allowed.
pub(super) fn data_url(url: &str, bare: bool) -> crate::Result<Arc<[u8]>> {
    let payload = match url.strip_prefix("data:") {
        Some(rest) => {
            let (header, payload) = rest
                .split_once(',')
                .ok_or_else(|| invalid("a data URL needs a comma before its data"))?;
            if !header.ends_with(";base64") {
                return Err(invalid("media data URLs must be base64 encoded"));
            }
            payload
        }
        None if bare => url,
        None => return Err(invalid("not a data URL")),
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|error| invalid(format!("media data is not valid base64: {error}")))?;
    if bytes.len() > MAX_MEDIA_BYTES {
        return Err(invalid(format!(
            "an image or video may hold at most {MAX_MEDIA_BYTES} bytes"
        )));
    }
    with_scope(|scope| scope.charge(bytes.len()))?;
    Ok(bytes.into())
}

/// The bytes a part's URL names: a base64 data URL, or a remote `http(s)`
/// URL, fetched.
pub(super) fn url(url: &str, kind: Kind) -> crate::Result<Arc<[u8]>> {
    if url.starts_with("http://") || url.starts_with("https://") {
        return remote(url, kind);
    }
    data_url(url, false).map_err(|error| match error {
        crate::Error::InvalidArgument(message) if message == "not a data URL" => invalid(format!(
            "{} URL must be a base64 data URL or an http(s) URL",
            kind.noun()
        )),
        other => other,
    })
}

/// The bytes of uploaded file `file_id`, read once per request.
pub(super) fn file(files: Option<&Store>, file_id: &str) -> crate::Result<Arc<[u8]>> {
    let files = files.ok_or_else(|| {
        invalid("file_id parts need the Files API, which this server was started without")
    })?;
    let key = format!("file:{file_id}");
    if let Some(body) = with_scope(|scope| {
        let (_, body) = scope.source(&key)?;
        Some(scope.charge(body.len()).map(|()| body))
    }) {
        return body;
    }
    let mut content = files
        .open_file_content(file_id)
        .map_err(|error| invalid(format!("file {file_id:?}: {error}")))?;
    if content.file.bytes > MAX_MEDIA_BYTES as u64 {
        return Err(invalid(format!(
            "file {file_id:?} holds {} bytes; an image or video may hold at most \
             {MAX_MEDIA_BYTES}",
            content.file.bytes
        )));
    }
    // Refused before reading when the request would pass its total.
    with_scope(|scope| scope.charge(content.file.bytes as usize))?;
    let mut bytes = Vec::with_capacity(content.file.bytes as usize);
    content
        .content
        .read_to_end(&mut bytes)
        .map_err(|error| crate::Error::Generation(format!("file {file_id:?}: {error}")))?;
    let body: Arc<[u8]> = bytes.into();
    with_scope(|scope| scope.sources.push((key, None, Arc::clone(&body))));
    Ok(body)
}

/// One fetched URL.
struct Fetched {
    url: String,
    at: Instant,
    content_type: Option<String>,
    body: Arc<[u8]>,
}

/// Recently fetched URLs, oldest first.
static FETCHED: Mutex<Vec<Fetched>> = Mutex::new(Vec::new());

/// The body of remote `url`, which must hold a `kind`, from the cache when
/// it was fetched in the last ten minutes (or earlier in this request).
fn remote(url: &str, kind: Kind) -> crate::Result<Arc<[u8]>> {
    let cached = with_scope(|scope| scope.source(url)).or_else(|| {
        let mut cache = FETCHED.lock().unwrap_or_else(PoisonError::into_inner);
        cache.retain(|entry| entry.at.elapsed() < FETCH_FRESH);
        cache
            .iter()
            .find(|entry| entry.url == url)
            .map(|entry| (entry.content_type.clone(), Arc::clone(&entry.body)))
    });
    let (content_type, body) = if let Some(hit) = cached {
        hit
    } else {
        // One deadline for all of the request's fetches.
        let timeout = with_scope(|scope| {
            let deadline = *scope
                .deadline
                .get_or_insert_with(|| Instant::now() + REQUEST_FETCH_TIMEOUT);
            deadline.saturating_duration_since(Instant::now())
        })
        .min(FETCH_TIMEOUT);
        if timeout.is_zero() {
            return Err(invalid(format!(
                "cannot fetch {url}: the request's media fetches took over {}s",
                REQUEST_FETCH_TIMEOUT.as_secs()
            )));
        }
        let download = block_on(async {
            tokio::time::timeout(
                timeout,
                local_services::tools::fetch::download(url, MAX_MEDIA_BYTES),
            )
            .await
        })?
        .map_err(|_| {
            invalid(format!(
                "cannot fetch {url}: no complete answer within {}s",
                timeout.as_secs()
            ))
        })?
        .map_err(|error| invalid(format!("cannot fetch {url}: {error}")))?;
        let body: Arc<[u8]> = download.body.into();
        let mut cache = FETCHED.lock().unwrap_or_else(PoisonError::into_inner);
        cache.retain(|entry| entry.url != url);
        cache.push(Fetched {
            url: url.to_owned(),
            at: Instant::now(),
            content_type: download.content_type.clone(),
            body: Arc::clone(&body),
        });
        let mut bytes = cache.iter().map(|entry| entry.body.len()).sum::<usize>();
        while cache.len() > 1 && bytes > FETCH_CACHE_BYTES {
            bytes -= cache.remove(0).body.len();
        }
        drop(cache);
        (download.content_type, body)
    };
    check_content_type(url, content_type.as_deref(), kind)?;
    with_scope(|scope| {
        scope.charge(body.len())?;
        if scope.source(url).is_none() {
            scope
                .sources
                .push((url.to_owned(), content_type, Arc::clone(&body)));
        }
        Ok::<_, crate::Error>(())
    })?;
    Ok(body)
}

/// Refuse a response whose `Content-Type` says it is not a `kind`. Absent
/// and generic binary types pass; the decoder decides.
fn check_content_type(url: &str, content_type: Option<&str>, kind: Kind) -> crate::Result<()> {
    let essence = content_type
        .unwrap_or_default()
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let matches = match kind {
        Kind::Image => essence.starts_with("image/"),
        Kind::Video => essence.starts_with("video/") || essence == "application/mp4",
    };
    if matches
        || matches!(
            essence.as_str(),
            "" | "application/octet-stream" | "binary/octet-stream"
        )
    {
        return Ok(());
    }
    Err(invalid(format!(
        "{url} answered with content type {essence}, not {}",
        kind.noun()
    )))
}

/// Run `future` to completion from synchronous request parsing, which runs
/// on the blocking pool (or, in tests, anywhere): on a thread of its own
/// with a private runtime, so it never needs the caller's.
fn block_on<T: Send>(future: impl Future<Output = T> + Send) -> crate::Result<T> {
    let run = || {
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map(|runtime| runtime.block_on(future))
                        .map_err(|error| crate::Error::Generation(error.to_string()))
                })
                .join()
                .unwrap_or_else(|_| Err(crate::Error::Generation("media fetch panicked".into())))
        })
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(run)
        }
        _ => run(),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use super::{
        Kind, MAX_MEDIA_BYTES, MAX_REQUEST_PARTS, check_content_type, data_url, file,
        request_scope, url,
    };

    /// One request reads a `file_id` once however often it names it, and
    /// its parts are bounded together: the fifth reference to a 64 MiB file
    /// passes 256 MiB, and the 65th part passes the part limit.
    #[test]
    fn a_requests_media_are_read_once_and_bounded_together() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let root = dir
            .path()
            .canonicalize()
            .expect("canonical")
            .join("services");
        let store = local_services::Store::open(&root).expect("store");
        let id = store
            .create_file(
                &local_services::files::CreateFile {
                    filename: "big.png".into(),
                    purpose: local_services::files::FilePurpose::Vision,
                    expires_after: None,
                },
                std::io::Read::take(std::io::repeat(7), MAX_MEDIA_BYTES as u64),
            )
            .expect("upload")
            .id;
        let error = request_scope(|| {
            let first = file(Some(&store), &id).expect("read");
            for _ in 0..3 {
                let again = file(Some(&store), &id).expect("again");
                assert!(Arc::ptr_eq(&first, &again), "read once");
            }
            file(Some(&store), &id).expect_err("over 256 MiB")
        });
        assert!(error.to_string().contains("bytes together"), "{error}");
        // Outside the scope (and in the next request) each part stands alone.
        drop(file(Some(&store), &id).expect("a new request"));

        let small = "data:image/png;base64,AAAA";
        let error = request_scope(|| {
            for _ in 0..MAX_REQUEST_PARTS {
                data_url(small, false).expect("within");
            }
            data_url(small, false).expect_err("one part too many")
        });
        assert!(error.to_string().contains("at most 64"), "{error}");
    }

    #[test]
    fn remote_media_types_must_match_the_part() {
        let ok = |content_type, kind| check_content_type("u", content_type, kind).is_ok();
        assert!(ok(Some("image/png"), Kind::Image));
        assert!(ok(Some("IMAGE/JPEG; charset=binary"), Kind::Image));
        assert!(ok(Some("video/mp4"), Kind::Video));
        assert!(ok(Some("application/octet-stream"), Kind::Video));
        assert!(ok(None, Kind::Image));
        assert!(!ok(Some("text/html; charset=utf-8"), Kind::Image));
        assert!(!ok(Some("video/mp4"), Kind::Image));
        assert!(!ok(Some("image/png"), Kind::Video));
    }

    #[test]
    fn non_public_media_urls_are_refused_before_any_request() {
        for address in [
            "http://127.0.0.1:9/a.png",
            "http://localhost:9/a.png",
            "https://[::1]/a.png",
            "http://169.254.169.254/latest",
            "http://10.0.0.1/v.mp4",
        ] {
            let error = url(address, Kind::Image).expect_err("refused");
            assert!(
                error.to_string().contains("non-public"),
                "{address}: {error}"
            );
        }
    }
}
