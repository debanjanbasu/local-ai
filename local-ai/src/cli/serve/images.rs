//! Image parts of Chat Completions and Responses messages.
//!
//! An image arrives as a `data:` URL (base64), or as an uploaded file named
//! by `file_id` (the Files API store). Remote `http`/`https` URLs are refused:
//! the server never fetches a URL a request names on the request's behalf;
//! clients send the bytes or upload them first. Each image keeps its place
//! among the text parts, and its `OpenAI` `detail` (`low`, `auto`, `high`)
//! sets its pixel budget ([`ImageDetail`]). The engine decodes, resizes and
//! encodes it; it needs the vision projector installed.

use std::io::Read as _;
use std::sync::Arc;

use base64::Engine as _;
use local_engine::{ChatImage, ImageDetail};
use local_services::Store;
use serde_json::Value;

/// Largest image file accepted, encoded.
const MAX_IMAGE_BYTES: u64 = 32 << 20;

fn invalid(message: impl Into<String>) -> crate::Error {
    crate::Error::InvalidArgument(message.into())
}

/// Message text and the images placed in it.
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
        });
    }
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
                return Err(invalid("image data URLs must be base64 encoded"));
            }
            payload
        }
        None if bare => url,
        None => return Err(invalid("not a data URL")),
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|error| invalid(format!("image data is not valid base64: {error}")))?;
    if bytes.len() as u64 > MAX_IMAGE_BYTES {
        return Err(invalid(format!(
            "an image may hold at most {MAX_IMAGE_BYTES} bytes"
        )));
    }
    Ok(bytes.into())
}

/// The image an `image_url` names: a data URL. Remote URLs are refused.
pub(super) fn url(url: &str) -> crate::Result<Arc<[u8]>> {
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(invalid(
            "remote image URLs are not fetched; send the image as a base64 data URL or upload \
             it to /v1/files and refer to its file_id",
        ));
    }
    data_url(url, false).map_err(|error| match error {
        crate::Error::InvalidArgument(message) if message == "not a data URL" => {
            invalid("an image URL must be a base64 data URL")
        }
        other => other,
    })
}

/// The bytes of uploaded file `file_id`.
pub(super) fn file(files: Option<&Store>, file_id: &str) -> crate::Result<Arc<[u8]>> {
    let files = files.ok_or_else(|| {
        invalid("file_id images need the Files API, which this server was started without")
    })?;
    let mut content = files
        .open_file_content(file_id)
        .map_err(|error| invalid(format!("image file {file_id:?}: {error}")))?;
    if content.file.bytes > MAX_IMAGE_BYTES {
        return Err(invalid(format!(
            "image file {file_id:?} holds {} bytes; the limit is {MAX_IMAGE_BYTES}",
            content.file.bytes
        )));
    }
    let mut bytes = Vec::with_capacity(content.file.bytes as usize);
    content
        .content
        .read_to_end(&mut bytes)
        .map_err(|error| crate::Error::Generation(format!("image file {file_id:?}: {error}")))?;
    Ok(bytes.into())
}
