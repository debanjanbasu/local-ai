//! Image input for Bonsai 2 27B: the checkpoint's Qwen3.5 vision tower and
//! the decoder's multimodal positions.
//!
//! Prism ships the 0.46B, 27-block vision tower as a separate mmproj
//! (`Ternary-Bonsai-2-27B-mmproj-BF16.gguf`, the BF16 reference export). When
//! it is installed beside the GGUF it is hash-verified and loaded at startup
//! ([`VisionEncoder::open`]); image parts of a request then become
//! `<|vision_start|>` + one `<|image_pad|>` per merged 2x2 patch block +
//! `<|vision_end|>`, exactly as the checkpoint's chat template and processor
//! lay them out, and the tower's 5120-wide outputs replace those pads'
//! embeddings.
//!
//! - Preprocessing ([`preprocess`]) is `Qwen2VLImageProcessorFast`'s:
//!   `smart_resize` to multiples of 32 within the pixel budget of the
//!   request's `detail`, torch's antialiased bicubic `uint8` resize, then
//!   `(x - 127.5) / 127.5`.
//! - The tower ([`tower`]) runs on Metal in F32 with the BF16 weights as
//!   shipped: bilinearly interpolated learned positions, 27 pre-LN blocks with
//!   2-D `RoPE` and bidirectional attention, then the 2x2 merger.
//! - Image tokens take the interleaved M-RoPE positions of the processor's
//!   `get_rope_index` ([`PromptImages::rope`]): token `k` of an image whose
//!   first pad sits at text position `t0` rotates by `(t0, t0 + k / cols,
//!   t0 + k % cols)`, and the text after it resumes at `t0 + max(rows, cols)`.
//!
//! Video parts ([`video`]) become frame groups of two frames, each encoded
//! like an image and laid out like one after its timestamp text.
//!
//! Prompt caching keys on token ids, so each image token gets a virtual id
//! above the vocabulary ([`image_token_id`]) derived from the mmproj and the
//! preprocessed pixels: equal images share prefixes, different ones never do.

mod preprocess;
mod tower;
pub mod video;

use std::sync::{Arc, Mutex, PoisonError};

use sha2::{Digest as _, Sha256};

use crate::bonsai::VOCAB;
use crate::bonsai::WIDTH;

pub use self::preprocess::{ImageDetail, ImageFile, PreparedImage, prepare_decoded, prepare_image};
pub use self::tower::{VISION_PROJECTOR_FILE, VisionEncoder};
pub use self::video::{PreparedVideo, VIDEO_PAD, VIDEO_PLACEHOLDER, prepare_video};

/// `<|vision_start|>`.
pub const VISION_START: u32 = 248_053;
/// `<|vision_end|>`.
pub const VISION_END: u32 = 248_054;
/// `<|image_pad|>`: the template writes one per image; the processor expands
/// it to one per image token.
pub const IMAGE_PAD: u32 = 248_056;
/// What the chat template renders for each image part.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Most image and video parts one request may hold.
pub const MAX_REQUEST_PARTS: usize = 64;
/// Most encoded bytes of one request's images and videos together.
pub const MAX_REQUEST_MEDIA_BYTES: usize = 256 << 20;
/// Most decoded pixels of one request's images together (512 MP; each image
/// is decoded and resized on its own, so this bounds CPU time, not memory).
pub const MAX_REQUEST_IMAGE_PIXELS: usize = 512 << 20;
/// Most track frames of one request's videos together (two videos at the
/// per-video limit), checked as each video is read.
pub const MAX_REQUEST_VIDEO_FRAMES: usize = 2 * video::MAX_TRACK_FRAMES;

/// The error of work stopped because its request was cancelled.
pub(crate) fn cancelled() -> crate::Error {
    crate::Error::Generation("the request was cancelled".into())
}

/// First virtual image-token id: everything at or above it is an image token.
pub const IMAGE_TOKEN_BASE: u32 = VOCAB as u32;

/// Whether `token` is a virtual image-token id.
#[must_use]
pub const fn is_image_token(token: u32) -> bool {
    token >= IMAGE_TOKEN_BASE
}

/// The virtual id of token `index` of an image, above the vocabulary.
///
/// A hash of the projector, the image digest and the index: identity for
/// prompt caching only; the embedding comes from the image span the token
/// sits in.
#[must_use]
pub fn image_token_id(projector: &[u8; 32], image: &[u8; 32], index: u32) -> u32 {
    let digest = Sha256::new()
        .chain_update(b"bonsai-vision-v1")
        .chain_update(projector)
        .chain_update(image)
        .chain_update(index.to_le_bytes())
        .finalize();
    let value = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    IMAGE_TOKEN_BASE + value % (u32::MAX - IMAGE_TOKEN_BASE)
}

/// One image's tokens in a prompt.
#[derive(Clone, Debug)]
pub struct ImageSpan {
    /// Prompt position of the first image token.
    pub start: usize,
    /// Merged grid rows and columns: `rows * cols` tokens.
    pub rows: usize,
    pub cols: usize,
    /// `[rows * cols][WIDTH]` decoder input rows, in the checkpoint's
    /// signed-Hadamard embedding basis (as `token_embd` rows are stored).
    pub embedding: Arc<[f32]>,
}

impl ImageSpan {
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.rows * self.cols
    }

    const fn end(&self) -> usize {
        self.start + self.tokens()
    }

    /// How many positions the text after this image resumes short of its
    /// token count.
    const fn position_shrink(&self) -> usize {
        let extent = if self.rows > self.cols {
            self.rows
        } else {
            self.cols
        };
        self.tokens() - extent
    }
}

/// The images of one sequence's prompt, by position.
#[derive(Clone, Debug, Default)]
pub struct PromptImages {
    spans: Vec<ImageSpan>,
}

impl PromptImages {
    /// # Errors
    ///
    /// [`crate::Error::InvalidArgument`] for overlapping, unordered, empty or
    /// wrongly sized spans.
    pub fn new(spans: Vec<ImageSpan>) -> crate::Result<Self> {
        let mut end = 0;
        for span in &spans {
            if span.start < end
                || span.tokens() == 0
                || span.embedding.len() != span.tokens() * WIDTH
            {
                return Err(crate::Error::InvalidArgument(
                    "invalid prompt image spans".into(),
                ));
            }
            end = span.end();
        }
        Ok(Self { spans })
    }

    #[must_use]
    pub fn spans(&self) -> &[ImageSpan] {
        &self.spans
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    fn span_at(&self, position: usize) -> Option<&ImageSpan> {
        let index = self.spans.partition_point(|span| span.end() <= position);
        self.spans.get(index).filter(|span| span.start <= position)
    }

    /// The decoder input row of the image token at `position`.
    #[must_use]
    pub fn embedding(&self, position: usize) -> Option<&[f32]> {
        let span = self.span_at(position)?;
        let row = position - span.start;
        Some(&span.embedding[row * WIDTH..(row + 1) * WIDTH])
    }

    /// Positions the images before `position` (whole spans) take back.
    fn shrink_before(&self, position: usize) -> usize {
        self.spans
            .iter()
            .take_while(|span| span.end() <= position)
            .map(ImageSpan::position_shrink)
            .sum()
    }

    /// The `(t, h, w, 0)` rotary position of the token at `position`.
    #[must_use]
    pub fn rope(&self, position: usize) -> [u32; 4] {
        if let Some(span) = self.span_at(position) {
            let t0 = (span.start - self.shrink_before(span.start)) as u32;
            let k = position - span.start;
            return [
                t0,
                t0 + (k / span.cols) as u32,
                t0 + (k % span.cols) as u32,
                0,
            ];
        }
        let p = (position - self.shrink_before(position)) as u32;
        [p, p, p, 0]
    }

    /// For `count` tokens from `start` that hold no image token, the offset
    /// every one of them rotates by relative to its position (plain `RoPE`
    /// at `position + offset`); `None` when an image token is among them.
    #[must_use]
    pub fn text_offset(&self, start: usize, count: usize) -> Option<i32> {
        let end = start + count;
        if self
            .spans
            .iter()
            .any(|span| span.start < end && start < span.end())
        {
            return None;
        }
        i32::try_from(self.shrink_before(start)).ok().map(|s| -s)
    }

    /// Whether `token` at `position` is this prompt's image token there.
    #[must_use]
    pub fn covers(&self, position: usize) -> bool {
        self.span_at(position).is_some()
    }
}

/// Validate that a sequence's virtual image tokens all fall in its image
/// spans, the one thing a token id above the vocabulary may be.
pub(crate) fn check_tokens(
    tokens: &[u32],
    first_position: usize,
    images: Option<&PromptImages>,
) -> crate::Result<()> {
    for (offset, &token) in tokens.iter().enumerate() {
        if token as usize >= VOCAB
            && !images.is_some_and(|images| images.covers(first_position + offset))
        {
            return Err(crate::Error::InvalidArgument(
                "Bonsai token beyond vocabulary".into(),
            ));
        }
    }
    Ok(())
}

/// What a prompt needs of an image or frame group: its merged grid and the
/// digest its token ids derive from ([`PreparedImage::layout`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub digest: [u8; 32],
}

impl Grid {
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

/// One image or video of a prompt, encoded, for [`expand_prompt`].
#[derive(Clone, Debug)]
pub enum Visual {
    /// An image and its decoder rows (`[tokens][WIDTH]`, embedding basis).
    Image(Grid, Arc<[f32]>),
    /// A video: its frame groups in order.
    Video(Vec<FrameGroup>),
}

/// A video frame group of a [`Visual::Video`]: the token ids of its
/// timestamp text, the group and its rows.
pub type FrameGroup = (Vec<u32>, Grid, Arc<[f32]>);

impl Visual {
    /// The template token this part's expansion replaces.
    const fn pad(&self) -> u32 {
        match self {
            Self::Image(..) => IMAGE_PAD,
            Self::Video(_) => VIDEO_PAD,
        }
    }

    /// Tokens the pad becomes.
    fn len(&self) -> usize {
        match self {
            Self::Image(image, _) => image.tokens(),
            Self::Video(groups) => groups
                .iter()
                .map(|(text, group, _)| text.len() + group.tokens() + 2)
                .sum(),
        }
    }
}

fn check_pads(ids: &[u32], visuals: &[Visual]) -> crate::Result<()> {
    let mut expected = visuals.iter().map(Visual::pad);
    let matched = ids
        .iter()
        .filter(|&&id| id == IMAGE_PAD || id == VIDEO_PAD)
        .all(|&id| expected.next() == Some(id));
    if matched && expected.next().is_none() {
        return Ok(());
    }
    Err(crate::Error::InvalidArgument(format!(
        "the prompt's <|image_pad|> and <|video_pad|> tokens do not match its {} images and \
         videos; message text may not contain them",
        visuals.len()
    )))
}

/// Replace each template `<|image_pad|>` and `<|video_pad|>` in `ids` with
/// its part's tokens, returning the expanded ids and the image spans (one
/// per image and per video frame group, in prompt order).
///
/// An image becomes its tokens. A video becomes, for each frame group, its
/// timestamp text, `<|vision_start|>`, the group's tokens and
/// `<|vision_end|>`, as the processor expands it.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the prompt's pads do not match
/// `visuals` one to one, in order (text may not spell one).
pub fn expand_prompt(
    ids: &[u32],
    projector: &[u8; 32],
    visuals: &[Visual],
) -> crate::Result<(Vec<u32>, PromptImages)> {
    check_pads(ids, visuals)?;
    let extra = visuals.iter().map(Visual::len).sum::<usize>();
    let mut expanded = Vec::with_capacity(ids.len() + extra);
    let mut spans = Vec::new();
    let mut grid = |expanded: &mut Vec<u32>, image: &Grid, embedding: &Arc<[f32]>| {
        let start = expanded.len();
        expanded.extend(
            (0..image.tokens() as u32).map(|k| image_token_id(projector, &image.digest, k)),
        );
        spans.push(ImageSpan {
            start,
            rows: image.rows,
            cols: image.cols,
            embedding: Arc::clone(embedding),
        });
    };
    let mut next = visuals.iter();
    for &id in ids {
        if id != IMAGE_PAD && id != VIDEO_PAD {
            expanded.push(id);
            continue;
        }
        match next.next() {
            Some(Visual::Image(image, embedding)) => grid(&mut expanded, image, embedding),
            Some(Visual::Video(groups)) => {
                for (text, group, embedding) in groups {
                    expanded.extend_from_slice(text);
                    expanded.push(VISION_START);
                    grid(&mut expanded, group, embedding);
                    expanded.push(VISION_END);
                }
            }
            None => break,
        }
    }
    Ok((expanded, PromptImages::new(spans)?))
}

/// The prompt length `ids` (holding one pad per part) expands to; the rows
/// of `visuals` are not read.
///
/// # Errors
///
/// As [`expand_prompt`].
pub fn expanded_len(ids: &[u32], visuals: &[Visual]) -> crate::Result<usize> {
    check_pads(ids, visuals)?;
    Ok(ids.len() + visuals.iter().map(|visual| visual.len() - 1).sum::<usize>())
}

/// Forward signed-Hadamard rotation of decoder input rows into the basis the
/// checkpoint stores `token_embd` in: per 1024-wide block, `y = H(s * x) /
/// 32`, the exact inverse of the GPU's input transform.
#[doc(hidden)]
pub fn rotate_rows(rows: &mut [f32], signs: &[f32]) {
    for row in rows.as_chunks_mut::<WIDTH>().0 {
        for (block, signs) in row
            .as_chunks_mut::<1024>()
            .0
            .iter_mut()
            .zip(signs.as_chunks::<1024>().0)
        {
            for (value, &sign) in block.iter_mut().zip(signs) {
                *value *= sign;
            }
            let mut distance = 1;
            while distance < 1024 {
                for start in (0..1024).step_by(2 * distance) {
                    for index in start..start + distance {
                        let (a, b) = (block[index], block[index + distance]);
                        block[index] = a + b;
                        block[index + distance] = a - b;
                    }
                }
                distance *= 2;
            }
            for value in block.iter_mut() {
                *value /= 32.0;
            }
        }
    }
}

/// An encoded image: its digest and decoder input rows.
type CachedImage = ([u8; 32], Arc<[f32]>);

/// Recently encoded images, by digest, within a byte budget.
pub(crate) struct EmbeddingCache {
    entries: Mutex<Vec<CachedImage>>,
    budget: usize,
}

impl EmbeddingCache {
    pub(crate) const fn new(budget: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            budget,
        }
    }

    pub(crate) fn get(&self, digest: &[u8; 32]) -> Option<Arc<[f32]>> {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let index = entries.iter().position(|(key, _)| key == digest)?;
        let entry = entries.remove(index);
        let value = Arc::clone(&entry.1);
        entries.push(entry);
        drop(entries);
        Some(value)
    }

    pub(crate) fn insert(&self, digest: [u8; 32], value: Arc<[f32]>) {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        entries.retain(|(key, _)| *key != digest);
        entries.push((digest, value));
        let mut bytes = entries
            .iter()
            .map(|(_, value)| value.len() * size_of::<f32>())
            .sum::<usize>();
        while entries.len() > 1 && bytes > self.budget {
            bytes -= entries.remove(0).1.len() * size_of::<f32>();
        }
        drop(entries);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
