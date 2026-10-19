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
//! Prompt caching keys on token ids, so each image token gets a virtual id
//! above the vocabulary ([`image_token_id`]) derived from the mmproj and the
//! preprocessed pixels: equal images share prefixes, different ones never do.

mod preprocess;
mod tower;

use std::sync::{Arc, Mutex, PoisonError};

use sha2::{Digest as _, Sha256};

use crate::bonsai::VOCAB;
use crate::bonsai::WIDTH;

pub use self::preprocess::{ImageDetail, PreparedImage, prepare_decoded, prepare_image};
pub use self::tower::{VISION_PROJECTOR_FILE, VisionEncoder};

/// `<|vision_start|>`.
pub const VISION_START: u32 = 248_053;
/// `<|vision_end|>`.
pub const VISION_END: u32 = 248_054;
/// `<|image_pad|>`: the template writes one per image; the processor expands
/// it to one per image token.
pub const IMAGE_PAD: u32 = 248_056;
/// What the chat template renders for each image part.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

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

/// Replace each template `<|image_pad|>` in `ids` with its image's tokens,
/// returning the expanded ids and spans for `embeddings` (one per image, in
/// prompt order, each `[tokens][WIDTH]` in the embedding basis).
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] when the prompt holds a different number
/// of `<|image_pad|>` tokens than images (text may not spell one).
pub fn expand_prompt(
    ids: &[u32],
    projector: &[u8; 32],
    images: &[(&PreparedImage, Arc<[f32]>)],
) -> crate::Result<(Vec<u32>, PromptImages)> {
    let pads = ids.iter().filter(|&&id| id == IMAGE_PAD).count();
    if pads != images.len() {
        return Err(crate::Error::InvalidArgument(format!(
            "the prompt has {pads} <|image_pad|> tokens for {} images; message text may not \
             contain <|image_pad|>",
            images.len()
        )));
    }
    let extra = images
        .iter()
        .map(|(image, _)| image.tokens() - 1)
        .sum::<usize>();
    let mut expanded = Vec::with_capacity(ids.len() + extra);
    let mut spans = Vec::with_capacity(images.len());
    let mut next = images.iter();
    for &id in ids {
        if id != IMAGE_PAD {
            expanded.push(id);
            continue;
        }
        let Some((image, embedding)) = next.next() else {
            break;
        };
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
    }
    Ok((expanded, PromptImages::new(spans)?))
}

/// The prompt length `ids` (holding one `<|image_pad|>` per image) expands
/// to.
///
/// # Errors
///
/// As [`expand_prompt`].
pub fn expanded_len(ids: &[u32], images: &[PreparedImage]) -> crate::Result<usize> {
    let pads = ids.iter().filter(|&&id| id == IMAGE_PAD).count();
    if pads != images.len() {
        return Err(crate::Error::InvalidArgument(format!(
            "the prompt has {pads} <|image_pad|> tokens for {} images; message text may not \
             contain <|image_pad|>",
            images.len()
        )));
    }
    Ok(ids.len() + images.iter().map(|image| image.tokens() - 1).sum::<usize>())
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
