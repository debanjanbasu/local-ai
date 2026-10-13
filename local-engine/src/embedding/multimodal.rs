//! Multimodal inputs: one sequence assembly for text, image, video and audio.
//!
//! Every input runs through the shared text encoder as one `<bos> … <eos>`
//! sequence laid out like the reference processor: text tokens as given;
//! each image or video frame as `<|image>`, its vision soft tokens,
//! `<image|>` ([`super::vision`]); each audio clip as `<start_of_audio>`,
//! its audio soft tokens, `<end_of_audio>` (`super::audio`). The soft tokens
//! of every modality are computed first (each tower in its own passes), then
//! the sequences are packed into text-encoder passes
//! ([`EmbeddingModel::encode_embedded`]), so an image, a video, an audio clip
//! and a text land in one 768-d space. A text-only input matches
//! [`EmbeddingModel::embed_texts`]; an audio-only input is the reference's
//! audio sequence.

use super::audio::{audio_tokens, frame_counts, validate_clip};
use super::vision::ImagePatches;
use super::{EmbeddingModel, HIDDEN, MEDIA_PLACEHOLDERS, PASS_TOKENS};
use crate::Error;

/// Longest multimodal input in tokens (the model's context), `<bos>`,
/// `<eos>`, the media delimiters and every soft token included.
pub const MAX_MULTIMODAL_TOKENS: usize = 8192;

const BOS: u32 = 2;
const EOS: u32 = 1;
const BOI: u32 = 255_999;
const EOI: u32 = 258_882;
const BOA: u32 = 256_000;
const EOA: u32 = 258_883;

/// One part of a multimodal input, in order. Text is embedded as given (the
/// model card uses no prompt for media).
#[derive(Clone, Copy, Debug)]
pub enum MediaPart<'a> {
    Text(&'a str),
    /// One preprocessed image ([`super::vision::preprocess::patchify`]).
    Image(&'a ImagePatches),
    /// Preprocessed, already sampled frames of one video.
    Video(&'a [ImagePatches]),
    /// One 16 kHz mono clip of at most 30 s ([`super::decode_audio`]).
    Audio(&'a [f32]),
}

/// One position of a laid-out input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Token(u32),
    /// The next vision soft token (images and frames in input order).
    Vision,
    /// The next audio soft token (clips in input order).
    Audio,
}

impl EmbeddingModel {
    /// Embed one multimodal input (full width, L2-normalized).
    pub(super) fn single(
        &mut self,
        parts: &[MediaPart<'_>],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<f32>> {
        self.embed_multimodal(&[parts.to_vec()], pace)?
            .pop()
            .ok_or_else(|| Error::Generation("no embedding produced".into()))
    }

    /// Token sequence of one multimodal input, with `None` marking soft-token
    /// slots, as the reference processor lays it out: `<bos>`, then each part
    /// in order (text tokens; per image or video frame `<|image>`, one slot
    /// per soft token, `<image|>`; per audio clip `<start_of_audio>`, one slot
    /// per soft token, `<end_of_audio>`), then `<eos>`.
    ///
    /// # Errors
    ///
    /// Tokenizer failures, media placeholder ids inside text, empty videos,
    /// invalid audio clips, or more than [`MAX_MULTIMODAL_TOKENS`] tokens.
    pub fn multimodal_layout(&self, parts: &[MediaPart<'_>]) -> crate::Result<Vec<Option<u32>>> {
        Ok(self
            .layout(parts)?
            .into_iter()
            .map(|slot| match slot {
                Slot::Token(id) => Some(id),
                Slot::Vision | Slot::Audio => None,
            })
            .collect())
    }

    fn layout(&self, parts: &[MediaPart<'_>]) -> crate::Result<Vec<Slot>> {
        let mut layout = vec![Slot::Token(BOS)];
        let mut text = String::new();
        let flush = |text: &mut String, layout: &mut Vec<Slot>| -> crate::Result<()> {
            if !text.is_empty() {
                let encoding = self
                    .tokenizer
                    .encode(text.as_str(), false)
                    .map_err(|error| Error::Tokenizer(error.to_string()))?;
                if encoding
                    .get_ids()
                    .iter()
                    .any(|id| MEDIA_PLACEHOLDERS.contains(id))
                {
                    return Err(Error::InvalidArgument(
                        "text part contains an image, audio or video placeholder token".into(),
                    ));
                }
                layout.extend(encoding.get_ids().iter().map(|&id| Slot::Token(id)));
                text.clear();
            }
            Ok(())
        };
        let media = |open: u32, slot: Slot, soft: usize, close: u32, layout: &mut Vec<Slot>| {
            layout.push(Slot::Token(open));
            layout.extend(std::iter::repeat_n(slot, soft));
            layout.push(Slot::Token(close));
        };
        for part in parts {
            match part {
                MediaPart::Text(t) => text.push_str(t),
                MediaPart::Image(image) => {
                    flush(&mut text, &mut layout)?;
                    media(BOI, Slot::Vision, image.soft_tokens(), EOI, &mut layout);
                }
                MediaPart::Video(frames) => {
                    if frames.is_empty() {
                        return Err(Error::InvalidArgument("video has no frames".into()));
                    }
                    flush(&mut text, &mut layout)?;
                    for f in *frames {
                        media(BOI, Slot::Vision, f.soft_tokens(), EOI, &mut layout);
                    }
                }
                MediaPart::Audio(clip) => {
                    validate_clip(clip)?;
                    flush(&mut text, &mut layout)?;
                    let soft = audio_tokens(frame_counts(clip.len()).1);
                    media(BOA, Slot::Audio, soft, EOA, &mut layout);
                }
            }
            if layout.len() > MAX_MULTIMODAL_TOKENS {
                break;
            }
        }
        flush(&mut text, &mut layout)?;
        layout.push(Slot::Token(EOS));
        if layout.len() > MAX_MULTIMODAL_TOKENS {
            return Err(Error::ContextOverflow(format!(
                "multimodal input has {} tokens; the limit is {MAX_MULTIMODAL_TOKENS}",
                layout.len()
            )));
        }
        Ok(layout)
    }

    /// Embed multimodal inputs (each a sequence of text, image, video and
    /// audio parts; a text-only input matches [`Self::embed_texts`]); one
    /// full-width L2-normalized vector per input, in order.
    ///
    /// Every image and frame runs through the vision encoder (in passes of up
    /// to 2,520 patches, one command buffer per layer, `pace` after each),
    /// every audio clip through the audio encoder, then the sequences through
    /// the text encoder (packed up to 1,024 tokens per pass; a longer input
    /// takes its own). Each encoder artifact loads on the first input that
    /// needs it.
    ///
    /// # Errors
    ///
    /// [`Self::multimodal_layout`] errors, a missing vision or audio artifact
    /// (for media parts), a `pace` error, or a Metal failure.
    pub fn embed_multimodal(
        &mut self,
        inputs: &[Vec<MediaPart<'_>>],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let layouts = inputs
            .iter()
            .map(|parts| self.layout(parts))
            .collect::<crate::Result<Vec<_>>>()?;
        let frames: Vec<&ImagePatches> = inputs
            .iter()
            .flatten()
            .flat_map(|part| match part {
                MediaPart::Text(_) | MediaPart::Audio(_) => &[][..],
                MediaPart::Image(image) => std::slice::from_ref(*image),
                MediaPart::Video(frames) => frames,
            })
            .collect();
        let clips: Vec<&[f32]> = inputs
            .iter()
            .flatten()
            .filter_map(|part| match part {
                MediaPart::Audio(clip) => Some(*clip),
                _ => None,
            })
            .collect();
        let vision = if frames.is_empty() {
            Vec::new()
        } else {
            self.vision_soft_tokens(&frames, pace)?
        };
        let audio = if clips.is_empty() {
            Vec::new()
        } else {
            self.audio_soft_tokens(&clips, pace)?
        };
        for (clip, soft) in clips.iter().zip(&audio) {
            if soft.len() != audio_tokens(frame_counts(clip.len()).1) * HIDDEN {
                return Err(Error::Generation(
                    "audio soft tokens differ from the layout".into(),
                ));
            }
        }
        let mut vision_rows = vision.as_chunks::<HIDDEN>().0.iter();
        let mut audio_rows = audio
            .iter()
            .flat_map(|clip| clip.as_chunks::<HIDDEN>().0.iter());
        let missing = || Error::Generation("fewer soft tokens than placeholders".into());
        let mut out = Vec::with_capacity(inputs.len());
        let mut start = 0;
        while start < layouts.len() {
            let mut end = start + 1;
            let mut tokens = layouts[start].len();
            while end < layouts.len() && tokens + layouts[end].len() <= PASS_TOKENS {
                tokens += layouts[end].len();
                end += 1;
            }
            let mut x0 = Vec::with_capacity(tokens * HIDDEN);
            for slot in layouts[start..end].iter().flatten() {
                match slot {
                    Slot::Token(id) => self.push_token_embedding(&mut x0, *id),
                    Slot::Vision => x0.extend_from_slice(vision_rows.next().ok_or_else(missing)?),
                    Slot::Audio => x0.extend_from_slice(audio_rows.next().ok_or_else(missing)?),
                }
            }
            let lens: Vec<usize> = layouts[start..end].iter().map(Vec::len).collect();
            out.extend(self.encode_embedded(&x0, &lens, pace)?);
            start = end;
        }
        if vision_rows.next().is_some() || audio_rows.next().is_some() {
            return Err(Error::Generation(
                "more soft tokens than placeholders".into(),
            ));
        }
        Ok(out)
    }
}
