//! `Qwen2VLImageProcessorFast` preprocessing for the Bonsai vision tower.
//!
//! 1. Decode to 8-bit RGB as the embedding vision path does
//!    ([`crate::embedding::vision::preprocess::decode_image`]: libjpeg-turbo
//!    for JPEG, no EXIF rotation, alpha dropped).
//! 2. `smart_resize`: sides rounded (Python's half-to-even `round`) to
//!    multiples of 32 (patch 16 x merge 2); scaled down (`floor`) to at most
//!    the detail's pixel budget, or up (`ceil`) to at least 65,536 pixels;
//!    aspect ratios above 200 are refused.
//! 3. torch's antialiased bicubic `uint8` resize (always applied, an identity
//!    at the same size).
//! 4. `(f32(x) - 127.5) / 127.5`, patchified into rows of 3 x 16 x 16
//!    (channel, row, column) in merge order: 2x2 blocks of patches in raster
//!    order of the blocks. The processor stacks two identical temporal
//!    slices; the tower folds them into its patch weights instead.

use sha2::{Digest as _, Sha256};

use crate::Error;
use crate::embedding::vision::preprocess::{RgbImage, decode_image, resize_bicubic};

/// Patch side in pixels.
pub const PATCH: usize = 16;
/// Merge block side in patches.
pub const MERGE: usize = 2;
/// `smart_resize`'s factor.
const FACTOR: usize = PATCH * MERGE;
/// Values per patch row (one temporal slice).
pub const PATCH_VALUES: usize = 3 * PATCH * PATCH;
/// `smart_resize`'s factor, for the video processor's variant.
pub(super) const RESIZE_FACTOR: usize = FACTOR;
/// The processor's `shortest_edge` (minimum pixels).
const MIN_PIXELS: usize = 65_536;
/// Largest aspect ratio the processor accepts.
const MAX_ASPECT: f64 = 200.0;

/// The `OpenAI` `detail` of an image part: its pixel budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ImageDetail {
    /// At most 262,144 pixels (256 tokens).
    Low,
    /// At most 4,194,304 pixels (4,096 tokens): the image budget Prism's
    /// llama.cpp build runs the published vision evaluations at.
    #[default]
    Auto,
    /// At most 16,777,216 pixels (16,384 tokens): the processor's own
    /// default ceiling.
    High,
}

impl ImageDetail {
    /// Most pixels an image of this detail keeps.
    #[must_use]
    pub const fn max_pixels(self) -> usize {
        match self {
            Self::Low => 262_144,
            Self::Auto => 4_194_304,
            Self::High => 16_777_216,
        }
    }

    /// Parse an `OpenAI` `detail` value.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for anything but `low`, `auto` or `high`.
    pub fn parse(value: &str) -> crate::Result<Self> {
        match value {
            "low" => Ok(Self::Low),
            "auto" => Ok(Self::Auto),
            "high" => Ok(Self::High),
            other => Err(Error::InvalidArgument(format!(
                "image detail must be low, auto or high, not {other:?}"
            ))),
        }
    }
}

/// A decoded, resized image ready for the tower.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    /// The resized image (`height = rows * 32`, `width = cols * 32`).
    pub image: RgbImage,
    /// A video frame group's second frame, the same size; `None` for a
    /// still image, whose one frame fills both temporal slices.
    pub next: Option<RgbImage>,
    /// Merged grid: `rows * cols` image tokens.
    pub rows: usize,
    pub cols: usize,
    /// SHA-256 of the grid and the resized pixels.
    pub digest: [u8; 32],
}

impl PreparedImage {
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.rows * self.cols
    }

    /// Patch rows (`rows * cols * 4`).
    #[must_use]
    pub const fn patches(&self) -> usize {
        self.tokens() * MERGE * MERGE
    }

    /// Patch grid rows and columns.
    #[must_use]
    pub const fn grid(&self) -> (usize, usize) {
        (self.rows * MERGE, self.cols * MERGE)
    }

    /// Values per patch row: one temporal slice for a still image, both
    /// (frame then next) for a video frame group.
    #[must_use]
    pub const fn patch_values(&self) -> usize {
        if self.next.is_some() {
            2 * PATCH_VALUES
        } else {
            PATCH_VALUES
        }
    }

    /// Normalized patch rows `[patches][patch_values]` in merge order, each
    /// slice `(channel, row, column)`.
    #[must_use]
    pub fn patch_rows(&self) -> Vec<f32> {
        let (grid_h, grid_w) = self.grid();
        let width = self.image.width;
        let frames = std::iter::once(&self.image)
            .chain(&self.next)
            .map(|frame| frame.pixels.as_slice())
            .collect::<Vec<_>>();
        let mut out = Vec::with_capacity(self.patches() * self.patch_values());
        for (h, w) in merge_order(grid_h, grid_w) {
            for pixels in &frames {
                for channel in 0..3 {
                    for y in 0..PATCH {
                        let row = (h * PATCH + y) * width + w * PATCH;
                        for x in 0..PATCH {
                            let value = f32::from(pixels[(row + x) * 3 + channel]);
                            out.push((value - 127.5) / 127.5);
                        }
                    }
                }
            }
        }
        out
    }
}

/// Patch `(row, column)` of each patch in merge order.
pub fn merge_order(grid_h: usize, grid_w: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..grid_h / MERGE).flat_map(move |block_row| {
        (0..grid_w / MERGE).flat_map(move |block_col| {
            (0..MERGE).flat_map(move |dy| {
                (0..MERGE).map(move |dx| (block_row * MERGE + dy, block_col * MERGE + dx))
            })
        })
    })
}

/// Python's `round` (half to even) of a non-negative ratio.
pub(super) const fn round_half_even(value: f64) -> usize {
    value.round_ties_even() as usize
}

/// The processor's `smart_resize`: `(height, width)` for the tower.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for an aspect ratio above 200.
pub fn smart_resize(
    height: usize,
    width: usize,
    max_pixels: usize,
) -> crate::Result<(usize, usize)> {
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > MAX_ASPECT {
        return Err(Error::InvalidArgument(format!(
            "image of {width}x{height} has an aspect ratio above {MAX_ASPECT}"
        )));
    }
    let factor = FACTOR as f64;
    let mut h_bar = (round_half_even(h / factor) * FACTOR).max(FACTOR);
    let mut w_bar = (round_half_even(w / factor) * FACTOR).max(FACTOR);
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = FACTOR.max((h / beta / factor).floor() as usize * FACTOR);
        w_bar = FACTOR.max((w / beta / factor).floor() as usize * FACTOR);
    } else if h_bar * w_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        h_bar = (h * beta / factor).ceil() as usize * FACTOR;
        w_bar = (w * beta / factor).ceil() as usize * FACTOR;
    }
    Ok((h_bar, w_bar))
}

/// Decode and resize an encoded image file for `detail`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for undecodable data or an unsupported shape.
pub fn prepare_image(bytes: &[u8], detail: ImageDetail) -> crate::Result<PreparedImage> {
    prepare_decoded(&decode_image(bytes)?, detail.max_pixels())
}

/// [`prepare_image`] of a decoded image with an explicit pixel budget.
///
/// # Errors
///
/// As [`smart_resize`].
pub fn prepare_decoded(image: &RgbImage, max_pixels: usize) -> crate::Result<PreparedImage> {
    let (height, width) = smart_resize(image.height, image.width, max_pixels)?;
    let image = resize_bicubic(image, height, width);
    let (rows, cols) = (height / FACTOR, width / FACTOR);
    let digest = Sha256::new()
        .chain_update((rows as u32).to_le_bytes())
        .chain_update((cols as u32).to_le_bytes())
        .chain_update(&image.pixels)
        .finalize()
        .into();
    Ok(PreparedImage {
        image,
        next: None,
        rows,
        cols,
        digest,
    })
}
