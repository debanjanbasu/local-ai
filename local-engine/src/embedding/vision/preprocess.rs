//! Image and video preprocessing, matching the reference processors
//! (transformers 5.19.0 `Gemma4ImageProcessor` and
//! `EmbeddingGemma2VideoProcessor`, torchvision backend) exactly.
//!
//! 1. Decode to 8-bit RGB (alpha dropped, grey replicated; no EXIF rotation,
//!    as `torchvision.io.decode_image` and `PIL.Image.open` do). JPEG goes
//!    through libjpeg-turbo's decoder (the `mozjpeg` crate), the decoder
//!    Pillow and torchvision use, so the pixels are identical; other formats
//!    through the `image` crate.
//! 2. Aspect-preserving resize to the largest size whose sides are multiples
//!    of 48 (patch 16 x pooling 3) and that holds at most `soft_tokens * 9`
//!    patches, with torch's antialiased bicubic `uint8` kernel (Keys a = -0.5,
//!    int16 fixed-point weights, horizontal pass then vertical, rounding and
//!    clamping to `uint8` after each pass). Same size: no resize.
//! 3. Rescale: `f32(u8) * f32(1 / 255)`.
//! 4. Patchify into 16 x 16 x 3 rows (row, column, channel order), patches in
//!    raster order, positions `(x, y)` in patch units.
//!
//! Video frames are sampled as the reference does: 1 frame per second of the
//! stream (`floor(i * fps)` for `i < floor(duration)`), then at most 32 frames
//! chosen uniformly (`numpy.linspace(..., dtype=int)`).

use crate::Error;

/// Patch side in pixels.
pub const PATCH_SIZE: usize = 16;
/// Pooling kernel side in patches.
pub const POOLING: usize = 3;
/// Values per patch row (16 x 16 x 3).
pub const PATCH_VALUES: usize = PATCH_SIZE * PATCH_SIZE * 3;
/// The image soft-token budget the processor uses by default.
pub const IMAGE_SOFT_TOKENS: usize = 280;
/// The per-frame soft-token budget the video processor uses by default.
pub const VIDEO_SOFT_TOKENS: usize = 140;
/// Budgets the reference processors accept.
pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];
/// Default video sampling rate (frames per second of the stream).
pub const VIDEO_FPS: f64 = 1.0;
/// Default cap on sampled video frames.
pub const VIDEO_MAX_FRAMES: usize = 32;
/// Largest accepted decoded image side, in pixels.
pub const MAX_IMAGE_SIDE: u32 = 16_384;

/// The processor's `rescale_factor` (0.00392156862745098) as torch applies
/// it to a `uint8` tensor: an F32 multiply.
pub const RESCALE: f32 = (1.0_f64 / 255.0) as f32;

/// An 8-bit RGB image, rows top to bottom, `[height][width][3]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl RgbImage {
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if `pixels` is not `width * height * 3`
    /// bytes or a side is zero.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> crate::Result<Self> {
        if width == 0
            || height == 0
            || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(pixels.len())
        {
            return Err(Error::InvalidArgument(format!(
                "RGB image of {width}x{height} needs {} bytes, got {}",
                width.saturating_mul(height).saturating_mul(3),
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
}

/// One preprocessed image or video frame: `patches` rows of
/// [`PATCH_VALUES`] rescaled values and their `(x, y)` patch positions, on a
/// grid of `columns` x `rows` patches (both multiples of 3).
#[derive(Clone, Debug, PartialEq)]
pub struct ImagePatches {
    pub columns: usize,
    pub rows: usize,
    pub values: Vec<f32>,
}

impl ImagePatches {
    #[must_use]
    pub const fn patch_count(&self) -> usize {
        self.columns * self.rows
    }

    /// Soft tokens this image becomes (one per 3 x 3 patches).
    #[must_use]
    pub const fn soft_tokens(&self) -> usize {
        self.patch_count() / (POOLING * POOLING)
    }

    /// `(x, y)` of patch `index` (raster order).
    #[must_use]
    pub const fn position(&self, index: usize) -> (usize, usize) {
        (index % self.columns, index / self.columns)
    }
}

/// Decode a baseline, progressive or arithmetic-coded JPEG with
/// libjpeg-turbo (islow IDCT, fancy upsampling: the library defaults Pillow
/// and torchvision use). `Ok(None)` for CMYK/YCCK, left to [`image`].
fn decode_jpeg(bytes: &[u8]) -> crate::Result<Option<RgbImage>> {
    let decoded =
        std::panic::catch_unwind(|| -> std::io::Result<Option<(usize, usize, Vec<u8>)>> {
            let decompress = mozjpeg::Decompress::new_mem(bytes)?;
            let (width, height) = decompress.size();
            if width.max(height) > MAX_IMAGE_SIDE as usize {
                return Err(std::io::Error::other(format!(
                    "{width}x{height} exceeds {MAX_IMAGE_SIDE} pixels per side"
                )));
            }
            if matches!(
                decompress.color_space(),
                mozjpeg::ColorSpace::JCS_CMYK | mozjpeg::ColorSpace::JCS_YCCK
            ) {
                return Ok(None);
            }
            let mut started = decompress.rgb()?;
            let pixels = started.read_scanlines::<u8>()?;
            started.finish()?;
            Ok(Some((width, height, pixels)))
        });
    match decoded {
        Ok(Ok(Some((width, height, pixels)))) => RgbImage::new(width, height, pixels).map(Some),
        Ok(Ok(None)) => Ok(None),
        Ok(Err(error)) => Err(Error::InvalidArgument(format!("image: JPEG: {error}"))),
        Err(payload) => Err(Error::InvalidArgument(format!(
            "image: JPEG: {}",
            payload
                .downcast_ref::<String>()
                .map_or("undecodable data", String::as_str)
        ))),
    }
}

/// Decode an encoded image (JPEG, PNG, GIF, BMP or WebP; the first frame of
/// an animation) to 8-bit RGB.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for undecodable data or a side over
/// [`MAX_IMAGE_SIDE`].
pub fn decode_image(bytes: &[u8]) -> crate::Result<RgbImage> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF])
        && let Some(image) = decode_jpeg(bytes)?
    {
        return Ok(image);
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|error| Error::InvalidArgument(format!("image: {error}")))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    limits.max_alloc = Some(2 << 30);
    reader.limits(limits);
    let decoded = reader
        .decode()
        .map_err(|error| Error::InvalidArgument(format!("image: {error}")))?;
    let rgb = decoded.into_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    RgbImage::new(width, height, rgb.into_raw())
}

/// The reference's `get_aspect_ratio_preserving_size`: `(height, width)`
/// after resizing for a budget of `max_patches` patches.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when no size fits (the reference raises too).
pub fn target_size(
    height: usize,
    width: usize,
    max_patches: usize,
) -> crate::Result<(usize, usize)> {
    let side = PATCH_SIZE * POOLING;
    let total = (height * width) as f64;
    let target_px = max_patches * PATCH_SIZE * PATCH_SIZE;
    let factor = (target_px as f64 / total).sqrt();
    let ideal_height = factor * height as f64;
    let ideal_width = factor * width as f64;
    let mut target_height = (ideal_height / side as f64).floor() as usize * side;
    let mut target_width = (ideal_width / side as f64).floor() as usize * side;
    if target_height == 0 && target_width == 0 {
        return Err(Error::InvalidArgument(format!(
            "image of {width}x{height} cannot be resized to the patch grid"
        )));
    }
    let max_side = (max_patches / (POOLING * POOLING)) * side;
    if target_height == 0 {
        target_height = side;
        target_width = ((width as f64 / height as f64).floor() as usize * side).min(max_side);
    } else if target_width == 0 {
        target_width = side;
        target_height = ((height as f64 / width as f64).floor() as usize * side).min(max_side);
    }
    if target_height * target_width > target_px {
        return Err(Error::InvalidArgument(format!(
            "image of {width}x{height} exceeds {max_patches} patches after resizing"
        )));
    }
    Ok((target_height, target_width))
}

/// Antialiased bicubic weights of one axis, as torch computes them for
/// `uint8` input (`_compute_index_ranges_int16_weights`).
struct AxisWeights {
    starts: Vec<usize>,
    sizes: Vec<usize>,
    stride: usize,
    weights: Vec<i32>,
    precision: u32,
}

// Unfused arithmetic on purpose: the same operations as torch's C++.
#[allow(clippy::suboptimal_flops)]
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

#[allow(clippy::cast_possible_wrap, clippy::suboptimal_flops)]
fn axis_weights(input: usize, output: usize) -> AxisWeights {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let stride = support.ceil() as usize * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut starts = Vec::with_capacity(output);
    let mut sizes = Vec::with_capacity(output);
    let mut real = vec![0.0_f64; output * stride];
    let mut max_weight = 0.0_f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        let start = ((center - support + 0.5) as i64).max(0);
        let size = ((center + support + 0.5) as i64)
            .min(input as i64)
            .saturating_sub(start)
            .clamp(0, stride as i64) as usize;
        let row = &mut real[i * stride..(i + 1) * stride];
        let mut total = 0.0;
        for (j, w) in row.iter_mut().enumerate().take(size) {
            *w = cubic(((j as i64 + start) as f64 - center + 0.5) * invscale);
            total += *w;
        }
        if total != 0.0 {
            for w in row.iter_mut().take(size) {
                *w /= total;
                max_weight = max_weight.max(*w);
            }
        }
        starts.push(start as usize);
        sizes.push(size);
    }
    let mut precision = 0;
    while precision < 22 {
        let next = (0.5 + max_weight * f64::from(1_u32 << (precision + 1))) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let unit = f64::from(1_u32 << precision);
    let weights = real
        .iter()
        .map(|w| {
            let v = w * unit;
            if v < 0.0 {
                (-0.5 + v) as i32
            } else {
                (0.5 + v) as i32
            }
        })
        .collect();
    AxisWeights {
        starts,
        sizes,
        stride,
        weights,
        precision,
    }
}

fn convolve(sum: i64, precision: u32) -> u8 {
    ((sum + (1 << (precision - 1))) >> precision).clamp(0, 255) as u8
}

/// torch's antialiased bicubic resize of an RGB `uint8` image (CPU kernel,
/// `F.interpolate(..., mode="bicubic", antialias=True)` on `uint8`).
#[must_use]
pub fn resize_bicubic(image: &RgbImage, height: usize, width: usize) -> RgbImage {
    let mut current = image.clone();
    if width != current.width {
        let w = axis_weights(current.width, width);
        let mut out = vec![0_u8; current.height * width * 3];
        for y in 0..current.height {
            let src = &current.pixels[y * current.width * 3..(y + 1) * current.width * 3];
            for x in 0..width {
                let k = &w.weights[x * w.stride..x * w.stride + w.sizes[x]];
                for c in 0..3 {
                    let sum: i64 = k
                        .iter()
                        .enumerate()
                        .map(|(j, &kj)| i64::from(kj) * i64::from(src[(w.starts[x] + j) * 3 + c]))
                        .sum();
                    out[(y * width + x) * 3 + c] = convolve(sum, w.precision);
                }
            }
        }
        current = RgbImage {
            width,
            height: current.height,
            pixels: out,
        };
    }
    if height != current.height {
        let w = axis_weights(current.height, height);
        let row = current.width * 3;
        let mut out = vec![0_u8; height * row];
        for y in 0..height {
            let k = &w.weights[y * w.stride..y * w.stride + w.sizes[y]];
            for i in 0..row {
                let sum: i64 = k
                    .iter()
                    .enumerate()
                    .map(|(j, &kj)| {
                        i64::from(kj) * i64::from(current.pixels[(w.starts[y] + j) * row + i])
                    })
                    .sum();
                out[y * row + i] = convolve(sum, w.precision);
            }
        }
        current = RgbImage {
            width: current.width,
            height,
            pixels: out,
        };
    }
    current
}

/// Resize (if needed), rescale and patchify one image for a budget of
/// `soft_tokens` (one of [`SUPPORTED_SOFT_TOKENS`]).
///
/// # Errors
///
/// [`Error::InvalidArgument`] for an unsupported budget or an image that
/// cannot be fit to the patch grid.
pub fn patchify(image: &RgbImage, soft_tokens: usize) -> crate::Result<ImagePatches> {
    if !SUPPORTED_SOFT_TOKENS.contains(&soft_tokens) {
        return Err(Error::InvalidArgument(format!(
            "soft-token budget {soft_tokens} is not one of {SUPPORTED_SOFT_TOKENS:?}"
        )));
    }
    let (height, width) = target_size(image.height, image.width, soft_tokens * POOLING * POOLING)?;
    let resized;
    let image = if (height, width) == (image.height, image.width) {
        image
    } else {
        resized = resize_bicubic(image, height, width);
        &resized
    };
    let (columns, rows) = (width / PATCH_SIZE, height / PATCH_SIZE);
    let mut values = Vec::with_capacity(columns * rows * PATCH_VALUES);
    for py in 0..rows {
        for px in 0..columns {
            for dy in 0..PATCH_SIZE {
                let start = ((py * PATCH_SIZE + dy) * width + px * PATCH_SIZE) * 3;
                values.extend(
                    image.pixels[start..start + PATCH_SIZE * 3]
                        .iter()
                        .map(|&v| f32::from(v) * RESCALE),
                );
            }
        }
    }
    Ok(ImagePatches {
        columns,
        rows,
        values,
    })
}

/// Stream facts the frame sampler needs (as torchcodec reports them in
/// exact mode): decoded frame count, average rate and duration in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoTiming {
    pub frames: usize,
    pub fps: Option<f64>,
    pub duration: Option<f64>,
}

/// Indices of the frames the reference samples: `fps` frames per second of
/// the stream when the rate and duration are known (otherwise every frame),
/// then at most `max_frames` chosen uniformly.
#[must_use]
pub fn sample_frame_indices(timing: VideoTiming, fps: f64, max_frames: usize) -> Vec<usize> {
    if timing.frames == 0 {
        return Vec::new();
    }
    let indices: Vec<usize> = match (timing.fps, timing.duration) {
        (Some(native), Some(duration)) if native > 0.0 && duration.is_finite() => {
            let step = native / fps;
            let count = ((duration * fps) as usize).max(1);
            (0..count)
                .map(|i| ((i as f64 * step) as usize).min(timing.frames - 1))
                .collect()
        }
        _ => (0..timing.frames).collect(),
    };
    if max_frames == 0 || indices.len() <= max_frames {
        return indices;
    }
    let last = indices.len() - 1;
    if max_frames == 1 {
        return vec![indices[0]];
    }
    (0..max_frames)
        .map(|i| {
            let at = if i + 1 == max_frames {
                last
            } else {
                // numpy.linspace: i * ((n - 1) / (num - 1)), truncated.
                (i as f64 * (last as f64 / (max_frames - 1) as f64)) as usize
            };
            indices[at]
        })
        .collect()
}
