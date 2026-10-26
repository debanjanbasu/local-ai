//! Video input: `Qwen3VLVideoProcessor` (the processor the checkpoint names)
//! over frames `AVFoundation` decodes.
//!
//! 1. Decode (`AVFoundation` through `objc2`: MP4, MOV and M4V containers with
//!    any codec macOS decodes, H.264 and HEVC everywhere; Matroska and `WebM`
//!    are refused). A first pass over the compressed samples counts the
//!    frames and their duration, as the reference's decoders report them
//!    (frame count, `frames / duration`); the second decodes in presentation
//!    order and keeps only the sampled frames, each resized as soon as it is
//!    decoded. `VideoToolbox` hands over its native 8-bit 4:2:0 planes (codec
//!    decoding is exact), converted to RGB with swscale's fixed-point
//!    arithmetic and the stream's colour matrix (BT.601 when untagged), as
//!    torchcodec, the processor's default decoder, converts them
//!    ([`crate::embedding::vision::video`] holds that conversion). 10-bit
//!    streams are reduced to 8 bits by `VideoToolbox`.
//! 2. `sample_frames`: `int(total / fps * 2)` frames (2 per second), at
//!    least 4 and at most 768 and the video's own count, at
//!    `round(linspace(0, total - 1, n))`.
//! 3. The video `smart_resize` (sides in multiples of 32, the clip's frames
//!    together at least 4,096 pixels) with the reference per-frame cap
//!    (`cap_pixels_per_frame`, qwen-vl-utils' behaviour and the transformers
//!    default from v5.22): each frame at most 786,432 pixels (768 tokens)
//!    and the clip at most 25,165,824, then the image path's antialiased
//!    bicubic resize.
//! 4. An odd frame count repeats the last frame; each pair of frames is one
//!    temporal patch group, encoded like an image of the same grid (the
//!    tower attends within a group only) with both frames in its patches.
//! 5. The prompt holds, for each group, `<{t:.1} seconds>` (the mean of its
//!    two frames' times) and the group's tokens between `<|vision_start|>`
//!    and `<|vision_end|>`, all inside the template's own
//!    `<|vision_start|>...<|vision_end|>`, exactly as the processor expands
//!    `<|video_pad|>`. Each group takes image M-RoPE positions (`get_rope_index`
//!    splits the grid into `t = 1` grids); the timestamps carry the time.
//!
//! Prepared videos are cached by content, so a conversation that keeps a
//! video in its history decodes it once.

use std::cell::Cell;
use std::sync::{Arc, Mutex, PoisonError};

use sha2::{Digest as _, Sha256};

use super::preprocess::{PreparedImage, RESIZE_FACTOR, round_half_even};
use crate::Error;
use crate::bonsai_model::CancelToken;
use crate::embedding::vision::preprocess::{
    MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE, RgbImage, resize_bicubic,
};

/// `<|video_pad|>`: the template writes one per video; the prompt expands it
/// to the frame groups.
pub const VIDEO_PAD: u32 = 248_057;
/// What the chat template renders for each video part.
pub const VIDEO_PLACEHOLDER: &str = "<|vision_start|><|video_pad|><|vision_end|>";

/// Frames sampled per second of video.
const FPS: f64 = 2.0;
const MIN_FRAMES: usize = 4;
/// Most frames sampled from one video (384 frame groups).
pub const MAX_FRAMES: usize = 768;
/// Frames per temporal patch group.
const TEMPORAL: usize = 2;
/// `shortest_edge`: fewest pixels of the whole clip.
const MIN_PIXELS: usize = 4096;
/// `longest_edge`: most pixels of the whole clip (12,288 tokens).
const MAX_PIXELS: usize = 25_165_824;
/// `max_video_tokens`: most tokens of one frame group.
const MAX_FRAME_TOKENS: usize = 768;
const MAX_ASPECT: f64 = 200.0;
/// Longest video accepted, in seconds of its track. The probe and the
/// decoding pass both read every frame up to the last sampled one (the
/// last frame), so this bounds a request's decoding time.
pub const MAX_SECONDS: f64 = 600.0;
/// Most frames (and sample buffers) a video track may hold: 600 s at 60
/// frames per second.
pub const MAX_TRACK_FRAMES: usize = 36_000;
/// Prepared videos kept for reuse (a clip at the pixel budget holds 72 MiB).
const CACHE_BYTES: usize = 256 << 20;

/// A decoded, sampled and resized video ready for the tower.
#[derive(Debug)]
pub struct PreparedVideo {
    /// Frame groups in order, each two frames of one grid.
    pub groups: Vec<PreparedImage>,
    /// Each group's time in seconds.
    pub timestamps: Vec<f64>,
    /// Sampled frame indices (before the odd tail is repeated).
    pub indices: Vec<usize>,
    /// The video's frame count and frame rate.
    pub total_frames: usize,
    pub fps: f64,
}

impl PreparedVideo {
    /// Image tokens over all groups.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.groups.iter().map(PreparedImage::tokens).sum()
    }

    fn bytes(&self) -> usize {
        self.groups
            .iter()
            .map(|group| 2 * group.image.pixels.len())
            .sum()
    }
}

/// The text before a frame group's tokens.
#[must_use]
pub fn timestamp_text(seconds: f64) -> String {
    format!("<{seconds:.1} seconds>")
}

/// `sample_frames`: the frame indices sampled from `total` frames at `fps`.
#[must_use]
pub fn sample_indices(total: usize, fps: f64) -> Vec<usize> {
    let wanted = (total as f64 / fps * FPS) as usize;
    let count = wanted.clamp(MIN_FRAMES, MAX_FRAMES).min(total);
    if count <= 1 {
        return vec![0; count];
    }
    // numpy's linspace: `i * step`, the last exactly `total - 1`.
    let last = (total - 1) as f64;
    let step = last / (count - 1) as f64;
    (0..count)
        .map(|index| {
            let value = if index == count - 1 {
                last
            } else {
                index as f64 * step
            };
            round_half_even(value)
        })
        .collect()
}

/// The video `smart_resize` with the per-frame cap: `(height, width)` of
/// every frame of a clip of `frames` sampled frames.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for an aspect ratio above 200.
pub fn smart_resize(frames: usize, height: usize, width: usize) -> crate::Result<(usize, usize)> {
    let factor = RESIZE_FACTOR;
    let per_frame = (MAX_FRAME_TOKENS * factor * factor)
        .min(MAX_PIXELS / frames.max(1))
        .max(MIN_PIXELS * 105 / 100);
    let max_pixels = per_frame * frames;
    // The reference refuses a single frame; it is repeated instead.
    let frames = frames.max(TEMPORAL);
    let (mut height, mut width) = (height, width);
    if height < factor || width < factor {
        let scale = (factor as f64 / height as f64).max(factor as f64 / width as f64);
        height = (height as f64 * scale) as usize;
        width = (width as f64 * scale) as usize;
    }
    let (h, w) = (height as f64, width as f64);
    if h.max(w) / h.min(w) > MAX_ASPECT {
        return Err(Error::InvalidArgument(format!(
            "video of {width}x{height} has an aspect ratio above {MAX_ASPECT}"
        )));
    }
    let f = factor as f64;
    let mut h_bar = round_half_even(h / f) * factor;
    let mut w_bar = round_half_even(w / f) * factor;
    let t_bar = round_half_even(frames as f64 / TEMPORAL as f64) * TEMPORAL;
    let pixels = (frames * height * width) as f64;
    if t_bar * h_bar * w_bar > max_pixels {
        let beta = (pixels / max_pixels as f64).sqrt();
        h_bar = factor.max((h / beta / f).floor() as usize * factor);
        w_bar = factor.max((w / beta / f).floor() as usize * factor);
    } else if t_bar * h_bar * w_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / pixels).sqrt();
        h_bar = (h * beta / f).ceil() as usize * factor;
        w_bar = (w * beta / f).ceil() as usize * factor;
    }
    Ok((h_bar, w_bar))
}

/// Each frame group's time: the mean of its two frames' `index / fps`.
#[must_use]
pub fn timestamps(indices: &[usize], fps: f64) -> Vec<f64> {
    indices
        .chunks(TEMPORAL)
        .map(|pair| {
            let first = pair[0] as f64 / fps;
            let last = *pair.last().unwrap_or(&pair[0]) as f64 / fps;
            f64::midpoint(first, last)
        })
        .collect()
}

/// Group resized `frames` (sampled at `indices` of a `total`-frame video at
/// `fps`) into temporal patch groups.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when there are no frames or their sizes differ.
pub fn group_frames(
    mut frames: Vec<RgbImage>,
    indices: Vec<usize>,
    total_frames: usize,
    fps: f64,
) -> crate::Result<PreparedVideo> {
    let first = frames
        .first()
        .ok_or_else(|| Error::InvalidArgument("the video has no frames".into()))?;
    let (height, width) = (first.height, first.width);
    if frames
        .iter()
        .any(|frame| (frame.height, frame.width) != (height, width))
        || height % RESIZE_FACTOR != 0
        || width % RESIZE_FACTOR != 0
    {
        return Err(Error::InvalidArgument(
            "video frames must share one resized size".into(),
        ));
    }
    if !frames.len().is_multiple_of(TEMPORAL)
        && let Some(last) = frames.last().cloned()
    {
        frames.push(last);
    }
    let mut padded = indices.clone();
    if !padded.len().is_multiple_of(TEMPORAL)
        && let Some(&last) = padded.last()
    {
        padded.push(last);
    }
    let (rows, cols) = (height / RESIZE_FACTOR, width / RESIZE_FACTOR);
    let mut groups = Vec::with_capacity(frames.len() / TEMPORAL);
    let mut frames = frames.into_iter();
    while let (Some(image), Some(next)) = (frames.next(), frames.next()) {
        let digest = Sha256::new()
            .chain_update(b"video-group")
            .chain_update((rows as u32).to_le_bytes())
            .chain_update((cols as u32).to_le_bytes())
            .chain_update(&image.pixels)
            .chain_update(&next.pixels)
            .finalize()
            .into();
        groups.push(PreparedImage {
            image,
            next: Some(next),
            rows,
            cols,
            digest,
        });
    }
    Ok(PreparedVideo {
        groups,
        timestamps: timestamps(&padded, fps),
        indices,
        total_frames,
        fps,
    })
}

/// Recently prepared videos, by SHA-256 of the file.
static CACHE: Mutex<Vec<([u8; 32], Arc<PreparedVideo>)>> = Mutex::new(Vec::new());

/// The prepared video of file digest `key`, if cached (now the newest).
fn cached(key: &[u8; 32]) -> Option<Arc<PreparedVideo>> {
    let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    let index = cache.iter().position(|(digest, _)| digest == key)?;
    let entry = cache.remove(index);
    let video = Arc::clone(&entry.1);
    cache.push(entry);
    drop(cache);
    Some(video)
}

/// Decode, sample and resize an encoded video file, from the cache when the
/// same file was prepared recently. `cancel` is checked before each sample
/// the decoder reads.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a file `AVFoundation` cannot read, one with
/// no video track, one longer than [`MAX_SECONDS`] or [`MAX_TRACK_FRAMES`],
/// or an unsupported shape; [`Error::Generation`] when cancelled.
pub fn prepare_video(
    bytes: &[u8],
    cancel: Option<&CancelToken>,
) -> crate::Result<Arc<PreparedVideo>> {
    let key: [u8; 32] = Sha256::digest(bytes).into();
    if let Some(video) = cached(&key) {
        return Ok(video);
    }
    let mut frames = Vec::new();
    let size = Cell::new((0, 0));
    let mut sampled = Vec::new();
    let probe = decode::decode(
        bytes,
        cancel,
        |probe| {
            sampled = sample_indices(probe.total_frames, probe.fps);
            size.set(smart_resize(sampled.len(), probe.height, probe.width)?);
            Ok(sampled.clone())
        },
        |frame| {
            let (height, width) = size.get();
            frames.push(resize_bicubic(frame, height, width));
            Ok(())
        },
    )?;
    let video = Arc::new(group_frames(
        frames,
        sampled,
        probe.total_frames,
        probe.fps,
    )?);
    let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    cache.retain(|(digest, _)| *digest != key);
    cache.push((key, Arc::clone(&video)));
    let mut bytes = cache.iter().map(|(_, video)| video.bytes()).sum::<usize>();
    while cache.len() > 1 && bytes > CACHE_BYTES {
        bytes -= cache.remove(0).1.bytes();
    }
    drop(cache);
    Ok(video)
}

/// Refuse a frame over the image limits ([`MAX_IMAGE_SIDE`] a side,
/// [`MAX_IMAGE_PIXELS`] in all).
fn check_frame(width: usize, height: usize) -> crate::Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidArgument(
            "the video track holds no frames".into(),
        ));
    }
    if width.max(height) > MAX_IMAGE_SIDE as usize || width * height > MAX_IMAGE_PIXELS {
        return Err(Error::InvalidArgument(format!(
            "video frames of {width}x{height} exceed {MAX_IMAGE_SIDE} pixels a side or \
             {MAX_IMAGE_PIXELS} in all"
        )));
    }
    Ok(())
}

/// The error of a cancelled decode.
fn cancelled() -> Error {
    Error::Generation("the request was cancelled".into())
}

/// What the first pass learns about a video track.
#[derive(Clone, Copy, Debug)]
pub struct Probe {
    pub total_frames: usize,
    /// `frames x timescale / duration`, as the reference decoders report it.
    pub fps: f64,
    pub width: usize,
    pub height: usize,
    /// The stream's `FFmpeg` colour-space name, when it declares one.
    pub color_space: Option<&'static str>,
}

/// Whether `bytes` look like a video file (ISO base media or Matroska),
/// rather than an image.
#[must_use]
pub fn is_video(bytes: &[u8]) -> bool {
    container_suffix(bytes).is_ok() || bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3])
}

/// The container suffix `AVFoundation` needs to open `bytes`, if it is an
/// ISO base media (MP4, MOV, M4V, 3GP) file. HEIF and AVIF images share
/// the container and are not videos.
fn container_suffix(bytes: &[u8]) -> crate::Result<&'static str> {
    let kind = bytes.get(4..8).unwrap_or_default();
    let brand = bytes.get(8..12).unwrap_or_default();
    match kind {
        b"ftyp"
            if matches!(
                brand,
                b"heic" | b"heix" | b"heim" | b"heis" | b"mif1" | b"msf1" | b"avif" | b"avis"
            ) =>
        {
            Err(Error::InvalidArgument(
                "HEIF and AVIF images are not supported".into(),
            ))
        }
        b"ftyp" if brand == b"qt  " => Ok(".mov"),
        b"ftyp" => Ok(".mp4"),
        b"moov" | b"mdat" | b"wide" | b"free" | b"skip" => Ok(".mov"),
        _ if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) => Err(Error::InvalidArgument(
            "Matroska and WebM videos are not supported; send MP4 or MOV (H.264 or HEVC)".into(),
        )),
        _ => Err(Error::InvalidArgument(
            "the video is not an MP4 or MOV file".into(),
        )),
    }
}

/// `AVFoundation` decoding.
///
/// [`decode::decode`] probes the track, lets its caller choose the frame
/// indices (ascending), and lends each selected frame, at its source size,
/// in order. A video whose decoder yields fewer frames than probed
/// repeats its last decoded frame for the rest.
#[allow(unsafe_code)]
pub mod decode {
    use objc2::rc::{Retained, autoreleasepool};
    use objc2::runtime::AnyObject;
    use objc2_av_foundation::{
        AVAsset, AVAssetReader, AVAssetReaderStatus, AVAssetReaderTrackOutput,
        AVAssetReferenceRestrictions, AVAssetTrack, AVMediaTypeVideo, AVURLAsset,
        AVURLAssetReferenceRestrictionsKey,
    };
    use objc2_core_media::{CMSampleBuffer, kCMFormatDescriptionExtension_YCbCrMatrix};
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
        CVPixelBufferGetHeight, CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType,
        CVPixelBufferGetPlaneCount, CVPixelBufferGetWidth, CVPixelBufferGetWidthOfPlane,
        CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
        kCVImageBufferYCbCrMatrix_ITU_R_709_2, kCVImageBufferYCbCrMatrix_ITU_R_2020,
        kCVImageBufferYCbCrMatrix_SMPTE_240M_1995, kCVPixelBufferPixelFormatTypeKey,
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    };
    use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

    use std::io::Write as _;

    use super::{MAX_SECONDS, MAX_TRACK_FRAMES, Probe, cancelled, check_frame, container_suffix};
    use crate::Error;
    use crate::bonsai_model::CancelToken;
    use crate::embedding::vision::preprocess::RgbImage;
    use crate::embedding::vision::video::{yuv2rgb_coefficients, yuv420_to_rgb};

    fn invalid(message: impl Into<String>) -> Error {
        Error::InvalidArgument(message.into())
    }

    /// One step of the decoding pass.
    enum Next {
        End,
        /// A sample buffer without an image.
        Marker,
        /// A frame that is not sampled.
        Skipped,
        Frame(RgbImage),
    }

    fn described(context: &str, error: Option<&NSError>) -> Error {
        let reason = error.map_or_else(
            || "unknown error".to_owned(),
            |error| error.localizedDescription().to_string(),
        );
        invalid(format!("{context}: {reason}"))
    }

    fn reader(
        asset: &AVAsset,
        track: &AVAssetTrack,
        settings: Option<&NSDictionary<NSString, AnyObject>>,
    ) -> crate::Result<(Retained<AVAssetReader>, Retained<AVAssetReaderTrackOutput>)> {
        // SAFETY: plain `AVFoundation` calls on objects this thread owns; the
        // settings dictionary holds an NSString key and an NSNumber value.
        unsafe {
            let reader = AVAssetReader::assetReaderWithAsset_error(asset)
                .map_err(|error| described("cannot read the video", Some(&error)))?;
            let output = AVAssetReaderTrackOutput::assetReaderTrackOutputWithTrack_outputSettings(
                track, settings,
            );
            output.setAlwaysCopiesSampleData(false);
            if !reader.canAddOutput(&output) {
                return Err(invalid("cannot decode the video track"));
            }
            reader.addOutput(&output);
            if !reader.startReading() {
                return Err(described(
                    "cannot read the video",
                    reader.error().as_deref(),
                ));
            }
            Ok((reader, output))
        }
    }

    fn finished(reader: &AVAssetReader) -> crate::Result<()> {
        // SAFETY: status and error are thread-safe properties.
        unsafe {
            if reader.status() == AVAssetReaderStatus::Failed {
                return Err(described(
                    "cannot decode the video",
                    reader.error().as_deref(),
                ));
            }
        }
        Ok(())
    }

    /// A `CFString` constant as an `NSString` (toll-free bridged).
    const fn ns(string: &objc2_core_foundation::CFString) -> &NSString {
        // SAFETY: CFString and NSString are toll-free bridged.
        unsafe { &*std::ptr::from_ref(string).cast::<NSString>() }
    }

    /// The `FFmpeg` colour-space name of the YCbCr matrix a sample's format
    /// description declares (`None` when the stream does not say, which
    /// swscale reads as BT.601). Decoded frames carry `VideoToolbox`'s guess
    /// instead (BT.709 for untagged HD), so the compressed sample is read.
    fn color_space(sample: &CMSampleBuffer) -> Option<&'static str> {
        // SAFETY: a valid sample buffer; the extension, when present, is a
        // CFString, toll-free bridged to NSString; the constants are
        // immutable framework strings.
        unsafe {
            let description = sample.format_description()?;
            let matrix = description.extension(kCMFormatDescriptionExtension_YCbCrMatrix)?;
            let matrix = &*std::ptr::from_ref(&*matrix).cast::<NSString>();
            if matrix == ns(kCVImageBufferYCbCrMatrix_ITU_R_709_2) {
                Some("bt709")
            } else if matrix == ns(kCVImageBufferYCbCrMatrix_SMPTE_240M_1995) {
                Some("smpte240m")
            } else if matrix == ns(kCVImageBufferYCbCrMatrix_ITU_R_2020) {
                Some("bt2020nc")
            } else {
                None
            }
        }
    }

    /// The RGB pixels of a decoded biplanar 4:2:0 frame of a stream with
    /// colour space `color_space`.
    fn rgb(buffer: &CVPixelBuffer, color_space: Option<&str>) -> crate::Result<RgbImage> {
        let format = CVPixelBufferGetPixelFormatType(buffer);
        let full_range = format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
        if !full_range && format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
            return Err(invalid("the video decoder did not produce 4:2:0 frames"));
        }
        // The decoded size, not the track header's: check it again, and
        // that both planes hold it.
        let (width, height) = (
            CVPixelBufferGetWidth(buffer),
            CVPixelBufferGetHeight(buffer),
        );
        check_frame(width, height)?;
        let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
        if CVPixelBufferGetPlaneCount(buffer) != 2
            || CVPixelBufferGetWidthOfPlane(buffer, 0) < width
            || CVPixelBufferGetHeightOfPlane(buffer, 0) < height
            || CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) < width
            || CVPixelBufferGetWidthOfPlane(buffer, 1) < chroma_width
            || CVPixelBufferGetHeightOfPlane(buffer, 1) < chroma_height
            || CVPixelBufferGetBytesPerRowOfPlane(buffer, 1) < 2 * chroma_width
        {
            return Err(invalid(
                "a decoded video frame's planes do not hold its size",
            ));
        }
        let mut planes = vec![0_u8; width * height + 2 * chroma_width * chroma_height];
        // SAFETY: each plane's base address is valid for its rows times its
        // stride while the buffer is locked read-only, and is only read.
        unsafe {
            if CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags::ReadOnly) != 0 {
                return Err(invalid("cannot read a decoded video frame"));
            }
            let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0).cast::<u8>();
            let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer, 1).cast::<u8>();
            let readable = !luma.is_null() && !chroma.is_null();
            if readable {
                let (y_plane, uv) = planes.split_at_mut(width * height);
                let (u_plane, v_plane) = uv.split_at_mut(chroma_width * chroma_height);
                let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
                for (row, out) in y_plane.chunks_exact_mut(width).enumerate() {
                    out.copy_from_slice(std::slice::from_raw_parts(luma.add(row * stride), width));
                }
                let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1);
                for row in 0..chroma_height {
                    let line =
                        std::slice::from_raw_parts(chroma.add(row * stride), 2 * chroma_width);
                    for (column, pair) in line.as_chunks::<2>().0.iter().enumerate() {
                        u_plane[row * chroma_width + column] = pair[0];
                        v_plane[row * chroma_width + column] = pair[1];
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags::ReadOnly);
            if !readable {
                return Err(invalid("cannot read a decoded video frame"));
            }
        }
        let coefficients = yuv2rgb_coefficients(color_space);
        RgbImage::new(
            width,
            height,
            yuv420_to_rgb(&planes, width, height, coefficients, full_range),
        )
    }

    /// See the module documentation of [`super`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for files `AVFoundation` cannot read, no
    /// video track, oversized frames, a track over [`MAX_SECONDS`] or
    /// [`MAX_TRACK_FRAMES`], or errors from the callbacks;
    /// [`Error::Generation`] once `cancel` is cancelled (checked before each
    /// sample is read).
    pub fn decode(
        bytes: &[u8],
        cancel: Option<&CancelToken>,
        select: impl FnOnce(&Probe) -> crate::Result<Vec<usize>>,
        mut frame: impl FnMut(&RgbImage) -> crate::Result<()>,
    ) -> crate::Result<Probe> {
        let suffix = container_suffix(bytes)?;
        let mut file = tempfile::Builder::new()
            .prefix("local-ai-video-")
            .suffix(suffix)
            .tempfile()?;
        file.write_all(bytes)?;
        file.flush()?;
        let path = file.path().to_string_lossy().into_owned();
        autoreleasepool(|_| {
            // SAFETY: `AVFoundation` objects created and used on this thread
            // only; `AVMediaTypeVideo` is an immutable framework constant.
            #[allow(deprecated)]
            let (asset, track) = unsafe {
                let asset = open(&path);
                let kind =
                    AVMediaTypeVideo.ok_or_else(|| invalid("AVFoundation is unavailable"))?;
                let track = asset
                    .tracksWithMediaType(kind)
                    .firstObject()
                    .ok_or_else(|| {
                        invalid("the file holds no video track AVFoundation can read")
                    })?;
                (asset, track)
            };
            let probe = probe(&asset, &track, cancel)?;
            let wanted = select(&probe)?;
            let Some(&last_wanted) = wanted.last() else {
                return Ok(probe);
            };
            // The decoder's own 4:2:0 layout, either range: no conversion.
            // SAFETY: an immutable framework constant.
            let key = ns(unsafe { kCVPixelBufferPixelFormatTypeKey });
            let formats = NSArray::from_retained_slice(&[
                NSNumber::new_u32(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
                NSNumber::new_u32(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
            ]);
            let settings = NSDictionary::<NSString, AnyObject>::from_slices(
                &[key],
                &[AsRef::<AnyObject>::as_ref(&*formats)],
            );
            let (reader, output) = reader(&asset, &track, Some(&settings))?;
            let mut index = 0;
            let mut wanted = wanted.iter().copied().peekable();
            let mut last = None;
            // Sample buffers read, frames or not: at most the probe's.
            let mut buffers = 0;
            while wanted.peek().is_some() && index <= last_wanted {
                if cancel.is_some_and(CancelToken::is_cancelled) {
                    return Err(cancelled());
                }
                buffers += 1;
                if buffers > MAX_TRACK_FRAMES {
                    return Err(too_long());
                }
                let next = autoreleasepool(|_| -> crate::Result<Next> {
                    // SAFETY: the reader is started and owned by this thread.
                    let Some(sample) = (unsafe { output.copyNextSampleBuffer() }) else {
                        return Ok(Next::End);
                    };
                    // SAFETY: as above.
                    let Some(buffer) = (unsafe { sample.image_buffer() }) else {
                        return Ok(Next::Marker);
                    };
                    if wanted.peek() == Some(&index) {
                        rgb(&buffer, probe.color_space).map(Next::Frame)
                    } else {
                        Ok(Next::Skipped)
                    }
                })?;
                match next {
                    Next::End => break,
                    Next::Marker => {}
                    Next::Skipped => index += 1,
                    Next::Frame(image) => {
                        while wanted.peek() == Some(&index) {
                            wanted.next();
                            frame(&image)?;
                        }
                        last = Some(image);
                        index += 1;
                    }
                }
            }
            // SAFETY: reading has stopped on this thread.
            unsafe { reader.cancelReading() };
            finished(&reader).or_else(|error| if last.is_some() { Ok(()) } else { Err(error) })?;
            let last = last.ok_or_else(|| invalid("the video decoded to no frames"))?;
            for _ in wanted {
                frame(&last)?;
            }
            Ok(probe)
        })
    }

    /// The asset of the file at `path`, resolving no reference to media
    /// outside it (a movie may name other local files or URLs as its media
    /// data; `AVFoundation` follows local ones by default).
    ///
    /// # Safety
    ///
    /// Call on the thread that uses the asset.
    pub(in crate::bonsai_vision) unsafe fn open(path: &str) -> Retained<AVURLAsset> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(path));
        let forbid = NSNumber::new_usize(AVAssetReferenceRestrictions::ForbidAll.0);
        // SAFETY: an immutable framework constant; the value is the NSNumber
        // the key documents.
        let options = NSDictionary::<NSString, AnyObject>::from_slices(
            &[unsafe { AVURLAssetReferenceRestrictionsKey }],
            &[AsRef::<AnyObject>::as_ref(&*forbid)],
        );
        // SAFETY: as documented for this function.
        unsafe { AVURLAsset::URLAssetWithURL_options(&url, Some(&options)) }
    }

    fn too_long() -> Error {
        invalid(format!(
            "the video is longer than {MAX_SECONDS} s or {MAX_TRACK_FRAMES} frames"
        ))
    }

    /// Count the track's frames and sum their durations without decoding,
    /// refusing a track over [`MAX_SECONDS`] or [`MAX_TRACK_FRAMES`] as soon
    /// as it passes either.
    fn probe(
        asset: &AVAsset,
        track: &AVAssetTrack,
        cancel: Option<&CancelToken>,
    ) -> crate::Result<Probe> {
        let (reader, output) = reader(asset, track, None)?;
        let mut frames = 0_u64;
        let (mut duration, mut timescale) = (0_i128, 0_i32);
        let mut buffers = 0_usize;
        let mut matrix = None;
        loop {
            if cancel.is_some_and(CancelToken::is_cancelled) {
                return Err(cancelled());
            }
            buffers += 1;
            if buffers > MAX_TRACK_FRAMES
                || frames > MAX_TRACK_FRAMES as u64
                || (timescale > 0 && duration as f64 / f64::from(timescale) > MAX_SECONDS)
            {
                return Err(too_long());
            }
            let more = autoreleasepool(|_| {
                // SAFETY: the reader is started and owned by this thread.
                let Some(sample) = (unsafe { output.copyNextSampleBuffer() }) else {
                    return false;
                };
                // SAFETY: a valid sample buffer.
                let (count, length) = unsafe { (sample.num_samples(), sample.duration()) };
                if count > 0 && length.timescale > 0 {
                    if frames == 0 {
                        matrix = color_space(&sample);
                    }
                    frames += count as u64;
                    if timescale == 0 {
                        timescale = length.timescale;
                    }
                    duration += i128::from(length.value) * i128::from(timescale)
                        / i128::from(length.timescale);
                }
                true
            });
            if !more {
                break;
            }
        }
        finished(&reader)?;
        // SAFETY: a valid, retained track.
        let size = unsafe { track.naturalSize() };
        let (width, height) = (size.width.abs() as usize, size.height.abs() as usize);
        if frames == 0 || duration <= 0 {
            return Err(invalid("the video track holds no frames"));
        }
        if frames > MAX_TRACK_FRAMES as u64 || duration as f64 / f64::from(timescale) > MAX_SECONDS
        {
            return Err(too_long());
        }
        check_frame(width, height)?;
        Ok(Probe {
            total_frames: frames as usize,
            fps: (i128::from(frames) * i128::from(timescale)) as f64 / duration as f64,
            width,
            height,
            color_space: matrix,
        })
    }
}

/// Decode `bytes` and return the sampled frames at their source size, with
/// the probe (for parity checks against the reference processor).
///
/// # Errors
///
/// As [`prepare_video`].
#[doc(hidden)]
pub fn decode_sampled(bytes: &[u8]) -> crate::Result<(Probe, Vec<usize>, Vec<RgbImage>)> {
    let mut frames = Vec::new();
    let mut sampled = Vec::new();
    let probe = decode::decode(
        bytes,
        None,
        |probe| {
            sampled = sample_indices(probe.total_frames, probe.fps);
            Ok(sampled.clone())
        },
        |frame| {
            frames.push(frame.clone());
            Ok(())
        },
    )?;
    Ok((probe, sampled, frames))
}
