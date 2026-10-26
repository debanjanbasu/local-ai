//! Video files, decoded with the `FFmpeg` command-line tools.
//!
//! There is no maintained pure-Rust H.264/HEVC/VP9 decoder, so decoding a
//! video *file* is delegated to `ffprobe` and `ffmpeg` (looked up on `PATH`;
//! an external dependency, used only by [`decode_video_file`]). Callers that
//! decode frames themselves pass them to [`super::prepare_video`] instead.
//!
//! The reference decodes with torchcodec (`FFmpeg` libraries, exact seek mode),
//! so this mirrors it:
//!
//! - stream facts from a packet scan: the frame count, `duration = max(pts +
//!   duration) − min(pts)` (each `pts · num / den` seconds) and `fps = frames
//!   / duration`; frames numbered in presentation order;
//! - frames of 8-bit 4:2:0 streams (`yuv420p`, `yuvj420p`) are taken from
//!   `ffmpeg` as raw planes (codec decoding is exact) and converted to RGB here
//!   with swscale's fixed-point `yuv2rgb` arithmetic (16-bit coefficients,
//!   `pmulhw`, nearest chroma) and the stream's colour matrix (BT.601 when
//!   unspecified; limited range unless `yuvj`), which is what torchcodec's
//!   `sws_scale` produced for the reference (bit-identical on its fixtures).
//!   Other pixel formats fall back to `ffmpeg`'s own RGB24 conversion, which
//!   can differ from the reference by a few levels.

use std::io::Read as _;
use std::path::Path;
use std::process::{Command, Stdio};

use serde::Deserialize;

use super::preprocess::{self, ImagePatches, RgbImage, VideoTiming};
use super::{VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_SOFT_TOKENS};
use crate::Error;

/// Largest accepted video frame side, in pixels.
const MAX_SIDE: usize = 8192;

#[derive(Deserialize)]
struct Probe {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    #[serde(default)]
    packets: Vec<ProbePacket>,
}

#[derive(Deserialize)]
struct ProbeStream {
    width: usize,
    height: usize,
    time_base: String,
    #[serde(default)]
    pix_fmt: Option<String>,
    #[serde(default)]
    color_space: Option<String>,
}

#[derive(Deserialize)]
struct ProbePacket {
    pts: Option<i64>,
    duration: Option<i64>,
}

/// What [`probe`] learns about the first video stream.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoStream {
    pub timing: VideoTiming,
    pub width: usize,
    pub height: usize,
    /// `FFmpeg` pixel format name (`yuv420p`, …), if reported.
    pub pixel_format: Option<String>,
    /// `FFmpeg` colour-space name (`bt709`, …), if reported.
    pub color_space: Option<String>,
}

fn failure(message: impl Into<String>) -> Error {
    Error::InvalidArgument(format!("video: {}", message.into()))
}

/// Stream facts of the first video stream.
///
/// # Errors
///
/// `ffprobe` missing or failing, no video stream, or no packets.
pub fn probe(path: &Path) -> crate::Result<VideoStream> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,time_base,pix_fmt,color_space:packet=pts,duration",
            "-of",
            "json",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| failure(format!("cannot run ffprobe: {error}")))?;
    if !output.status.success() {
        return Err(failure(format!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let probe: Probe = serde_json::from_slice(&output.stdout)?;
    let stream = probe
        .streams
        .into_iter()
        .next()
        .ok_or_else(|| failure("no video stream"))?;
    let (num, den) = stream
        .time_base
        .split_once('/')
        .and_then(|(n, d)| Some((n.parse::<f64>().ok()?, d.parse::<f64>().ok()?)))
        .filter(|(n, d)| *n > 0.0 && *d > 0.0)
        .ok_or_else(|| failure("bad stream time base"))?;
    let mut first = i64::MAX;
    let mut last = i64::MIN;
    let mut frames = 0;
    for packet in &probe.packets {
        let Some(pts) = packet.pts else { continue };
        frames += 1;
        first = first.min(pts);
        last = last.max(pts + packet.duration.unwrap_or(0));
    }
    if frames == 0 {
        return Err(failure("no video packets"));
    }
    // end - begin as torchcodec computes them (pts * num / den seconds each).
    let seconds = |pts: i64| pts as f64 * num / den;
    let duration = seconds(last) - seconds(first);
    Ok(VideoStream {
        timing: VideoTiming {
            frames,
            fps: (duration > 0.0).then(|| frames as f64 / duration),
            duration: (duration > 0.0).then_some(duration),
        },
        width: stream.width,
        height: stream.height,
        pixel_format: stream.pix_fmt,
        color_space: stream.color_space,
    })
}

/// swscale's `ff_yuv2rgb_coeffs` row for an `FFmpeg` colour-space name
/// (`sws_getCoefficients`: unknown and unsupported spaces use BT.601).
pub(crate) fn yuv2rgb_coefficients(color_space: Option<&str>) -> [i64; 4] {
    const BT709: [i64; 4] = [117_489, 138_438, 13_975, 34_925];
    const BT601: [i64; 4] = [104_597, 132_201, 25_675, 53_279];
    match color_space {
        Some("gbr" | "bt709") => BT709,
        Some("fcc") => [104_448, 132_798, 24_759, 53_109],
        Some("smpte240m") => [117_579, 136_230, 16_907, 35_559],
        Some("bt2020nc" | "bt2020c") => [110_013, 140_363, 12_277, 42_626],
        _ => BT601,
    }
}

/// swscale's fixed-point YUV 4:2:0 → RGB24 conversion (the x86 `yuv2rgb`
/// path: coefficients rounded to 16 bits, `(a · b) >> 16` products,
/// saturating 16-bit adds, nearest chroma), default contrast and saturation.
#[expect(clippy::many_single_char_names)]
pub(crate) fn yuv420_to_rgb(
    planes: &[u8],
    width: usize,
    height: usize,
    coefficients: [i64; 4],
    full_range: bool,
) -> Vec<u8> {
    let round16 = |f: i64| ((f + (1 << 15)) >> 16).clamp(-32_768, 32_767) as i32;
    let sat16 = |x: i32| x.clamp(-32_768, 32_767);
    let [mut crv, mut cbu, cgu, cgv] = coefficients;
    let (mut cgu, mut cgv) = (-cgu, -cgv);
    let (mut cy, mut oy) = (1_i64 << 16, 0_i64);
    if full_range {
        crv = crv * 224 / 255;
        cbu = cbu * 224 / 255;
        cgu = cgu * 224 / 255;
        cgv = cgv * 224 / 255;
    } else {
        cy = cy * 255 / 219;
        oy = 16 << 16;
    }
    let y_coeff = round16(cy * (1 << 13));
    let vr = round16(crv * (1 << 13));
    let ub = round16(cbu * (1 << 13));
    let vg = round16(cgv * (1 << 13));
    let ug = round16(cgu * (1 << 13));
    let y_offset = round16(oy * 8);
    let chroma_width = width.div_ceil(2);
    let chroma_plane = chroma_width * height.div_ceil(2);
    let (luma, chroma) = planes.split_at(width * height);
    let (u_plane, v_plane) = chroma.split_at(chroma_plane);
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let c = (y / 2) * chroma_width + x / 2;
            let yy = sat16((i32::from(luma[y * width + x]) << 3) - y_offset);
            let u = sat16((i32::from(u_plane[c]) << 3) - 1024);
            let v = sat16((i32::from(v_plane[c]) << 3) - 1024);
            let l = (yy * y_coeff) >> 16;
            let r = sat16(l + ((v * vr) >> 16));
            let g = sat16(l + sat16(((u * ug) >> 16) + ((v * vg) >> 16)));
            let b = sat16(l + ((u * ub) >> 16));
            rgb.extend([r, g, b].map(|channel| channel.clamp(0, 255) as u8));
        }
    }
    rgb
}

/// Decode the frames at `indices` (presentation order, ascending; repeats
/// allowed) of `stream` as RGB.
///
/// # Errors
///
/// `ffmpeg` missing or failing, or fewer frames than an index needs.
pub fn decode_frames(
    path: &Path,
    stream: &VideoStream,
    indices: &[usize],
) -> crate::Result<Vec<RgbImage>> {
    let (width, height) = (stream.width, stream.height);
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(failure(format!("unsupported frame size {width}x{height}")));
    }
    let planar = matches!(stream.pixel_format.as_deref(), Some("yuv420p" | "yuvj420p"));
    let pixel_format = if planar {
        stream.pixel_format.as_deref().unwrap_or("yuv420p")
    } else {
        "rgb24"
    };
    let frame_bytes = if planar {
        width * height + 2 * width.div_ceil(2) * height.div_ceil(2)
    } else {
        width * height * 3
    };
    let coefficients = yuv2rgb_coefficients(stream.color_space.as_deref());
    let full_range = stream.pixel_format.as_deref() == Some("yuvj420p");
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| failure(format!("cannot run ffmpeg: {error}")))?;
    let mut frames = Vec::with_capacity(indices.len());
    let mut wanted = indices.iter().copied().peekable();
    let mut buffer = vec![0_u8; frame_bytes];
    let result = (|| -> crate::Result<()> {
        let stdout = child
            .stdout
            .as_mut()
            .ok_or_else(|| failure("ffmpeg has no output"))?;
        let mut index = 0;
        while let Some(&next) = wanted.peek() {
            stdout
                .read_exact(&mut buffer)
                .map_err(|_| failure(format!("stream ended before frame {next}")))?;
            if wanted.peek() == Some(&index) {
                let pixels = if planar {
                    yuv420_to_rgb(&buffer, width, height, coefficients, full_range)
                } else {
                    buffer.clone()
                };
                while wanted.peek() == Some(&index) {
                    frames.push(RgbImage::new(width, height, pixels.clone())?);
                    wanted.next();
                }
            }
            index += 1;
        }
        Ok(())
    })();
    let _ = child.kill();
    let _ = child.wait();
    result?;
    Ok(frames)
}

/// Decode, sample (1 frame/s, at most 32) and preprocess a video file as the
/// reference video processor does.
///
/// # Errors
///
/// As [`probe`], [`decode_frames`] and [`preprocess::patchify`].
pub fn decode_video_file(path: &Path) -> crate::Result<Vec<ImagePatches>> {
    let stream = probe(path)?;
    let indices = preprocess::sample_frame_indices(stream.timing, VIDEO_FPS, VIDEO_MAX_FRAMES);
    let frames = decode_frames(path, &stream, &indices)?;
    frames
        .iter()
        .map(|frame| preprocess::patchify(frame, VIDEO_SOFT_TOKENS))
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn yuv_conversion_follows_swscale_fixed_point() {
        // 2x2 frame, one chroma sample: limited-range BT.601 grey and colour.
        let planes = [16, 235, 128, 81, 90, 240];
        let rgb = yuv420_to_rgb(&planes, 2, 2, yuv2rgb_coefficients(None), false);
        // Y 16 / 235 with neutral chroma would be black / white; here U 90,
        // V 240 push red up and blue down.
        assert_eq!(rgb.len(), 12);
        assert!(rgb[0] > rgb[2] && rgb[3] == 255);
        let grey = yuv420_to_rgb(
            &[16, 235, 126, 126, 128, 128],
            2,
            2,
            yuv2rgb_coefficients(None),
            false,
        );
        assert_eq!(&grey[..3], &[0, 0, 0]);
        assert_eq!(&grey[3..6], &[255, 255, 255]);
        assert_eq!(grey[6], grey[7]);
    }
}
