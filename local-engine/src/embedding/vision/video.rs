//! Video files, decoded with the `FFmpeg` command-line tools.
//!
//! There is no maintained pure-Rust H.264/HEVC/VP9 decoder, so decoding a
//! video *file* is delegated to `ffprobe` and `ffmpeg` (looked up on `PATH`;
//! an external dependency, used only by [`decode_video_file`]). Callers that
//! decode frames themselves pass them to [`super::prepare_video`] instead.
//!
//! The reference decodes with torchcodec (`FFmpeg` libraries, exact seek mode),
//! so this mirrors its stream facts: a packet scan gives the frame count, the
//! first and last presentation times (`duration = max(pts + duration) −
//! min(pts)` in stream time base) and `fps = frames / duration`; frames are
//! numbered in presentation order and converted to RGB24.

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
}

#[derive(Deserialize)]
struct ProbePacket {
    pts: Option<i64>,
    duration: Option<i64>,
}

fn failure(message: impl Into<String>) -> Error {
    Error::InvalidArgument(format!("video: {}", message.into()))
}

/// Stream facts of the first video stream: timing and frame size.
///
/// # Errors
///
/// `ffprobe` missing or failing, no video stream, or no packets.
#[allow(clippy::suboptimal_flops)]
pub fn probe(path: &Path) -> crate::Result<(VideoTiming, usize, usize)> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,time_base:packet=pts,duration",
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
        .first()
        .ok_or_else(|| failure("no video stream"))?;
    let (num, den) = stream
        .time_base
        .split_once('/')
        .and_then(|(n, d)| Some((n.parse::<f64>().ok()?, d.parse::<f64>().ok()?)))
        .filter(|(n, d)| *n > 0.0 && *d > 0.0)
        .ok_or_else(|| failure("bad stream time base"))?;
    let base = num / den;
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
    // end - begin as torchcodec computes them (seconds each, then subtract).
    let duration = (last as f64 * base) - (first as f64 * base);
    let timing = VideoTiming {
        frames,
        fps: (duration > 0.0).then(|| frames as f64 / duration),
        duration: (duration > 0.0).then_some(duration),
    };
    Ok((timing, stream.width, stream.height))
}

/// Decode the frames at `indices` (presentation order, ascending) as RGB24.
///
/// # Errors
///
/// `ffmpeg` missing or failing, or fewer frames than an index needs.
pub fn decode_frames(
    path: &Path,
    width: usize,
    height: usize,
    indices: &[usize],
) -> crate::Result<Vec<RgbImage>> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(failure(format!("unsupported frame size {width}x{height}")));
    }
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-sws_flags",
            "bilinear",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| failure(format!("cannot run ffmpeg: {error}")))?;
    let frame_bytes = width * height * 3;
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
            while wanted.peek() == Some(&index) {
                frames.push(RgbImage::new(width, height, buffer.clone())?);
                wanted.next();
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
    let (timing, width, height) = probe(path)?;
    let indices = preprocess::sample_frame_indices(timing, VIDEO_FPS, VIDEO_MAX_FRAMES);
    let frames = decode_frames(path, width, height, &indices)?;
    frames
        .iter()
        .map(|frame| preprocess::patchify(frame, VIDEO_SOFT_TOKENS))
        .collect()
}
