use std::fs;
use std::path::{Path, PathBuf};

use half::f16;

use local_core::config::Gemma4MultimodalConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedRgbImage {
    pub width: u32,
    pub height: u32,
    /// Packed RGB888 pixels in row-major order: `width * height * 3` bytes.
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcmAudio {
    /// Interleaved or mono f32 PCM samples. Values are expected to be in
    /// `[-1, 1]`; out-of-range values are accepted and passed through.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedVideoFrame {
    pub image: DecodedRgbImage,
    pub timestamp_seconds: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MediaInput {
    Image {
        path: PathBuf,
    },
    DecodedImage {
        image: DecodedRgbImage,
    },
    Audio {
        path: PathBuf,
    },
    PcmAudio {
        audio: PcmAudio,
    },
    /// A sampled video input. `path` may be a single decoded frame image or a
    /// directory of decoded frame images; container decode stays with the
    /// app/OS media stack.
    Video {
        path: PathBuf,
    },
    DecodedVideo {
        frames: Vec<DecodedVideoFrame>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalPrompt {
    pub text: String,
    pub media: Vec<MediaInput>,
}

#[derive(Debug, Clone)]
pub struct SoftTokenOverride {
    pub position: usize,
    pub embedding: Vec<f16>,
}

#[derive(Debug, Clone)]
pub struct PreparedMultimodalPrompt {
    pub tokens: Vec<u32>,
    /// Token IDs used for PLE. Gemma 4 uses PAD-token identity for media soft
    /// slots and masks the main embedding stream separately.
    pub ple_tokens: Vec<u32>,
    pub soft_tokens: Vec<SoftTokenOverride>,
}

impl MultimodalPrompt {
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            media: Vec::new(),
        }
    }

    #[must_use]
    pub fn has_images(&self) -> bool {
        self.media.iter().any(|media| {
            matches!(
                media,
                MediaInput::Image { .. } | MediaInput::DecodedImage { .. }
            )
        })
    }

    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.media.iter().any(|media| {
            matches!(
                media,
                MediaInput::Audio { .. } | MediaInput::PcmAudio { .. }
            )
        })
    }

    #[must_use]
    pub fn has_video(&self) -> bool {
        self.media.iter().any(|media| {
            matches!(
                media,
                MediaInput::Video { .. } | MediaInput::DecodedVideo { .. }
            )
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct MultimodalSupport {
    pub config_declares_image: bool,
    pub config_declares_audio: bool,
    pub config_declares_video: bool,
    pub has_image_tensors: bool,
    pub has_audio_tensors: bool,
    pub has_video_tensors: bool,
}

impl MultimodalSupport {
    #[must_use]
    pub const fn supports_images(self) -> bool {
        self.config_declares_image && self.has_image_tensors
    }

    #[must_use]
    pub const fn supports_audio(self) -> bool {
        self.config_declares_audio && self.has_audio_tensors
    }

    #[must_use]
    pub const fn supports_video(self) -> bool {
        self.config_declares_video && self.has_video_tensors
    }

    #[must_use]
    pub const fn declares_multimodal(self) -> bool {
        self.config_declares_image || self.config_declares_audio || self.config_declares_video
    }
}

/// Number of Gemma 4 image soft-token placeholders to allocate for one image.
#[must_use]
pub fn image_soft_token_count(config: &Gemma4MultimodalConfig) -> usize {
    config.vision_soft_tokens_per_image.unwrap_or(280).max(1) as usize
}

/// Build the hard-token span that reserves one image's soft-token slots.
///
/// The `image_token_id` positions are later replaced with soft embeddings;
/// begin/end image tokens remain ordinary hard tokens.
#[must_use]
pub fn image_placeholder_tokens(config: &Gemma4MultimodalConfig) -> Option<Vec<u32>> {
    let image_token = config.image_token_id?;
    let mut out = Vec::with_capacity(image_soft_token_count(config) + 2);
    if let Some(boi) = config.boi_token_id {
        out.push(boi);
    }
    out.extend(std::iter::repeat_n(
        image_token,
        image_soft_token_count(config),
    ));
    if let Some(eoi) = config.eoi_token_id {
        out.push(eoi);
    }
    Some(out)
}

/// Number of Gemma 4 video soft-token placeholders to allocate for one sampled frame.
///
/// HF's `Gemma4VideoProcessor` defaults to 70; model configs may carry
/// an override when processor metadata is folded into the runtime config.
#[must_use]
pub fn video_soft_token_count(config: &Gemma4MultimodalConfig) -> usize {
    config.video_soft_tokens_per_frame.unwrap_or(70).max(1) as usize
}

/// Number of frames to sample from a decoded-frame directory for one video.
#[must_use]
pub fn video_frame_count(config: &Gemma4MultimodalConfig) -> usize {
    config.video_frames_per_video.unwrap_or(32).max(1) as usize
}

/// Build the hard-token span that reserves one sampled video's soft-token slots.
///
/// The caller inserts timestamp text before each frame span to mirror
/// `Gemma4Processor`'s `MM:SS <|image><|video|>...<image|>` layout.
#[must_use]
pub fn video_frame_placeholder_tokens(config: &Gemma4MultimodalConfig) -> Option<Vec<u32>> {
    let video_token = config.video_token_id?;
    let mut out = Vec::with_capacity(video_soft_token_count(config) + 2);
    if let Some(boi) = config.boi_token_id {
        out.push(boi);
    }
    out.extend(std::iter::repeat_n(
        video_token,
        video_soft_token_count(config),
    ));
    if let Some(eoi) = config.eoi_token_id {
        out.push(eoi);
    }
    Some(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampledVideoFrame {
    pub path: PathBuf,
    pub timestamp_seconds: usize,
}

/// Resolve a video input into decoded frame images.
///
/// A file is treated as a one-frame video; a directory is sampled uniformly after lexicographic sort.
/// Heavy container decode is intentionally left to the app/OS boundary.
pub fn sampled_video_frames(
    path: &Path,
    max_frames: usize,
) -> crate::Result<Vec<SampledVideoFrame>> {
    let max_frames = max_frames.max(1);
    if path.is_file() {
        return Ok(vec![SampledVideoFrame {
            path: path.to_path_buf(),
            timestamp_seconds: 0,
        }]);
    }
    if !path.is_dir() {
        return Err(crate::Error::InvalidArgument(format!(
            "video input {} must be a decoded frame image or directory of decoded frames",
            path.display()
        )));
    }

    let mut frames: Vec<PathBuf> = fs::read_dir(path)
        .map_err(|e| {
            crate::Error::InvalidArgument(format!(
                "failed to read video frame directory {}: {e}",
                path.display()
            ))
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && is_supported_frame_image(path))
        .collect();
    frames.sort();
    if frames.is_empty() {
        return Err(crate::Error::InvalidArgument(format!(
            "video frame directory {} contains no supported frame images (.png/.jpg/.jpeg/.webp)",
            path.display()
        )));
    }

    let total = frames.len();
    let take = total.min(max_frames);
    let mut sampled = Vec::with_capacity(take);
    for out_idx in 0..take {
        let src_idx = if take == total {
            out_idx
        } else {
            out_idx * (total - 1) / (take - 1).max(1)
        };
        sampled.push(SampledVideoFrame {
            path: frames[src_idx].clone(),
            // With pre-decoded frames we do not have container timestamps, so
            // use the sampled-frame ordinal as a stable one-frame/sec timeline.
            timestamp_seconds: out_idx,
        });
    }
    Ok(sampled)
}

fn is_supported_frame_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp"
            )
        })
}

/// Gemma 4 audio soft-token cadence is 25 tokens/sec at 16 kHz, i.e. one
/// decoder placeholder per 640 waveform samples (40 ms).
#[must_use]
pub const fn audio_soft_token_count_for_samples(samples_16khz: usize) -> usize {
    let count = samples_16khz.div_ceil(640);
    if count == 0 { 1 } else { count }
}

/// Build the hard-token span that reserves one audio's soft-token slots.
///
/// The `audio_token_id` positions are later replaced with soft embeddings;
/// begin/end audio tokens remain ordinary hard tokens.
#[must_use]
pub fn audio_placeholder_tokens(
    config: &Gemma4MultimodalConfig,
    soft_token_count: usize,
) -> Option<Vec<u32>> {
    let audio_token = config.audio_token_id?;
    let mut out = Vec::with_capacity(soft_token_count + 2);
    if let Some(boa) = config.boa_token_id {
        out.push(boa);
    }
    out.extend(std::iter::repeat_n(audio_token, soft_token_count.max(1)));
    if let Some(eoa) = config.eoa_token_id {
        out.push(eoa);
    }
    Some(out)
}

#[must_use]
pub fn tensor_names_indicate_image_support<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    names.into_iter().any(|name| {
        name.contains("vision")
            || name.contains("image")
            || name.contains("patch_embed")
            || name.contains("patch_embd")
            || name.contains("embed_vision")
            || name.contains("vision_tower")
            || name.starts_with("v.blk.")
            || name.starts_with("v.patch_")
            || name == "mm.input_projection.weight"
    })
}

#[must_use]
pub fn tensor_names_indicate_audio_support<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    names.into_iter().any(|name| {
        name.contains("audio")
            || name.contains("embed_audio")
            || name.contains("audio_tower")
            || name.contains("subsample_conv")
            || name.starts_with("a.blk.")
            || name.starts_with("a.conv1d.")
            || name == "mm.a.input_projection.weight"
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "local-ai-multimodal-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create temp dir");
        path
    }

    #[test]
    fn prompt_reports_requested_media_types() {
        let prompt = MultimodalPrompt {
            text: "describe".into(),
            media: vec![
                MediaInput::Image {
                    path: "image.png".into(),
                },
                MediaInput::Audio {
                    path: "speech.wav".into(),
                },
                MediaInput::Video {
                    path: "frames".into(),
                },
            ],
        };

        assert!(prompt.has_images());
        assert!(prompt.has_audio());
        assert!(prompt.has_video());
    }

    #[test]
    fn tensor_name_detection_finds_gemma4_towers() {
        assert!(tensor_names_indicate_image_support([
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight"
        ]));
        assert!(tensor_names_indicate_image_support([
            "embed_vision.embedding_projection.weight"
        ]));
        assert!(tensor_names_indicate_image_support([
            "v.patch_embd.weight",
            "mm.input_projection.weight"
        ]));
        assert!(tensor_names_indicate_audio_support([
            "model.audio_tower.layers.0.self_attn.q_proj.weight"
        ]));
        assert!(tensor_names_indicate_audio_support([
            "subsample_conv_projection.layer0.conv.weight"
        ]));
        assert!(tensor_names_indicate_audio_support([
            "a.blk.0.attn_q.weight",
            "mm.a.input_projection.weight"
        ]));
        assert!(!tensor_names_indicate_image_support([
            "blk.0.attn_q.weight",
            "token_embd.weight"
        ]));
        assert!(!tensor_names_indicate_audio_support([
            "blk.0.attn_q.weight",
            "token_embd.weight"
        ]));
    }

    #[test]
    fn support_requires_config_and_tensors() {
        let config_only = MultimodalSupport {
            config_declares_image: true,
            config_declares_audio: true,
            config_declares_video: true,
            has_image_tensors: false,
            has_audio_tensors: false,
            has_video_tensors: false,
        };
        assert!(!config_only.supports_images());
        assert!(!config_only.supports_audio());
        assert!(!config_only.supports_video());

        let ready = MultimodalSupport {
            has_image_tensors: true,
            has_audio_tensors: true,
            has_video_tensors: true,
            ..config_only
        };
        assert!(ready.supports_images());
        assert!(ready.supports_audio());
        assert!(ready.supports_video());
    }

    #[test]
    fn image_placeholder_uses_configured_soft_token_budget() {
        let config = Gemma4MultimodalConfig {
            boi_token_id: Some(255_999),
            image_token_id: Some(258_880),
            eoi_token_id: Some(258_882),
            vision_soft_tokens_per_image: Some(70),
            ..Gemma4MultimodalConfig::default()
        };

        let tokens = image_placeholder_tokens(&config).expect("image tokens");
        assert_eq!(tokens.len(), 72);
        assert_eq!(tokens[0], 255_999);
        assert_eq!(tokens[1], 258_880);
        assert_eq!(tokens[70], 258_880);
        assert_eq!(tokens[71], 258_882);
    }

    #[test]
    fn audio_placeholder_uses_duration_derived_soft_token_budget() {
        let config = Gemma4MultimodalConfig {
            boa_token_id: Some(256_000),
            audio_token_id: Some(258_881),
            eoa_token_id: Some(258_883),
            ..Gemma4MultimodalConfig::default()
        };
        assert_eq!(audio_soft_token_count_for_samples(16_000), 25);

        let tokens = audio_placeholder_tokens(&config, 25).expect("audio tokens");
        assert_eq!(tokens.len(), 27);
        assert_eq!(tokens[0], 256_000);
        assert_eq!(tokens[1], 258_881);
        assert_eq!(tokens[25], 258_881);
        assert_eq!(tokens[26], 258_883);
    }

    #[test]
    fn video_frame_placeholder_uses_video_soft_tokens() {
        let config = Gemma4MultimodalConfig {
            boi_token_id: Some(255_999),
            video_token_id: Some(258_884),
            eoi_token_id: Some(258_882),
            video_soft_tokens_per_frame: Some(70),
            ..Gemma4MultimodalConfig::default()
        };

        let tokens = video_frame_placeholder_tokens(&config).expect("video frame tokens");
        assert_eq!(tokens.len(), 72);
        assert_eq!(tokens[0], 255_999);
        assert_eq!(tokens[1], 258_884);
        assert_eq!(tokens[70], 258_884);
        assert_eq!(tokens[71], 258_882);
    }

    #[test]
    fn sampled_video_frames_sort_and_subsample_decoded_frames() {
        let dir = temp_dir("frames");
        for name in ["003.png", "001.jpg", "002.webp", "ignore.txt", "004.jpeg"] {
            fs::write(dir.join(name), []).expect("write frame marker");
        }

        let frames = sampled_video_frames(&dir, 2).expect("sample frames");
        fs::remove_dir_all(&dir).expect("remove temp dir");

        assert_eq!(frames.len(), 2);
        assert_eq!(
            frames[0].path.file_name().and_then(|n| n.to_str()),
            Some("001.jpg")
        );
        assert_eq!(
            frames[1].path.file_name().and_then(|n| n.to_str()),
            Some("004.jpeg")
        );
        assert_eq!(frames[0].timestamp_seconds, 0);
        assert_eq!(frames[1].timestamp_seconds, 1);
    }
}
