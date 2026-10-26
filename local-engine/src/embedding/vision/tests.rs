#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::preprocess::*;

#[test]
fn target_sizes_follow_the_reference_rule() {
    // 280 tokens -> at most 2,520 patches; sides are multiples of 48.
    assert_eq!(target_size(300, 451, 2520).unwrap(), (624, 960));
    assert_eq!(target_size(768, 768, 2520).unwrap(), (768, 768));
    // Very wide: the height rounds to 0 and takes one pooling cell.
    assert_eq!(target_size(10, 4000, 2520).unwrap(), (48, 280 * 48));
    assert!(target_size(375, 500, 2520).unwrap().0.is_multiple_of(48));
    assert_eq!(
        RESCALE.to_bits(),
        (0.003_921_568_627_450_98_f64 as f32).to_bits()
    );
}

#[test]
fn frames_sample_at_one_per_second_then_uniformly() {
    let timing = VideoTiming {
        frames: 300,
        fps: Some(30.0),
        duration: Some(10.0),
    };
    assert_eq!(
        sample_frame_indices(timing, 1.0, 32),
        (0..10).map(|i| i * 30).collect::<Vec<_>>()
    );
    let long = VideoTiming {
        frames: 400,
        fps: Some(10.0),
        duration: Some(40.0),
    };
    let picked = sample_frame_indices(long, 1.0, 32);
    assert_eq!(picked.len(), 32);
    assert_eq!((picked[0], picked[31]), (0, 390));
    // numpy.linspace(0, 39, 32, dtype=int)[1] == 1, [2] == 2, [3] == 3, [4] == 5
    assert_eq!(&picked[..5], &[0, 10, 20, 30, 50]);
    let unknown = VideoTiming {
        frames: 5,
        fps: None,
        duration: None,
    };
    assert_eq!(sample_frame_indices(unknown, 1.0, 32), vec![0, 1, 2, 3, 4]);
}

#[test]
fn patches_are_raster_ordered_row_column_channel() {
    let (w, h) = (384, 384);
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
    let image = RgbImage::new(w, h, pixels.clone()).unwrap();
    // 384 x 384 is already the 70-token size (smaller images are scaled up).
    let patches = patchify(&image, 70).unwrap();
    assert_eq!((patches.columns, patches.rows), (24, 24));
    assert_eq!(patches.soft_tokens(), 64);
    // Patch 25 = (x 1, y 1); its row 2, column 5, channel 1.
    let value = patches.values[25 * PATCH_VALUES + (2 * 16 + 5) * 3 + 1];
    let source = pixels[((16 + 2) * w + 16 + 5) * 3 + 1];
    assert_eq!(value, f32::from(source) * RESCALE);
    assert_eq!(patches.position(26), (2, 1));
}

#[test]
fn resizing_a_constant_image_keeps_it_constant() {
    let image = RgbImage::new(37, 23, vec![200; 37 * 23 * 3]).unwrap();
    let out = resize_bicubic(&image, 96, 144);
    assert!(out.pixels.iter().all(|&v| v == 200));
    let down = resize_bicubic(
        &RgbImage::new(500, 333, vec![7; 500 * 333 * 3]).unwrap(),
        48,
        96,
    );
    assert!(down.pixels.iter().all(|&v| v == 7));
}

#[test]
fn unsupported_budgets_and_bad_buffers_are_refused() {
    let image = RgbImage::new(4, 4, vec![0; 48]).unwrap();
    assert!(patchify(&image, 100).is_err());
    assert!(RgbImage::new(4, 4, vec![0; 47]).is_err());
    assert!(decode_image(b"not an image").is_err());
    // A JPEG signature with a corrupt header is an error, not an abort.
    assert!(decode_image(&[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', 0xFF, 0xC0]).is_err());
}

/// CRC-32 (IEEE) of a PNG chunk's type and data.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A PNG of `width` x `height` RGB pixels with no pixel data: its header
/// reads, its pixels do not decode.
pub fn png_header(width: u32, height: u32) -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = b"IHDR".to_vec();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    png.extend_from_slice(&13_u32.to_be_bytes());
    png.extend_from_slice(&ihdr);
    png.extend_from_slice(&crc32(&ihdr).to_be_bytes());
    for chunk in [&b"IDAT"[..], b"IEND"] {
        png.extend_from_slice(&0_u32.to_be_bytes());
        png.extend_from_slice(chunk);
        png.extend_from_slice(&crc32(chunk).to_be_bytes());
    }
    png
}

/// 10,000 x 10,000 is within the side limit but over the pixel limit: the
/// header is refused before 300 MB of pixels are allocated, for JPEG
/// (libjpeg-turbo) and the `image` crate's formats alike.
#[test]
fn images_over_the_pixel_limit_are_refused_from_their_header() {
    let mut jpeg = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vision/syn_wide_1600x90.jpg"
    ))
    .unwrap();
    // The SOF0 segment's height and width.
    assert_eq!(&jpeg[158..160], &[0xFF, 0xC0]);
    jpeg[163..167].copy_from_slice(&[0x27, 0x10, 0x27, 0x10]);
    let error = decode_image(&jpeg).expect_err("refused").to_string();
    assert!(error.contains("pixels"), "{error}");
    let error = decode_image(&png_header(10_000, 10_000))
        .expect_err("refused")
        .to_string();
    assert!(error.contains("exceeds 67108864 pixels"), "{error}");
}

#[derive(serde::Deserialize)]
struct Reference {
    images: Vec<ReferenceImage>,
}

#[derive(serde::Deserialize)]
struct ReferenceImage {
    file: String,
    columns: usize,
    rows: usize,
    pixel_values_u8_sha256: String,
    ids_len: usize,
    embedding: Vec<f32>,
}

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/vision");

fn reference() -> Reference {
    serde_json::from_str(
        &std::fs::read_to_string(format!("{FIXTURES}/reference.json")).expect("fixture"),
    )
    .expect("reference")
}

/// The committed images decode and preprocess to exactly the reference
/// processor's pixel values (transformers 5.19.0 `Gemma4ImageProcessor`,
/// fed the same files): a PNG upscaled from 7x5 and a 1600x90 JPEG (4:2:0,
/// libjpeg-turbo decode) downscaled to a 210 x 9 patch grid.
#[test]
fn committed_images_preprocess_exactly_like_the_reference() {
    use sha2::{Digest as _, Sha256};
    for image in reference().images {
        let bytes = std::fs::read(format!("{FIXTURES}/{}", image.file)).expect("image");
        let patches = super::prepare_image(&bytes).expect("prepare");
        assert_eq!(
            (patches.columns, patches.rows),
            (image.columns, image.rows),
            "{}",
            image.file
        );
        assert_eq!(patches.soft_tokens() + 4, image.ids_len);
        let quantized: Vec<u8> = patches
            .values
            .iter()
            .map(|v| (v * 255.0).round() as u8)
            .collect();
        assert!(
            patches
                .values
                .iter()
                .zip(&quantized)
                .all(|(v, q)| v.to_bits() == (f32::from(*q) * RESCALE).to_bits())
        );
        let digest = Sha256::digest(&quantized)
            .iter()
            .fold(String::new(), |mut out, b| {
                use std::fmt::Write as _;
                let _ = write!(out, "{b:02x}");
                out
            });
        assert_eq!(digest, image.pixel_values_u8_sha256, "{}", image.file);
    }
}

/// Native image embeddings of the committed images against the FP32
/// reference (the parity gate: cosine ≥ 0.999 each).
#[test]
#[ignore = "requires the installed embedding model with the vision artifact"]
fn native_image_embeddings_match_the_reference_fixture() {
    use crate::embedding::{EmbeddingModel, installed_model};
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let dir = installed_model(&std::path::Path::new(root).join("models")).expect("installed model");
    let mut model = EmbeddingModel::open(&dir).expect("open");
    for image in reference().images {
        let bytes = std::fs::read(format!("{FIXTURES}/{}", image.file)).expect("image");
        let patches = super::prepare_image(&bytes).expect("prepare");
        let native = model.embed_image(&patches, &mut || Ok(())).expect("embed");
        let cosine: f32 = native
            .iter()
            .zip(&image.embedding)
            .map(|(a, b)| a * b)
            .sum();
        assert!(cosine >= 0.999, "{}: cosine {cosine}", image.file);
    }
}
