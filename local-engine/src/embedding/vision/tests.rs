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
}
