use std::sync::Arc;

use super::preprocess::smart_resize;
use super::*;

fn span(start: usize, rows: usize, cols: usize) -> ImageSpan {
    ImageSpan {
        start,
        rows,
        cols,
        embedding: vec![0.0; rows * cols * WIDTH].into(),
    }
}

/// Positions as `get_rope_index` assigns them, worked by hand: text 0..3,
/// a 2x3 image at 3..9 (t0 3, next text at 3 + max(2, 3) = 6), text at
/// 9..11 (6, 7), a 4x1 image at 11..15 (t0 8, next text at 8 + 4 = 12).
#[test]
fn rope_positions_follow_the_processor_layout() {
    let images = PromptImages::new(vec![span(3, 2, 3), span(11, 4, 1)]).expect("spans");
    let expected = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [3, 3, 4],
        [3, 3, 5],
        [3, 4, 3],
        [3, 4, 4],
        [3, 4, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
        [8, 9, 8],
        [8, 10, 8],
        [8, 11, 8],
        [12, 12, 12],
        [13, 13, 13],
    ];
    for (position, want) in expected.iter().enumerate() {
        let got = images.rope(position);
        assert_eq!(&got[..3], want, "position {position}");
    }
    assert_eq!(images.text_offset(0, 3), Some(0));
    assert_eq!(
        images.text_offset(2, 2),
        None,
        "the block reaches the image"
    );
    assert_eq!(images.text_offset(9, 2), Some(-3));
    assert_eq!(images.text_offset(15, 40), Some(-3));
    assert!(images.covers(8) && !images.covers(9) && images.covers(14));
    assert!(PromptImages::new(vec![span(3, 2, 3), span(8, 1, 1)]).is_err());
}

#[test]
fn rotation_is_the_inverse_of_the_input_transform() {
    let signs = (0..WIDTH)
        .map(|index| if (index * 7 + 3) % 5 < 2 { -1.0 } else { 1.0 })
        .collect::<Vec<f32>>();
    let original = (0..2 * WIDTH)
        .map(|index| ((index * 37 % 101) as f32 - 50.0) / 17.0)
        .collect::<Vec<f32>>();
    let mut rows = original.clone();
    rotate_rows(&mut rows, &signs);
    // The GPU inverse: x = s * H(y) / 32 per 1024 block, by the definition
    // H[i][j] = (-1)^popcount(i & j).
    for (row, want) in rows.chunks(WIDTH).zip(original.chunks(WIDTH)) {
        for block in 0..WIDTH / 1024 {
            let y = &row[block * 1024..(block + 1) * 1024];
            for i in (0..1024).step_by(97) {
                let sum = y
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| {
                        let sign = if (i & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        f64::from(value) * sign
                    })
                    .sum::<f64>();
                let x = f64::from(signs[block * 1024 + i]) * sum / 32.0;
                assert!((x - f64::from(want[block * 1024 + i])).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn smart_resize_matches_the_processor() {
    // No resize: both sides already multiples of 32.
    assert_eq!(smart_resize(480, 640, 16_777_216).expect("ok"), (480, 640));
    // Half to even: 1648 / 32 = 51.5 rounds to 52, 1616 / 32 = 50.5 to 50.
    assert_eq!(
        smart_resize(1648, 1616, 16_777_216).expect("ok"),
        (1664, 1600)
    );
    // The minimum-pixels branch rounds up.
    assert_eq!(smart_resize(5, 7, 16_777_216).expect("ok"), (224, 320));
    // The maximum-pixels branch rounds down.
    assert_eq!(
        smart_resize(1800, 2400, 4_194_304).expect("ok"),
        (1760, 2336)
    );
    assert_eq!(smart_resize(384, 512, 262_144).expect("ok"), (384, 512));
    assert_eq!(smart_resize(1800, 2400, 262_144).expect("ok"), (416, 576));
    assert!(smart_resize(10, 2010, 16_777_216).is_err());
}

#[test]
fn expansion_replaces_each_pad_with_its_image_tokens() {
    let image = |rows, cols, seed: u8| PreparedImage {
        image: crate::embedding::vision::preprocess::RgbImage::new(1, 1, vec![0, 0, 0])
            .expect("image"),
        rows,
        cols,
        digest: [seed; 32],
    };
    let (a, b) = (image(2, 3, 1), image(1, 2, 2));
    let rows = |image: &PreparedImage| -> Arc<[f32]> { vec![0.5; image.tokens() * WIDTH].into() };
    let projector = [9; 32];
    let ids = [
        10,
        VISION_START,
        IMAGE_PAD,
        VISION_END,
        11,
        VISION_START,
        IMAGE_PAD,
        VISION_END,
    ];
    let (expanded, images) =
        expand_prompt(&ids, &projector, &[(&a, rows(&a)), (&b, rows(&b))]).expect("expand");
    assert_eq!(expanded.len(), ids.len() + 5 + 1);
    assert_eq!(
        expanded_len(&ids, &[a.clone(), b]).expect("len"),
        expanded.len()
    );
    assert_eq!(&expanded[..2], &[10, VISION_START]);
    assert!(expanded[2..8].iter().all(|&id| is_image_token(id)));
    assert_eq!(&expanded[8..11], &[VISION_END, 11, VISION_START]);
    assert_eq!(expanded[2], image_token_id(&projector, &a.digest, 0));
    assert_ne!(expanded[2], expanded[3]);
    assert_ne!(expanded[11], image_token_id(&projector, &a.digest, 0));
    let spans = images.spans();
    assert_eq!((spans[0].start, spans[1].start), (2, 11));
    // A different projector gives different ids for the same image.
    assert_ne!(image_token_id(&[8; 32], &a.digest, 0), expanded[2]);
    // Text that spells a pad cannot shift the images.
    let typed = [IMAGE_PAD, VISION_START, IMAGE_PAD, VISION_END];
    assert!(expand_prompt(&typed, &projector, &[(&a, rows(&a))]).is_err());
    assert!(check_tokens(&expanded, 0, Some(&images)).is_ok());
    assert!(check_tokens(&expanded, 1, Some(&images)).is_err());
    assert!(check_tokens(&expanded, 0, None).is_err());
}
