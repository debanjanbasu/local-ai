use half::f16;

use super::AneKernels;
use crate::ane::{Surface, SurfaceLayout};
use crate::batch::CommandBatch;
use crate::bonsai::{PTQ1_BLOCK_BYTES, PTQ1_BLOCK_ELEMENTS, Ptq1Matrix};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn uniform(&mut self) -> f32 {
        (self.next() as f32 / (1_u64 << 31) as f32).mul_add(2.0, -1.0)
    }
}

fn setup() -> (MetalContext, AneKernels) {
    let context = MetalContext::new().expect("context");
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let kernels = AneKernels::new(&context, &shaders).expect("kernels");
    (context, kernels)
}

#[expect(unsafe_code)]
fn read<T: bytemuck::Pod>(buffer: &MetalBuffer) -> Vec<T> {
    // SAFETY: the batch that wrote the buffer has completed.
    unsafe { buffer.as_slice::<T>().to_vec() }
}

#[test]
fn pack_quantizes_each_token_channel_major_under_its_fp16_scale() {
    let (context, kernels) = setup();
    let (width, tokens, rows) = (5120_u32, 100_u32, 128_u32);
    let mut rng = Rng(7);
    let x = (0..tokens * width)
        .map(|i| rng.uniform() * if i % 977 == 0 { 40.0 } else { 3.0 })
        .collect::<Vec<f32>>();
    let input = MetalBuffer::from_slice(context.device(), &x).expect("input");
    let scales = Surface::new(&context, SurfaceLayout::new(1, rows, true)).expect("scales");
    let halves = [0, 1].map(|_| {
        Surface::new(&context, SurfaceLayout::new(width / 2, rows, false)).expect("codes")
    });
    // Padding must come out as zeros whatever the surfaces held.
    scales
        .buffer()
        .copy_from_bytes(&vec![0x55; scales.layout().bytes], 0);
    for half in &halves {
        half.buffer()
            .copy_from_bytes(&vec![0x55; half.layout().bytes], 0);
    }
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .pack(
            &mut batch,
            &input,
            width,
            tokens,
            rows,
            &scales,
            [&halves[0], &halves[1]],
        )
        .expect("pack");
    batch.commit_and_wait().expect("gpu");
    let tx = read::<f16>(scales.buffer());
    let codes = halves.each_ref().map(|half| read::<i8>(half.buffer()));
    let stride = halves[0].layout().stride;
    let (mut exact, mut total) = (0_usize, 0_usize);
    for t in 0..rows as usize {
        if t >= tokens as usize {
            assert_eq!(tx[t].to_f32(), 0.0, "padding scale {t}");
        } else {
            let row = &x[t * width as usize..(t + 1) * width as usize];
            let peak = row.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            assert_eq!(tx[t], f16::from_f32(peak / 127.0 * 128.0), "scale {t}");
        }
        for c in 0..width as usize {
            let code = codes[c / (width as usize / 2)][(c % (width as usize / 2)) * stride + t];
            if t >= tokens as usize {
                assert_eq!(code, 0, "padding code {t} {c}");
                continue;
            }
            let expected = (x[t * width as usize + c] * (128.0 / tx[t].to_f32()))
                .round_ties_even()
                .clamp(-127.0, 127.0);
            let difference = (f32::from(code) - expected).abs();
            assert!(difference <= 1.0, "code {t} {c}: {code} against {expected}");
            exact += usize::from(difference == 0.0);
            total += 1;
        }
    }
    assert!(
        exact * 1000 >= total * 999,
        "{exact} of {total} codes exact"
    );
}

fn trit(block: &[u8], element: usize) -> i32 {
    const POWERS: [u32; 5] = [1, 3, 9, 27, 81];
    let (index, power) = if element < 80 {
        (element % 16, element / 16)
    } else if element < 120 {
        (16 + (element - 80) % 8, (element - 80) / 8)
    } else {
        (24 + (element - 120) % 2, (element - 120) / 2)
    };
    let digit = (((u32::from(block[index]) * POWERS[power]) & 255) * 3) >> 8;
    i32::try_from(digit).expect("a trit digit is 0, 1 or 2") - 1
}

#[test]
fn stage_converts_ternary_rows_to_per_row_int8() {
    let (context, kernels) = setup();
    let (rows, blocks) = (16_u32, 6_u32);
    let mut rng = Rng(11);
    let mut packed = vec![0_u8; 2 + (rows * blocks) as usize * PTQ1_BLOCK_BYTES];
    // An odd 2-byte offset, as checkpoint tensors may have.
    let data = &mut packed[2..];
    for (index, block) in data
        .as_chunks_mut::<PTQ1_BLOCK_BYTES>()
        .0
        .iter_mut()
        .enumerate()
    {
        for byte in &mut block[..26] {
            *byte = rng.next() as u8;
        }
        let scale = if index % 7 == 3 {
            0.0
        } else {
            rng.uniform() * 0.05
        };
        block[26..].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    }
    let weights = MetalBuffer::from_slice(context.device(), &packed).expect("weights");
    let matrix =
        Ptq1Matrix::new(&weights, 2, rows, blocks * PTQ1_BLOCK_ELEMENTS as u32).expect("matrix");
    let segments = [(0_u32, 4_u32), (4, 2)];
    let outputs = segments.map(|(_, count)| {
        Surface::new(
            &context,
            SurfaceLayout::new(rows, count * PTQ1_BLOCK_ELEMENTS as u32, false),
        )
        .expect("codes")
    });
    let scales = Surface::new(&context, SurfaceLayout::new(rows, 1, true)).expect("scales");
    let mut batch = CommandBatch::new(&context).expect("batch");
    for (index, ((first, count), output)) in segments.iter().zip(&outputs).enumerate() {
        kernels
            .stage(
                &mut batch,
                matrix,
                *first,
                *count,
                output,
                (index == 0).then_some(&scales),
            )
            .expect("stage");
    }
    batch.commit_and_wait().expect("gpu");
    let data = &packed[2..];
    let scale_rows = read::<f16>(scales.buffer());
    for row in 0..rows as usize {
        let row_blocks = &data[row * blocks as usize * PTQ1_BLOCK_BYTES..];
        let block_scale = |b: usize| {
            let block = &row_blocks[b * PTQ1_BLOCK_BYTES..];
            f16::from_le_bytes([block[26], block[27]]).to_f32()
        };
        let peak = (0..blocks as usize).fold(0.0_f32, |m, b| m.max(block_scale(b).abs()));
        assert_eq!(
            scale_rows[row * scales.layout().stride / 2],
            f16::from_f32(peak / 127.0 * 128.0),
            "row {row} scale"
        );
        for ((first, count), output) in segments.iter().zip(&outputs) {
            let codes = read::<i8>(output.buffer());
            for b in 0..*count as usize {
                let block = &row_blocks[(*first as usize + b) * PTQ1_BLOCK_BYTES..];
                let q =
                    (block_scale(*first as usize + b) * (127.0 / peak)).round_ties_even() as i32;
                for element in 0..PTQ1_BLOCK_ELEMENTS {
                    let code = codes[row * output.layout().stride + b * 128 + element];
                    assert_eq!(
                        i32::from(code),
                        trit(block, element) * q,
                        "row {row} block {} element {element}",
                        *first as usize + b
                    );
                }
            }
        }
    }
}

#[test]
fn join_applies_both_token_scales_and_flags_non_finite_rows() {
    let (context, kernels) = setup();
    let (width, tokens) = (256_u32, 70_u32);
    let mut rng = Rng(3);
    let partial = Surface::new(&context, SurfaceLayout::new(width + 1, 128, true)).expect("y");
    let scales = Surface::new(&context, SurfaceLayout::new(1, 128, true)).expect("tx");
    let stride = partial.layout().stride / 2;
    let mut y = vec![f16::ZERO; partial.layout().bytes / 2];
    for value in &mut y {
        *value = f16::from_f32(rng.uniform() * 4.0);
    }
    let tx = (0..scales.layout().bytes / 2)
        .map(|_| f16::from_f32(rng.uniform().abs() + 0.1))
        .collect::<Vec<_>>();
    partial
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&y), 0);
    scales
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&tx), 0);
    let output = MetalBuffer::empty(context.device(), (tokens * width) as usize * 4).expect("out");
    let status = MetalBuffer::from_slice(context.device(), &[0_u32]).expect("status");
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .join(
            &mut batch, &partial, &scales, &output, &status, width, tokens,
        )
        .expect("join");
    batch.commit_and_wait().expect("gpu");
    let out = read::<f32>(&output);
    for t in 0..tokens as usize {
        for c in 0..width as usize {
            let expected = y[c * stride + t].to_f32()
                * (y[width as usize * stride + t].to_f32() * tx[t].to_f32());
            assert_eq!(out[t * width as usize + c], expected, "{t} {c}");
        }
    }
    assert_eq!(read::<u32>(&status)[0], 0);
    // A non-finite value past the block is ignored; one inside it is flagged.
    y[3 * stride + tokens as usize + 1] = f16::INFINITY;
    partial
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&y), 0);
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .join(
            &mut batch, &partial, &scales, &output, &status, width, tokens,
        )
        .expect("join");
    batch.commit_and_wait().expect("gpu");
    assert_eq!(read::<u32>(&status)[0], 0);
    y[3 * stride + 5] = f16::NAN;
    partial
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&y), 0);
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .join(
            &mut batch, &partial, &scales, &output, &status, width, tokens,
        )
        .expect("join");
    batch.commit_and_wait().expect("gpu");
    assert_eq!(read::<u32>(&status)[0], 1);
}

#[test]
fn join_split_takes_the_ane_channels_then_the_gpu_rest() {
    let (context, kernels) = setup();
    let (width, split, tokens) = (256_u32, 96_u32, 70_u32);
    let mut rng = Rng(4);
    // Taller than the split, as the engine's surfaces are.
    let partial = Surface::new(&context, SurfaceLayout::new(width, 128, true)).expect("y");
    let scales = Surface::new(&context, SurfaceLayout::new(1, 128, true)).expect("tx");
    let stride = partial.layout().stride / 2;
    let mut y = vec![f16::ZERO; partial.layout().bytes / 2];
    for value in &mut y {
        *value = f16::from_f32(rng.uniform() * 4.0);
    }
    let tx = (0..scales.layout().bytes / 2)
        .map(|_| f16::from_f32(rng.uniform().abs() + 0.1))
        .collect::<Vec<_>>();
    let rest = (0..tokens * (width - split))
        .map(|_| rng.uniform())
        .collect::<Vec<_>>();
    partial
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&y), 0);
    scales
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&tx), 0);
    let rest_buffer = MetalBuffer::from_slice(context.device(), &rest).expect("rest");
    let output = MetalBuffer::empty(context.device(), (tokens * width) as usize * 4).expect("out");
    let status = MetalBuffer::from_slice(context.device(), &[0_u32]).expect("status");
    let join = |status: &MetalBuffer| {
        let mut batch = CommandBatch::new(&context).expect("batch");
        kernels
            .join_split(
                &mut batch,
                &partial,
                &scales,
                &rest_buffer,
                &output,
                status,
                width,
                split,
                tokens,
            )
            .expect("join");
        batch.commit_and_wait().expect("gpu");
    };
    join(&status);
    let out = read::<f32>(&output);
    let (width, split) = (width as usize, split as usize);
    for t in 0..tokens as usize {
        for c in 0..width {
            let expected = if c < split {
                y[c * stride + t].to_f32() * tx[t].to_f32()
            } else {
                rest[t * (width - split) + c - split]
            };
            assert_eq!(out[t * width + c], expected, "{t} {c}");
        }
    }
    assert_eq!(read::<u32>(&status)[0], 0);
    // Rows past the split and tokens past the block are not read.
    y[(split + 1) * stride + 3] = f16::NAN;
    y[3 * stride + tokens as usize + 1] = f16::INFINITY;
    partial
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&y), 0);
    join(&status);
    assert_eq!(read::<u32>(&status)[0], 0);
    y[(split - 1) * stride + 5] = f16::NAN;
    partial
        .buffer()
        .copy_from_bytes(bytemuck::cast_slice(&y), 0);
    join(&status);
    assert_eq!(read::<u32>(&status)[0], 1);
}
