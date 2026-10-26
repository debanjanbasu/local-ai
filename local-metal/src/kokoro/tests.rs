#![expect(
    unsafe_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use super::*;

fn gpu_or_skip() -> Option<(MetalContext, KokoroKernels)> {
    let context = MetalContext::new();
    if matches!(context, Err(crate::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    let context = context.expect("Metal context failed for a reason other than a missing device");
    let shaders = ShaderLibrary::new(context.device()).expect("shader library");
    let kernels = KokoroKernels::new(&context, &shaders).expect("kokoro kernels");
    Some((context, kernels))
}

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 20_000) as f32 / 10_000.0 - 1.0
        })
        .collect()
}

fn buffer(context: &MetalContext, data: &[f32]) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), data).expect("buffer")
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: {a} vs {e}"
        );
    }
}

/// CPU `Conv1d`: x `[cin, t_in]`, w `[cout, cin, k]` (the `PyTorch` layout).
#[expect(clippy::too_many_arguments)]
fn conv_reference(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    cin: usize,
    cout: usize,
    k: usize,
    t_in: usize,
    stride: usize,
    dilation: usize,
    pad: usize,
) -> Vec<f32> {
    let t_out = (t_in + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let mut y = vec![0.0; cout * t_out];
    for o in 0..cout {
        for t in 0..t_out {
            let mut acc = f64::from(b[o]);
            for i in 0..cin {
                for tap in 0..k {
                    let src = (t * stride + tap * dilation) as isize - pad as isize;
                    if src >= 0 && (src as usize) < t_in {
                        acc += f64::from(w[(o * cin + i) * k + tap])
                            * f64::from(x[i * t_in + src as usize]);
                    }
                }
            }
            y[o * t_out + t] = acc as f32;
        }
    }
    y
}

/// [cout, cin, k] -> [cout, k, cin], the kernel's layout.
fn to_kernel_layout(w: &[f32], cout: usize, cin: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    for o in 0..cout {
        for i in 0..cin {
            for tap in 0..k {
                out[(o * k + tap) * cin + i] = w[(o * cin + i) * k + tap];
            }
        }
    }
    out
}

#[test]
fn conv_matches_reference_for_strides_dilations_and_ragged_tiles() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    for (cin, cout, k, t_in, stride, dilation, pad) in [
        (
            37_usize, 45_usize, 3_usize, 71_usize, 1_usize, 1_usize, 1_usize,
        ),
        (22, 33, 12, 120, 6, 1, 3),
        (40, 40, 7, 90, 1, 5, 15),
        (1, 1, 3, 50, 2, 1, 1),
        (70, 20, 1, 33, 1, 1, 0),
    ] {
        let x = values(cin * t_in, 1);
        let w = values(cout * cin * k, 2);
        let b = values(cout, 3);
        let expected = conv_reference(&x, &w, &b, cin, cout, k, t_in, stride, dilation, pad);
        let t_out = expected.len() / cout;
        let (xb, wb, bb) = (
            buffer(&context, &x),
            buffer(&context, &to_kernel_layout(&w, cout, cin, k)),
            buffer(&context, &b),
        );
        let yb = MetalBuffer::empty(context.device(), expected.len() * 4).expect("y");
        let mut batch = CommandBatch::new(&context).expect("batch");
        kernels
            .conv(
                &mut batch,
                View::new(&wb, 0),
                Some(View::new(&bb, 0)),
                View::new(&xb, 0),
                View::new(&yb, 0),
                Conv {
                    cout: cout as u32,
                    cin: cin as u32,
                    taps: k as u32,
                    columns: t_out as u32,
                    in_stride: stride as i32,
                    in_dil: dilation as i32,
                    in_offset: -(pad as i32),
                    in_len: t_in as u32,
                    in_row: t_in as u32,
                    out_stride: 1,
                    out_offset: 0,
                    out_row: t_out as u32,
                    flags: Conv::BIAS,
                    scale: 1.0,
                },
            )
            .expect("conv");
        batch.commit_and_wait().expect("gpu");
        // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
        close(
            &unsafe { yb.as_slice::<f32>() }[..expected.len()],
            &expected,
            1e-4,
        );
    }
}

#[test]
fn transposed_conv_phases_match_reference() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    // ConvTranspose1d(cin, cout, K=12, stride 6, padding 3).
    let (cin, cout, k, s, pad, t_in) = (9_usize, 7_usize, 12_usize, 6_usize, 3_usize, 11_usize);
    let x = values(cin * t_in, 4);
    let w = values(cin * cout * k, 5); // [cin, cout, k]
    let b = values(cout, 6);
    let t_out = (t_in - 1) * s - 2 * pad + k;
    let mut expected = vec![0.0_f32; cout * t_out];
    for o in 0..cout {
        for t in 0..t_out {
            let mut acc = f64::from(b[o]);
            for i in 0..cin {
                for j in 0..t_in {
                    let tap = t as isize + pad as isize - (j * s) as isize;
                    if (0..k as isize).contains(&tap) {
                        acc += f64::from(w[(i * cout + o) * k + tap as usize])
                            * f64::from(x[i * t_in + j]);
                    }
                }
            }
            expected[o * t_out + t] = acc as f32;
        }
    }
    // Phase layout [s][cout][k/s][cin] with tap kk = r + kk * s.
    let kp = k / s;
    let mut phased = vec![0.0_f32; w.len()];
    for r in 0..s {
        for o in 0..cout {
            for kk in 0..kp {
                for i in 0..cin {
                    phased[((r * cout + o) * kp + kk) * cin + i] =
                        w[(i * cout + o) * k + r + kk * s];
                }
            }
        }
    }
    let (xb, wb, bb) = (
        buffer(&context, &x),
        buffer(&context, &phased),
        buffer(&context, &b),
    );
    let yb = MetalBuffer::empty(context.device(), expected.len() * 4).expect("y");
    let mut batch = CommandBatch::new(&context).expect("batch");
    for r in 0..s {
        let q0 = ((pad as i32 - r as i32).max(0) + s as i32 - 1) / s as i32;
        let first = q0 * s as i32 + r as i32 - pad as i32;
        let columns = (t_out - 1 - first as usize) / s + 1;
        kernels
            .conv(
                &mut batch,
                View::new(&wb, 0).at(r * cout * kp * cin),
                Some(View::new(&bb, 0)),
                View::new(&xb, 0),
                View::new(&yb, 0),
                Conv {
                    cout: cout as u32,
                    cin: cin as u32,
                    taps: kp as u32,
                    columns: columns as u32,
                    in_stride: 1,
                    in_dil: -1,
                    in_offset: q0,
                    in_len: t_in as u32,
                    in_row: t_in as u32,
                    out_stride: s as u32,
                    out_offset: first as u32,
                    out_row: t_out as u32,
                    flags: Conv::BIAS,
                    scale: 1.0,
                },
            )
            .expect("conv");
    }
    batch.commit_and_wait().expect("gpu");
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(
        &unsafe { yb.as_slice::<f32>() }[..expected.len()],
        &expected,
        1e-4,
    );
}

#[test]
fn lstm_matches_reference_in_both_directions() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let h = LSTM_HIDDEN as usize;
    let steps = 9_usize;
    let gates: Vec<f32> = values(8 * h * steps, 7).iter().map(|v| v * 0.5).collect();
    let whh: Vec<f32> = values(2 * 4 * h * h, 8).iter().map(|v| v * 0.05).collect(); // [dir][r][j]
    let mut whh_t = vec![0.0_f32; whh.len()];
    for d in 0..2 {
        for r in 0..4 * h {
            for j in 0..h {
                whh_t[(d * h + j) * 4 * h + r] = whh[(d * 4 * h + r) * h + j];
            }
        }
    }
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut expected = vec![0.0_f32; 2 * h * steps];
    for d in 0..2 {
        let (mut hs, mut cs) = (vec![0.0_f64; h], vec![0.0_f64; h]);
        for s in 0..steps {
            let t = if d == 0 { s } else { steps - 1 - s };
            let g: Vec<f64> = (0..4 * h)
                .map(|r| {
                    let mut acc = f64::from(gates[(d * 4 * h + r) * steps + t]);
                    for j in 0..h {
                        acc += f64::from(whh[(d * 4 * h + r) * h + j]) * hs[j];
                    }
                    acc
                })
                .collect();
            for j in 0..h {
                let (i, f, c, o) = (
                    sigmoid(g[j]),
                    sigmoid(g[h + j]),
                    g[2 * h + j].tanh(),
                    sigmoid(g[3 * h + j]),
                );
                cs[j] = f * cs[j] + i * c;
                hs[j] = o * cs[j].tanh();
                expected[(d * h + j) * steps + t] = hs[j] as f32;
            }
        }
    }
    let (gb, wb) = (buffer(&context, &gates), buffer(&context, &whh_t));
    let out = MetalBuffer::empty(context.device(), expected.len() * 4).expect("out");
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .lstm(
            &mut batch,
            View::new(&gb, 0),
            View::new(&wb, 0),
            View::new(&out, 0),
            steps as u32,
            steps as u32,
            steps as u32,
        )
        .expect("lstm");
    batch.commit_and_wait().expect("gpu");
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(
        &unsafe { out.as_slice::<f32>() }[..expected.len()],
        &expected,
        1e-4,
    );
}

#[test]
fn instance_norm_snake_and_istft_match_reference() {
    let Some((context, kernels)) = gpu_or_skip() else {
        return;
    };
    let (channels, columns) = (5_usize, 300_usize);
    let x = values(channels * columns, 9);
    let scale = values(channels, 10);
    let shift = values(channels, 11);
    let alpha: Vec<f32> = values(channels, 12).iter().map(|v| 0.5 + v.abs()).collect();
    let mut expected = vec![0.0_f32; x.len()];
    for c in 0..channels {
        let row = &x[c * columns..(c + 1) * columns];
        let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / columns as f64;
        let var = row
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / columns as f64;
        for t in 0..columns {
            let n = (f64::from(row[t]) - mean) / (var + 1e-5).sqrt() * f64::from(scale[c])
                + f64::from(shift[c]);
            let a = f64::from(alpha[c]);
            expected[c * columns + t] = (n + (a * n).sin().powi(2) / a) as f32;
        }
    }
    let (xb, sb, hb, ab) = (
        buffer(&context, &x),
        buffer(&context, &scale),
        buffer(&context, &shift),
        buffer(&context, &alpha),
    );
    let mut batch = CommandBatch::new(&context).expect("batch");
    kernels
        .instance_norm(
            &mut batch,
            View::new(&xb, 0),
            View::new(&xb, 0),
            View::new(&sb, 0),
            View::new(&hb, 0),
            Some(View::new(&ab, 0)),
            channels as u32,
            columns as u32,
            (columns as u32, columns as u32),
            Activation::Snake,
        )
        .expect("norm");
    // iSTFT of a random head: compare with a direct overlap-add.
    let frames = 40_usize;
    let head: Vec<f32> = values(22 * frames, 13).iter().map(|v| v * 0.5).collect();
    let hb2 = buffer(&context, &head);
    let samples = 5 * (frames - 1);
    let audio = MetalBuffer::empty(context.device(), samples * 4).expect("audio");
    kernels
        .istft(
            &mut batch,
            View::new(&hb2, 0),
            View::new(&audio, 0),
            frames as u32,
            frames as u32,
        )
        .expect("istft");
    batch.commit_and_wait().expect("gpu");
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(&unsafe { xb.as_slice::<f32>() }[..x.len()], &expected, 1e-4);
    let window: Vec<f64> = (0..20)
        .map(|m| 0.5 - 0.5 * (std::f64::consts::TAU * m as f64 / 20.0).cos())
        .collect();
    let padded = samples + 20;
    let (mut acc, mut env) = (vec![0.0_f64; padded], vec![0.0_f64; padded]);
    for f in 0..frames {
        for m in 0..20 {
            let mut v = 0.0;
            for k in 0..11 {
                let mag = f64::from(head[k * frames + f]).exp();
                let phase = f64::from(head[(11 + k) * frames + f]).sin();
                let (re, im) = (mag * phase.cos(), mag * phase.sin());
                let angle = std::f64::consts::TAU * (k * m) as f64 / 20.0;
                let weight = if k == 0 || k == 10 { 1.0 } else { 2.0 };
                v += weight
                    * (re * angle.cos()
                        - if k == 0 || k == 10 {
                            0.0
                        } else {
                            im * angle.sin()
                        });
            }
            v /= 20.0;
            if f * 5 + m < padded {
                acc[f * 5 + m] += window[m] * v;
                env[f * 5 + m] += window[m] * window[m];
            }
        }
    }
    let expected_audio: Vec<f32> = (0..samples)
        .map(|n| (acc[n + 10] / env[n + 10]) as f32)
        .collect();
    // SAFETY: no GPU work on this buffer is in flight and nothing else touches it meanwhile.
    close(
        &unsafe { audio.as_slice::<f32>() }[..samples],
        &expected_audio,
        1e-3,
    );
}
