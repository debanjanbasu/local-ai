//! The Kokoro-82M network on Metal: phoneme ids and a style vector in,
//! 24 kHz samples out (`kokoro/model.py` `KModel.forward_with_tokens`).
//!
//! Three GPU phases with CPU work between them, exactly as the reference
//! orders them: (A) PL-BERT, the duration encoder, the duration LSTM and the
//! text encoder, then durations are rounded on the CPU; (B) alignment, the
//! shared LSTM and the F0/energy branches, then the harmonic source is built
//! on the CPU from F0 ([`super::source`]); (C) the decoder, the iSTFTNet
//! generator and the inverse STFT. Work is submitted in command buffers of
//! bounded size with the caller's pacing hook between them.

use std::collections::BTreeMap;

use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::kokoro::{Activation, Conv, KokoroKernels, View};
use local_metal::shaders::ShaderLibrary;

use super::artifact::{Artifact, STYLE_WIDTH};
use super::source::{self, Noise};
use crate::Error;

const HIDDEN: usize = 512;
const STYLE: usize = 128;
const BERT: usize = 768;
const BERT_EMBED: usize = 128;
const BERT_HEADS: u32 = 12;
const BERT_FFN: usize = 2048;
const BERT_LAYERS: usize = 12;
const MAX_DUR: usize = 50;
const GATES: usize = 2048;
/// Longest input in token ids, boundary ids included (PL-BERT's positions).
pub(super) const MAX_IDS: usize = 512;
/// Most acoustic frames (40 per second) one call may produce. Bounds GPU
/// memory: the generator's last stage holds about five `[128, 120 * frames]`
/// F32 buffers (about 600 MB at this limit).
pub(super) const MAX_FRAMES: usize = 2000;
/// Multiply-accumulates encoded per command buffer before the pacing hook
/// runs (larger convolutions are split by columns). About 3 ms of GPU time
/// on an M4 Pro, so a decode step waiting behind one speech submission is
/// delayed by about that much.
const SUBMIT_MACS: u64 = 4_000_000_000;
const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
const BERT_EPS: f32 = 1e-12;
const NORM_EPS: f32 = 1e-5;

/// One synthesized chunk.
pub struct Synthesis {
    /// Mono F32 samples at 24 kHz.
    pub samples: Vec<f32>,
    /// Frames per input id (each frame is 600 samples).
    pub durations: Vec<u32>,
}

/// Named intermediate activations, `[channels, time]` flattened, for parity
/// checks against the reference (`kokoro-ref/fixtures/intermediates`).
pub type Taps = BTreeMap<String, Vec<f32>>;

struct StyleFc {
    name: String,
    channels: usize,
    decoder: bool,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

pub(super) struct Network {
    context: MetalContext,
    kernels: KokoroKernels,
    weights: MetalBuffer,
    artifact: Artifact,
    word: Vec<f32>,
    position: Vec<f32>,
    token_type: Vec<f32>,
    embed_norm: (Vec<f32>, Vec<f32>),
    text_embedding: Vec<f32>,
    style_fcs: Vec<StyleFc>,
    source_linear: ([f32; 9], f32),
}

// SAFETY: Metal devices, queues, pipelines and buffers may be transferred
// between threads. The network owns every object exclusively and is `!Sync`,
// so nothing is encoded concurrently; sharing goes through a mutex.
#[expect(unsafe_code)]
#[expect(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Network {}

/// The pacing wrapper: one live command batch and the work encoded into it.
struct Pacer<'a> {
    context: &'a MetalContext,
    batch: CommandBatch,
    work: u64,
    pace: &'a mut dyn FnMut() -> crate::Result<()>,
}

impl Pacer<'_> {
    fn add(&mut self, macs: u64) -> crate::Result<()> {
        self.work += macs;
        if self.work >= SUBMIT_MACS {
            self.flush()?;
        }
        Ok(())
    }

    /// Submit, wait, and run the pacing hook.
    fn flush(&mut self) -> crate::Result<()> {
        self.batch.commit_and_renew(self.context)?;
        self.work = 0;
        (self.pace)()
    }

    /// Submit and wait without pacing (before CPU reads).
    fn sync(&mut self) -> crate::Result<()> {
        self.batch.commit_and_renew(self.context)?;
        self.work = 0;
        Ok(())
    }
}

/// A GPU activation block `[channels, columns]` with row stride `row`.
#[derive(Clone, Copy)]
struct Act<'a> {
    view: View<'a>,
    channels: u32,
    columns: u32,
    row: u32,
}

impl<'a> Act<'a> {
    const fn new(buffer: &'a MetalBuffer, channels: usize, columns: usize) -> Self {
        Self {
            view: View::new(buffer, 0),
            channels: channels as u32,
            columns: columns as u32,
            row: columns as u32,
        }
    }

    /// Rows `start..start + channels` of this block.
    const fn rows(self, start: usize, channels: usize) -> Self {
        Self {
            view: self.view.at(start * self.row as usize),
            channels: channels as u32,
            ..self
        }
    }
}

fn buffer(context: &MetalContext, floats: usize) -> crate::Result<MetalBuffer> {
    Ok(MetalBuffer::empty(context.device(), floats.max(1) * 4)?)
}

#[expect(unsafe_code)]
fn read(act: Act<'_>) -> Vec<f32> {
    // SAFETY: callers read taps after the pacer has synced, and the borrow
    // ends with the copy below.
    let all: &[f32] = unsafe { act.view.buffer.as_slice() };
    let start = act.view.offset / 4;
    let mut out = Vec::with_capacity(act.channels as usize * act.columns as usize);
    for c in 0..act.channels as usize {
        let at = start + c * act.row as usize;
        out.extend_from_slice(&all[at..at + act.columns as usize]);
    }
    out
}

fn write(act: Act<'_>, values: &[f32]) {
    let columns = act.columns as usize;
    for c in 0..act.channels as usize {
        let at = act.view.offset + c * act.row as usize * 4;
        act.view.buffer.copy_from_bytes(
            bytemuck::cast_slice(&values[c * columns..(c + 1) * columns]),
            at,
        );
    }
}

fn tap(taps: &mut Option<&mut Taps>, name: &str, act: Act<'_>) {
    if let Some(taps) = taps.as_deref_mut() {
        taps.insert(name.to_owned(), read(act));
    }
}

fn layer_norm(values: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for ((v, w), b) in values.iter_mut().zip(weight).zip(bias) {
        *v = (*v - mean) * inv * w + b;
    }
}

impl Network {
    pub(super) fn open(artifact: Artifact) -> crate::Result<Self> {
        let context = MetalContext::new()?;
        let shaders = ShaderLibrary::new(context.device())?;
        let kernels = KokoroKernels::new(&context, &shaders)?;
        let weights = MetalBuffer::from_slice(context.device(), artifact.payload())?;
        let word = artifact.values("bert.embeddings.word_embeddings.weight", &[178, BERT_EMBED])?;
        let position = artifact.values(
            "bert.embeddings.position_embeddings.weight",
            &[MAX_IDS, BERT_EMBED],
        )?;
        let token_type = artifact.values(
            "bert.embeddings.token_type_embeddings.weight",
            &[2, BERT_EMBED],
        )?;
        let embed_norm = (
            artifact.values("bert.embeddings.LayerNorm.weight", &[BERT_EMBED])?,
            artifact.values("bert.embeddings.LayerNorm.bias", &[BERT_EMBED])?,
        );
        let text_embedding = artifact.values("text_encoder.embedding.weight", &[178, HIDDEN])?;
        let mut style_fcs = Vec::new();
        let names: Vec<String> = artifact
            .tensor_names()
            .filter(|n| n.ends_with(".fc.weight"))
            .map(str::to_owned)
            .collect();
        for name in names {
            let prefix = name.trim_end_matches(".fc.weight").to_owned();
            let rows = artifact.rows(&name)?;
            let weight = artifact.values(&name, &[rows, STYLE])?;
            let bias = artifact.values(&format!("{prefix}.fc.bias"), &[rows])?;
            style_fcs.push(StyleFc {
                decoder: prefix.starts_with("decoder."),
                name: prefix,
                channels: rows / 2,
                weight,
                bias,
            });
        }
        let linear = artifact.values("decoder.generator.m_source.l_linear.weight", &[1, 9])?;
        let linear_bias = artifact.values("decoder.generator.m_source.l_linear.bias", &[1])?[0];
        let mut source_weights = [0.0_f32; 9];
        source_weights.copy_from_slice(&linear);
        Ok(Self {
            context,
            kernels,
            weights,
            artifact,
            word,
            position,
            token_type,
            embed_norm,
            text_embedding,
            style_fcs,
            source_linear: (source_weights, linear_bias),
        })
    }

    pub(super) const fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    fn w(&self, name: &str, shape: &[usize]) -> crate::Result<View<'_>> {
        Ok(View::new(&self.weights, self.artifact.offset(name, shape)?))
    }

    /// Per-norm `(1 + gamma, beta)` from the style vector, one buffer.
    fn styles(&self, style: &[f32]) -> crate::Result<(MetalBuffer, BTreeMap<String, usize>)> {
        let mut values = Vec::new();
        let mut index = BTreeMap::new();
        for fc in &self.style_fcs {
            let s = if fc.decoder {
                &style[..STYLE]
            } else {
                &style[STYLE..]
            };
            let rows = fc.channels * 2;
            index.insert(fc.name.clone(), values.len());
            let start = values.len();
            values.resize(start + rows, 0.0);
            for (r, out) in values[start..].iter_mut().enumerate() {
                let w = &fc.weight[r * STYLE..(r + 1) * STYLE];
                let dot: f32 = w.iter().zip(s).map(|(a, b)| a * b).sum();
                *out = dot + fc.bias[r];
            }
            for gamma in &mut values[start..start + fc.channels] {
                *gamma += 1.0;
            }
        }
        Ok((
            MetalBuffer::from_slice(self.context.device(), &values)?,
            index,
        ))
    }

    /// Convolution `y = conv(x)` over `columns` outputs (split by columns
    /// into bounded submissions). `weight` is `[cout, taps, cin]` (taps 1
    /// for a linear map, stored `[cout, cin]` when `linear`).
    #[expect(clippy::too_many_arguments, clippy::many_single_char_names)]
    fn conv(
        &self,
        pacer: &mut Pacer<'_>,
        weight: &str,
        bias: bool,
        linear: bool,
        x: Act<'_>,
        taps: usize,
        geometry: (i32, i32, i32),
        y: Act<'_>,
        accumulate: bool,
        scale: f32,
    ) -> crate::Result<()> {
        let (stride, dil, pad) = geometry;
        let (cout, cin) = (y.channels as usize, x.channels as usize);
        let w = if linear {
            self.w(weight, &[cout, cin])?
        } else {
            self.w(weight, &[cout, taps, cin])?
        };
        let b = if bias {
            Some(self.w(
                &format!("{}.bias", weight.trim_end_matches(".weight")),
                &[cout],
            )?)
        } else {
            None
        };
        let p = Conv {
            cout: cout as u32,
            cin: cin as u32,
            taps: taps as u32,
            columns: y.columns,
            in_stride: stride,
            in_dil: dil,
            in_offset: -pad,
            in_len: x.columns,
            in_row: x.row,
            out_stride: 1,
            out_offset: 0,
            out_row: y.row,
            flags: (Conv::BIAS * u32::from(b.is_some()))
                | (Conv::ACCUMULATE * u32::from(accumulate)),
            scale,
        };
        self.dispatch_conv(pacer, w, b, x.view, y.view, p)
    }

    #[expect(clippy::many_single_char_names)]
    fn dispatch_conv(
        &self,
        pacer: &mut Pacer<'_>,
        w: View<'_>,
        b: Option<View<'_>>,
        x: View<'_>,
        y: View<'_>,
        p: Conv,
    ) -> crate::Result<()> {
        let per_column = u64::from(p.cout) * u64::from(p.cin) * u64::from(p.taps);
        let total = per_column * u64::from(p.columns);
        let pieces = total.div_ceil(SUBMIT_MACS).max(1);
        let step = (u64::from(p.columns).div_ceil(pieces)).next_multiple_of(32) as u32;
        let mut start = 0_u32;
        while start < p.columns {
            let columns = step.min(p.columns - start);
            let mut piece = p;
            piece.columns = columns;
            piece.in_offset = p.in_offset + start.cast_signed() * p.in_stride;
            piece.out_offset = p.out_offset + start * p.out_stride;
            self.kernels.conv(&mut pacer.batch, w, b, x, y, piece)?;
            pacer.add(per_column * u64::from(columns))?;
            start += columns;
        }
        Ok(())
    }

    /// `y = W x + b` for a `[cout, cin]` weight over every column of `x`.
    fn linear(
        &self,
        pacer: &mut Pacer<'_>,
        weight: &str,
        bias: bool,
        x: Act<'_>,
        y: Act<'_>,
        accumulate: bool,
    ) -> crate::Result<()> {
        self.conv(
            pacer,
            weight,
            bias,
            true,
            x,
            1,
            (1, 1, 0),
            y,
            accumulate,
            1.0,
        )
    }

    /// Bidirectional LSTM `prefix` over `x` into `out` (512 rows).
    fn lstm(
        &self,
        pacer: &mut Pacer<'_>,
        prefix: &str,
        x: Act<'_>,
        gates: &MetalBuffer,
        out: Act<'_>,
    ) -> crate::Result<()> {
        let g = Act::new(gates, GATES, x.columns as usize);
        self.linear(pacer, &format!("{prefix}.ih.weight"), true, x, g, false)?;
        let whh = self.w(&format!("{prefix}.whh_t"), &[2, 256, 1024])?;
        self.kernels.lstm(
            &mut pacer.batch,
            g.view,
            whh,
            out.view,
            x.columns,
            g.row,
            out.row,
        )?;
        pacer.add(u64::from(x.columns) * 2 * 256 * 1024)
    }

    fn style_view<'b>(
        styles: &'b (MetalBuffer, BTreeMap<String, usize>),
        name: &str,
        channels: usize,
    ) -> crate::Result<(View<'b>, View<'b>)> {
        let at = *styles
            .1
            .get(name)
            .ok_or_else(|| Error::MissingTensor(format!("{name}.fc")))?;
        let base = View::new(&styles.0, 0).at(at);
        Ok((base, base.at(channels)))
    }

    /// `AdainResBlk1d` (`istftnet.py`): x `[cin, T]` to out `[cout, T or 2T]`.
    #[expect(clippy::too_many_lines)]
    fn adain_res_block(
        &self,
        pacer: &mut Pacer<'_>,
        styles: &(MetalBuffer, BTreeMap<String, usize>),
        prefix: &str,
        x: Act<'_>,
        out: Act<'_>,
        upsample: bool,
    ) -> crate::Result<()> {
        let (cin, cout) = (x.channels as usize, out.channels as usize);
        let t = x.columns as usize;
        let t2 = if upsample { 2 * t } else { t };
        let a = buffer(&self.context, cin * t)?;
        let a = Act::new(&a, cin, t);
        let (scale, shift) = Self::style_view(styles, &format!("{prefix}.norm1"), cin)?;
        self.kernels.instance_norm(
            &mut pacer.batch,
            x.view,
            a.view,
            scale,
            shift,
            None,
            x.channels,
            x.columns,
            (x.row, a.row),
            Activation::Leaky(0.2),
        )?;
        let pooled;
        let mid_in = if upsample {
            pooled = buffer(&self.context, cin * t2)?;
            let p = Act::new(&pooled, cin, t2);
            self.kernels.depthwise_up(
                &mut pacer.batch,
                a.view,
                self.w(&format!("{prefix}.pool.weight"), &[cin, 1, 3])?,
                self.w(&format!("{prefix}.pool.bias"), &[cin])?,
                p.view,
                a.channels,
                a.columns,
                (a.row, p.row),
            )?;
            p
        } else {
            a
        };
        let mid = buffer(&self.context, cout * t2)?;
        let mid = Act::new(&mid, cout, t2);
        self.conv(
            pacer,
            &format!("{prefix}.conv1.weight"),
            true,
            false,
            mid_in,
            3,
            (1, 1, 1),
            mid,
            false,
            1.0,
        )?;
        let (scale, shift) = Self::style_view(styles, &format!("{prefix}.norm2"), cout)?;
        self.kernels.instance_norm(
            &mut pacer.batch,
            mid.view,
            mid.view,
            scale,
            shift,
            None,
            mid.channels,
            mid.columns,
            (mid.row, mid.row),
            Activation::Leaky(0.2),
        )?;
        self.conv(
            pacer,
            &format!("{prefix}.conv2.weight"),
            true,
            false,
            mid,
            3,
            (1, 1, 1),
            out,
            false,
            SQRT_HALF,
        )?;
        // Shortcut: nearest x2 upsampling, then the learned 1x1 when widths differ.
        let up;
        let shortcut = if upsample {
            up = buffer(&self.context, cin * t2)?;
            let u = Act::new(&up, cin, t2);
            let index: Vec<u32> = (0..t2 as u32).map(|i| i / 2).collect();
            let index = MetalBuffer::from_slice(self.context.device(), &index)?;
            self.kernels.gather(
                &mut pacer.batch,
                x.view,
                View::new(&index, 0),
                u.view,
                x.channels,
                x.columns,
                t2 as u32,
                (x.row, u.row),
            )?;
            u
        } else {
            x
        };
        if cin == cout {
            self.kernels.copy(
                &mut pacer.batch,
                shortcut.view,
                out.view,
                out.channels,
                out.columns,
                (shortcut.row, out.row),
                SQRT_HALF,
                1.0,
                true,
            )?;
        } else {
            self.conv(
                pacer,
                &format!("{prefix}.conv1x1.weight"),
                false,
                false,
                shortcut,
                1,
                (1, 1, 0),
                out,
                true,
                SQRT_HALF,
            )?;
        }
        Ok(())
    }

    /// `AdaINResBlock1` in place on `x` (`[C, L]`), using `t1`, `t2` scratch.
    #[expect(clippy::too_many_arguments)]
    fn adain_res_block1(
        &self,
        pacer: &mut Pacer<'_>,
        styles: &(MetalBuffer, BTreeMap<String, usize>),
        prefix: &str,
        x: Act<'_>,
        kernel: usize,
        t1: Act<'_>,
        t2: Act<'_>,
    ) -> crate::Result<()> {
        let c = x.channels as usize;
        let kernel_i = i32::try_from(kernel).unwrap_or(1);
        for (i, dilation) in [1_i32, 3, 5].into_iter().enumerate() {
            let (scale, shift) = Self::style_view(styles, &format!("{prefix}.adain1.{i}"), c)?;
            self.kernels.instance_norm(
                &mut pacer.batch,
                x.view,
                t1.view,
                scale,
                shift,
                Some(self.w(&format!("{prefix}.alpha1.{i}"), &[c])?),
                x.channels,
                x.columns,
                (x.row, t1.row),
                Activation::Snake,
            )?;
            let pad = dilation * (kernel_i - 1) / 2;
            self.conv(
                pacer,
                &format!("{prefix}.convs1.{i}.weight"),
                true,
                false,
                t1,
                kernel,
                (1, dilation, pad),
                t2,
                false,
                1.0,
            )?;
            let (scale, shift) = Self::style_view(styles, &format!("{prefix}.adain2.{i}"), c)?;
            self.kernels.instance_norm(
                &mut pacer.batch,
                t2.view,
                t2.view,
                scale,
                shift,
                Some(self.w(&format!("{prefix}.alpha2.{i}"), &[c])?),
                t2.channels,
                t2.columns,
                (t2.row, t2.row),
                Activation::Snake,
            )?;
            self.conv(
                pacer,
                &format!("{prefix}.convs2.{i}.weight"),
                true,
                false,
                t2,
                kernel,
                (1, 1, (kernel_i - 1) / 2),
                x,
                true,
                1.0,
            )?;
        }
        Ok(())
    }

    /// Synthesize one chunk: `ids` framed by the boundary id 0, `style` the
    /// voice pack row for the chunk's phoneme count (256 values).
    #[expect(unsafe_code, clippy::too_many_lines, clippy::many_single_char_names)]
    pub(super) fn synthesize(
        &self,
        ids: &[u32],
        style: &[f32],
        speed: f32,
        noise: Noise,
        pace: &mut dyn FnMut() -> crate::Result<()>,
        mut taps: Option<&mut Taps>,
    ) -> crate::Result<Synthesis> {
        let t = ids.len();
        if !(3..=MAX_IDS).contains(&t) || style.len() != STYLE_WIDTH {
            return Err(Error::InvalidArgument(format!(
                "speech chunk must hold 3 to {MAX_IDS} ids"
            )));
        }
        if ids.iter().any(|id| *id >= 178) {
            return Err(Error::InvalidArgument("phoneme id out of range".into()));
        }
        if !(speed.is_finite() && speed > 0.0) {
            return Err(Error::InvalidArgument("speed must be positive".into()));
        }
        let ctx = &self.context;
        let styles = self.styles(style)?;
        let mut pacer = Pacer {
            context: ctx,
            batch: CommandBatch::new(ctx)?,
            work: 0,
            pace,
        };

        // ---------------------------------------------------- phase A
        let mut x0 = vec![0.0_f32; BERT_EMBED * t];
        for (i, &id) in ids.iter().enumerate() {
            let mut e: Vec<f32> = (0..BERT_EMBED)
                .map(|c| {
                    self.word[id as usize * BERT_EMBED + c]
                        + self.position[i * BERT_EMBED + c]
                        + self.token_type[c]
                })
                .collect();
            layer_norm(&mut e, &self.embed_norm.0, &self.embed_norm.1, BERT_EPS);
            for (c, v) in e.into_iter().enumerate() {
                x0[c * t + i] = v;
            }
        }
        let emb = MetalBuffer::from_slice(ctx.device(), &x0)?;
        let emb = Act::new(&emb, BERT_EMBED, t);
        let h_buf = buffer(ctx, BERT * t)?;
        let h = Act::new(&h_buf, BERT, t);
        self.linear(
            &mut pacer,
            "bert.encoder.embedding_hidden_mapping_in.weight",
            true,
            emb,
            h,
            false,
        )?;
        let qkv = buffer(ctx, 3 * BERT * t)?;
        let q = Act::new(&qkv, 3 * BERT, t).rows(0, BERT);
        let k = Act::new(&qkv, 3 * BERT, t).rows(BERT, BERT);
        let v = Act::new(&qkv, 3 * BERT, t).rows(2 * BERT, BERT);
        let ctx_buf = buffer(ctx, BERT * t)?;
        let attn = Act::new(&ctx_buf, BERT, t);
        let a_buf = buffer(ctx, BERT * t)?;
        let a = Act::new(&a_buf, BERT, t);
        let f_buf = buffer(ctx, BERT_FFN * t)?;
        let f = Act::new(&f_buf, BERT_FFN, t);
        let layer = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        for _ in 0..BERT_LAYERS {
            for (name, out) in [("query", q), ("key", k), ("value", v)] {
                self.linear(
                    &mut pacer,
                    &format!("{layer}.attention.{name}.weight"),
                    true,
                    h,
                    out,
                    false,
                )?;
            }
            self.kernels.attention(
                &mut pacer.batch,
                q.view,
                k.view,
                v.view,
                attn.view,
                BERT_HEADS,
                t as u32,
                t as u32,
            )?;
            self.kernels.copy(
                &mut pacer.batch,
                h.view,
                a.view,
                a.channels,
                a.columns,
                (h.row, a.row),
                1.0,
                1.0,
                false,
            )?;
            self.linear(
                &mut pacer,
                &format!("{layer}.attention.dense.weight"),
                true,
                attn,
                a,
                true,
            )?;
            self.kernels.channel_norm(
                &mut pacer.batch,
                a.view,
                a.view,
                Some((
                    self.w(&format!("{layer}.attention.LayerNorm.weight"), &[BERT])?,
                    self.w(&format!("{layer}.attention.LayerNorm.bias"), &[BERT])?,
                )),
                a.channels,
                a.columns,
                a.row,
                BERT_EPS,
                None,
            )?;
            self.linear(
                &mut pacer,
                &format!("{layer}.ffn.weight"),
                true,
                a,
                f,
                false,
            )?;
            self.kernels
                .gelu(&mut pacer.batch, f.view, (BERT_FFN * t) as u32)?;
            self.kernels.copy(
                &mut pacer.batch,
                a.view,
                h.view,
                h.channels,
                h.columns,
                (a.row, h.row),
                1.0,
                1.0,
                false,
            )?;
            self.linear(
                &mut pacer,
                &format!("{layer}.ffn_output.weight"),
                true,
                f,
                h,
                true,
            )?;
            self.kernels.channel_norm(
                &mut pacer.batch,
                h.view,
                h.view,
                Some((
                    self.w(&format!("{layer}.full_layer_layer_norm.weight"), &[BERT])?,
                    self.w(&format!("{layer}.full_layer_layer_norm.bias"), &[BERT])?,
                )),
                h.channels,
                h.columns,
                h.row,
                BERT_EPS,
                None,
            )?;
        }
        // Duration encoder input: [d_en (512); style (128)] per column.
        let dx_buf = buffer(ctx, (HIDDEN + STYLE) * t)?;
        let dx = Act::new(&dx_buf, HIDDEN + STYLE, t);
        let s_rows: Vec<f32> = style[STYLE..]
            .iter()
            .flat_map(|v| std::iter::repeat_n(*v, t))
            .collect();
        write(dx.rows(HIDDEN, STYLE), &s_rows);
        self.linear(
            &mut pacer,
            "bert_encoder.weight",
            true,
            h,
            dx.rows(0, HIDDEN),
            false,
        )?;
        let gates = buffer(ctx, GATES * t.max(1))?;
        if taps.is_some() {
            pacer.sync()?;
            tap(&mut taps, "bert", h);
            tap(&mut taps, "bert_encoder", dx.rows(0, HIDDEN));
        }
        for i in 0..3 {
            let prefix = format!("predictor.text_encoder.lstms.{}", 2 * i);
            self.lstm(&mut pacer, &prefix, dx, &gates, dx.rows(0, HIDDEN))?;
            let (scale, shift) = Self::style_view(
                &styles,
                &format!("predictor.text_encoder.lstms.{}", 2 * i + 1),
                HIDDEN,
            )?;
            self.kernels.channel_norm(
                &mut pacer.batch,
                dx.view,
                dx.view,
                Some((scale, shift)),
                HIDDEN as u32,
                t as u32,
                dx.row,
                NORM_EPS,
                None,
            )?;
        }
        let xl_buf = buffer(ctx, HIDDEN * t)?;
        let xl = Act::new(&xl_buf, HIDDEN, t);
        self.lstm(&mut pacer, "predictor.lstm", dx, &gates, xl)?;
        let dur_buf = buffer(ctx, MAX_DUR * t)?;
        let dur = Act::new(&dur_buf, MAX_DUR, t);
        self.linear(
            &mut pacer,
            "predictor.duration_proj.linear_layer.weight",
            true,
            xl,
            dur,
            false,
        )?;
        // Text encoder: embedding, 3 x (conv k5, LayerNorm, LeakyReLU 0.2), LSTM.
        let mut te = vec![0.0_f32; HIDDEN * t];
        for (i, &id) in ids.iter().enumerate() {
            for c in 0..HIDDEN {
                te[c * t + i] = self.text_embedding[id as usize * HIDDEN + c];
            }
        }
        let te_a = MetalBuffer::from_slice(ctx.device(), &te)?;
        let te_a = Act::new(&te_a, HIDDEN, t);
        let te_b = buffer(ctx, HIDDEN * t)?;
        let te_b = Act::new(&te_b, HIDDEN, t);
        for i in 0..3 {
            let (src, dst) = if i % 2 == 0 {
                (te_a, te_b)
            } else {
                (te_b, te_a)
            };
            self.conv(
                &mut pacer,
                &format!("text_encoder.cnn.{i}.0.weight"),
                true,
                false,
                src,
                5,
                (1, 1, 2),
                dst,
                false,
                1.0,
            )?;
            self.kernels.channel_norm(
                &mut pacer.batch,
                dst.view,
                dst.view,
                Some((
                    self.w(&format!("text_encoder.cnn.{i}.1.gamma"), &[HIDDEN])?,
                    self.w(&format!("text_encoder.cnn.{i}.1.beta"), &[HIDDEN])?,
                )),
                HIDDEN as u32,
                t as u32,
                dst.row,
                NORM_EPS,
                Some(0.2),
            )?;
        }
        let t_en_buf = buffer(ctx, HIDDEN * t)?;
        let t_en = Act::new(&t_en_buf, HIDDEN, t);
        self.lstm(&mut pacer, "text_encoder.lstm", te_b, &gates, t_en)?;
        pacer.sync()?;
        tap(&mut taps, "predictor.text_encoder", dx);
        tap(&mut taps, "predictor.lstm", xl);
        tap(&mut taps, "text_encoder", t_en);
        let logits = read(dur);
        tap(&mut taps, "predictor.duration_proj", dur);
        let mut durations = Vec::with_capacity(t);
        for i in 0..t {
            let total: f32 = (0..MAX_DUR)
                .map(|j| 1.0 / (1.0 + (-logits[j * t + i]).exp()))
                .sum();
            durations.push((total / speed).round_ties_even().max(1.0) as u32);
        }
        let frames: usize = durations.iter().map(|d| *d as usize).sum();
        if frames > MAX_FRAMES {
            return Err(Error::ContextOverflow(format!(
                "speech chunk needs {frames} frames; the limit is {MAX_FRAMES}"
            )));
        }
        (pacer.pace)()?;

        // ---------------------------------------------------- phase B
        let index: Vec<u32> = durations
            .iter()
            .enumerate()
            .flat_map(|(i, d)| std::iter::repeat_n(i as u32, *d as usize))
            .collect();
        let index = MetalBuffer::from_slice(ctx.device(), &index)?;
        let en_buf = buffer(ctx, (HIDDEN + STYLE) * frames)?;
        let en = Act::new(&en_buf, HIDDEN + STYLE, frames);
        self.kernels.gather(
            &mut pacer.batch,
            dx.view,
            View::new(&index, 0),
            en.view,
            en.channels,
            t as u32,
            frames as u32,
            (dx.row, en.row),
        )?;
        // Decoder input [asr (512); F0 (1); N (1)].
        let dec_in_buf = buffer(ctx, (HIDDEN + 2) * frames)?;
        let dec_in = Act::new(&dec_in_buf, HIDDEN + 2, frames);
        self.kernels.gather(
            &mut pacer.batch,
            t_en.view,
            View::new(&index, 0),
            dec_in.view,
            HIDDEN as u32,
            t as u32,
            frames as u32,
            (t_en.row, dec_in.row),
        )?;
        let shared_gates = buffer(ctx, GATES * frames)?;
        let sh_buf = buffer(ctx, HIDDEN * frames)?;
        let sh = Act::new(&sh_buf, HIDDEN, frames);
        self.lstm(&mut pacer, "predictor.shared", en, &shared_gates, sh)?;
        let t0 = 2 * frames;
        let curves = buffer(ctx, 2 * t0)?;
        let f0_curve = Act::new(&curves, 2, t0).rows(0, 1);
        let n_curve = Act::new(&curves, 2, t0).rows(1, 1);
        for (branch, curve) in [("F0", f0_curve), ("N", n_curve)] {
            let b0 = buffer(ctx, HIDDEN * frames)?;
            let b0 = Act::new(&b0, HIDDEN, frames);
            self.adain_res_block(
                &mut pacer,
                &styles,
                &format!("predictor.{branch}.0"),
                sh,
                b0,
                false,
            )?;
            let b1 = buffer(ctx, 256 * t0)?;
            let b1 = Act::new(&b1, 256, t0);
            self.adain_res_block(
                &mut pacer,
                &styles,
                &format!("predictor.{branch}.1"),
                b0,
                b1,
                true,
            )?;
            let b2 = buffer(ctx, 256 * t0)?;
            let b2 = Act::new(&b2, 256, t0);
            self.adain_res_block(
                &mut pacer,
                &styles,
                &format!("predictor.{branch}.2"),
                b1,
                b2,
                false,
            )?;
            self.conv(
                &mut pacer,
                &format!("predictor.{branch}_proj.weight"),
                true,
                false,
                b2,
                1,
                (1, 1, 0),
                curve,
                false,
                1.0,
            )?;
        }
        pacer.sync()?;
        tap(&mut taps, "predictor.shared", sh);
        tap(&mut taps, "predictor.F0_proj", f0_curve);
        tap(&mut taps, "predictor.N_proj", n_curve);
        let f0 = read(f0_curve);
        (pacer.pace)()?;

        // ---------------------------------------------------- phase C
        let har = source::harmonics(&f0, self.source_linear, noise);
        let har_frames = har.len() / 22;
        let har = MetalBuffer::from_slice(ctx.device(), &har)?;
        let har = Act::new(&har, 22, har_frames);
        for (name, curve, row) in [
            ("decoder.F0_conv.weight", f0_curve, HIDDEN),
            ("decoder.N_conv.weight", n_curve, HIDDEN + 1),
        ] {
            self.conv(
                &mut pacer,
                name,
                true,
                false,
                curve,
                3,
                (2, 1, 1),
                dec_in.rows(row, 1),
                false,
                1.0,
            )?;
        }
        let cat_a = buffer(ctx, 1090 * frames)?;
        let cat_a = Act::new(&cat_a, 1090, frames);
        let cat_b = buffer(ctx, 1090 * frames)?;
        let cat_b = Act::new(&cat_b, 1090, frames);
        for cat in [cat_a, cat_b] {
            self.conv(
                &mut pacer,
                "decoder.asr_res.0.weight",
                true,
                false,
                dec_in.rows(0, HIDDEN),
                1,
                (1, 1, 0),
                cat.rows(1024, 64),
                false,
                1.0,
            )?;
            self.kernels.copy(
                &mut pacer.batch,
                dec_in.rows(HIDDEN, 2).view,
                cat.rows(1088, 2).view,
                2,
                frames as u32,
                (dec_in.row, cat.row),
                1.0,
                1.0,
                false,
            )?;
        }
        self.adain_res_block(
            &mut pacer,
            &styles,
            "decoder.encode",
            dec_in,
            cat_a.rows(0, 1024),
            false,
        )?;
        if taps.is_some() {
            pacer.sync()?;
            tap(&mut taps, "decoder.encode", cat_a.rows(0, 1024));
            tap(&mut taps, "decoder.asr_res", cat_a.rows(1024, 64));
        }
        let mut cur = cat_a;
        let mut next = cat_b;
        for i in 0..3 {
            self.adain_res_block(
                &mut pacer,
                &styles,
                &format!("decoder.decode.{i}"),
                cur,
                next.rows(0, 1024),
                false,
            )?;
            if i == 0 && taps.is_some() {
                pacer.sync()?;
                tap(&mut taps, "decoder.decode.0", next.rows(0, 1024));
            }
            std::mem::swap(&mut cur, &mut next);
        }
        let g0_buf = buffer(ctx, HIDDEN * t0)?;
        let g0 = Act::new(&g0_buf, HIDDEN, t0);
        self.adain_res_block(&mut pacer, &styles, "decoder.decode.3", cur, g0, true)?;
        if taps.is_some() {
            pacer.sync()?;
            tap(&mut taps, "decoder.decode.3", g0);
        }
        // Generator.
        let l1 = 60 * t0 + 1;
        let big = 128 * l1;
        let bx = buffer(ctx, big)?;
        let bs = buffer(ctx, big)?;
        let br = buffer(ctx, big)?;
        let b1 = buffer(ctx, big)?;
        let b2 = buffer(ctx, big)?;
        let x_prev_buf = buffer(ctx, 256 * 10 * t0)?;
        let mut x_prev = g0;
        for stage in 0..2 {
            let (ch, len, kernel_res) = if stage == 0 {
                (256_usize, 10 * t0, 7)
            } else {
                (128, l1, 11)
            };
            let s_act = Act::new(&bs, ch, len);
            let t1 = Act::new(&b1, ch, len);
            let t2 = Act::new(&b2, ch, len);
            // x = leaky_relu(x, 0.1), in place.
            self.kernels.copy(
                &mut pacer.batch,
                x_prev.view,
                x_prev.view,
                x_prev.channels,
                x_prev.columns,
                (x_prev.row, x_prev.row),
                1.0,
                0.1,
                false,
            )?;
            if stage == 0 {
                self.conv(
                    &mut pacer,
                    "decoder.generator.noise_convs.0.weight",
                    true,
                    false,
                    har,
                    12,
                    (6, 1, 3),
                    s_act,
                    false,
                    1.0,
                )?;
            } else {
                self.conv(
                    &mut pacer,
                    "decoder.generator.noise_convs.1.weight",
                    true,
                    false,
                    har,
                    1,
                    (1, 1, 0),
                    s_act,
                    false,
                    1.0,
                )?;
            }
            self.adain_res_block1(
                &mut pacer,
                &styles,
                &format!("decoder.generator.noise_res.{stage}"),
                s_act,
                kernel_res,
                t1,
                t2,
            )?;
            // Transposed convolution, one dispatch per output phase.
            let x_act = Act::new(&bx, ch, len);
            let (stride, pad, kernel) = if stage == 0 {
                (10_usize, 5_i32, 20_usize)
            } else {
                (6, 3, 12)
            };
            let stride_i = if stage == 0 { 10_i32 } else { 6 };
            let out_offset_extra = u32::from(stage == 1);
            let up_len = x_prev.columns as usize * stride;
            let w = self.w(
                &format!("decoder.generator.ups.{stage}.weight"),
                &[stride, ch, kernel / stride, 2 * ch],
            )?;
            let b = self.w(&format!("decoder.generator.ups.{stage}.bias"), &[ch])?;
            let per_phase = ch * (kernel / stride) * 2 * ch;
            for r in 0..stride {
                // Outputs t = q * stride + r - pad for q >= q0 with t >= 0.
                let r_i = i32::try_from(r).unwrap_or(0);
                let q0 = ((pad - r_i).max(0) + stride_i - 1) / stride_i;
                let first = q0 * stride_i + r_i - pad;
                if first < 0 || first as usize >= up_len {
                    continue;
                }
                let columns = (up_len - 1 - first as usize) / stride + 1;
                let p = Conv {
                    cout: ch as u32,
                    cin: 2 * ch as u32,
                    taps: (kernel / stride) as u32,
                    columns: columns as u32,
                    in_stride: 1,
                    in_dil: -1,
                    in_offset: q0,
                    in_len: x_prev.columns,
                    in_row: x_prev.row,
                    out_stride: stride as u32,
                    out_offset: first as u32 + out_offset_extra,
                    out_row: x_act.row,
                    flags: Conv::BIAS,
                    scale: 1.0,
                };
                self.dispatch_conv(
                    &mut pacer,
                    w.at(r * per_phase),
                    Some(b),
                    x_prev.view,
                    x_act.view,
                    p,
                )?;
            }
            if stage == 1 {
                // Reflection pad (1, 0): column 0 = unpadded column 1.
                self.kernels.copy(
                    &mut pacer.batch,
                    x_act.view.at(2),
                    x_act.view,
                    x_act.channels,
                    1,
                    (x_act.row, x_act.row),
                    1.0,
                    1.0,
                    false,
                )?;
            }
            // x += x_source.
            self.kernels.copy(
                &mut pacer.batch,
                s_act.view,
                x_act.view,
                x_act.channels,
                x_act.columns,
                (s_act.row, x_act.row),
                1.0,
                1.0,
                true,
            )?;
            // x = mean_j resblock_j(x); the source buffer becomes the sum.
            let r_act = Act::new(&br, ch, len);
            for j in 0..3 {
                self.kernels.copy(
                    &mut pacer.batch,
                    x_act.view,
                    r_act.view,
                    ch as u32,
                    len as u32,
                    (x_act.row, r_act.row),
                    1.0,
                    1.0,
                    false,
                )?;
                self.adain_res_block1(
                    &mut pacer,
                    &styles,
                    &format!("decoder.generator.resblocks.{}", stage * 3 + j),
                    r_act,
                    [3, 7, 11][j],
                    t1,
                    t2,
                )?;
                self.kernels.copy(
                    &mut pacer.batch,
                    r_act.view,
                    s_act.view,
                    ch as u32,
                    len as u32,
                    (r_act.row, s_act.row),
                    1.0 / 3.0,
                    1.0,
                    j > 0,
                )?;
            }
            if stage == 0 {
                let keep = Act::new(&x_prev_buf, ch, len);
                self.kernels.copy(
                    &mut pacer.batch,
                    s_act.view,
                    keep.view,
                    ch as u32,
                    len as u32,
                    (s_act.row, keep.row),
                    1.0,
                    1.0,
                    false,
                )?;
                x_prev = keep;
            } else {
                x_prev = s_act;
            }
        }
        // leaky_relu (slope 0.01), conv_post, iSTFT.
        self.kernels.copy(
            &mut pacer.batch,
            x_prev.view,
            x_prev.view,
            x_prev.channels,
            x_prev.columns,
            (x_prev.row, x_prev.row),
            1.0,
            0.01,
            false,
        )?;
        let post_buf = buffer(ctx, 22 * l1)?;
        let post = Act::new(&post_buf, 22, l1);
        self.conv(
            &mut pacer,
            "decoder.generator.conv_post.weight",
            true,
            false,
            x_prev,
            7,
            (1, 1, 3),
            post,
            false,
            1.0,
        )?;
        let samples = 5 * (l1 - 1);
        let audio = buffer(ctx, samples)?;
        self.kernels.istft(
            &mut pacer.batch,
            post.view,
            View::new(&audio, 0),
            l1 as u32,
            post.row,
        )?;
        pacer.sync()?;
        tap(&mut taps, "decoder.generator.conv_post", post);
        // SAFETY: `pacer.sync` waited for the batch; the borrow ends with this copy.
        let samples = unsafe { audio.as_slice::<f32>()[..samples].to_vec() };
        Ok(Synthesis { samples, durations })
    }
}
