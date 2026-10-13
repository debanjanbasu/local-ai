//! Kernels of the Kokoro-82M speech synthesizer (`shaders/kokoro.metal`).
//!
//! Activations are F32 `[channels, time]` blocks with a row stride (PyTorch's
//! layout for one sequence), so a producer can write straight into a channel
//! slice of a concatenation. Every operation names its buffers as
//! [`View`]s (buffer plus byte offset) and checks that the elements it will
//! touch lie inside them.

use core::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _, MTLSize,
};

use crate::Error;
use crate::batch::CommandBatch;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Encoder = ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>;

/// LSTM hidden width the recurrence kernel is built for.
pub const LSTM_HIDDEN: u32 = 256;
/// Attention head width of [`KokoroKernels::attention`].
pub const ATTENTION_HEAD_DIM: u32 = 64;

/// A position inside a buffer: F32 elements start at `offset` bytes.
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub buffer: &'a MetalBuffer,
    pub offset: usize,
}

impl<'a> View<'a> {
    #[must_use]
    pub const fn new(buffer: &'a MetalBuffer, offset: usize) -> Self {
        Self { buffer, offset }
    }

    /// The view `floats` F32 elements further on.
    #[must_use]
    pub const fn at(self, floats: usize) -> Self {
        Self {
            buffer: self.buffer,
            offset: self.offset + floats * 4,
        }
    }

    fn check(self, floats: u64) -> crate::Result<()> {
        let end = floats
            .checked_mul(4)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .and_then(|bytes| bytes.checked_add(self.offset));
        match end {
            Some(end) if end <= self.buffer.length() && self.offset % 4 == 0 => Ok(()),
            _ => Err(arg("kokoro kernel view out of bounds")),
        }
    }
}

/// Extent of a `[rows, columns]` block with row stride `row`.
fn block(rows: u32, columns: u32, row: u32) -> u64 {
    if rows == 0 || columns == 0 {
        0
    } else {
        u64::from(rows - 1) * u64::from(row) + u64::from(columns)
    }
}

/// One convolution as a gathered GEMM (see `kk_conv` in the shader):
/// `y[co, col * out_stride + out_offset] (+)= scale * (bias[co] + sum
/// w[co, tap, ci] x[ci, col * in_stride + tap * in_dil + in_offset])`, inputs
/// outside `[0, in_len)` reading zero. Weights are `[cout][taps][cin]`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Conv {
    pub cout: u32,
    pub cin: u32,
    pub taps: u32,
    pub columns: u32,
    pub in_stride: i32,
    pub in_dil: i32,
    pub in_offset: i32,
    pub in_len: u32,
    pub in_row: u32,
    pub out_stride: u32,
    pub out_offset: u32,
    pub out_row: u32,
    pub flags: u32,
    pub scale: f32,
}

impl Conv {
    /// Bias is added.
    pub const BIAS: u32 = 1;
    /// The result is added to `y` instead of replacing it.
    pub const ACCUMULATE: u32 = 2;
}

/// Activation applied after a normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    None,
    Leaky(f32),
    /// Snake with per-channel `alpha` (a view of `channels` floats).
    Snake,
}

pub struct KokoroKernels {
    conv: Pipeline,
    lstm: Pipeline,
    channel_norm: Pipeline,
    instance_norm: Pipeline,
    copy: Pipeline,
    gather: Pipeline,
    depthwise_up: Pipeline,
    gelu: Pipeline,
    attention: Pipeline,
    istft: Pipeline,
    lstm_threads: usize,
}

fn arg(message: &str) -> Error {
    Error::InvalidArgument(message.to_owned())
}

#[allow(unsafe_code)]
fn set<T: Copy>(encoder: &Encoder, value: &T, index: usize) {
    // SAFETY: `value` is a live plain value; Metal copies the bytes.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(value).cast_mut().cast::<c_void>()),
            size_of::<T>(),
            index,
        );
    }
}

#[allow(unsafe_code)]
fn bind(encoder: &Encoder, view: View<'_>, index: usize) {
    // SAFETY: the buffer is retained by the caller until GPU completion.
    unsafe { encoder.setBuffer_offset_atIndex(Some(view.buffer.raw()), view.offset, index) };
}

fn dispatch(encoder: &Encoder, groups: (usize, usize), threads: (usize, usize)) {
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups.0,
            height: groups.1,
            depth: 1,
        },
        MTLSize {
            width: threads.0,
            height: threads.1,
            depth: 1,
        },
    );
}

fn groups(count: u32, per: usize) -> usize {
    (count as usize).div_ceil(per)
}

impl KokoroKernels {
    /// # Errors
    ///
    /// Fails when a kernel is missing or cannot run its threadgroup size.
    pub fn new(context: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let pipeline = |name: &str, threads: usize| -> crate::Result<Pipeline> {
            let function = shaders.get_function(name)?;
            let pipeline = context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < threads
            {
                return Err(arg("kokoro pipeline cannot run its threadgroup size"));
            }
            Ok(pipeline)
        };
        let lstm = pipeline("kk_lstm", 256)?;
        let lstm_threads = lstm.maxTotalThreadsPerThreadgroup().min(1024) / 32 * 32;
        Ok(Self {
            conv: pipeline("kk_conv", 128)?,
            lstm,
            channel_norm: pipeline("kk_channel_norm", 64)?,
            instance_norm: pipeline("kk_instance_norm", 256)?,
            copy: pipeline("kk_copy", 64)?,
            gather: pipeline("kk_gather", 64)?,
            depthwise_up: pipeline("kk_depthwise_up", 64)?,
            gelu: pipeline("kk_gelu_tanh", 64)?,
            attention: pipeline("kk_attention", 32)?,
            istft: pipeline("kk_istft", 64)?,
            lstm_threads,
        })
    }

    /// Convolution / linear map (see [`Conv`]). `bias` is read only with
    /// [`Conv::BIAS`].
    ///
    /// # Errors
    ///
    /// Invalid shapes or views too short for the elements touched.
    pub fn conv(
        &self,
        batch: &mut CommandBatch,
        weights: View<'_>,
        bias: Option<View<'_>>,
        x: View<'_>,
        y: View<'_>,
        mut p: Conv,
    ) -> crate::Result<()> {
        if p.cout == 0 || p.cin == 0 || p.taps == 0 || p.columns == 0 || !p.scale.is_finite() {
            return Err(arg("invalid kokoro conv shape"));
        }
        weights.check(u64::from(p.cout) * u64::from(p.taps) * u64::from(p.cin))?;
        x.check(block(p.cin, p.in_len, p.in_row))?;
        let last_out = u64::from(p.columns - 1) * u64::from(p.out_stride) + u64::from(p.out_offset);
        y.check(u64::from(p.cout - 1) * u64::from(p.out_row) + last_out + 1)?;
        if p.in_len > p.in_row && p.cin > 1 {
            return Err(arg("kokoro conv input rows overlap"));
        }
        match bias {
            Some(bias) => bias.check(u64::from(p.cout))?,
            None => p.flags &= !Conv::BIAS,
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.conv);
        bind(encoder, weights, 0);
        bind(encoder, x, 1);
        bind(encoder, y, 2);
        bind(encoder, bias.unwrap_or(weights), 3);
        set(encoder, &p, 4);
        dispatch(
            encoder,
            (groups(p.cout, 32), groups(p.columns, 32)),
            (128, 1),
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Bidirectional LSTM recurrence over `steps` columns: `gates` holds the
    /// input projections `[2 * 4 * 256][g_row]` (forward rows first), `whh`
    /// the transposed recurrent weights `[2][256][1024]`; h goes to rows
    /// `0..512` of `out` (forward, then backward).
    ///
    /// # Errors
    ///
    /// Views too short or an empty sequence.
    pub fn lstm(
        &self,
        batch: &mut CommandBatch,
        gates: View<'_>,
        whh: View<'_>,
        out: View<'_>,
        steps: u32,
        g_row: u32,
        o_row: u32,
    ) -> crate::Result<()> {
        let h = LSTM_HIDDEN;
        if steps == 0 || steps > g_row || steps > o_row {
            return Err(arg("invalid kokoro LSTM shape"));
        }
        gates.check(block(8 * h, steps, g_row))?;
        whh.check(u64::from(2 * h) * u64::from(4 * h))?;
        out.check(block(2 * h, steps, o_row))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.lstm);
        bind(encoder, gates, 0);
        bind(encoder, whh, 1);
        bind(encoder, out, 2);
        set(encoder, &steps, 3);
        set(encoder, &g_row, 4);
        set(encoder, &o_row, 5);
        dispatch(encoder, (2, 1), (self.lstm_threads, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// `LayerNorm` over the channels of each column, then optional
    /// per-channel `scale`/`shift` and optional `LeakyReLU`. In place allowed.
    ///
    /// # Errors
    ///
    /// Views too short.
    #[allow(clippy::too_many_arguments)]
    pub fn channel_norm(
        &self,
        batch: &mut CommandBatch,
        x: View<'_>,
        y: View<'_>,
        affine: Option<(View<'_>, View<'_>)>,
        channels: u32,
        columns: u32,
        row: u32,
        eps: f32,
        leaky: Option<f32>,
    ) -> crate::Result<()> {
        if channels == 0 || columns == 0 || columns > row {
            return Err(arg("invalid kokoro norm shape"));
        }
        x.check(block(channels, columns, row))?;
        y.check(block(channels, columns, row))?;
        if let Some((scale, shift)) = affine {
            scale.check(u64::from(channels))?;
            shift.check(u64::from(channels))?;
        }
        let flags = u32::from(affine.is_some()) | (u32::from(leaky.is_some()) << 1);
        let slope = leaky.unwrap_or(1.0);
        let (scale, shift) = affine.unwrap_or((x, x));
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.channel_norm);
        bind(encoder, x, 0);
        bind(encoder, y, 1);
        bind(encoder, scale, 2);
        bind(encoder, shift, 3);
        set(encoder, &channels, 4);
        set(encoder, &columns, 5);
        set(encoder, &row, 6);
        set(encoder, &eps, 7);
        set(encoder, &flags, 8);
        set(encoder, &slope, 9);
        dispatch(encoder, (groups(columns, 64), 1), (64, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// Instance norm of each channel over `columns` (`AdaIN1d`):
    /// `y = norm(x) * scale[c] + shift[c]`, then `activation` (`Snake` reads
    /// `alpha`). `y` may be `x`.
    ///
    /// # Errors
    ///
    /// Views too short, or `Snake` without `alpha`.
    #[allow(clippy::too_many_arguments)]
    pub fn instance_norm(
        &self,
        batch: &mut CommandBatch,
        x: View<'_>,
        y: View<'_>,
        scale: View<'_>,
        shift: View<'_>,
        alpha: Option<View<'_>>,
        channels: u32,
        columns: u32,
        rows: (u32, u32),
        activation: Activation,
    ) -> crate::Result<()> {
        if channels == 0 || columns == 0 || columns > rows.0 || columns > rows.1 {
            return Err(arg("invalid kokoro instance norm shape"));
        }
        x.check(block(channels, columns, rows.0))?;
        y.check(block(channels, columns, rows.1))?;
        scale.check(u64::from(channels))?;
        shift.check(u64::from(channels))?;
        let (mode, slope) = match activation {
            Activation::None => (0_u32, 1.0_f32),
            Activation::Leaky(slope) => (1, slope),
            Activation::Snake => (2, 1.0),
        };
        if mode == 2 {
            alpha
                .ok_or_else(|| arg("Snake needs alpha"))?
                .check(u64::from(channels))?;
        }
        let eps = 1e-5_f32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.instance_norm);
        bind(encoder, x, 0);
        bind(encoder, y, 1);
        bind(encoder, scale, 2);
        bind(encoder, shift, 3);
        bind(encoder, alpha.unwrap_or(scale), 4);
        set(encoder, &columns, 5);
        set(encoder, &rows.0, 6);
        set(encoder, &rows.1, 7);
        set(encoder, &eps, 8);
        set(encoder, &mode, 9);
        set(encoder, &slope, 10);
        dispatch(encoder, (channels as usize, 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// `y = (accumulate ? y : 0) + scale * leaky(x)` over a `[channels,
    /// columns]` block (`slope` 1 is the identity).
    ///
    /// # Errors
    ///
    /// Views too short.
    #[allow(clippy::too_many_arguments)]
    pub fn copy(
        &self,
        batch: &mut CommandBatch,
        x: View<'_>,
        y: View<'_>,
        channels: u32,
        columns: u32,
        rows: (u32, u32),
        scale: f32,
        slope: f32,
        accumulate: bool,
    ) -> crate::Result<()> {
        if channels == 0 || columns == 0 {
            return Err(arg("invalid kokoro copy shape"));
        }
        x.check(block(channels, columns, rows.0))?;
        y.check(block(channels, columns, rows.1))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.copy);
        bind(encoder, x, 0);
        bind(encoder, y, 1);
        set(encoder, &columns, 2);
        set(encoder, &rows.0, 3);
        set(encoder, &rows.1, 4);
        set(encoder, &scale, 5);
        set(encoder, &slope, 6);
        set(encoder, &u32::from(accumulate), 7);
        dispatch(
            encoder,
            (groups(columns, 64), channels as usize),
            (64, 1),
        );
        batch.record_dispatch();
        Ok(())
    }

    /// `y[c, t] = x[c, index[t]]` for `columns` output columns; `index` is a
    /// view of `columns` `u32` values below `x_columns`.
    ///
    /// # Errors
    ///
    /// Views too short.
    #[allow(clippy::too_many_arguments)]
    pub fn gather(
        &self,
        batch: &mut CommandBatch,
        x: View<'_>,
        index: View<'_>,
        y: View<'_>,
        channels: u32,
        x_columns: u32,
        columns: u32,
        rows: (u32, u32),
    ) -> crate::Result<()> {
        if channels == 0 || columns == 0 || x_columns == 0 {
            return Err(arg("invalid kokoro gather shape"));
        }
        x.check(block(channels, x_columns, rows.0))?;
        index.check(u64::from(columns))?;
        y.check(block(channels, columns, rows.1))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gather);
        bind(encoder, x, 0);
        bind(encoder, index, 1);
        bind(encoder, y, 2);
        set(encoder, &columns, 3);
        set(encoder, &rows.0, 4);
        set(encoder, &rows.1, 5);
        dispatch(
            encoder,
            (groups(columns, 64), channels as usize),
            (64, 1),
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Depthwise `ConvTranspose1d(k=3, stride 2, padding 1, output_padding
    /// 1)`: `[channels, in_columns]` to `[channels, 2 * in_columns]`.
    ///
    /// # Errors
    ///
    /// Views too short.
    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_up(
        &self,
        batch: &mut CommandBatch,
        x: View<'_>,
        weights: View<'_>,
        bias: View<'_>,
        y: View<'_>,
        channels: u32,
        in_columns: u32,
        rows: (u32, u32),
    ) -> crate::Result<()> {
        if channels == 0 || in_columns == 0 {
            return Err(arg("invalid kokoro upsampling shape"));
        }
        x.check(block(channels, in_columns, rows.0))?;
        y.check(block(channels, 2 * in_columns, rows.1))?;
        weights.check(u64::from(channels) * 3)?;
        bias.check(u64::from(channels))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.depthwise_up);
        bind(encoder, x, 0);
        bind(encoder, weights, 1);
        bind(encoder, bias, 2);
        bind(encoder, y, 3);
        set(encoder, &in_columns, 4);
        set(encoder, &rows.0, 5);
        set(encoder, &rows.1, 6);
        dispatch(
            encoder,
            (groups(2 * in_columns, 64), channels as usize),
            (64, 1),
        );
        batch.record_dispatch();
        Ok(())
    }

    /// GELU (tanh approximation) in place over `n` contiguous values.
    ///
    /// # Errors
    ///
    /// View too short.
    pub fn gelu(&self, batch: &mut CommandBatch, x: View<'_>, n: u32) -> crate::Result<()> {
        if n == 0 {
            return Err(arg("empty GELU"));
        }
        x.check(u64::from(n))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gelu);
        bind(encoder, x, 0);
        set(encoder, &n, 1);
        dispatch(encoder, (groups(n, 256), 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// Bidirectional softmax attention over `[heads * 64, tokens]` blocks
    /// with row stride `row`, scale 1/8.
    ///
    /// # Errors
    ///
    /// Views too short.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        batch: &mut CommandBatch,
        q: View<'_>,
        k: View<'_>,
        v: View<'_>,
        out: View<'_>,
        heads: u32,
        tokens: u32,
        row: u32,
    ) -> crate::Result<()> {
        if heads == 0 || tokens == 0 || tokens > row {
            return Err(arg("invalid kokoro attention shape"));
        }
        let extent = block(heads * ATTENTION_HEAD_DIM, tokens, row);
        for view in [q, k, v, out] {
            view.check(extent)?;
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.attention);
        bind(encoder, q, 0);
        bind(encoder, k, 1);
        bind(encoder, v, 2);
        bind(encoder, out, 3);
        set(encoder, &tokens, 4);
        set(encoder, &row, 5);
        dispatch(encoder, (groups(tokens, 32), heads as usize), (32, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// Inverse STFT of the generator head `[22, frames]` (row stride `row`)
    /// into `5 * (frames - 1)` samples.
    ///
    /// # Errors
    ///
    /// Views too short or fewer than two frames.
    pub fn istft(
        &self,
        batch: &mut CommandBatch,
        x: View<'_>,
        out: View<'_>,
        frames: u32,
        row: u32,
    ) -> crate::Result<()> {
        if frames < 2 || frames > row {
            return Err(arg("invalid kokoro iSTFT shape"));
        }
        x.check(block(22, frames, row))?;
        let samples = 5 * (frames - 1);
        out.check(u64::from(samples))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.istft);
        bind(encoder, x, 0);
        bind(encoder, out, 1);
        set(encoder, &frames, 2);
        set(encoder, &row, 3);
        dispatch(encoder, (groups(samples, 256), 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }
}
