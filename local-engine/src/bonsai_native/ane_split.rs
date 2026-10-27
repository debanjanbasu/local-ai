//! Bonsai prefill with every layer's FFN and part of each mixer's output
//! projection on the Neural Engine ("ANE") while the GPU runs the rest of
//! the mixers (Gated `DeltaNet` and full attention).
//!
//! Ported from Kapitsa's `src/ane/ffn_split.{h,cpp}`, `src/kernels/ane_ffn.metal`
//! and `enc_prefill_wave` in `src/model/qwen35_ane.cpp` (nace-ai/kapitsa at
//! `a6a7bca`, Apache-2.0; see `THIRD_PARTY_NOTICES.md`), with the whole FFN on
//! the ANE instead of a channel split.
//!
//! - **Program.** One MIL program with a function per row count
//!   ([`FUNCTION_ROWS`]): the gate and up projections, `SiLU` through `tanh`,
//!   the checkpoint's 17,408-wide forward rotation (a grouped convolution
//!   whose weights are the program's only constant) and the down projection;
//!   and per row count and split ([`OUT_STEPS`]) the leading output rows of
//!   a mixer's 5,120 x 6,144 output projection.
//!   Weights are int8 surfaces with one scale per row, activations int8 with
//!   one scale per token (and the rotated intermediate re-quantized per
//!   token), so the ANE returns a partial result and a scale row that the GPU
//!   joins in F32.
//! - **Weights.** No resident int8 copy: the GPU converts each layer's
//!   ternary blocks into one of two weight sets (about 0.60 GB) while the ANE
//!   runs the previous layer from the other. A ternary row is its trits
//!   times block scales, so a row's codes are the trits times
//!   `rint(127 s_b / max |s|)`.
//! - **Wavefront.** A prefill block of up to 1,024 rows runs as two halves of
//!   at most 512 (the GPU's scratch rows): per layer the GPU runs half A's
//!   mixer, hands its FFN to the ANE, joins half B's previous FFN, runs half
//!   B's mixer and hands that over, stages the next layer's weights, then
//!   joins half A. Recurrent state and K/V flow from half A to half B in the
//!   GPU's order, so the result is the sequential one up to the ANE's
//!   rounding.
//! - **Balance.** The ANE's FFN (about 19 ms per 512-row half) is shorter
//!   than the GPU's mixer and staging (21-27 ms, growing with context), so
//!   each half's output projection is split by output row: the ANE takes the
//!   leading rows in its idle time between FFNs while the GPU computes the
//!   rest, and the GPU joins both. [`OutSplit`] chooses the split per
//!   half-block size from measured chunk times as prefill runs.
//! - **Ordering.** Two `MTLSharedEvent`s ([`Handoff`]) carry GPU-to-ANE and
//!   ANE-to-GPU order. A watchdog retires a handoff whose evaluation stops
//!   answering, which releases every GPU wait.
//! - **Fallback.** The GPU path is always there: an unsupported chip, a
//!   missing framework, a failed compile or start-up check leaves prefill on
//!   the GPU, and a chunk whose ANE work failed, timed out or returned
//!   non-finite rows is restored from a checkpoint and rerun on the GPU,
//!   after which the ANE is not used again by this model.
//!
//! Only prefill blocks take this path; decode, verification and capture stay
//! on the GPU.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use local_metal::ane::{self, Binding, Handoff, Program, Surface, SurfaceLayout};
use local_metal::bonsai_ane::AneKernels;

use super::{
    AttentionLayer, BlockOutput, BonsaiMetalTensor, BonsaiModel, BonsaiPackage, BufferCopyRequest,
    CommandBatch, Described, FFN, HadamardDirection, LAYERS, Layer, MetalBuffer, MetalContext,
    PROMPT_CHECKPOINT_BYTES, PromptCheckpoint, Rope, Selection, ShaderLibrary, Speculation, WIDTH,
    decode_embeddings,
};

/// Rows of the program's functions: a half block of `r` rows runs the
/// smallest that holds it. Below about 300 rows the ANE is bound by fetching
/// the weights, so smaller functions would not be faster.
const FUNCTION_ROWS: [u32; 4] = [128, 256, 384, 512];
/// Most rows in a half block: the GPU scratch rows (`MAX_PREFILL_TOKENS`).
const HALF_ROWS: usize = 512;
/// Prefill blocks from this many rows run as a wavefront. Bound by fetching
/// the weights, the wavefront takes about 1.40 s for any block of 160-256
/// rows and the GPU 7.3 ms per row: on the M4 Pro 0.89x the GPU's speed at
/// 160 rows, 1.01x at 192, 1.35x at 256 and 2.3x at 512.
const WAVE_MIN_ROWS: usize = 192;
/// Rows of a prefill block while the ANE runs its FFN.
pub(super) const WAVE_ROWS: usize = 2 * HALF_ROWS;
/// The first half's rows are a multiple of this.
const HALF_ALIGN: usize = 128;
/// Input channels per matmul: the ANE's int8 matmul is fastest below 4,096.
const SEGMENT: u32 = 2560;
/// The checkpoint's rotation block.
const ROTATION_BLOCK: usize = 1024;
/// Longest one ANE evaluation may run once the GPU has released it: a
/// 512-row FFN takes about 19 ms, and a full queue of both halves under 50.
const JOB_BOUND: Duration = Duration::from_secs(2);
/// Last resort for a whole chunk, should an ANE job's start never report.
const CHUNK_BOUND: Duration = Duration::from_secs(300);
/// Longest the ANE daemon may take to compile and load the program.
const COMPILE_LIMIT: Duration = Duration::from_secs(180);
/// Rows and tolerance of the start-up comparison with the GPU FFN: W8A8
/// measured 0.42% from it (relative Frobenius) on these rows; the bound only
/// has to catch a program that computes something else.
const CHECK_ROWS: u32 = 256;
const CHECK_TOLERANCE: f64 = 0.1;
/// Keepalive: the ANE sleeps after about a second idle and its first
/// evaluation then takes 70-120 ms longer. Within [`KEEP_WINDOW`] of the last
/// chunk a side thread runs one 128-row FFN evaluation (the smallest) every
/// [`KEEP_PERIOD`]. The ANE has no wake or idle
/// notification to wait on, so this is a bounded periodic task.
const KEEP_PERIOD: Duration = Duration::from_millis(800);
const KEEP_WINDOW: Duration = Duration::from_secs(60);
/// Longest a chunk waits for a keepalive evaluation in flight.
const KEEP_WAIT: Duration = Duration::from_secs(2);
/// Program cache directory under `$(DARWIN_USER_CACHE_DIR)`.
const CACHE_TAG: &str = "local-ai-bonsai-ffn";

/// Input channels of a mixer's output projection: Gated `DeltaNet`'s 48 x
/// 128 values, attention's 24 x 256 gated heads.
const MIXED: usize = 6144;
/// Leading output rows of a mixer's output projection the ANE may take (0:
/// none); the GPU computes the rest at the same time. [`OutSplit`] picks one
/// per function from measured timing. The ANE computes 1,024 rows of 512
/// tokens in about 0.95 ms, about the GPU's rate, so the gain is the overlap.
const OUT_STEPS: [u32; 9] = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096];
/// Where [`OutSplit`] starts on a machine without a learned split: 3,584
/// rows, the best fixed split for 4K prompts on the M4 Pro.
const OUT_START: usize = 7;
/// [`OutSplit`] moves to a probed split whose chunk ran faster per row by
/// more than this fraction, and takes a step away when the GPU waited over
/// [`OUT_WAIT_LIMIT_MS`] per layer on the ANE's FFNs.
const OUT_MARGIN: f64 = 0.002;
const OUT_WAIT_LIMIT_MS: f64 = 0.5;
/// The learned splits, under the program cache directory.
const OUT_SPLIT_FILE: &str = "out-split.json";

const X: SurfaceLayout = SurfaceLayout::new(SEGMENT, HALF_ROWS as u32, false);
const O: SurfaceLayout = SurfaceLayout::new(MIXED as u32 / 2, HALF_ROWS as u32, false);
const OUT: SurfaceLayout = SurfaceLayout::new(WIDTH as u32, MIXED as u32 / 2, false);
const TX: SurfaceLayout = SurfaceLayout::new(1, HALF_ROWS as u32, true);
const Y: SurfaceLayout = SurfaceLayout::new(WIDTH as u32 + 1, HALF_ROWS as u32, true);
const GATE: SurfaceLayout = SurfaceLayout::new(FFN as u32, SEGMENT, false);
const FFN_SCALES: SurfaceLayout = SurfaceLayout::new(FFN as u32, 1, true);
const DOWN_SCALES: SurfaceLayout = SurfaceLayout::new(WIDTH as u32, 1, true);

/// Layouts of the down projection's input-channel segments.
fn down_layouts() -> impl Iterator<Item = SurfaceLayout> {
    (0..FFN as u32)
        .step_by(SEGMENT as usize)
        .map(|start| SurfaceLayout::new(WIDTH as u32, SEGMENT.min(FFN as u32 - start), false))
}

/// Bytes the ANE path allocates once used: two weight sets, two lanes of
/// surfaces, the second half's GPU rows and the rerun checkpoint.
fn reserved_bytes() -> u64 {
    let set = 4 * GATE.bytes
        + 2 * FFN_SCALES.bytes
        + down_layouts().map(|layout| layout.bytes).sum::<usize>()
        + 2 * DOWN_SCALES.bytes
        + 2 * OUT.bytes;
    let lane =
        2 * X.bytes + 2 * O.bytes + TX.bytes + Y.bytes + HALF_ROWS * WIDTH * size_of::<f32>();
    let rows = 3 * HALF_ROWS * WIDTH * size_of::<f32>() + HALF_ROWS * 4 * size_of::<u32>();
    (2 * set + 2 * lane + rows + PROMPT_CHECKPOINT_BYTES) as u64
}

/// `0x1.8p+6` style hexadecimal of a normal `value`, as MIL literals take.
fn hex(value: f64) -> String {
    let bits = value.to_bits();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff).cast_signed() - 1023;
    let fraction = format!("{:013x}", bits & ((1 << 52) - 1));
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        format!("{sign}0x1p{exponent:+}")
    } else {
        format!("{sign}0x1.{fraction}p{exponent:+}")
    }
}

fn fp16(value: f64) -> String {
    format!("fp16({})", hex(value))
}

/// One function's MIL text.
struct Function {
    rows: u32,
    params: Vec<String>,
    body: String,
}

impl Function {
    fn line(&mut self, text: &str) {
        self.body.push_str("        ");
        self.body.push_str(text);
        self.body.push_str(";\n");
    }

    fn tensor(ty: &str, rows: u32, width: u32) -> String {
        format!("tensor<{ty}, [1, 1, {rows}, {width}]>")
    }

    fn half(&mut self, name: &str, rows: u32, width: u32, expression: &str) {
        let tensor = Self::tensor("fp16", rows, width);
        self.line(&format!("{tensor} {name} = {expression}"));
    }

    fn input(&mut self, name: &str, layout: SurfaceLayout, rows: u32, width: u32) {
        self.params
            .push(format!("{} {name}", layout.buffer_type(rows, width)));
        let tensor = Self::tensor(layout.mil_type(), rows, width);
        self.line(&format!(
            "{tensor} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})"
        ));
    }

    fn dequantize(&mut self, name: &str, rows: u32, width: u32) {
        let unit = fp16(1.0 / 128.0);
        self.half(
            &format!("{name}_d"),
            rows,
            width,
            &format!("dequantize(input = {name}_t, scale = {unit})"),
        );
    }

    fn matmul(&mut self, name: &str, weights: &str, input: &str, rows: u32) {
        let tensor = Self::tensor("fp16", rows, self.rows);
        self.line(&format!(
            "{tensor} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {weights}, y = {input})"
        ));
    }

    /// `prefix0 + prefix1 + ...`, returning the sum's name.
    fn sum(&mut self, prefix: &str, count: usize, rows: u32) -> String {
        let mut total = format!("{prefix}0");
        for index in 1..count {
            let next = format!("{prefix}_s{index}");
            self.half(
                &next,
                rows,
                self.rows,
                &format!("add(x = {total}, y = {prefix}{index})"),
            );
            total = next;
        }
        total
    }
}

/// The FFN of `rows` tokens: inputs `x0, x1` (int8 channel-major halves of
/// the rotated input), `tx` (fp16 token scales times 128), the gate/up
/// segments and row scales, the down segments and row scales; output `y`,
/// the down projection before its token scales and, in its last row, the
/// intermediate's token scales.
#[expect(clippy::too_many_lines)]
fn function(rows: u32) -> String {
    let (width, ffn, block) = (WIDTH as u32, FFN as u32, ROTATION_BLOCK as u32);
    let mut f = Function {
        rows,
        params: Vec::new(),
        body: String::new(),
    };
    for k in 0..2 {
        f.input(&format!("x{k}"), X, SEGMENT, rows);
    }
    f.input("tx", TX, 1, rows);
    for k in 0..2 {
        f.input(&format!("wg{k}"), GATE, ffn, SEGMENT);
        f.input(&format!("wu{k}"), GATE, ffn, SEGMENT);
    }
    f.input("sg", FFN_SCALES, ffn, 1);
    f.input("su", FFN_SCALES, ffn, 1);
    let down = down_layouts().collect::<Vec<_>>();
    for (index, layout) in down.iter().enumerate() {
        f.input(&format!("wd{index}"), *layout, width, layout.width);
    }
    f.input("sd", DOWN_SCALES, width, 1);
    for k in 0..2 {
        f.dequantize(&format!("x{k}"), SEGMENT, rows);
    }
    for projection in ["g", "u"] {
        for k in 0..2 {
            let weights = format!("w{projection}{k}");
            f.dequantize(&weights, ffn, SEGMENT);
            f.matmul(
                &format!("{projection}m{k}"),
                &format!("{weights}_d"),
                &format!("x{k}_d"),
                ffn,
            );
        }
        let total = f.sum(&format!("{projection}m"), 2, ffn);
        f.half(
            &format!("{projection}s"),
            ffn,
            rows,
            &format!("mul(x = {total}, y = s{projection}_t)"),
        );
    }
    // SiLU(g) = g/2 (1 + tanh(g/2)); the ANE has tanh but no sigmoid.
    f.half("gt", ffn, rows, "mul(x = gs, y = tx_t)");
    f.half("gh", ffn, rows, &format!("mul(x = gt, y = {})", fp16(0.5)));
    f.half("th", ffn, rows, "tanh(x = gh)");
    f.half("tp", ffn, rows, &format!("add(x = th, y = {})", fp16(1.0)));
    f.half("si", ffn, rows, "mul(x = gh, y = tp)");
    f.half("h", ffn, rows, "mul(x = si, y = us)");
    f.line(&format!(
        "tensor<fp16, [1, {ffn}, 1, {rows}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {ffn}, 1, {rows}]))"
    ));
    f.line(&format!(
        "tensor<fp16, [{ffn}, {block}, 1, 1]> rot = const()[name = string(\"rot\"), val = tensor<fp16, [{ffn}, {block}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]"
    ));
    f.line(&format!(
        "tensor<fp16, [1, {ffn}, 1, {rows}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32({}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rot, x = h4)",
        ffn / block
    ));
    f.half(
        "hr",
        ffn,
        rows,
        &format!("reshape(x = hr4, shape = tensor<int32, [4]>([1, 1, {ffn}, {rows}]))"),
    );
    // Per-token int8 of the rotated intermediate; its scale rides in y's last row.
    f.half("ha", ffn, rows, "abs(x = hr)");
    f.half(
        "pk",
        1,
        rows,
        "reduce_max(x = ha, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))",
    );
    f.half(
        "fl",
        1,
        rows,
        &format!("maximum(x = pk, y = {})", fp16(f64::powi(2.0, -9))),
    );
    f.half(
        "iv",
        1,
        rows,
        &format!("real_div(x = {}, y = fl)", fp16(127.0)),
    );
    f.half("hs", ffn, rows, "mul(x = hr, y = iv)");
    let quantized = Function::tensor("int8", ffn, rows);
    f.line(&format!(
        "{quantized} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))"
    ));
    f.half(
        "hd",
        ffn,
        rows,
        &format!("dequantize(input = hq, scale = {})", fp16(1.0 / 128.0)),
    );
    f.half(
        "hsc",
        1,
        rows,
        &format!("mul(x = fl, y = {})", fp16(128.0 / 127.0)),
    );
    let mut begin = 0;
    for (index, layout) in down.iter().enumerate() {
        let columns = layout.width;
        f.half(
            &format!("hd{index}"),
            columns,
            rows,
            &format!(
                "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>([1, 1, {columns}, {rows}]))"
            ),
        );
        let weights = format!("wd{index}");
        f.dequantize(&weights, width, columns);
        f.matmul(
            &format!("dm{index}"),
            &format!("{weights}_d"),
            &format!("hd{index}"),
            width,
        );
        begin += columns;
    }
    let total = f.sum("dm", down.len(), width);
    f.half("ds", width, rows, &format!("mul(x = {total}, y = sd_t)"));
    f.half(
        "yt",
        width + 1,
        rows,
        "concat(axis = int32(2), interleave = bool(false), values = (ds, hsc))",
    );
    let stride = Y.stride / Y.element_bytes();
    let plane = stride * Y.rows as usize;
    f.line(&format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {stride}, 1]))",
        Y.buffer_type(width + 1, rows)
    ));
    format!(
        "    func ffn{rows}<ios18>({}) {{\n{}    }} -> (y);\n",
        f.params.join(", "),
        f.body
    )
}

/// The leading `split` output rows of a mixer's output projection for `rows`
/// tokens: inputs `o0, o1` (int8 channel-major halves of the rotated mixer
/// output), `tx` (its fp16 token scales times 128), the two input-channel
/// halves of the weights and their row scales (surfaces of all
/// [`WIDTH`] rows, of which the function reads the first `split`); output
/// `y`, the projection before its token scales.
fn out_function(rows: u32, split: u32) -> String {
    let segment = MIXED as u32 / 2;
    let mut f = Function {
        rows,
        params: Vec::new(),
        body: String::new(),
    };
    for k in 0..2 {
        f.input(&format!("o{k}"), O, segment, rows);
    }
    f.input("tx", TX, 1, rows);
    for k in 0..2 {
        f.input(&format!("wo{k}"), OUT, split, segment);
    }
    f.input("so", DOWN_SCALES, split, 1);
    for k in 0..2 {
        f.dequantize(&format!("o{k}"), segment, rows);
        f.dequantize(&format!("wo{k}"), split, segment);
        f.matmul(
            &format!("om{k}"),
            &format!("wo{k}_d"),
            &format!("o{k}_d"),
            split,
        );
    }
    let total = f.sum("om", 2, split);
    f.half("ys", split, rows, &format!("mul(x = {total}, y = so_t)"));
    let stride = Y.stride / Y.element_bytes();
    let plane = stride * Y.rows as usize;
    f.line(&format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = ys, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {stride}, 1]))",
        Y.buffer_type(split, rows)
    ));
    format!(
        "    func out{rows}_{split}<ios18>({}) {{\n{}    }} -> (y);\n",
        f.params.join(", "),
        f.body
    )
}

/// The FFN program.
fn program_text() -> String {
    let mut text = String::from("program(1.3)\n{\n");
    for rows in FUNCTION_ROWS {
        text.push_str(&function(rows));
    }
    text.push_str("}\n");
    text
}

/// The output projections' program, separate from the FFN's: the ANE
/// compiler rejects a program of the four FFN functions and more than 26 of
/// these (`ANECCompile() FAILED`), while either alone compiles.
fn out_program_text() -> String {
    let mut text = String::from("program(1.3)\n{\n");
    for rows in FUNCTION_ROWS {
        for split in &OUT_STEPS[1..] {
            text.push_str(&out_function(rows, *split));
        }
    }
    text.push_str("}\n");
    text
}

/// The output projections' program has no constants, but the compiler wants
/// a weight file: one empty record.
fn placeholder_blob() -> Vec<u8> {
    let mut blob = vec![0_u8; 192];
    blob[0..4].copy_from_slice(&1_u32.to_le_bytes());
    blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
    blob[64..68].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    blob[68..72].copy_from_slice(&1_u32.to_le_bytes());
    blob[72..80].copy_from_slice(&64_u64.to_le_bytes());
    blob[80..88].copy_from_slice(&128_u64.to_le_bytes());
    blob
}

/// The program's weight blob: one fp16 record, the rotation's grouped
/// convolution weights `rot[g 1024 + o][i] = H[o][i] s[g 1024 + i] / 32`, with
/// `H` the natural-order Sylvester Hadamard matrix and `s` the checkpoint's
/// FFN signs: the forward rotation the GPU applies before the down
/// projection.
fn rotation_blob(signs: &[f32]) -> Vec<u8> {
    let elements = FFN * ROTATION_BLOCK;
    let mut blob = vec![0_u8; 128 + elements * 2];
    blob[0..4].copy_from_slice(&1_u32.to_le_bytes());
    blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
    blob[64..68].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    blob[68..72].copy_from_slice(&1_u32.to_le_bytes());
    blob[72..80].copy_from_slice(&(elements as u64 * 2).to_le_bytes());
    blob[80..88].copy_from_slice(&128_u64.to_le_bytes());
    let plus = half::f16::from_f32(1.0 / 32.0).to_bits().to_le_bytes();
    let minus = half::f16::from_f32(-1.0 / 32.0).to_bits().to_le_bytes();
    let (values, _) = blob[128..].as_chunks_mut::<2>();
    for (row, chunk) in values
        .as_chunks_mut::<ROTATION_BLOCK>()
        .0
        .iter_mut()
        .enumerate()
    {
        let (group, output) = (row / ROTATION_BLOCK, row % ROTATION_BLOCK);
        let signs = &signs[group * ROTATION_BLOCK..][..ROTATION_BLOCK];
        for (input, value) in chunk.iter_mut().enumerate() {
            let negative = (output & input).count_ones() & 1 == 1;
            *value = if negative == (signs[input] < 0.0) {
                plus
            } else {
                minus
            };
        }
    }
    blob
}

/// The smallest function holding `rows` rows.
fn function_for(rows: u32) -> crate::Result<usize> {
    FUNCTION_ROWS
        .iter()
        .position(|&capacity| capacity >= rows)
        .ok_or_else(|| crate::Error::InvalidArgument("ANE half block too long".into()))
}

/// The smallest prefill block the wavefront takes; tests lower it to
/// measure where the ANE stops paying.
#[cfg_attr(
    not(test),
    expect(clippy::missing_const_for_fn, reason = "tests read an override")
)]
fn wave_min_rows() -> usize {
    #[cfg(test)]
    {
        let rows = WAVE_MIN_ROWS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if rows != 0 {
            return rows;
        }
    }
    WAVE_MIN_ROWS
}

/// Tests fix every layer's output projection split at step `n - 1` (0:
/// learn).
#[cfg(test)]
pub(super) static OUT_STEP_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(super) static WAVE_MIN_ROWS_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Pseudo-random start-up check rows of unit variance.
fn fill_check_rows(values: &mut [f32]) {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for value in values {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *value = ((state >> 40) as f32 / (1 << 24) as f32 - 0.5) * 3.4;
    }
}

/// `|ane - gpu| / |gpu|` (Frobenius).
fn relative_difference(gpu: &[f32], ane: &[f32]) -> f64 {
    let (difference, reference) =
        gpu.iter()
            .zip(ane)
            .fold((0.0_f64, 0.0_f64), |(difference, reference), (&g, &a)| {
                let (g, a) = (f64::from(g), f64::from(a));
                ((g - a).mul_add(g - a, difference), g.mul_add(g, reference))
            });
    (difference / reference).sqrt()
}

/// The `N` of an "Apple M`N` ..." device name.
fn chip_generation(device: &str) -> Option<u32> {
    device
        .split_whitespace()
        .find_map(|word| word.strip_prefix('M')?.parse().ok())
}

/// Whether and how prefill uses the Neural Engine.
pub(super) enum AnePrefill {
    /// Prefill stays on the GPU, for this reason.
    Off(String),
    /// The program compiles on a background thread started at load; the
    /// first prefill block that can use it waits for it.
    Pending(mpsc::Receiver<crate::Result<[Program; 2]>>),
    Ready(Box<AneSplit>),
    /// Taken by the block in flight.
    Busy,
}

impl AnePrefill {
    /// Start compiling when this machine and configuration can use the ANE.
    pub(super) fn start(
        context: &MetalContext,
        package: &BonsaiPackage,
        prefill_chunk: usize,
    ) -> Self {
        if prefill_chunk != HALF_ROWS {
            return Self::Off(format!(
                "{prefill_chunk}-row prefill chunks; the wavefront runs {HALF_ROWS}-row halves"
            ));
        }
        let device = context.device_name();
        if chip_generation(&device).is_none_or(|generation| generation < 4) {
            return Self::Off(format!("measured on Apple M4 and later; this is {device}"));
        }
        if let Some(reason) = ane::unavailable() {
            return Self::Off(reason);
        }
        let signs = match package.hadamard().signs(FFN as u32) {
            Ok(signs) => signs.to_vec(),
            Err(error) => return Self::Off(error.to_string()),
        };
        let (sender, receiver) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("ane-compile".into())
            .spawn(move || {
                let programs = Program::compile(
                    &program_text(),
                    &rotation_blob(&signs),
                    CACHE_TAG,
                    COMPILE_LIMIT,
                )
                .and_then(|ffn| {
                    let out = Program::compile(
                        &out_program_text(),
                        &placeholder_blob(),
                        CACHE_TAG,
                        COMPILE_LIMIT,
                    )?;
                    Ok([ffn, out])
                });
                // The model may be gone; nothing waits then.
                let _ = sender.send(programs.map_err(crate::Error::Metal));
            });
        match spawned {
            Ok(_) => Self::Pending(receiver),
            Err(error) => Self::Off(format!("compile thread: {error}")),
        }
    }

    pub(super) const fn active(&self) -> bool {
        !matches!(self, Self::Off(_))
    }

    /// Bytes to admit for the ANE path, if it may be used.
    pub(super) fn reserved_bytes(&self) -> u64 {
        if self.active() { reserved_bytes() } else { 0 }
    }

    pub(super) fn policy(&self) -> serde_json::Value {
        match self {
            Self::Off(reason) => serde_json::json!({"enabled": false, "reason": reason}),
            _ => serde_json::json!({
                "enabled": true,
                "weights": "int8 per row",
                "activations": "int8 per token",
                "block_rows": WAVE_ROWS,
                "min_block_rows": wave_min_rows(),
                "half_rows": FUNCTION_ROWS,
                "mixer_output_rows": OUT_STEPS,
                "reserved_bytes": reserved_bytes(),
            }),
        }
    }
}

/// One half block's ANE surfaces and joined FFN rows. The output projection
/// and the FFN of a half take turns on `tx` and `y`: each is joined before
/// the other is packed.
struct Lane {
    x: [Surface; 2],
    /// The output projection's input.
    o: [Surface; 2],
    tx: Surface,
    y: Surface,
    /// The joined F32 FFN rows the next residual add reads.
    ffn: MetalBuffer,
    /// `done` value of this lane's evaluation in flight.
    done: Cell<u64>,
}

/// One layer's int8 FFN weights and the ANE's rows of its mixer's output
/// projection.
struct WeightSet {
    gate: [Surface; 2],
    up: [Surface; 2],
    gate_scales: Surface,
    up_scales: Surface,
    down: Vec<Surface>,
    down_scales: Surface,
    out: [Surface; 2],
    out_scales: Surface,
}

impl WeightSet {
    /// Surfaces by program input name.
    fn input(&self, name: &str) -> Option<&Surface> {
        Some(match name {
            "wg0" => &self.gate[0],
            "wg1" => &self.gate[1],
            "wu0" => &self.up[0],
            "wu1" => &self.up[1],
            "sg" => &self.gate_scales,
            "su" => &self.up_scales,
            "sd" => &self.down_scales,
            "wo0" => &self.out[0],
            "wo1" => &self.out[1],
            "so" => &self.out_scales,
            _ => self
                .down
                .get(name.strip_prefix("wd")?.parse::<usize>().ok()?)?,
        })
    }
}

/// The ANE path once compiled and checked.
///
/// Fields drop in order: the keepalive thread and the handoff (which waits
/// for evaluations in flight) go before the surfaces those evaluations use.
pub(super) struct AneSplit {
    keep: Option<Keepalive>,
    handoff: Handoff,
    kernels: AneKernels,
    lanes: [Lane; 2],
    sets: [WeightSet; 2],
    /// `[set][lane][function]`.
    bindings: Vec<Arc<Binding>>,
    /// `[set][lane][function][split - 1]`: the output projection's.
    out_bindings: Vec<Binding>,
    /// The chunk's jobs in queue order, matching [`Handoff::take_timeline`].
    jobs: RefCell<Vec<Job>>,
    /// The output projection split, learned from the jobs' timing.
    out_split: RefCell<OutSplit>,
    /// Set by a join that met a non-finite value.
    status: MetalBuffer,
    /// The second half's embeddings, residual stream and rotary rows (the
    /// first half's are the GPU scratch).
    embedding: MetalBuffer,
    hidden: MetalBuffer,
    hidden_alt: MetalBuffer,
    rope: MetalBuffer,
    /// Recurrent state at the start of the chunk in flight, for a rerun.
    checkpoint: Option<PromptCheckpoint>,
    /// Test fault: the layer whose first-half evaluation is never started.
    #[cfg(test)]
    fault: Cell<Option<usize>>,
}

impl AneSplit {
    fn new(context: &MetalContext, [program, out_program]: &[Program; 2]) -> crate::Result<Self> {
        let shaders = ShaderLibrary::new(context.device())?;
        let surface = |layout| Surface::new(context, layout).map_err(crate::Error::Metal);
        let rows = |width: usize| {
            MetalBuffer::empty(context.device(), HALF_ROWS * width * size_of::<f32>())
        };
        let lane = || -> crate::Result<Lane> {
            Ok(Lane {
                x: [surface(X)?, surface(X)?],
                o: [surface(O)?, surface(O)?],
                tx: surface(TX)?,
                y: surface(Y)?,
                ffn: rows(WIDTH)?,
                done: Cell::new(0),
            })
        };
        let set = || -> crate::Result<WeightSet> {
            Ok(WeightSet {
                gate: [surface(GATE)?, surface(GATE)?],
                up: [surface(GATE)?, surface(GATE)?],
                gate_scales: surface(FFN_SCALES)?,
                up_scales: surface(FFN_SCALES)?,
                down: down_layouts()
                    .map(surface)
                    .collect::<crate::Result<Vec<_>>>()?,
                down_scales: surface(DOWN_SCALES)?,
                out: [surface(OUT)?, surface(OUT)?],
                out_scales: surface(DOWN_SCALES)?,
            })
        };
        let lanes = [lane()?, lane()?];
        let sets = [set()?, set()?];
        let bind = |program: &Program, set: &WeightSet, lane: &Lane, name: &str| {
            let procedure = program
                .procedure(name)
                .ok_or_else(|| crate::Error::Generation(format!("ANE program has no {name}")))?;
            let inputs = program
                .inputs(procedure)
                .iter()
                .map(|name| {
                    match name.as_str() {
                        "x0" => Some(&lane.x[0]),
                        "x1" => Some(&lane.x[1]),
                        "o0" => Some(&lane.o[0]),
                        "o1" => Some(&lane.o[1]),
                        "tx" => Some(&lane.tx),
                        name => set.input(name),
                    }
                    .ok_or_else(|| crate::Error::Generation(format!("unknown ANE input {name}")))
                })
                .collect::<crate::Result<Vec<_>>>()?;
            crate::Result::Ok(program.bind(procedure, &inputs, &lane.y)?)
        };
        let mut bindings = Vec::with_capacity(2 * 2 * FUNCTION_ROWS.len());
        let mut out_bindings = Vec::with_capacity(bindings.capacity() * OUT_STEPS.len());
        for set in &sets {
            for lane in &lanes {
                for rows in FUNCTION_ROWS {
                    bindings.push(Arc::new(bind(program, set, lane, &format!("ffn{rows}"))?));
                    for split in &OUT_STEPS[1..] {
                        out_bindings.push(bind(
                            out_program,
                            set,
                            lane,
                            &format!("out{rows}_{split}"),
                        )?);
                    }
                }
            }
        }
        Ok(Self {
            keep: None,
            handoff: Handoff::new(context, JOB_BOUND)?,
            kernels: AneKernels::new(context, &shaders)?,
            lanes,
            sets,
            bindings,
            out_bindings,
            jobs: RefCell::new(Vec::new()),
            out_split: RefCell::new(OutSplit::load()),
            status: MetalBuffer::empty(context.device(), 16)?,
            embedding: rows(WIDTH)?,
            hidden: rows(WIDTH)?,
            hidden_alt: rows(WIDTH)?,
            rope: MetalBuffer::empty(context.device(), HALF_ROWS * 4 * size_of::<u32>())?,
            checkpoint: None,
            #[cfg(test)]
            fault: Cell::new(None),
        })
    }

    fn binding(&self, set: usize, lane: usize, function: usize) -> &Binding {
        &self.bindings[(set * 2 + lane) * FUNCTION_ROWS.len() + function]
    }

    fn out_binding(&self, set: usize, lane: usize, function: usize, step: usize) -> &Binding {
        let splits = OUT_STEPS.len() - 1;
        &self.out_bindings[((set * 2 + lane) * FUNCTION_ROWS.len() + function) * splits + step - 1]
    }

    /// Convert the first `split` rows (none for 0) of a mixer's output
    /// projection `weights` into weight set `set`.
    fn stage_out(
        &self,
        batch: &mut CommandBatch,
        weights: &BonsaiMetalTensor,
        set: usize,
        split: u32,
    ) -> crate::Result<()> {
        if split == 0 {
            return Ok(());
        }
        let set = &self.sets[set];
        let matrix = weights.ptq1_matrix()?;
        if matrix.rows() != WIDTH as u32 || matrix.columns() != MIXED as u32 {
            return Err(crate::Error::InvalidArgument(
                "unexpected mixer output projection shape".into(),
            ));
        }
        let rows = matrix.row_slice(0, split)?;
        let segment = MIXED as u32 / 2 / 128;
        for (index, surface) in set.out.iter().enumerate() {
            self.kernels.stage(
                batch,
                rows,
                index as u32 * segment,
                segment,
                surface,
                (index == 0).then_some(&set.out_scales),
            )?;
        }
        Ok(())
    }

    /// Convert `layer`'s FFN weights into weight set `set`.
    fn stage(&self, batch: &mut CommandBatch, layer: &Layer, set: usize) -> crate::Result<()> {
        let set = &self.sets[set];
        let segment = SEGMENT / 128;
        for (matrix, surfaces, scales) in [
            (&layer.gate, &set.gate, &set.gate_scales),
            (&layer.up, &set.up, &set.up_scales),
        ] {
            let matrix = matrix.ptq1_matrix()?;
            for (index, surface) in surfaces.iter().enumerate() {
                self.kernels.stage(
                    batch,
                    matrix,
                    index as u32 * segment,
                    segment,
                    surface,
                    (index == 0).then_some(scales),
                )?;
            }
        }
        let down = layer.down.ptq1_matrix()?;
        let mut first = 0;
        for (index, surface) in set.down.iter().enumerate() {
            let count = surface.layout().width / 128;
            self.kernels.stage(
                batch,
                down,
                first,
                count,
                surface,
                (index == 0).then_some(&set.down_scales),
            )?;
            first += count;
        }
        Ok(())
    }

    /// Pack `rows` rows of `input` into `lane`'s surfaces and hand them to
    /// the ANE with weight set `set`, submitting the GPU work so far.
    #[expect(clippy::too_many_arguments)]
    fn launch(
        &self,
        batch: &mut CommandBatch,
        context: &MetalContext,
        input: &MetalBuffer,
        lane: usize,
        set: usize,
        rows: u32,
        layer: usize,
    ) -> crate::Result<()> {
        let function = function_for(rows)?;
        let surfaces = &self.lanes[lane];
        self.kernels.pack(
            batch,
            input,
            WIDTH as u32,
            rows,
            FUNCTION_ROWS[function],
            &surfaces.tx,
            [&surfaces.x[0], &surfaces.x[1]],
        )?;
        let job = Job {
            out: false,
            lane,
            layer,
        };
        self.hand_over(batch, context, self.binding(set, lane, function), job)
    }

    /// Pack `rows` rows of a mixer's rotated output `input` into `lane`'s
    /// surfaces and hand the first `OUT_STEPS[step]` rows of its output
    /// projection (weight set `set`) to the ANE, submitting the GPU work so
    /// far.
    #[expect(clippy::too_many_arguments)]
    fn launch_out(
        &self,
        batch: &mut CommandBatch,
        context: &MetalContext,
        input: &MetalBuffer,
        lane: usize,
        set: usize,
        rows: u32,
        step: usize,
        layer: usize,
    ) -> crate::Result<()> {
        let function = function_for(rows)?;
        let surfaces = &self.lanes[lane];
        self.kernels.pack(
            batch,
            input,
            MIXED as u32,
            rows,
            FUNCTION_ROWS[function],
            &surfaces.tx,
            [&surfaces.o[0], &surfaces.o[1]],
        )?;
        let job = Job {
            out: true,
            lane,
            layer,
        };
        self.hand_over(
            batch,
            context,
            self.out_binding(set, lane, function, step),
            job,
        )
    }

    /// Queue `binding` behind the GPU work so far and submit that.
    fn hand_over(
        &self,
        batch: &mut CommandBatch,
        context: &MetalContext,
        binding: &Binding,
        job: Job,
    ) -> crate::Result<()> {
        let ready = self.handoff.next_ready();
        let done = self.handoff.next_done();
        #[cfg(test)]
        let skip = job.lane == 0 && !job.out && self.fault.get() == Some(job.layer) && {
            self.fault.set(None);
            true
        };
        #[cfg(not(test))]
        let skip = false;
        self.handoff.queue(binding, ready, done, skip);
        self.jobs.borrow_mut().push(job);
        self.lanes[job.lane].done.set(done);
        batch.signal_event(self.handoff.ready_event(), ready)?;
        batch.submit_and_renew(context)?;
        Ok(())
    }

    /// Wait for `lane`'s evaluation and join its `rows` into the lane's F32
    /// FFN rows.
    fn join(&self, batch: &mut CommandBatch, lane: usize, rows: u32) -> crate::Result<()> {
        let surfaces = &self.lanes[lane];
        batch.wait_event(self.handoff.done_event(), surfaces.done.get())?;
        self.kernels.join(
            batch,
            &surfaces.y,
            &surfaces.tx,
            &surfaces.ffn,
            &self.status,
            WIDTH as u32,
            rows,
        )?;
        Ok(())
    }

    /// Wait for `lane`'s output projection rows and join its first `split`
    /// rows of each of `rows` tokens with the GPU's `rest` into `output`.
    fn join_out(
        &self,
        batch: &mut CommandBatch,
        lane: usize,
        rows: u32,
        split: u32,
        rest: &MetalBuffer,
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        let surfaces = &self.lanes[lane];
        batch.wait_event(self.handoff.done_event(), surfaces.done.get())?;
        self.kernels.join_split(
            batch,
            &surfaces.y,
            &surfaces.tx,
            rest,
            output,
            &self.status,
            WIDTH as u32,
            split,
            rows,
        )?;
        Ok(())
    }

    /// Prepare for a chunk: `false` if a keepalive evaluation did not return.
    fn begin_chunk(&self) -> bool {
        self.jobs.borrow_mut().clear();
        drop(self.handoff.take_timeline());
        if self.keep.as_ref().is_some_and(|keep| !keep.enter()) {
            self.handoff.retire("a keepalive evaluation did not return");
            return false;
        }
        self.status.copy_from_bytes(&[0; 4], 0);
        self.handoff.arm(CHUNK_BOUND);
        true
    }

    /// After the chunk's GPU work: whether every evaluation succeeded and
    /// every joined row was finite.
    #[expect(unsafe_code)]
    fn finish_chunk(&self) -> Result<(), String> {
        let finished = self.handoff.finish();
        if let Some(keep) = &self.keep {
            keep.leave();
        }
        if !finished {
            return Err(self
                .handoff
                .retired()
                .unwrap_or_else(|| "a Neural Engine evaluation failed".into()));
        }
        // SAFETY: the chunk's GPU work has completed; nothing writes it now.
        if unsafe { self.status.as_slice::<u32>()[0] } != 0 {
            return Err("the Neural Engine returned non-finite FFN rows".into());
        }
        Ok(())
    }
}

/// One evaluation handed to the ANE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Job {
    /// The mixer's output projection rather than the FFN.
    out: bool,
    lane: usize,
    layer: usize,
}

/// How long the GPU waited, on average over a chunk's layers after the
/// first, for the second half's previous FFN after handing over the first
/// half's (ms), from the chunk's `jobs` and their release and report times
/// in queue order: above zero when the ANE has no time to spare.
fn mean_ffn_wait(
    jobs: &[Job],
    timeline: &[(Option<Instant>, Option<Instant>)],
    layers: usize,
) -> Option<f64> {
    let mut ffn = vec![[(None, None); 2]; layers];
    for (job, &(ready, done)) in jobs.iter().zip(timeline) {
        if !job.out && job.layer < layers {
            ffn[job.layer][job.lane] = (ready, done);
        }
    }
    mean((1..layers).filter_map(|layer| {
        let (done, ready): (Instant, Instant) = (ffn[layer - 1][1].1?, ffn[layer][0].0?);
        Some(done.saturating_duration_since(ready).as_secs_f64() * 1e3)
    }))
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, count) = values.fold((0.0, 0_u32), |(sum, count), value| (sum + value, count + 1));
    (count > 0).then(|| sum / f64::from(count))
}

/// One finished wavefront chunk, as [`OutSplit`] compares them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Chunk {
    function: usize,
    rows: usize,
    step: usize,
    probe: bool,
    ms_per_row: f64,
}

/// How many leading rows of each mixer's output projection the ANE takes,
/// per function (half-block rows), as an index into [`OUT_STEPS`], tuned
/// online from measured chunk times. Contiguous chunks of a prompt
/// alternate between the current split and a probe one step up or down
/// (alternating by comparison); a probe chunk that ran faster per row than
/// the mean of the equal-sized chunks on either side, by more than
/// [`OUT_MARGIN`], becomes the split. Averaging the neighbours cancels the
/// steady growth of chunk time with context, so the split follows the prompt
/// as attention's GPU work grows. A chunk whose GPU waited on the ANE's
/// FFNs takes a step away instead. Splits persist per machine.
///
/// Per layer comparisons do not work: a layer's extra ANE rows delay the
/// FFN the next layer joins, so its cost lands on its neighbour.
#[derive(Debug, PartialEq)]
struct OutSplit {
    steps: [usize; FUNCTION_ROWS.len()],
    probe_up: [bool; FUNCTION_ROWS.len()],
    /// The last finished chunks of the current prompt, oldest first.
    history: Vec<Chunk>,
    /// Where the last finished chunk ended.
    end: Option<usize>,
    /// The chunk in flight.
    current: Option<Chunk>,
}

impl OutSplit {
    const fn new(steps: [usize; FUNCTION_ROWS.len()]) -> Self {
        Self {
            steps,
            probe_up: [true; FUNCTION_ROWS.len()],
            history: Vec::new(),
            end: None,
            current: None,
        }
    }

    fn path() -> Option<std::path::PathBuf> {
        Some(
            ane::user_cache_dir()
                .ok()?
                .join(CACHE_TAG)
                .join(OUT_SPLIT_FILE),
        )
    }

    /// The learned splits, or [`OUT_START`] everywhere.
    fn load() -> Self {
        let mut steps = [OUT_START; FUNCTION_ROWS.len()];
        let learned = Self::path()
            .and_then(|path| std::fs::read(path).ok())
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        if let Some(rows) = learned.as_ref().and_then(|value| value["rows"].as_array()) {
            for (step, rows) in steps.iter_mut().zip(rows) {
                if let Some(index) = rows
                    .as_u64()
                    .and_then(|rows| OUT_STEPS.iter().position(|&s| u64::from(s) == rows))
                {
                    *step = index;
                }
            }
        }
        Self::new(steps)
    }

    fn rows(&self) -> serde_json::Value {
        serde_json::json!(self.steps.map(|step| OUT_STEPS[step]))
    }

    /// Best effort: a split that cannot be saved is learned again.
    fn save(&self) {
        if let Some(path) = Self::path() {
            let text = serde_json::json!({"rows": self.rows()}).to_string();
            let _ = std::fs::write(path, text);
        }
    }

    /// The step for a chunk of `rows` rows from position `start` whose halves
    /// run `function`.
    fn plan(&mut self, function: usize, rows: usize, start: usize) -> usize {
        if self.end != Some(start) {
            self.history.clear();
        }
        let base = self.steps[function];
        let mut chunk = Chunk {
            function,
            rows,
            step: base,
            probe: false,
            ms_per_row: 0.0,
        };
        if self
            .history
            .last()
            .is_some_and(|last| !last.probe && last.function == function && last.rows == rows)
        {
            chunk.step = if self.probe_up[function] {
                (base + 1).min(OUT_STEPS.len() - 1)
            } else {
                base.saturating_sub(1)
            };
            chunk.probe = chunk.step != base;
        }
        #[cfg(test)]
        if let Some(step) = OUT_STEP_OVERRIDE
            .load(std::sync::atomic::Ordering::Relaxed)
            .checked_sub(1)
        {
            chunk.step = step;
        }
        self.current = Some(chunk);
        chunk.step
    }

    /// Learn from the chunk in flight, which ended at position `end` after
    /// `ms`, its GPU having waited `wait` ms per layer on average for the
    /// ANE's FFNs: whether the split changed.
    fn learn(&mut self, end: usize, ms: f64, wait: Option<f64>) -> bool {
        let Some(mut chunk) = self.current.take() else {
            return false;
        };
        self.end = Some(end);
        #[cfg(test)]
        if OUT_STEP_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) != 0 {
            return false;
        }
        let step = &mut self.steps[chunk.function];
        let before = *step;
        if wait.is_some_and(|wait| wait > OUT_WAIT_LIMIT_MS) {
            *step = chunk.step.min(*step).saturating_sub(1);
            self.history.clear();
            return *step != before;
        }
        chunk.ms_per_row = ms / chunk.rows as f64;
        self.history.push(chunk);
        if self.history.len() > 3 {
            self.history.remove(0);
        }
        if let [first, probe, last] = self.history[..]
            && probe.probe
            && [first, last].iter().all(|base| {
                !base.probe
                    && base.step == before
                    && base.function == probe.function
                    && base.rows == probe.rows
            })
        {
            if probe.ms_per_row
                < f64::midpoint(first.ms_per_row, last.ms_per_row) * (1.0 - OUT_MARGIN)
            {
                *step = probe.step;
            }
            self.probe_up[chunk.function] = !self.probe_up[chunk.function];
            // The last base chunk opens the next comparison.
            self.history.drain(..2);
        }
        *step != before
    }
}

/// The keepalive thread's shared state; the thread holds its own clone, so
/// one wedged in an evaluation can be detached.
struct KeepState {
    inner: Mutex<KeepInner>,
    changed: Condvar,
}

struct KeepInner {
    stop: bool,
    in_chunk: bool,
    evaluating: bool,
    last_use: Instant,
}

struct Keepalive {
    state: Arc<KeepState>,
    thread: Option<JoinHandle<()>>,
}

fn lock(state: &KeepState) -> MutexGuard<'_, KeepInner> {
    state.inner.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Keepalive {
    fn start(binding: Arc<Binding>) -> crate::Result<Self> {
        let state = Arc::new(KeepState {
            inner: Mutex::new(KeepInner {
                stop: false,
                in_chunk: false,
                evaluating: false,
                last_use: Instant::now(),
            }),
            changed: Condvar::new(),
        });
        let shared = Arc::clone(&state);
        let thread = std::thread::Builder::new()
            .name("ane-keepalive".into())
            .spawn(move || Self::run(&shared, &binding))
            .map_err(|error| crate::Error::Generation(format!("keepalive thread: {error}")))?;
        Ok(Self {
            state,
            thread: Some(thread),
        })
    }

    #[expect(clippy::significant_drop_tightening)]
    fn run(state: &KeepState, binding: &Binding) {
        let mut inner = lock(state);
        loop {
            inner = state
                .changed
                .wait_timeout(inner, KEEP_PERIOD)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
            if inner.stop {
                return;
            }
            let idle = inner.last_use.elapsed();
            if inner.in_chunk || idle < KEEP_PERIOD || idle > KEEP_WINDOW {
                continue;
            }
            inner.evaluating = true;
            drop(inner);
            let result = binding.evaluate();
            inner = lock(state);
            inner.evaluating = false;
            state.changed.notify_all();
            if result.is_err() {
                // The next chunk meets whatever went wrong and falls back.
                return;
            }
        }
    }

    /// Keep the thread off the surfaces for a chunk: `false` if an
    /// evaluation in flight did not return within [`KEEP_WAIT`].
    #[expect(clippy::significant_drop_tightening)]
    fn enter(&self) -> bool {
        let mut inner = lock(&self.state);
        inner.in_chunk = true;
        let deadline = Instant::now() + KEEP_WAIT;
        while inner.evaluating {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            inner = self
                .state
                .changed
                .wait_timeout(inner, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }

    fn leave(&self) {
        let mut inner = lock(&self.state);
        inner.in_chunk = false;
        inner.last_use = Instant::now();
    }
}

impl Drop for Keepalive {
    fn drop(&mut self) {
        let mut inner = lock(&self.state);
        inner.stop = true;
        self.state.changed.notify_all();
        let deadline = Instant::now() + KEEP_WAIT;
        while inner.evaluating && Instant::now() < deadline {
            inner = self
                .state
                .changed
                .wait_timeout(inner, deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let wedged = inner.evaluating;
        drop(inner);
        if let Some(thread) = self.thread.take() {
            if wedged {
                // Program::evaluate has no timeout: leave the thread (and
                // the binding it holds) rather than hang.
                eprintln!("local-ai: a Neural Engine keepalive evaluation did not return");
            } else {
                let _ = thread.join();
            }
        }
    }
}

impl Layer {
    /// The mixer's output projection (`ssm_out` or `attn_output`).
    const fn mixer_output(&self) -> &BonsaiMetalTensor {
        match &self.attention {
            AttentionLayer::Recurrent(recurrent) => &recurrent.output,
            AttentionLayer::Full(full) => &full.output,
        }
    }
}

/// One half block of the wavefront.
struct Half<'a> {
    rows: u32,
    position: usize,
    embedding: &'a MetalBuffer,
    hidden: &'a MetalBuffer,
    hidden_alt: &'a MetalBuffer,
    rope: Rope<'a>,
}

impl BonsaiModel {
    /// Rows per prefill block.
    pub(super) fn prefill_rows(&self) -> usize {
        if self.ane.active() {
            self.info.prefill_chunk_size
        } else {
            self.info.prefill_chunk_size.min(self.block_rows)
        }
    }

    /// The first half's rows when a prefill block of `rows` runs as a
    /// wavefront.
    fn wave_split(&self, rows: usize) -> Option<usize> {
        (self.ane.active()
            && self.block_rows >= HALF_ROWS
            && (wave_min_rows().max(HALF_ALIGN + 1)..=WAVE_ROWS).contains(&rows))
        .then(|| rows.div_ceil(2).next_multiple_of(HALF_ALIGN).min(HALF_ROWS))
    }

    /// The Neural Engine policy reported at load.
    pub(crate) fn neural_engine_policy(&self) -> serde_json::Value {
        self.ane.policy()
    }

    /// Keep prefill on the GPU from now on.
    #[cfg(test)]
    pub(super) fn disable_ane(&mut self, reason: &str) {
        self.ane = super::AnePrefill::Off(reason.into());
    }

    /// One prefill block, its MTP head ingestion in the same batches: on the
    /// GPU and the ANE together when it can, else on the GPU.
    pub(super) fn encode_prefill_block(
        &mut self,
        tokens: &[u32],
        output: BlockOutput<'_>,
    ) -> crate::Result<()> {
        let valid = !matches!(output, BlockOutput::Verify(_))
            && crate::bonsai_vision::check_tokens(tokens, self.position, self.images.as_deref())
                .is_ok()
            && self
                .position
                .checked_add(tokens.len())
                .is_some_and(|end| end <= self.info.context);
        if valid
            && let Some(first) = self.wave_split(tokens.len())
            && let Some(split) = self.take_ane()
        {
            return self.prefill_wave(split, tokens, output, first);
        }
        self.prefill_on_gpu(tokens, output, tokens.len().min(self.block_rows))
    }

    /// `tokens` as GPU blocks of `first` rows and the rest.
    fn prefill_on_gpu(
        &mut self,
        tokens: &[u32],
        output: BlockOutput<'_>,
        first: usize,
    ) -> crate::Result<()> {
        let (head, tail) = tokens.split_at(first);
        if tail.is_empty() {
            return self.encode_with_head(head, output);
        }
        let early = if self.speculation.is_some() {
            // The head ingests every committed row's output-normalized
            // hidden, so the first block must produce them too.
            BlockOutput::Hidden
        } else {
            BlockOutput::None
        };
        self.encode_with_head(head, early)?;
        self.prefill_on_gpu(tail, output, tail.len().min(self.block_rows))
    }

    /// [`Self::encode_block`] with head ingestion riding in the same command
    /// batch: one GPU round trip per chunk instead of two.
    fn encode_with_head(&mut self, tokens: &[u32], output: BlockOutput<'_>) -> crate::Result<()> {
        let mut speculation = self.speculation.take();
        let result = self.encode_block(tokens, output, speculation.as_mut());
        self.speculation = speculation;
        result
    }

    /// The compiled ANE path, built on first use; `None` (leaving the reason
    /// in `self.ane`) when it cannot be used.
    fn take_ane(&mut self) -> Option<Box<AneSplit>> {
        match std::mem::replace(&mut self.ane, super::AnePrefill::Busy) {
            super::AnePrefill::Ready(split) => Some(split),
            super::AnePrefill::Pending(receiver) => {
                // Bounded by the compile limit.
                let built = receiver
                    .recv()
                    .map_err(|_| crate::Error::Generation("the ANE compile thread ended".into()))
                    .and_then(|programs| programs)
                    .and_then(|programs| self.build_ane(&programs));
                match built {
                    Ok(split) => Some(split),
                    Err(error) => {
                        eprintln!("local-ai: prefill stays on the GPU: Neural Engine {error}");
                        self.ane = super::AnePrefill::Off(error.to_string());
                        None
                    }
                }
            }
            other => {
                self.ane = other;
                None
            }
        }
    }

    fn build_ane(&mut self, programs: &[Program; 2]) -> crate::Result<Box<AneSplit>> {
        let mut split = AneSplit::new(&self.context, programs)?;
        for (what, error) in [
            ("FFN", self.check_ane(&split)?),
            ("output projection", self.check_out(&split)?),
        ] {
            if error.is_nan() || error > CHECK_TOLERANCE {
                return Err(crate::Error::Generation(format!(
                    "{what} differs from the GPU's by {error:.3} (relative), over {CHECK_TOLERANCE}"
                )));
            }
        }
        split.keep = Some(Keepalive::start(Arc::clone(&split.bindings[0]))?);
        Ok(Box::new(split))
    }

    /// Relative difference between the ANE's and the GPU's FFN of layer 0 on
    /// [`CHECK_ROWS`] pseudo-random rows of unit variance.
    #[expect(unsafe_code)]
    fn check_ane(&mut self, split: &AneSplit) -> crate::Result<f64> {
        let rows = CHECK_ROWS as usize;
        // SAFETY: no GPU work is in flight between blocks.
        fill_check_rows(unsafe {
            &mut self.scratch.rotated_hidden.as_mut_slice::<f32>()[..rows * WIDTH]
        });
        let layer = &self.layers[0];
        let lane = &split.lanes[0];
        let handoff = &split.handoff;
        split.status.copy_from_bytes(&[0; 4], 0);
        let mut batch = CommandBatch::new(&self.context)?;
        let encoded = (|| -> crate::Result<()> {
            split.stage(&mut batch, layer, 0)?;
            split.kernels.pack(
                &mut batch,
                &self.scratch.rotated_hidden,
                WIDTH as u32,
                CHECK_ROWS,
                CHECK_ROWS,
                &lane.tx,
                [&lane.x[0], &lane.x[1]],
            )?;
            let ready = handoff.next_ready();
            let done = handoff.next_done();
            handoff.queue(
                split.binding(0, 0, function_for(CHECK_ROWS)?),
                ready,
                done,
                false,
            );
            batch.signal_event(handoff.ready_event(), ready)?;
            self.feed_forward_branch(&mut batch, layer, CHECK_ROWS)?;
            batch.wait_event(handoff.done_event(), done)?;
            split.kernels.join(
                &mut batch,
                &lane.y,
                &lane.tx,
                &lane.ffn,
                &split.status,
                WIDTH as u32,
                CHECK_ROWS,
            )?;
            Ok(())
        })();
        if let Err(error) = encoded {
            handoff.retire("the start-up check failed to encode");
            return Err(error);
        }
        self.finish(batch)?;
        if !handoff.finish() {
            return Err(crate::Error::Generation(
                handoff
                    .retired()
                    .unwrap_or_else(|| "start-up evaluation failed".into()),
            ));
        }
        // SAFETY: the batch has completed.
        let (gpu, ane, status) = unsafe {
            (
                &self.scratch.branch.as_slice::<f32>()[..rows * WIDTH],
                &lane.ffn.as_slice::<f32>()[..rows * WIDTH],
                split.status.as_slice::<u32>()[0],
            )
        };
        if status != 0 {
            return Err(crate::Error::Generation(
                "start-up evaluation returned non-finite rows".into(),
            ));
        }
        Ok(relative_difference(gpu, ane))
    }

    /// Relative difference between layer 0's mixer output projection with
    /// its largest split on the ANE and on the GPU alone, on [`CHECK_ROWS`]
    /// pseudo-random rows of unit variance.
    #[expect(unsafe_code)]
    fn check_out(&mut self, split: &AneSplit) -> crate::Result<f64> {
        let rows = CHECK_ROWS as usize;
        // SAFETY: no GPU work is in flight between blocks.
        fill_check_rows(unsafe {
            &mut self.scratch.rotated_attention.as_mut_slice::<f32>()[..rows * MIXED]
        });
        let scratch = &self.scratch;
        let weights = self.layers[0].mixer_output();
        let step = OUT_STEPS.len() - 1;
        let split_rows = OUT_STEPS[step];
        let lane = &split.lanes[0];
        split.status.copy_from_bytes(&[0; 4], 0);
        split.jobs.borrow_mut().clear();
        let mut batch = CommandBatch::new(&self.context)?;
        let encoded = (|| -> crate::Result<()> {
            split.stage_out(&mut batch, weights, 0, split_rows)?;
            split.launch_out(
                &mut batch,
                &self.context,
                &scratch.rotated_attention,
                0,
                0,
                CHECK_ROWS,
                step,
                0,
            )?;
            let matrix = weights.ptq1_matrix()?;
            self.kernels.matmul(
                &mut batch,
                matrix.row_slice(split_rows, matrix.rows() - split_rows)?,
                &scratch.rotated_attention,
                &scratch.ffn_gate,
                CHECK_ROWS,
            )?;
            self.project(
                &mut batch,
                weights,
                &scratch.rotated_attention,
                &scratch.branch,
                CHECK_ROWS,
            )?;
            split.join_out(
                &mut batch,
                0,
                CHECK_ROWS,
                split_rows,
                &scratch.ffn_gate,
                &lane.ffn,
            )
        })();
        let handoff = &split.handoff;
        if let Err(error) = encoded {
            handoff.retire("the start-up check failed to encode");
            return Err(error);
        }
        self.finish(batch)?;
        if !handoff.finish() {
            return Err(crate::Error::Generation(
                handoff
                    .retired()
                    .unwrap_or_else(|| "start-up evaluation failed".into()),
            ));
        }
        // SAFETY: the batch has completed.
        let (gpu, ane, status) = unsafe {
            (
                &self.scratch.branch.as_slice::<f32>()[..rows * WIDTH],
                &lane.ffn.as_slice::<f32>()[..rows * WIDTH],
                split.status.as_slice::<u32>()[0],
            )
        };
        if status != 0 {
            return Err(crate::Error::Generation(
                "start-up evaluation returned non-finite rows".into(),
            ));
        }
        Ok(relative_difference(gpu, ane))
    }

    /// A prefill block as a wavefront, restored and rerun on the GPU (which
    /// keeps prefill from then on) when the ANE's part failed.
    fn prefill_wave(
        &mut self,
        mut split: Box<AneSplit>,
        tokens: &[u32],
        output: BlockOutput<'_>,
        first: usize,
    ) -> crate::Result<()> {
        let start = self.position;
        let checkpoint = split.checkpoint.take().map_or_else(
            || self.pinned_prompt_checkpoint(),
            |checkpoint| {
                self.copy_prompt_state(&checkpoint, false)
                    .map(|()| checkpoint)
            },
        );
        let checkpoint = match checkpoint {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                self.ane = super::AnePrefill::Ready(split);
                return Err(error);
            }
        };
        let started = Instant::now();
        let failure = if split.begin_chunk() {
            let mut speculation = self.speculation.take();
            let encoded = self.encode_wave(&split, tokens, output, speculation.as_mut(), first);
            self.speculation = speculation;
            let finished = split.finish_chunk();
            match (encoded, finished) {
                (Ok(()), Ok(())) => {
                    let wait = mean_ffn_wait(
                        &split.jobs.borrow(),
                        &split.handoff.take_timeline(),
                        self.layers.len(),
                    );
                    let ms = started.elapsed().as_secs_f64() * 1e3;
                    let mut out_split = split.out_split.borrow_mut();
                    if out_split.learn(start + tokens.len(), ms, wait) {
                        out_split.save();
                    }
                    drop(out_split);
                    self.position = start + tokens.len();
                    if self.take_wave_logits(output) {
                        split.checkpoint = Some(checkpoint);
                        self.ane = super::AnePrefill::Ready(split);
                        return Ok(());
                    }
                    "the chunk's logits were non-finite".to_owned()
                }
                (Err(error), Ok(())) => format!("the chunk failed: {error}"),
                (_, Err(reason)) => reason,
            }
        } else {
            "a keepalive evaluation did not return".to_owned()
        };
        eprintln!(
            "local-ai: Neural Engine prefill stopped ({failure}); \
             the chunk at token {start} reruns on the GPU, which keeps prefill"
        );
        split.handoff.retire(&failure);
        self.ane = super::AnePrefill::Off(format!("retired: {failure}"));
        drop(split);
        self.copy_prompt_state(&checkpoint, true)?;
        self.position = start;
        self.prefill_on_gpu(tokens, output, first)
    }

    /// After a wavefront producing `output`: the greedy selection, and
    /// whether its logits (if any) are finite.
    fn take_wave_logits(&mut self, output: BlockOutput<'_>) -> bool {
        if !matches!(output, BlockOutput::LastLogits) {
            return true;
        }
        if self.device_greedy {
            self.selection = Selection::Last;
        }
        let finite = self
            .greedy
            .results(1)
            .iter()
            .all(|result| result.nonfinite == 0);
        if !finite {
            self.selection = Selection::None;
        }
        finite
    }

    /// Encode, submit and wait for the wavefront of `tokens` split after
    /// `first` rows. The caller advances the position once the ANE's part is
    /// known good.
    #[expect(unsafe_code)]
    fn encode_wave(
        &mut self,
        split: &AneSplit,
        tokens: &[u32],
        output: BlockOutput<'_>,
        mut ingest: Option<&mut Speculation>,
        first: usize,
    ) -> crate::Result<()> {
        let start = self.position;
        let (head, tail) = tokens.split_at(first);
        self.selection = Selection::None;
        self.distributions.described = Described::None;
        self.candidates.described = Described::None;
        if ingest.is_some() && matches!(output, BlockOutput::None) {
            return Err(crate::Error::InvalidArgument(
                "MTP ingestion needs output-normalized rows".into(),
            ));
        }
        self.reserve_kv(start + tokens.len(), ingest.as_deref_mut())?;
        let images = self.images.as_deref();
        let mut tail_embedding = split.embedding.clone();
        for (rows, buffer, position) in [
            (head, &mut self.scratch.embedding, start),
            (tail, &mut tail_embedding, start + first),
        ] {
            decode_embeddings(
                &self.package,
                rows,
                // SAFETY: no GPU work is in flight between blocks.
                unsafe { buffer.as_mut_slice::<f32>() },
                images.map(|images| (images, position)),
            )?;
        }
        if let Some(speculation) = ingest.as_deref_mut() {
            decode_embeddings(
                &self.package,
                head,
                speculation.mtp.embedding_rows(head.len())?,
                images.map(|images| (images, start)),
            )?;
        }
        let logits = matches!(output, BlockOutput::LastLogits);
        if logits {
            if self.top_k.is_some() {
                self.candidates.reserve(&self.context, 1)?;
            }
            if self.logprobs.is_some() {
                self.distributions.reserve(&self.context, 1)?;
            }
        }
        let head_rope = self.write_rope(start, head.len(), 0);
        // As `write_rope`, into the second half's rows.
        let tail_rope = images.map_or(Rope::Text(0), |images| {
            images.text_offset(start + first, tail.len()).map_or_else(
                || {
                    super::block::write_rope_rows(
                        &split.rope,
                        images,
                        start + first,
                        tail.len(),
                        0,
                    );
                    Rope::Rows(&split.rope, 0)
                },
                Rope::Text,
            )
        });
        let scratch = &self.scratch;
        let halves = [
            Half {
                rows: head.len() as u32,
                position: start,
                embedding: &scratch.embedding,
                hidden: &scratch.hidden,
                hidden_alt: &scratch.hidden_alt,
                rope: head_rope.map_or(Rope::Rows(&scratch.rope, 0), Rope::Text),
            },
            Half {
                rows: tail.len() as u32,
                position: start + first,
                embedding: &split.embedding,
                hidden: &split.hidden,
                hidden_alt: &split.hidden_alt,
                rope: tail_rope,
            },
        ];
        let mut batch = CommandBatch::new(&self.context)?;
        if let Err(error) =
            self.encode_wave_batch(&mut batch, split, &halves, output, ingest.as_deref())
        {
            // Release every GPU wait before the batch waits for its buffers.
            split.handoff.retire("the wavefront failed to encode");
            return Err(error);
        }
        self.finish(batch)?;
        if logits {
            if let Some(k) = self.top_k {
                self.candidates.described = Described::Last;
                self.candidates.k = k;
            }
            if let Some(top) = self.logprobs {
                self.distributions.described = Described::Last;
                self.distributions.top = top;
            }
        }
        Ok(())
    }

    fn encode_wave_batch(
        &self,
        batch: &mut CommandBatch,
        split: &AneSplit,
        halves: &[Half<'_>; 2],
        output: BlockOutput<'_>,
        ingest: Option<&Speculation>,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        for half in halves {
            self.kernels.transform(
                batch,
                &self.input_rotation,
                half.embedding,
                half.hidden,
                half.rows,
                HadamardDirection::Inverse,
            )?;
        }
        let step = split.out_split.borrow_mut().plan(
            function_for(halves[0].rows)?,
            (halves[0].rows + halves[1].rows) as usize,
            halves[0].position,
        );
        let split_rows = OUT_STEPS[step];
        split.stage(batch, &self.layers[0], 0)?;
        split.stage_out(batch, self.layers[0].mixer_output(), 0, split_rows)?;
        batch.submit_and_renew(&self.context)?;
        let launch = |batch: &mut CommandBatch, lane: usize, layer: usize| {
            split.launch(
                batch,
                &self.context,
                &scratch.rotated_hidden,
                lane,
                layer & 1,
                halves[lane].rows,
                layer,
            )
        };
        // Per layer, the ANE runs half A's FFN during half B's mixer and half
        // B's during half A's next mixer, and between them each half's
        // leading output projection rows while the GPU computes the rest.
        for (index, layer) in self.layers.iter().enumerate() {
            self.wave_mixer(batch, split, layer, index, &halves[0], 0, step)?;
            launch(batch, 0, index)?;
            if index > 0 {
                split.join(batch, 1, halves[1].rows)?;
            }
            self.wave_mixer(batch, split, layer, index, &halves[1], 1, step)?;
            launch(batch, 1, index)?;
            // The other set's last readers (the previous layer's jobs) were
            // joined above.
            if let Some(next) = self.layers.get(index + 1) {
                let set = (index + 1) & 1;
                split.stage(batch, next, set)?;
                split.stage_out(batch, next.mixer_output(), set, split_rows)?;
            }
            split.join(batch, 0, halves[0].rows)?;
        }
        split.join(batch, 1, halves[1].rows)?;
        debug_assert_eq!(self.layers.len(), LAYERS);
        if matches!(output, BlockOutput::None) {
            return Ok(());
        }
        // Every row is output-normalized, half by half: `scratch.normalized`
        // then holds the hidden rows the MTP head consumes for that half.
        for (lane, half) in halves.iter().enumerate() {
            self.ops.residual_normalize_transform(
                batch,
                &self.input_rotation,
                half.hidden_alt,
                &split.lanes[lane].ffn,
                &self.output_norm,
                [half.hidden, &scratch.normalized, &scratch.rotated_hidden],
                half.rows,
                self.epsilon,
            )?;
            if let Some(speculation) = ingest {
                if lane == 1 {
                    batch.blit_buffer_copies([BufferCopyRequest {
                        source: half.embedding,
                        source_offset: 0,
                        destination: speculation.mtp.embedding_buffer(),
                        destination_offset: 0,
                        size: half.rows as usize * WIDTH * size_of::<f32>(),
                    }])?;
                }
                self.encode_head_rows(batch, speculation, half.rows as usize, half.position)?;
            }
        }
        if matches!(output, BlockOutput::LastLogits) {
            // Only the last row is projected to the vocabulary.
            batch.blit_buffer_copies([BufferCopyRequest {
                source: &scratch.rotated_hidden,
                source_offset: (halves[1].rows as usize - 1) * WIDTH * size_of::<f32>(),
                destination: &scratch.embedding,
                destination_offset: 0,
                size: WIDTH * size_of::<f32>(),
            }])?;
            self.project(batch, &self.output, &scratch.embedding, &scratch.logits, 1)?;
            self.greedy.encode(batch, &scratch.logits, 1)?;
            if let Some(k) = self.top_k {
                self.candidates.encode(batch, &scratch.logits, 1, k)?;
            }
            if let Some(top) = self.logprobs {
                self.distributions.encode(batch, &scratch.logits, 1, top)?;
            }
        }
        Ok(())
    }

    /// A layer's mixer for `half` in `lane`: the residual add of the
    /// previous FFN with this layer's input normalization, the mixer with
    /// the first `OUT_STEPS[step]` rows of its output projection on the
    /// ANE, and the residual add with the FFN's normalization into
    /// `scratch.rotated_hidden`.
    #[expect(clippy::too_many_arguments)]
    fn wave_mixer(
        &self,
        batch: &mut CommandBatch,
        split: &AneSplit,
        layer: &Layer,
        index: usize,
        half: &Half<'_>,
        lane: usize,
        step: usize,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let ffn = &split.lanes[lane].ffn;
        if index == 0 {
            self.normalize_input(batch, half.hidden, &layer.attention_norm, half.rows)?;
        } else {
            self.ops.residual_normalize_transform(
                batch,
                &self.input_rotation,
                half.hidden_alt,
                ffn,
                &layer.attention_norm,
                [half.hidden, &scratch.normalized, &scratch.rotated_hidden],
                half.rows,
                self.epsilon,
            )?;
        }
        // `half.hidden` now holds the half's residual stream after the
        // previous layer, at positions from `half.position`: what
        // `encode_block` exposes to per-layer readers at this point.
        match &layer.attention {
            AttentionLayer::Recurrent(recurrent) => {
                self.recurrent_core(batch, recurrent, half.rows, None)?;
            }
            AttentionLayer::Full(full) => {
                self.attention_core_at(batch, full, half.rows, half.rope, half.position)?;
                self.rotate_attention(batch, half.rows)?;
            }
        }
        let rows = OUT_STEPS[step];
        if rows == 0 {
            let weights = layer.mixer_output();
            self.project(
                batch,
                weights,
                &scratch.rotated_attention,
                &scratch.branch,
                half.rows,
            )?;
            return self.residual_normalize(
                batch,
                half.hidden,
                half.hidden_alt,
                &layer.post_attention_norm,
                half.rows,
            );
        }
        split.launch_out(
            batch,
            &self.context,
            &scratch.rotated_attention,
            lane,
            index & 1,
            half.rows,
            step,
            index,
        )?;
        // The GPU's rows meanwhile, into a buffer the FFN alone uses.
        let matrix = layer.mixer_output().ptq1_matrix()?;
        self.kernels.matmul(
            batch,
            matrix.row_slice(rows, matrix.rows() - rows)?,
            &scratch.rotated_attention,
            &scratch.ffn_gate,
            half.rows,
        )?;
        split.join_out(
            batch,
            lane,
            half.rows,
            rows,
            &scratch.ffn_gate,
            &scratch.branch,
        )?;
        self.residual_normalize(
            batch,
            half.hidden,
            half.hidden_alt,
            &layer.post_attention_norm,
            half.rows,
        )
    }

    /// Make the next wavefront skip starting its first-half evaluation of
    /// `layer`, as an ANE that never answers would.
    #[cfg(test)]
    pub(super) fn inject_ane_fault(&mut self, layer: usize) -> bool {
        if let Some(split) = self.take_ane() {
            split.fault.set(Some(layer));
            self.ane = super::AnePrefill::Ready(split);
            true
        } else {
            false
        }
    }

    /// Whether the ANE path is still in use (compiled or compiling).
    #[cfg(test)]
    pub(super) const fn ane_active(&self) -> bool {
        self.ane.active()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_literals_round_trip() {
        assert_eq!(hex(1.0 / 128.0), "0x1p-7");
        assert_eq!(hex(1.0), "0x1p+0");
        assert_eq!(hex(127.0), "0x1.fcp+6");
        assert_eq!(hex(128.0 / 127.0), "0x1.0204081020408p+0");
        assert_eq!(fp16(0.5), "fp16(0x1p-1)");
    }

    #[test]
    fn halves_cover_the_block() {
        for rows in HALF_ALIGN + 1..=WAVE_ROWS {
            let first = rows.div_ceil(2).next_multiple_of(HALF_ALIGN).min(HALF_ROWS);
            assert!(first <= HALF_ROWS && rows - first <= HALF_ROWS && rows > first);
            assert!(FUNCTION_ROWS.iter().any(|&r| r as usize >= first));
        }
    }

    #[test]
    fn rotation_blob_is_the_forward_rotation() {
        let signs = (0..FFN)
            .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let blob = rotation_blob(&signs);
        assert_eq!(blob.len(), 128 + FFN * ROTATION_BLOCK * 2);
        let value = |row: usize, input: usize| {
            let at = 128 + (row * ROTATION_BLOCK + input) * 2;
            half::f16::from_le_bytes([blob[at], blob[at + 1]]).to_f32()
        };
        // Row 0 of each group is the signs over 32; row 1 alternates them.
        for group in [0, 5, 16] {
            for input in [0, 1, 2, 1023] {
                let sign = signs[group * ROTATION_BLOCK + input];
                assert_eq!(value(group * ROTATION_BLOCK, input), sign / 32.0);
                let alternate = if input % 2 == 1 { -sign } else { sign };
                assert_eq!(value(group * ROTATION_BLOCK + 1, input), alternate / 32.0);
            }
        }
    }

    #[test]
    fn program_declares_every_input_once() {
        let text = program_text() + &out_program_text();
        let ffns = FUNCTION_ROWS.len();
        let outs = ffns * (OUT_STEPS.len() - 1);
        for rows in FUNCTION_ROWS {
            assert!(text.contains(&format!("func ffn{rows}<ios18>(")));
            for split in &OUT_STEPS[1..] {
                assert!(text.contains(&format!("func out{rows}_{split}<ios18>(")));
                // The function reads the first `split` rows of the
                // full-height weight surfaces.
                assert!(text.contains(&format!(
                    "shape=[1, 1, {split}, 3072], strides=[{}, ",
                    OUT.stride * WIDTH
                )));
            }
        }
        let ffn_inputs = ["x0", "x1", "wg0", "wu0", "wg1", "wu1", "sg", "su"]
            .into_iter()
            .map(str::to_owned)
            .chain((0..7).map(|index| format!("wd{index}")))
            .chain(["sd".to_owned()])
            .map(|name| (name, ffns));
        let out_inputs = ["o0", "o1", "wo0", "wo1", "so"]
            .into_iter()
            .map(|name| (name.to_owned(), outs));
        for (name, count) in ffn_inputs
            .chain(out_inputs)
            .chain([("tx".to_owned(), ffns + outs)])
        {
            let declared = text.matches(&format!("]> {name},")).count()
                + text.matches(&format!("]> {name})")).count();
            assert_eq!(declared, count, "{name}");
            assert_eq!(
                text.matches(&format!("(input = {name})")).count(),
                count,
                "{name}"
            );
        }
        assert_eq!(text.matches("-> (y);").count(), ffns + outs);
    }

    fn finish(split: &mut OutSplit, rows: usize, start: usize, ms: f64) -> (usize, bool) {
        let step = split.plan(3, rows, start);
        let changed = split.learn(start + rows, ms, Some(0.0));
        (step, changed)
    }

    #[test]
    fn out_split_alternates_base_and_probe_chunks() {
        let mut split = OutSplit::new([4; FUNCTION_ROWS.len()]);
        // Base, probe up, base: the probe ran no faster, so nothing moves and
        // the next probe goes down.
        assert_eq!(finish(&mut split, 1024, 0, 2000.0), (4, false));
        assert_eq!(finish(&mut split, 1024, 1024, 2010.0), (5, false));
        assert_eq!(finish(&mut split, 1024, 2048, 2020.0), (4, false));
        assert!(!split.probe_up[3]);
        // Probe down, faster than its neighbours' mean by over the margin.
        assert_eq!(finish(&mut split, 1024, 3072, 2000.0), (3, false));
        assert_eq!(finish(&mut split, 1024, 4096, 2040.0), (4, true));
        assert_eq!(split.steps[3], 3);
        assert_eq!(split.steps[0], 4, "other functions keep theirs");
        // The next chunk probes up from the new split.
        assert_eq!(split.plan(3, 1024, 5120), 4);
    }

    #[test]
    fn out_split_compares_only_contiguous_equal_chunks() {
        let mut split = OutSplit::new([4; FUNCTION_ROWS.len()]);
        assert_eq!(finish(&mut split, 1024, 0, 2000.0), (4, false));
        assert_eq!(finish(&mut split, 1024, 1024, 1000.0), (5, false));
        // A new prompt: the comparison restarts with a base chunk.
        assert_eq!(finish(&mut split, 1024, 0, 2000.0), (4, false));
        assert_eq!(finish(&mut split, 1024, 1024, 1000.0), (5, false));
        // A shorter last chunk is not compared with full ones.
        assert_eq!(finish(&mut split, 600, 2048, 100.0), (4, false));
        assert_eq!(split.steps[3], 4);
        // At the top the probe would equal the split: a base chunk instead.
        let mut top = OutSplit::new([OUT_STEPS.len() - 1; FUNCTION_ROWS.len()]);
        finish(&mut top, 1024, 0, 2000.0);
        assert_eq!(top.plan(3, 1024, 1024), OUT_STEPS.len() - 1);
        assert_eq!(top.current.map(|chunk| chunk.probe), Some(false));
    }

    #[test]
    fn out_split_backs_off_when_the_gpu_waits_on_the_ane() {
        let mut split = OutSplit::new([4; FUNCTION_ROWS.len()]);
        split.plan(3, 1024, 0);
        assert!(split.learn(1024, 2000.0, Some(0.8)));
        assert_eq!(split.steps[3], 3);
        split.plan(3, 1024, 1024);
        assert!(!split.learn(2048, 2000.0, Some(0.2)));
        assert_eq!(OUT_STEPS[0], 0, "step 0 leaves the projection to the GPU");
    }

    #[test]
    fn ffn_wait_pairs_each_layers_halves() {
        let base = Instant::now();
        let at = |ms: u64| Some(base + Duration::from_millis(ms));
        let job = |out, lane, layer| Job { out, lane, layer };
        // Layer 0: out A, FFN A, out B, FFN B; layer 1 likewise.
        let jobs = [
            job(true, 0, 0),
            job(false, 0, 0),
            job(true, 1, 0),
            job(false, 1, 0),
            job(true, 0, 1),
            job(false, 0, 1),
            job(true, 1, 1),
            job(false, 1, 1),
            job(false, 0, 2),
        ];
        let timeline = [
            (at(0), at(2)),
            (at(5), at(25)),
            (at(20), at(22)),
            (at(27), at(47)),
            (at(40), at(42)),
            (at(44), at(64)),
            (at(60), at(62)),
            (at(70), at(90)),
            (at(92), None),
        ];
        // FFN B of layer 0 reported 3 ms after FFN A of layer 1 left; FFN B
        // of layer 1 2 ms before FFN A of layer 2 did.
        assert_eq!(mean_ffn_wait(&jobs, &timeline, 3), Some(1.5));
        assert_eq!(mean_ffn_wait(&jobs[..2], &timeline[..2], 3), None);
    }

    #[test]
    fn chip_generations() {
        assert_eq!(chip_generation("Apple M4 Pro"), Some(4));
        assert_eq!(chip_generation("Apple M2"), Some(2));
        assert_eq!(chip_generation("AMD Radeon"), None);
    }
}
