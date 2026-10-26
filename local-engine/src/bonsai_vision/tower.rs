//! The Qwen3.5 vision tower of the Bonsai mmproj, on Metal.
//!
//! Width 1152, 27 pre-LN blocks (16 heads x 72, 2-D `RoPE` θ 10,000 over
//! rows then columns, bidirectional attention over the whole image or video
//! frame group,
//! GELU-tanh MLP 4304), a learned 48 x 48 position table interpolated
//! bilinearly to the patch grid, then the merger: `LayerNorm`, 2x2 patch
//! concatenation (4608), GELU (exact), 5120. F32 activations throughout; the
//! BF16 matrices and F32 biases bind straight from the mapped GGUF.

use std::io::Read as _;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use local_metal::batch::CommandBatch;
use local_metal::bonsai_vision::{Epilogue, Linear, VisionKernels};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::shaders::ShaderLibrary;
use sha2::{Digest as _, Sha256};

use super::preprocess::{PATCH_VALUES, PreparedImage, merge_order};
use super::{EmbeddingCache, cancelled, rotate_rows};
use crate::activity::EngineActivity;
use crate::bonsai::{BonsaiMetalTensor, BonsaiPackage, BonsaiTensorType, WIDTH};
use crate::bonsai_model::CancelToken;

/// The projector file, installed beside the Bonsai GGUF.
pub const VISION_PROJECTOR_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-BF16.gguf";
/// Its size and SHA-256 at the pinned repository revision.
pub const VISION_PROJECTOR_BYTES: u64 = 931_145_856;
pub const VISION_PROJECTOR_SHA256: &str =
    "e287342d92332fa3577ed1d42e921dac9370c08da58ba9337fa450f6cc76cfd7";

const HIDDEN: usize = 1152;
const HEAD_DIM: usize = 72;
const FFN: usize = 4304;
const BLOCKS: usize = 27;
const MERGED: usize = 4 * HIDDEN;
const GRID: usize = 48;
const EPSILON: f32 = 1e-6;
const ROPE_THETA: f32 = 10_000.0;
/// Query x key pairs per attention submission, so one command buffer of a
/// large image stays short beside running decode steps.
const ATTENTION_PAIRS: usize = 1 << 24;
/// Longest wait for a decode step between submissions.
const DECODE_YIELD_LIMIT: Duration = Duration::from_millis(250);
/// Encoded images kept for reuse (a 1,024-token image holds 20 MiB).
const CACHE_BYTES: usize = 512 << 20;

struct Block {
    ln1: (BonsaiMetalTensor, BonsaiMetalTensor),
    ln2: (BonsaiMetalTensor, BonsaiMetalTensor),
    qkv: (BonsaiMetalTensor, BonsaiMetalTensor),
    out: (BonsaiMetalTensor, BonsaiMetalTensor),
    up: (BonsaiMetalTensor, BonsaiMetalTensor),
    down: (BonsaiMetalTensor, BonsaiMetalTensor),
}

struct Gpu {
    context: MetalContext,
    kernels: VisionKernels,
    /// `W(t=0) + W(t=1)` of the patch convolution, `[1152][768]` F32: the
    /// processor feeds two identical temporal slices.
    patch: MetalBuffer,
    /// `[W(t=0) | W(t=1)]` per output row, `[1152][1536]` F32: a video frame
    /// group's two frames.
    patch_pair: MetalBuffer,
    patch_bias: BonsaiMetalTensor,
    blocks: Vec<Block>,
    post_ln: (BonsaiMetalTensor, BonsaiMetalTensor),
    mm0: (BonsaiMetalTensor, BonsaiMetalTensor),
    mm2: (BonsaiMetalTensor, BonsaiMetalTensor),
    /// The learned position table, `[48 * 48][1152]`.
    positions: Vec<f32>,
    /// Keeps the mapping the tensors bind alive.
    _package: BonsaiPackage,
}

/// The loaded, verified vision tower; shareable across threads.
pub struct VisionEncoder {
    gpu: Mutex<Gpu>,
    digest: [u8; 32],
    /// The decoder's input signs (width 5120).
    signs: Vec<f32>,
    cache: EmbeddingCache,
}

// SAFETY: Metal devices, queues, pipelines and buffers may move between
// threads; every GPU object sits behind the mutex, so none is used by two
// threads at once.
#[allow(unsafe_code)]
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Gpu {}

const fn linear(
    pair: &(BonsaiMetalTensor, BonsaiMetalTensor),
    rows: usize,
    columns: usize,
) -> Linear<'_> {
    Linear {
        f32: false,
        weights: pair.0.buffer(),
        weights_offset: pair.0.offset(),
        bias: pair.1.buffer(),
        bias_offset: pair.1.offset(),
        rows: rows as u32,
        columns: columns as u32,
    }
}

fn sha256_file(path: &Path) -> crate::Result<[u8; 32]> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 8 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
            out
        })
}

impl VisionEncoder {
    /// Verify (size and SHA-256 against the pinned export) and load the
    /// projector at `path` for a decoder whose input signs (width 5120) are
    /// `signs`.
    ///
    /// # Errors
    ///
    /// Fails when the file is not the pinned projector, a tensor is missing
    /// or misshapen, or Metal cannot be set up.
    #[allow(clippy::too_many_lines)]
    pub fn open(path: &Path, signs: &[f32]) -> crate::Result<Self> {
        let bytes = std::fs::metadata(path)?.len();
        if bytes != VISION_PROJECTOR_BYTES {
            return Err(crate::Error::InvalidFormat(format!(
                "vision projector {} is {bytes} bytes, not the pinned {VISION_PROJECTOR_BYTES}",
                path.display()
            )));
        }
        let digest = sha256_file(path)?;
        if hex(&digest) != VISION_PROJECTOR_SHA256 {
            return Err(crate::Error::InvalidFormat(format!(
                "vision projector {} has sha256 {}, not the pinned {VISION_PROJECTOR_SHA256}",
                path.display(),
                hex(&digest)
            )));
        }
        if signs.len() != WIDTH {
            return Err(crate::Error::InvalidArgument(
                "decoder input signs must be 5120 wide".into(),
            ));
        }
        let package = BonsaiPackage::open_projector(path)?;
        if package.metadata_string("clip.projector_type")? != "qwen3vl_merger"
            || package.metadata_u32("clip.vision.block_count")? as usize != BLOCKS
            || package.metadata_u32("clip.vision.embedding_length")? as usize != HIDDEN
            || package.metadata_u32("clip.vision.projection_dim")? as usize != WIDTH
        {
            return Err(crate::Error::InvalidFormat(
                "vision projector is not the Bonsai Qwen3.5 tower".into(),
            ));
        }
        let context = MetalContext::new()?;
        let shaders = ShaderLibrary::new(context.device())?;
        let kernels = VisionKernels::new(&context, &shaders)?;
        let tensor = |name: &str, kind: BonsaiTensorType, dims: &[u64]| {
            let found = package.tensor(name)?;
            if found.tensor_type() != kind || found.dimensions() != dims {
                return Err(crate::Error::InvalidFormat(format!(
                    "vision tensor {name} is {:?} {:?}",
                    found.tensor_type(),
                    found.dimensions()
                )));
            }
            package.metal_tensor(context.device(), name)
        };
        let pair = |name: &str, rows: usize, columns: usize| -> crate::Result<_> {
            Ok((
                tensor(
                    &format!("{name}.weight"),
                    BonsaiTensorType::Bf16,
                    &[columns as u64, rows as u64],
                )?,
                tensor(
                    &format!("{name}.bias"),
                    BonsaiTensorType::F32,
                    &[rows as u64],
                )?,
            ))
        };
        let norm = |name: &str| -> crate::Result<_> {
            Ok((
                tensor(
                    &format!("{name}.weight"),
                    BonsaiTensorType::F32,
                    &[HIDDEN as u64],
                )?,
                tensor(
                    &format!("{name}.bias"),
                    BonsaiTensorType::F32,
                    &[HIDDEN as u64],
                )?,
            ))
        };
        let f32s = |name: &str, dims: &[u64]| -> crate::Result<Vec<f32>> {
            let found = package.tensor(name)?;
            if found.tensor_type() != BonsaiTensorType::F32 || found.dimensions() != dims {
                return Err(crate::Error::InvalidFormat(format!(
                    "vision tensor {name} is misshapen"
                )));
            }
            Ok(package
                .bytes(name)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&bytes| f32::from_le_bytes(bytes))
                .collect())
        };
        let patch_dims = [16, 16, 3, HIDDEN as u64];
        let (first, second) = (
            f32s("v.patch_embd.weight", &patch_dims)?,
            f32s("v.patch_embd.weight.1", &patch_dims)?,
        );
        let patch_weights = first
            .iter()
            .zip(&second)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let patch_pair = first
            .chunks(PATCH_VALUES)
            .zip(second.chunks(PATCH_VALUES))
            .flat_map(|(a, b)| a.iter().chain(b))
            .copied()
            .collect::<Vec<_>>();
        let blocks = (0..BLOCKS)
            .map(|index| {
                let name = |suffix: &str| format!("v.blk.{index}.{suffix}");
                Ok(Block {
                    ln1: norm(&name("ln1"))?,
                    ln2: norm(&name("ln2"))?,
                    qkv: pair(&name("attn_qkv"), 3 * HIDDEN, HIDDEN)?,
                    out: pair(&name("attn_out"), HIDDEN, HIDDEN)?,
                    up: pair(&name("ffn_up"), FFN, HIDDEN)?,
                    down: pair(&name("ffn_down"), HIDDEN, FFN)?,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let gpu = Gpu {
            patch: MetalBuffer::from_slice(context.device(), &patch_weights)?,
            patch_pair: MetalBuffer::from_slice(context.device(), &patch_pair)?,
            patch_bias: tensor("v.patch_embd.bias", BonsaiTensorType::F32, &[HIDDEN as u64])?,
            blocks,
            post_ln: norm("v.post_ln")?,
            mm0: pair("mm.0", MERGED, MERGED)?,
            mm2: pair("mm.2", WIDTH, MERGED)?,
            positions: f32s(
                "v.position_embd.weight",
                &[HIDDEN as u64, (GRID * GRID) as u64],
            )?,
            kernels,
            context,
            _package: package,
        };
        Ok(Self {
            gpu: Mutex::new(gpu),
            digest,
            signs: signs.to_vec(),
            cache: EmbeddingCache::new(CACHE_BYTES),
        })
    }

    /// The projector's SHA-256.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The cached decoder input rows of the image or frame group whose
    /// digest is `digest`, if it was encoded recently.
    #[must_use]
    pub fn cached(&self, digest: &[u8; 32]) -> Option<Arc<[f32]>> {
        self.cache.get(digest)
    }

    /// The decoder input rows (`[tokens][5120]`, in the checkpoint's
    /// embedding basis) of `image`, from the cache when it was encoded
    /// recently. With `activity`, every submission after the first waits for
    /// a running decode step first; `cancel` is checked before each
    /// submission.
    ///
    /// # Errors
    ///
    /// Metal failures; [`crate::Error::Generation`] when cancelled.
    pub fn encode(
        &self,
        image: &PreparedImage,
        activity: Option<&EngineActivity>,
        cancel: Option<&CancelToken>,
    ) -> crate::Result<Arc<[f32]>> {
        if let Some(rows) = self.cache.get(&image.digest) {
            return Ok(rows);
        }
        let mut rows = self.encode_raw(image, activity, cancel, &mut |_, _| {})?;
        rotate_rows(&mut rows, &self.signs);
        let rows: Arc<[f32]> = rows.into();
        self.cache.insert(image.digest, Arc::clone(&rows));
        Ok(rows)
    }

    /// The tower's output for `image` (`[tokens][5120]`, as the reference
    /// returns it), reporting `block0_in` (the embedded patches) and
    /// `block{i}_out` to `probe`. `activity` and `cancel` as for
    /// [`Self::encode`].
    ///
    /// # Errors
    ///
    /// Metal failures; [`crate::Error::Generation`] when cancelled.
    #[doc(hidden)]
    #[allow(unsafe_code, clippy::too_many_lines)]
    pub fn encode_raw(
        &self,
        image: &PreparedImage,
        activity: Option<&EngineActivity>,
        cancel: Option<&CancelToken>,
        probe: &mut dyn FnMut(&str, &[f32]),
    ) -> crate::Result<Vec<f32>> {
        let gpu = self.gpu.lock().unwrap_or_else(PoisonError::into_inner);
        let (grid_h, grid_w) = image.grid();
        let patches = image.patches();
        let tokens = image.tokens();
        let count = patches as u32;
        let device = gpu.context.device();
        let floats = |width: usize| MetalBuffer::empty(device, patches * width * 4);
        let input = MetalBuffer::from_slice(device, &image.patch_rows())?;
        let hidden =
            MetalBuffer::from_slice(device, &position_rows(&gpu.positions, grid_h, grid_w))?;
        let (cosines, sines) = rope_tables(grid_h, grid_w);
        let cosines = MetalBuffer::from_slice(device, &cosines)?;
        let sines = MetalBuffer::from_slice(device, &sines)?;
        let normed = floats(HIDDEN)?;
        let qkv = floats(3 * HIDDEN)?;
        let attention = floats(HIDDEN)?;
        let ffn = floats(FFN)?;
        let kernels = &gpu.kernels;
        let mut first = true;
        // A cancelled request submits nothing more (and gives the tower's
        // lock back); a submitted buffer always runs to completion.
        let mut submit = |batch: CommandBatch| -> crate::Result<()> {
            if !first && let Some(activity) = activity {
                activity.yield_to_decode(DECODE_YIELD_LIMIT);
            }
            if cancel.is_some_and(CancelToken::is_cancelled) {
                return Err(cancelled());
            }
            first = false;
            batch.commit_and_wait()?;
            Ok(())
        };
        let mut batch = CommandBatch::new(&gpu.context)?;
        kernels.gemm(
            &mut batch,
            Linear {
                f32: true,
                weights: if image.next.is_some() {
                    &gpu.patch_pair
                } else {
                    &gpu.patch
                },
                weights_offset: 0,
                bias: gpu.patch_bias.buffer(),
                bias_offset: gpu.patch_bias.offset(),
                rows: HIDDEN as u32,
                columns: image.patch_values() as u32,
            },
            &input,
            &hidden,
            count,
            Epilogue::Accumulate,
        )?;
        submit(batch)?;
        // SAFETY: `submit` waited for the batch, and nothing is encoded while
        // the probe reads.
        probe("block0_in", unsafe {
            &hidden.as_slice::<f32>()[..patches * HIDDEN]
        });
        let chunk = (ATTENTION_PAIRS / patches).max(64) / 64 * 64;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        for (index, block) in gpu.blocks.iter().enumerate() {
            let mut batch = CommandBatch::new(&gpu.context)?;
            let norm = |batch: &mut CommandBatch, pair: &(BonsaiMetalTensor, BonsaiMetalTensor)| {
                kernels.layernorm(
                    batch,
                    &hidden,
                    (pair.0.buffer(), pair.0.offset()),
                    (pair.1.buffer(), pair.1.offset()),
                    &normed,
                    count,
                    HIDDEN as u32,
                    EPSILON,
                )
            };
            norm(&mut batch, &block.ln1)?;
            kernels.gemm(
                &mut batch,
                linear(&block.qkv, 3 * HIDDEN, HIDDEN),
                &normed,
                &qkv,
                count,
                Epilogue::Store,
            )?;
            kernels.rope(&mut batch, &qkv, &cosines, &sines, count)?;
            let mut start = 0;
            while start < count {
                let end = (start + chunk as u32).min(count);
                kernels.attention(&mut batch, &qkv, 0, &attention, 0, count, start..end, scale)?;
                start = end;
                if start < count {
                    submit(batch)?;
                    batch = CommandBatch::new(&gpu.context)?;
                }
            }
            kernels.gemm(
                &mut batch,
                linear(&block.out, HIDDEN, HIDDEN),
                &attention,
                &hidden,
                count,
                Epilogue::Accumulate,
            )?;
            norm(&mut batch, &block.ln2)?;
            kernels.gemm(
                &mut batch,
                linear(&block.up, FFN, HIDDEN),
                &normed,
                &ffn,
                count,
                Epilogue::GeluTanh,
            )?;
            kernels.gemm(
                &mut batch,
                linear(&block.down, HIDDEN, FFN),
                &ffn,
                &hidden,
                count,
                Epilogue::Accumulate,
            )?;
            submit(batch)?;
            probe(
                &format!("block{index}_out"),
                // SAFETY: as for `block0_in`.
                unsafe { &hidden.as_slice::<f32>()[..patches * HIDDEN] },
            );
        }
        let mut batch = CommandBatch::new(&gpu.context)?;
        kernels.layernorm(
            &mut batch,
            &hidden,
            (gpu.post_ln.0.buffer(), gpu.post_ln.0.offset()),
            (gpu.post_ln.1.buffer(), gpu.post_ln.1.offset()),
            &normed,
            count,
            HIDDEN as u32,
            EPSILON,
        )?;
        // Four consecutive patch rows are one merged row of 4608.
        kernels.gemm(
            &mut batch,
            linear(&gpu.mm0, MERGED, MERGED),
            &normed,
            &ffn,
            tokens as u32,
            Epilogue::GeluErf,
        )?;
        kernels.gemm(
            &mut batch,
            linear(&gpu.mm2, WIDTH, MERGED),
            &ffn,
            &qkv,
            tokens as u32,
            Epilogue::Store,
        )?;
        submit(batch)?;
        drop(gpu);
        // SAFETY: `submit` waited for the batch; the borrow ends with this copy.
        let rows = unsafe { qkv.as_slice::<f32>()[..tokens * WIDTH].to_vec() };
        if rows.iter().any(|value| !value.is_finite()) {
            return Err(crate::Error::Generation(
                "non-finite vision tower output".into(),
            ));
        }
        Ok(rows)
    }
}

/// torch's float32 `linspace(0, 47, steps)`, rounded as torch rounds it
/// (separate multiply and subtract, not a fused multiply-add).
#[allow(clippy::suboptimal_flops)]
fn linspace(steps: usize) -> Vec<f32> {
    let end = (GRID - 1) as f32;
    if steps == 1 {
        return vec![0.0];
    }
    let step = end / (steps - 1) as f32;
    let halfway = steps / 2;
    (0..steps)
        .map(|index| {
            if index < halfway {
                step * index as f32
            } else {
                end - step * (steps - 1 - index) as f32
            }
        })
        .collect()
}

/// The learned positions bilinearly interpolated to a `grid_h x grid_w`
/// patch grid (`fast_pos_embed_interpolate`), in merge order, with torch's
/// unfused products and sums.
#[allow(clippy::suboptimal_flops)]
pub(super) fn position_rows(table: &[f32], grid_h: usize, grid_w: usize) -> Vec<f32> {
    let (hs, ws) = (linspace(grid_h), linspace(grid_w));
    let corners = |value: f32| {
        let floor = value as usize;
        (floor, (floor + 1).min(GRID - 1), value - floor as f32)
    };
    let mut out = Vec::with_capacity(grid_h * grid_w * HIDDEN);
    for (h, w) in merge_order(grid_h, grid_w) {
        let (h0, h1, dh) = corners(hs[h]);
        let (w0, w1, dw) = corners(ws[w]);
        let terms = [
            (h0 * GRID + w0, (1.0 - dh) * (1.0 - dw)),
            (h0 * GRID + w1, (1.0 - dh) * dw),
            (h1 * GRID + w0, dh * (1.0 - dw)),
            (h1 * GRID + w1, dh * dw),
        ];
        let row = |index: usize| &table[index * HIDDEN..(index + 1) * HIDDEN];
        for column in 0..HIDDEN {
            let mut sum = row(terms[0].0)[column] * terms[0].1;
            for &(index, weight) in &terms[1..] {
                sum += row(index)[column] * weight;
            }
            out.push(sum);
        }
    }
    out
}

/// Cosine and sine tables `[patches][36]` of the 2-D vision `RoPE` in merge
/// order: angles `row * inv_freq[i]` for the first 18 dims, `column *
/// inv_freq[i]` for the next 18, `inv_freq[i] = 10000^(-2i / 36)`.
pub(super) fn rope_tables(grid_h: usize, grid_w: usize) -> (Vec<f32>, Vec<f32>) {
    let half = HEAD_DIM / 4;
    let inverse = (0..half)
        .map(|index| 1.0 / ROPE_THETA.powf((2 * index) as f32 / (HEAD_DIM / 2) as f32))
        .collect::<Vec<_>>();
    let patches = grid_h * grid_w;
    let mut cosines = Vec::with_capacity(patches * 2 * half);
    let mut sines = Vec::with_capacity(patches * 2 * half);
    for position in merge_order(grid_h, grid_w) {
        for coordinate in <[usize; 2]>::from(position) {
            for &frequency in &inverse {
                let angle = coordinate as f32 * frequency;
                cosines.push(angle.cos());
                sines.push(angle.sin());
            }
        }
    }
    (cosines, sines)
}
