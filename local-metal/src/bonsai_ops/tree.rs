//! `KFlash` tree verify: the operations that let one verify pass run a tree of
//! draft rows, each from its own path, and commit the accepted path.
//!
//! The tree descriptor is a small buffer of `i32` words: each row's parent
//! (row 0, the anchor, has -1) at [`TREE_PARENTS`], each row's
//! ancestor-or-self bit mask at [`TREE_MASKS`], and an accepted path's rows at
//! [`TREE_PATH`] for the commit.

use super::{
    AttentionKernel, AttentionWorkspace, BonsaiOps, CommandBatch, GdnStateFormat, KV_HEADS,
    KvLayout, MetalBuffer, Q_HEADS, SPLIT, SPLIT_HEADS, SPLIT_TENSOR_LONG_MIN_PREFIX,
    SPLIT_TENSOR_MIN_PREFIX, SPLIT_TENSOR_SHORT, arg, need, no_alias, sequence_bytes,
    tensor_split_tokens,
};

/// Rows of the largest tree (`BONSAI_TREE_ROWS` in the shader).
pub const TREE_ROWS: u32 = 8;
/// Word index of the parents in the tree descriptor.
pub const TREE_PARENTS: usize = 0;
/// Word index of the ancestor-or-self masks.
pub const TREE_MASKS: usize = TREE_ROWS as usize;
/// Word index of the accepted path's rows.
pub const TREE_PATH: usize = 2 * TREE_ROWS as usize;
/// Words of a tree descriptor.
pub const TREE_DESCRIPTOR_WORDS: usize = 32;

/// Indexes in `BonsaiOps::p` of the tree pipelines (appended to its names).
const CONV_TREE: usize = 27;
const ATTENTION_TAIL: usize = 31;
const COPY_ROWS: usize = 32;
const GATHER_ROWS: usize = 33;

impl GdnStateFormat {
    /// Index of the tree recurrence pipeline for this state format.
    const fn tree_pipeline(self) -> usize {
        match self {
            Self::F32 => 28,
            Self::F16 => 29,
            Self::Bf16 => 30,
        }
    }
}

/// How a tree verify attends the history below its anchor: the split kernel
/// and split size [`BonsaiOps::attention_row_kv`] would use for that prefix.
#[derive(Clone, Copy)]
enum History {
    Simd { split_tokens: u32 },
    Tensor { pipeline: usize, split_tokens: u32 },
}

fn check_rows(tokens: u32) -> crate::Result<()> {
    if (1..=TREE_ROWS).contains(&tokens) {
        Ok(())
    } else {
        Err(arg("a tree verify holds 1..=8 rows"))
    }
}

impl BonsaiOps {
    fn tree_history(&self, layout: KvLayout, prefix: u32) -> History {
        let tensor = if layout.is_f16() {
            Some(18)
        } else {
            self.quantized_tensor(layout)
        };
        if let Some(pipeline) = tensor
            && self.attention_kernel != AttentionKernel::SimdF32
            && prefix >= SPLIT_TENSOR_MIN_PREFIX
        {
            let split_tokens = if prefix >= SPLIT_TENSOR_LONG_MIN_PREFIX {
                tensor_split_tokens(prefix)
            } else {
                SPLIT_TENSOR_SHORT
            };
            return History::Tensor {
                pipeline,
                split_tokens,
            };
        }
        History::Simd {
            split_tokens: if prefix < SPLIT_TENSOR_MIN_PREFIX {
                self.short_split_tokens
            } else {
                SPLIT
            },
        }
    }

    /// Partial records per (row, query head) of a tree verify whose anchor
    /// sits at `position`: the history's splits plus the tree zone's one.
    fn tree_splits(&self, layout: KvLayout, position: u32) -> u32 {
        if position == 0 {
            return 1;
        }
        let split_tokens = match self.tree_history(layout, position) {
            History::Simd { split_tokens } | History::Tensor { split_tokens, .. } => split_tokens,
        };
        position.div_ceil(split_tokens) + 1
    }

    /// Bytes of partial records [`Self::attention_tree_kv`] needs for `tokens`
    /// rows anchored at `position`.
    #[must_use]
    pub fn tree_partial_bytes(&self, layout: KvLayout, position: u32, tokens: u32) -> usize {
        self.tree_splits(layout, position) as usize * tokens as usize * Q_HEADS as usize * 258 * 4
    }

    /// [`Self::conv_l2_decay`] over a tree of `tokens` rows: each row's
    /// convolution window follows its path (`tree`, a tree descriptor) from
    /// `history`, which is read only.
    pub fn conv_l2_decay_tree(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        history: &MetalBuffer,
        out: &MetalBuffer,
        epsilon: f32,
        decay_inputs: [&MetalBuffer; 4],
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        tree: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        check_rows(tokens)?;
        let n = 10240usize;
        let [a, alpha, dt, beta_raw] = decay_inputs;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(arg("invalid L2 epsilon"));
        }
        let bytes = sequence_bytes(tokens, n as u32)?;
        need(input, bytes)?;
        need(weights, n * 4 * 4)?;
        need(history, n * 3 * 4)?;
        need(out, bytes)?;
        need(tree, TREE_DESCRIPTOR_WORDS * 4)?;
        for v in [a, dt] {
            need(v, 48 * 4)?;
        }
        for v in [alpha, beta_raw, decay, beta] {
            need(v, sequence_bytes(tokens, 48)?)?;
        }
        no_alias(
            out,
            &[
                input, weights, history, a, alpha, dt, beta_raw, decay, beta, tree,
            ],
        )?;
        no_alias(decay, &[input, a, alpha, dt, beta_raw, beta, history, tree])?;
        no_alias(beta, &[input, a, alpha, dt, beta_raw, history, tree])?;
        self.go(
            b,
            CONV_TREE,
            &[
                (input, 0),
                (weights, 0),
                (history, 0),
                (out, 0),
                (a, 0),
                (alpha, 0),
                (dt, 0),
                (beta_raw, 0),
                (decay, 0),
                (beta, 0),
                (tree, 0),
            ],
            &[tokens],
            &[epsilon],
            n / 128 + 1,
            128,
        );
        Ok(())
    }

    /// The recurrence over a tree of `tokens` rows, each from its parent's
    /// state (`tree`, a tree descriptor), reading `state` (in `format`) and
    /// writing only each row's output.
    pub fn gdn_tree(
        &self,
        b: &mut CommandBatch,
        format: GdnStateFormat,
        qkv: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        out: &MetalBuffer,
        tree: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        check_rows(tokens)?;
        need(qkv, sequence_bytes(tokens, 10240)?)?;
        need(decay, sequence_bytes(tokens, 48)?)?;
        need(beta, sequence_bytes(tokens, 48)?)?;
        need(state, format.state_bytes())?;
        need(out, sequence_bytes(tokens, 6144)?)?;
        need(tree, TREE_DESCRIPTOR_WORDS * 4)?;
        no_alias(out, &[qkv, decay, beta, state, tree])?;
        self.go(
            b,
            format.tree_pipeline(),
            &[
                (qkv, 0),
                (decay, 0),
                (beta, 0),
                (state, 0),
                (out, 0),
                (tree, 0),
            ],
            &[tokens],
            &[],
            48 * 128 / 4,
            32,
        );
        Ok(())
    }

    /// The history half of [`Self::attention_tree_kv`] on a Q8 cache: the
    /// verify block's two-to-four-row kernel with every row's prefix pinned at
    /// `position`, in groups of 4+4, 4+3, 4+2, 3+2, 4 or 3 rows. Each row
    /// writes its own records, as the one-row form does.
    fn attention_tree_rows_q8(
        &self,
        b: &mut CommandBatch,
        [q, k_cache, v_cache, partials]: [&MetalBuffer; 4],
        workspace: &AttentionWorkspace,
        [position, tokens, splits, split_tokens]: [u32; 4],
        rows_pipeline: usize,
    ) -> crate::Result<()> {
        need(&workspace.gathered, (4 * Q_HEADS * 256) as usize * 4)?;
        let row_records = Q_HEADS as usize * splits as usize * 258 * 4;
        let groups: &[u32] = match tokens {
            8 => &[4, 4],
            7 => &[4, 3],
            6 => &[4, 2],
            5 => &[3, 2],
            _ => &[tokens],
        };
        let mut first = 0;
        for &rows in groups {
            self.go(
                b,
                rows_pipeline,
                &[(q, first as usize * 6144 * 4), (&workspace.gathered, 0)],
                &[rows],
                &[],
                (rows * Q_HEADS) as usize,
                256,
            );
            // The tree kernels follow the gather, the three split kernels and
            // the prefill kernel.
            self.go(
                b,
                rows_pipeline + 3 + rows as usize,
                &[
                    (&workspace.gathered, 0),
                    (k_cache, 0),
                    (v_cache, 0),
                    (partials, first as usize * row_records),
                ],
                &[position, splits, split_tokens],
                &[],
                (KV_HEADS * splits) as usize,
                128,
            );
            first += rows;
        }
        Ok(())
    }

    /// Full attention of a tree verify's `tokens` query rows, whose K/V rows
    /// sit at cache rows `position + row`: row `t` attends the history
    /// below `position` and, among the pass's rows, its ancestors and itself
    /// (`tree`, a tree descriptor). The history goes through the split kernel
    /// a decode at prefix `position` uses, one row at a time, or, for a Q8
    /// cache on the tensor path, through the verify block's two-to-four-row
    /// kernel with every row's prefix pinned at `position` (groups of 4+4,
    /// 4+3, 4+2, 3+2, 4 or 3 rows; `workspace` holds the gathered queries).
    /// The tree zone goes
    /// through `bo_attn_tree_tail`, key by key in path order, as the last
    /// partial record; `bo_attn_reduce` joins them. A row's numbers depend on
    /// `position`, its query and its path's keys alone, never on the other
    /// rows or where its ancestors sit in the pass.
    pub fn attention_tree_kv(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        gate: Option<&MetalBuffer>,
        out: &MetalBuffer,
        position: u32,
        tokens: u32,
        tree: &MetalBuffer,
        partials: &MetalBuffer,
        workspace: &AttentionWorkspace,
    ) -> crate::Result<()> {
        check_rows(tokens)?;
        let end = position
            .checked_add(tokens)
            .ok_or_else(|| arg("tree attention position overflow"))?;
        let bytes = sequence_bytes(tokens, 6144)?;
        need(q, bytes)?;
        need(k_cache, end as usize * layout.key.token_bytes())?;
        need(v_cache, end as usize * layout.value.token_bytes())?;
        need(out, bytes)?;
        need(tree, TREE_DESCRIPTOR_WORDS * 4)?;
        need(partials, self.tree_partial_bytes(layout, position, tokens))?;
        if let Some(g) = gate {
            need(g, bytes)?;
            no_alias(out, &[g])?;
        }
        no_alias(out, &[q, k_cache, v_cache, partials, tree])?;
        no_alias(partials, &[q, k_cache, v_cache, tree])?;
        let splits = self.tree_splits(layout, position);
        let row_records = Q_HEADS as usize * splits as usize * 258 * 4;
        let codes = layout.codes();
        let history = self.tree_history(layout, position);
        if let History::Tensor { split_tokens, .. } = history
            && let Some(rows_pipeline) = self.tensor_rows
            && layout == KvLayout::Q8
            && tokens >= 3
        {
            self.attention_tree_rows_q8(
                b,
                [q, k_cache, v_cache, partials],
                workspace,
                [position, tokens, splits, split_tokens],
                rows_pipeline,
            )?;
        } else if position > 0 {
            // Each row writes its own records; the split past the history is
            // empty and leaves (-inf, 0, zeros) for the tree zone to replace.
            b.independent(|b| {
                for row in 0..tokens as usize {
                    let caches = [
                        (q, row * 6144 * 4),
                        (k_cache, 0),
                        (v_cache, 0),
                        (partials, row * row_records),
                    ];
                    match history {
                        History::Tensor {
                            pipeline,
                            split_tokens,
                        } => self.go(
                            b,
                            pipeline,
                            &caches,
                            &[position, splits, split_tokens],
                            &[],
                            (KV_HEADS * splits) as usize,
                            128,
                        ),
                        History::Simd { split_tokens } => self.go(
                            b,
                            10,
                            &caches,
                            &[position, splits, codes[0], codes[1], split_tokens],
                            &[],
                            (Q_HEADS / SPLIT_HEADS * splits) as usize,
                            32,
                        ),
                    }
                }
                crate::Result::Ok(())
            })?;
        }
        self.go(
            b,
            ATTENTION_TAIL,
            &[(q, 0), (k_cache, 0), (v_cache, 0), (partials, 0), (tree, 0)],
            &[position, splits, codes[0], codes[1]],
            &[],
            (Q_HEADS * tokens) as usize,
            32,
        );
        // Quantized outputs rotate back inside the reduction before gating.
        self.go(
            b,
            11,
            &[(partials, 0), (gate.unwrap_or(q), 0), (out, 0)],
            &[
                splits,
                u32::from(gate.is_some()),
                u32::from(!layout.is_f16()),
            ],
            &[],
            (Q_HEADS * tokens) as usize,
            256,
        );
        Ok(())
    }

    /// Copy `rows` rows of `width` floats from `source` (rows
    /// `source_stride` floats apart from its byte offset) to rows
    /// `(first + row) % capacity` of `destination` (`destination_stride`
    /// floats apart from its byte offset). Offsets, widths and strides are
    /// whole 16-byte vectors.
    pub fn copy_rows(
        &self,
        b: &mut CommandBatch,
        source: (&MetalBuffer, usize),
        destination: (&MetalBuffer, usize),
        width: usize,
        rows: usize,
        strides: [usize; 2],
        first: usize,
        capacity: usize,
    ) -> crate::Result<()> {
        let [source_stride, destination_stride] = strides;
        if rows == 0 {
            return Ok(());
        }
        if capacity == 0
            || first >= capacity
            || rows > capacity
            || !width.is_multiple_of(4)
            || !source_stride.is_multiple_of(4)
            || !destination_stride.is_multiple_of(4)
            || !source.1.is_multiple_of(16)
            || !destination.1.is_multiple_of(16)
        {
            return Err(arg("invalid row copy layout"));
        }
        need(
            source.0,
            source.1 + ((rows - 1) * source_stride + width) * 4,
        )?;
        need(
            destination.0,
            destination.1 + ((capacity - 1) * destination_stride + width) * 4,
        )?;
        no_alias(destination.0, &[source.0])?;
        let narrow = |value: usize| u32::try_from(value).map_err(|_| arg("row copy overflow"));
        self.go(
            b,
            COPY_ROWS,
            &[source, destination],
            &[
                narrow(width / 4)?,
                narrow(source_stride / 4)?,
                narrow(destination_stride / 4)?,
                narrow(first)?,
                narrow(capacity)?,
            ],
            &[],
            rows,
            256,
        );
        Ok(())
    }

    /// Move rows `path[i]` to rows `i` for `i` in `1..count`, in place, in
    /// each of `parts` (a buffer, the byte offset of its row 0 and its bytes
    /// per row, a multiple of 4). `path` is `count` ascending `i32` rows from
    /// a byte offset of a buffer, starting with 0.
    pub fn gather_rows(
        &self,
        b: &mut CommandBatch,
        parts: &[(&MetalBuffer, usize, usize)],
        path: (&MetalBuffer, usize),
        count: usize,
    ) -> crate::Result<()> {
        if parts.is_empty() || parts.len() > 3 || !(1..=TREE_ROWS as usize + 1).contains(&count) {
            return Err(arg("a row gather moves 1..=9 rows of 1..=3 buffers"));
        }
        need(path.0, path.1 + count * 4)?;
        let mut bound = [(parts[0].0, parts[0].1); 3];
        let mut words = [0u32; 3];
        for (index, &(buffer, offset, row_bytes)) in parts.iter().enumerate() {
            if !row_bytes.is_multiple_of(4) || !offset.is_multiple_of(4) {
                return Err(arg("gathered rows must be whole words"));
            }
            need(buffer, offset + count * row_bytes)?;
            no_alias(buffer, &[path.0])?;
            bound[index] = (buffer, offset);
            words[index] = u32::try_from(row_bytes / 4).map_err(|_| arg("row gather overflow"))?;
        }
        if count == 1 {
            return Ok(());
        }
        let total = words.iter().map(|&value| value as usize).sum::<usize>();
        self.go(
            b,
            GATHER_ROWS,
            &[bound[0], bound[1], bound[2], path],
            &[words[0], words[1], words[2], count as u32],
            &[],
            total.div_ceil(256),
            256,
        );
        Ok(())
    }
}
