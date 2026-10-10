use super::{
    AttentionKernel, AttentionWorkspace, BonsaiOps, CommandBatch, KV_HEADS, KvLayout,
    MAX_PREFILL_TOKENS, MetalBuffer, Q_HEADS, ROW_BLOCK_MIN_PREFIX, ROW_BLOCK_TOKENS, SPLIT,
    SPLIT_HEADS, SPLIT_TENSOR_LONG_MIN_PREFIX, SPLIT_TENSOR_MIN_PREFIX, SPLIT_TENSOR_ROWS,
    SPLIT_TENSOR_SHORT, arg, need, no_alias, sequence_bytes, tensor_split_tokens,
};

impl BonsaiOps {
    pub fn attention(
        &self,
        b: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        gate: Option<&MetalBuffer>,
        out: &MetalBuffer,
        prefix: u32,
        workspace: &AttentionWorkspace,
    ) -> crate::Result<()> {
        self.attention_row(b, q, k_cache, v_cache, gate, out, prefix, workspace, 0)
    }

    /// F16-cache form of [`Self::attention_row_kv`].
    pub fn attention_row(
        &self,
        b: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        gate: Option<&MetalBuffer>,
        out: &MetalBuffer,
        prefix: u32,
        workspace: &AttentionWorkspace,
        row: u32,
    ) -> crate::Result<()> {
        self.attention_row_kv(
            KvLayout::F16,
            b,
            q,
            k_cache,
            v_cache,
            gate,
            out,
            prefix,
            workspace,
            row,
        )
    }

    /// Attend one query row, reusing a single linear-size workspace for all rows.
    /// The caller supplies that row's causal prefix, not the full token block.
    ///
    /// On a tensor build, F16 caches use `bo_attn_split_tensor` and quantized
    /// ones `bo_attn_split_tensor_<layout>` (the six GQA heads as one Q tile per
    /// split); the SIMD build uses `bo_attn_split` with `SPLIT_HEADS` heads per
    /// SIMD group. All feed the same partial records to `bo_attn_reduce`.
    pub fn attention_row_kv(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        gate: Option<&MetalBuffer>,
        out: &MetalBuffer,
        prefix: u32,
        workspace: &AttentionWorkspace,
        row: u32,
    ) -> crate::Result<()> {
        if prefix == 0 || prefix > workspace.max_context {
            return Err(arg("attention prefix exceeds workspace"));
        }
        let rows = row
            .checked_add(1)
            .ok_or_else(|| arg("attention row overflow"))?;
        let bytes = sequence_bytes(rows, 6144)?;
        let offset = bytes - 6144 * 4;
        let codes = layout.codes();
        need(q, bytes)?;
        need(k_cache, prefix as usize * layout.key.token_bytes())?;
        need(v_cache, prefix as usize * layout.value.token_bytes())?;
        need(out, bytes)?;
        if let Some(g) = gate {
            need(g, bytes)?;
            no_alias(out, &[g])?;
        }
        no_alias(out, &[q, k_cache, v_cache, &workspace.partials])?;
        let caches = [
            (q, offset),
            (k_cache, 0),
            (v_cache, 0),
            (&workspace.partials, 0),
        ];
        let tensor_split = if layout.is_f16() {
            Some(18)
        } else {
            self.quantized_tensor(layout)
        };
        let splits = if let Some(pipeline) = tensor_split
            && self.attention_kernel != AttentionKernel::SimdF32
            && prefix >= SPLIT_TENSOR_MIN_PREFIX
        {
            let split_tokens = if prefix >= SPLIT_TENSOR_LONG_MIN_PREFIX {
                tensor_split_tokens(prefix)
            } else {
                SPLIT_TENSOR_SHORT
            };
            let splits = prefix.div_ceil(split_tokens);
            need(
                &workspace.partials,
                splits as usize * Q_HEADS as usize * 258 * 4,
            )?;
            self.go(
                b,
                pipeline,
                &caches,
                &[prefix, splits, split_tokens],
                &[],
                (KV_HEADS * splits) as usize,
                128,
            );
            splits
        } else {
            let splits = prefix.div_ceil(SPLIT);
            need(
                &workspace.partials,
                splits as usize * Q_HEADS as usize * 258 * 4,
            )?;
            self.go(
                b,
                10,
                &caches,
                &[prefix, splits, codes[0], codes[1]],
                &[],
                (24 / SPLIT_HEADS * splits) as usize,
                32,
            );
            splits
        };
        // Quantized outputs rotate back inside the reduction before gating.
        let rotated = !layout.is_f16();
        let fallback = gate.unwrap_or(q);
        self.go(
            b,
            11,
            &[(&workspace.partials, 0), (fallback, offset), (out, offset)],
            &[splits, u32::from(gate.is_some()), u32::from(rotated)],
            &[],
            24,
            256,
        );
        Ok(())
    }

    /// Pack two to four Q8 query rows so their GQA tiles share each KV load.
    /// The caller validates buffers and keeps every row on the same split size.
    fn attention_rows_q8(
        &self,
        b: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        gate: Option<&MetalBuffer>,
        out: &MetalBuffer,
        first_prefix: u32,
        rows: u32,
        row: u32,
        split_tokens: u32,
        pipeline: usize,
        workspace: &AttentionWorkspace,
    ) -> crate::Result<()> {
        let offset = row as usize * 6144 * 4;
        let splits = (first_prefix + rows - 1).div_ceil(split_tokens);
        need(
            &workspace.partials,
            (rows * Q_HEADS * splits * 258) as usize * 4,
        )?;
        self.go(
            b,
            pipeline,
            &[(q, offset), (&workspace.gathered, 0)],
            &[rows],
            &[],
            (rows * Q_HEADS) as usize,
            256,
        );
        self.go(
            b,
            pipeline + rows as usize - 1,
            &[
                (&workspace.gathered, 0),
                (k_cache, 0),
                (v_cache, 0),
                (&workspace.partials, 0),
            ],
            &[first_prefix, splits, split_tokens],
            &[],
            (KV_HEADS * splits) as usize,
            128,
        );
        self.go(
            b,
            11,
            &[
                (&workspace.partials, 0),
                (gate.unwrap_or(q), offset),
                (out, offset),
            ],
            &[splits, u32::from(gate.is_some()), 1],
            &[],
            (rows * Q_HEADS) as usize,
            256,
        );
        Ok(())
    }

    /// Rotate `rows` attention output rows starting at byte `offset` back from
    /// a quantized cache's Hadamard basis, then apply `gate` if given.
    fn unrotate(
        &self,
        b: &mut CommandBatch,
        out: &MetalBuffer,
        offset: usize,
        gate: Option<&MetalBuffer>,
        rows: u32,
    ) {
        self.go(
            b,
            20,
            &[(out, offset), (gate.unwrap_or(out), offset)],
            &[u32::from(gate.is_some())],
            &[],
            24 * rows as usize,
            256,
        );
    }

    /// F16-cache form of [`Self::attention_block_kv`].
    pub fn attention_block(
        &self,
        b: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        gate: Option<&MetalBuffer>,
        out: &MetalBuffer,
        position: u32,
        tokens: u32,
        workspace: &AttentionWorkspace,
    ) -> crate::Result<()> {
        self.attention_block_kv(
            KvLayout::F16,
            b,
            q,
            k_cache,
            v_cache,
            gate,
            out,
            position,
            tokens,
            workspace,
        )
    }

    /// Causal full-attention prefill for a contiguous query block. Query row
    /// `r` attends through cache position `position + r`, inclusive.
    ///
    /// On a tensor build, F16 caches use `bo_attn_tensor` and quantized ones
    /// `bo_attn_tensor_<layout>`; the SIMD build takes the SIMD block kernel,
    /// which dequantizes in registers.
    #[allow(clippy::too_many_lines)]
    pub fn attention_block_kv(
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
        workspace: &AttentionWorkspace,
    ) -> crate::Result<()> {
        let end = position
            .checked_add(tokens)
            .ok_or_else(|| arg("attention block position overflow"))?;
        if !(1..=MAX_PREFILL_TOKENS).contains(&tokens) || end > workspace.max_context || end == 0 {
            return Err(arg("invalid attention block extent"));
        }
        let bytes = sequence_bytes(tokens, 6144)?;
        need(q, bytes)?;
        need(k_cache, end as usize * layout.key.token_bytes())?;
        need(v_cache, end as usize * layout.value.token_bytes())?;
        need(out, bytes)?;
        if let Some(g) = gate {
            need(g, bytes)?;
            no_alias(out, &[g])?;
        }
        no_alias(out, &[q, k_cache, v_cache, &workspace.partials])?;
        if tokens <= ROW_BLOCK_TOKENS && end > ROW_BLOCK_MIN_PREFIX {
            let mut row = 0;
            while row < tokens {
                let first_prefix = position + row + 1;
                let remaining = tokens - row;
                // Avoid leaving one row after a four-row group.
                let rows = if remaining == 5 {
                    3
                } else {
                    remaining.min(SPLIT_TENSOR_ROWS)
                };
                let split_size = |prefix| {
                    if prefix >= SPLIT_TENSOR_LONG_MIN_PREFIX {
                        tensor_split_tokens(prefix)
                    } else {
                        SPLIT_TENSOR_SHORT
                    }
                };
                if let Some(pipeline) = self.tensor_rows
                    && layout == KvLayout::Q8
                    && rows >= 2
                    && first_prefix >= SPLIT_TENSOR_MIN_PREFIX
                    && split_size(first_prefix) == split_size(first_prefix + rows - 1)
                {
                    self.attention_rows_q8(
                        b,
                        q,
                        k_cache,
                        v_cache,
                        gate,
                        out,
                        first_prefix,
                        rows,
                        row,
                        split_size(first_prefix),
                        pipeline,
                        workspace,
                    )?;
                    row += rows;
                    continue;
                }
                self.attention_row_kv(
                    layout,
                    b,
                    q,
                    k_cache,
                    v_cache,
                    gate,
                    out,
                    position + row + 1,
                    workspace,
                    row,
                )?;
                row += 1;
            }
            return Ok(());
        }
        let fallback = gate.unwrap_or(q);
        let bufs = [(q, 0), (k_cache, 0), (v_cache, 0), (fallback, 0), (out, 0)];
        let groups = 24 * tokens.div_ceil(8) as usize;
        // Rotated (quantized) caches gate after rotating back; see `unrotate`.
        let rotated = !layout.is_f16();
        let gated = u32::from(gate.is_some() && !rotated);
        let codes = layout.codes();
        if layout.is_f16() {
            // `bo_attn_block` reads the format codes at slots 8 and 9 and
            // dispatches on them: any non-F16 code selects the dequantizing
            // instantiation, which reads F16 bytes as Q8. The tensor kernel
            // takes `half*` directly and never dispatches, which is why this
            // only showed up on the SIMD path.
            self.go(
                b,
                13,
                &bufs,
                &[position, tokens, gated, codes[0], codes[1]],
                &[],
                groups,
                self.attention_kernel.threads(),
            );
        } else if let Some(split) = self.quantized_tensor(layout) {
            self.go(
                b,
                split + 1,
                &bufs,
                &[position, tokens, gated],
                &[],
                groups,
                self.attention_kernel.threads(),
            );
        } else {
            self.go(
                b,
                17,
                &bufs,
                &[position, tokens, gated, codes[0], codes[1]],
                &[],
                groups,
                256,
            );
        }
        if rotated {
            self.unrotate(b, out, 0, gate, tokens);
        }
        Ok(())
    }
}
