use super::{
    Bf16Matrix, BonsaiOps, CommandBatch, MetalBuffer, arg, matrix_bytes, need, no_alias,
    sequence_bytes,
};

/// Activation rows per SIMD group of `bo_bf16_mv_tokens`; must match the shader.
const BF16_GROUP_TOKENS: usize = 4;

impl BonsaiOps {
    pub fn bf16_matvec(
        &self,
        b: &mut CommandBatch,
        m: Bf16Matrix<'_>,
        x: &MetalBuffer,
        y: &MetalBuffer,
    ) -> crate::Result<()> {
        self.bf16_matmul(b, m, x, y, 1)
    }

    /// `[tokens, rows]` = `[tokens, columns]` x BF16 `[rows, columns]`ᵀ.
    ///
    /// Every row's output is bitwise the single-row matvec's, whatever the
    /// row count: blocks share each weight load across four rows but keep
    /// the matvec's per-row summation order.
    pub fn bf16_matmul(
        &self,
        b: &mut CommandBatch,
        m: Bf16Matrix<'_>,
        x: &MetalBuffer,
        y: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.bf16_matmul_group(b, &[(m, y)], x, tokens)
    }

    /// [`Self::bf16_matmul`] of two equal-shape matrices over the same rows in
    /// one dispatch (a recurrent layer's alpha and beta projections); each
    /// output is bitwise what `bf16_matmul` writes.
    pub fn bf16_matmul_pair(
        &self,
        b: &mut CommandBatch,
        matrices: [(Bf16Matrix<'_>, &MetalBuffer); 2],
        x: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let [(first, _), (second, _)] = matrices;
        if (first.rows, first.columns) != (second.rows, second.columns) {
            return Err(arg("paired BF16 matrices differ in shape"));
        }
        no_alias(matrices[0].1, &[matrices[1].1])?;
        self.bf16_matmul_group(b, &matrices, x, tokens)
    }

    fn bf16_matmul_group(
        &self,
        b: &mut CommandBatch,
        matrices: &[(Bf16Matrix<'_>, &MetalBuffer)],
        x: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        for &(m, y) in matrices {
            if m.rows == 0 || m.columns == 0 || !m.offset.is_multiple_of(2) {
                return Err(arg("invalid BF16 matrix"));
            }
            let end = m
                .offset
                .checked_add(matrix_bytes(m.rows, m.columns, 2)?)
                .ok_or_else(|| arg("BF16 offset overflow"))?;
            need(m.buffer, end)?;
            need(x, sequence_bytes(tokens, m.columns)?)?;
            need(y, sequence_bytes(tokens, m.rows)?)?;
            no_alias(y, &[x])?;
            for &(other, _) in matrices {
                no_alias(y, &[other.buffer])?;
            }
        }
        let (first, y0) = matrices[0];
        let (second, y1) = matrices[matrices.len() - 1];
        if tokens == 1 && matrices.len() == 1 {
            self.go(
                b,
                3,
                &[(first.buffer, first.offset), (x, 0), (y0, 0)],
                &[first.rows, first.columns],
                &[],
                first.rows as usize,
                32,
            );
            return Ok(());
        }
        let groups =
            matrices.len() * first.rows as usize * (tokens as usize).div_ceil(BF16_GROUP_TOKENS);
        self.go(
            b,
            16,
            &[
                (first.buffer, first.offset),
                (second.buffer, second.offset),
                (x, 0),
                (y0, 0),
                (y1, 0),
            ],
            &[first.rows, first.columns, tokens, matrices.len() as u32],
            &[],
            groups,
            32,
        );
        Ok(())
    }
}
