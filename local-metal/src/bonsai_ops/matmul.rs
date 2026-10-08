use super::{
    BF16_TILE_COLUMNS, BF16_TILE_MIN_TOKENS, BF16_TILE_ROWS, BF16_TILE_TOKENS, Bf16Matrix,
    BonsaiOps, CommandBatch, MetalBuffer, arg, matrix_bytes, need, no_alias, sequence_bytes,
};

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

    pub fn bf16_matmul(
        &self,
        b: &mut CommandBatch,
        m: Bf16Matrix<'_>,
        x: &MetalBuffer,
        y: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if m.rows == 0 || m.columns == 0 || !m.offset.is_multiple_of(2) {
            return Err(arg("invalid BF16 matrix"));
        }
        let bytes = matrix_bytes(m.rows, m.columns, 2)?;
        let end = m
            .offset
            .checked_add(bytes)
            .ok_or_else(|| arg("BF16 offset overflow"))?;
        need(m.buffer, end)?;
        need(x, sequence_bytes(tokens, m.columns)?)?;
        need(y, sequence_bytes(tokens, m.rows)?)?;
        no_alias(y, &[m.buffer, x])?;
        if tokens >= BF16_TILE_MIN_TOKENS && m.columns.is_multiple_of(BF16_TILE_COLUMNS) {
            // Token-tiled GEMM: the matrix is streamed once per 32 tokens
            // instead of once per token.
            let groups = (m.rows as usize).div_ceil(BF16_TILE_ROWS as usize)
                * (tokens as usize).div_ceil(BF16_TILE_TOKENS as usize);
            self.go(
                b,
                16,
                &[(m.buffer, m.offset), (x, 0), (y, 0)],
                &[m.rows, m.columns, tokens],
                &[],
                groups,
                128,
            );
            return Ok(());
        }
        self.go(
            b,
            3,
            &[(m.buffer, m.offset), (x, 0), (y, 0)],
            &[m.rows, m.columns],
            &[],
            m.rows as usize * tokens as usize,
            32,
        );
        Ok(())
    }
}
