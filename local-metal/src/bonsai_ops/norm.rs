use super::{
    BonsaiOps, CommandBatch, MetalBuffer, RmsNormParams, arg, matrix_bytes, need, no_alias, same,
    trio,
};

impl BonsaiOps {
    pub fn rms_norm(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        output: &MetalBuffer,
        params: RmsNormParams,
    ) -> crate::Result<()> {
        if params.dimension == 0
            || params.rows == 0
            || params.stride < params.dimension
            || !params.epsilon.is_finite()
            || params.epsilon <= 0.0
            || !params.weight_offset.is_multiple_of(4)
        {
            return Err(arg("invalid RMSNorm parameters"));
        }
        let bytes = matrix_bytes(params.rows, params.stride, 4)?;
        let weight_end = params
            .weight_offset
            .checked_add(params.dimension as usize * 4)
            .ok_or_else(|| arg("RMSNorm weight offset overflow"))?;
        need(input, bytes)?;
        need(weights, weight_end)?;
        need(output, bytes)?;
        no_alias(output, &[weights])?;
        self.go(
            batch,
            0,
            &[(input, 0), (weights, params.weight_offset), (output, 0)],
            &[params.dimension, params.rows, params.stride],
            &[params.epsilon],
            params.rows as usize,
            128,
        );
        Ok(())
    }
    pub fn residual_add(
        &self,
        b: &mut CommandBatch,
        x: &MetalBuffer,
        residual: &MetalBuffer,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        if count == 0 {
            return Err(arg("add count is zero"));
        }
        for v in [x, residual, out] {
            need(v, count as usize * 4)?;
        }
        if same(out, x) && !same(out, residual) {
            return Err(arg("only exact in-place residual addition is supported"));
        }
        self.go(
            b,
            1,
            &[(x, 0), (residual, 0), (out, 0)],
            &[count],
            &[],
            (count as usize).div_ceil(256),
            256,
        );
        Ok(())
    }
    pub fn swiglu(
        &self,
        b: &mut CommandBatch,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        trio(gate, up, out, count)?;
        self.go(
            b,
            2,
            &[(gate, 0), (up, 0), (out, 0)],
            &[count],
            &[],
            (count as usize).div_ceil(256),
            256,
        );
        Ok(())
    }
    pub fn sigmoid_mul(
        &self,
        b: &mut CommandBatch,
        x: &MetalBuffer,
        gate: &MetalBuffer,
        out: &MetalBuffer,
        count: u32,
    ) -> crate::Result<()> {
        trio(x, gate, out, count)?;
        self.go(
            b,
            12,
            &[(x, 0), (gate, 0), (out, 0)],
            &[count],
            &[],
            (count as usize).div_ceil(256),
            256,
        );
        Ok(())
    }
}
