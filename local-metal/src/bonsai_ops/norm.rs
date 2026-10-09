use super::{
    BonsaiOps, CommandBatch, MetalBuffer, RmsNormParams, arg, matrix_bytes, need, no_alias, same,
    trio,
};
use crate::bonsai::{HADAMARD_BLOCK_ELEMENTS, SignedHadamard};

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
    /// `sum = hidden + branch` (as [`Self::residual_add`]), then RMS-normalize
    /// `sum` by `weights` into `normalized` and write its forward `rotation`
    /// to `output`, all in one dispatch over `tokens` rows of the rotation's
    /// width. Values are bitwise those of the residual add followed by
    /// `BonsaiKernels::normalize_transform`. Every threadgroup reads whole
    /// rows of `hidden` and `branch`, so no output may alias an input.
    pub fn residual_normalize_transform(
        &self,
        b: &mut CommandBatch,
        rotation: &SignedHadamard,
        hidden: &MetalBuffer,
        branch: &MetalBuffer,
        weights: &MetalBuffer,
        outputs: [&MetalBuffer; 3],
        tokens: u32,
        epsilon: f32,
    ) -> crate::Result<()> {
        let [sum, normalized, output] = outputs;
        let elements = rotation
            .columns
            .checked_mul(tokens)
            .filter(|&count| count != 0)
            .ok_or_else(|| arg("residual normalize-rotate shape overflow/empty"))?;
        let bytes = elements as usize * 4;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(arg("invalid RMSNorm epsilon"));
        }
        for v in [hidden, branch, sum, normalized, output] {
            need(v, bytes)?;
        }
        need(weights, rotation.columns as usize * 4)?;
        let inputs = [hidden, branch, weights, &rotation.signs];
        no_alias(sum, &inputs)?;
        no_alias(normalized, &inputs)?;
        no_alias(output, &inputs)?;
        if same(sum, normalized) || same(sum, output) || same(normalized, output) {
            return Err(arg("residual normalize-rotate outputs alias each other"));
        }
        self.go(
            b,
            26,
            &[
                (hidden, 0),
                (branch, 0),
                (weights, 0),
                (&rotation.signs, 0),
                (sum, 0),
                (normalized, 0),
                (output, 0),
            ],
            &[rotation.columns / HADAMARD_BLOCK_ELEMENTS as u32],
            &[epsilon],
            elements as usize / HADAMARD_BLOCK_ELEMENTS,
            128,
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
