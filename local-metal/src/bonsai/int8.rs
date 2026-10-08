//! Per-row int8 projections: `y[t, r] = scale[r] · Σ_c int8[r, c] · x[t, c]`.
//!
//! These multiply the same F32 activations the `PTQ1_0` kernels read (already
//! forward-rotated where the weights live in a rotated basis), so int8 and
//! ternary matrices can be mixed freely within one model. The weights stay int8
//! in memory; nothing expands them.

use core::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder as _, MTLComputePipelineState};

use super::{BonsaiKernels, bind, dispatch, set_u32};
use crate::Error;
use crate::batch::CommandBatch;
use crate::buffer::MetalBuffer;

/// Column counts must be a multiple of this: the vector kernels give each lane
/// 16 consecutive columns and the wide kernel walks 64-column chunks.
pub const INT8_COLUMN_MULTIPLE: u32 = 64;

/// Largest activation-row count one int8 dispatch handles; larger blocks are
/// split into even chunks of at most this many rows, each reading the matrix
/// once.
pub const INT8_CHUNK_TOKENS: u32 = 8;

/// Chunks of up to this many rows use the vector kernels (one row per
/// activation row in registers); larger ones the wide simdgroup-matrix kernel.
pub const INT8_VECTOR_TOKENS: u32 = 2;

/// Output rows per threadgroup of the wide kernel.
const WIDE_ROWS: usize = 32;

/// Checked row-major int8 matrix with one F32 scale per row.
///
/// Both buffers must stay unmodified until GPU completion.
#[derive(Clone, Copy)]
pub struct Int8Matrix<'a> {
    pub(crate) weights: &'a MetalBuffer,
    pub(crate) scales: &'a MetalBuffer,
    pub(crate) rows: u32,
    pub(crate) columns: u32,
}

impl<'a> Int8Matrix<'a> {
    /// `weights` holds `rows · columns` int8 values from offset 0, row-major;
    /// `scales` holds `rows` F32 values.
    pub fn new(
        weights: &'a MetalBuffer,
        scales: &'a MetalBuffer,
        rows: u32,
        columns: u32,
    ) -> crate::Result<Self> {
        let bytes = (rows as usize).checked_mul(columns as usize);
        if rows == 0
            || columns == 0
            || !columns.is_multiple_of(INT8_COLUMN_MULTIPLE)
            || bytes.is_none_or(|bytes| bytes > weights.length())
            || scales.length() < rows as usize * size_of::<f32>()
            || std::ptr::eq(weights.raw(), scales.raw())
        {
            return Err(Error::InvalidArgument(format!(
                "int8 matrix is empty, its columns are not a multiple of \
                 {INT8_COLUMN_MULTIPLE}, or it exceeds its buffers"
            )));
        }
        Ok(Self {
            weights,
            scales,
            rows,
            columns,
        })
    }

    #[must_use]
    pub const fn rows(self) -> u32 {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> u32 {
        self.columns
    }

    fn writes(self, output: &MetalBuffer) -> bool {
        std::ptr::eq(self.weights.raw(), output.raw())
            || std::ptr::eq(self.scales.raw(), output.raw())
    }
}

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(super) struct Int8Kernels {
    /// Vector kernels for 1..=`INT8_VECTOR_TOKENS` activation rows.
    vector: Vec<Pipeline>,
    wide: Pipeline,
    concat: Pipeline,
    swiglu: Pipeline,
}

impl Int8Kernels {
    pub(super) fn new(
        pipeline: impl Fn(&str, usize) -> crate::Result<Pipeline>,
    ) -> crate::Result<Self> {
        Ok(Self {
            vector: vec![
                pipeline("bonsai_int8_matvec", 32)?,
                pipeline("bonsai_int8_matmul_2", 32)?,
            ],
            wide: pipeline("bonsai_int8_matmul_wide", 128)?,
            concat: pipeline("bonsai_int8_matvec_concat", 32)?,
            swiglu: pipeline("bonsai_int8_matvec_swiglu", 32)?,
        })
    }
}

/// Weight rows one SIMD group reduces in the vector kernels.
const VECTOR_ROWS: usize = 4;

fn extent(tokens: u32, width: u32) -> crate::Result<usize> {
    (tokens as usize)
        .checked_mul(width as usize)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| Error::InvalidArgument("int8 matmul extent overflow".into()))
}

impl BonsaiKernels {
    /// Single-row int8 projection; see [`Self::int8_matmul`].
    pub fn int8_matvec(
        &self,
        batch: &mut CommandBatch,
        matrix: Int8Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        self.int8_matmul(batch, matrix, input, output, 1)
    }

    /// Int8 projection of contiguous `[tokens, columns]` F32 rows into
    /// `[tokens, rows]` F32 rows. Blocks of up to [`INT8_CHUNK_TOKENS`] rows
    /// take one dispatch that reads the matrix once: the vector kernels for up
    /// to [`INT8_VECTOR_TOKENS`] rows, the wide simdgroup-matrix kernel above.
    /// Larger blocks take even chunks of at most [`INT8_CHUNK_TOKENS`] rows.
    #[allow(unsafe_code)]
    pub fn int8_matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: Int8Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0
            || input.length() < extent(tokens, matrix.columns)?
            || output.length() < extent(tokens, matrix.rows)?
            || std::ptr::eq(input.raw(), output.raw())
            || matrix.writes(output)
        {
            return Err(Error::InvalidArgument(
                "int8 matmul buffers are empty, too short, or aliased".into(),
            ));
        }
        // Even chunks: 9 rows run as 5 + 4, never 8 + 1.
        let chunks = tokens.div_ceil(INT8_CHUNK_TOKENS);
        let mut start = 0;
        for chunk in 0..chunks {
            let end = (tokens * (chunk + 1)).div_ceil(chunks);
            let count = end - start;
            let wide = count > INT8_VECTOR_TOKENS;
            let encoder = batch.encoder();
            encoder.setComputePipelineState(if wide {
                &self.int8.wide
            } else {
                &self.int8.vector[count as usize - 1]
            });
            unsafe {
                bind(encoder, matrix.weights, 0, 0);
                bind(encoder, matrix.scales, 0, 1);
                bind(encoder, input, extent(start, matrix.columns)?, 2);
                bind(encoder, output, extent(start, matrix.rows)?, 3);
                set_u32(encoder, &matrix.rows, 4);
                set_u32(encoder, &matrix.columns, 5);
                if wide {
                    set_u32(encoder, &count, 6);
                }
            }
            if wide {
                dispatch(encoder, (matrix.rows as usize).div_ceil(WIDE_ROWS), 128);
            } else {
                dispatch(encoder, (matrix.rows as usize).div_ceil(VECTOR_ROWS), 32);
            }
            batch.record_dispatch();
            start = end;
        }
        Ok(())
    }

    /// Single-row projections of one input by one to three int8 matrices with
    /// the same column count, in one dispatch. Outputs must be distinct from
    /// the input and every weight buffer.
    #[allow(unsafe_code)]
    pub fn int8_matvec_concat(
        &self,
        batch: &mut CommandBatch,
        projections: &[(Int8Matrix<'_>, &MetalBuffer)],
        input: &MetalBuffer,
    ) -> crate::Result<()> {
        let Some(&(first, first_output)) = projections.first() else {
            return Err(Error::InvalidArgument(
                "int8 concatenated matvec needs one to three matrices".into(),
            ));
        };
        if projections.len() > 3
            || input.length() < first.columns as usize * size_of::<f32>()
            || projections.iter().any(|(matrix, output)| {
                matrix.columns != first.columns
                    || output.length() < matrix.rows as usize * size_of::<f32>()
                    || std::ptr::eq(input.raw(), output.raw())
                    || projections.iter().any(|(other, _)| other.writes(output))
            })
        {
            return Err(Error::InvalidArgument(
                "int8 concatenated matvec shapes differ, buffers are short, or aliased".into(),
            ));
        }
        let mut rows = [0_u32; 3];
        let mut groups = 0;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.int8.concat);
        unsafe {
            // Unused segments have zero rows but still bind valid buffers.
            for (segment, rows) in rows.iter_mut().enumerate() {
                let (matrix, output) = match projections.get(segment) {
                    Some(&projection) => {
                        *rows = projection.0.rows;
                        groups += (projection.0.rows as usize).div_ceil(4);
                        projection
                    }
                    None => (first, first_output),
                };
                bind(encoder, matrix.weights, 0, 3 * segment);
                bind(encoder, matrix.scales, 0, 3 * segment + 1);
                bind(encoder, output, 0, 3 * segment + 2);
            }
            bind(encoder, input, 0, 9);
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(rows.as_ptr().cast_mut().cast::<c_void>()),
                size_of_val(&rows),
                10,
            );
            set_u32(encoder, &first.columns, 11);
        }
        dispatch(encoder, groups, 32);
        batch.record_dispatch();
        Ok(())
    }

    /// Single-row `silu(gate · input) · (up · input)` for two same-shape int8
    /// matrices in one dispatch; neither projection is stored.
    #[allow(unsafe_code)]
    pub fn int8_matvec_swiglu(
        &self,
        batch: &mut CommandBatch,
        gate: Int8Matrix<'_>,
        up: Int8Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        if gate.rows != up.rows
            || gate.columns != up.columns
            || input.length() < gate.columns as usize * size_of::<f32>()
            || output.length() < gate.rows as usize * size_of::<f32>()
            || std::ptr::eq(input.raw(), output.raw())
            || gate.writes(output)
            || up.writes(output)
        {
            return Err(Error::InvalidArgument(
                "int8 SwiGLU shapes differ, buffers are too short, or output aliases".into(),
            ));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.int8.swiglu);
        unsafe {
            bind(encoder, gate.weights, 0, 0);
            bind(encoder, gate.scales, 0, 1);
            bind(encoder, up.weights, 0, 2);
            bind(encoder, up.scales, 0, 3);
            bind(encoder, input, 0, 4);
            bind(encoder, output, 0, 5);
            set_u32(encoder, &gate.rows, 6);
            set_u32(encoder, &gate.columns, 7);
        }
        dispatch(encoder, (gate.rows as usize).div_ceil(2), 32);
        batch.record_dispatch();
        Ok(())
    }
}
