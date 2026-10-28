//! Group-wise 4-bit projections: `y[t, r] = Σ_c (scale[r, g] · code[r, c] +
//! bias[r, g]) · x[t, c]`, `g = c / 64`, codes in [0, 15].
//!
//! Codes are row-major, two per byte (column `2i` in the low nibble of byte
//! `i`), and each group of [`INT4_GROUP`] consecutive columns of a row has an
//! F16 `(scale, bias)` pair. They multiply the same F32 activations the int8
//! kernels read, and the weights stay packed in memory.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder as _, MTLComputePipelineState};

use super::{BonsaiKernels, bind, dispatch, set_u32};
use crate::Error;
use crate::batch::CommandBatch;
use crate::buffer::MetalBuffer;

/// Column counts must be a multiple of this: the wide kernel walks 128-column
/// chunks.
pub const INT4_COLUMN_MULTIPLE: u32 = 128;

/// Columns per `(scale, bias)` pair.
pub const INT4_GROUP: u32 = 64;

/// Output rows and threads per threadgroup of the wide kernel.
const WIDE_ROWS: usize = 8;
const WIDE_THREADS: usize = 256;
/// Matrices of at least this many rows take the wide kernel with two 8-row
/// tiles per threadgroup, which share each activation load: at eight
/// activation rows on an M4 Pro, 17408x5120 264 us against 291 with one tile,
/// 5120x17408 269 against 326, 5120x25600 395 against 493, 4096x5120 67
/// against 72; 1024 rows (64 threadgroups) are slower paired, 24 us against
/// 22.
const PAIRED_TILE_ROWS: u32 = 4096;
/// Weight rows one SIMD group reduces in the vector kernels.
const VECTOR_ROWS: usize = 4;
/// Largest activation-row count of one dispatch; larger blocks are split into
/// even chunks of at most this many rows, each reading the matrix once.
const CHUNK_TOKENS: u32 = 8;

/// Checked group-wise 4-bit matrix.
///
/// Both buffers must stay unmodified until GPU completion.
#[derive(Clone, Copy)]
pub struct Int4Matrix<'a> {
    pub(crate) codes: &'a MetalBuffer,
    pub(crate) groups: &'a MetalBuffer,
    pub(crate) rows: u32,
    pub(crate) columns: u32,
}

impl<'a> Int4Matrix<'a> {
    /// `codes` holds `rows · columns / 2` bytes from offset 0, row-major;
    /// `groups` holds `rows · columns / 64` F16 `(scale, bias)` pairs,
    /// row-major.
    pub fn new(
        codes: &'a MetalBuffer,
        groups: &'a MetalBuffer,
        rows: u32,
        columns: u32,
    ) -> crate::Result<Self> {
        let elements = (rows as usize).checked_mul(columns as usize);
        if rows == 0
            || columns == 0
            || !columns.is_multiple_of(INT4_COLUMN_MULTIPLE)
            || elements.is_none_or(|elements| {
                elements / 2 > codes.length()
                    || elements / INT4_GROUP as usize * 4 > groups.length()
            })
            || std::ptr::eq(codes.raw(), groups.raw())
        {
            return Err(Error::InvalidArgument(format!(
                "int4 matrix is empty, its columns are not a multiple of \
                 {INT4_COLUMN_MULTIPLE}, or it exceeds its buffers"
            )));
        }
        Ok(Self {
            codes,
            groups,
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
        std::ptr::eq(self.codes.raw(), output.raw())
            || std::ptr::eq(self.groups.raw(), output.raw())
    }
}

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(super) struct Int4Kernels {
    /// The vector kernel for one activation row. Two rows take the wide
    /// kernel: on an M4 Pro a two-row vector kernel took 300 us at 17408x5120
    /// and 17 at 256x5120, against 264 and 9 wide.
    vector: Pipeline,
    wide: Pipeline,
    /// The wide kernel with two 8-row tiles per threadgroup.
    paired: Pipeline,
}

impl Int4Kernels {
    pub(super) fn new(
        pipeline: impl Fn(&str, usize) -> crate::Result<Pipeline>,
    ) -> crate::Result<Self> {
        Ok(Self {
            vector: pipeline("bonsai_int4_matvec", 32)?,
            wide: pipeline("bonsai_int4_matmul_wide", WIDE_THREADS)?,
            paired: pipeline("bonsai_int4_matmul_wide_2", WIDE_THREADS)?,
        })
    }
}

fn extent(tokens: u32, width: u32) -> crate::Result<usize> {
    (tokens as usize)
        .checked_mul(width as usize)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| Error::InvalidArgument("int4 matmul extent overflow".into()))
}

impl BonsaiKernels {
    /// 4-bit projection of contiguous `[tokens, columns]` F32 rows into
    /// `[tokens, rows]` F32 rows, as [`Self::int8_matmul`] routes them: one
    /// dispatch per even chunk of at most eight rows, the vector kernel for one
    /// row and the wide simdgroup-matrix kernel above.
    #[expect(unsafe_code)]
    pub fn int4_matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: Int4Matrix<'_>,
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
                "int4 matmul buffers are empty, too short, or aliased".into(),
            ));
        }
        let tiles = if matrix.rows >= PAIRED_TILE_ROWS {
            2
        } else {
            1
        };
        let chunks = tokens.div_ceil(CHUNK_TOKENS);
        let mut start = 0;
        for chunk in 0..chunks {
            let end = (tokens * (chunk + 1)).div_ceil(chunks);
            let count = end - start;
            let wide = count > 1;
            let encoder = batch.encoder();
            encoder.setComputePipelineState(if wide {
                if tiles == 2 {
                    &self.int4.paired
                } else {
                    &self.int4.wide
                }
            } else {
                &self.int4.vector
            });
            unsafe {
                bind(encoder, matrix.codes, 0, 0);
                bind(encoder, matrix.groups, 0, 1);
                bind(encoder, input, extent(start, matrix.columns)?, 2);
                bind(encoder, output, extent(start, matrix.rows)?, 3);
                set_u32(encoder, &matrix.rows, 4);
                set_u32(encoder, &matrix.columns, 5);
                if wide {
                    set_u32(encoder, &count, 6);
                }
            }
            if wide {
                dispatch(
                    encoder,
                    (matrix.rows as usize).div_ceil(WIDE_ROWS * tiles),
                    WIDE_THREADS,
                );
            } else {
                dispatch(encoder, (matrix.rows as usize).div_ceil(VECTOR_ROWS), 32);
            }
            batch.record_dispatch();
            start = end;
        }
        Ok(())
    }
}
