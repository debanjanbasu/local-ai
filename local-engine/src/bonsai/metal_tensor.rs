use local_metal::bonsai::Ptq1Matrix;
use local_metal::buffer::MetalBuffer;

use super::{BonsaiTensor, BonsaiTensorType, METAL_PAGE, align_up, invalid};

/// One checkpoint tensor bound as a Metal buffer: a page-aligned no-copy
/// window of the mapping when the device allows it, otherwise a copy.
pub struct BonsaiMetalTensor {
    pub(super) buffer: MetalBuffer,
    pub(super) offset: usize,
    pub(super) tensor: BonsaiTensor,
    pub(super) copied: bool,
}

impl BonsaiMetalTensor {
    /// Read only the original tensor bytes, excluding mapping-page padding.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.buffer.as_slice::<u8>()[self.offset..self.offset + self.tensor.bytes as usize]
    }
    // Runtime-only binding. Do not expose the mmap's mutable MetalBuffer API.
    pub(crate) const fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
    #[must_use]
    pub const fn copied(&self) -> bool {
        self.copied
    }
    #[must_use]
    pub const fn tensor(&self) -> &BonsaiTensor {
        &self.tensor
    }
    pub fn ptq1_matrix(&self) -> crate::Result<Ptq1Matrix<'_>> {
        if self.tensor.tensor_type != BonsaiTensorType::Ptq1 || self.tensor.dimensions.len() != 2 {
            return Err(crate::Error::InvalidFormat(
                "PTQ1 matrix view requires a rank-2 PTQ1 tensor".into(),
            ));
        }
        let columns = u32::try_from(self.tensor.dimensions[0])
            .map_err(|_| crate::Error::InvalidFormat("PTQ1 columns exceed u32".into()))?;
        let rows = u32::try_from(self.tensor.dimensions[1])
            .map_err(|_| crate::Error::InvalidFormat("PTQ1 rows exceed u32".into()))?;
        Ptq1Matrix::new(&self.buffer, self.offset, rows, columns).map_err(crate::Error::Metal)
    }
}

/// The page-aligned mapping window a no-copy Metal buffer may cover, or
/// `None` when the tensor must be copied instead (window past EOF or over the
/// device limit while the bare tensor still fits).
pub(super) fn metal_window(
    absolute: usize,
    bytes: usize,
    file_len: usize,
    maximum: usize,
) -> crate::Result<Option<std::ops::Range<usize>>> {
    let end = absolute
        .checked_add(bytes)
        .ok_or_else(|| crate::Error::InvalidFormat("Metal tensor range overflow".into()))?;
    if bytes == 0 || bytes > maximum || end > file_len {
        return invalid("tensor exceeds file or Metal buffer limit");
    }
    let start = absolute / METAL_PAGE * METAL_PAGE;
    let rounded_end = align_up(end, METAL_PAGE)?;
    Ok((rounded_end <= file_len && rounded_end - start <= maximum).then_some(start..rounded_end))
}
