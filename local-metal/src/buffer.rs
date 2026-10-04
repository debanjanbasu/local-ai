use core::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLPurgeableState, MTLResource,
    MTLResourceOptions,
};

use crate::Error;

fn contents_survived(previous: MTLPurgeableState) -> bool {
    previous != MTLPurgeableState::Empty
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPlacement {
    Shared,
    PrivatePreferred,
}

/// Safe wrapper around a Metal buffer.
#[derive(Clone)]
pub struct MetalBuffer {
    inner: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalBuffer {
    /// Make an idle buffer reclaimable by macOS.
    pub fn make_volatile(&self) {
        self.inner.setPurgeableState(MTLPurgeableState::Volatile);
    }

    /// Make a purgeable buffer resident before use. Returns `false` when its
    /// contents were discarded while volatile.
    #[must_use]
    pub fn make_nonvolatile(&self) -> bool {
        contents_survived(self.inner.setPurgeableState(MTLPurgeableState::NonVolatile))
    }

    /// Discard the contents of this buffer. Intended for cache-pressure tests.
    #[doc(hidden)]
    pub fn discard(&self) {
        self.inner.setPurgeableState(MTLPurgeableState::Empty);
    }

    /// Create a buffer initialised from a byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferAllocation`] if Metal cannot allocate the buffer.
    #[allow(unsafe_code)]
    pub fn from_slice<T: bytemuck::NoUninit>(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> crate::Result<Self> {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let len = bytes.len();
        // SAFETY: pointer is valid for `len` bytes; shared storage mode is CPU+GPU accessible.
        let inner = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(bytes.as_ptr().cast_mut().cast::<c_void>()),
                len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| Error::BufferAllocation {
            size: len,
            reason: "newBufferWithBytes returned nil".to_owned(),
        })?;
        Ok(Self { inner })
    }

    /// Create an uninitialised buffer of the given byte length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferAllocation`] if Metal cannot allocate the buffer.
    pub fn empty(device: &ProtocolObject<dyn MTLDevice>, byte_len: usize) -> crate::Result<Self> {
        let inner = device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::BufferAllocation {
                size: byte_len,
                reason: "newBufferWithLength returned nil".to_owned(),
            })?;
        Ok(Self { inner })
    }
    /// Wrap page-aligned, externally-owned shared memory without copying it.
    ///
    /// # Safety
    ///
    /// `owner` must keep `pointer..pointer+byte_len` readable and page-backed
    /// for its lifetime. The pointer and length must satisfy Metal's
    /// page-alignment requirements. Native buffer retention also retains
    /// `owner`, including after all Rust wrappers have been dropped.
    #[allow(unsafe_code)]
    pub unsafe fn from_bytes_no_copy(
        device: &ProtocolObject<dyn MTLDevice>,
        pointer: NonNull<u8>,
        byte_len: usize,
        owner: Arc<dyn Send + Sync>,
    ) -> crate::Result<Self> {
        let raw = pointer.cast::<c_void>();
        let deallocator = RcBlock::new(move |_pointer: NonNull<c_void>, _length: usize| {
            // Metal retains the block until native buffer destruction. The
            // capture is released with that block, not consumed by this Fn.
            let _ = &owner;
        });
        // SAFETY: the caller supplies valid pages. The escaping block owns the
        // backing allocation; its only capture is Send + Sync and 'static.
        let inner = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                raw,
                byte_len,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked,
                Some(&deallocator),
            )
        }
        .ok_or_else(|| Error::BufferAllocation {
            size: byte_len,
            reason: "newBufferWithBytesNoCopy returned nil".to_owned(),
        })?;
        Ok(Self { inner })
    }

    /// Buffer size in bytes.
    #[must_use]
    pub fn length(&self) -> usize {
        self.inner.length()
    }

    /// Reinterpret the buffer contents as a typed slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length is not a multiple of `size_of::<T>()` or if
    /// the pointer is not suitably aligned.
    #[must_use]
    #[allow(unsafe_code)]
    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        let ptr = self.inner.contents().as_ptr().cast::<u8>();
        let len = self.inner.length();
        // SAFETY: Metal shared-mode buffer contents are CPU-accessible and valid
        // for the lifetime of the buffer.  We hold `&self` so the buffer is alive.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        bytemuck::cast_slice(bytes)
    }

    /// Typed mutable view over the buffer contents.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length is not a multiple of `size_of::<T>()` or if
    /// the pointer is not suitably aligned.
    #[allow(unsafe_code)]
    pub fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        let ptr = self.inner.contents().as_ptr().cast::<u8>();
        let len = self.inner.length();
        // SAFETY: Metal shared-mode buffer contents are CPU-accessible and valid
        // for the lifetime of the buffer.  We hold `&mut self` so the buffer is
        // alive and uniquely borrowed.
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        bytemuck::cast_slice_mut(bytes)
    }

    /// Raw mutable pointer to the buffer contents (for weight-pool bulk copies).
    #[must_use]
    pub fn contents_mut_ptr(&self) -> *mut u8 {
        self.inner.contents().as_ptr().cast::<u8>()
    }

    /// Copy raw bytes into the buffer at the given byte offset.
    ///
    /// # Panics
    ///
    /// Panics if `offset + src.len()` exceeds the buffer length.
    #[allow(unsafe_code)]
    pub fn copy_from_bytes(&self, src: &[u8], offset: usize) {
        assert!(
            offset + src.len() <= self.length(),
            "copy_from_bytes: out of bounds (offset={offset}, src_len={}, buf_len={})",
            src.len(),
            self.length(),
        );
        // SAFETY: bounds checked above; shared buffer is CPU-writable.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.contents_mut_ptr().add(offset),
                src.len(),
            );
        }
    }

    /// Fill the shared buffer with zero bytes.
    #[allow(unsafe_code)]
    pub fn clear(&self) {
        // SAFETY: `contents_mut_ptr` is valid for the retained buffer length.
        unsafe { std::ptr::write_bytes(self.contents_mut_ptr(), 0, self.length()) };
    }

    /// Access the underlying `MTLBuffer` protocol object.
    #[must_use]
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.inner
    }

    /// Current logical placement hint for this buffer.
    #[must_use]
    pub const fn placement(&self) -> BufferPlacement {
        BufferPlacement::Shared
    }
}

#[cfg(test)]
mod purgeable_tests {
    use super::*;

    #[test]
    fn empty_previous_state_requires_cache_miss() {
        assert!(!contents_survived(MTLPurgeableState::Empty));
        assert!(contents_survived(MTLPurgeableState::Volatile));
        assert!(contents_survived(MTLPurgeableState::NonVolatile));
    }
}

/// Obtain the system default Metal device.
///
/// # Errors
///
/// Returns [`Error::NoDevice`] if no Metal device is available.
pub fn get_default_device() -> crate::Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn buffer_from_slice_roundtrip() {
        let device = get_default_device().expect("Metal device");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = MetalBuffer::from_slice(&device, &data).expect("create buffer");
        assert_eq!(buf.length(), data.len() * size_of::<f32>());
        let out: &[f32] = buf.as_slice();
        assert_eq!(out, &data[..]);
    }

    #[test]
    fn buffer_empty() {
        let device = get_default_device().expect("Metal device");
        let buf = MetalBuffer::empty(&device, 1024).expect("create buffer");
        assert_eq!(buf.length(), 1024);
    }

    #[test]
    #[allow(unsafe_code)]
    fn no_copy_owner_follows_native_retains() {
        #[repr(align(16384))]
        struct Page([u8; 16384]);

        let device = get_default_device().expect("Metal device");
        let backing = Arc::new(Page([0x5a; 16384]));
        let owner = Arc::new(Arc::clone(&backing));
        let weak = Arc::downgrade(&owner);
        let pointer = NonNull::new(backing.0.as_ptr().cast_mut()).expect("page pointer");
        // SAFETY: the aligned page supplies valid memory owned by `owner`.
        // An independent guard keeps it alive even if this regression fails.
        let buffer =
            unsafe { MetalBuffer::from_bytes_no_copy(&device, pointer, backing.0.len(), owner) }
                .expect("no-copy buffer");
        let cloned = buffer.clone();
        let native = buffer.inner.clone();
        drop(buffer);
        assert!(
            weak.upgrade().is_some(),
            "the Rust clone must retain backing"
        );
        drop(cloned);
        assert!(
            weak.upgrade().is_some(),
            "native MTLBuffer retention must also retain backing"
        );
        drop(native);
        assert!(
            weak.upgrade().is_none(),
            "the final release must not leak backing"
        );
        drop(backing);
    }

    #[test]
    fn buffer_copy_from_bytes() {
        let device = get_default_device().expect("Metal device");
        let buf = MetalBuffer::empty(&device, 256).expect("create buffer");
        let data: Vec<f32> = vec![10.0, 20.0, 30.0];
        let bytes = bytemuck::cast_slice::<f32, u8>(&data);
        buf.copy_from_bytes(bytes, 0);
        let out: &[f32] = buf.as_slice();
        assert_eq!(&out[..3], &data[..]);
    }

    #[test]
    fn buffer_from_slice_f16() {
        let device = get_default_device().expect("Metal device");
        let data: Vec<half::f16> = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ];
        let buf = MetalBuffer::from_slice(&device, &data).expect("create buffer");
        assert_eq!(buf.length(), data.len() * size_of::<half::f16>());
    }
}
