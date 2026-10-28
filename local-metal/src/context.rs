use std::sync::{Arc, Mutex, PoisonError};

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResourceOptions,
};

use crate::Error;
use crate::buffer::MetalBuffer;

/// Capabilities of the active GPU, queried once at startup.
///
/// Lets the engine size itself to the hardware it actually runs on instead of
/// assuming a fixed machine.
#[derive(Debug, Clone)]
pub struct DeviceCaps {
    pub name: String,
    /// Apple GPU family generation (e.g. 7 = A14/M1, 8 = A15/A16/M2, 9 = A17/M3+);
    /// 0 if no Apple family is reported.
    pub apple_family: u32,
    /// Approximate bytes of memory the device can use with good performance
    /// before it is overcommitted ([`MTLDevice::recommendedMaxWorkingSetSize`]).
    pub recommended_working_set: u64,
    /// Largest single Metal buffer supported by the active device.
    pub max_buffer_length: u64,
    /// Nominal host-to-device transfer rate; zero for integrated GPUs.
    pub max_transfer_rate: u64,
    /// Whether CPU and GPU share one physical memory pool (all Apple Silicon).
    pub has_unified_memory: bool,
    /// Device-level maximum threads per threadgroup (width).
    pub max_threads_per_threadgroup: usize,
}

/// Safe wrapper around the Metal device and command queue.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    residency: QueueResidency,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    pub fn new() -> crate::Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoMetalDevice)?;
        let queue = device.newCommandQueue().ok_or(Error::NoCommandQueue)?;
        let residency = QueueResidency {
            device: device.clone(),
            queue: queue.clone(),
            sets: Arc::default(),
        };
        Ok(Self {
            device,
            queue,
            residency,
        })
    }

    /// Get the Metal device.
    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Get the retained Metal device (for passing to buffer creation, etc.).
    #[must_use]
    pub fn device_retained(&self) -> &Retained<ProtocolObject<dyn MTLDevice>> {
        &self.device
    }

    /// Get the device name.
    #[must_use]
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Query the active GPU's capabilities (memory budget, family, limits).
    #[must_use]
    pub fn caps(&self) -> DeviceCaps {
        let families: &[(MTLGPUFamily, u32)] = &[
            (MTLGPUFamily::Apple10, 10),
            (MTLGPUFamily::Apple9, 9),
            (MTLGPUFamily::Apple8, 8),
            (MTLGPUFamily::Apple7, 7),
            (MTLGPUFamily::Apple6, 6),
            (MTLGPUFamily::Apple5, 5),
            (MTLGPUFamily::Apple4, 4),
        ];
        let apple_family = families
            .iter()
            .find(|(f, _)| self.device.supportsFamily(*f))
            .map_or(0, |&(_, n)| n);
        DeviceCaps {
            name: self.device_name(),
            apple_family,
            recommended_working_set: self.device.recommendedMaxWorkingSetSize(),
            max_buffer_length: self.device.maxBufferLength() as u64,
            max_transfer_rate: self.device.maxTransferRate(),
            has_unified_memory: self.device.hasUnifiedMemory(),
            max_threads_per_threadgroup: self.device.maxThreadsPerThreadgroup().width,
        }
    }

    /// Bytes currently allocated through this Metal device.
    #[must_use]
    pub fn current_allocated_size(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }

    /// Create a new command buffer from the queue.
    pub fn new_command_buffer(
        &self,
    ) -> crate::Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.queue
            .commandBuffer()
            .ok_or_else(|| Error::CommandBuffer("Failed to create command buffer".to_owned()))
    }

    /// Get the command queue.
    #[must_use]
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Keep `buffers` resident for every command buffer of the queue while
    /// the returned set lives. Without one, Metal evicts an allocation no
    /// command buffer has used for about a second, and the next command
    /// buffer that uses it pays to make it resident again in the driver
    /// before the GPU starts. A set only holds while the GPU stays in use
    /// (see [`QueueResidency`]). `None` where residency sets are unavailable:
    /// before macOS 15, or where the device declines to make one (the
    /// paravirtual GPU of a macOS VM). Residency is only a speed-up, so that
    /// is not an error.
    pub fn keep_resident<'a>(
        &self,
        buffers: impl IntoIterator<Item = &'a MetalBuffer>,
    ) -> Option<ResidencySet> {
        if !self
            .device
            .respondsToSelector(sel!(newResidencySetWithDescriptor:error:))
        {
            return None;
        }
        let descriptor = MTLResidencySetDescriptor::new();
        let set = self
            .device
            .newResidencySetWithDescriptor_error(&descriptor)
            .ok()?;
        for buffer in buffers {
            set.addAllocation(ProtocolObject::from_ref(buffer.raw()));
        }
        set.commit();
        let mut sets = self.residency.lock();
        if !sets.suspended {
            set.requestResidency();
            self.queue.addResidencySet(&set);
        }
        sets.sets.push(Set(set.clone()));
        drop(sets);
        Some(ResidencySet {
            set,
            residency: self.residency.clone(),
        })
    }

    /// The queue's residency sets, for a thread that keeps them resident
    /// between requests or lets them go.
    #[must_use]
    pub fn residency(&self) -> QueueResidency {
        self.residency.clone()
    }
}

/// Allocations [`MetalContext::keep_resident`] keeps resident; dropping it
/// lets Metal evict them again.
pub struct ResidencySet {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    residency: QueueResidency,
}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        let mut sets = self.residency.lock();
        sets.sets
            .retain(|set| Retained::as_ptr(&set.0) != Retained::as_ptr(&self.set));
        if !sets.suspended {
            self.residency.queue.removeResidencySet(&self.set);
            self.set.endResidency();
        }
    }
}

/// Every residency set of one command queue, shareable with another thread.
///
/// A residency set holds only while the GPU is in use. After 2-3 s with no
/// command buffer the driver lets the process's residency go (wired memory
/// falls back), and the next command buffer makes every set on the queue
/// resident again before it starts: 123-136 ms for Bonsai's target and
/// drafter weights, which made a request after a 5 s pause slower than
/// without the sets. [`Self::ping`] at a shorter period keeps the sets
/// resident; [`Self::suspend`] takes them off the queue so Metal may evict
/// and unwire them, and [`Self::resume`] puts them back.
#[derive(Clone)]
pub struct QueueResidency {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    sets: Arc<Mutex<Sets>>,
}

#[derive(Default)]
struct Sets {
    sets: Vec<Set>,
    /// Whether the sets are off the queue.
    suspended: bool,
}

struct Set(Retained<ProtocolObject<dyn MTLResidencySet>>);

// SAFETY: Metal objects may be transferred between threads; a `Set` is only
// used under the `Sets` mutex.
#[expect(unsafe_code)]
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "Metal objects are thread-safe to transfer; access is serialized"
)]
unsafe impl Send for Set {}

impl QueueResidency {
    fn lock(&self) -> std::sync::MutexGuard<'_, Sets> {
        self.sets.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Commit a command buffer that clears 16 bytes, and wait for it. (An
    /// empty one holds nothing: after it the next command buffer still
    /// waited 122 ms.)
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferAllocation`] or [`Error::CommandBuffer`] if
    /// Metal cannot create the buffer or the command buffer, or reports an
    /// error.
    pub fn ping(&self) -> crate::Result<()> {
        const BYTES: usize = 16;
        let buffer = self
            .device
            .newBufferWithLength_options(BYTES, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| Error::BufferAllocation {
                size: BYTES,
                reason: "newBufferWithLength returned nil".to_owned(),
            })?;
        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::CommandBuffer("Failed to create command buffer".to_owned()))?;
        let encoder = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create blit encoder".to_owned()))?;
        encoder.fillBuffer_range_value(&buffer, NSRange::new(0, BYTES), 0);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        cmd_buf.error().map_or(Ok(()), |err| {
            Err(Error::CommandBuffer(format!("GPU error: {err}")))
        })
    }

    /// Take every set off the queue and end its residency.
    pub fn suspend(&self) {
        let mut sets = self.lock();
        if sets.suspended {
            return;
        }
        for set in &sets.sets {
            self.queue.removeResidencySet(&set.0);
            set.0.endResidency();
        }
        sets.suspended = true;
    }

    /// Put suspended sets back on the queue.
    pub fn resume(&self) {
        let mut sets = self.lock();
        if !sets.suspended {
            return;
        }
        for set in &sets.sets {
            set.0.requestResidency();
            self.queue.addResidencySet(&set.0);
        }
        sets.suspended = false;
    }
}

/// Thread-safe shared context.
pub type SharedMetalContext = Arc<MetalContext>;

#[cfg(test)]
mod tests {
    use super::*;

    /// `None` when this machine exposes no Metal device, so GPU tests skip
    /// instead of failing. Any other failure is still a failure: a missing GPU
    /// is an environment, a broken context is a bug.
    fn gpu_or_skip() -> Option<MetalContext> {
        let context = MetalContext::new();
        if matches!(context, Err(crate::Error::NoMetalDevice)) {
            eprintln!("skipping GPU test: this machine has no Metal device");
            return None;
        }
        Some(context.expect("Metal context failed for a reason other than a missing device"))
    }

    #[test]
    fn create_context() {
        let Some(ctx) = gpu_or_skip() else { return };
        assert_ne!(ctx.device_name(), "");
    }

    #[test]
    fn device_name_contains_apple() {
        let Some(ctx) = gpu_or_skip() else { return };
        let name = ctx.device_name();
        assert!(
            name.contains("Apple") || name.contains("apple"),
            "Unexpected device name: {name}"
        );
    }

    #[test]
    fn create_command_buffer() {
        let Some(ctx) = gpu_or_skip() else { return };
        let _cb = ctx.new_command_buffer().expect("create command buffer");
    }

    #[test]
    fn sets_made_or_dropped_while_suspended_follow_the_queue() {
        let Some(ctx) = gpu_or_skip() else { return };
        let buffer = MetalBuffer::empty(ctx.device(), 4096).expect("buffer");
        let residency = ctx.residency();
        let Some(kept) = ctx.keep_resident([&buffer]) else {
            return;
        };
        residency.suspend();
        let later = ctx.keep_resident([&buffer]).expect("set");
        assert_eq!(residency.lock().sets.len(), 2);
        drop(kept);
        assert_eq!(residency.lock().sets.len(), 1);
        residency.ping().expect("ping while suspended");
        residency.resume();
        assert!(!residency.lock().suspended);
        drop(later);
        assert!(residency.lock().sets.is_empty());
        residency.ping().expect("ping");
    }
}
