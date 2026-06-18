use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
};

use crate::Error;

/// Capabilities of the active GPU, queried once at startup.
///
/// Lets the engine size itself to the hardware it actually runs on — from an
/// iPhone SE's A15 to an M-series Max — instead of assuming a fixed machine.
#[derive(Debug, Clone)]
pub struct DeviceCaps {
    pub name: String,
    /// Apple GPU family generation (e.g. 7 = A14/M1, 8 = A15/A16/M2, 9 = A17/M3+);
    /// 0 if no Apple family is reported.
    pub apple_family: u32,
    /// Approximate bytes of memory the device can use with good performance
    /// before it is overcommitted ([`MTLDevice::recommendedMaxWorkingSetSize`]).
    pub recommended_working_set: u64,
    /// Whether CPU and GPU share one physical memory pool (all Apple Silicon).
    pub has_unified_memory: bool,
    /// Device-level maximum threads per threadgroup (width).
    pub max_threads_per_threadgroup: usize,
}

/// Safe wrapper around the Metal device and command queue.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    pub fn new() -> crate::Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoMetalDevice)?;
        let queue = device.newCommandQueue().ok_or(Error::NoCommandQueue)?;
        Ok(Self { device, queue })
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
            has_unified_memory: self.device.hasUnifiedMemory(),
            max_threads_per_threadgroup: self.device.maxThreadsPerThreadgroup().width,
        }
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
}

/// Thread-safe shared context.
pub type SharedMetalContext = Arc<MetalContext>;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn create_context() {
        let ctx = MetalContext::new().expect("create Metal context");
        assert!(!ctx.device_name().is_empty());
    }

    #[test]
    fn device_name_contains_apple() {
        let ctx = MetalContext::new().expect("create Metal context");
        let name = ctx.device_name();
        assert!(
            name.contains("Apple") || name.contains("apple"),
            "Unexpected device name: {name}"
        );
    }

    #[test]
    fn create_command_buffer() {
        let ctx = MetalContext::new().expect("create Metal context");
        let _cb = ctx.new_command_buffer().expect("create command buffer");
    }
}
