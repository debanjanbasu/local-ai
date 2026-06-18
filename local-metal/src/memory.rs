use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Reserved bytes for OS overhead (200 MiB).
pub const RESERVE_BYTES: usize = 200 * 1024 * 1024;

const MIN_POOL_BYTES: usize = 64 * 1024 * 1024;
const MAX_POOL_BYTES: usize = 512 * 1024 * 1024;
const FALLBACK_AVAILABLE: usize = 1_500_000_000; // 1.5 GB

/// Query the amount of memory available to this process.
///
/// Falls back to 1.5 GB if the platform probe is unavailable.
#[must_use]
pub fn available_memory() -> usize {
    available_memory_now().map_or(FALLBACK_AVAILABLE, |v| v as usize)
}

/// Memory actually free for this process to take **right now**, in bytes.
///
/// - **iOS**: [`os_proc_available_memory`] — the remaining allowance before
///   the jetsam limit kills the process (the authoritative per-app figure).
/// - **macOS**: reclaimable VM pages (`free + inactive + purgeable +
///   speculative`) via `host_statistics64`. macOS has no hard per-process
///   limit, so this is the soft "take this much without forcing swap" signal.
///
/// Returns `None` when the probe fails — callers should treat that as
/// "unknown" (do not constrain), not as zero.
#[must_use]
#[allow(unsafe_code)]
pub fn available_memory_now() -> Option<u64> {
    #[cfg(target_os = "ios")]
    {
        // SAFETY: simple C getter with no preconditions (iOS 13+).
        let val = unsafe { os_proc_available_memory() };
        (val > 0).then_some(val as u64)
    }
    #[cfg(target_os = "macos")]
    {
        macos_reclaimable_memory()
    }
    #[cfg(not(any(target_os = "ios", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "ios")]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn os_proc_available_memory() -> usize;
}

/// `free + inactive + purgeable + speculative` pages from `host_statistics64`.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn macos_reclaimable_memory() -> Option<u64> {
    /// `struct vm_statistics64` from `<mach/vm_statistics.h>` (38 × 4 bytes).
    #[repr(C)]
    #[derive(Default)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u32,
        throttled_count: u32,
        external_page_count: u32,
        internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    }
    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics64(host: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    const HOST_VM_INFO64: i32 = 4;
    const KERN_SUCCESS: i32 = 0;

    let mut stats = VmStatistics64::default();
    let mut count = (size_of::<VmStatistics64>() / size_of::<i32>()) as u32;
    // SAFETY: `stats` is a correctly sized/aligned vm_statistics64 buffer and
    // `count` carries its length in 32-bit words, per the Mach contract.
    let kr = unsafe {
        host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            (&raw mut stats).cast(),
            &raw mut count,
        )
    };
    if kr != KERN_SUCCESS {
        return None;
    }
    // SAFETY: sysconf is a safe libc query.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return None;
    }
    let pages = u64::from(stats.free_count)
        + u64::from(stats.inactive_count)
        + u64::from(stats.purgeable_count)
        + u64::from(stats.speculative_count);
    Some(pages * page as u64)
}

/// Hint the kernel to prefetch the given memory region.
///
/// The pointer is page-aligned internally. No-op on non-Darwin platforms.
#[allow(unsafe_code)]
pub fn prefetch_region(ptr: *const u8, len: usize) {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if len == 0 || ptr.is_null() {
            return;
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return;
        }
        #[allow(clippy::cast_sign_loss)]
        let page_mask = page_size as usize - 1;
        let aligned = (ptr as usize) & !page_mask;
        let extra = (ptr as usize) - aligned;
        let total = len + extra;
        // SAFETY: `madvise` with `MADV_WILLNEED` is advisory and safe to call.
        unsafe {
            libc::madvise(aligned as *mut libc::c_void, total, libc::MADV_WILLNEED);
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = (ptr, len);
    }
}

/// Hardware capabilities probed from the Metal device.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Recommended max GPU working set (bytes).
    pub gpu_memory_bytes: usize,
    /// Available system memory (bytes).
    pub system_memory_bytes: usize,
    /// Device name string.
    pub gpu_name: String,
    /// Max single buffer size (bytes).
    pub max_buffer_bytes: usize,
}

impl DeviceCapabilities {
    /// Probe capabilities from the given Metal device.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn probe(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let gpu_memory_bytes = device.recommendedMaxWorkingSetSize() as usize;
        let gpu_name = device.name().to_string();
        let max_buffer_bytes = device.maxBufferLength();
        let system_memory_bytes = available_memory();
        Self {
            gpu_memory_bytes,
            system_memory_bytes,
            gpu_name,
            max_buffer_bytes,
        }
    }
}

/// Memory allocation plan for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    /// Number of weight pool slots (round-robin layer decompression).
    pub pool_slots: usize,
    /// Bytes per pool slot.
    pub pool_slot_bytes: usize,
    /// Maximum context length (tokens) the KV cache can hold.
    pub max_context_length: usize,
    /// Number of expert cache slots (for `MoE` models).
    pub expert_cache_slots: usize,
    /// Bytes per expert slot.
    pub expert_slot_bytes: usize,
    /// Total scratch buffer bytes.
    pub scratch_bytes: usize,
}

/// Fraction of GPU memory to use (80%).
const USABLE_FRACTION_NUM: usize = 80;
const USABLE_FRACTION_DEN: usize = 100;

/// Model memory sizing parameters for [`MemoryPlan::compute`].
pub struct ModelMemoryParams {
    pub num_layers: usize,
    pub layer_weight_bytes: usize,
    pub scratch_bytes: usize,
    pub kv_entry_bytes: usize,
    pub num_kv_layers: usize,
    pub expert_slot_bytes: usize,
    pub requested_context: usize,
}

impl MemoryPlan {
    /// Compute memory plan given device capabilities and model parameters.
    ///
    /// Priority: weights → scratch → KV cache (maximize context) → expert cache (fill remaining).
    /// Uses 80% of `gpu_memory_bytes` as usable budget (leave headroom).
    #[must_use]
    pub fn compute(caps: &DeviceCapabilities, model: &ModelMemoryParams) -> Self {
        let budget = caps.gpu_memory_bytes * USABLE_FRACTION_NUM / USABLE_FRACTION_DEN;

        // 1. Pool slots: minimum 2 for double-buffering
        let pool_slot_bytes = model.layer_weight_bytes;
        let pool_slots = 2;
        let pool_total = pool_slots * pool_slot_bytes;

        // 2. Scratch
        let after_pool_scratch = budget
            .saturating_sub(pool_total)
            .saturating_sub(model.scratch_bytes);

        // 3. KV cache — bytes per token = kv_entry_bytes * num_kv_layers
        let kv_per_token = model.kv_entry_bytes.saturating_mul(model.num_kv_layers);
        let max_context_length = (after_pool_scratch)
            .checked_div(kv_per_token)
            .map_or(model.requested_context, |possible| {
                possible.min(model.requested_context)
            });
        let kv_total = max_context_length * kv_per_token;
        let after_kv = after_pool_scratch.saturating_sub(kv_total);

        // 4. Expert cache slots (fill remaining)
        let expert_cache_slots = after_kv.checked_div(model.expert_slot_bytes).unwrap_or(0);

        Self {
            pool_slots,
            pool_slot_bytes,
            max_context_length,
            expert_cache_slots,
            expert_slot_bytes: model.expert_slot_bytes,
            scratch_bytes: model.scratch_bytes,
        }
    }
}

/// Memory budget split across pool, KV cache, and scratch space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Bytes available for the weight buffer pool.
    pub pool_bytes: usize,
    /// Bytes available for the KV cache.
    pub kv_bytes: usize,
    /// Bytes available for scratch / activations.
    pub scratch_bytes: usize,
}

impl MemoryBudget {
    /// Compute a budget from the total available memory.
    ///
    /// Subtracts [`RESERVE_BYTES`], then splits 40/40/20 with the pool
    /// clamped to \[64 MiB, 512 MiB\]. Any excess is redistributed to KV.
    #[must_use]
    pub fn from_available(available: usize) -> Self {
        let usable = available.saturating_sub(RESERVE_BYTES);

        let raw_pool = usable * 40 / 100;
        let raw_kv = usable * 40 / 100;
        let raw_scratch = usable - raw_pool - raw_kv;

        let pool_bytes = raw_pool.clamp(MIN_POOL_BYTES, MAX_POOL_BYTES);
        // If pool was clamped down, excess goes to KV; if clamped up, KV shrinks.
        let kv_bytes = (raw_kv + raw_pool).saturating_sub(pool_bytes);
        let scratch_bytes = raw_scratch;

        Self {
            pool_bytes,
            kv_bytes,
            scratch_bytes,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn available_memory_returns_nonzero() {
        assert!(available_memory() > 0);
    }

    #[test]
    fn device_capabilities_probe() {
        let ctx = crate::context::MetalContext::new().expect("Metal context");
        let caps = DeviceCapabilities::probe(ctx.device());
        assert!(
            caps.gpu_memory_bytes > 0,
            "gpu_memory_bytes should be non-zero"
        );
        assert!(
            caps.system_memory_bytes > 0,
            "system_memory_bytes should be non-zero"
        );
        assert!(!caps.gpu_name.is_empty(), "gpu_name should be non-empty");
        assert!(
            caps.max_buffer_bytes > 0,
            "max_buffer_bytes should be non-zero"
        );
    }

    #[test]
    fn memory_plan_m4_pro_scenario() {
        let caps = DeviceCapabilities {
            gpu_memory_bytes: 36 * 1024 * 1024 * 1024,
            system_memory_bytes: 36 * 1024 * 1024 * 1024,
            gpu_name: "Apple M4 Pro".to_owned(),
            max_buffer_bytes: 16 * 1024 * 1024 * 1024,
        };
        let plan = MemoryPlan::compute(
            &caps,
            &ModelMemoryParams {
                num_layers: 26,
                layer_weight_bytes: 50 * 1024 * 1024,
                scratch_bytes: 64 * 1024 * 1024,
                kv_entry_bytes: 512,
                num_kv_layers: 26,
                expert_slot_bytes: 32 * 1024 * 1024,
                requested_context: 8192,
            },
        );
        assert!(plan.pool_slots >= 2);
        assert_eq!(plan.max_context_length, 8192);
        assert!(plan.expert_cache_slots > 0);
    }

    #[test]
    fn memory_plan_iphone_scenario() {
        let caps = DeviceCapabilities {
            gpu_memory_bytes: 6 * 1024 * 1024 * 1024,
            system_memory_bytes: 6 * 1024 * 1024 * 1024,
            gpu_name: "Apple A18 Pro".to_owned(),
            max_buffer_bytes: 4 * 1024 * 1024 * 1024,
        };
        let plan = MemoryPlan::compute(
            &caps,
            &ModelMemoryParams {
                num_layers: 26,
                layer_weight_bytes: 50 * 1024 * 1024,
                scratch_bytes: 64 * 1024 * 1024,
                kv_entry_bytes: 512,
                num_kv_layers: 26,
                expert_slot_bytes: 32 * 1024 * 1024,
                requested_context: 8192,
            },
        );
        assert!(plan.pool_slots >= 2);
        assert!(plan.max_context_length > 0);
        assert!(plan.max_context_length <= 8192);
    }

    #[test]
    fn memory_plan_dense_no_experts() {
        let caps = DeviceCapabilities {
            gpu_memory_bytes: 8 * 1024 * 1024 * 1024,
            system_memory_bytes: 8 * 1024 * 1024 * 1024,
            gpu_name: "Test".to_owned(),
            max_buffer_bytes: 4 * 1024 * 1024 * 1024,
        };
        let plan = MemoryPlan::compute(
            &caps,
            &ModelMemoryParams {
                num_layers: 18,
                layer_weight_bytes: 20 * 1024 * 1024,
                scratch_bytes: 32 * 1024 * 1024,
                kv_entry_bytes: 256,
                num_kv_layers: 18,
                expert_slot_bytes: 0,
                requested_context: 4096,
            },
        );
        assert_eq!(plan.expert_cache_slots, 0);
        assert_eq!(plan.expert_slot_bytes, 0);
    }

    #[test]
    fn memory_budget_reasonable() {
        let budget = MemoryBudget::from_available(4 * 1024 * 1024 * 1024); // 4 GiB
        // Pool should be clamped to max 512 MB
        assert!(budget.pool_bytes <= MAX_POOL_BYTES);
        assert!(budget.pool_bytes >= MIN_POOL_BYTES);
        // KV should get the lion's share
        assert!(budget.kv_bytes > budget.pool_bytes);
        // All three should be nonzero
        assert!(budget.kv_bytes > 0);
        assert!(budget.scratch_bytes > 0);
        // Sum should be roughly usable
        let total = budget.pool_bytes + budget.kv_bytes + budget.scratch_bytes;
        let usable = 4 * 1024 * 1024 * 1024 - RESERVE_BYTES;
        assert_eq!(total, usable);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod avail_tests {
    #[test]
    fn available_memory_now_is_sane() {
        let avail = super::available_memory_now().expect("probe works on macOS");
        // Between 100 MiB and 1 TiB — catches unit errors (pages vs bytes).
        assert!(avail > 100 * 1024 * 1024, "too small: {avail}");
        assert!(avail < 1 << 40, "too large: {avail}");
        eprintln!(
            "available now: {:.2} GiB",
            avail as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }
}
