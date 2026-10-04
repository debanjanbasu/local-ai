//! Command buffer batching for reduced GPU synchronization overhead.
//!
//! Groups multiple compute dispatches into a single Metal command buffer,
//! reducing per-layer GPU syncs from ~15 to 4 (dense) or fewer.

use std::sync::{Arc, Mutex};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
};

use crate::Error;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;

/// A batch of GPU compute dispatches sharing one command buffer + encoder.
///
/// Create with [`CommandBatch::new`], encode multiple dispatches via
/// [`encoder()`](CommandBatch::encoder), then call [`commit_and_wait()`](CommandBatch::commit_and_wait)
/// or [`commit_async()`](CommandBatch::commit_async).
pub struct CommandBatch {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    encoder_open: bool,
    leases: Option<CompletionLeases>,
    dispatch_count: u32,
    pending: Vec<PendingCommandBuffer>,
}

type CompletionLeases = Arc<Mutex<Vec<Arc<dyn Send + Sync>>>>;

#[derive(Clone)]
pub struct PendingCommandBuffer {
    cmd_bufs: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

pub struct BufferCopyRequest<'a> {
    pub source: &'a MetalBuffer,
    pub source_offset: usize,
    pub destination: &'a MetalBuffer,
    pub destination_offset: usize,
    pub size: usize,
}

#[allow(unsafe_code)]
fn encode_buffer_copy(
    encoder: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    copy: &BufferCopyRequest<'_>,
) -> crate::Result<()> {
    if copy
        .source_offset
        .checked_add(copy.size)
        .is_none_or(|end| end > copy.source.length())
        || copy
            .destination_offset
            .checked_add(copy.size)
            .is_none_or(|end| end > copy.destination.length())
    {
        return Err(Error::CommandBuffer(format!(
            "Buffer copy out of bounds (src_off={}, dst_off={}, size={}, src_len={}, dst_len={})",
            copy.source_offset,
            copy.destination_offset,
            copy.size,
            copy.source.length(),
            copy.destination.length(),
        )));
    }

    // SAFETY: bounds checked above; buffers remain alive for the duration
    // of the command buffer; same-queue submission preserves ordering.
    unsafe {
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            copy.source.raw(),
            copy.source_offset,
            copy.destination.raw(),
            copy.destination_offset,
            copy.size,
        );
    }
    Ok(())
}

/// Submit one blit command buffer that performs all requested buffer copies.
///
/// # Errors
///
/// Returns [`Error::CommandBuffer`] if Metal cannot allocate a blit encoder.
pub fn submit_buffer_copies_iter<'a, I>(
    ctx: &MetalContext,
    copies: I,
) -> crate::Result<PendingCommandBuffer>
where
    I: IntoIterator<Item = BufferCopyRequest<'a>>,
{
    let cmd_buf = ctx.new_command_buffer()?;
    let encoder = cmd_buf
        .blitCommandEncoder()
        .ok_or_else(|| Error::CommandBuffer("Failed to create blit encoder".to_owned()))?;

    let result = copies
        .into_iter()
        .try_for_each(|copy| encode_buffer_copy(&encoder, &copy));
    encoder.endEncoding();
    result?;
    cmd_buf.commit();
    Ok(PendingCommandBuffer::single(cmd_buf))
}

/// Submit one blit command buffer that performs all requested buffer copies.
///
/// # Errors
///
/// Returns [`Error::CommandBuffer`] if Metal cannot allocate a blit encoder.
pub fn submit_buffer_copies(
    ctx: &MetalContext,
    copies: &[BufferCopyRequest<'_>],
) -> crate::Result<PendingCommandBuffer> {
    submit_buffer_copies_iter(
        ctx,
        copies.iter().map(|copy| BufferCopyRequest {
            source: copy.source,
            source_offset: copy.source_offset,
            destination: copy.destination,
            destination_offset: copy.destination_offset,
            size: copy.size,
        }),
    )
}

impl PendingCommandBuffer {
    fn single(cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self {
            cmd_bufs: vec![cmd_buf],
        }
    }

    fn extend(&mut self, other: Self) {
        self.cmd_bufs.extend(other.cmd_bufs);
    }

    /// Block until the submitted command buffer completes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] if the GPU reports an error.
    pub fn wait(self) -> crate::Result<()> {
        self.wait_checked()
    }

    /// Wait, then sum command-buffer GPU execution times, excluding CPU encoding.
    ///
    /// Metal publishes these timestamps only after completion. The returned time
    /// includes all work inside each buffer, including blits and encoder changes,
    /// but not the gaps between separately submitted buffers on the same queue.
    pub fn wait_with_gpu_time(self) -> crate::Result<std::time::Duration> {
        self.wait_checked()?;
        self.cmd_bufs
            .iter()
            .try_fold(std::time::Duration::ZERO, |elapsed, cmd_buf| {
                let seconds = cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime();
                let duration =
                    std::time::Duration::try_from_secs_f64(seconds).map_err(|error| {
                        Error::CommandBuffer(format!("Invalid Metal GPU timestamps: {error}"))
                    })?;
                elapsed
                    .checked_add(duration)
                    .ok_or_else(|| Error::CommandBuffer("Metal GPU duration overflow".into()))
            })
    }

    fn wait_checked(&self) -> crate::Result<()> {
        let mut first_error = None;
        for cmd_buf in &self.cmd_bufs {
            cmd_buf.waitUntilCompleted();
            if let Some(err) = cmd_buf.error()
                && first_error.is_none()
            {
                first_error = Some(Error::CommandBuffer(format!("GPU error: {err}")));
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl CommandBatch {
    /// Create a new batch from the given Metal context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] if Metal cannot allocate a command
    /// buffer or compute encoder.
    pub fn new(ctx: &MetalContext) -> crate::Result<Self> {
        let cmd_buf = ctx.new_command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        Ok(Self {
            cmd_buf,
            encoder,
            encoder_open: true,
            leases: None,
            dispatch_count: 0,
            pending: Vec::new(),
        })
    }

    /// Access the shared compute encoder for encoding dispatches.
    #[must_use]
    pub fn encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        &self.encoder
    }

    /// Record that a dispatch was encoded (for diagnostics).
    pub const fn record_dispatch(&mut self) {
        self.dispatch_count += 1;
    }

    /// Number of dispatches encoded so far.
    #[must_use]
    pub const fn dispatch_count(&self) -> u32 {
        self.dispatch_count
    }

    /// Pin a cache entry until this command buffer has finished using its bytes.
    ///
    /// Native buffer retention prevents deallocation, not CPU reuse of a pool
    /// slot. The completion handler releases this separate lease even while a
    /// caller retains the completed command buffer. An uncommitted command
    /// buffer releases the capture on destruction instead.
    #[allow(unsafe_code)]
    pub fn retain_until_completed<T: Send + Sync + 'static>(&mut self, lease: Arc<T>) {
        let leases = self.leases.get_or_insert_with(|| {
            let leases = Arc::new(Mutex::new(Vec::<Arc<dyn Send + Sync>>::new()));
            let capture = Arc::clone(&leases);
            let completed = RcBlock::new(move |_| {
                capture
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clear();
            });
            // SAFETY: registered before commit; the escaping block owns only
            // synchronized, Send + Sync leases and no executor-local state.
            unsafe {
                self.cmd_buf
                    .addCompletedHandler(RcBlock::as_ptr(&completed));
            };
            leases
        });
        leases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(lease);
    }

    fn end_compute_encoding(&mut self) {
        if self.encoder_open {
            self.encoder.endEncoding();
            self.encoder_open = false;
        }
    }

    /// Track an already-submitted command buffer so a later batch-wide wait
    /// also reports its completion/errors.
    pub fn track_pending(&mut self, pending: PendingCommandBuffer) {
        self.pending.push(pending);
    }

    /// Wait for every command buffer previously submitted by this batch.
    ///
    /// The current encoder remains open, so CPU work can overlap submitted GPU work and
    /// encoding can continue after the synchronization point.
    pub fn wait_pending(&mut self) -> crate::Result<()> {
        let mut first_error = None;
        for pending in std::mem::take(&mut self.pending) {
            if let Err(error) = pending.wait()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Encode one or more buffer blits into the current command buffer.
    ///
    /// The compute encoder is ended, a temporary blit encoder is used for the
    /// copies, and then a fresh compute encoder is opened on the same command
    /// buffer so later compute dispatches remain ordered behind the blits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] if Metal cannot allocate the blit or
    /// renewed compute encoder, or if any copy would be out of bounds.
    pub fn blit_buffer_copies<'a, I>(&mut self, copies: I) -> crate::Result<()>
    where
        I: IntoIterator<Item = BufferCopyRequest<'a>>,
    {
        self.end_compute_encoding();
        let encoder = self
            .cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create blit encoder".to_owned()))?;
        let mut result = Ok(());
        for copy in copies {
            if let Err(error) = encode_buffer_copy(&encoder, &copy) {
                result = Err(error);
                break;
            }
        }
        encoder.endEncoding();
        self.encoder = self
            .cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        self.encoder_open = true;
        result
    }

    fn submit_current_and_renew(
        &mut self,
        ctx: &MetalContext,
    ) -> crate::Result<PendingCommandBuffer> {
        // Finish every fallible allocation before committing. An allocation
        // failure must leave the old encoder open and its leases unsubmitted.
        let next = ctx.new_command_buffer()?;
        let next_encoder = next
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        self.end_compute_encoding();
        self.cmd_buf.commit();
        // The completion handler, not this now-renewed batch, owns these leases.
        self.leases = None;
        let cmd_buf = std::mem::replace(&mut self.cmd_buf, next);
        self.encoder = next_encoder;
        self.encoder_open = true;
        self.dispatch_count = 0;
        Ok(PendingCommandBuffer::single(cmd_buf))
    }

    /// End encoding, commit, and block until GPU completes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] if the GPU reports an error.
    pub fn commit_and_wait(self) -> crate::Result<()> {
        self.commit_async().wait()
    }

    /// Submit the current batch, keep encoding into a fresh command buffer,
    /// and defer the wait/error check until a later [`commit_and_wait`] or
    /// [`commit_async`] on this batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] if Metal cannot allocate the renewed
    /// command buffer or encoder.
    pub fn submit_and_renew(&mut self, ctx: &MetalContext) -> crate::Result<()> {
        let pending = self.submit_current_and_renew(ctx)?;
        self.pending.push(pending);
        Ok(())
    }

    /// Commit the current batch, wait for GPU, then reinitialise with a fresh
    /// command buffer + encoder so callers can keep encoding.
    ///
    /// This is useful inside `TransformerLayer::forward_decode` where the layer
    /// needs a CPU sync point (e.g. to read GPU results for CPU-side post-processing)
    /// but doesn't own the batch lifecycle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CommandBuffer`] if the GPU reports an error or if the
    /// new command buffer / encoder cannot be created.
    pub fn commit_and_renew(&mut self, ctx: &MetalContext) -> crate::Result<()> {
        let current = self.submit_current_and_renew(ctx)?;
        self.pending.push(current);
        self.wait_pending()?;
        Ok(())
    }

    /// End encoding and commit without waiting (for pipelining).
    /// Returns a handle that can be waited on later.
    #[must_use]
    pub fn commit_async(mut self) -> PendingCommandBuffer {
        self.end_compute_encoding();
        self.cmd_buf.commit();
        self.leases = None;
        let mut pending = PendingCommandBuffer::single(self.cmd_buf.clone());
        for earlier in self.pending.drain(..).rev() {
            pending.extend(earlier);
        }
        pending.cmd_bufs.reverse();
        pending
    }
}

impl Drop for CommandBatch {
    fn drop(&mut self) {
        self.end_compute_encoding();
        // The current buffer was never submitted if it still has a lease group.
        // Do not wait for an autorelease pool to discard that unused capture.
        if let Some(leases) = self.leases.take() {
            leases
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clear();
        }
        // Early returns must not leave already submitted writes racing a reset.
        // Explicit waits report errors; Drop can only finish outstanding work.
        let _ = self.wait_pending();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
