//! Command buffer batching for reduced GPU synchronization overhead.
//!
//! Groups multiple compute dispatches into a single Metal command buffer,
//! reducing per-layer GPU syncs from ~15 to 4 (dense) or fewer.

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
    dispatch_count: u32,
    pending: Vec<PendingCommandBuffer>,
}

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
    if copy.source_offset + copy.size > copy.source.length()
        || copy.destination_offset + copy.size > copy.destination.length()
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

    for copy in copies {
        encode_buffer_copy(&encoder, &copy)?;
    }

    encoder.endEncoding();
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
        for cmd_buf in self.cmd_bufs {
            cmd_buf.waitUntilCompleted();
            if let Some(err) = cmd_buf.error() {
                return Err(Error::CommandBuffer(format!("GPU error: {err}")));
            }
        }
        Ok(())
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

    /// Track an already-submitted command buffer so a later batch-wide wait
    /// also reports its completion/errors.
    pub fn track_pending(&mut self, pending: PendingCommandBuffer) {
        self.pending.push(pending);
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
        self.encoder.endEncoding();
        let encoder = self
            .cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create blit encoder".to_owned()))?;
        for copy in copies {
            encode_buffer_copy(&encoder, &copy)?;
        }
        encoder.endEncoding();
        self.encoder = self
            .cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
        Ok(())
    }

    fn submit_current_and_renew(
        &mut self,
        ctx: &MetalContext,
    ) -> crate::Result<PendingCommandBuffer> {
        self.encoder.endEncoding();
        self.cmd_buf.commit();
        let cmd_buf = std::mem::replace(&mut self.cmd_buf, ctx.new_command_buffer()?);
        self.encoder = self
            .cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| Error::CommandBuffer("Failed to create compute encoder".to_owned()))?;
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
        let mut pending = std::mem::take(&mut self.pending);
        pending.push(current);
        for pending in pending {
            pending.wait()?;
        }
        Ok(())
    }

    /// End encoding and commit without waiting (for pipelining).
    /// Returns a handle that can be waited on later.
    #[must_use]
    pub fn commit_async(mut self) -> PendingCommandBuffer {
        self.encoder.endEncoding();
        self.cmd_buf.commit();
        let mut pending = PendingCommandBuffer::single(self.cmd_buf);
        for earlier in self.pending.drain(..).rev() {
            pending.extend(earlier);
        }
        pending.cmd_bufs.reverse();
        pending
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn create_and_commit_empty_batch() {
        let ctx = Arc::new(MetalContext::new().expect("Metal context"));
        let batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
        assert_eq!(batch.dispatch_count(), 0);
        batch.commit_and_wait().expect("commit_and_wait");
    }

    #[test]
    fn dispatch_count_increments() {
        let ctx = Arc::new(MetalContext::new().expect("Metal context"));
        let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
        batch.record_dispatch();
        batch.record_dispatch();
        assert_eq!(batch.dispatch_count(), 2);
        batch.commit_and_wait().expect("commit_and_wait");
    }

    #[test]
    fn commit_async_returns_command_buffer() {
        let ctx = Arc::new(MetalContext::new().expect("Metal context"));
        let batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
        let pending = batch.commit_async();
        pending.wait().expect("wait");
    }

    #[test]
    fn submit_buffer_copies_moves_data_between_shared_buffers() {
        let ctx = Arc::new(MetalContext::new().expect("Metal context"));
        let src =
            crate::buffer::MetalBuffer::from_slice(ctx.device(), &[1_u32, 2, 3, 4]).expect("src");
        let dst = crate::buffer::MetalBuffer::empty(ctx.device(), src.length()).expect("dst");

        let pending = submit_buffer_copies(
            &ctx,
            &[BufferCopyRequest {
                source: &src,
                source_offset: std::mem::size_of::<u32>(),
                destination: &dst,
                destination_offset: 0,
                size: std::mem::size_of::<u32>() * 2,
            }],
        )
        .expect("submit copies");
        pending.wait().expect("copy wait");

        assert_eq!(&dst.as_slice::<u32>()[..2], &[2, 3]);
    }

    #[test]
    fn submit_buffer_copies_iter_moves_data_between_shared_buffers() {
        let ctx = Arc::new(MetalContext::new().expect("Metal context"));
        let src = crate::buffer::MetalBuffer::from_slice(ctx.device(), &[1_u32, 2, 3, 4, 5, 6])
            .expect("src");
        let dst = crate::buffer::MetalBuffer::empty(ctx.device(), src.length()).expect("dst");

        let pending = submit_buffer_copies_iter(
            &ctx,
            [
                BufferCopyRequest {
                    source: &src,
                    source_offset: std::mem::size_of::<u32>(),
                    destination: &dst,
                    destination_offset: 0,
                    size: std::mem::size_of::<u32>() * 2,
                },
                BufferCopyRequest {
                    source: &src,
                    source_offset: std::mem::size_of::<u32>() * 4,
                    destination: &dst,
                    destination_offset: std::mem::size_of::<u32>() * 2,
                    size: std::mem::size_of::<u32>() * 2,
                },
            ],
        )
        .expect("submit copies");
        pending.wait().expect("copy wait");

        assert_eq!(&dst.as_slice::<u32>()[..4], &[2, 3, 5, 6]);
    }

    #[test]
    fn submit_and_renew_can_track_intermediate_copy_work() {
        let ctx = Arc::new(MetalContext::new().expect("Metal context"));
        let src = crate::buffer::MetalBuffer::from_slice(ctx.device(), &[10_u32, 20, 30, 40])
            .expect("src");
        let dst = crate::buffer::MetalBuffer::empty(ctx.device(), src.length()).expect("dst");

        let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
        batch.record_dispatch();
        batch.submit_and_renew(&ctx).expect("submit_and_renew");
        batch.track_pending(
            submit_buffer_copies(
                &ctx,
                &[BufferCopyRequest {
                    source: &src,
                    source_offset: std::mem::size_of::<u32>(),
                    destination: &dst,
                    destination_offset: 0,
                    size: std::mem::size_of::<u32>() * 2,
                }],
            )
            .expect("submit copies"),
        );
        batch.commit_and_wait().expect("commit_and_wait");

        assert_eq!(&dst.as_slice::<u32>()[..2], &[20, 30]);
    }

    #[test]
    fn blit_buffer_copies_moves_data_inside_existing_batch() {
        let ctx = Arc::new(MetalContext::new().expect("Metal context"));
        let src =
            crate::buffer::MetalBuffer::from_slice(ctx.device(), &[7_u32, 8, 9, 10]).expect("src");
        let dst = crate::buffer::MetalBuffer::empty(ctx.device(), src.length()).expect("dst");

        let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
        batch.record_dispatch();
        batch
            .blit_buffer_copies([BufferCopyRequest {
                source: &src,
                source_offset: std::mem::size_of::<u32>(),
                destination: &dst,
                destination_offset: 0,
                size: std::mem::size_of::<u32>() * 2,
            }])
            .expect("blit copies");
        batch.commit_and_wait().expect("commit_and_wait");

        assert_eq!(&dst.as_slice::<u32>()[..2], &[8, 9]);
    }
}
