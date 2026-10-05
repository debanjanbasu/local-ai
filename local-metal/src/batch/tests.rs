use super::*;
use std::sync::Arc;

fn drop_probe() -> (Arc<()>, std::sync::Weak<()>) {
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    (lease, weak)
}

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
fn create_and_commit_empty_batch() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
    assert_eq!(batch.dispatch_count(), 0);
    batch.commit_and_wait().expect("commit_and_wait");
}

#[test]
fn dispatch_count_increments() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
    batch.record_dispatch();
    batch.record_dispatch();
    assert_eq!(batch.dispatch_count(), 2);
    batch.commit_and_wait().expect("commit_and_wait");
}

#[test]
fn commit_async_returns_command_buffer() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
    let pending = batch.commit_async();
    pending.wait().expect("wait");
}

#[test]
fn submit_buffer_copies_moves_data_between_shared_buffers() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let src = crate::buffer::MetalBuffer::from_slice(ctx.device(), &[1_u32, 2, 3, 4]).expect("src");
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
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let src =
        crate::buffer::MetalBuffer::from_slice(ctx.device(), &[1_u32, 2, 3, 4, 5, 6]).expect("src");
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
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let src =
        crate::buffer::MetalBuffer::from_slice(ctx.device(), &[10_u32, 20, 30, 40]).expect("src");
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
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
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

#[test]
fn completed_command_releases_lease_while_pending_clone_is_alive() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let src = MetalBuffer::from_slice(ctx.device(), &[11_u32, 22, 33, 44, 55]).expect("src");
    let dst = MetalBuffer::empty(ctx.device(), src.length()).expect("dst");
    let (lease, weak) = drop_probe();

    let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
    batch.retain_until_completed(lease);
    batch
        .blit_buffer_copies([BufferCopyRequest {
            source: &src,
            source_offset: 2 * size_of::<u32>(),
            destination: &dst,
            destination_offset: size_of::<u32>(),
            size: 2 * size_of::<u32>(),
        }])
        .expect("blit copy");
    let pending = batch.commit_async();
    let completed_handle = pending.clone();
    pending.wait().expect("copy wait");

    assert_eq!(&dst.as_slice::<u32>()[1..3], &[33, 44]);
    assert!(weak.upgrade().is_none());
    drop(completed_handle);
}

#[test]
fn unsubmitted_batch_drop_releases_lease_immediately() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let (lease, weak) = drop_probe();
    let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
    batch.retain_until_completed(lease);

    drop(batch);

    assert!(weak.upgrade().is_none());
}

#[test]
fn submit_and_renew_keeps_distinct_per_command_leases() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let src = MetalBuffer::from_slice(ctx.device(), &[3_u32, 5, 7, 11, 13, 17]).expect("src");
    let dst = MetalBuffer::empty(ctx.device(), src.length()).expect("dst");
    let (first_lease, first_weak) = drop_probe();
    let (second_lease, second_weak) = drop_probe();
    let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");

    batch.retain_until_completed(first_lease);
    batch
        .blit_buffer_copies([BufferCopyRequest {
            source: &src,
            source_offset: size_of::<u32>(),
            destination: &dst,
            destination_offset: 3 * size_of::<u32>(),
            size: 2 * size_of::<u32>(),
        }])
        .expect("first copy");
    batch.submit_and_renew(&ctx).expect("first submit");
    batch.retain_until_completed(second_lease);
    assert!(second_weak.upgrade().is_some());

    batch.wait_pending().expect("first wait");
    assert!(first_weak.upgrade().is_none());
    assert!(second_weak.upgrade().is_some());
    assert_eq!(&dst.as_slice::<u32>()[3..5], &[5, 7]);

    batch.submit_and_renew(&ctx).expect("second submit");
    batch.wait_pending().expect("second wait");
    assert!(second_weak.upgrade().is_none());
}

#[test]
fn dropping_renewed_batch_drains_earlier_copy() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let src = MetalBuffer::from_slice(ctx.device(), &[101_u32, 202, 303, 404, 505]).expect("src");
    let dst = MetalBuffer::empty(ctx.device(), src.length()).expect("dst");
    let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");
    batch
        .blit_buffer_copies([BufferCopyRequest {
            source: &src,
            source_offset: size_of::<u32>(),
            destination: &dst,
            destination_offset: 2 * size_of::<u32>(),
            size: 3 * size_of::<u32>(),
        }])
        .expect("copy");
    batch.submit_and_renew(&ctx).expect("submit_and_renew");

    drop(batch);

    assert_eq!(&dst.as_slice::<u32>()[2..5], &[202, 303, 404]);
}

#[test]
fn invalid_blits_restore_encoder_and_allow_valid_work() {
    let Some(ctx) = gpu_or_skip().map(Arc::new) else {
        return;
    };
    let src = MetalBuffer::from_slice(ctx.device(), &[2_u32, 4, 8, 16, 32, 64]).expect("src");
    let dst = MetalBuffer::empty(ctx.device(), src.length()).expect("dst");
    let mut batch = CommandBatch::new(&ctx).expect("CommandBatch::new");

    let out_of_bounds = batch.blit_buffer_copies([BufferCopyRequest {
        source: &src,
        source_offset: 5 * size_of::<u32>(),
        destination: &dst,
        destination_offset: size_of::<u32>(),
        size: 2 * size_of::<u32>(),
    }]);
    assert!(matches!(out_of_bounds, Err(Error::CommandBuffer(_))));

    let overflow = batch.blit_buffer_copies([BufferCopyRequest {
        source: &src,
        source_offset: usize::MAX - 1,
        destination: &dst,
        destination_offset: 0,
        size: 4,
    }]);
    assert!(matches!(overflow, Err(Error::CommandBuffer(_))));

    batch
        .blit_buffer_copies([BufferCopyRequest {
            source: &src,
            source_offset: 2 * size_of::<u32>(),
            destination: &dst,
            destination_offset: size_of::<u32>(),
            size: 3 * size_of::<u32>(),
        }])
        .expect("valid copy after rejected copies");
    batch.commit_and_wait().expect("valid copy wait");
    assert_eq!(&dst.as_slice::<u32>()[1..4], &[8, 16, 32]);
}
