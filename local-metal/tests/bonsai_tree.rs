#![expect(unsafe_code, clippy::expect_used, clippy::too_many_lines)]

use half::f16;
use local_metal::{
    batch::CommandBatch,
    bonsai_ops::{BonsaiOps, GdnStateFormat, TREE_DESCRIPTOR_WORDS, TREE_MASKS, TREE_PATH},
    buffer::MetalBuffer,
    context::MetalContext,
    shaders::ShaderLibrary,
};

const QKV: usize = 10240;
const OUTPUT: usize = 6144;
const STATE: usize = 48 * 128 * 128;
/// A tree whose rows mix rows that follow the row before them (computed in
/// sequence) and rows that branch (rebuilt by replaying their ancestors).
const PARENTS: [i32; 8] = [-1, 0, 0, 1, 2, 4, 2, 4];

/// `None` when this machine exposes no Metal device, so GPU tests skip
/// instead of failing. Any other failure is still a failure: a missing GPU
/// is an environment, a broken context is a bug.
fn setup() -> Option<(MetalContext, BonsaiOps)> {
    let context = MetalContext::new();
    if matches!(context, Err(local_metal::Error::NoMetalDevice)) {
        eprintln!("skipping GPU test: this machine has no Metal device");
        return None;
    }
    let context = context.expect("Metal context failed for a reason other than a missing device");
    let shaders = ShaderLibrary::new(context.device()).expect("shaders");
    let ops = BonsaiOps::new(&context, &shaders).expect("ops");
    Some((context, ops))
}

fn signal(index: usize, salt: usize) -> f32 {
    ((index * 17 + index / 29 * 11 + salt * 37) % 251) as f32 / 131.0 - 0.91
}

fn floats(context: &MetalContext, values: &[f32]) -> MetalBuffer {
    MetalBuffer::from_slice(context.device(), values).expect("float buffer")
}

fn read(buffer: &MetalBuffer) -> Vec<u32> {
    // SAFETY: every batch below is waited for before its buffers are read.
    unsafe { buffer.as_slice::<u32>() }.to_vec()
}

/// The rows from the root to `row`.
fn path(row: usize) -> Vec<usize> {
    let mut path = vec![row];
    let mut row = row;
    while let Ok(parent) = usize::try_from(PARENTS[row]) {
        path.push(parent);
        row = parent;
    }
    path.reverse();
    path
}

fn rows_of(values: &[f32], width: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|&row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}

/// Every row of a tree verify's convolution and recurrence is bitwise what
/// the chain kernels compute for that row's path alone, whether the row
/// continues the row before it or replays its ancestors; the committed
/// history and state are only read.
#[test]
fn tree_rows_convolve_and_recur_exactly_as_their_paths_do() {
    let Some((context, ops)) = setup() else {
        return;
    };
    let rows = PARENTS.len();
    let raw = (0..rows * QKV)
        .map(|i| signal(i, 3) * if i % QKV < 4096 { 0.4 } else { 0.9 })
        .collect::<Vec<_>>();
    let alpha = (0..rows * 48)
        .map(|i| signal(i, 7) * 2.0)
        .collect::<Vec<_>>();
    let raw_beta = (0..rows * 48).map(|i| signal(i, 11)).collect::<Vec<_>>();
    let weights = floats(
        &context,
        &(0..QKV * 4).map(|i| signal(i, 13)).collect::<Vec<_>>(),
    );
    let history_values = (0..QKV * 3).map(|i| signal(i, 17)).collect::<Vec<_>>();
    let history = floats(&context, &history_values);
    let a = floats(
        &context,
        &(0..48)
            .map(|i| -0.5 - signal(i, 19).abs())
            .collect::<Vec<_>>(),
    );
    let dt = floats(
        &context,
        &(0..48).map(|i| signal(i, 23)).collect::<Vec<_>>(),
    );
    let mut descriptor = [0i32; TREE_DESCRIPTOR_WORDS];
    let mut masks = [0u32; 8];
    for (row, &parent) in PARENTS.iter().enumerate() {
        descriptor[row] = parent;
        masks[row] = usize::try_from(parent).map_or(0, |parent| masks[parent]) | 1 << row;
        descriptor[TREE_MASKS + row] = masks[row].cast_signed();
    }
    let tree = MetalBuffer::from_slice(context.device(), &descriptor).expect("tree");
    for format in [GdnStateFormat::F16, GdnStateFormat::F32] {
        let state_values = (0..STATE).map(|i| signal(i, 29) * 0.05).collect::<Vec<_>>();
        let state = match format {
            GdnStateFormat::F16 => MetalBuffer::from_slice(
                context.device(),
                &state_values
                    .iter()
                    .map(|&v| f16::from_f32(v))
                    .collect::<Vec<_>>(),
            ),
            _ => MetalBuffer::from_slice(context.device(), &state_values),
        }
        .expect("state");
        let state_bits = read(&state);
        let scratch = |count| floats(&context, &vec![f32::NAN; count]);
        let (convolved, decay, beta, output) = (
            scratch(rows * QKV),
            scratch(rows * 48),
            scratch(rows * 48),
            scratch(rows * OUTPUT),
        );
        let (raw_buffer, alpha_buffer, beta_buffer) = (
            floats(&context, &raw),
            floats(&context, &alpha),
            floats(&context, &raw_beta),
        );
        let mut batch = CommandBatch::new(&context).expect("batch");
        ops.conv_l2_decay_tree(
            &mut batch,
            &raw_buffer,
            &weights,
            &history,
            &convolved,
            1e-6,
            [&a, &alpha_buffer, &dt, &beta_buffer],
            &decay,
            &beta,
            &tree,
            rows as u32,
        )
        .expect("tree convolution");
        ops.gdn_tree(
            &mut batch,
            format,
            &convolved,
            &decay,
            &beta,
            &state,
            &output,
            &tree,
            rows as u32,
        )
        .expect("tree recurrence");
        batch.commit_and_wait().expect("tree pass");
        assert_eq!(read(&history), bits(&history_values));
        assert_eq!(read(&state), state_bits);
        let (tree_convolved, tree_output) = (read(&convolved), read(&output));

        for row in 0..rows {
            let path = path(row);
            let length = path.len();
            let (chain_convolved, chain_decay, chain_beta, chain_output) = (
                scratch(length * QKV),
                scratch(length * 48),
                scratch(length * 48),
                scratch(length * OUTPUT),
            );
            let (final_history, final_state) = (
                scratch(QKV * 3),
                MetalBuffer::empty(context.device(), format.state_bytes()).expect("state"),
            );
            let (path_raw, path_alpha, path_beta) = (
                floats(&context, &rows_of(&raw, QKV, &path)),
                floats(&context, &rows_of(&alpha, 48, &path)),
                floats(&context, &rows_of(&raw_beta, 48, &path)),
            );
            let mut batch = CommandBatch::new(&context).expect("batch");
            ops.conv_l2_decay_into(
                &mut batch,
                &path_raw,
                &weights,
                [&history, &final_history],
                &chain_convolved,
                1e-6,
                [&a, &path_alpha, &dt, &path_beta],
                &chain_decay,
                &chain_beta,
                length as u32,
            )
            .expect("chain convolution");
            ops.gdn_sequence_into(
                &mut batch,
                format,
                &chain_convolved,
                &chain_decay,
                &chain_beta,
                &state,
                &final_state,
                &chain_output,
                length as u32,
            )
            .expect("chain recurrence");
            batch.commit_and_wait().expect("chain pass");
            let last = length - 1;
            assert_eq!(
                tree_convolved[row * QKV..(row + 1) * QKV],
                read(&chain_convolved)[last * QKV..length * QKV],
                "{format:?} row {row}: convolution"
            );
            assert_eq!(
                tree_output[row * OUTPUT..(row + 1) * OUTPUT],
                read(&chain_output)[last * OUTPUT..length * OUTPUT],
                "{format:?} row {row}: recurrence"
            );
        }
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A commit's in-place gather moves exactly the path's rows of every part
/// to consecutive rows from the part's offset, and the tap copies wrap
/// around their ring.
#[test]
fn row_gathers_and_ring_copies_move_whole_rows() {
    let Some((context, ops)) = setup() else {
        return;
    };
    let path = [0usize, 2, 3, 5];
    let mut words = [0i32; TREE_DESCRIPTOR_WORDS];
    for (word, &row) in words[TREE_PATH..].iter_mut().zip(&path) {
        *word = (row as u32).cast_signed();
    }
    let descriptor = MetalBuffer::from_slice(context.device(), &words).expect("descriptor");
    // Two parts with different row widths, the first from a row offset.
    let widths = [17usize, 3];
    let parts = widths.map(|width| (0..10 * width).map(|i| i as f32 + 0.25).collect::<Vec<_>>());
    let buffers = parts.each_ref().map(|values| floats(&context, values));
    let offset = 2;
    let mut batch = CommandBatch::new(&context).expect("batch");
    ops.gather_rows(
        &mut batch,
        &[
            (&buffers[0], offset * widths[0] * 4, widths[0] * 4),
            (&buffers[1], 0, widths[1] * 4),
        ],
        (&descriptor, TREE_PATH * 4),
        path.len(),
    )
    .expect("gather");
    batch.commit_and_wait().expect("gather pass");
    for ((values, buffer), (width, first)) in parts
        .iter()
        .zip(&buffers)
        .zip([(widths[0], offset), (widths[1], 0)])
    {
        let mut expected = values.clone();
        for (index, &row) in path.iter().enumerate() {
            expected.copy_within(
                (first + row) * width..(first + row + 1) * width,
                (first + index) * width,
            );
        }
        assert_eq!(read(buffer), bits(&expected));
    }

    // Three rows of eight floats to a four-row ring from its row 3.
    let source = floats(&context, &(0..24).map(|i| i as f32).collect::<Vec<_>>());
    let ring = floats(&context, &[f32::NAN; 4 * 12]);
    let mut batch = CommandBatch::new(&context).expect("batch");
    ops.copy_rows(&mut batch, (&source, 0), (&ring, 16), 8, 3, [8, 12], 3, 4)
        .expect("ring copy");
    batch.commit_and_wait().expect("ring pass");
    // SAFETY: the batch above was waited for.
    let ring = unsafe { ring.as_slice::<f32>() }.to_vec();
    for (row, ring_row) in [(0usize, 3usize), (1, 0), (2, 1)] {
        let copied = &ring[ring_row * 12 + 4..ring_row * 12 + 12];
        assert_eq!(
            copied,
            (row * 8..row * 8 + 8).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert!(
            ring[ring_row * 12..ring_row * 12 + 4]
                .iter()
                .all(|v| v.is_nan())
        );
    }
    assert!(ring[2 * 12..3 * 12].iter().all(|v| v.is_nan()));
}
