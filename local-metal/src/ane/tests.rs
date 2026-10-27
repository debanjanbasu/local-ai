use std::time::{Duration, Instant};

use super::{Handoff, Program, Surface, SurfaceLayout, unavailable};
use crate::batch::CommandBatch;
use crate::context::MetalContext;

const CHANNELS: u32 = 256;
const ROWS: u32 = 64;
const TOKENS: u32 = 64;

/// `y[r][t] = sum_c (w[r][c] / 128) (x[c][t] / 128)` in one ANE function.
fn program(x: SurfaceLayout, w: SurfaceLayout, y: SurfaceLayout) -> String {
    let unit = "fp16(0x1p-7)";
    format!(
        "program(1.3)\n{{\n    func main<ios18>({} x, {} w) {{\n\
        tensor<int8, [1, 1, {CHANNELS}, {TOKENS}]> x_t = tensor_buffer_to_tensor<ios17>(input = x);\n\
        tensor<int8, [1, 1, {ROWS}, {CHANNELS}]> w_t = tensor_buffer_to_tensor<ios17>(input = w);\n\
        tensor<fp16, [1, 1, {CHANNELS}, {TOKENS}]> x_d = dequantize(input = x_t, scale = {unit});\n\
        tensor<fp16, [1, 1, {ROWS}, {CHANNELS}]> w_d = dequantize(input = w_t, scale = {unit});\n\
        tensor<fp16, [1, 1, {ROWS}, {TOKENS}]> yt = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w_d, y = x_d);\n\
        {} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}));\n\
    }} -> (y);\n}}\n",
        x.buffer_type(CHANNELS, TOKENS),
        w.buffer_type(ROWS, CHANNELS),
        y.buffer_type(ROWS, TOKENS),
        y.strides(),
    )
}

/// An empty MIL blob (storage version 2, no records).
fn empty_blob() -> Vec<u8> {
    let mut blob = vec![0_u8; 64];
    blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
    blob
}

struct Fixture {
    context: MetalContext,
    x: Surface,
    w: Surface,
    y: Surface,
    program: Program,
    expected: Vec<f32>,
}

fn fixture() -> Option<Fixture> {
    if let Some(reason) = unavailable() {
        eprintln!("skipping Neural Engine test: {reason}");
        return None;
    }
    let context = MetalContext::new().expect("context");
    let x = Surface::new(&context, SurfaceLayout::new(CHANNELS, TOKENS, false)).expect("x");
    let w = Surface::new(&context, SurfaceLayout::new(ROWS, CHANNELS, false)).expect("w");
    let y = Surface::new(&context, SurfaceLayout::new(ROWS, TOKENS, true)).expect("y");
    let code = |seed: u32| i32::from(((seed.wrapping_mul(2_654_435_761) >> 24) % 255) as u8) - 127;
    let mut xs = vec![0_i8; x.layout().bytes];
    let mut ws = vec![0_i8; w.layout().bytes];
    for c in 0..CHANNELS as usize {
        for t in 0..TOKENS as usize {
            xs[c * x.layout().stride + t] = code((c * 977 + t) as u32) as i8;
        }
    }
    for r in 0..ROWS as usize {
        for c in 0..CHANNELS as usize {
            ws[r * w.layout().stride + c] = code((r * 131 + c * 7 + 5) as u32) as i8;
        }
    }
    x.buffer().copy_from_bytes(bytemuck::cast_slice(&xs), 0);
    w.buffer().copy_from_bytes(bytemuck::cast_slice(&ws), 0);
    let expected = (0..ROWS as usize)
        .flat_map(|r| {
            let (xs, ws, x, w) = (&xs, &ws, &x, &w);
            (0..TOKENS as usize).map(move |t| {
                (0..CHANNELS as usize)
                    .map(|c| {
                        f32::from(ws[r * w.layout().stride + c])
                            * f32::from(xs[c * x.layout().stride + t])
                    })
                    .sum::<f32>()
                    / 16_384.0
            })
        })
        .collect();
    let source = program(x.layout(), w.layout(), y.layout());
    let program = Program::compile(
        &source,
        &empty_blob(),
        "local-ai-ane-test",
        Duration::from_secs(120),
    )
    .expect("compile");
    Some(Fixture {
        context,
        x,
        w,
        y,
        program,
        expected,
    })
}

#[expect(unsafe_code)]
fn output(fixture: &Fixture) -> Vec<f32> {
    // SAFETY: no GPU or ANE work on the output is in flight.
    let halves = unsafe { fixture.y.buffer().as_slice::<half::f16>() };
    let stride = fixture.y.layout().stride / 2;
    (0..ROWS as usize)
        .flat_map(|r| (0..TOKENS as usize).map(move |t| halves[r * stride + t].to_f32()))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 2e-3_f32.mul_add(e.abs(), 2e-3),
            "element {index}: {a} against {e}"
        );
    }
}

#[test]
fn synchronous_evaluation_matches_cpu_reference() {
    let Some(fixture) = fixture() else { return };
    let procedure = fixture.program.procedure("main").expect("main");
    let inputs = fixture.program.inputs(procedure).to_vec();
    let surfaces = inputs
        .iter()
        .map(|name| if name == "x" { &fixture.x } else { &fixture.w })
        .collect::<Vec<_>>();
    let binding = fixture
        .program
        .bind(procedure, &surfaces, &fixture.y)
        .expect("bind");
    fixture.y.buffer().clear();
    binding.evaluate().expect("evaluate");
    assert_close(&output(&fixture), &fixture.expected);
}

#[test]
fn shared_events_order_the_gpu_and_the_ane_and_the_watchdog_releases_the_gpu() {
    let Some(fixture) = fixture() else { return };
    let procedure = fixture.program.procedure("main").expect("main");
    let surfaces = fixture
        .program
        .inputs(procedure)
        .iter()
        .map(|name| if name == "x" { &fixture.x } else { &fixture.w })
        .collect::<Vec<_>>();
    let binding = fixture
        .program
        .bind(procedure, &surfaces, &fixture.y)
        .expect("bind");
    let handoff = Handoff::new(&fixture.context, Duration::from_millis(500)).expect("handoff");
    fixture.y.buffer().clear();
    for _ in 0..3 {
        let (ready, done) = (handoff.next_ready(), handoff.next_done());
        handoff.queue(&binding, ready, done, false);
        let mut batch = CommandBatch::new(&fixture.context).expect("batch");
        batch
            .signal_event(handoff.ready_event(), ready)
            .expect("signal");
        batch.wait_event(handoff.done_event(), done).expect("wait");
        batch.commit_and_wait().expect("gpu");
        assert!(handoff.finish(), "{:?}", handoff.retired());
        assert_close(&output(&fixture), &fixture.expected);
    }
    // A job that never starts: the watchdog retires the handoff and the
    // GPU's wait passes within about one bound.
    let (ready, done) = (handoff.next_ready(), handoff.next_done());
    handoff.queue(&binding, ready, done, true);
    let started = Instant::now();
    let mut batch = CommandBatch::new(&fixture.context).expect("batch");
    batch
        .signal_event(handoff.ready_event(), ready)
        .expect("signal");
    batch.wait_event(handoff.done_event(), done).expect("wait");
    batch.commit_and_wait().expect("gpu");
    assert!(!handoff.finish());
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(handoff.retired().is_some());
    // After retirement, later values are released at once.
    let (ready, done) = (handoff.next_ready(), handoff.next_done());
    handoff.queue(&binding, ready, done, false);
    let mut batch = CommandBatch::new(&fixture.context).expect("batch");
    batch.wait_event(handoff.done_event(), done).expect("wait");
    batch.commit_and_wait().expect("gpu");
    assert!(!handoff.finish());
}
