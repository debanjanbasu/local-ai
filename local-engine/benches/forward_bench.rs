//! Benchmark scaffold for forward pass timing.
//!
//! Run with:
//!   `GEMMA4_MODEL_PATH=/path/to/model cargo bench -p gemma4-engine`
//!
//! Current scope: this bench is intended for `gemma-4-e2b-it` only.
//! The model path must be set via environment variable.

#![allow(clippy::expect_used)]

use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use local_engine::{Engine, EngineConfig, GenerateParams};

fn bench_forward_one_token(c: &mut Criterion) {
    let Ok(model_path) = std::env::var("GEMMA4_MODEL_PATH") else {
        eprintln!("GEMMA4_MODEL_PATH not set, skipping benchmark");
        return;
    };

    let config = EngineConfig {
        model_dir: PathBuf::from(&model_path),
        max_context_length: 8192,
    };
    let mut engine = Engine::new(&config).expect("load engine");

    // Warm up: generate 5 tokens
    let warmup_params = GenerateParams {
        temperature: 0.0,
        max_tokens: 5,
        ..GenerateParams::default()
    };
    let _ = engine.generate("Hello", &warmup_params);
    engine.reset();

    // Benchmark single token generation at position 5
    let params = GenerateParams {
        temperature: 0.0,
        max_tokens: 1,
        ..GenerateParams::default()
    };

    c.bench_function("forward_one_token", |b| {
        b.iter(|| {
            engine.reset();
            let _ = engine.generate("The capital of France is", &params);
        });
    });
}

criterion_group!(benches, bench_forward_one_token);
criterion_main!(benches);
