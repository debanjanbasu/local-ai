# local-engine

Synchronous, network-free Rust library for native Ternary Bonsai 2 27B
inference on Apple Silicon. It provides model discovery, Metal execution,
prompt caching, lossless speculation, tokenization, and sampling.

```rust,no_run
let mut engine = local_engine::Engine::open()?;
let answer = engine.chat("What is 17 * 23?")?;
println!("{}", answer.content);
# Ok::<(), local_engine::Error>(())
```

`Engine` is `Send` but not `Sync`. Use `Engine::into_handle()` for a cloneable
`EngineHandle`; requests return cancellable `EventStream`s that are both a
blocking `Iterator` and a `Stream`. The crate depends on `tokio` (features `rt`
and `sync`) for its event transport and on `futures-core` for the `Stream`
implementation, and has no HTTP server dependency. See the repository
[README](../README.md#library-use) and [technical reference](../docs/BONSAI.md).
