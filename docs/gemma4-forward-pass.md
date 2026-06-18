# Gemma 4 E2B forward pass — authoritative spec

Transcribed from llama.cpp `src/models/gemma4.cpp` (architecture `gemma4`), the
graph that consumes this exact GGUF. This is the reference the engine's forward
pass must match.

## Dimensions (from GGUF metadata)

- `n_embd = 1536`, `n_head = 8`, `n_head_kv = 1`, `n_layer = 35`
- `n_embd_per_layer = 256` (PLE width)
- Per-layer attention type from `sliding_window_pattern = [T,T,T,T,F]×7` (T = SWA, F = full):
  - **SWA layers**: `head_dim = 256`, `rope_freq_base = 10000`, `n_rot = 256`
  - **full layers**: `head_dim = 512`, `rope_freq_base = 1000000`, `n_rot = 512`, plus `rope_freqs` (proportional RoPE)
- `shared_kv_layers = 20` → layers `[15..35)` reuse an earlier layer's KV
- FFN width: layers 0–14 → 6144, layers 15–34 → 12288 (double-wide)
- `f_attention_scale = 1.0` (Gemma4 uses **no** `1/sqrt(head_dim)` softmax scaling)
- `final_logit_softcapping = 30`

## KV sharing

A shared layer (`il >= 15`) computes **only Q**; it reuses the K/V cache of the
**last non-shared layer with the same attention type**:
- shared SWA layer  → source layer 13 (last SWA before 15)
- shared full layer → source layer 14 (last full before 15)

(`Gemma4QATConfig::kv_source_layer` implements this.)

## Embedding + per-layer inputs (once per token)

```
h        = tok_embd[token]                         # [1536]
h        = h * sqrt(n_embd)                         # ≈ ×39.19

ple_tok  = per_layer_tok_embd[token]               # [256*35] -> reshape [256, 35]
ple_tok *= sqrt(n_embd_per_layer)                   # ×16

proj     = per_layer_model_proj @ h                # -> [256*35] -> reshape [256,35]
proj    *= 1/sqrt(n_embd)                            # ×1/39.19
proj     = rms_norm(proj, per_layer_proj_norm)      # per 256-row
per_layer_input = (proj + ple_tok) * (1/sqrt(2))    # [256, 35]; row l feeds layer l
```

## Per layer `il`

```
cur = rms_norm(h, attn_norm)                         # input_layernorm

# Q (all layers)
Q = wq @ cur                                          # [n_head * head_dim]
Q = reshape(Q, [head_dim, n_head])
Q = rms_norm(Q, attn_q_norm)                          # weight size head_dim
Q = rope(Q, freq_base, n_rot, rope_freqs?)            # rope_freqs only on full layers

if il < 15:                                           # has own KV
    K = wk @ cur ;  V = (wv ? wv@cur : K)
    K = reshape(K,[head_dim,n_head_kv]); V = reshape(V,[head_dim,n_head_kv])
    K = rms_norm(K, attn_k_norm)
    V = rms_norm(V, eps)                              # NOTE: rms_norm with NO weight
    K = rope(K, freq_base, n_rot, rope_freqs?)
    attn = attention(Q, K, V, scale = 1.0)            # writes K,V to cache[il]
else:                                                 # shared
    attn = attention(Q, cache[source].K, cache[source].V, scale = 1.0)
cur = wo @ attn

cur      = rms_norm(cur, attn_post_norm)              # post_attention_norm
attn_out = cur + h                                    # residual #1

cur      = rms_norm(attn_out, ffn_norm)               # pre_feedforward_layernorm
cur      = ffn_down @ ( gelu(ffn_gate @ cur) * (ffn_up @ cur) )   # GeGLU
cur      = rms_norm(cur, ffn_post_norm)               # post_feedforward_layernorm
cur      = cur + attn_out                             # residual #2

# per-layer input injection
pe_in = cur
g     = gelu(per_layer_inp_gate @ cur)                # [1536]->[256]
g     = g * per_layer_input[il]                       # elementwise [256]
g     = per_layer_proj @ g                            # [256]->[1536]
g     = rms_norm(g, per_layer_post_norm)              # "post_norm.weight"
cur   = pe_in + g                                     # residual #3

if out_scale: cur = cur * out_scale[il]               # layer_scalar [1]
h = cur
```

## Final

```
h      = rms_norm(h, output_norm)
logits = token_embd @ h                                # tied embeddings
logits = 30 * tanh(logits / 30)                        # final_logit_softcapping
```

## Implementation status

All of the above is implemented in `layer.rs` / `pipeline.rs` and the model
**generates correct, coherent text**:

1. ✅ Embedding scale ×√n_embd.
2. ✅ Per-layer head_dim 256/512 + per-layer RoPE base (10000/1000000).
3. ✅ NEOX rotation (was interleaved — the bug that scrambled every layer).
4. ✅ Attention softmax scale = 1.0 (the flash-attention kernel already does this).
5. ✅ V rms_norm (no weight) before caching.
6. ✅ KV sharing for layers 15–34 (caches restructured into the pipeline).
7. ✅ Norm order: `attn_post_norm` → residual; `ffn_norm`/`ffn_post_norm` around FFN.
8. ✅ Per-layer-input (PLE) injection + `per_layer_post_norm` + `out_scale`.
9. ✅ Final logit soft-capping (via the sampler's `logit_softcap`).

10. ✅ **Sliding-window masking** — SWA layers enforce the 512-position window
    (`q − 512 < k ≤ q`) in both the decode and batched-prefill attention kernels.
11. ✅ **Proportional RoPE** — full layers apply the HF `rope_type: "proportional"`
    `rope_freqs` schedule (only the first `0.25·head_dim/2` NEOX pairs rotate).

### Root-cause fixes that made it work

The initial output was all-NaN/garbage. Root causes, now fixed:
- **GGUF reader**: data section must be 32-byte aligned, and tensor offsets are
  *relative* to it — the reader used unaligned offsets and never added the base.
- **Q4_0 dequant**: 18-byte blocks (not 20), `(nibble-8)*d` (the `-8` was
  missing), low-nibbles-then-high layout (not interleaved).
- **Q8_0 dequant**: 34-byte blocks (not 36), signed int8 (was read unsigned).
- **RoPE**: NEOX half-split pairing (was GPT-J interleaved).
