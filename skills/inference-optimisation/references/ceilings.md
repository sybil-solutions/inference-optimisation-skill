# Computing theoretical ceilings

Measure the hardware; don't trust the spec sheet. A device-to-device copy of a large buffer gives realistic bandwidth. A large square FP16/BF16 GEMM (4096³ and 8192³) gives realistic compute; the larger one shows thermal and clock drop. Practical ceilings below assume about 100% of *measured* bandwidth and about 50% of measured peak FLOPs.

## Decode (memory bound)

Bytes read per target forward at short context:

```
B_target = dense_weight_bytes                    # attention / linear-attention, shared experts, router, norms
         + expert_bytes × (top_k / n_experts)    # MoE: only routed experts are read
         + lm_head_bytes                         # often 10–25% of the total on large vocabularies!
         + kv_bytes_per_token × context          # grows with context; dominates at long context
         + recurrent_state_bytes × 2             # SSM / GDN state read + write, if hybrid
```

Use the checkpoint's tensor sizes, which already include quantization. Don't use parameter count × bits.

With speculative decoding (k drafts, accepted length `a` tokens per step):

```
verify_bytes ≈ dense + lm_head + expert_bytes × (1 − (1 − top_k/n_experts)^(k+1))   # union of routed experts
draft_bytes  ≈ k × (draft_layer_bytes + draft_lm_head_bytes)                         # prune the draft head!
bytes_per_emitted_token = (verify_bytes + draft_bytes) / a
c1_ceiling = measured_bandwidth / bytes_per_emitted_token
```

For a concurrency of N, dense weights are read once per step for all streams; experts follow the union of all rows' routing; KV is per stream. The aggregate ceiling therefore grows almost linearly with N until the expert union and per-stream KV dominate the bytes.

## Prefill (compute bound)

```
flops_per_token ≈ 2 × active_linear_params
                + 4 × n_full_attn_layers × n_heads × head_dim × position   (causal: average position = L/2)
                + linear-attention / SSM terms
prefill_ceiling(L) ≈ 0.5 × measured_peak_flops / mean_flops_per_token(L)
```

Int8/FP8 tensor cores double the peak for the parts that use them. For MoE, small per-expert row counts at a given chunk size lower GEMM efficiency; that is usually the biggest gap.

## Three ways the ceiling comes out too high

These cost one hill-climb a wrong headline: a looped MoE on an RTX PRO 6000 (Inkling 2 Mini, Basalt engine) first read as "55% of the c1 ceiling, 5% of the c8 ceiling". Corrected, it was close to normal on both.

- **Looped (recurrent-depth) models read some weights more than once per token.** Count layer *visits*, not layers. Inkling 2 Mini has 24 physical layers; the 12-layer core runs 3 times, so a token makes 48 visits and reads the dense and expert bytes about twice. Leaving that out doubled the c1 ceiling (784 to about 390 tok/s).
- **Spec-sheet FP4/FP8 peaks often assume 2:4 structured sparsity.** Dense prompt-processing GEMMs get about half. Use the dense figure, or measure a dense GEMM at that precision. On the same box, using the sparse peak and the physical active parameters (not the looped effective ones) put the 32K prefill ceiling at 99K tok/s; the corrected figure was about 34K, against 16.4K measured.
- **Engines cap the rows in one decode step.** If the cap is R rows and there are N streams, each stream gets about R/N rows, so drafting stops when N reaches R. Check the engine for a compile-time window limit (Basalt: `kDecodeRows = 8`; its MMVQ products take at most 8 columns). Above that point, model cN with no drafts. The server log can show it: tokens kept per step fell from 1.5 to 3.3 per request solo to exactly 1.0 for every request at 8 streams.

Also: the union-of-experts formula treats draft rows as independent. Rows of one conversation share most of their experts, so with many drafts it can predict *less* than you measure. When it does, report the no-draft row as the bound.

## TTFT, startup, KV

- **Short-prompt TTFT floor:** one forward pass (bytes ÷ bandwidth) plus tokenize and schedule. That is typically 30–60 ms on a laptop and a few ms on datacenter GPUs.
- **Startup floor:** weight bytes ÷ disk or page-cache speed, plus allocation. That is about 10 s when kernel JIT, autotune and graph-capture caches persist.
- **KV:** `bytes/token = n_kv_layers × 2 × n_kv_heads × head_dim × bytes_per_elem` (+ the draft layer's own KV). Then `max_full_length_streams = (memory_cap − weights − state − runtime) / (bytes/token × context)`.

## Worked example: Ornith-1.5-35B-A3B (Qwen3.5-MoE hybrid), EXL3 4 bpw, Intel Arc B390 iGPU

**Hardware (measured):**
- 115 GB/s copy bandwidth on 64 GB of LPDDR5X (153.6 GB/s spec).
- 35.8 TFLOPS FP16 at 4096³, dropping to 20 at 8192³.

**Model:** 40 layers: 30 GDN linear-attention plus 10 full-attention. 256 experts, top-8.

| Component | GB / target token |
|---|---:|
| Routed experts: 16.26 GB × 8/256 | 0.508 |
| Dense: GDN 0.515 + full attention 0.137 + shared expert 0.064 + router 0.042 | 0.758 |
| LM head (6-bit, 248K vocab) | 0.382 |
| **Total** | **1.648** |

**Single-user decode ceilings (practical, at 115 GB/s):**

| Mode | GB per emitted token | Ceiling |
|---|---:|---:|
| No speculation | 1.68 | 68 tok/s |
| 1 MTP draft, full draft head, a = 1.85 | 1.38 | 83 tok/s |
| 1 draft, pruned 32K-token draft head | 1.19 | 97 tok/s |
| 2 drafts, a ≈ 2.5 | 1.11 | ~103 tok/s |

The pruned draft head, about 8× fewer head bytes, was worth more than an extra draft.

**What happened in practice:**

- **c1 decode:** it reached ~53 tok/s, about 55% of the ceiling. The model's MTP head accepted only ~14% at depth 2, so the 2-draft row was unreachable. The remaining gap was the quantized GEMV kernels, at ~76 GB/s effective in the verify graph.
- **Prefill:** about 670 tok/s practical at 256K versus 628 achieved. Long-context prefill was close to its ceiling.
- **KV cache:** FP8 KV is 10,240 B/token, so 2.7 GB per 256K stream. Four full-length streams fit in about 11 GB under a 46 GiB cap.

`scripts/ceiling.py` does this arithmetic from a small JSON of component sizes.
