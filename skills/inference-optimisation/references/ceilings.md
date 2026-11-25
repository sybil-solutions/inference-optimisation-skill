# Computing theoretical ceilings

Measure the hardware, not the spec sheet: a large device-to-device copy gives bandwidth; FP16/BF16 GEMMs at 4096³ and 8192³ give compute (the larger shows thermal and clock drop). Practical ceilings assume ~100% of *measured* bandwidth and ~50% of measured peak FLOPs.

## Decode (memory bound)

Bytes read per target forward at short context:

```
B_target = dense_weight_bytes                    # attention / linear-attention, shared experts, router, norms
         + expert_bytes × (top_k / n_experts)    # MoE: only routed experts are read
         + lm_head_bytes                         # often 10–25% of the total on large vocabularies!
         + kv_bytes_per_token × context          # grows with context; dominates at long context
         + recurrent_state_bytes × 2             # SSM / GDN state read + write, if hybrid
```

Use the checkpoint's tensor sizes, not parameter count × bits.

With speculative decoding (k drafts, accepted length `a` tokens per step):

```
verify_bytes ≈ dense + lm_head + expert_bytes × (1 − (1 − top_k/n_experts)^(k+1))   # union of routed experts
draft_bytes  ≈ k × (draft_layer_bytes + draft_lm_head_bytes)                         # prune the draft head!
bytes_per_emitted_token = (verify_bytes + draft_bytes) / a
c1_ceiling = measured_bandwidth / bytes_per_emitted_token
```

At concurrency N, dense weights are read once a step for all streams, experts follow the union of routing, KV is per stream: the aggregate grows almost linearly with N until expert union and KV dominate.

## Prefill (compute bound)

```
flops_per_token ≈ 2 × active_linear_params
                + 4 × n_full_attn_layers × n_heads × head_dim × position   (causal: average position = L/2)
                + linear-attention / SSM terms
prefill_ceiling(L) ≈ 0.5 × measured_peak_flops / mean_flops_per_token(L)
```

Int8/FP8 tensor cores double the peak where used. For MoE, few rows per expert at a given chunk size lower GEMM efficiency; usually the biggest gap.

## Batched serving: the per-component roofline

One total hides where time goes. For a batched step of R rows, give each component (dense, routed experts, attention at real context lengths, recurrent state, LM head, sampler, host) a floor:

```
floor_c(R) = bytes_c(R) / bandwidth + flops_c(R) / peak_flops        # no overlap
floor_c(R) = max(bytes_c(R) / bandwidth, flops_c(R) / peak_flops)     # full overlap
```

`bytes_c` counts shared weights once a step and per-row data (KV, state, distinct experts) per row. Time each component (profiler, stage stamps), report `floor_c / measured_c` for each, and attribute wall time outside any component (gaps, host stalls, prompt rows) separately.

Illustrative example (two GPUs, a pipelined MoE pool serving 32 coding agents): the pool ran at 23–30% of a 735–980 tok/s roofline. Only ~40% of a stage's wall was decode rows (17% fixed per step, 20% prompt rows, 21% gaps), and a decode step cost 1.8–2.4× its floor. The capacity plan priced each extra row at its new bytes and forgot its FLOPs (~0.3 ms a row per stage at the measured GEMM rate). No single item was worth 2×: the per-component table says so, the total does not.

## TTFT, startup, KV

- **Short-prompt TTFT floor:** one forward pass plus tokenize and schedule: ~30–60 ms on a laptop, a few ms on datacenter GPUs.
- **Startup floor:** weight bytes ÷ disk or page-cache speed, plus allocation: ~10 s with JIT, autotune and graph caches persisted.
- **KV:** `bytes/token = n_kv_layers × 2 × n_kv_heads × head_dim × bytes_per_elem` (+ the draft layer's KV). `max_full_length_streams = (memory_cap − weights − state − runtime) / (bytes/token × context)`.

## Worked example: Ornith-1.5-35B-A3B (Qwen3.5-MoE hybrid), EXL3 4 bpw, Intel Arc B390 iGPU

**Hardware (measured):** 115 GB/s copy bandwidth on 64 GB LPDDR5X (153.6 spec); 35.8 TFLOPS FP16 at 4096³, 20 at 8192³.

**Model:** 40 layers (30 GDN linear-attention, 10 full-attention), 256 experts, top-8.

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

The pruned draft head (~8× fewer head bytes) was worth more than an extra draft.

**In practice:** c1 reached ~53 tok/s, ~55% of the ceiling; MTP accepted ~14% at depth 2 (2-draft row unreachable), and the rest was GEMV kernels at ~76 GB/s effective. Prefill at 256K: ~670 tok/s ceiling vs 628 achieved. FP8 KV is 10,240 B/token (2.7 GB per 256K stream): four full-length streams fit in ~11 GB under a 46 GiB cap.

`scripts/ceiling.py` does this arithmetic from a small JSON of component sizes.
