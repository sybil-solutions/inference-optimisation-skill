# Ceilings - Qwen3.8-Flash-Next Uncensored W4A16, TP2 on 2x A100-SXM4-80GB

Computed 2026-11-21 with `scripts/ceiling.py flash-ceiling.json`.
Bandwidth and FLOPs are spec-sheet values (2x 2039 GB/s, 2x 312 TFLOPS bf16), not measured yet.
Measure them during the first window and rerun.

## Bytes per decoded token (from safetensors headers)

| Component | GB on disk | Read per token at c1 |
| --- | --- | --- |
| Routed experts (512, top-10, W4A16) | 67.95 | 1.33 |
| Dense: linear attention 4.17, hyper-connections etc. 1.41, full attention 1.23, shared expert 0.47 | 7.28 | 7.28 |
| LM head (bf16) | 1.27 | 1.27 |
| PLE n-gram table (bf16, pinned host RAM) | 102.47 | one row per layer, a CPU gather, not modeled |
| MTP layer (bf16, unquantized) | 5.21 | ~0.3 per draft step, plus the shared LM head 1.27 |
| Visual tower | 0.90 | 0 for text |

KV: only 12 of 48 layers are full attention, 2 KV heads x 256 dim, so 24 KiB per token and 6.4 GB per 262k-token stream.

## Ceilings (practical upper bounds)

| Metric | Ceiling | Measured | Share |
| --- | --- | --- | --- |
| Decode c1, 4K context, no spec | ~409 tok/s | ~84-90 tok/s | ~22% |
| Decode c1, 4K, MTP 1 draft at 1.7 accepted | ~540 tok/s | - | - |
| Decode c1, 4K, MTP 2 drafts at 2.2 accepted | ~572 tok/s | - | - |
| Prefill 4K | ~24.8k tok/s | ~9.0k tok/s | ~36% |
| Prefill 128K | ~10.0k tok/s | ~8.5k tok/s (228k prompt, 2026-09-22) | ~85% |

Acceptance lengths in the MTP rows are assumptions until the runs report `c1_accept_len`.

## What this says

Decode at one user sits at about a fifth of the bandwidth bound, so step time is overhead, not weight reads.
Likely contributors: two TP all-reduces in each of 48 layers, the hyper-connection mixers, and the per-token PLE gather from host RAM.
When a step is overhead-bound, verifying k+1 tokens costs about the same as decoding 1, which is exactly where MTP pays.
Long-context prefill is already near its compute ceiling, so prefill levers are low priority.
