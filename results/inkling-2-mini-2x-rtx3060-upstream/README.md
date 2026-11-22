# Inkling 2 Mini on 2× RTX 3060 12 GB, upstream llama.cpp (2026-11-20/22)

A placement study complementing #12, which covers one RTX 4070 on Unsloth's mix build. Here: two 12 GB cards on **upstream** llama.cpp, with ggml-org's `inkling2_loop` GGUF.

## Setup

| | |
|---|---|
| GPUs | 2× RTX 3060 12 GB (334 GB/s copy each). CUDA1 also drives the desktop (~0.7 GB) |
| Host | i9-9980XE, 64 GB DDR4, Linux, driver 615.78, CUDA 13.4 |
| Engine | upstream llama.cpp 1.0.0-dev **8d60873df** (also seen on 46eb1742d), CUDA sm_86, flash attention on |
| Model | `ggml-org/Inkling-2-Mini-GGUF` IQ4_NL (17.63 GB, arch `inkling2_loop`) + mmproj BF16 on CUDA0. Unsloth's UD GGUFs (arch label `inkling2`) and their dspark draft do not load on upstream (`blk.24.attn_q.weight not found`, draft vocab 201024 vs 200058) |
| Requests | thinking off (`thinking_budget_tokens: 0`; `reasoning_effort` and `enable_thinking` are ignored by upstream's template handling), temperature 1.0 |

## What happens on two GPUs

| Attempt | Result |
|---|---|
| `--split-mode tensor` | `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'inkling2_loop'` |
| `--split-mode layer` (with or without `--tensor-split 1,1`) | the whole 15.9 GB weight buffer goes to CUDA0 → OOM |
| auto-fit (no `-ngl`) at 128K | weights fit, then a **33 GB** prompt-processing compute buffer is requested on CUDA0 |
| **`-ot` with explicit `blk.N.=CUDA0/CUDA1`** | weights load on both cards. #12's tensor-override trick also works on upstream for multi-GPU placement |

## The limit after placement: the compute buffer scales with ubatch × context

| ubatch | 32K | 64K | 128K |
|---|---|---|---|
| 512 | 3.6–3.7 GB | 6.5–6.8 GB | 13.4 GB |
| 256 | 1.85 GB | 3.5 GB | |
| 128 | 0.96 GB | 1.9 GB | 3.5 GB |
| 64 | | | 2.1 GB (+ 2 GB KV) |

Measured from the failed allocations (`failed to allocate compute pp buffers`). Roughly ubatch × context × ~220 bytes, which looks like materialised attention scores. Plan VRAM for it before weights.

## Placements that load, and what they do

The placement that fits (`p3`) keeps the 3×-executed core on one card, to avoid cross-GPU hops inside the loop, and pushes only the last two 1×-executed blocks' experts to RAM (#12's byte-ROI argument):

```
-ot 'blk\.2[23]\.ffn_(gate_up|down)_exps\.=CPU,blk\.([6-9]|1[0-7])\.=CUDA1,blk\.([0-5]|1[89]|2[0-3])\.=CUDA0'
--split-mode layer -ngl 99 --cache-type-k f16 --cache-type-v f16 --mmproj-device CUDA0
```

| Config | Decode (short reply) | Prefill, 6K prompt | VRAM CUDA0 / CUDA1 |
|---|---|---|---|
| p3, 32K, ub256 | 25.3 tok/s | 296 tok/s | 10.3 / 10.9 GB |
| p3, 64K, ub128 | 25.4 tok/s | 257 tok/s | 10.8 / 11.5 GB |
| p3, 32K, ub256, `-t 8 -tb 8` | 25.1 | 296 | same |
| p3, 16K, ub512, `-t 8 -tb 8` | 25.1 | 311 | 9.9 / 10.9 GB |
| p4 (every weight on GPU), 16K, ub256 | 26.6 | 305 | 11.1 / 9.8 GB |
| p4, 16K, ub512 | 26.4 | 317 | 11.9 / 10.9 GB |
| auto-fit (`--fit`), 32K / 8K, ub512 | 23.9 / 25.7 | | 10.6 / 10.8 GB, ~26 GB RAM |
| p3 at 32K-ub512, 64K-ub256, 128K-ub128 and 128K-ub64 | OOM | | |

**Placement, thread count and CPU experts make no difference here: decode stays at 24–27 tok/s even with every weight in VRAM.** A rough ceiling from this skill's method is ~3 GB read per token (5.4B effective active parameters at ~4.5 bpw; layer mode reads one card at a time) at 334 GB/s, about 100 tok/s. So upstream's `inkling2_loop` path runs at roughly a quarter of the bandwidth ceiling on Ampere. #12 measured 34–37 tok/s and ~800 tok/s prefill on a single RTX 4070 with half the experts on the CPU using Unsloth's mix build, which suggests that build's Inkling kernels are considerably faster than upstream's at 8d60873df. Upstream is moving quickly (two Inkling commits on 11-21), so treat these as a dated snapshot.

## Not covered

No interleaved-control bench (`bench.py`) for these configs: the load tests used one short reply and one 6K prefill each. Treat the decode and prefill numbers as ±10%. Unsloth's mix build was not tested on this machine.
