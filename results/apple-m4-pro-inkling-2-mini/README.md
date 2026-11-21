# Inkling 2 Mini on Apple M4 Pro (Mac mini, 48 GB)

Hill-climb of Thinking Machines' Inkling 2 Mini (26.4B total, 3.4B physical / 5.4B effective active, looped MoE, native NVFP4/MXFP8, 20 GB) on a Mac mini, using this skill's method.

| | |
|---|---|
| Hardware | Apple M4 Pro, 48 GB unified LPDDR5X, 273 GB/s, on AC power |
| Engine | Ollama v0.40.17-rc3, MLX runner, built-in MTP drafter ("shared-verification" speculation) |
| Model / quant | `inkling-2-mini`, shipped mixed NVFP4 + MXFP8 (no re-quantization) |
| Harness | `scripts/bench.py` unchanged (OpenAI-compatible `/v1`), `python3 -I`, serialized runs, memory watchdog on `/api/ps` size (cap 36 GiB) |
| Host state | **Not idle** (other background jobs running), so run-to-run noise is about +-10% |

## Before -> after

| Metric | Before | After | Ceiling |
|---|---|---|---|
| Aggregate decode, 2 concurrent streams | 18.14 tok/s | **29.45 / 28.66 tok/s** (+60%) | n/a |
| 4 concurrent streams | n/a | 28.71 tok/s (no better than 2) | n/a |
| Decode, 1 stream (median of 6) | 23.1 tok/s | unchanged (21.8 - 24.6 across controls) | ~53 without speculation |
| Prefill 4K / 16K tokens | 253 / 251 tok/s | unchanged | ~790 |
| Long-context needle gate | not run | **3/3 at 31,285 and at 62,553 tokens** (96% of the 65,536 context) | n/a |
| Peak memory | 25.6 GiB | 28.9 GiB at 62.5K tokens | cap 36 |

Ceiling arithmetic: 5.4B effective-active params x ~0.76 bytes/param (20 GB / 26.4B) ~= 4.1 GB read per token; 273 GB/s x ~0.8 / 4.1 GB ~= 53 tok/s. Prefill: 2 x 5.4B = 10.8 GFLOP/token against ~50% of ~17 TFLOPS ~= 790 tok/s.

## What worked
- **`OLLAMA_NUM_PARALLEL=2`** (the MLX runner picks parallelism from free-memory headroom; pin it so it does not change when other models are loaded or unloaded). Decode is bandwidth-bound, so a second stream is nearly free: +60% aggregate. Four streams gave nothing more.

## Rejected / unproven (interleaved controls, n=6 per run)
- Draft length via Modelfile `PARAMETER draft_num_predict` 4 and 2: 23.29 and 24.38 tok/s against neighbouring controls 22.35 / 24.56 / 24.19. Inside noise. The server log kept reporting `max_draft=7`, so the parameter may not be honoured by the MLX runner.
- `draft_num_predict 0` (speculation possibly off): 26.45 against controls 21.78 and 23.91. The median beat both neighbours, but per-prompt differences were inside the spread, and we could not confirm drafting was actually disabled. Not adopted; needs an idle host and a code/JSON panel.

## Not a speed lever, but large: reasoning effort
Inkling exposes `reasoning_effort` (none, minimal, low, medium, high, xhigh, max; default high). On three small tasks, `low` used 2-3x fewer completion tokens (alarm triage 666 -> 213) with the same pass/fail. `none` saved only about 1.5x. Harder work was not measured.

## Not run
Concurrent full-KV soak (2 x 62K at once) and a greedy-equivalence check (nothing lossy was adopted).

Raw data: `ledger.jsonl` (one line per run, from `bench.py`) and `PLAN.md` (experiment log with verdicts).
