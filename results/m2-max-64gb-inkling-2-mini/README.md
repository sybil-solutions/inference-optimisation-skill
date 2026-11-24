# Inkling-2-Mini on a 64 GB M2 Max, mlx-dspark with FP16 fused kernels

Inkling-2-Mini (26B MoE, 128 experts top-8, a 3-pass loop core and a bundled MTP head) runs at a median **37.0 tok/s single-stream decode** on a 30-core M2 Max, and **45.1 tok/s** with speculation off, inside a sampled 45 GiB footprint at 64K context.
Prefill holds about 319 to 358 tok/s at 3,302 prompt tokens and 315 to 350 at 12,900, depending on profile.
Four concurrent requests aggregate 59.2 tok/s, and a 63,968-token needle prompt still answers 3/3.

Those are medians across repeated runs, with the spread below. A single quiet run reached 39.1 tok/s decode and 62.7 tok/s aggregate, and quoting that would overstate what the same machine delivers with a busy desktop.

## Setup

| | |
| --- | --- |
| Hardware | MacBook Pro, Apple M2 Max, 30-core GPU, 64 GB unified memory, AC power, macOS 26 |
| Measured hardware | 368 GB/s read, 370 GB/s copy, 9.2 FP16 TFLOPS at 4096^3 |
| Model | `thinkingmachines/Inkling-2-Mini` at `7a67ae2`, 20.2 GB including the bundled `mtp.safetensors`. NVFP4 routed experts, 8-bit attention and LM head, FP16 activations |
| Engine | mlx-dspark, fork of `ARahim3/mlx-dspark` v0.21.1 with the FP16 fused Metal kernels, on MLX 0.32.4 and mlx-vlm `d998a62` |
| Launchers | `launch.sh` here, three profiles: agent (lookup drafts), plain (no speculation), multi (shared-prefix KV) |
| Harness | `bench.py`, `dev/screen_fast.py`, `dev/agentbench/agentbench.py`, `dev/branch_bench.py` and `dev/gate.py` from `fda968e` |

## Results

Pooled across every validated run on this machine. Agent: 3 runs, 18 decode samples. Plain: 2 runs, 12 samples.

| Measurement | Agent profile | Plain profile |
| --- | ---: | ---: |
| Decode, 1 stream (median) | **37.0 tok/s** | **45.1 tok/s** |
| Decode, per-run medians | 35.0 / 33.5 / 39.1 | 42.2 / 46.2 |
| Decode, best single sample | 52.3 | 48.2 |
| Decode, 4 streams aggregate | **59.2 tok/s** (16.6/stream) | — |
| Decode, 2 streams aggregate | — | 53.3 tok/s |
| Prefill, 3,302-token prompt | 319.3 tok/s (mean 327.4) | 357.7 tok/s |
| Prefill, 12,900-token prompt | 315.1 tok/s (mean 323.5) | 350.25 tok/s |
| Short TTFT | 557-568 ms | 560-566 ms |
| Sampled peak footprint, 4 streams | 45.0 GiB at 64K needle | 26.0 GiB |

Workload-shaped figures:

| Measurement | Value | Runs |
| --- | ---: | --- |
| Code-edit decode, 2 KB fast screen | 76.5 tok/s, 20.9 s end to end, edit and tool-call gates pass | 3 |
| Edit decode on a 21K prefix | 113.4 tok/s, 3/3 pass at a 6K output cap | 1 (see caveats) |
| Branch fan-out, 23.5K shared prompt | 1 branch 43.6 tok/s, 4 branches 17.0 tok/s each, 43.6 aggregate, 3.36 s and 10.1 s wall | 2 |
| Warm 21K second turn | TTFT 0.71 s from 21,323 cached tokens | 3 |
| Cold 21K first turn | TTFT 65-71 s at 21,334 prompt tokens | 3 |
| 63,968-token needle | TTFT 232-287 s, decode 34.9-37.7 tok/s, retrieval 3/3 | 3 |

The fast-screen edit rate in the table is the quiet run. Across the three runs it reads 71.2, 70.4 and 76.5 tok/s, so the median is 71.2.

Earlier baselines on the same machine, from separate sessions and not interleaved with these runs: oMLX 0.8.2 stock 32.4 tok/s, stock mlx-dspark 31.1, MTPLX 2.13.2 35.7, mlx-serve 26.11.4-pre1 32.7, llama.cpp 20-26. Treat that column as context for engine choice rather than a controlled delta.

## What moved the number

- **FP16 fused Metal kernels**, templated so mask, short-conv, layer, attention, KV and MoE run as one launch each. Prefill went from about 280 to 350-plus tok/s.
- **NVFP4 experts with exact FP16 block scales**, which cut expert bytes without a measurable quality loss on the token-forcing gate.
- **8-bit attention and LM head** through `--target-bits 8`. Dense bytes are 59% of the per-token read on the bf16 checkpoint, so this is the largest single lever.
- **A 32k pruned draft vocabulary** and lookup drafts, which lift code edits well above prose decode.
- **Boundary snapshots** for warm prefix reuse, taking second-turn TTFT at 21K from 20.5 s to 0.71 s.
- **Shared-prefix global KV** in the multi profile, which saves about 8 GiB across four 23.5K branches.

## Gates

| Gate | Result |
| --- | --- |
| Token forcing, 12 prompts x 256 tokens | PASS. Candidate mismatch 1.074% against reference 1.469%, rate rule passes at -0.39 pp with a -0.87 pp one-sided 95% lower bound, 0 large-margin events against an allowance of 3.00 |
| Kernel self-tests | mask, sconv, layer, attn, kv, moe, res, ctrl, model, wide all inside tolerance, maximum absolute difference 1.19e-07 |
| Long-context needle | 3/3 exact at 63,968 prompt tokens |
| Edit gate | 3/3 pass, `finish_reason=stop`, 3,103 tokens at a 6K cap |
| Tool-call round trip | pass |
| Concurrency | 4/4 branches ok, no memory-cap trip |

This gate measures next-token agreement and margin sizes on a fixed prompt set. It is a release check for numeric drift, not a semantic or perplexity evaluation. Reference and candidate decisions were identical on a rerun, so it is deterministic on this build.

## Caveats

- **Two to three runs per profile.** Identical launchers and configs produced a best-to-worst spread of up to 11% on decode and prefill, while the edit screen's decode finished 2.3% down and branch fan-out moved by under 6%. The spread fits desktop drawing load, which was the one thing that changed between sessions, but this data cannot prove that cause. That is why the headline is a median across runs.
- **Memory pressure appeared in the agent runs.** Both agent sessions logged a memory-pressure warning and discarded cached prefix data, with no cap trip. Sampled peak footprint is a periodic physical-footprint reading, not a recorded high-water mark.
- **The 21K edit figure is one run.** An earlier attempt capped output at 3,000 tokens and truncated at the cap, so only the 6K-cap run qualifies.
- **The branch bench prompt is 23,484 tokens**, above the 16-21K band the config nominally targets, in every run. Figures are comparable across runs, not directly to the agent-profile prefills.
- **One machine, no thermal instrumentation.** WindowServer or another GPU client moves decode by several percent, so measure on a quiet desktop and check GPU ownership first.
- **The NVFP4 and 8-bit paths are lossy** by design. The token-forcing gate is the release check, not a proof of lossless output.

## Files

- `THEORY.md` - bytes per token, ceilings and where the remaining gap sits
- `PLAN.md` - hill-climb log: what was accepted, parked and rejected
- `launch.sh` - the three profiles exactly as benchmarked
- `configs/` - harness configs for agent, plain and multi
- `ledger.jsonl`, `ledger-fast.jsonl`, `ledger-agent.jsonl`, `ledger-branches.jsonl` - raw per-run records for every run behind the medians. Repeats share a name, so the `time` field identifies the run.
- `gate.json`, `gate.log` - token-forcing gate detail
- `context-scaling.png`, `concurrency-scaling.png` - decode against context length and against concurrency

