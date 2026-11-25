---
name: inference-optimisation
description: >
  Hill-climb the serving speed of a local LLM on any hardware (NVIDIA, AMD, Intel,
  Apple, unified memory, multi-GPU, CPU offload) and quantization (GGUF, EXL2/EXL3,
  AWQ/GPTQ, FP8, MLX, BF16): c1/cN decode, speculative decoding, prefill, KV size,
  startup, TTFT, under a memory cap with full-context stability. Per-component
  ceilings, a frozen forecast per arm, noise-controlled A/B on synthetic and real
  traffic, a ledger, and qualification before publishing. Use when asked to optimise, tune, speed up,
  benchmark or hill-climb a model server (vLLM, SGLang, llama.cpp, ExLlama/TabbyAPI,
  MLX, TensorRT-LLM...), find the maximum tok/s on a machine, or publish a tuned
  recipe. Triggers: hill climb, tok/s, decode/prefill speed, MTP, EAGLE, draft model,
  speculative decoding, KV cache, TTFT, startup time, theoretical maximum, roofline.
---

# Inference optimisation (hill-climbing a model server)

You make a model server faster on one machine, with evidence:

1. Measure the baseline on a synthetic benchmark and on the real workload.
2. Compute the ceiling per component.
3. Freeze a forecast for the next arm.
4. Run the arm against interleaved controls.
5. Score the forecast, keep or reject, and write down why.
6. Repeat until the gap needs out-of-scope work; then qualify and publish.

References, read when needed: `references/ceilings.md` (maxima, per-component roofline, worked example), `methodology.md` (forecasts, counting, noise, shared hosts), `levers.md` (levers with measured evidence), `gotchas.md` (engine and hardware traps), `publishing.md` (gates, pinning, report). `scripts/`: bench, ceiling calculator, memory probe, serialized queue, ledger.

## 0. Before touching anything

- **Write down targets and constraints;** ask only if a missing one changes what you do. Metrics: c1 and cN decode, prefill at several lengths, TTFT, startup, and the real workload's own rate (e.g. agent tasks per hour). Memory cap, context that must stay stable, allowed quality trades (lossless only, or labelled lossy ones such as int8 prefill).
- **Inventory:** accelerator, memory (type, size, unified?), CPU cores and NUMA, driver, engine version, model, quant, launch command, who else uses the host.
- **Measure, don't trust spec sheets:** copy bandwidth, a large GEMM, clocks under load.
- **Read existing logs and forecasts** for this machine and model. Never rerun a rejected experiment without a new reason.
- **Working directory:** `PLAN.md` (timestamped: forecast, numbers, verdict, reason), `THEORY.md`, `forecasts.jsonl`, `ledger.jsonl`, `configs/` (one JSON per run), `best-config.json`.

## 1. Ceilings first

Before optimising, compute (`scripts/ceiling.py`, `references/ceilings.md`):

- **Decode** is bandwidth bound: bytes per emitted token (dense, active experts, LM head, KV) ÷ measured bandwidth; speculation divides target bytes by accepted length and adds the draft's.
- **Prefill** is compute bound: FLOPs per token at ~50% of measured peak. Also TTFT, startup and KV floors.
- **Batched serving gets a roofline per component** (dense, experts, attention, state, LM head, sampler, host) from bytes and FLOPs per row at real context lengths, with percent of ceiling per component: a total hides which part is far from its floor, and batching amortizes shared bytes, not per-row FLOPs.

Report progress as percent of the practical ceiling. If the target is above it, say so early, with the arithmetic.

## 2. Ground rules (non-negotiable)

1. **Forecast before every arm** in `forecasts.jsonl`: metric, value with band, mechanism, falsifier. Score it after; keep the calibration (N of M inside) in `PLAN.md`. A miss is information about your model of the machine, not just the config: find the wrong term before the next arm.
2. **Size arms by attribution.** Bundle levers in one arm when each has its own frozen forecast and per-stage timing (profiler, engine stage stamps) can attribute the gain; confirm against interleaved controls. Change one thing at a time only when attribution is unclear, or to split a bundle that missed. Derive configs from the current best.
3. **Control for noise** (typically ±5–10%): ≥6 decode samples, median; control, candidate, control. Accept only if the candidate beats both neighbours and regresses nothing by >~3%. Repeat surprising wins.
4. **Fix the environment, or record it.** AC power and a fixed performance profile, enforced by the harness; restore what you changed, and say so. On a shared host, book the accelerator plus a fixed core set, record host load through each run, and mark a run CONTAMINATED from when load changed, keeping the clean span.
5. **One accelerator job at a time,** under a lock (`scripts/queue.sh`). Kill by PID, never with `pkill -f`/`pgrep -f` patterns that can match your own shell.
6. **Every launch has a blocking startup check:** wait for health with a deadline, log progress, exit non-zero on failure. Failing fast needs a time-limited liveness probe on the serving process itself (container state only sees the container exit). A server that never came up fails the launch; it never leaves a queue waiting silently.
7. **Long jobs run as harness background tasks or service-manager units** (`systemd-run --user`, launchd): logged, visible, stoppable. Never `nohup`, `setsid` or `& disown`.
8. **Memory cap watchdog** that kills the run; on unified memory count driver residency plus anon (`methodology.md`).
9. **Stability gates** before "best": needles at ~50% then ~100% context, found exactly; a concurrent full-length soak if KV is sized for N users; greedy equivalence if the change claims to be lossless.
10. **Never force-kill a server mid-batch.** Abort, drain, stop with a timeout; a forced stop mid-prefill has caused GPU driver faults.
11. **Verify outputs, not just speed:** flag looping output, failed requests, an idle GPU. A suspiciously fast result is usually broken.
12. **Log everything in `PLAN.md`,** failures included; negative results are the most reusable output.
13. **Credentials:** never prompt for, store, print or publish one. Use what the environment provides (credential helper, environment variable); if one is missing, stop and say what is needed.

## 3. The loop

```
baseline → per-component ceiling → pick levers (gap × likelihood) → freeze forecast → arm vs controls
   ↑                                                                                       ↓
   └──── update best-config + PLAN.md + ledger ←──── score forecast, accept / reject ←─────┘
```

**Synthetic vs real.** A synthetic run is the perfect-world ceiling probe; the real workload (agent sessions, their prompt mix) is the truth, and the gap between them names the bottleneck: real agents send long prompts mid-decode, so mixed prefill+decode steps, rare in synthetic runs, can dominate wall time. Keep a lever only if it holds on real traffic.

### Lever order

Cheapest first; evidence in `levers.md`.

1. **Persistent JIT/compile/graph caches** (startup −50–70%); eager loading where mmap is slow.
2. **Decode graphs:** often 2× c1 alone.
3. **Speculative decoding:** method; draft length per concurrency and context; pruned draft LM head; GPU sampling; decode-style verify.
4. **Concurrency and KV:** N × context pool, hybrid state slots, prefix caching (`cached_tokens` > 0).
5. **Prefill and scheduling:** chunk size, where prompts run beside decode, faster attention, int8/fp8 prefill GEMMs (labelled), no MoE padding cliffs.
6. **Host path:** copies, syncs, samplers on the critical path; threads and NUMA for CPU tiers.
7. **Kernel thread and tile sizes:** defaults suit a bigger card.
8. **Kernel work** after config levers plateau: profile, read the ISA, fix the dominant kernel.

**When a lever fails:** fusion gives nothing under graphs → time is inside the big kernels; stop fusing. Higher acceptance without higher tok/s → the draft costs too much, measure per-step time. A weird result → suspect the measurement (chunk-counted tokens, prompt length not from the real tokenizer, cache hits, thermals, battery, host load).

**Plateau:** three levers in a row inside noise. Write a STATUS section in `PLAN.md`: best vs baseline vs ceiling per component, what limits further gains with evidence, forecast calibration, remaining levers ranked with cost. Keep looping slower if told "don't stop", else hand back.

## 4. Qualify and publish

See `publishing.md`: pin weights, engine and kernels; build clean from the pins; run the acceptance gates (load, chat, reasoning, tools, needle, speed); A/B the clean build against your best. Gate and hill-climb numbers are not comparable; say which is which.

## 5. Reporting to the user

- **Start with a table:** metric | baseline | now | ceiling | how measured (median of N, cold or warm, harness, synthetic or real). Add percent of ceiling per component and forecast calibration.
- **Then:** what got the gains, why it stops here, known limitations, contaminated runs, and the system state you changed (power profile, running servers and units, uncommitted code, branches that must stay).
- **Say plainly** what was not verified.
- **To free the GPU:** drain, stop the server, confirm no processes and an idle clock, end any loop or unit.
