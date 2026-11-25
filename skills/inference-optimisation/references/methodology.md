# Measurement methodology

Most "wins" and "regressions" in inference tuning are measurement errors. Each check below changed a conclusion.

## Forecast, then measure

- **Freeze a forecast before each arm** in `forecasts.jsonl`: arm, metric, value and band, mechanism, falsifier. Never edit it once the run starts.
- **Derive it from structure first:** the per-component ledger, measured kernel times, the engine source (default-off settings), then a one-kernel microbenchmark. A full run confirms a derived forecast; it is not the only way to learn.
- **Score every forecast** and keep the running calibration. In one record 117 of 146 hand forecasts over five sessions fell inside their bands (44–100% a session), and the misses taught most: one blamed host work for an idle gap that was the pipeline waiting on its own prompt step; one priced a RAM expert tier at compute cost when it was read-bound. Fix the model, then pick the next arm.
- **Bundled arms:** one forecast per lever, and per-stage timing that separates them. A missed bundle is split.

## Counting

- **Count tokens with usage stats or the tokenizer, never streamed chunks:** with speculation one SSE chunk holds several tokens. Use `stream_options: {include_usage: true}`, `continuous_usage_stats`, or re-tokenize.
- **Size prompts with the real tokenizer:** "random word" documents ran ~3.5 tokens a word, so "32K/128K" cells were really 58K/232K.
- **Build prompts before the clock starts,** unique per run (random prefix) so prefix caching cannot hit, unless you measure hits on purpose.
- **Decode tok/s:** completion tokens after the first ÷ time since it. **Prefill tok/s:** prompt tokens ÷ TTFT, cold unique prompt; warm (cache-hit) is reported apart, never as prefill.
- **cN throughput:** total completion tokens ÷ wall time, first start to last finish (per-stream sums overstate it). Verify the server ran N at once (`max_running`, state slots).

## Synthetic vs real traffic

- A synthetic run (uniform streams, fixed prompts) probes the perfect-world ceiling; the real workload is the truth. Run both; the gap names the bottleneck.
- Example: a two-GPU pool made 334 tok/s on 32 synthetic streams and ~225 on 32 real agent sessions. On real traffic 78–84% of the long waits were one pipeline stage waiting while the other ran a step carrying prompt rows: agents send prompts mid-decode, synthetic streams almost never do.
- Compare real-traffic arms over spans with admitted load; demand gaps (agents thinking, tools running) are not engine time.

## Noise

- Expect ±5–10% run-to-run: thermals, clocks, power source (battery or a balanced profile ran 8–10% slower), background CPU load (a laptop RTX 4080 went 107 → 75 tok/s), prompt content.
- ≥6 decode samples, median, on a fixed seeded prompt panel; random prompts moved acceptance 2.4 → 3.7 and made c1 comparisons noise.
- **Interleave** control, candidate, control; compare only neighbours. One config measured 53.8, then 49.5 an hour later.
- **Comparing engines:** same card, same day, back to back; "B is faster" only if a bootstrap 95% CI of median(B)/median(A) is above 1.0.
- Discard cold first runs and boots after a JIT cache miss (36 vs 90 tok/s once).
- Microbenchmarks need interleaved min-of-N timing; clock ramp-up once made a kernel look 2.5× slower.

## Shared hosts

- An idle box is often unavailable. Book the accelerator plus a cpuset you can read back per process; a reservation or cgroup weight reserves capacity, not cores, so such a run is shared: record the contention. Keep the arm and its controls in one booking.
- Record host load through every run: busy cores outside your set, page cache held by the model file, clocks, other accelerator users. On one machine (illustrative): a CPU expert tier ran 7.07 tok/s quiet and 4.37 with 9–19 other busy cores, and losing page cache tripled its dispatch time.
- When load changes mid-run, mark the run CONTAMINATED from then and keep the clean span. Once a booking ended mid-run (other-core load 1.3 → 20.7 median); the clean first half still confirmed the mechanism and the rest was kept, labelled contaminated.

## Speculative decoding specifics

- Judge step time and accepted length separately (tok/s = a ÷ step_time): a 7-token draft accepted 3.26 vs MTP's 2.85 yet ran 11–14% slower. Counters may update only at request end; probe whole requests.
- Acceptance depends on content and temperature (code and greedy draft far better than sampled prose); report panel, temperature and thinking mode.
- Greedy equivalence can diverge from numerics (M=1 vs M=k kernels); compare logits there: a top-2 tie of 0.016 nats is no bug.

## Long context and output sanity

- A fixed decode window can end inside a long prefill; run long context separately (cold, then warm same document). Needles (3 random codes at different depths, answered exactly) prove the whole context works; speed doesn't.
- Flag looping output, failed requests, an idle GPU (<60% busy), prefill inside the decode window, KV-bound admission. A 117 tok/s "record" was a looping numbered list.

## Memory

- On unified memory (Apple, Intel/AMD iGPU, GB10-class), GPU allocations bypass cgroups: MemAvailable showed 14 GiB of a real 29.4 GiB. Sum driver residency (`drm-resident-*` in `/proc/<pid>/fdinfo` by `drm-client-id`, `nvidia-smi`, `ioreg`/`vm_stat`) plus cgroup `anon`; sample every 1–5 s, kill at the cap.

## Profiling

- Profile with graphs off per kernel, then time the graph replay for the real step.
- Profile by stage (GEMM, attention, MoE, linear attention, norms, sampler, host) before picking a lever. Host stacks (`perf`) named one stall in minutes: a deep copy on the command thread, 11% of a stage's wall.
- Debug builds that remove work change codegen; they are no lower bound.
- Read the ISA when a kernel is "at the limit": register moves, spills and serialized tails hide there.
