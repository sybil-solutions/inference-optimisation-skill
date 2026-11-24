# Measurement methodology

Most "wins" and "regressions" in inference tuning are measurement errors. In every case below, a check like this changed a conclusion.

## Counting

- **Count tokens with the tokenizer or the server's usage stats, not streamed chunks.** With speculative decoding, engines emit several tokens per SSE chunk; counting chunks under-reported MTP decode. Use `stream_options: {include_usage: true}` or `continuous_usage_stats`, or re-tokenize the output.
- **Size prompts with the real tokenizer.** "Random word" documents were ~3.5 tokens per word, not ~1.3–1.9, so nominal "32K/128K" cells were really 58K/232K.
- **Build prompts before the clock starts.** Make them unique per run, with a random prefix, so prefix caching cannot hit, unless you are measuring cache hits deliberately.
- **Decode tok/s:** completion tokens after the first token ÷ time since the first token.
- **Prefill tok/s:** prompt tokens ÷ TTFT on a cold, unique prompt. Report cold and warm (cache-hit) separately; never publish a cache-hit prefill as prefill.
- **Concurrent throughput (cN):** total completion tokens across N streams ÷ wall time from first start to last finish. Per-stream sums overstate it when streams don't overlap. Verify the server actually ran N at once (`max_running`, state slots).

## Noise

- Expect ±5–10% run-to-run. Sources include:
  - thermals and clocks;
  - power source (a laptop on battery or a balanced profile ran 8–10% slower);
  - background CPU load (a laptop RTX 4080 went 107 → 75 tok/s under light CPU load);
  - prompt content, because acceptance depends on text type.
- Take ≥6 decode samples and use the median. Use a fixed, seeded prompt panel; random prompts moved acceptance from 2.4 to 3.7 and made c1 comparisons pure noise.
- **Interleave runs:** control, candidate, control. Compare a candidate only with its neighbours. Drift over an hour can exceed the effect you are measuring: the same config measured 53.8 and then 49.5 an hour apart.
- **When comparing engines,** run them on the same card on the same day, back to back. Accept "B is faster" only if a bootstrap 95% CI of median(B)/median(A) is above 1.0.
- Discard cold first runs and the first boot after a JIT cache miss (36 vs 90 tok/s on one vLLM boot).
- Microbenchmarks need interleaved min-of-N timing; clock ramp-up made one kernel look 2.5× slower than it was.

## Speculative decoding specifics

- Judge per-step time (ms per verify step) and accepted length separately. tok/s = a ÷ step_time.
- Higher acceptance can be slower: a 7-token draft accepted 3.26 vs MTP's 2.85, yet ran 11–14% slower.
- Acceptance counters may only update on request completion, so a short window sees a biased sample. Probe whole requests.
- Acceptance depends on content and temperature. Code and greedy decoding draft far better than sampled prose. Report the panel, temperature and thinking mode with every number.
- A greedy equivalence check (spec on vs off) can diverge because of numerics: different kernels for M=1 vs M=k rows. Compare logits at the divergence point before calling it a bug. A top-2 tie of 0.016 nats is not a bug.

## Long context

- A fixed decode window can end inside a long prefill. Use a dedicated long-context run: a cold request, then a warm same-document request.
- Retrieval needles (3 random codes at different depths, answered exactly) prove the whole context works. Speed alone doesn't.

## Fast screening and quality gates

- **Screen before measuring.** `bench.py screen` spends at most its `--budget` (default 90 seconds) after the server is healthy. It runs three fixed prose reps, a cold 4K greedy prefill, a small code edit, an OpenAI `tool_calls` round-trip, and a two-turn prefix-cache probe. A screen is a pass/fail smoke test, not evidence of a speedup.
- **A greedy request must finish.** The screen bounds both total elapsed time and each socket read, so a `temperature: 0` request that hangs or dribbles data is rejected instead of stalling the queue. A stream that closes before `[DONE]` or a terminal `finish_reason` is also a failure; the tool probe additionally requires a terminal `tool_calls` finish reason and server usage.
- **Prefix caching must be visible.** Turn 2 must report at least 80% of turn 1's prompt tokens as `cached_tokens` and start within `screen_prefix_max_s` (default 1 second). Servers that omit usage details fail this gate; do not infer a cache hit from latency alone.
- **Code and tool gates validate structure, not correctness of the model.** The code check statically evaluates a tiny clamp fix without executing model output. The tool check requires a `get_weather` call with valid JSON arguments. Neither replaces the full long-context and agent-task gates.
- **Capture a quality baseline before lossy changes.** `bench.py quality --write-baseline` records deterministic greedy outputs; `--baseline` compares the candidate exactly and reports the first divergence. The default total budget is 300 seconds including startup and watchdog cleanup (`--budget`). Whitespace-only differences are counted separately. Investigate every exact mismatch before calling a change lossless.

## Output sanity

- Flag looping output (repeated shingles), failed requests, an idle GPU (<60% busy in the window), prefill inside the decode window, and KV-bound admission. A 117 tok/s "record" was a model emitting a numbered list in a loop.

## Memory

- On unified memory (Apple, Intel/AMD iGPU, GB10/Spark-class), GPU allocations bypass cgroup accounting. A MemAvailable delta showed 14 GiB where the real footprint was 29.4 GiB.
- Sum driver residency (Linux DRM: `drm-resident-*` in `/proc/<pid>/fdinfo`, deduplicated by `drm-client-id`) plus cgroup `anon`.
- Track the peak with a ~1–5 s sampler, and kill the run at the cap. A missing, failing, or timed-out probe voids the run; a configured cap is never treated as enforced without a successful sample.

## Profiling

- Profile with graphs off to get a per-kernel breakdown, but remember that graphs change launch overhead. Time the graph replay itself for the real step time.
- Profile prefill by stage (GEMM, attention, MoE, linear attention, norms/Hadamards) before picking the next lever.
- Debug builds that remove work (e.g. skipping dequantization) change code generation, so they are not valid lower bounds.
- Read the ISA or assembly when a kernel is "at the limit". Register-region moves, spills and serialized tails hide there.
