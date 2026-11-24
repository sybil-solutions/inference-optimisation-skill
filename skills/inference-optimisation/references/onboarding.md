# Mac onboarding and the best-engine-first check

This workflow is for a new machine, especially Apple silicon, where the engine choice can matter more than the first round of flags. It uses only the bundled standard-library tools.

## 1. Preflight

From the experiment directory:

```bash
S=/path/to/inference-optimisation-skill/skills/inference-optimisation
python3 "$S/scripts/preflight.py" \
  --model my-served-model \
  --model-path /path/to/local/checkpoint \
  --out . \
  --memory-cap-gib 48 \
  --context 131072
```

Preflight reads the host inventory, checks AC power, chooses a free localhost port, and writes:

- `ceiling.json`: starter hardware/model sizes for `ceiling.py`;
- `config.json`: a starter `bench.py` config using that port;
- `DEAD-ENDS.md`: a reusable rejection log;
- an inventory/assumptions record in the generated ceiling file.

Review the output before trusting it. A local Hugging Face-style checkpoint can be measured from its safetensors headers; remote names, GGUF metadata and unknown hardware may require explicit size or performance overrides. Bandwidth and FLOPs marked as assumptions are starting points, not measurements. Run the device-copy and GEMM measurements from `ceilings.md` before publishing a ceiling.

Plug in AC power, copy the config into the queue's expected location, and edit the served model name or launch command:

```bash
mkdir -p configs
cp config.json configs/base.json
python3 "$S/scripts/ceiling.py" ceiling.json
```

The generated `suggested_memory_cap_gib` is not enforced by default. To enable the watchdog, rerun preflight with a command that prints the server's real footprint in bytes, for example `--memory-probe 'bash /path/to/mem.sh my-container'`. A config that sets `memory.cap_gib` without `memory.probe` is refused rather than silently running without a cap.

The generated config benchmarks an already-running server by default. If `launch` is supplied, it must return after the server starts (a daemonized process or `docker run -d`, not a foreground server). During `screen`, `launch_timeout_s` and `stop_timeout_s` default to 30 seconds and `screen_health_timeout_s` defaults to 90 seconds, so three broken engines cannot consume the whole bake-off. Set `stop` to the matching graceful stop command.

## 2. Step 0.1: a 15-minute engine bake-off

Skip this only when the user has already fixed the engine. Otherwise choose two or three **installed** engines that support the exact model, quantization and chip. Examples of families to check on Apple silicon include `mlx-lm`, `oMLX` and `llama.cpp`; this is not a ranking, and a model may only run well on one of them.

Create one config per engine. Give each a unique `name`, port, `launch` and `stop`. Keep the model, context, memory cap and benchmark settings identical. Then screen them serially:

```bash
BENCH_MODE=screen \
BENCH_FLAGS="--budget 90 --prefill 4096" \
bash "$S/scripts/queue.sh" engine-a engine-b engine-c

python3 "$S/scripts/bench.py" report
python3 "$S/scripts/bench.py" show
```

Keep the whole bake-off, including cold startup, under 15 minutes. Reject an engine that:

- cannot start or stay healthy;
- fails the code-edit or OpenAI `tool_calls` check;
- hangs on the greedy prefill;
- does not reuse the turn-1 prefix on turn 2;
- loops, returns empty output, or runs on battery;
- cannot hold the model under the memory cap;
- spends the entire budget starting.

`screen` is deliberately short: three prose reps, one cold 4K prefill, one code edit, one tool call and one two-turn prefix probe. It is a smoke gate, not a statistical benchmark. Run the full interleaved A/B harness on the survivors before choosing a winner.

For a non-agent model, tool support can be disabled explicitly with `"screen_require_tools": false`. Prefix caching defaults to required because slow agent turns are one of the failures this check is meant to catch; disable it only with a recorded reason.

## 3. Baseline and quality

Capture deterministic greedy outputs from the control before trying a lossy change:

```bash
python3 "$S/scripts/bench.py" quality configs/base.json \
  --write-baseline quality-baseline.json --prompts 3 --max-tokens 256 --budget 300

python3 "$S/scripts/bench.py" run configs/base.json --long 131072
bash "$S/scripts/queue.sh" base cand base
python3 "$S/scripts/bench.py" quality configs/cand.json \
  --baseline quality-baseline.json --prompts 3 --max-tokens 256 --budget 300
python3 "$S/scripts/bench.py" report
```

An exact output mismatch voids the candidate. The report keeps full-run and short-screen leaderboards separate, shows the latest quality gate even when it fails, and includes the first differing character and a short snippet from each side. A quality row never replaces a performance row. Whitespace-only matches are counted separately; investigate early or repeated divergences rather than waiving the gate.

## 4. Record dead ends

Every rejected config gets a row in `DEAD-ENDS.md`:

| Date | Config / change | Evidence | Why rejected | Safe to retry when |
|---|---|---|---|---|
| YYYY-MM-DD | engine/config + one changed lever | screen flags, A/B numbers, quality divergence | concrete failure mode | new engine/model/hardware reason |

`PLAN.md` keeps the full timestamped history. `DEAD-ENDS.md` is the short list an agent checks before repeating work.

## macOS notes

- `queue.sh` uses Python's portable `fcntl.flock`, so it does not need the Linux-only `flock(1)` executable. The default lock lives in a private per-user runtime directory; a custom `QUEUE_LOCK` must be a regular file owned by the current user. The queue exits nonzero if any benchmark fails.
- Detach a long queue portably with:
  ```bash
  nohup bash "$S/scripts/queue.sh" base cand base > queue.log 2>&1 < /dev/null & disown
  ```
  `setsid` is Linux-only and is not required.
- The bundled `mem.sh` currently targets Linux/containers. For a bare-metal Mac server, provide a `memory.probe` command that prints real unified-memory bytes; do not substitute RSS for GPU residency and call it a memory cap.
