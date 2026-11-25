# Prompt: run an inference hill-climb with the `inference-optimisation` skill

Paste the block below as the task for a coding agent with shell access on the machine that serves the model. Fill in the `<…>` fields first. Delete lines that don't apply.

---

You are optimising the serving speed of a local LLM on this machine. Use the `inference-optimisation` skill. If your harness doesn't load skills, read `skills/inference-optimisation/SKILL.md` now, and read its `references/` files when the skill points to them. Follow its ground rules exactly.

## Target

- **Model:** `<model name / repo @ revision>`, quantization `<e.g. EXL3 4bpw / GGUF Q4_K_M / AWQ / FP8 / MLX 4-bit>`
- **Engine:** `<vLLM / SGLang / llama.cpp / TabbyAPI / MLX-LM / …>`
- **Current launch:** `<command, or "the server on http://127.0.0.1:8000">`
- **Hardware:** `<GPU(s), memory, unified or discrete>`
- **Metrics to maximise, in priority order:** `<e.g. c1 decode tok/s, c4 aggregate decode, prefill tok/s at 4K/32K/256K, KV capacity, startup time, TTFT>`
- **Hard constraints:**
  - memory cap `<e.g. 46 GiB for the whole stack>`;
  - context `<e.g. 262,144 tokens>` that must stay stable;
  - concurrency `<N>` users at full context;
  - quality `<lossless only | labelled lossy trades allowed (e.g. int8 prefill)>`.
- **Stretch goal from the user:** `<e.g. "100 tok/s c1">`. Check it against the ceiling and report honestly.

## How to manage the work

1. **Set up and measure.**
   - Set up the experiment directory: `PLAN.md`, `THEORY.md`, `forecasts.jsonl`, `configs/`, `ledger.jsonl`, `best-config.json`.
   - Inventory the machine, and measure real bandwidth and matmul throughput.
   - Compute the ceilings with `scripts/ceiling.py` and write them to `THEORY.md`.
   - Measure the baseline with `scripts/bench.py`, and on the real workload if there is one.
2. **Loop:**
   - Pick the lever with the biggest expected gain, using the lever order in `SKILL.md` and the evidence in `references/levers.md`.
   - Freeze a forecast in `forecasts.jsonl` (value, band, mechanism, falsifier), then make one change in a new config, or a bundle whose levers each have a forecast and can be told apart by per-stage timing.
   - Run it between two controls with `scripts/queue.sh ctl cand ctl`.
   - Score the forecast, accept or reject the change, and log the numbers and the reason in `PLAN.md`.
   - Never rerun something `PLAN.md` already rejected without a new reason.
3. **Gate every new best:**
   - long-context needle retrieval at ~50%, then ~100% of the context;
   - a concurrent full-length soak if the KV pool is sized for N users;
   - a greedy equivalence check for changes claimed to be lossless.
4. **Safety, always:**
   - one GPU job at a time;
   - an AC power and performance-profile guard; on a shared host, the GPU plus a fixed core set booked and host load recorded per run;
   - the memory watchdog on;
   - drain requests before stopping a server, never force-kill mid-batch;
   - long queues run as harness background tasks or service-manager units, never `nohup`/`setsid`.
5. **Credentials:** never prompt for, store, print or commit one, or put its path in anything published. Use what the environment already provides; if something is missing, stop and say what is needed.
6. **Cadence:** if told "don't stop", keep looping. When three levers in a row land inside noise, write a STATUS/plateau section in `PLAN.md` and keep going at a slower cadence: profile, then try kernel-level levers. Otherwise, hand back at the plateau.
7. **Afterwards:** restore the best server on its usual port, unless the user asked for the GPU to be freed.

## How to report

Every report, whether progress or final, starts with this table:

| Metric | Baseline | Now | Ceiling | How measured |
|---|---:|---:|---:|---|
| c1 decode | | | | median of N, thinking on/off, prompt panel, temperature |
| cN decode | | | | wall-clock aggregate |
| Prefill 4K / 32K / long | | | | cold unique prompt, prompt_tokens ÷ TTFT |
| KV capacity | | | | tokens, and N × context proven by soak |
| Startup / TTFT | | | | |
| Peak memory | | | | driver residency + anon, vs cap |

Then:

- **What got the gains,** one line each, with the measured effect.
- **What limits further gains,** with evidence: profile numbers, acceptance by depth, percentage of ceiling per component.
- **Forecast calibration:** how many forecasts landed inside their bands, and what the misses taught.
- **Contaminated runs,** if any, and what changed on the host.
- **Known limitations,** e.g. decode stalls during a long prefill.
- **System state you changed:**
  - power profile;
  - running containers;
  - uncommitted code;
  - branches that must not be deleted;
  - files to clean up.
- **What you did *not* verify.**

Rules for the report:

- Use medians, not best runs. Label single runs.
- Never mix numbers from different harnesses without saying which is which.
- If asked to publish, follow `references/publishing.md`:
  - land the code on a pinned commit;
  - build cleanly from pins;
  - run the acceptance gates;
  - prove the clean build matches the best;
  - grep for secrets;
  - open a PR from the current main.

## When the user says "take it down" or "free the GPU"

1. Abort and drain requests.
2. Stop the server.
3. Confirm no model processes remain and the GPU is idle.
4. Stop any loop.
5. Mention any power profile or setting that is still changed.