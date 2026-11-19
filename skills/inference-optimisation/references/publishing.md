# Qualifying and publishing a tuned recipe

A tuned config is only useful if someone else can rebuild it and get the same result.

## 1. Land the code

- Commit kernel and engine-plugin changes to a branch of a public repo. A published build must fetch a pinned commit, never a local patch or a dirty tree.
- Put data files the config needs (draft vocabularies, configs) in the recipe source. Copy them into the image.

## 2. Pin and build cleanly

- **Pin everything:**
  - the base image by digest;
  - the engine and kernel source by commit;
  - Python packages by version (fail the build if `pip freeze` differs);
  - weights by repository revision.
- **Build from a clean checkout of the exact recipe commit.** Keep the build log as evidence.
- **Build every library with the same flags you benchmarked.** Never load two builds of the same op library in one process.

## 3. Write the launch exactly as benchmarked

- Copy the environment and arguments from the best config, then diff them mechanically. Change nothing "while you're there".
- Make advertised numbers true. If a registry derives KV capacity as `ctx × seqs (+ margin)`, set the engine's pool to that value. Note any difference from what you soak-tested.

## 4. Acceptance gates (all required, on the clean build)

1. **Load:** the server becomes healthy, and `/v1/models` lists the model.
2. **Chat:** a short answer with a stop finish.
3. **Reasoning:** thinking separated from content, with the correct answer.
4. **Tools:** a full round trip. The model calls the tool, receives the result, and quotes a code from it.
5. **Context:** a needle in a document at about 85% of the window, answered exactly.
6. **Speed:** single-user decode over the first 30 s after the first token, with no output cap, counted with the server tokenizer, above a floor.
- Also check **vision** if the model has it, plus a **concurrent full-length soak** if the recipe advertises N users at full context.
- Gate the same config more than once if acceptance ramps up (draft methods with warm-up). Never publish a "validated" result from a single wave.
- Record peak memory: driver residency plus anon on unified memory.
- **Agent tasks (optional, recommended for lossy recipes).** The gates above prove the server answers; they do not prove an agent on top of it still finishes real work. Run a code-graded agent benchmark through a harness (opencode, hermes, ...) whose model is the clean build, and the same harness and version on the baseline. Publish the pass count and the end-to-end wall time with the harness name.
  - Example: the [airbench.ai](https://airbench.ai) quick test, 15 challenges (math, vision, code), graded server-side. An agent registers itself for a test key with a one-second proof of work and reads back `passed / total / wallTimeMs`; the [`local-inference-optimisation`](https://github.com/dh7/airbench-skills) skill wraps it. The orchestrating agent must hand the briefing to the harness and never solve the challenges itself.
  - Agent runs are noisy. In a calibration of 5 runs per hosted model through one fixed harness, the same model moved by up to 3 challenges (12 to 15), and a weaker model occasionally scored 0 because it gave up on the harness. A short test like this separates tiers (a drop of 3 or more), not near neighbours, and a near-frontier model sits close to 15/15.
  - So: void a run that ends with challenges still pending (that is the harness, not the server). Run each config at least three times, alternating order. Reject when the candidate's best clean run is below the control's worst and the means differ by more than 2. Treat it as a gate, not as a hill-climb metric.
  - Measured example: RTX 5090, Qwen3.8-27B NVFP4, vLLM 0.27.1. `--kv-cache-dtype fp8` vs `auto` gave the same c1 decode (86.7 vs 86.9 tok/s) and +3.5% KV tokens, and scored 9 and 13 /15 against 14 and 15. Flagged, not yet rejected: the 9 is outside the noise band, the 13 is inside it, and two runs per config is below the three this rule asks for.

## 5. Prove the published build equals the best

- Run your harness back to back on the clean build and your hill-climb best image, against interleaved controls. They should match within noise.
- If they don't, find the difference before publishing.

## 6. Publish

- **Branch from the current main.** Squash-merged repos don't contain your old branch history.
- **Keep any branch that holds a pinned build commit alive.** Deleting it breaks reproducibility, unless the commit is also reachable from main or a PR ref.
- **Evidence to commit:**
  - the gate run JSON;
  - the rendered launch and its hash;
  - build and server logs;
  - memory samples;
  - the vision response;
  - the check and test logs;
  - the experiment log (`PLAN.md`, `THEORY.md`, ledger).
- **Grep the bundle** for secrets, credential-file paths, tokens, personal emails and home-directory usernames before pushing.
- **State in the README** which numbers come from the gates and which from the hill-climb harness, and how each was measured.

## 7. Results card or poster (optional)

- Use medians, not best runs, and show honest `was X → now Y (+Z%)` deltas.
- Label cold vs warm and single runs vs medians.
- If you have a house template, change only the values. Render headless:
  ```bash
  chromium --headless=new --force-device-scale-factor=2 --window-size=1024,574 --screenshot=out.png file://$PWD/card.html
  ```
