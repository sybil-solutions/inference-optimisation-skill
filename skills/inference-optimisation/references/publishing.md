# Qualifying and publishing a tuned recipe

A tuned config is useful only if someone else can rebuild it and get the same result.

## 1. Land the code

- Commit kernel and engine-plugin changes to a branch of a public repo. A published build fetches a pinned commit, never a local patch or a dirty tree.
- Put data files the config needs (draft vocabularies) in the recipe source; copy them into the image.

## 2. Pin and build cleanly

- **Pin everything:** base image by digest; engine and kernel source by commit; Python packages by version (fail the build if `pip freeze` differs); weights by repository revision.
- **Build from a clean checkout of the exact recipe commit;** keep the build log as evidence.
- **Build every library with the benchmarked flags;** never load two builds of one op library in a process.

## 3. Write the launch exactly as benchmarked

- Copy environment and arguments from the best config; diff mechanically; change nothing "while you're there".
- Keep the blocking startup check in the published launch: it fails if the server exits or misses its health deadline.
- Make advertised numbers true. If a registry derives KV capacity as `ctx × seqs (+ margin)`, set the engine's pool to that value, and note any difference from what you soak-tested.

## 4. Acceptance gates (all required, on the clean build)

1. **Load:** healthy, and `/v1/models` lists the model.
2. **Chat:** a short answer with a stop finish.
3. **Reasoning:** thinking separated from content, correct answer.
4. **Tools:** a full round trip: the model calls the tool, gets the result, quotes a code from it.
5. **Context:** a needle in a document at ~85% of the window, answered exactly.
6. **Speed:** single-user decode over the first 30 s after the first token, no output cap, server tokenizer, above a floor.
- Also check **vision** if the model has it, and a **concurrent full-length soak** if the recipe advertises N users at full context.
- Gate more than once if acceptance ramps up (draft warm-up); never "validated" from one wave.
- Record peak memory (driver residency plus anon).

## 5. Prove the published build equals the best

Run your harness back to back on the clean build and the hill-climb best, with interleaved controls. They should match within noise; if not, find the difference before publishing.

## 6. Publish

- **Branch from the current main;** squash-merged repos don't contain your old branch history.
- **Keep branches holding a pinned build commit,** unless main or a PR ref reaches it.
- **Commit the evidence:** gate JSON, rendered launch and hash, build and server logs, memory samples, vision response, test logs, `PLAN.md`, `THEORY.md`, `forecasts.jsonl`, ledger.
- **Grep the bundle** for secrets, credential-file paths, tokens, personal emails, hostnames and home-directory usernames before pushing.
- **State in the README** which numbers come from the gates and which from the hill-climb harness, synthetic or real traffic, and how each was measured.

## 7. Results card or poster (optional)

Medians, honest `was X → now Y (+Z%)` deltas, cold/warm and single/median labelled. With a house template, change only values. Render headless:
  ```bash
  chromium --headless=new --force-device-scale-factor=2 --window-size=1024,574 --screenshot=out.png file://$PWD/card.html
  ```
