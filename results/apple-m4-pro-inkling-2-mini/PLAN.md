# Inkling 2 Mini on a Mac mini (M4 Pro, 48 GB, 273 GB/s) — hill-climb log
Skill: this repo @ fda968e. Engine: Ollama v0.40.17-rc3 MLX runner, port 11436. Quant: NVFP4/MXFP8 mixed (native), 20 GB.
## Ceilings (THEORY)
- Decode: 5.4B effective-active params (3.4B physical + 3 loop passes) x ~0.76 B/param (20 GB / 26.4B) ~= 4.1 GB read/token; 273 GB/s x ~0.8 achievable / 4.1 GB ~= 53 tok/s without speculation; ~1.5-1.8 accepted/step with the shipped MTP drafter -> ~80-95 tok/s upper bound.
- Prefill: 2 x 5.4B = 10.8 GFLOP/token; M4 Pro GPU ~17 TFLOPS fp16 peak x 50% -> ~790 tok/s.
## Baseline (ctl-a, 2026-11-20 17:2x, host not idle: other background jobs running)
c1 decode median 23.1 tok/s (14.8-29.2, 6 reps) = 44% of no-spec ceiling; prefill 4K 253, 16K 251 tok/s (32% of ceiling); needle 3/3 at 18.8K tokens; peak mem 25.6 GiB; TTFT short 445 ms. Speculative acceptance in server log ~0.5, avg_draft ~3 (max 7).
## Experiments
- d4 / d2 (Modelfile PARAMETER draft_num_predict 4 / 2) vs interleaved controls ctl-b 22.35, ctl-c 24.56, ctl-d 24.19: d4 23.29, d2 24.38 -> inside noise (+-10%), REJECT. Server log still reports max_draft=7, so the parameter may not be honoured by the MLX runner at all. Acceptance ~0.43-0.56 on prose.
- d0 (draft_num_predict 0, speculation maybe off) 26.45 vs ctl-e 21.78 / ctl-f 23.91: per-prompt gains small and inside noise; unproven (could not confirm the parameter disables the MTP drafter). NOT adopted. Needs a quiet host and a code/JSON panel.
- PARALLEL: baseline c2 aggregate 18.14 tok/s (server then at parallel=1) -> par-c2 29.45 and 28.66, par-c4 28.71. +60% aggregate at c2 (decode is bandwidth bound; two streams share weight reads). c4 no better than c2. ACCEPTED: OLLAMA_NUM_PARALLEL=2 pinned in run-inkling.sh. Peak mem 25.5-28.4 GiB (cap 36).
- Reasoning effort (API reasoning_effort): low cuts completion tokens 2-3x with same pass/fail on 3 small tasks (classify 666->213 tok, extract 257->183, math 267->149 where tasks passed); 'none' only ~1.5x. Efficiency lever for agents; not a tok/s lever. Quality on harder work NOT measured.
## Gates (server restarted under run-inkling.sh: ctx 65536, NUM_PARALLEL=2, KEEP_ALIVE=-1)
- gate-50: 31,285-token prompt, needles 3/3, prefill 216 tok/s, decode 31.4 tok/s (repetitive text), peak mem 25.6 GiB.
- gate-96: 62,553-token prompt (96% of 65,536), needles 3/3, prefill 173 tok/s, decode 16.5 tok/s, peak 28.9 GiB (cap 36). No flags (no loop, no failed requests, AC power).
- NOT run: concurrent full-KV soak (2 x 62K at once, ~12 min of prefill) and a greedy-equivalence check (nothing lossy was adopted).
## STATUS
Plateau for config levers on this engine: draft length (2/4) and spec-off inside noise; parallel=2 was the only accepted win (+60% aggregate at c2). Decode c1 23-24 tok/s = ~44% of the 53 tok/s no-spec ceiling; prefill ~250 tok/s = ~32% of ~790 ceiling. Remaining gap is engine/kernel level (Ollama MLX runner; looped-layer weight re-reads; other daemons share the GPU/bandwidth) — out of scope. Host was never idle (background jobs), so noise is high (+-10%).
Next if wanted: (1) llama.cpp GGUF / mlx-lm 4-bit comparison on the same prompts (lossy 4-bit vs native NVFP4, ~16 GB download each); (2) wired-memory limit (needs sudo); (3) per-agent reasoning_effort=low defaults.
