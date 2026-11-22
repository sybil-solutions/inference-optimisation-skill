# Qwen3.8-27B UD-Q4_K_XL (native MTP) on 2× RTX 3060 12 GB, llama.cpp CUDA (2026-11-22)

The same base model and machine as `results/arete-27b-2x-rtx3060` (#16), but Qwen's own MTP head instead of a grafted one, the larger unsloth UD-Q4_K_XL quant, and 96K context so the vision projector fits on GPU 0. The questions: does a native draft head change the best draft count, and do the levers that worked there fit here?

## Setup

| | |
|---|---|
| GPUs | 2× RTX 3060 12 GB, measured 334 GB/s copy each |
| Engine | llama.cpp v0.7.0 (620d2225c), CUDA sm_86, `--split-mode tensor --tensor-split 1,1`, flash attention on, `--parallel 1` |
| Model | unsloth Qwen3.8-27B-UD-Q4_K_XL with its MTP head; mmproj F16 on GPU 0 |
| Control | 96K context (98,304), q4_0 K and V, MTP 3 drafts |
| Bench | `bench.py`, 8 decode reps of 512 tokens, temperature 1.0 (top-p 0.95, top-k 20), thinking off; needle on every run (87,374 real tokens); a tool call and a 1920×1080 image after every run; an image after an 87.5K-token prompt on the controls |

## Results

Controls (9 runs): c1 47.9–50.9, median 50.7; needle 3/3 every run; long-context decode 39.3–42.8.

| Candidate | c1, round 1 / round 2 | Decode at 87K (needle answer) | Verdict |
|---|---|---|---|
| MTP 2 drafts | 50.6 / 50.5 | 36.6 / 36.9 vs 42.5–42.8 | rejected (−14% long) |
| MTP 4 drafts | 48.4 / 49.2 | 45.1 / 45.3 vs 39.6–42.8 | rejected for general use (−3 to −5% short; wins only copy-from-context answers) |
| q8_0 KV, 96K | OOM at load | | |
| `--ubatch-size 1024` | healthy, then OOM at the first request (`cudaGraphInstantiate`, device 0) | | does not fit (it fit on #16's smaller IQ4_XS quant) |

### Decode vs context with freshly written output (`ctxsweep.jsonl`)

| Drafts | ~0 | 16K | 32K | 64K | runs |
|---|---|---|---|---|---|
| 3 (control) | 48.7–52.5 | 40.7–49.2 | 41.2–43.5 | 33.8–40.0 | 5 |
| 2 | 53.1 / 53.4 | 45.4 / 48.7 | 43.7 / 42.4 | 35.9 / 35.2 | 2 |
| 4 | 48.7 / 46.4 | 45.3 / 44.3 | 40.6 / 40.7 | 34.8 / 33.1 | 2 |

The native head behaves like the grafted one in #16. 2 drafts are a little faster when the context is short and slower when it is long. 4 drafts win only on answers that copy from the context. 3 drafts is still the best all-round choice for an agent that works at 20–90K.

## Findings that went into gotchas.md

- With `--split-mode tensor`, llama.cpp silently samples on the CPU (`backend sampling not supported with SPLIT_MODE_TENSOR`). That applies to every run here and in #16.
- A config can pass the health check and still OOM at the first request (CUDA graph instantiation), so gate on real requests.

## Files

`ledger.jsonl` (bench runs), `ctxsweep.jsonl` (decode vs context), `toolcall.jsonl`, `vision.jsonl` (image probes, including after an 87.5K prompt). The `ok: false` vision rows are the configs whose server had died (ub1024, kvq8).
