# Arete-27B on 2× RTX 3060 12 GB, llama.cpp CUDA (2026-11-16/17)

A hill-climb run with this skill's method and `bench.py`: four queues of about 90 benches, every candidate between two controls.

## Setup

| | |
|---|---|
| GPUs | 2× RTX 3060 12 GB (Ampere), measured 334 GB/s copy each; fp16 GEMM 8192³ 19.6 / 26.6 TFLOPS |
| Host | i9-9980XE, 64 GB, Arch Linux, driver 615.78, CUDA 13.4 |
| Engine | llama.cpp v0.7.0 (620d2225c), CUDA, sm_86, `--split-mode tensor --tensor-split 1,1`, flash attention on, `--parallel 1` |
| Model | NousResearch Arete-27B (Qwen3.8-27B fine-tune), bartowski IQ4_XS, with Qwen3.8's MTP head grafted on. 64 layers + 1 MTP; full attention every 4th layer, so 16 KV layers (4 KV heads × 256) |
| Control | 128K context, q4_0 K and V, MTP 3 drafts, vision projector on GPU 0 |
| Bench | `bench.py`, 8 decode reps of 512 tokens, temperature 1.0 (top-p 0.95, top-k 20), thinking off; needle (`--long`) on every run in the later queues |

## Results

Decode (c1, median of 8) per run; controls ranged 52.6–59.8 across 37 runs (median 55.7).

| Candidate | Decode, round 1 / round 2 | Decode at ~100K (needle answer) | Verdict |
|---|---|---|---|
| MTP 2 drafts | 57.6 / 59.3 (beat both neighbours twice) | 35.7 vs ~39.8 | rejected: −10% at long context |
| MTP 1 draft | 50.0 / 49.2 | 35.6 at 68K | rejected |
| no MTP | 34.9 / 34.9 | 22.9 at 68K | rejected (MTP ≈ 1.6×) |
| MTP 4 drafts | 52.0 / 52.7 | **48.0 / 47.5** vs 38.9–40.1 | wins only copy-from-context output (see the sweep) |
| MTP 5 drafts | sweep only | | no gain |
| `--spec-draft-p-min` 0.5 / 0.75 | 41.9, 42.9 / 36.5, 37.1 | ~39.8 (unchanged) | rejected |
| KV K-q8/V-q4, 128K | 55.2 / 57.2 | 51.5 at 68K (= control) | no difference |
| KV q8_0, 96K | 55.0 / 56.4 | | no difference |
| KV f16, 48K | 52.6 / 58.4 | | no difference |
| KV q8_0, 128K | OOM at load | | |
| `--ubatch-size 1024` | decode 54.4 / 55.7 / 55.1 / 53.7 | prefill 597 vs ~580 at 102K | **accepted**: prefill +3.4% at 4K/32K, +2.5% at 102K; +0.9 GiB VRAM; image after a 104K prompt OK |
| `--ubatch-size 2048` | OOM at load | | |
| `--split-mode layer` | 33.7 / 35.6 | 24.8 | rejected |

### Decode vs context with freshly written output (`ctxsweep.jsonl`)

Three panel prompts appended to a filler document (document prefix cached), 512 tokens each, median per context:

| Drafts | ~0 | 15K | 31K | 63K | 99K | runs |
|---|---|---|---|---|---|---|
| 3 (control) | 54.6 | 50.1 | 45.4 | 39.1 | 33.2 | 28 |
| 2 | 56.1 | 48.8 | 43.7 | 37.9 | 31.6 | 2 |
| 4 | 57.5 | 48.0 | 42.3 | 37.9 | 32.2 | 1 |
| 5 | 51.4 | 41.8 | 40.3 | 36.7 | 32.4 | 1 |

The control's spread per column is about ±7% (for example 30.6–36.2 at 99K), so none of these beats 3 drafts. The +20% for 4 drafts in the needle runs comes from the answer quoting the document, which MTP drafts almost always guess right. (The 28 control sweeps are partly an accident: a loop bug in my own driver repeated the control. They are still valid control runs.)

## Ceilings (from `references/ceilings.md`)

- Decode without speculation: ~14.9 GB read per token (blocks 13.86 + LM head 1.04), split over two GPUs, so ~45 tok/s at 334 GB/s. Measured no-MTP 34.9 ≈ 78%.
- Prefill: ~54 GFLOP/token over ~46 TFLOPS combined ≈ 850 tok/s at 100% of measured; measured ~770–800, close to the ceiling.

## Files

- `ledger.jsonl`: every `bench.py` run (names as in the table; `ctl` = control, `ub1024` = ubatch 1024, and so on).
- `ctxsweep.jsonl`: the decode-vs-context sweep.
- `vision.jsonl`: image probes per run, including an image after a ~104K prompt. Early entries are failures of my own probe script, not of the server.

Harness notes: `bench.py`'s filler document came out at ~0.66 of the requested tokens on this tokenizer (#11 fixes that). The server needs an API key, which this PR's `api_key_env` handles.
