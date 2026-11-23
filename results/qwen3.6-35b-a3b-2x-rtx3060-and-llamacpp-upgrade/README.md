# Qwen3.6-35B-A3B (MoE, experts partly in RAM) on 2× RTX 3060, and a llama.cpp upgrade across three models (2026-11-22)

Same machine as #16 / #17 / #19: 2× RTX 3060 12 GB (334 GB/s copy each), i9-9980XE + 64 GB DDR4. `bench.py` with 8 decode reps, temperature 1.0, thinking off, needle (~102K) on every run, a tool call and a 1920×1080 image after every run, every candidate between two controls, 2 rounds.

## 1. Upgrading llama.cpp v0.7.0 → 1.0.0-dev 8d60873 (same flags, same GGUFs)

| Model (preset) | Decode c1 | Prefill | Long context | Runs |
|---|---|---|---|---|
| Arete-27B IQ4_XS, tensor split, 128K, MTP 3 | **+5%** (median 56.8 vs 53.9; 3 of 4 beat both neighbours) | +4% at 32K (808–814 vs 776–780) | prefill +2%, decode +2.5% at 102K | 4 vs 5 |
| Qwen3.8-27B Q4_K_XL, tensor split, 96K, MTP 3 | **+12%** (57.95 / 57.12 vs 51.79 / 51.65 / 49.54) | +1–2% | decode +5–8% at 87K | 2 vs 3 |
| Qwen3.6-35B-A3B Q4_K_XL, layer split + auto-fit, 128K, MTP 3 | **+5%** (75.3 / 71.5, both beat neighbours) | **+14% / +12%** at 4K / 32K | prefill +8%, decode +3% at 102K; startup 13 s vs 15 | 2 vs 3 |

All gates pass on the new build: needle 3/3, tool calls and images. Both builds still sample on the CPU under tensor split (see #17). Pin and re-measure; we only compared these two points.

## 2. Qwen3.6-35B-A3B levers (auto-fit leaves some experts in system RAM)

Production preset: `--split-mode layer`, no `-ngl` (auto-fit), q8_0 KV, mmproj on CPU, MTP 3 drafts.

On v0.7.0 (control c1 68.6–75.0, median ~71.1, 15 runs):

| Candidate | c1 | Prefill 4K / 32K | Long (102K) prefill / decode | Verdict |
|---|---|---|---|---|
| `--fit-target 512` (default margin 1024 MiB) | 74.8 / 72.5 | +8% / +7% | +5% / ≈ | win |
| `--fit-target 256` | 78.0 / 71.4 | +12% / +11% | +7% / +3% | **rejected: the first image request crashed the server (OOM), both rounds** |
| q4_0 KV | 74.4 / 73.9 | +5% / +4% | +3% / ≈ | small; a quality trade |
| `--ubatch-size 1024` | 68.4 / 69.4 | **+15% / +17%** | +13% / −2% | trade: the bigger compute buffer means auto-fit keeps fewer experts on GPU, so decode drops |
| MTP 2 drafts | 73.7 / 73.9 | ≈ | ≈ / **−10%** | rejected |
| MTP 4 drafts | 70.3 / 72.5 | ≈ | ≈ / ≈ | no gain |

Combined on 8d60873 (control = production flags on the new build, c1 median 75.1):

| Candidate | c1 | Prefill 4K / 32K | Long prefill / decode | Verdict |
|---|---|---|---|---|
| `--fit-target 512` | 76.8 / 77.3 | +8% / +7% | +5% / +3% | **win** |
| `--ubatch-size 1024` | 70.3 / 67.4 (−6 to −10%) | +16% / +15% | +11% / ≈ | trade |
| ubatch 1024 + fit-target 512 | 72.6 / 69.4 | **+25% / +22%** | +16% / ≈ | **rejected: image request crashed the server, both rounds** |
| q4_0 KV | 73.2 / 74.5 | +6% / +4% | +3% / one outlier | not worth the trade |

Net, from production (v0.7.0, default margin) to 8d60873 + `--fit-target 512`: decode ~71 → ~77 (+8%), prefill 4K ~1000 → 1262 (+26%), 32K ~1185 → 1428 (+20%), 102K ~920 → 1043 (+13%), long decode ~59 → ~63.

## Release-candidate check (all three on 8d60873, before switching a production router)

| Preset | c1 vs neighbouring controls | Image after a long prompt | Tool call |
|---|---|---|---|
| Arete (ubatch 1024) | 55.0 vs 53.3 / 55.7 (v0.7.0 controls) | OK after 103,923 tokens | OK |
| Qwen3.8-27B | 54.9 vs 49.0 / 52.3 (v0.7.0 controls) | OK after 87,509 tokens | OK |
| Qwen3.6-35B-A3B + `--fit-target 512` | 77.9 vs 76.3 / 75.9 (8d60873 controls) | OK after 103,923 tokens | OK |

Router: `llama-server --models-preset <ini> --models-max 1` on 8d60873 with the production preset file switched arete → qwen-coder → qwen-fast → arete, each answering text, a tool call and an image.

## Lessons

- **Auto-fit's margin is a lever for MoE models that spill experts to RAM.** Every MiB of margin returned holds experts that otherwise decode from system RAM. Shrink it only as far as the vision path still fits.
- **Gate with the request types you serve.** Two configs that passed every text gate (needle, tool call, prefill) crashed on their first image: the vision encoder's compute buffer is allocated lazily. A config that "fits" by a few hundred MiB will fail on real multimodal traffic.
- **ubatch trades against auto-fit.** On a split-to-RAM MoE, a bigger ubatch raises prefill but takes VRAM from experts, so decode falls. Measure both.

## Files

`ledger-qwen-fast-v0.7.0.jsonl`, `ledger-qwen-fast-8d60873.jsonl` (control = production flags on 8d60873), `ledger-arete-upgrade.jsonl`, `ledger-qwen-coder-upgrade.jsonl` (`ctl` = v0.7.0; `arete-8d6` / `b8d6` = 8d60873).
