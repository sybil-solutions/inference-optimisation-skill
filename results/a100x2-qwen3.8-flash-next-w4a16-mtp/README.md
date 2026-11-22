# Qwen3.8-Flash-Next W4A16 on 2x A100-SXM4-80GB, vLLM + MTP

MTP with 2 draft tokens took single-user decode from 91.9 to 131.2 tok/s (+43%) and 2-user aggregate decode from 168.8 to 215.7 tok/s (+28%), at a cost of about 4% prefill.
vLLM's own recipe for this model reports MTP "made throughput worse at every concurrency tested", which is why the box ran without it.
That result was measured for throughput on H-class GPUs and did not carry over to 1-2 users on A100.

## Setup

| | |
| --- | --- |
| Hardware | DGX A100, 2 of its A100-SXM4-80GB (NVLink), TP=2, driver 575.57.08, shared host (other tenants on other GPUs) |
| Model | Abliterated build of `Qwen/Qwen3.8-Flash-Next` (125B MoE, 512 experts, top-10, hybrid linear/full attention, 1 MTP layer), W4A16 routed experts; attention, MTP, LM head and vision kept BF16. Not publicly hosted. |
| Engine | vLLM `0.29.1rc1.dev128+gcd10ed6f9` cu129 nightly, torch 2.13.0+cu129 |
| Local patch | `ngram_embedding.py`: `Qwen4ExpPLEEmbeddingMethod.from_quant_config` returns the unquantized PLE method for compressed-tensors configs (otherwise `NotImplementedError` on load) |
| PLE | 102 GB n-gram embedding table offloaded to pinned host RAM (`VLLM_PLE_CPU_OFFLOAD=1`), required at TP=2 on 80 GB cards |
| Launch | `launch.sh` (the winning config); candidates only appended `--speculative-config` |

## Method

`scripts/bench.py` from this repo at `fda968e`, run by `scripts/queue.sh` as `base mtp1 base mtp2 base mtp3 base`, each run a fresh server boot.
c1 = median of 6 fixed prompts x 512 tokens, c2 = 2 concurrent streams, prefill on unique cold prompts (real lengths 2,839 and 21,573 tokens from server usage).
Sampling as the box's clients use it: thinking on, temperature 1.0, top_p 0.95, top_k 20.
Memory watchdog on host RAM (PLE offload pins ~130 GiB), cap 253 GiB used.
Three local changes to `bench.py`: an `api_key_file` config key (vLLM with `VLLM_API_KEY` answers 401 otherwise), acceptance read from vLLM's `/metrics` spec-decode counters, and an `alive` command so a crashed boot fails in ~30 s instead of waiting out the health timeout.

## Results

| Run | c1 tok/s | accept len | draft accept | c2 agg tok/s | prefill 2.8K | prefill 21.6K | TTFT ms | startup s (warm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 91.91 | - | - | 168.8 | 8,976 | 9,517 | 53.0 | 297 |
| mtp1 | 123.23 | 1.61 | 60.8% | 197.1 | 2,420 * | 9,152 | 54.8 | 349 |
| base | 91.87 | - | - | 168.5 | 8,958 | 9,520 | 53.4 | 294 |
| **mtp2** | **131.23** | 1.88 | 43.9% | **215.7** | 8,543 | 9,112 | 55.8 | 349 |
| base | 91.81 | - | - | 168.5 | 8,954 | 9,510 | 53.3 | 294 |
| mtp3 | 134.44 | 2.12 | 37.3% | 225.3 | 8,581 | 9,090 | 56.1 | 313 |
| base | 91.99 | - | - | 168.8 | 8,962 | 9,513 | 53.5 | 295 |

\* Single outlier on the first long request after boot; not repeated.

The four base controls span 0.2%, so every MTP gain is far outside noise.
mtp3 has the highest median, but its six prompts range from 126 to 151 tok/s and it beats mtp2 on only 3 of 6, so mtp2 was chosen for production pending a 12-sample rematch.
The prefill cost matches a vLLM warning at boot: with speculative decoding on, `max_num_scheduled_tokens` drops to 2048.

## Ceilings (`THEORY.md`, `flash-ceiling.json`)

Bytes per token at c1 come to ~9.9 GB (1.33 GB active experts, 7.28 GB dense incl. linear attention, 1.27 GB LM head), so the bandwidth ceiling is ~409 tok/s on spec-sheet bandwidth.
Base decode sat at ~22% of that, so the step is overhead-bound (two TP all-reduces per layer across 48 layers, hyper-connection mixers, the per-token PLE gather from host RAM).
That is the regime where verifying k+1 tokens costs about one decode step, and why MTP pays here.
Prefill at long context was already ~85% of its compute ceiling.

## Gates (production launch, MTP on)

Run against `launch.sh` as deployed, two waves (`gates.jsonl`, last row of `ledger.jsonl`):

| Gate | Result |
| --- | --- |
| Load, chat | pass x2 |
| Reasoning | pass x2: `391` in content, reasoning in its own field, no `<think>` leak |
| Tools | pass x2: model called `get_secret`, then quoted `ZX-4471-QM` from the tool result |
| Vision | pass x2: 64x64 solid red PNG answered "Red" |
| Context | 3/3 needles at 210,103 tokens (80% of the 262,144 window), 23.9 s prefill (8,797 tok/s), 121.7 tok/s decode at that length |
| Speed | c1 135.6 tok/s (median of 6), accept length 1.97, c2 223.1 tok/s, TTFT 52.8 ms |

Host RAM peak 171 GiB used during the run.

## Costs

- Prefill about 4% slower (above).
- KV pool 2,322,270 -> 1,787,213 tokens (-23%), so 8.9 -> 6.8 full-length 262k streams. The MTP layer and its BF16 weights take that memory at the same `--gpu-memory-utilization 0.90`.
- Startup +50 s warm (295 -> 349 s).

## Not verified

- `--max-num-batched-tokens 8192` to win the prefill back.
- No concurrent full-context soak (the box serves 1-2 users).
- Bandwidth and FLOPs were not measured; ceilings use spec-sheet values.
- No greedy-equivalence check; MTP with rejection sampling is lossless by construction.
