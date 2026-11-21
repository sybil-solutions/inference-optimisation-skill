# Qwen3.8-27B-pi-adapted on Apple M4 Max

On Apple M4 Max with 128 GiB of unified memory, depth-3 multi-token prediction (MTP) raised median short decode from **15.29 to 33.13 tokens/s** and cut median Pi edit time from **10.67 to 6.31 seconds**. Both configurations passed the three edit fixtures and retrieved all three planted codes at both tested lengths. Fresh first-token latency increased about 3–4%.

## Configuration

- Apple Metal, [llama.cpp b12359](https://github.com/ggml-org/llama.cpp/releases/tag/b12359) (arm64, commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`), Pi 1.5.2.
- Qwen3.8-27B Q8_0 base, F16 Pi LoRA at scale 1, and the stock Q4_0 MTP draft when drafting was enabled. The server API confirmed the active adapter and scale.
- One slot, 65,536-token context, full Metal offload, Flash Attention, physical microbatch 512 for both qualification runs, and thinking disabled in the chat template, server, and Pi.

| Artifact | Pinned file | SHA-256 |
| --- | --- | --- |
| Q8_0 base | [Qwen3.8-27B-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/blob/c1869413dece09109e55b7dcc2c77ef382b0d0bd/Qwen3.8-27B-Q8_0.gguf) | `6bc9e7220d8400574fd62df25f04a961cb5cbb0d766cf0118c21c36c0059c35a` |
| F16 Pi LoRA | [Qwen3.8-27B-pi-lora-F16.gguf](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted-GGUF/blob/e8c52527e0683205fb130dbbf17f68da129547db/Qwen3.8-27B-pi-lora-F16.gguf) | `a64f91844f19e67da500d8628bdf9383aa334e61b7ab687805802e7d272d7b67` |
| Q4_0 MTP draft | [mtp-Qwen3.8-27B-Q4_0.gguf](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/blob/c1869413dece09109e55b7dcc2c77ef382b0d0bd/mtp-Qwen3.8-27B-Q4_0.gguf) | `98a72f16e648f77e15b3dcfc7828702e2211f05b95777cb27dc85cb87e621db4` |

Common server arguments use the paths to those downloaded files.

```sh
llama-server --model "$BASE_GGUF" --lora "$LORA_GGUF" \
  --alias qwen/qwen3.8-27b-pi-adapted --ctx-size 65536 --parallel 1 \
  --gpu-layers all --fit off --flash-attn on --jinja --ubatch-size 512 \
  --chat-template-kwargs '{"enable_thinking":false}' --reasoning off
```

For the depth-3 run, append `--spec-type draft-mtp --model-draft "$DRAFT_GGUF" --spec-draft-n-max 3`.

## Workload comparison

| Measurement | MTP off | MTP depth 3 |
| --- | ---: | ---: |
| Short decode, median of six | 15.29 tok/s | 33.13 tok/s |
| Pi edit, median wall time | 10.67 s | 6.31 s |
| Exact Pi edits | 3/3 | 3/3 |
| Fresh 4,001-token prompt, first visible output | 16.48 s | 16.96 s |
| Fresh 4,025-token follow-up, first visible output | 16.68 s | 17.17 s |
| Cached 4,025-token follow-up, first visible output | 0.29 s | 0.30 s |
| Fresh 16,003-token retrieval, first visible output | 69.87 s | 72.09 s |
| Fresh 61,004-token retrieval, first visible output | 332.31 s | 345.16 s |
| Codes found at each length | 3/3 | 3/3 |
| Peak Metal in-use system memory | 31.2 GiB | 33.5 GiB |
| Peak worker RSS | 38.1 GiB | 40.6 GiB |
| Swap growth | 0 MiB | 0 MiB |

The interleaved MTP-off / depth-3 / MTP-off screen measured **15.41 / 33.45 / 15.32 tok/s**. The same Pi edit took **12.88 / 8.93 / 13.26 seconds**. Depth 3 missed the screen’s 3% fresh-prefill tolerance by 0.01 percentage point against the faster control. The subsequent workload comparison quantified that cost alongside the larger decode and edit-time gains. The sweep covered draft depths 0 and 3.

With MTP off, a 1,024-token physical microbatch took 68.82 s to first visible output on the fresh 16K prompt, between 512-token controls at 70.00 and 69.88 s. That change fell short of the 5% gain threshold, so the workload runs used 512. The larger microbatch was tested with drafting off.

## Measurement method and raw data

- Ran one server at a time. Direct chat requests used temperature 0 and seed 724. The six short decode requests per run had an 80-token cap. Decode uses the server-reported completion count minus one divided by elapsed time after the first visible chunk.
- Fresh prompts disabled prefix reuse. The cache check replayed the same follow-up messages with reuse enabled and disabled. The server reported 4,003 reused tokens with caching and zero when fresh in both workload runs.
- Pi called its real read and edit tools on isolated Python fixtures. A check compared the parsed file to the expected syntax tree. The workloads used three small synthetic edits and one synthetic retrieval prompt at each length.
- Peak Metal memory is IOKit’s system-wide in-use measure. Worker RSS and swap change were sampled separately. These are single-user measurements with short, capped decode probes.

[Sweep and MTP-off workload rows](sweep-and-baseline.jsonl) · [Depth-3 workload rows](mtp-depth-3-qualification.jsonl)

The ledgers contain the per-run measurements and synthetic fixture outcomes. Free-form Pi replies were omitted. This submission reports the single-user benchmark. The uncapped 30-second decode and clean-build acceptance rerun remain unmeasured.
