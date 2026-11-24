# PLAN: the hill-climb log

Starting point: stock engines on the same machine, 32.4 tok/s decode (oMLX 0.8.2), 31.1 (stock mlx-dspark), 35.7 (MTPLX 2.13.2), 32.7 (mlx-serve 26.11.4-pre1), 20-26 (llama.cpp and stock mlx-vlm).
Ceiling from THEORY.md: 54.6 tok/s at bf16 attention, about 83 at 8-bit attention.

## Lever order and what each bought

| Lever | Why it was next | Result |
| --- | --- | --- |
| NVFP4 experts with exact FP16 block scales | Expert read is 1.84 GB of 6.74 per token | Accepted. Experts stopped being the dominant read without a gate regression. |
| 8-bit attention and LM head | Dense is 59% of the per-token read | Accepted. Largest single lever; takes the ceiling from 54.6 to about 83 tok/s. |
| FP16 fused Metal kernels, templated | 48 layer visits with several launches each | Accepted. Prefill 280 to 350-plus tok/s, decode up on every profile. |
| 32k pruned draft vocabulary | Draft head cost dominated short edits | Accepted. Code edits moved from 39 to 71-76 tok/s on the fast screen. |
| Lookup drafts with an automatic cap | Literal matches are cheap accepts on file edits | Accepted for agent and multi profiles. Plain profile keeps speculation off. |
| Boundary snapshot for prefix reuse | 21K second turn cost 20.5 s | Accepted. Warm TTFT 0.71 s, 28x better, at no measurable decode cost. |
| Shared-prefix global KV | Four 23.5K branches duplicated the prefix | Accepted in the multi profile. Saves about 8 GiB and keeps 4/4 branches ok. |

## Tried and rejected, with the numbers

| Change | Measurement | Verdict |
| --- | --- | --- |
| Draft length 1 and 3 instead of the default | Edits landed inside the 74.3-76.6 tok/s control band; draft 3 prose fell to 35.3 tok/s | Rejected, inside noise or slower |
| One threadgroup per expert route on verify batches | Bit-identical output, 9-17% slower at 2-4 verify rows, about 1% slower on independent rows | Rejected. The grouped kernel already shares expert reads. |
| Whole decode step as one compiled graph | Bit-exact, 783 compiled steps, one trace, no retraces; 40.32 tok/s against controls at 40.09 and 43.73 | Parked. Little overhead left after the fusions. |
| Fewer prompt-token loops in prefill (LoopDuet) | 1.34x prefill with 2 loops, 2.09x with 1; greedy output matched a median of 9.5 and 4.5 of 64 tokens | Rejected as lossy. Inkling was not trained for shallow prompt passes. |
| Direct single-token expert routing | Slower than the fused path | Rejected |

Engines pruned on this machine, all measured, no retest needed: oMLX 0.8.2 stock (32.4), oMLX plus the inkling-m1max-lab v2 kernels (18.3 NVFP4, 21.4 edits), MTPLX 2.13.2 (35.7, slow startup), mlx-serve 26.11.4-pre1 (32.7, drafting disabled on M2), llama.cpp (20-26) and stock mlx-vlm (about 19).

## Quiet-desktop rule

Identical recipes measured 2.5 hours apart differed by up to 11% on decode and prefill, with an identical configuration banner, the same launcher and the same MLX version. The edit screen's decode was 2.3% down and branch fan-out moved by under 6% in that comparison. Desktop drawing load is the likely cause, since that was what changed between sessions, and this data does not prove it. A controlled rerun with load recorded during the run would settle it.

Every run is kept in this directory's ledger, and the published figures are medians across runs rather than the best run. Before a timed run, check GPU time per process through `ioreg` and refuse to start while another client owns the GPU.

## Where it ended

Plain profile, 45.1 tok/s median decode at 350-358 tok/s prefill, inside 26 GiB.
Agent profile, 37.0 tok/s median decode, 59.2 tok/s across four streams, 0.71 s warm TTFT at 21K and 3/3 needle at 64K, inside 45 GiB.
Quality gate passes on the token-forcing check, the edit gate and the tool round trip.

Next levers, already scouted and queued in the experiment log: barrier-free NVFP4 expert kernels, an ordered in-place sliding-window KV append, flash decode attention for batched and long-context work, and a fused MoE epilogue. Each was rejected or parked once before on other hardware, so each needs its own control on this machine.

