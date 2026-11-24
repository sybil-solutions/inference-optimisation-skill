# THEORY: Inkling-2-Mini on a 64 GB M2 Max

## Hardware, measured

- Memory bandwidth: 368 GB/s read, 370 GB/s copy, against a 400 GB/s spec.
- FP16 GEMM at 4096^3: 9.2 TFLOPS.

## Bytes per decoded token

The checkpoint has 24 layers. Layers 6-17 run the loop core three times, so a token visits 12 + 12x3 = 48 layers.

| Component | Per token |
| --- | ---: |
| Dense weights, bf16 outer layers 0.93 GB plus 3 x 1.01 GB looped | 3.97 GB |
| Routed experts, NVFP4 and MXFP8, 8 of 128 per layer | 1.84 GB |
| LM head, bf16 | 0.93 GB |
| **Total at bf16 attention and head** | **6.74 GB** |

At bf16 the ceiling is 368 / 6.74 = **54.6 tok/s**, and the dense weights are 59% of the read. That is why attention precision, not expert precision, is the dominant decode lever on this checkpoint.

`--target-bits 8` moves attention and the head to 8-bit affine, taking the read to about **4.4 GB per token** and the ceiling to about **83 tok/s**.

## Ceilings against what was reached

| Path | Ceiling | Reached | Share of ceiling |
| --- | ---: | ---: | ---: |
| bf16 attention, plain decode | 54.6 tok/s | 32.4 (oMLX 0.8.2) | 59% |
| 8-bit attention, plain decode | ~83 tok/s | 45.1 (median) | 54% |
| 8-bit attention, speculative decode | acceptance-dependent, around 142 tok/s at an accepted length of 2.4 | 37.0 (median) | 26% |
| Prefill, short context | ~425 tok/s at 50% of peak FP16 | 319-358 | 75-84% |

Prefill is close to the compute ceiling. Decode is not, which is why the remaining work is kernel-level.

## Speculative decode arithmetic

A verify pass with k=1 reads the union of experts for 2 rows. With 8 of 128 experts routed, that union is 1-(1-1/16)^2 = 12.1% of experts rather than 6.25%, so the verify step reads about 5.6 GB plus 0.64 GB of drafter.

Break-even needs roughly 1.5 accepted tokens per step. With an accepted length of a, the ceiling is about 368 x a / 6.2 GB.

That is the shape of the trade in the ledger: lookup drafts help code edits a lot, where long literal matches are accepted, and help novel prose much less, where the accepted length sits near break-even.

## Where the remaining gap is

- **Decode:** about 54% of the 8-bit ceiling. The gap is small-M matrix-vector efficiency and launch overhead across 48 layer visits, not memory traffic.
- **Batched decode:** 4 streams aggregate 59.2 tok/s, so per-stream throughput falls as batch grows. The batch path serialises attention across rows.
- **Prefill:** within 16-25% of the FP16 compute bound, with tile shape and memory pressure as the remaining levers.

