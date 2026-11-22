# A/A noise floor: 45 runs of one unchanged config (Apple M4 Max, llama.cpp)

A check of the accept rule, not a speed-up. One config was run 45 times back to back with the stock
`bench.py` (commit `fda968e`), nothing changed between runs. Every "win" is therefore a false positive.

## Setup

| | |
|---|---|
| Hardware | Mac Studio, Apple M4 Max (12P + 4E cores), 64 GB unified memory, AC power, macOS 27.0.1 |
| Engine | llama.cpp `llama-server` version 8980 (41a63be28), Metal, `--ctx-size 8192 -ngl 99`, 4 slots |
| Model | Orpheus-3b-FT, GGUF Q4_K_M (3.78 B parameters, 2.36 GB; a Llama-3.2-3B fine-tune that was already loaded) |
| Harness | `bench.py run` unchanged: `concurrency` 2, `prefill_lengths` [512, 1024], `temperature` 0.6, 6 decode samples per run |
| Host | A desktop in normal use, not an idle box. Four runs dipped while it was busy; they are kept. |

## Result

| Measure | Value |
|---|---:|
| Runs | 45 (about 17 s each, 16:22 to 16:35 local) |
| c1 decode, median of run medians | 164.3 tok/s |
| Run medians, min to max | 141.9 to 166.6 tok/s |
| Run medians, p10 to p90 spread | 2.9% |
| Middle runs that beat both neighbours | **15 of 43 (35%)** |
| ... and also regressed nothing else by more than 3% | 11 of 43 (26%) |
| Largest no-change win over the better neighbour | 1.54% |
| Middle runs that beat both neighbours by more than 2% | 0 of 43 |

Three identical runs have a fastest one, and it is the middle one about one time in three. So "beats both
neighbouring controls" keeps about a third of changes that do nothing. A margin sized from the A/A series
removes them: here 2% was enough. The price is sensitivity. Replaying these 43 triplets with a pretend
speed-up added to the middle run, a real 2% gain clears a 2% margin 35% of the time, 3% clears it 65%,
4% clears it 86% and 5% clears it 93%.

Single-request metrics move more than the decode median: `prefill_512` had a p10 to p90 spread of 8.6%
(min to max 21%), and `c2_aggregate` 7.8%. `bench.py` times each prefill length once per run.

## Files

- `ledger.jsonl`: the 45 rows as `bench.py` wrote them.
- `aa-output.txt`: `python3 scripts/aa.py ledger.jsonl`.

## Limits

One machine, one small model, one engine, a short prompt panel. The size of the floor will differ
elsewhere; measure it on your own box. Not run: the long-context needle, the memory watchdog, a soak.
The model is a speech fine-tune, so its replies were used for timing only.
