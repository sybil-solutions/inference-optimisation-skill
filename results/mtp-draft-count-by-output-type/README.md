# MTP draft count by output type: fresh text vs code edits vs real agent turns

Where #16 and #17 left off: on Qwen3.8-27B-class models with MTP on 2× RTX 3060 (llama.cpp v0.7.0, tensor split), deeper drafting won the needle test (copy-from-context answers) but not freshly written prose. This adds two benchmarks that pin down the answer for an agent workload.

Setup as in `results/arete-27b-2x-rtx3060`: Arete-27B IQ4_XS + grafted MTP head, 128K, q4_0 KV, temperature 1.0 (top-p 0.95, top-k 20), thinking off. Every candidate between two controls, 2 rounds.

## 1. Code-reproducing output (`editsweep.py`)

Context = filler document + a real ~260-line Python file (`bench.py`). Three tasks that answer by reproducing code with small changes (rewrite a class with type hints, return a function with one substitution, an `old_string`/`new_string` edit), 700 tokens each, prefix cached. Copy-heavy output decodes very consistently (control spread ±1%).

| Drafts | 3K | 16K | 32K | 64K | 100K | runs |
|---|---|---|---|---|---|---|
| 3 (control) | 77.0 | 70.9 | 65.4 | 56.5 | 48.9 | 7 |
| 5 | **88.9** (+15%) | **83.0** (+17%) | **76.8** (+17%) | **67.3** (+19%) | **59.7** (+22%) | 2 |
| 4 | 84.9 (+10%) | 78.9 (+11%) | 73.0 (+12%) | 63.2 (+12%) | 55.4 (+13%) | 2 |
| 2 | 67.8 (−12%) | 62.3 | 56.6 | 48.1 | 41.0 (−16%) | 2 |

On freshly written prose and code (`ctxsweep.py`, numbers in #16), 4 and 5 drafts did not win, and 5 lost 11–17% at 15–31K (one sweep).

## 2. Replayed real agent turns (`replay.py`)

Twelve fixed points from real local Claude Code sessions, each just after a tool result. The preceding conversation is rendered as plain chat (tool calls as `<tool_call>` JSON, tool results as user text, hidden thinking dropped) and trimmed from the front to three buckets (real prompt sizes 15–23K, 35–40K and 72–87K tokens). The model then writes the next turn itself, 500 tokens: tool calls, edits, new code and prose in their real proportions. Only aggregate speeds are published (`replay.jsonl`); the transcripts stay on the machine.

Per-point speeds vary a lot (25–70 tok/s), so the comparison is **paired per point** (each point's mean for the candidate ÷ its mean for the control, 5 control runs and 2 per candidate):

| Drafts | Paired ratio vs 3 drafts (12 points) | 95% CI | 16K / 32K / 64K buckets |
|---|---|---|---|
| 4 | **+2.6%** | −3.8% to +9.6% | +4.5% / +1.9% / +1.6% |
| 5 | −1.9% | −9.0% to +5.8% | −0.1% / −13% / +8.8% |

## Conclusion

Draft depth amplifies whatever the output is made of. Copying from the context rewards deep drafts (+15–22% for 5), and fresh writing punishes them. A real coding agent's turns mix the two, and on those 3 drafts is within noise of the best. So a single server-wide draft count should be chosen on replayed real traffic, not on a needle or a prose panel. If a workload is mostly edits (for example an apply-patch worker), a separate preset with 4–5 drafts is worth +10–20%.

## Files

- `editsweep.py`, `ctxsweep.py`, `replay.py`: run from an experiment directory with `configs/NAME.json` (same format as `bench.py`). They import `bench.py` from `../../skills/inference-optimisation/scripts` or `$SKILL_SCRIPTS`. `replay.py` reads Claude Code transcripts from `~/.claude/projects/*/*.jsonl`; adapt `render()` for other agents' logs.
- `editsweep.jsonl`, `replay.jsonl`: per-run medians and per-request decode rates.
