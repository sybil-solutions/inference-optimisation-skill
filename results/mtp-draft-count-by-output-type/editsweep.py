#!/usr/bin/env python3
"""Decode tok/s vs context for edit-style output (answers that quote or lightly modify code from the context).

  python3 editsweep.py <config-name> <round>

Context = filler document (sized per target) + a real source file (bench.py from the skill, ~260 lines).
Three edit tasks per context, each answered by reproducing code from the file with small changes, the way an
agent's Edit/Write calls quote existing code. Prefix (filler + file) is cached after the first task.
Writes one JSON line per config to editsweep.jsonl.
"""
import json, os, statistics, sys, time
sys.path.insert(0, os.environ.get("SKILL_SCRIPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../skills/inference-optimisation/scripts")))
from bench import Server, document, looping

SRC = open(os.path.join(sys.path[0], "bench.py")).read()
TASKS = [
    "Rewrite the whole `Server` class from the file above with type hints added to every method signature. "
    "Output only the complete class, unchanged otherwise.",
    "Output the complete `run` function from the file above, with every `log(...)` call replaced by `print(...)`. "
    "Change nothing else.",
    "Give an edit for the file above as JSON with keys old_string and new_string, where old_string is the entire "
    "`document` function exactly as written and new_string is the same function with the parameter `needles` renamed "
    "to `n_needles` everywhere inside it.",
]
name, rnd = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"configs/{name}.json"))
srv = Server(cfg)
ctx = cfg.get("ctx", 131072)
# requested filler sizes; bench.document() yields ~0.66 real tokens per requested token; the file adds ~2.7K tokens
TARGETS = [("3k", 0), ("16k", 20000), ("32k", 44000), ("64k", 92000), ("100k", 148000)]
TARGETS = [(l, r) for l, r in TARGETS if r * 0.66 + 4000 < ctx - 2000]
res = {"name": name, "round": rnd, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "ctx": {}}
flags = []
srv.chat("Say hello.", 16)
for label, req in TARGETS:
    filler = document(req, seed=200 + req)[0].rsplit("\n\nList every secret code", 1)[0] + "\n\n" if req else ""
    base = filler + "Here is a Python file:\n\n```python\n" + SRC + "\n```\n\n"
    rates, ptoks = [], []
    for t in TASKS:
        r = srv.chat(base + t, 700)
        if r["decode"]:
            rates.append(round(r["decode"], 2))
        ptoks.append(r["prompt_tokens"])
        if looping(r["text"]):
            flags.append(f"LOOP@{label}")
    res["ctx"][label] = {"prompt_tokens": max(ptoks), "decode_median": round(statistics.median(rates), 2) if rates else None,
                         "decode_all": rates}
    print(time.strftime("%H:%M:%S"), name, label, res["ctx"][label], flush=True)
res["flags"] = sorted(set(flags))
with open("editsweep.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
