#!/usr/bin/env python3
"""Decode tok/s vs context length on the running test server (127.0.0.1:8181).

  python3 ctxsweep.py <config-name> <round>

For each context length, one filler document is sent with three prompts from bench.py's fixed panel appended
("ignore the document, do this instead"), so the document prefix is prefill-cached after the first prompt and
each measurement is decode at that real context. Writes one JSON line per config to ctxsweep.jsonl.
"""
import json, os, statistics, sys, time
sys.path.insert(0, os.environ.get("SKILL_SCRIPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../skills/inference-optimisation/scripts")))
from bench import Server, PROMPTS, document, looping

name, rnd = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"configs/{name}.json"))
srv = Server(cfg)
# requested sizes: bench.document() yields ~0.66 real tokens per requested token on this tokenizer
TARGETS = [("0k", 0), ("16k", 24000), ("32k", 48500), ("64k", 97000), ("100k", 152000)]
res = {"name": name, "round": rnd, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "ctx": {}}
flags = []
srv.chat("Say hello.", 16)  # warm-up
for label, req in TARGETS:
    doc = document(req, seed=100 + req)[0].rsplit("\n\nList every secret code", 1)[0] if req else ""
    rates, ptoks = [], []
    for p in PROMPTS[:3]:
        prompt = (doc + "\n\nIgnore the document above. Do this instead:\n" + p) if doc else p
        r = srv.chat(prompt, 512)
        if r["decode"]:
            rates.append(round(r["decode"], 2))
        ptoks.append(r["prompt_tokens"])
        if looping(r["text"]):
            flags.append(f"LOOP@{label}")
    res["ctx"][label] = {"prompt_tokens": max(ptoks), "decode_median": round(statistics.median(rates), 2) if rates else None,
                         "decode_all": rates}
    print(time.strftime("%H:%M:%S"), name, label, res["ctx"][label], flush=True)
res["flags"] = sorted(set(flags))
with open("ctxsweep.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
