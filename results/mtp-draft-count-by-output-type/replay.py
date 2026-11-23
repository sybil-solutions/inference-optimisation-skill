#!/usr/bin/env python3
"""Decode tok/s on replayed real agent turns (Claude Code transcripts on this machine).

  python3 replay.py <config-name> <round>

Picks fixed points in local Claude Code sessions (~/.claude/projects/*/*.jsonl, main thread only), each just before
an assistant turn. Renders the preceding conversation as plain chat (tool calls as <tool_call>JSON</tool_call>, tool
results as user text; hidden thinking dropped), trims it from the front to ~16K / 32K / 64K tokens, and lets the
model write the next turn itself. The mix of edits, new code and prose therefore comes from real sessions.
Transcript text is only sent to the local test server; only aggregate numbers are written to replay.jsonl.
"""
import glob, json, os, random, statistics, sys, time
sys.path.insert(0, os.environ.get("SKILL_SCRIPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../skills/inference-optimisation/scripts")))
from bench import Server, looping

SYSTEM = ("You are an expert coding agent working in the user's repository. Continue this session by writing your next "
          "message. To use a tool, write <tool_call>{\"name\": ..., \"input\": {...}}</tool_call>.")
CHARS_PER_TOKEN = 3.6           # rough for code/English on Qwen tokenizers; the real count comes back as prompt_tokens
BUCKETS = [("16k", 16000), ("32k", 32000), ("64k", 64000)]
POINTS_PER_BUCKET = 4


def render(path):
    """Linear main-thread conversation as [(role, text)], consecutive same-role messages merged."""
    out = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") not in ("user", "assistant") or d.get("isSidechain"):
            continue
        c = (d.get("message") or {}).get("content")
        parts = []
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                t = b.get("type")
                if t == "text":
                    parts.append(b.get("text", ""))
                elif t == "tool_use":
                    parts.append("<tool_call>" + json.dumps({"name": b.get("name"), "input": b.get("input")})[:8000] + "</tool_call>")
                elif t == "tool_result":
                    r = b.get("content")
                    if isinstance(r, list):
                        r = "\n".join(x.get("text", "") for x in r if isinstance(x, dict))
                    parts.append("Tool result:\n" + str(r)[:12000])
        text = "\n".join(p for p in parts if p).strip()
        if not text:
            continue
        role = d["type"]
        if out and out[-1][0] == role:
            out[-1] = (role, out[-1][1] + "\n\n" + text)
        else:
            out.append((role, text))
    return out


def points():
    """Deterministic sample per bucket: assistant turns whose history fills the bucket, spread across sessions."""
    files = sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")))
    cands = []  # (file, idx, history_chars)
    for f in files:
        conv = render(f)
        hist = 0
        for i in range(1, len(conv)):
            hist += len(conv[i - 1][1])
            if conv[i][0] == "assistant" and conv[i - 1][0] == "user" and hist >= 50000:
                cands.append((f, i, hist))
    rng = random.Random(20261122)
    picks = []
    for label, tokens in BUCKETS:
        need = tokens * CHARS_PER_TOKEN * 1.05
        by_file = {}
        for f, i, h in cands:
            if h >= need:
                by_file.setdefault(f, []).append(i)
        files_ok = sorted(by_file)
        rng.shuffle(files_ok)
        chosen = [(f, rng.choice(by_file[f])) for f in files_ok[:POINTS_PER_BUCKET]]
        while len(chosen) < POINTS_PER_BUCKET and files_ok:   # fewer sessions than points: reuse sessions, other turns
            f = rng.choice(files_ok); chosen.append((f, rng.choice(by_file[f])))
        picks += chosen
    return picks


def messages(path, idx, budget_tokens):
    conv = render(path)[:idx]
    budget = int(budget_tokens * CHARS_PER_TOKEN)
    kept, total = [], 0
    for role, text in reversed(conv):
        if total + len(text) > budget:
            if not kept:
                kept.append((role, text[-budget:]))
            break
        kept.append((role, text)); total += len(text)
    kept.reverse()
    while kept and kept[0][0] != "user":
        kept.pop(0)
    return [{"role": "system", "content": SYSTEM}] + [{"role": r, "content": t} for r, t in kept]


class ChatServer(Server):
    def chat_msgs(self, msgs, max_tokens):
        body_prompt = msgs  # reuse Server.chat's streaming/usage logic with a full message list
        orig = self.cfg.get("extra_body", {})
        self.cfg = dict(self.cfg, extra_body=dict(orig, messages=body_prompt))
        try:
            return self.chat("", max_tokens)
        finally:
            self.cfg = dict(self.cfg, extra_body=orig)


name, rnd = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"configs/{name}.json"))
srv = ChatServer(cfg)
picks = points()
res = {"name": name, "round": rnd, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "points": len(picks), "ctx": {}}
flags = []
srv.chat("Say hello.", 16)
for b, (label, tokens) in enumerate(BUCKETS):
    rates, ptoks = [], []
    for f, idx in picks[b * POINTS_PER_BUCKET:(b + 1) * POINTS_PER_BUCKET]:
        r = srv.chat_msgs(messages(f, idx, tokens), 500)
        if r["decode"]:
            rates.append(round(r["decode"], 2))
        ptoks.append(r["prompt_tokens"])
        if looping(r["text"]):
            flags.append(f"LOOP@{label}")
    res["ctx"][label] = {"prompt_tokens": ptoks, "decode_median": round(statistics.median(rates), 2) if rates else None,
                         "decode_all": rates}
    print(time.strftime("%H:%M:%S"), name, label, res["ctx"][label], flush=True)
res["flags"] = sorted(set(flags))
with open("replay.jsonl", "a") as fo:
    fo.write(json.dumps(res) + "\n")
