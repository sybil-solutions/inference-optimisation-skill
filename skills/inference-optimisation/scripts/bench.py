#!/usr/bin/env python3
"""Engine-agnostic hill-climb benchmark for any OpenAI-compatible server (vLLM, SGLang, llama.cpp server,
TabbyAPI, MLX-LM server, LM Studio, Ollama's /v1 ...). Standard library only.

  python3 bench.py run     configs/NAME.json [--long N] [--skip-cN] [--reps 6] [--run-id ID] [--group G --pos I]
  python3 bench.py soak    configs/NAME.json --n 4 --tokens 262144      # N concurrent full-length needle requests
  python3 bench.py greedy  configs/NAME.json --out greedy/NAME.json     # capture greedy outputs of the panel
  python3 bench.py equiv   greedy/A.json greedy/B.json                  # compare two greedy captures
  python3 bench.py compare --control ctl [--group G | --runs ID ID ID] [--metric c1_decode] [--tolerance 0.03]
  python3 bench.py show    [ledger.jsonl]

Config (JSON):
  {
    "name": "ctl-a",                              # config name (the ledger also gets a unique run_id)
    "base_url": "http://127.0.0.1:8000",
    "model": "my-model",                           # served model name
    "launch": ["docker", "run", "-d", "..."],      # optional; omit to benchmark an already-running server
    "stop":   ["docker", "stop", "-t", "60", "srv"],  # optional; should abort + drain before stopping
    "health_timeout_s": 900,
    "concurrency": 4,                              # N for the cN test
    "prefill_lengths": [4096, 32768],              # approximate prompt tokens for cold prefill
    "extra_body": {"chat_template_kwargs": {"enable_thinking": false}},  # merged into every request
    "temperature": 0.6,
    "memory": {"cap_gib": 46, "probe": ["bash", "mem.sh", "srv"], "interval_s": 3},  # probe prints bytes
    "gpu_busy_probe": ["bash", "-c", "..."],       # optional; prints GPU busy %; flags GPU_IDLE if median < 60
    "require_ac": true,                            # refuse to run on battery (Linux / macOS)
    "require_profile": "performance",              # optional; refuse unless the power profile matches
    "max_load": 0.5                                # optional; refuse if 1-min loadavg / CPUs exceeds this
  }

Measurements (all token counts come from the server's usage stats, never from streamed chunks):
  startup_s, ttft_short_ms, c1_decode (median of --reps), c1_all, cN_aggregate (wall clock), cN_per_stream,
  cN_overlap (how much the N decode windows really overlapped), prefill_<L> (prompt_tokens / TTFT on a unique
  cold prompt), long-context needle retrieval (--long), mem_peak_gib, env (AC, power profile, load at start/end).
Flags void a run (ok=false): REQ_FAIL, MEM_CAP, ON_BATTERY, ENV_CHANGED, LOOP, NO_USAGE, NEEDLE_MISS.
Warning flags (run stays ok): NOT_CONCURRENT, GPU_IDLE, SHORT_OUTPUT.
Results are appended to ledger.jsonl. The server is left running after a run; stop it yourself when done.
"""
import argparse, glob, hashlib, json, os, random, statistics, string, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [  # fixed panel: acceptance and speed depend on content, so never randomise these
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.",
]
PANEL_ID = hashlib.sha1("\n".join(PROMPTS).encode()).hexdigest()[:8]
WORDS = ("river stone paper cloud engine garden silver market winter orange signal harbor violet "
         "candle forest mirror thunder copper meadow lantern").split()
VOIDING = {"REQ_FAIL", "MEM_CAP", "ON_BATTERY", "ENV_CHANGED", "LOOP", "NO_USAGE", "NEEDLE_MISS"}
LOWER_IS_BETTER = {"ttft_short_ms", "startup_s", "mem_peak_gib"}
LEDGER = os.environ.get("BENCH_LEDGER", "ledger.jsonl")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------- server client
class Server:
    def __init__(self, cfg):
        self.cfg, self.base = cfg, cfg.get("base_url", "http://127.0.0.1:8000").rstrip("/")

    def get(self, path, timeout=5):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def healthy(self):
        try:
            return bool(self.get("/v1/models").get("data"))
        except Exception:
            return False

    def chat(self, prompt, max_tokens, temperature=None, timeout=7200):
        """Stream one chat completion. Returns ttft, decode tok/s, prompt/completion tokens (server usage)."""
        body = {"model": self.cfg["model"], "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                "temperature": self.cfg.get("temperature", 0.6) if temperature is None else temperature}
        body.update(self.cfg.get("extra_body", {}))
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic(); t_first = None; text = []; usage = {}; finish = None
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if piece:
                        if t_first is None:
                            t_first = time.monotonic()
                        text.append(piece)
        t_end = time.monotonic()
        ct, pt = usage.get("completion_tokens"), usage.get("prompt_tokens")
        if t_first is None:
            t_first = t_end
        decode = (ct - 1) / (t_end - t_first) if ct and ct > 1 and t_end > t_first else None
        return {"ttft": t_first - t0, "decode": decode, "completion_tokens": ct, "prompt_tokens": pt,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "wall": t_end - t0, "t0": t0, "t_first": t_first, "t_end": t_end, "text": "".join(text),
                "finish_reason": finish, "has_usage": bool(usage)}


# ------------------------------------------------------------------------------------------------ helpers
def looping(text, n=8, thresh=0.5):
    """Degenerate-output flag: share of repeated word n-grams."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(max(0, len(w) - n))]
    return len(grams) > 50 and 1 - len(set(grams)) / len(grams) > thresh


def document(tokens, seed, needles=3, unique=True):
    """Filler document of ~`tokens` tokens with `needles` retrieval codes at spread depths.
    Token count is approximate: the real count comes back as prompt_tokens and is what gets reported."""
    rng = random.Random(seed)
    n_words = int(tokens / 1.6)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(needles)]
    for i, c in enumerate(codes):
        pos = int(n_words * (i + 1) / (needles + 1))
        words.insert(pos, f". The secret code number {i + 1} is {c}. ")
    head = f"[doc {seed}-{time.time_ns() if unique else 0}] "  # unique head defeats prefix caching
    q = "\n\nList every secret code number and its value that appears in the document above. Answer only with the codes."
    return head + " ".join(words) + q, codes


def overlap(rs):
    """Fraction of the mean decode window during which all N streams were decoding at once (1.0 = fully concurrent)."""
    start, end = max(r["t_first"] for r in rs), min(r["t_end"] for r in rs)
    mean_win = statistics.mean(r["t_end"] - r["t_first"] for r in rs) or 1e-9
    return round(max(0.0, end - start) / mean_win, 3)


def _cmd(args, timeout=5):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return None


def on_ac():
    try:
        if sys.platform == "darwin":
            return "AC Power" in (_cmd(["pmset", "-g", "batt"]) or "AC Power")
        vals = []
        for p in glob.glob("/sys/class/power_supply/*"):
            try:
                kind = open(p + "/type").read().strip()
                if kind in ("Mains", "USB") and os.path.exists(p + "/online"):
                    vals.append(open(p + "/online").read().strip())
            except OSError:
                pass
        return not vals or "1" in vals
    except Exception:
        return True


def power_profile():
    if sys.platform == "darwin":
        out = _cmd(["pmset", "-g"]) or ""
        return "lowpower" if any(l.split()[-1:] == ["1"] for l in out.splitlines() if "lowpowermode" in l) else "normal"
    prof = _cmd(["powerprofilesctl", "get"])
    if prof:
        return prof
    try:
        return open("/sys/firmware/acpi/platform_profile").read().strip()
    except OSError:
        pass
    try:
        return "governor:" + open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        return None


def env_state():
    try:
        load = round(os.getloadavg()[0] / (os.cpu_count() or 1), 2)
    except OSError:
        load = None
    return {"ac": on_ac(), "profile": power_profile(), "load_per_cpu": load}


def guard_env(cfg, env):
    """Refuse to start when the environment is not the fixed benchmarking environment."""
    if cfg.get("require_ac") and not env["ac"]:
        sys.exit("refusing to run on battery (require_ac)")
    want = cfg.get("require_profile")
    if want and env["profile"] != want:
        sys.exit(f"refusing to run: power profile is {env['profile']!r}, config requires {want!r}")
    if cfg.get("max_load") is not None and env["load_per_cpu"] is not None and env["load_per_cpu"] > cfg["max_load"]:
        sys.exit(f"refusing to run: host load {env['load_per_cpu']} per CPU > max_load {cfg['max_load']}")


def config_hash(cfg):
    keep = {k: v for k, v in cfg.items() if k not in ("name",)}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


class Sampler(threading.Thread):
    """Runs a probe command every interval_s; keeps all samples; optional cap callback (memory watchdog)."""

    def __init__(self, probe, interval_s=3, cap=None, on_trip=None, scale=1.0):
        super().__init__(daemon=True)
        self.probe, self.interval, self.cap, self.on_trip, self.scale = probe, interval_s, cap, on_trip, scale
        self.samples, self.tripped, self.stop, self.errors = [], False, threading.Event(), 0

    def sample_once(self):
        out = _cmd(self.probe, timeout=30)
        try:
            v = float(out.split()[-1]) * self.scale
        except Exception:
            self.errors += 1
            return None
        self.samples.append(v)
        if self.cap is not None and v > self.cap and not self.tripped:
            self.tripped = True
            log(f"CAP EXCEEDED: {v / 2**30:.2f} GiB > {self.cap / 2**30:.2f} GiB")
            if self.on_trip:
                self.on_trip()
        return v

    def run(self):
        self.sample_once()
        while not self.stop.wait(self.interval):
            self.sample_once()

    @property
    def peak(self):
        return max(self.samples) if self.samples else 0


def mem_sampler(cfg, on_trip):
    m = cfg.get("memory") or {}
    if not m.get("probe"):
        return None
    return Sampler(m["probe"], m.get("interval_s", 3), m.get("cap_gib", 1e9) * 2**30, on_trip)


def start_server(cfg, srv, res, watch):
    """Launch (if configured) and wait for health. Returns False on failure."""
    if cfg.get("launch"):
        stop_server(cfg)
        t0 = time.monotonic()
        subprocess.run(cfg["launch"], check=True, capture_output=True)
        if watch:
            watch.start()
        deadline = t0 + cfg.get("health_timeout_s", 900)
        while not srv.healthy():
            if time.monotonic() > deadline or (watch and watch.tripped):
                res.update(ok=False, error="server did not become healthy")
                return False
            time.sleep(1)
        res["startup_s"] = round(time.monotonic() - t0, 1)
        log("healthy after", res["startup_s"], "s")
    else:
        if not srv.healthy():
            sys.exit("server not reachable and no launch command given")
        if watch:
            watch.start()
    return True


def stop_server(cfg):
    if cfg.get("stop"):
        subprocess.run(cfg["stop"], capture_output=True)


def check_text(r, flags, where):
    if not r["has_usage"]:
        flags.add("NO_USAGE")
        log(f"{where}: server sent no usage stats; enable stream_options.include_usage support")
    if looping(r["text"]):
        flags.add("LOOP")
        log(f"{where}: looping output detected")


# ---------------------------------------------------------------------------------------------------- run
def run(cfg_path, long_ctx=None, skip_cn=False, reps=6, run_id=None, group=None, pos=None):
    cfg = json.load(open(cfg_path))
    env0 = env_state()
    guard_env(cfg, env0)
    srv, flags = Server(cfg), set()
    res = {"name": cfg["name"], "run_id": run_id or f"{cfg['name']}@{time.strftime('%Y%m%dT%H%M%S')}",
           "group": group, "pos": pos, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": config_hash(cfg),
           "panel": PANEL_ID, "temperature": cfg.get("temperature", 0.6), "extra_body": cfg.get("extra_body", {}),
           "env_start": env0}
    watch = mem_sampler(cfg, lambda: stop_server(cfg))
    if not start_server(cfg, srv, res, watch):
        return finish(res, flags, watch, env0)
    try:
        srv.chat("Say hello.", 16)  # warm-up (not measured)
        r = srv.chat("Reply with the single word: ready", 8)
        res["ttft_short_ms"] = round(r["ttft"] * 1000, 1)

        busy = Sampler(cfg["gpu_busy_probe"], 1) if cfg.get("gpu_busy_probe") else None
        if busy:
            busy.start()
        c1 = []
        for i in range(reps):
            r = srv.chat(PROMPTS[i % len(PROMPTS)], 512)
            check_text(r, flags, f"c1[{i}]")
            if r["completion_tokens"] is not None and r["completion_tokens"] < 64:
                flags.add("SHORT_OUTPUT")  # decode rate over very few tokens is mostly noise
            if r["decode"]:
                c1.append(round(r["decode"], 2))
        if busy:
            busy.stop.set()
            if busy.samples:
                res["gpu_busy_median"] = round(statistics.median(busy.samples), 1)
                if res["gpu_busy_median"] < 60:
                    flags.add("GPU_IDLE")
        res["c1_decode"] = round(statistics.median(c1), 2) if c1 else None
        res["c1_all"] = c1
        log("c1 median", res["c1_decode"], c1)

        n = int(cfg.get("concurrency", 4))
        if n > 1 and not skip_cn:
            with ThreadPoolExecutor(n) as ex:
                rs = list(ex.map(lambda i: srv.chat(PROMPTS[i % len(PROMPTS)], 512), range(n)))
            for i, x in enumerate(rs):
                check_text(x, flags, f"c{n}[{i}]")
            toks = sum((x["completion_tokens"] or 0) for x in rs)
            wall = max(x["t_end"] for x in rs) - min(x["t0"] for x in rs)
            res[f"c{n}_aggregate"] = round(toks / wall, 2)
            res[f"c{n}_per_stream"] = [round(x["decode"] or 0, 2) for x in rs]
            res[f"c{n}_overlap"] = overlap(rs)
            if res[f"c{n}_overlap"] < 0.5:
                flags.add("NOT_CONCURRENT")  # the server likely serialised the streams (max_running / state slots)
            log(f"c{n} aggregate", res[f"c{n}_aggregate"], "overlap", res[f"c{n}_overlap"])

        for L in cfg.get("prefill_lengths", [4096, 32768]):
            prompt, _ = document(L, seed=L)
            r = srv.chat(prompt, 1, temperature=0.0)
            if not r["has_usage"]:
                flags.add("NO_USAGE")
            res[f"prefill_{L}"] = round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None
            res[f"prefill_{L}_tokens"] = r["prompt_tokens"]
            if r["cached_tokens"]:
                res[f"prefill_{L}_cached"] = r["cached_tokens"]  # should be ~0: the prompt is unique
            log(f"prefill ~{L}", res[f"prefill_{L}"], "tok/s over", r["prompt_tokens"], "tokens")

        if long_ctx:
            res["long"] = needle(srv, long_ctx, seed=7)
            if not res["long"]["all_found"]:
                flags.add("NEEDLE_MISS")
            log("long", res["long"])
        res["ok"] = True
    except Exception as e:
        res.update(ok=False, error=repr(e)); flags.add("REQ_FAIL")
    return finish(res, flags, watch, env0)


def needle(srv, tokens, seed):
    prompt, codes = document(tokens, seed=seed)
    r = srv.chat(prompt, 256, temperature=0.0)
    found = sum(c in r["text"] for c in codes)
    return {"prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft"], 1),
            "prefill": round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None,
            "decode": r["decode"] and round(r["decode"], 2), "retrieval": f"{found}/{len(codes)}",
            "all_found": found == len(codes)}


def finish(res, flags, watch, env0):
    if watch:
        watch.stop.set()
        res["mem_peak_gib"] = round(watch.peak / 2**30, 2)
        if watch.tripped:
            flags.add("MEM_CAP")
        if watch.errors and not watch.samples:
            log("memory probe never returned a number; check the probe command")
    env1 = env_state()
    res["env_end"] = env1
    if not env1["ac"]:
        flags.add("ON_BATTERY")
    if env1["ac"] != env0["ac"] or env1["profile"] != env0["profile"]:
        flags.add("ENV_CHANGED")
    res["flags"] = sorted(flags)
    if flags & VOIDING:
        res["ok"] = False
    with open(LEDGER, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------------------------------- soak
def soak(cfg_path, n, tokens):
    """N concurrent unique full-length needle requests: proves the KV pool really holds N x context."""
    cfg = json.load(open(cfg_path))
    env0 = env_state()
    guard_env(cfg, env0)
    srv, flags = Server(cfg), set()
    res = {"name": cfg["name"], "run_id": f"{cfg['name']}@soak-{time.strftime('%Y%m%dT%H%M%S')}", "kind": "soak",
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_hash": config_hash(cfg), "soak_n": n,
           "soak_tokens": tokens, "env_start": env0}
    watch = mem_sampler(cfg, lambda: stop_server(cfg))
    if not start_server(cfg, srv, res, watch):
        return finish(res, flags, watch, env0)
    try:
        t0 = time.monotonic()
        with ThreadPoolExecutor(n) as ex:
            rs = list(ex.map(lambda i: needle(srv, tokens, seed=1000 + i), range(n)))
        res["soak"] = rs
        res["soak_wall_s"] = round(time.monotonic() - t0, 1)
        res["soak_found"] = f"{sum(r['all_found'] for r in rs)}/{n}"
        if not all(r["all_found"] for r in rs):
            flags.add("NEEDLE_MISS")
        log("soak", res["soak_found"], "streams with all needles in", res["soak_wall_s"], "s")
        res["ok"] = True
    except Exception as e:
        res.update(ok=False, error=repr(e)); flags.add("REQ_FAIL")
    return finish(res, flags, watch, env0)


# ----------------------------------------------------------------------------------------- greedy / equiv
def greedy(cfg_path, out, max_tokens=256):
    cfg = json.load(open(cfg_path))
    srv = Server(cfg)
    if not srv.healthy():
        sys.exit("server not reachable")
    outs = [srv.chat(p, max_tokens, temperature=0.0)["text"] for p in PROMPTS]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"name": cfg["name"], "config_hash": config_hash(cfg), "panel": PANEL_ID, "max_tokens": max_tokens,
               "outputs": outs}, open(out, "w"), indent=1)
    log("greedy outputs written to", out)


def equiv(a_path, b_path):
    """Exact greedy equivalence per prompt; on divergence, report where, so you can inspect logits there."""
    a, b = json.load(open(a_path)), json.load(open(b_path))
    if a.get("panel") != b.get("panel"):
        sys.exit("captures use different prompt panels")
    same = 0
    for i, (x, y) in enumerate(zip(a["outputs"], b["outputs"])):
        if x == y:
            same += 1
            print(f"prompt {i}: identical ({len(x)} chars)")
            continue
        k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
        print(f"prompt {i}: DIVERGES at char {k} (word {len(x[:k].split())}):\n"
              f"   A: ...{x[max(0, k - 40):k + 40]!r}\n   B: ...{y[max(0, k - 40):k + 40]!r}")
    print(f"\n{same}/{len(a['outputs'])} identical. A late single divergence is often a numerics tie between "
          "kernels (compare logits there); an early or repeated one is a bug.")
    return same == len(a["outputs"])


# -------------------------------------------------------------------------------------------------- compare
def bootstrap_ratio_ci(cand, ctl, iters=4000, seed=0):
    """95% bootstrap CI of median(cand) / median(ctl)."""
    rng = random.Random(seed)
    ratios = sorted(statistics.median(rng.choices(cand, k=len(cand))) /
                    statistics.median(rng.choices(ctl, k=len(ctl))) for _ in range(iters))
    return ratios[int(0.025 * iters)], ratios[int(0.975 * iters) - 1]


def metric_keys(rows):
    keys = {"c1_decode", "ttft_short_ms", "startup_s", "mem_peak_gib"}
    for d in rows:
        keys |= {k for k in d if (k.startswith("c") and k.endswith("_aggregate")) or (k.startswith("prefill_") and k[8:].isdigit())}
    return sorted(keys)


def better(metric, a, b):
    """Relative improvement of a over b for this metric (positive = better)."""
    if a is None or b is None or b == 0:
        return None
    rel = (a - b) / b
    return -rel if metric in LOWER_IS_BETTER else rel


def compare(rows, control, metric="c1_decode", tolerance=0.03):
    """Judge every candidate against its neighbouring controls in run order.
    ACCEPT  : beats BOTH neighbours on `metric`, no other metric regresses by more than `tolerance`, and (when
              per-sample data exist) the bootstrap 95% CI of the median ratio vs pooled controls is above 1.
    REPEAT  : beats both neighbours, but the CI includes 1 -> run it again before believing it.
    REJECT  : otherwise. Voided runs (ok=false) are never judged."""
    verdicts = []
    seq = [d for d in rows if d.get("kind") != "soak"]
    for i, d in enumerate(seq):
        if d["name"] == control:
            continue
        prev = next((x for x in reversed(seq[:i]) if x["name"] == control), None)
        nxt = next((x for x in seq[i + 1:] if x["name"] == control), None)
        v = {"run_id": d.get("run_id", d["name"]), "name": d["name"], "metric": metric}
        if not prev or not nxt:
            v.update(verdict="NO_CONTROLS", reason="needs a control run on both sides (ctl cand ctl)")
        elif not all(x.get("ok") for x in (d, prev, nxt)):
            bad = [x.get("run_id", x["name"]) for x in (prev, d, nxt) if not x.get("ok")]
            v.update(verdict="VOID", reason=f"voided run(s): {', '.join(bad)}")
        else:
            gains = [better(metric, d.get(metric), c.get(metric)) for c in (prev, nxt)]
            v["vs_controls"] = [None if g is None else round(g * 100, 1) for g in gains]
            regress = []
            for k in metric_keys([d, prev, nxt]):
                if k == metric:
                    continue
                g = [better(k, d.get(k), c.get(k)) for c in (prev, nxt)]
                g = [x for x in g if x is not None]
                if g and statistics.mean(g) < -tolerance:
                    regress.append(f"{k} {statistics.mean(g) * 100:+.1f}%")
            ci = None
            if metric == "c1_decode" and d.get("c1_all") and prev.get("c1_all") and nxt.get("c1_all"):
                if len(d["c1_all"]) >= 3:
                    ci = bootstrap_ratio_ci(d["c1_all"], prev["c1_all"] + nxt["c1_all"])
                    v["ci95_ratio"] = [round(ci[0], 3), round(ci[1], 3)]
            beats = all(g is not None and g > 0 for g in gains)
            if not beats:
                v.update(verdict="REJECT", reason="does not beat both neighbouring controls")
            elif regress:
                v.update(verdict="REJECT", reason="regresses " + ", ".join(regress))
            elif ci and ci[0] <= 1.0:
                v.update(verdict="REPEAT", reason="beats both controls but the 95% CI includes no change")
            else:
                v.update(verdict="ACCEPT", reason="beats both controls" + (" with CI > 1" if ci else ""))
        verdicts.append(v)
    return verdicts


def load_ledger(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def select_rows(rows, group=None, run_ids=None):
    if run_ids:
        by = {d.get("run_id"): d for d in rows}
        missing = [r for r in run_ids if r not in by]
        if missing:
            sys.exit(f"run ids not in ledger: {', '.join(missing)}")
        return [by[r] for r in run_ids]
    groups = [d.get("group") for d in rows if d.get("group")]
    if group is None:
        if not groups:
            return rows  # no groups recorded: compare in file order
        group = groups[-1]
    sel = [d for d in rows if d.get("group") == group]
    return sorted(sel, key=lambda d: d.get("pos") or 0)


# ------------------------------------------------------------------------------------------------------ show
def show(path=LEDGER):
    rows = load_ledger(path)
    agg = sorted({k for d in rows for k in d if k.startswith("c") and k.endswith("_aggregate")})
    pre = sorted({k for d in rows for k in d if k.startswith("prefill_") and k[8:].isdigit()}, key=lambda k: int(k[8:]))
    keys = ["c1_decode", *agg, *pre, "ttft_short_ms", "startup_s", "mem_peak_gib"]
    print("run".ljust(30) + "".join(k[:13].rjust(14) for k in keys) + "  ok  flags")
    for d in rows:
        rid = d.get("run_id", d["name"])
        print(rid[:30].ljust(30) + "".join(str(d.get(k, "-"))[:13].rjust(14) for k in keys) +
              f"  {'y' if d.get('ok') else 'N'}  " + ",".join(d.get("flags", [])))


def main(argv):
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--long", type=int)
    r.add_argument("--skip-cN", dest="skip_cn", action="store_true"); r.add_argument("--reps", type=int, default=6)
    r.add_argument("--run-id"); r.add_argument("--group"); r.add_argument("--pos", type=int)
    s = sub.add_parser("soak"); s.add_argument("config"); s.add_argument("--n", type=int, required=True)
    s.add_argument("--tokens", type=int, required=True)
    g = sub.add_parser("greedy"); g.add_argument("config"); g.add_argument("--out", required=True)
    g.add_argument("--max-tokens", type=int, default=256)
    e = sub.add_parser("equiv"); e.add_argument("a"); e.add_argument("b")
    c = sub.add_parser("compare"); c.add_argument("--control", required=True); c.add_argument("--group")
    c.add_argument("--runs", nargs="+"); c.add_argument("--metric", default="c1_decode")
    c.add_argument("--tolerance", type=float, default=0.03); c.add_argument("--ledger", default=LEDGER)
    c.add_argument("--json", action="store_true")
    sh = sub.add_parser("show"); sh.add_argument("ledger", nargs="?", default=LEDGER)
    a = p.parse_args(argv)

    if a.cmd == "run":
        res = run(a.config, a.long, a.skip_cn, a.reps, a.run_id, a.group, a.pos)
        return 0 if res.get("ok") else 1
    if a.cmd == "soak":
        return 0 if soak(a.config, a.n, a.tokens).get("ok") else 1
    if a.cmd == "greedy":
        greedy(a.config, a.out, a.max_tokens); return 0
    if a.cmd == "equiv":
        return 0 if equiv(a.a, a.b) else 1
    if a.cmd == "compare":
        vs = compare(select_rows(load_ledger(a.ledger), a.group, a.runs), a.control, a.metric, a.tolerance)
        if a.json:
            print(json.dumps(vs, indent=1))
        else:
            for v in vs:
                extra = f" vs ctl {v['vs_controls']}%" if v.get("vs_controls") else ""
                extra += f" CI95 {v['ci95_ratio']}" if v.get("ci95_ratio") else ""
                print(f"{v['verdict']:<12}{v['run_id']:<34}{v['metric']}{extra}  — {v['reason']}")
        return 0
    if a.cmd == "show":
        show(a.ledger); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
