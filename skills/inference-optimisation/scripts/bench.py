#!/usr/bin/env python3
"""Engine-agnostic hill-climb benchmark for any OpenAI-compatible server (vLLM, SGLang, llama.cpp server,
TabbyAPI, MLX-LM server, LM Studio, Ollama's /v1 ...). Standard library only.

  python3 bench.py run configs/NAME.json [--long N] [--skip-cN] [--reps 6]
  python3 bench.py screen configs/NAME.json [--budget 90]    # fast onboarding / engine bake-off
  python3 bench.py quality configs/NAME.json --write-baseline quality-baseline.json
  python3 bench.py quality configs/NAME.json --baseline quality-baseline.json [--budget 300]
  python3 bench.py show [ledger.jsonl]
  python3 bench.py report [ledger.jsonl]                    # Markdown leaderboard

Config (JSON):
  {
    "name": "ctl-a",                              # unique run name (ledger key)
    "base_url": "http://127.0.0.1:8000",
    "model": "my-model",                           # served model name
    "launch": ["docker", "run", "-d", "..."],      # optional; omit to benchmark an already-running server
    "stop":   ["docker", "stop", "-t", "60", "srv"],  # optional; run after a drain wait
    "health_timeout_s": 900,
    "launch_timeout_s": 60,                       # optional; screen defaults to 30 seconds
    "stop_timeout_s": 60,                         # optional; screen defaults to 30 seconds
    "screen_health_timeout_s": 90,                # keeps a 2-3-engine bake-off near 15 minutes
    "concurrency": 4,                              # N for the cN test
    "prefill_lengths": [4096, 32768],              # approximate prompt tokens for cold prefill
    "extra_body": {"chat_template_kwargs": {"enable_thinking": false}},  # merged into every request
    "temperature": 0.6,
    "memory": {"cap_gib": 46, "probe": ["bash", "mem.sh", "srv"], "interval_s": 3},  # probe prints bytes
    "require_ac": true,                            # refuse to run on battery (Linux / macOS)
    "screen_require_tools": true,                  # screen fails if OpenAI tool_calls do not round-trip
    "screen_require_prefix_cache": true,           # screen fails if turn 2 does not reuse the turn-1 prefix
    "screen_prefix_max_s": 1.0
  }

Measurements (all token counts come from the server's usage stats, never from streamed chunks):
  startup_s, ttft_short_ms, c1_decode (median of --reps), c1_all, cN_aggregate (wall clock), cN_per_stream,
  prefill_<L> (prompt_tokens / TTFT on a unique cold prompt), long-context needle retrieval (--long),
  mem_peak_gib, flags (LOOP, REQ_FAIL, ON_BATTERY, MEM_CAP, MEM_PROBE_FAIL, SCREEN_*, QUALITY_*).
Results are appended to ledger.jsonl. The server is left running after a run; stop it yourself when done.
"""
import argparse, ast, http.client, json, os, random, re, signal, statistics, string, subprocess, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [  # fixed panel: acceptance and speed depend on content, so never randomise these
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.",
]
WORDS = ("river stone paper cloud engine garden silver market winter orange signal harbor violet "
         "candle forest mirror thunder copper meadow lantern").split()
SCREEN_PROSE = (PROMPTS[0], PROMPTS[1], PROMPTS[4])
CODE_BUG = """def clamp(value, low, high):
    if value < low:
        return high
    if value > high:
        return low
    return value"""
CODE_REQUEST = ("Fix only the bug in this Python function. Return the complete corrected function, "
                "with no explanation:\n\n" + CODE_BUG)
WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
QUALITY_PANEL = "inference-optimisation-quality-v1"
LOWER_IS_BETTER = {"ttft_short_ms", "startup_s", "mem_peak_gib"}
VOID_FLAGS = {"LOOP", "REQ_FAIL", "ON_BATTERY", "MEM_CAP", "MEM_PROBE_FAIL", "QUALITY_MISMATCH", "QUALITY_TIMEOUT"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _deadline_timeout(deadline_s, timeout):
    if deadline_s is None:
        return timeout
    remaining = deadline_s - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("operation exceeded its deadline")
    return min(timeout, remaining)


def _refresh_timeout(response, deadline_s, timeout):
    remaining = _deadline_timeout(deadline_s, timeout)
    sock = getattr(response, "sock", None)
    if sock is None:
        sock = getattr(getattr(response, "fp", None), "sock", None)
    if sock is None:
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        sock.settimeout(remaining)
    return remaining


def _read_body(response, deadline_s=None, timeout=5, limit=16 * 1024 * 1024):
    chunks, size = [], 0
    while True:
        _refresh_timeout(response, deadline_s, timeout)
        reader = getattr(response, "read1", response.read)
        chunk = reader(65536)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise ValueError("HTTP response exceeded the size limit")
        chunks.append(chunk)


def _sse_lines(response, deadline_s=None, timeout=7200):
    pending = b""
    while True:
        _refresh_timeout(response, deadline_s, timeout)
        reader = getattr(response, "read1", response.read)
        chunk = reader(65536)
        if not chunk:
            if pending:
                yield pending
            return
        pending += chunk
        if len(pending) > 1024 * 1024:
            raise ValueError("SSE line exceeded 1 MiB")
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line


class Server:
    def __init__(self, cfg):
        self.cfg, self.base = cfg, cfg.get("base_url", "http://127.0.0.1:8000").rstrip("/")
        self.url = urllib.parse.urlsplit(self.base)
        if self.url.scheme not in ("http", "https") or not self.url.hostname:
            raise ValueError(f"invalid base_url: {self.base}")

    def _connection(self, timeout):
        cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        return cls(self.url.hostname, self.url.port, timeout=timeout)

    def _target(self, path):
        return self.url.path.rstrip("/") + path

    def get(self, path, timeout=5, deadline_s=None):
        conn = self._connection(_deadline_timeout(deadline_s, timeout))
        try:
            conn.request("GET", self._target(path))
            response = conn.getresponse()
            return json.loads(_read_body(response, deadline_s, timeout) or b"{}")
        finally:
            conn.close()

    def healthy(self, deadline_s=None):
        try:
            return bool(self.get("/v1/models", deadline_s=deadline_s).get("data"))
        except Exception:
            return False

    def chat(self, prompt, max_tokens, temperature=None, timeout=7200, tools=None, messages=None, deadline_s=None):
        """Stream one chat completion. Returns ttft, decode tok/s, prompt/completion tokens (server usage)."""
        body = {"model": self.cfg["model"],
                "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                "temperature": self.cfg.get("temperature", 0.6) if temperature is None else temperature}
        body.update(self.cfg.get("extra_body", {}))
        if tools:
            body["tools"] = tools
        payload = json.dumps(body).encode()
        t0 = time.monotonic(); t_first = None; text = []; usage = {}; calls = {}
        done = False; finish_reason = None
        conn = self._connection(_deadline_timeout(deadline_s, timeout))
        try:
            conn.request("POST", self._target("/v1/chat/completions"), body=payload,
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            for raw in _sse_lines(response, deadline_s, timeout):
                if deadline_s is not None and time.monotonic() >= deadline_s:
                    raise TimeoutError("chat exceeded its deadline")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if piece:
                        if t_first is None:
                            t_first = time.monotonic()
                        text.append(piece)
                    for pos, tc in enumerate(d.get("tool_calls") or []):
                        i = tc.get("index")
                        if i is None:
                            i = pos
                        call = calls.setdefault(i, {"id": tc.get("id", ""), "type": "function",
                                                    "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            call["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name") and fn["name"] not in call["function"]["name"]:
                            call["function"]["name"] += fn["name"]
                        call["function"]["arguments"] += fn.get("arguments") or ""
        finally:
            conn.close()
        if not done and not finish_reason:
            raise ConnectionError("stream ended before [DONE] or a finish_reason")
        t_end = time.monotonic()
        ct, pt = usage.get("completion_tokens"), usage.get("prompt_tokens")
        if t_first is None:
            t_first = t_end
        decode = (ct - 1) / (t_end - t_first) if ct and ct > 1 and t_end > t_first else None
        details = usage.get("prompt_tokens_details") or {}
        return {"ttft": t_first - t0, "decode": decode, "completion_tokens": ct, "prompt_tokens": pt,
                "cached_tokens": details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0,
                "tool_calls": [calls[i] for i in sorted(calls)], "wall": t_end - t0,
                "t0": t0, "t_end": t_end, "text": "".join(text), "done": done,
                "finish_reason": finish_reason, "has_usage": bool(usage)}


def looping(text, n=8, thresh=0.5):
    """Degenerate-output flag: share of repeated word n-grams."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(max(0, len(w) - n))]
    return len(grams) > 50 and 1 - len(set(grams)) / len(grams) > thresh


def document(tokens, seed, needles=3):
    """Unique filler document of ~`tokens` tokens with `needles` retrieval codes at spread depths.
    Token count is approximate: the real count comes back as prompt_tokens and is what gets reported."""
    rng = random.Random(seed)
    n_words = int(tokens / 1.6)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(needles)]
    for i, c in enumerate(codes):
        pos = int(n_words * (i + 1) / (needles + 1))
        words.insert(pos, f". The secret code number {i + 1} is {c}. ")
    head = f"[doc {seed}-{time.time_ns()}] "  # defeats prefix caching
    q = "\n\nList every secret code number and its value that appears in the document above. Answer only with the codes."
    return head + " ".join(words) + q, codes


def on_ac():
    try:
        if sys.platform == "darwin":
            return "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        import glob
        vals = []
        for path in glob.glob("/sys/class/power_supply/*/online"):
            with open(path) as f:
                vals.append(f.read().strip())
        return not vals or "1" in vals
    except Exception:
        return True


def _kill_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _run_command(args, check=False, timeout=None, register=None, cancel_event=None):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    if register is not None:
        register(proc)
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("command cancelled")
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.communicate()
        raise
    finally:
        if proc.poll() is None:
            _kill_process_group(proc)
            proc.communicate()
        if register is not None:
            register(None)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, args, output=out, stderr=err)
    return subprocess.CompletedProcess(args, proc.returncode, out, err)


class MemWatch(threading.Thread):
    def __init__(self, mcfg, on_trip, deadline_s=None):
        super().__init__(daemon=True)
        self.mcfg, self.on_trip, self.peak, self.tripped, self.stop = mcfg, on_trip, 0, False, threading.Event()
        self.deadline_s, self.timed_out, self.samples, self.errors = deadline_s, False, 0, 0
        self.command_lock, self.command_proc = threading.Lock(), None

    def _set_command(self, proc):
        with self.command_lock:
            self.command_proc = proc

    def cancel(self):
        self.stop.set()
        with self.command_lock:
            proc = self.command_proc
        if proc is not None:
            _kill_process_group(proc)

    def run(self):
        cap = self.mcfg.get("cap_gib", 1e9) * 2**30
        while not self.stop.is_set():
            timeout = 30.0
            if self.deadline_s is not None:
                remaining = self.deadline_s - time.monotonic()
                if remaining <= 0:
                    self.timed_out = True
                    return
                timeout = min(timeout, remaining)
            try:
                out = _run_command(self.mcfg["probe"], check=True, timeout=timeout,
                                   register=self._set_command, cancel_event=self.stop).stdout
                b = int(out.strip() or 0)
            except subprocess.TimeoutExpired:
                self.timed_out = True
                log("memory probe exceeded its deadline")
                return
            except Exception as e:
                self.errors += 1
                if self.errors == 1:
                    log("memory probe failed:", repr(e))
                if self.stop.wait(self.mcfg.get("interval_s", 3)):
                    return
                continue
            self.samples += 1
            self.peak = max(self.peak, b)
            if b > cap and not self.tripped:
                self.tripped = True
                log(f"MEMORY CAP EXCEEDED: {b / 2**30:.1f} GiB > {cap / 2**30:.1f} GiB")
                if not self.stop.is_set():
                    self.on_trip(self)
            if self.stop.wait(self.mcfg.get("interval_s", 3)):
                return


def memory_watch(cfg, on_trip, deadline_s=None):
    memory = cfg.get("memory", {})
    if memory.get("cap_gib") is not None and not memory.get("probe"):
        sys.exit("memory.cap_gib requires memory.probe; refusing an unenforced memory cap")
    return MemWatch(memory, on_trip, deadline_s) if memory.get("probe") else None


def _bounded_timeout(cfg, key, default, deadline_s):
    timeout = cfg.get(key, default)
    if deadline_s is None:
        return timeout
    remaining = deadline_s - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("startup deadline exceeded")
    return remaining if timeout is None else min(timeout, remaining)


def stop_server(cfg, deadline_s=None, register=None, cancel_event=None):
    if cfg.get("stop"):
        try:
            timeout = _bounded_timeout(cfg, "stop_timeout_s", 30, deadline_s)
            _run_command(cfg["stop"], timeout=timeout, register=register, cancel_event=cancel_event)
        except subprocess.TimeoutExpired:
            log("stop command timed out")
            return False
        except TimeoutError:
            log("stop command had no time left in the budget")
            return False
        except OSError as e:
            log("stop command failed:", repr(e))
            return False
    return True


def prepare_server(cfg, srv, res, watch, deadline_s=None):
    if cfg.get("launch"):
        if not stop_server(cfg, deadline_s):
            res.update(ok=False, error="stop command failed or timed out")
            return False
        t0 = time.monotonic()
        try:
            timeout = _bounded_timeout(cfg, "launch_timeout_s", None, deadline_s)
            _run_command(cfg["launch"], check=True, timeout=timeout)
        except Exception as e:
            res.update(ok=False, error=f"launch failed: {e!r}")
            return False
        if watch:
            watch.start()
        deadline = t0 + cfg.get("health_timeout_s", 900)
        if deadline_s is not None:
            deadline = min(deadline, deadline_s)
        while not srv.healthy(deadline_s=deadline):
            if time.monotonic() > deadline or (watch and watch.tripped):
                res.update(ok=False, error="server did not become healthy")
                return False
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        res["startup_s"] = round(time.monotonic() - t0, 1)
        log("healthy after", res["startup_s"], "s")
    else:
        if not srv.healthy(deadline_s=deadline_s):
            res.update(ok=False, error="server not reachable and no launch command given")
            return False
        if watch:
            watch.start()
    return True


def finish(res, flags, watch, deadline_s=None, started_s=None):
    if watch:
        watch.stop.set()
        started = watch.ident is not None
        join_timeout = 65 if deadline_s is None else max(0, deadline_s - time.monotonic())
        if started:
            watch.join(timeout=join_timeout)
            if watch.is_alive():
                watch.cancel()
                watch.join(timeout=1)
        res["mem_peak_gib"] = round(watch.peak / 2**30, 2)
        if watch.tripped:
            flags.append("MEM_CAP"); res["ok"] = False
        if started and (watch.timed_out or watch.errors or not watch.samples):
            flags.append("MEM_PROBE_FAIL"); res["ok"] = False
    if not on_ac():
        flags.append("ON_BATTERY"); res["ok"] = False
    if deadline_s is not None and time.monotonic() > deadline_s:
        flags.append("QUALITY_TIMEOUT"); res["ok"] = False
    if started_s is not None and isinstance(res.get("quality"), dict):
        res["quality"]["elapsed_s"] = round(time.monotonic() - started_s, 1)
    res["flags"] = sorted(set(flags))
    if set(res["flags"]) & VOID_FLAGS:
        res["ok"] = False
    with open("ledger.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=1))
    return res


def run(cfg_path, long_ctx=None, skip_cn=False, reps=6):
    cfg = json.load(open(cfg_path))
    srv, res, flags = Server(cfg), {"name": cfg["name"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, []
    if cfg.get("require_ac") and not on_ac():
        sys.exit("refusing to run on battery (require_ac)")

    watch = memory_watch(cfg, lambda w: stop_server(cfg, register=w._set_command, cancel_event=w.stop))
    if not prepare_server(cfg, srv, res, watch):
        return finish(res, flags, watch)
    try:
        srv.chat("Say hello.", 16)  # warm-up (not measured)
        r = srv.chat("Reply with the single word: ready", 8)
        res["ttft_short_ms"] = round(r["ttft"] * 1000, 1)

        c1 = []
        for i in range(reps):
            r = srv.chat(PROMPTS[i % len(PROMPTS)], 512)
            if r["decode"]:
                c1.append(round(r["decode"], 2))
            if looping(r["text"]):
                flags.append("LOOP")
        res["c1_decode"] = round(statistics.median(c1), 2) if c1 else None
        res["c1_all"] = c1
        log("c1 median", res["c1_decode"], c1)

        n = int(cfg.get("concurrency", 4))
        if n > 1 and not skip_cn:
            with ThreadPoolExecutor(n) as ex:
                rs = list(ex.map(lambda i: srv.chat(PROMPTS[i % len(PROMPTS)], 512), range(n)))
            toks = sum((x["completion_tokens"] or 0) for x in rs)
            wall = max(x["t_end"] for x in rs) - min(x["t0"] for x in rs)
            res[f"c{n}_aggregate"] = round(toks / wall, 2)
            res[f"c{n}_per_stream"] = [round(x["decode"] or 0, 2) for x in rs]
            log(f"c{n} aggregate", res[f"c{n}_aggregate"])

        for L in cfg.get("prefill_lengths", [4096, 32768]):
            prompt, _ = document(L, seed=L)
            r = srv.chat(prompt, 1, temperature=0.0)
            res[f"prefill_{L}"] = round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None
            res[f"prefill_{L}_tokens"] = r["prompt_tokens"]
            log(f"prefill ~{L}", res[f"prefill_{L}"], "tok/s over", r["prompt_tokens"], "tokens")

        if long_ctx:
            prompt, codes = document(long_ctx, seed=7)
            r = srv.chat(prompt, 256, temperature=0.0)
            res["long"] = {"prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft"], 1),
                           "prefill": round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None,
                           "decode": r["decode"] and round(r["decode"], 2),
                           "retrieval": f"{sum(c in r['text'] for c in codes)}/{len(codes)}"}
            log("long", res["long"])
        res["ok"] = True
    except Exception as e:
        res.update(ok=False, error=repr(e)); flags.append("REQ_FAIL")
    return finish(res, flags, watch)


def _eval_expr(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_expr(node.operand, env)
    if isinstance(node, ast.BoolOp):
        values = [_eval_expr(v, env) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.IfExp):
        return _eval_expr(node.body if _eval_expr(node.test, env) else node.orelse, env)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
        if node.keywords or not node.args:
            raise ValueError("unsupported call")
        values = [_eval_expr(arg, env) for arg in node.args]
        return min(values) if node.func.id == "min" else max(values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        a, b = _eval_expr(node.left, env), _eval_expr(node.comparators[0], env)
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return a < b
        if isinstance(op, ast.LtE):
            return a <= b
        if isinstance(op, ast.Gt):
            return a > b
        if isinstance(op, ast.GtE):
            return a >= b
        if isinstance(op, ast.Eq):
            return a == b
    raise ValueError("unsupported expression")


def _eval_block(body, env):
    for stmt in body:
        if isinstance(stmt, ast.Return):
            return _eval_expr(stmt.value, env), True
        if isinstance(stmt, ast.If):
            branch = stmt.body if _eval_expr(stmt.test, env) else stmt.orelse
            value, returned = _eval_block(branch, env)
            if returned:
                return value, True
            continue
        if isinstance(stmt, (ast.Pass, ast.Expr)) and (isinstance(stmt, ast.Pass) or
                                                       isinstance(stmt.value, ast.Constant)):
            continue
        raise ValueError("unsupported statement")
    return None, False


def code_edit_ok(text):
    """Statically evaluate the tiny clamp fix; never execute model-generated code."""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    source = fenced.group(1) if fenced else text[text.find("def clamp"):]
    try:
        tree = ast.parse(source)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clamp"), None)
        if fn is None or [a.arg for a in fn.args.args] != ["value", "low", "high"]:
            return False
        for value, low, high, expected in ((-1, 0, 10, 0), (5, 0, 10, 5), (11, 0, 10, 10)):
            got, returned = _eval_block(fn.body, {"value": value, "low": low, "high": high})
            if not returned or got != expected:
                return False
        return True
    except (SyntaxError, ValueError, KeyError, TypeError):
        return False


def _reject_json_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def tool_call_ok(result):
    if not result.get("has_usage") or result.get("finish_reason") != "tool_calls":
        return False
    if len(result["tool_calls"]) != 1:
        return False
    fn = result["tool_calls"][0].get("function") or {}
    if fn.get("name") != "get_weather":
        return False
    try:
        args = json.loads(fn.get("arguments") or "{}", parse_constant=_reject_json_constant)
    except (TypeError, ValueError):
        return False
    return isinstance(args, dict) and str(args.get("city", "")).casefold() == "paris"


def screen(cfg_path, budget_s=90, reps=3, prefill=4096):
    """Fast, bounded smoke screen used before a full benchmark or a 2-3-engine bake-off."""
    cfg = json.load(open(cfg_path))
    cfg["health_timeout_s"] = min(float(cfg.get("health_timeout_s", 900)),
                                  float(cfg.get("screen_health_timeout_s", 90)))
    cfg.setdefault("launch_timeout_s", 30)
    cfg.setdefault("stop_timeout_s", 30)
    srv = Server(cfg)
    res = {"name": cfg["name"], "kind": "screen", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "screen": {"budget_s": budget_s, "reps": reps, "prefill": prefill}}
    flags = []
    if cfg.get("require_ac") and not on_ac():
        sys.exit("refusing to run on battery (require_ac)")
    watch = memory_watch(cfg, lambda w: stop_server(cfg, register=w._set_command, cancel_event=w.stop))
    if not prepare_server(cfg, srv, res, watch):
        return finish(res, flags, watch)
    started = time.monotonic()
    deadline = started + budget_s

    def stage(seconds):
        if time.monotonic() >= deadline:
            raise TimeoutError("screen budget exhausted")
        return min(deadline, time.monotonic() + seconds)

    try:
        srv.chat("Say hello.", 8, deadline_s=stage(10))
        values = []
        for i in range(reps):
            r = srv.chat(SCREEN_PROSE[i % len(SCREEN_PROSE)], 128, deadline_s=stage(20))
            if not r["text"].strip():
                flags.append("SCREEN_EMPTY")
            if looping(r["text"]):
                flags.append("SCREEN_LOOP")
            if r["decode"]:
                values.append(round(r["decode"], 2))
        res["screen"]["prose_decode"] = values
        res["c1_decode"] = round(statistics.median(values), 2) if values else None
        if not values:
            flags.append("SCREEN_NO_DECODE")

        prompt, _ = document(prefill, seed=4100)
        r = srv.chat(prompt, 1, temperature=0.0, deadline_s=stage(25))
        if not r["prompt_tokens"] or r["ttft"] <= 0:
            flags.append("SCREEN_GREEDY_FAIL")
        res["screen"]["greedy_prefill"] = {
            "prompt_tokens": r["prompt_tokens"], "ttft_ms": round(r["ttft"] * 1000, 1),
            "tok_s": round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] and r["ttft"] > 0 else None}
        res[f"prefill_{prefill}"] = res["screen"]["greedy_prefill"]["tok_s"]

        r = srv.chat(CODE_REQUEST, 256, temperature=0.0, deadline_s=stage(25))
        res["screen"]["code_edit_ok"] = code_edit_ok(r["text"])
        if not res["screen"]["code_edit_ok"]:
            flags.append("SCREEN_CODE_FAIL")

        if cfg.get("screen_require_tools", True):
            try:
                r = srv.chat("Use get_weather for Paris. Do not answer without calling the tool.", 64,
                             temperature=0.0, tools=[WEATHER_TOOL], deadline_s=stage(20))
                res["screen"]["tool_call_ok"] = tool_call_ok(r)
            except Exception as e:
                res["screen"]["tool_call_ok"] = False
                res["screen"]["tool_error"] = repr(e)
            if not res["screen"]["tool_call_ok"]:
                flags.append("SCREEN_TOOL_FAIL")

        if cfg.get("screen_require_prefix_cache", True):
            prompt, _ = document(4096, seed=4200)
            messages = [{"role": "user", "content": prompt + "\nReply with exactly READY."}]
            first = srv.chat(None, 8, temperature=0.0, messages=messages, deadline_s=stage(15))
            messages += [{"role": "assistant", "content": first["text"]},
                         {"role": "user", "content": "Reply with exactly SECOND."}]
            second = srv.chat(None, 8, temperature=0.0, messages=messages, deadline_s=stage(15))
            hit = bool(first["prompt_tokens"] and second["cached_tokens"] >= 0.8 * first["prompt_tokens"])
            fast = second["ttft"] <= float(cfg.get("screen_prefix_max_s", 1.0))
            res["screen"]["prefix_cache"] = {
                "turn1_prompt_tokens": first["prompt_tokens"], "turn1_ttft_ms": round(first["ttft"] * 1000, 1),
                "turn2_ttft_ms": round(second["ttft"] * 1000, 1), "turn2_cached_tokens": second["cached_tokens"],
                "hit": hit, "fast": fast}
            if not (hit and fast):
                flags.append("SCREEN_PREFIX_FAIL")

        res["screen"]["elapsed_s"] = round(time.monotonic() - started, 1)
        if time.monotonic() > deadline:
            flags.append("SCREEN_TIMEOUT")
        res["ok"] = not any(f.startswith("SCREEN_") for f in flags)
        log("screen", "passed" if res["ok"] else "failed", res["screen"]["elapsed_s"], "s", sorted(set(flags)))
    except Exception as e:
        res.update(ok=False, error=repr(e))
        flags.append("SCREEN_TIMEOUT" if isinstance(e, TimeoutError) else "REQ_FAIL")
        res["screen"]["elapsed_s"] = round(time.monotonic() - started, 1)
    return finish(res, flags, watch)


def load_quality(path):
    with open(path) as f:
        baseline = json.load(f)
    if baseline.get("panel") != QUALITY_PANEL:
        sys.exit(f"quality baseline {path} uses a different prompt panel")
    if baseline.get("version") != 1:
        sys.exit(f"quality baseline {path} uses an unsupported format version")
    return baseline


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def quality(cfg_path, baseline=None, write_baseline=None, prompts=3, max_tokens=256, budget_s=300):
    """Capture or compare deterministic greedy outputs; a mismatch voids the candidate."""
    if not 1 <= prompts <= len(PROMPTS):
        sys.exit(f"quality: --prompts must be between 1 and {len(PROMPTS)}")
    if bool(baseline) == bool(write_baseline):
        sys.exit("quality: choose exactly one of --baseline PATH or --write-baseline PATH")
    cfg = json.load(open(cfg_path))
    expected = load_quality(baseline) if baseline else None
    if expected and (expected.get("model") != cfg["model"] or expected.get("max_tokens") != max_tokens):
        sys.exit("quality baseline model/max_tokens does not match this run")
    if expected and expected.get("prompts") != PROMPTS[:prompts]:
        sys.exit("quality baseline prompts do not match this run")
    srv = Server(cfg)
    res = {"name": cfg["name"], "kind": "quality", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    flags = []
    if cfg.get("require_ac") and not on_ac():
        sys.exit("refusing to run on battery (require_ac)")
    started = time.monotonic()
    deadline = started + budget_s
    watch = memory_watch(cfg, lambda w: stop_server(cfg, deadline_s=deadline, register=w._set_command,
                                                    cancel_event=w.stop), deadline_s=deadline)
    if not prepare_server(cfg, srv, res, watch, deadline_s=deadline):
        res["quality"] = {"mode": "compare" if baseline else "capture", "elapsed_s": 0.0}
        return finish(res, flags, watch, deadline_s=deadline, started_s=started)
    try:
        panel = PROMPTS[:prompts]
        outputs = []
        for i, prompt in enumerate(panel):
            share = max(1.0, (deadline - time.monotonic()) / (len(panel) - i))
            outputs.append(srv.chat(prompt, max_tokens, temperature=0.0,
                                    deadline_s=min(deadline, time.monotonic() + share))["text"])
        if write_baseline:
            write_json(write_baseline, {"version": 1, "panel": QUALITY_PANEL, "model": cfg["model"],
                                        "max_tokens": max_tokens, "prompts": panel, "outputs": outputs})
            res["quality"] = {"mode": "capture", "path": write_baseline, "prompts": len(outputs)}
        else:
            if len(expected.get("outputs", [])) < len(outputs):
                sys.exit("quality baseline has fewer outputs than requested")
            identical = sum(a == b for a, b in zip(outputs, expected["outputs"]))
            normalized = sum(" ".join(a.split()) == " ".join(b.split())
                             for a, b in zip(outputs, expected["outputs"]))
            divergences = []
            for i, (a, b) in enumerate(zip(outputs, expected["outputs"])):
                if a == b:
                    continue
                k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                divergences.append({"prompt": i, "char": k, "candidate": a[max(0, k - 40):k + 40],
                                    "baseline": b[max(0, k - 40):k + 40]})
            res["quality"] = {"mode": "compare", "identical": identical, "normalized": normalized,
                              "prompts": len(outputs), "divergences": divergences}
            if identical != len(outputs):
                flags.append("QUALITY_MISMATCH")
        res["quality"]["elapsed_s"] = round(time.monotonic() - started, 1)
        res["ok"] = True
        log("quality", res["quality"])
    except Exception as e:
        res.update(ok=False, error=repr(e))
        flags.append("QUALITY_TIMEOUT" if isinstance(e, TimeoutError) else "REQ_FAIL")
        res["quality"] = {"mode": "compare" if baseline else "capture", "elapsed_s": round(time.monotonic() - started, 1)}
    return finish(res, flags, watch, deadline_s=deadline, started_s=started)


def load_ledger(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def metric_keys(rows):
    agg = sorted({k for d in rows for k in d if k.startswith("c") and k.endswith("_aggregate")})
    pre = sorted({k for d in rows for k in d if k.startswith("prefill_") and k[8:].isdigit()}, key=lambda k: int(k[8:]))
    return ["c1_decode", *agg, *pre, "ttft_short_ms", "startup_s", "mem_peak_gib"]


def show(path="ledger.jsonl"):
    rows = load_ledger(path)
    keys = metric_keys(rows)
    print("name".ljust(24) + "".join(k[:13].rjust(14) for k in keys) + "  ok  flags")
    for d in rows:
        print(d["name"][:24].ljust(24) + "".join(str(d.get(k, "-"))[:13].rjust(14) for k in keys) +
              f"  {'y' if d.get('ok') else 'N'}  " + ",".join(d.get("flags", [])))


def _md(value):
    text = str(value).replace("\n", " ")
    for old, new in (("\\", "\\\\"), ("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
                     ("|", "&#124;"), ("`", "\\`"), ("*", "\\*"), ("_", "\\_"),
                     ("[", "\\["), ("]", "\\]")):
        text = text.replace(old, new)
    return text


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def report(path="ledger.jsonl"):
    """Markdown leaderboard with separate full-run/screen boards and the latest quality gate per config."""
    if not os.path.exists(path):
        print(f"No ledger at {path}; run `bench.py run` or `bench.py screen` first.")
        return
    rows = load_ledger(path)
    if not rows:
        print("# Benchmark report\n\nNo runs recorded yet.")
        return
    latest_runs, latest_captures, latest_compares = {}, {}, {}
    for i, d in enumerate(rows):
        kind = d.get("kind", "run")
        if kind == "quality":
            target = latest_compares if (d.get("quality") or {}).get("mode", "compare") == "compare" else latest_captures
            target[d["name"]] = (i, d)
        elif kind in ("run", "screen"):
            latest_runs[(kind, d["name"])] = (i, d)

    boards, voided = {"run": [], "screen": []}, []
    for (kind, name), (i, d) in latest_runs.items():
        reasons = []
        if not d.get("ok"):
            reasons.append(d.get("error") or "run failed")
        flags = sorted(set(d.get("flags", [])) & VOID_FLAGS)
        if flags:
            reasons.extend(flags)
        quality_entry = latest_compares.get(name)
        if quality_entry and not quality_entry[1].get("ok"):
            reasons.extend(sorted(set(quality_entry[1].get("flags", [])) & VOID_FLAGS) or ["quality gate failed"])
        if reasons:
            voided.append((d, reasons))
        else:
            boards[kind].append(d)

    print("# Benchmark report\n")
    print(f"{len(rows)} ledger row(s); showing the latest benchmark for each config and protocol, "
          "plus the latest quality gate.\n")
    for kind, title in (("run", "Full runs"), ("screen", "Screens")):
        board = boards[kind]
        if not board:
            continue
        keys = metric_keys(board)
        best = {}
        for k in keys:
            vals = [d[k] for d in board if _number(d.get(k))]
            if vals:
                best[k] = min(vals) if k in LOWER_IS_BETTER else max(vals)
        print(f"## {title}\n")
        print("| Run | " + " | ".join(keys) + " | Flags |")
        print("|---|" + "---|" * len(keys) + "---|")
        for d in board:
            cells = []
            for k in keys:
                value = d.get(k)
                text = "-" if value is None else (f"{value:.1f}" if _number(value) else str(value))
                cells.append(f"**{text}**" if k in best and value == best[k] else text)
            print(f"| {_md(d['name'])} | " + " | ".join(cells) + " | " + _md(",".join(d.get("flags", [])) or "-") + " |")
        print("\n### Best valid measurements\n")
        for k in keys:
            if k not in best:
                continue
            names = [_md(d["name"]) for d in board if d.get(k) == best[k]]
            direction = "lowest" if k in LOWER_IS_BETTER else "highest"
            print(f"- **{k}:** {best[k]:.1f} ({direction}; {', '.join(names)})")
        print()

    quality_display = dict(latest_captures)
    quality_display.update(latest_compares)
    if quality_display:
        print("## Quality gates\n")
        print("| Config | Mode | Identical | Normalized | Status | First divergence |")
        print("|---|---|---:|---:|---|---|")
        for name, (_, q) in quality_display.items():
            info = q.get("quality") or {}
            divergences = info.get("divergences") or []
            if divergences:
                d = divergences[0]
                detail = (f"prompt {d.get('prompt', '?')}, char {d.get('char', '?')}; "
                         f"candidate: {d.get('candidate', '')}; baseline: {d.get('baseline', '')}")
            else:
                detail = "-"
            if info.get("mode") == "capture":
                status = "captured" if q.get("ok") else "VOID: " + (",".join(q.get("flags", [])) or q.get("error", "failed"))
            else:
                status = "pass" if q.get("ok") else "VOID: " + (",".join(q.get("flags", [])) or q.get("error", "failed"))
            print(f"| {_md(name)} | {_md(info.get('mode', '-'))} | {info.get('identical', '-')} | "
                  f"{info.get('normalized', '-')} | {_md(status)} | {_md(detail)} |")
        print()

    if voided:
        print("## Voided or failed latest benchmarks\n")
        for d, reasons in voided:
            print(f"- **{_md(d['name'])}** ({d.get('kind', 'run')}): {_md('; '.join(reasons))}")
        print()
    if not any(boards.values()):
        print("No valid benchmark rows yet. Check the quality and voided-run sections above.")


def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def main(argv):
    p = argparse.ArgumentParser(prog="bench.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--long", type=positive_int, dest="long_ctx")
    r.add_argument("--skip-cN", dest="skip_cn", action="store_true"); r.add_argument("--reps", type=positive_int, default=6)
    s = sub.add_parser("screen"); s.add_argument("config"); s.add_argument("--budget", type=positive_int, default=90)
    s.add_argument("--reps", type=positive_int, default=3); s.add_argument("--prefill", type=positive_int, default=4096)
    q = sub.add_parser("quality"); q.add_argument("config"); q.add_argument("--baseline")
    q.add_argument("--write-baseline", dest="write_baseline"); q.add_argument("--prompts", type=positive_int, default=3)
    q.add_argument("--max-tokens", type=positive_int, default=256)
    q.add_argument("--budget", type=positive_int, default=300, help="total budget including startup and cleanup, in seconds")
    sh = sub.add_parser("show"); sh.add_argument("ledger", nargs="?", default="ledger.jsonl")
    rep = sub.add_parser("report"); rep.add_argument("ledger", nargs="?", default="ledger.jsonl")
    a = p.parse_args(argv)

    if a.cmd == "run":
        return 0 if run(a.config, a.long_ctx, a.skip_cn, a.reps).get("ok") else 1
    if a.cmd == "screen":
        return 0 if screen(a.config, a.budget, a.reps, a.prefill).get("ok") else 1
    if a.cmd == "quality":
        return 0 if quality(a.config, a.baseline, a.write_baseline, a.prompts, a.max_tokens, a.budget).get("ok") else 1
    if a.cmd == "show":
        show(a.ledger)
    elif a.cmd == "report":
        report(a.ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
