"""Deterministic local OpenAI-compatible streaming server used by the harness tests."""
import argparse, hashlib, json, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NEEDLE = re.compile(r"secret code number (\d+) is ([A-Z0-9]{8})")


def make_handler(opts, state):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                if opts.drip_health:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    try:
                        self.wfile.write(b"{"); self.wfile.flush()
                        time.sleep(0.3)
                        self.wfile.write(b"}"); self.wfile.flush()
                        time.sleep(2)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.json({"data": [{"id": "fake"}]})
            else:
                self.json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.generate(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def event(self, obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        def generate(self, body):
            messages = body.get("messages", [])
            prompt = "\n".join(str(m.get("content", "")) for m in messages)
            pt = max(1, len(prompt.split()) + 8)
            first = messages[0].get("content", "") if messages else prompt
            with lock:
                cached = state["prefixes"].get(first, 0) if opts.prefix_cache and len(messages) > 1 else 0
                if len(messages) == 1:
                    state["prefixes"][first] = pt
            if opts.hang_greedy and body.get("temperature") == 0 and body.get("max_tokens") == 1:
                time.sleep(opts.hang_seconds)
            if opts.hang_all:
                time.sleep(opts.hang_seconds)
            if opts.drip_greedy and body.get("temperature") == 0 and body.get("max_tokens") == 1:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self.wfile.write(b"data: "); self.wfile.flush()
                    time.sleep(0.3)
                    self.wfile.write(b'{"choices":[]}\n\n'); self.wfile.flush()
                    time.sleep(2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            max_tokens = body.get("max_tokens", 16)
            usage_completion = max_tokens
            pieces = None
            tool_call = None
            if body.get("tools"):
                if opts.tool_fail:
                    toks, tool_call = ["no", "tool"], None
                else:
                    toks, tool_call = [], {"index": 0, "id": "call_1", "type": "function", "function": {
                        "name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}
                    if opts.fragment_tool:
                        tool_call = None
                    usage_completion = 12
            elif "Fix only the bug" in prompt:
                if opts.code_fail:
                    pieces = ["def clamp(value, low, high):\n    return high\n"]
                else:
                    pieces = ["def clamp(value, low, high):\n", "    if value < low:\n", "        return low\n",
                              "    if value > high:\n", "        return high\n", "    return value\n"]
                toks = pieces
            elif "Reply with exactly READY" in prompt:
                toks = ["READY"]
            elif "Reply with exactly SECOND" in prompt:
                toks = ["SECOND"]
            elif NEEDLE.search(prompt):
                toks = [piece for _, code in NEEDLE.findall(prompt) for piece in (code,)]
            elif opts.quality_bad:
                toks = ["MUTATED"] * max_tokens
            else:
                seed = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16)
                toks = [f"word{(seed + i * 7919) % 1000003}" for i in range(max_tokens)]
            toks = toks[:max_tokens]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                time.sleep(pt / opts.prefill_tps)
                for i, token in enumerate(toks):
                    if i:
                        time.sleep(1 / opts.tps)
                    piece = pieces[i] if pieces is not None else token + " "
                    self.event({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                if tool_call:
                    self.event({"choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}]})
                if opts.fragment_tool and body.get("tools") and not opts.tool_fail:
                    for part in ({"index": 0, "id": "call_1", "type": "function",
                                  "function": {"name": "get_", "arguments": '{"city":'}},
                                 {"index": 0, "function": {"name": "weather", "arguments": '"Paris"}'}}):
                        self.event({"choices": [{"index": 0, "delta": {"tool_calls": [part]}, "finish_reason": None}]})
                truncating = opts.truncate_tool and bool(body.get("tools"))
                if not truncating:
                    reason = "tool_calls" if (tool_call or (opts.fragment_tool and body.get("tools") and not opts.tool_fail)) else "stop"
                    self.event({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
                if not opts.no_usage:
                    self.event({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": usage_completion,
                                                          "total_tokens": pt + usage_completion,
                                                          "prompt_tokens_details": {"cached_tokens": cached}}})
                if truncating:
                    return
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--tps", type=float, default=4000)
    p.add_argument("--prefill-tps", type=float, default=100000)
    p.add_argument("--tool-fail", action="store_true")
    p.add_argument("--fragment-tool", action="store_true")
    p.add_argument("--truncate-tool", action="store_true")
    p.add_argument("--code-fail", action="store_true")
    p.add_argument("--no-prefix", action="store_true")
    p.add_argument("--no-usage", action="store_true")
    p.add_argument("--quality-bad", action="store_true")
    p.add_argument("--hang-greedy", action="store_true")
    p.add_argument("--drip-greedy", action="store_true")
    p.add_argument("--drip-health", action="store_true")
    p.add_argument("--hang-all", action="store_true")
    p.add_argument("--hang-seconds", type=float, default=1.5)
    opts = p.parse_args(argv)
    opts.prefix_cache = not opts.no_prefix
    return opts


def serve(opts):
    state = {"prefixes": {}}
    server = ThreadingHTTPServer(("127.0.0.1", opts.port), make_handler(opts, state))
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    server = serve(parse([]))
    print("PORT", server.server_address[1], flush=True)
    server.serve_forever()
