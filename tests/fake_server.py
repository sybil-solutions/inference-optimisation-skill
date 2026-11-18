"""A tiny fake OpenAI-compatible streaming server for end-to-end tests of bench.py (standard library only).

  python3 fake_server.py --port 0 --tps 400 --prefill-tps 50000 [--loop] [--no-usage] [--serial]
Prints "PORT <n>" on stdout once listening.

Behaviour: streams one token per SSE chunk at --tps, waits prompt_tokens / --prefill-tps before the first token,
echoes any "secret code number i is XXXX" needles it sees (so retrieval passes), and is deterministic at
temperature 0 so greedy-equivalence tests are meaningful.
"""
import argparse, hashlib, json, random, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOCAB = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon phi chi psi omega").split()
NEEDLE = re.compile(r"secret code number (\d+) is ([A-Z0-9]{8})")


def make_handler(opts):
    serial = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                return self._json({"data": [{"id": "fake"}]})
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json({"error": "not found"}, 404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][-1]["content"]
            if opts.serial:
                with serial:
                    self.generate(body, prompt)
            else:
                self.generate(body, prompt)

        def generate(self, body, prompt):
            pt = max(1, int(len(prompt.split()) * 1.3))
            max_tokens = body.get("max_tokens", 16)
            seed = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed if body.get("temperature", 1) == 0 else None)
            needles = NEEDLE.findall(prompt)
            if needles:
                toks = " ".join(f"{i}: {c}" for i, c in needles).split()
            elif opts.loop:
                toks = (["the", "same", "phrase", "again", "and", "again", "forever", "now"] * 200)
            else:
                toks = [rng.choice(VOCAB) for _ in range(max_tokens)]
            toks = toks[:max_tokens]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(pt / opts.prefill_tps)
            for i, t in enumerate(toks):
                if i:
                    time.sleep(1 / opts.tps)
                ev = {"choices": [{"index": 0, "delta": {"content": t + " "}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
            fin = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if len(toks) == max_tokens else "stop"}]}
            self.wfile.write(f"data: {json.dumps(fin)}\n\n".encode())
            if not opts.no_usage and (body.get("stream_options") or {}).get("include_usage"):
                u = {"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": len(toks), "total_tokens": pt + len(toks)}}
                self.wfile.write(f"data: {json.dumps(u)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

    return H


def serve(opts):
    srv = ThreadingHTTPServer(("127.0.0.1", opts.port), make_handler(opts))
    srv.daemon_threads = True
    return srv


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--tps", type=float, default=400)
    p.add_argument("--prefill-tps", type=float, default=50000)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--no-usage", action="store_true")
    p.add_argument("--serial", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    s = serve(parse(sys.argv[1:]))
    print("PORT", s.server_address[1], flush=True)
    s.serve_forever()
