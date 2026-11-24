"""Regression tests for the inference-optimisation harness (standard library only)."""
import contextlib, io, json, os, subprocess, sys, tempfile, threading, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "skills", "inference-optimisation", "scripts")
EXAMPLES = os.path.join(ROOT, "skills", "inference-optimisation", "examples")
sys.path[:0] = [SCRIPTS, os.path.dirname(os.path.abspath(__file__))]

import bench, ceiling, fake_server  # noqa: E402


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def ornith():
    return read_json(os.path.join(EXAMPLES, "ceiling-ornith-b390.json"))


class CeilingTests(unittest.TestCase):
    def test_issue_1_dense_concurrency_repro(self):
        model = {"bandwidth_gbs": 100, "dense_gb": 10, "lm_head_gb": 0}
        self.assertEqual([ceiling.decode_ceiling(model, 0, 1.0, 0, n) for n in (1, 2, 4)], [10.0, 20.0, 40.0])

    def test_b390_c1_is_unchanged_and_c4_is_physical(self):
        model = ornith()
        c1 = ceiling.decode_ceiling(model, 0, 1.0, 4096, 1)
        c4 = ceiling.decode_ceiling(model, 0, 1.0, 4096, 4)
        self.assertAlmostEqual(c1, 68, delta=1.5)
        self.assertAlmostEqual(c4, 142, delta=2)
        self.assertLess(c4, 4 * c1)

    def test_draft_weights_are_shared_by_the_batch(self):
        model = {"bandwidth_gbs": 100, "dense_gb": 1, "lm_head_gb": 0,
                 "draft_layer_gb": 1, "draft_head_gb": 0}
        # One 2 GB step emits accept * n = 8 tokens.
        self.assertAlmostEqual(ceiling.decode_ceiling(model, 1, 2.0, 0, 4), 400.0)


class BenchUnitTests(unittest.TestCase):
    def test_code_gate_accepts_fix_and_rejects_bad_or_unsafe_code(self):
        good = "```python\ndef clamp(value, low, high):\n    if value < low:\n        return low\n    if value > high:\n        return high\n    return value\n```"
        bad = "def clamp(value, low, high):\n    if value < low:\n        return high\n    return value\n"
        unsafe = "def clamp(value, low, high):\n    return __import__('os').system('true')\n"
        self.assertTrue(bench.code_edit_ok(good))
        self.assertTrue(bench.code_edit_ok("def clamp(value, low, high):\n    return max(low, min(value, high))\n"))
        self.assertFalse(bench.code_edit_ok(bad))
        self.assertFalse(bench.code_edit_ok(unsafe))
        self.assertFalse(bench.code_edit_ok("not python"))

    def test_tool_gate_requires_named_call_with_json_arguments(self):
        valid = {"tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}],
                 "done": True, "finish_reason": "tool_calls", "has_usage": True}
        truncated = {"tool_calls": valid["tool_calls"], "done": False, "finish_reason": None, "has_usage": True}
        wrong_name = {"tool_calls": [{"function": {"name": "other", "arguments": '{"city":"Paris"}'}}]}
        bad_json = {"tool_calls": [{"function": {"name": "get_weather", "arguments": "Paris"}}]}
        nan = {"tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Paris","n":NaN}'}}]}
        infinity = {"tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Paris","n":Infinity}'}}]}
        self.assertTrue(bench.tool_call_ok(valid))
        self.assertFalse(bench.tool_call_ok(wrong_name))
        self.assertFalse(bench.tool_call_ok(bad_json))
        self.assertFalse(bench.tool_call_ok(nan))
        self.assertFalse(bench.tool_call_ok(infinity))
        self.assertFalse(bench.tool_call_ok(truncated))
        self.assertFalse(bench.tool_call_ok({"tool_calls": []}))

    def test_report_uses_latest_valid_run_and_marks_best_metrics(self):
        rows = [
            {"name": "a", "ok": True, "c1_decode": 100, "ttft_short_ms": 50},
            {"name": "a", "ok": True, "c1_decode": 110, "ttft_short_ms": 40},
            {"name": "b", "ok": True, "c1_decode": 105, "ttft_short_ms": 60},
            {"name": "broken", "ok": False, "c1_decode": 999, "ttft_short_ms": 1},
            {"name": "loop", "ok": True, "flags": ["LOOP"], "c1_decode": 999, "ttft_short_ms": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        text = out.getvalue()
        self.assertIn("## Full runs", text)
        self.assertIn("| Run | c1_decode", text)
        self.assertIn("**110.0**", text)
        self.assertIn("**40.0**", text)
        self.assertNotIn("| broken |", text)
        self.assertNotIn("| loop |", text)
        self.assertIn("## Voided or failed latest benchmarks", text)
        self.assertIn("broken", text)
        self.assertIn("loop", text)

    def test_report_keeps_runs_when_quality_rows_exist(self):
        rows = [
            {"name": "cand", "kind": "run", "ok": True, "c1_decode": 120},
            {"name": "cand", "kind": "quality", "ok": True,
             "quality": {"mode": "capture", "prompts": 3}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        text = out.getvalue()
        self.assertIn("## Full runs", text)
        self.assertIn("| cand | **120.0** |", text)
        self.assertIn("**120.0**", text)
        self.assertIn("## Quality gates", text)

    def test_report_surfaces_a_quality_mismatch_that_voids_an_earlier_run(self):
        rows = [
            {"name": "cand", "kind": "run", "ok": True, "c1_decode": 120},
            {"name": "cand", "kind": "quality", "ok": False, "flags": ["QUALITY_MISMATCH"],
             "quality": {"mode": "compare", "identical": 2, "normalized": 2, "prompts": 3,
                         "divergences": [{"prompt": 0, "char": 4, "candidate": "candidate text",
                                          "baseline": "baseline text"}]}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        text = out.getvalue()
        self.assertIn("QUALITY\\_MISMATCH", text)
        self.assertIn("candidate text", text)
        self.assertIn("baseline text", text)
        self.assertNotIn("**120.0**", text)

    def test_report_separates_screen_and_full_run_leaderboards(self):
        rows = [
            {"name": "same", "kind": "run", "ok": True, "c1_decode": 100},
            {"name": "same", "kind": "screen", "ok": True, "c1_decode": 900},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        text = out.getvalue()
        self.assertIn("## Full runs", text)
        self.assertIn("## Screens", text)
        self.assertIn("| same | **100.0** |", text)
        self.assertIn("| same | **900.0** |", text)

    def test_report_capture_does_not_clear_an_earlier_quality_mismatch(self):
        rows = [
            {"name": "cand", "kind": "run", "ok": True, "c1_decode": 120},
            {"name": "cand", "kind": "quality", "ok": False, "flags": ["QUALITY_MISMATCH"],
             "quality": {"mode": "compare", "identical": 2, "normalized": 2, "prompts": 3, "divergences": []}},
            {"name": "cand", "kind": "quality", "ok": True,
             "quality": {"mode": "capture", "prompts": 3}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        text = out.getvalue()
        self.assertIn("QUALITY\\_MISMATCH", text)
        self.assertNotIn("**120.0**", text)

    def test_report_successful_comparison_can_clear_an_earlier_mismatch(self):
        rows = [
            {"name": "cand", "kind": "run", "ok": True, "c1_decode": 120},
            {"name": "cand", "kind": "quality", "ok": False, "flags": ["QUALITY_MISMATCH"],
             "quality": {"mode": "compare", "identical": 2, "normalized": 2, "prompts": 3, "divergences": []}},
            {"name": "cand", "kind": "quality", "ok": True,
             "quality": {"mode": "compare", "identical": 3, "normalized": 3, "prompts": 3, "divergences": []}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        self.assertIn("**120.0**", out.getvalue())

    def test_report_rerun_does_not_clear_an_earlier_quality_mismatch(self):
        rows = [
            {"name": "cand", "kind": "quality", "ok": False, "flags": ["QUALITY_MISMATCH"],
             "quality": {"mode": "compare", "identical": 2, "normalized": 2, "prompts": 3, "divergences": []}},
            {"name": "cand", "kind": "run", "ok": True, "c1_decode": 120},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        self.assertIn("QUALITY\\_MISMATCH", out.getvalue())
        self.assertNotIn("**120.0**", out.getvalue())

    def test_report_escapes_model_controlled_markdown(self):
        rows = [
            {"name": "cand", "kind": "run", "ok": True, "c1_decode": 120},
            {"name": "cand", "kind": "quality", "ok": False, "flags": ["QUALITY_MISMATCH"],
             "quality": {"mode": "compare", "identical": 2, "normalized": 2, "prompts": 3,
                         "divergences": [{"prompt": 0, "char": 0,
                                          "candidate": "![pixel](https://example.invalid/track.png)",
                                          "baseline": "<img src=x>"}]}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            write(path, "".join(json.dumps(r) + "\n" for r in rows))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                bench.report(path)
        text = out.getvalue()
        self.assertNotIn("![pixel](https://", text)
        self.assertNotIn("<img src=x>", text)
        self.assertIn("!\\[pixel\\]", text)
        self.assertIn("&lt;img", text)

    def test_quality_requires_a_matching_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            write(path, json.dumps({"panel": "other", "outputs": []}))
            with self.assertRaises(SystemExit):
                bench.load_quality(path)


class FakeServer:
    def __init__(self, *args):
        self.server = fake_server.serve(fake_server.parse(["--port", "0", *args]))
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        os.makedirs(os.path.join(self.dir, "configs"))
        os.chdir(self.dir)
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.close()
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def config(self, name, *server_args, **extra):
        server = FakeServer(*server_args)
        self.servers.append(server)
        cfg = {"name": name, "base_url": server.url, "model": "fake", "concurrency": 2,
               "prefill_lengths": [512], "temperature": 0.6, "require_ac": False}
        cfg.update(extra)
        path = os.path.join("configs", name + ".json")
        write(path, json.dumps(cfg))
        return path

    def bench(self, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(SCRIPTS, "bench.py"), *args],
                              cwd=self.dir, capture_output=True, text=True, timeout=30)

    def ledger(self):
        with open("ledger.jsonl") as f:
            return [json.loads(line) for line in f]

    def test_screen_passes_prose_prefill_code_tools_and_prefix_cache(self):
        result = self.bench("screen", self.config("screen"), "--budget", "10", "--prefill", "1024")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertTrue(row["ok"], row)
        self.assertEqual(row["kind"], "screen")
        self.assertTrue(row["screen"]["code_edit_ok"])
        self.assertTrue(row["screen"]["tool_call_ok"])
        self.assertTrue(row["screen"]["prefix_cache"]["hit"])
        self.assertGreater(row["c1_decode"], 0)
        self.assertGreater(row["prefill_1024"], 0)

    def test_screen_accepts_fragmented_tool_calls(self):
        result = self.bench("screen", self.config("fragmented", "--fragment-tool"), "--budget", "10", "--prefill", "1024")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue(self.ledger()[-1]["screen"]["tool_call_ok"])

    def test_screen_rejects_a_truncated_tool_stream(self):
        result = self.bench("screen", self.config("truncated", "--truncate-tool"), "--budget", "10", "--prefill", "1024")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertFalse(row["screen"]["tool_call_ok"])
        self.assertIn("SCREEN_TOOL_FAIL", row["flags"])

    def test_screen_voids_tool_code_and_prefix_failures(self):
        result = self.bench("screen", self.config("bad", "--tool-fail", "--code-fail", "--no-prefix"),
                            "--budget", "10", "--prefill", "1024")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertFalse(row["ok"])
        self.assertIn("SCREEN_TOOL_FAIL", row["flags"])
        self.assertIn("SCREEN_CODE_FAIL", row["flags"])
        self.assertIn("SCREEN_PREFIX_FAIL", row["flags"])

    def test_chat_deadline_is_total_not_per_read(self):
        server = FakeServer("--drip-greedy")
        self.servers.append(server)
        srv = bench.Server({"model": "fake", "base_url": server.url})
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            srv.chat("slow", 1, temperature=0.0, timeout=5, deadline_s=started + 0.5)
        self.assertLess(time.monotonic() - started, 0.75)

    def test_screen_deadline_stops_a_greedy_http_hang(self):
        result = self.bench("screen", self.config("hang", "--hang-greedy", "--hang-seconds", "2"),
                            "--budget", "1", "--prefill", "512")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertFalse(row["ok"])
        self.assertIn("SCREEN_TIMEOUT", row["flags"])
        self.assertLessEqual(row["screen"]["elapsed_s"], 2.5)

    def test_quality_capture_compare_and_mismatch(self):
        cfg = self.config("quality")
        result = self.bench("quality", cfg, "--write-baseline", "quality/base.json", "--prompts", "2",
                            "--max-tokens", "32")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        result = self.bench("quality", cfg, "--baseline", "quality/base.json", "--prompts", "2",
                            "--max-tokens", "32")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.ledger()[-1]["quality"]["identical"], 2)
        baseline = read_json("quality/base.json")
        baseline["outputs"][0] = "different output"
        write("quality/base.json", json.dumps(baseline))
        result = self.bench("quality", cfg, "--baseline", "quality/base.json", "--prompts", "2",
                            "--max-tokens", "32")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertFalse(row["ok"])
        self.assertIn("QUALITY_MISMATCH", row["flags"])
        self.assertEqual(len(row["quality"]["divergences"]), 1)

    def test_original_run_command_still_records_measurements(self):
        result = self.bench("run", self.config("run"), "--reps", "2", "--skip-cN")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertTrue(row["ok"], row)
        self.assertEqual(len(row["c1_all"]), 2)
        self.assertIn("prefill_512_tokens", row)

    def test_screen_records_an_unreachable_server_as_a_failed_run(self):
        write("configs/down.json", json.dumps({"name": "down", "base_url": "http://127.0.0.1:1",
                                               "model": "fake", "require_ac": False}))
        result = self.bench("screen", "configs/down.json", "--budget", "2")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertFalse(row["ok"])
        self.assertIn("server not reachable", row["error"])

    def test_screen_records_a_launch_failure(self):
        write("configs/launch-fail.json", json.dumps({"name": "launch-fail", "model": "fake",
                                                      "launch": ["false"], "require_ac": False}))
        result = self.bench("screen", "configs/launch-fail.json", "--budget", "2")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertFalse(row["ok"])
        self.assertIn("launch failed", row["error"])

    def test_queue_runs_the_screen_mode_end_to_end(self):
        self.config("queued")
        env = dict(os.environ, QUEUE_LOCK=os.path.join(self.dir, "queue.lock"), BENCH_MODE="screen",
                   BENCH_FLAGS="--budget 5 --prefill 512")
        result = subprocess.run(["bash", os.path.join(SCRIPTS, "queue.sh"), "queued"], cwd=self.dir,
                                capture_output=True, text=True, env=env, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("QUEUE_DONE", result.stdout)
        self.assertEqual(self.ledger()[-1]["kind"], "screen")
        self.assertTrue(self.ledger()[-1]["ok"])

    def test_screen_bounds_a_launch_that_never_daemonizes(self):
        write("configs/launch-hang.json", json.dumps({"name": "launch-hang", "model": "fake", "require_ac": False,
                                                      "launch_timeout_s": 0.1,
                                                      "launch": [sys.executable, "-c", "import time; time.sleep(2)"]}))
        result = self.bench("screen", "configs/launch-hang.json", "--budget", "2")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        row = self.ledger()[-1]
        self.assertFalse(row["ok"])
        self.assertIn("launch failed", row["error"])

    def test_health_deadline_stops_a_slow_drip(self):
        server = FakeServer("--drip-health")
        self.servers.append(server)
        srv = bench.Server({"base_url": server.url})
        started = time.monotonic()
        self.assertFalse(srv.healthy(deadline_s=started + 0.5))
        self.assertLess(time.monotonic() - started, 0.75)

    def test_quality_budget_stops_a_hung_request(self):
        cfg = self.config("quality-hang", "--hang-all", "--hang-seconds", "2")
        started = time.monotonic()
        result = self.bench("quality", cfg, "--write-baseline", "quality/hang.json", "--prompts", "1",
                            "--max-tokens", "16", "--budget", "1")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertLess(time.monotonic() - started, 2.5)
        row = self.ledger()[-1]
        self.assertIn("QUALITY_TIMEOUT", row["flags"])

    def test_stop_command_timeout_is_bounded_and_recorded(self):
        write("configs/stop-hang.json", json.dumps({"name": "stop-hang", "model": "fake", "require_ac": False,
                                                    "stop_timeout_s": 0.1,
                                                    "stop": [sys.executable, "-c", "import time; time.sleep(2)"],
                                                    "launch": ["false"]}))
        started = time.monotonic()
        result = self.bench("screen", "configs/stop-hang.json", "--budget", "2")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIn("stop command failed", self.ledger()[-1]["error"])

    def test_memory_cap_without_a_probe_is_refused(self):
        cfg = self.config("no-probe", memory={"cap_gib": 1})
        result = self.bench("screen", cfg, "--budget", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("memory.probe", result.stderr)
        self.assertFalse(os.path.exists("ledger.jsonl"))

    def test_memory_probe_failure_voids_the_run(self):
        cfg = self.config("bad-probe", memory={"cap_gib": 100, "probe": ["false"]})
        result = self.bench("screen", cfg, "--budget", "2")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertIn("MEM_PROBE_FAIL", self.ledger()[-1]["flags"])

    def test_working_memory_probe_allows_the_run(self):
        cfg = self.config("good-probe", memory={"cap_gib": 100,
                                                "probe": [sys.executable, "-c", "print(0)"]})
        result = self.bench("screen", cfg, "--budget", "2")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.ledger()[-1]["mem_peak_gib"], 0.0)

    def test_quality_budget_includes_a_hanging_stop_command(self):
        write("configs/quality-stop-hang.json", json.dumps({
            "name": "quality-stop-hang", "model": "fake", "require_ac": False,
            "stop": [sys.executable, "-c", "import time; time.sleep(5)"], "launch": ["true"]}))
        started = time.monotonic()
        result = self.bench("quality", "configs/quality-stop-hang.json", "--write-baseline", "quality/stop.json",
                            "--prompts", "1", "--max-tokens", "16", "--budget", "1")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertIn("stop command failed", self.ledger()[-1]["error"])

    def test_quality_budget_includes_slow_memory_probe_cleanup(self):
        cfg = self.config("quality-probe", "--tps", "2", memory={"cap_gib": 100, "interval_s": 0.1,
                                                   "probe": [sys.executable, "-c",
                                                             "import time; time.sleep(3); print(0)"]})
        started = time.monotonic()
        result = self.bench("quality", cfg, "--write-baseline", "quality/probe.json", "--prompts", "1",
                            "--max-tokens", "8", "--budget", "1")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertLess(time.monotonic() - started, 2.0)
        row = self.ledger()[-1]
        self.assertIn("QUALITY_TIMEOUT", row["flags"])
        self.assertGreaterEqual(row["quality"]["elapsed_s"], 0.9)

    def test_startup_failure_with_a_memory_probe_still_records_a_ledger_row(self):
        cases = {
            "launch-fail-probe": {"launch": ["false"]},
            "down-probe": {"base_url": "http://127.0.0.1:1"},
        }
        for name, extra in cases.items():
            with self.subTest(name=name):
                cfg = {"name": name, "model": "fake", "require_ac": False,
                       "memory": {"cap_gib": 10, "probe": ["true"]}}
                cfg.update(extra)
                write(os.path.join("configs", name + ".json"), json.dumps(cfg))
                result = self.bench("screen", os.path.join("configs", name + ".json"), "--budget", "2")
                self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
                self.assertFalse(self.ledger()[-1]["ok"])
                self.assertNotIn("cannot join thread", result.stderr)

    def test_quality_memory_cap_stop_does_not_survive_the_budget(self):
        pidfile = os.path.join(self.dir, "stop.pid")
        stop_code = (f"import os, time; open({pidfile!r}, 'w').write(str(os.getpid())); "
                     "time.sleep(5)")
        cfg = self.config("cap-stop", memory={"cap_gib": 0.1,
                                              "probe": [sys.executable, "-c", "print(2**30)"]},
                          stop=[sys.executable, "-c", stop_code])
        result = self.bench("quality", cfg, "--write-baseline", "quality/cap.json", "--prompts", "1",
                            "--max-tokens", "8", "--budget", "1")
        self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
        self.assertIn("MEM_CAP", self.ledger()[-1]["flags"])
        if not os.path.exists(pidfile):
            return  # cleanup cancelled the stop command before it could publish its PID
        with open(pidfile) as f:
            pid = int(f.read())
        for _ in range(100):
            stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
            if stat.returncode or stat.stdout.strip().startswith("Z"):
                break
            time.sleep(0.02)
        else:
            self.fail(f"stop process {pid} survived the quality budget")

    def test_report_cli_emits_markdown(self):
        write("ledger.jsonl", json.dumps({"name": "best", "ok": True, "c1_decode": 42}) + "\n")
        result = self.bench("report")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Benchmark report", result.stdout)
        self.assertIn("**42.0**", result.stdout)


if __name__ == "__main__":
    unittest.main()
