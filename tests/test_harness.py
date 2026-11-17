"""Tests for the inference-optimisation harness. Standard library only:  python3 -m unittest discover -s tests -v"""
import json, os, shutil, subprocess, sys, tempfile, threading, unittest

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
    def test_c1_matches_worked_example(self):
        # references/ceilings.md: no speculation, 1.68 GB/token at 115 GB/s -> ~68 tok/s at short context
        self.assertAlmostEqual(ceiling.decode_ceiling(ornith(), 0, 1.0, 4096, 1), 68, delta=1.5)

    def test_concurrency_is_not_double_counted(self):
        m = ornith()
        c1 = ceiling.decode_ceiling(m, 0, 1.0, 4096, 1)
        c4 = ceiling.decode_ceiling(m, 0, 1.0, 4096, 4)
        self.assertGreater(c4, c1)              # batching helps...
        self.assertLess(c4, 4 * c1)             # ...but can never beat N independent single-user ceilings
        self.assertAlmostEqual(c4, 142, delta=2)  # hand calculation: 115 * 4 / 3.25 GB

    def test_dense_model_scales_linearly_at_zero_context(self):
        m = {"bandwidth_gbs": 100, "dense_gb": 10, "lm_head_gb": 0}
        self.assertAlmostEqual(ceiling.decode_ceiling(m, 0, 1.0, 0, 8), 8 * ceiling.decode_ceiling(m, 0, 1.0, 0, 1))

    def test_draft_weights_read_once_per_step(self):
        m = {"bandwidth_gbs": 100, "dense_gb": 1, "lm_head_gb": 0, "draft_layer_gb": 1, "draft_head_gb": 0}
        self.assertAlmostEqual(ceiling.step_bytes_gb(m, 1, 0, 4), 2.0)

    def test_validation(self):
        with self.assertRaises(ceiling.ConfigError):
            ceiling.compute({"bandwidth_gbs": 100, "dense_gb": 1})
        with self.assertRaises(ceiling.ConfigError):
            ceiling.compute({"bandwidth_gbs": 100, "dense_gb": 1, "lm_head_gb": 0, "experts_gb": 5})
        with self.assertRaises(ceiling.ConfigError):
            ceiling.compute({"bandwidth_gbs": 100, "dense_gb": 1, "lm_head_gb": 0, "spec": [{"drafts": 1, "accept": 3}]})

    def test_target_and_kv_fit(self):
        m = ornith()
        res = ceiling.compute(m)
        self.assertTrue(ceiling.target_verdict(m, res, 100)["reachable"])
        self.assertFalse(ceiling.target_verdict(m, res, 150)["reachable"])
        m.update(memory_cap_gb=20, resident_gb=9.3)
        self.assertEqual(ceiling.kv_fit(m, 262144), 3)

    def test_cli_json(self):
        out = subprocess.run([sys.executable, "-I", os.path.join(SCRIPTS, "ceiling.py"),
                              os.path.join(EXAMPLES, "ceiling-ornith-b390.json"), "--json", "--target", "90"],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("reachable", json.loads(out)["target"])


class BenchUnitTests(unittest.TestCase):
    def test_looping(self):
        self.assertTrue(bench.looping("one two three four five six seven eight " * 30))
        self.assertFalse(bench.looping(" ".join(str(i) for i in range(400))))

    def test_document_has_needles_and_is_unique(self):
        p1, codes = bench.document(2000, seed=1)
        p2, codes2 = bench.document(2000, seed=1)
        self.assertEqual(codes, codes2)
        self.assertNotEqual(p1, p2)  # unique head defeats prefix caching
        for c in codes:
            self.assertIn(c, p1)

    def test_overlap(self):
        full = [{"t_first": 0, "t_end": 10}, {"t_first": 0, "t_end": 10}]
        serial = [{"t_first": 0, "t_end": 10}, {"t_first": 10, "t_end": 20}]
        self.assertEqual(bench.overlap(full), 1.0)
        self.assertEqual(bench.overlap(serial), 0.0)

    def test_bootstrap_ci(self):
        lo, hi = bench.bootstrap_ratio_ci([110, 111, 112, 109, 110, 111], [100, 101, 99, 100, 100, 102])
        self.assertGreater(lo, 1.0)
        lo, hi = bench.bootstrap_ratio_ci([100, 104, 96, 101, 99, 103], [100, 101, 99, 100, 103, 97])
        self.assertLessEqual(lo, 1.0)
        self.assertGreaterEqual(hi, 1.0)

    @staticmethod
    def row(name, c1, all_=None, ok=True, **kw):
        d = {"name": name, "run_id": name + str(id(kw)), "ok": ok, "c1_decode": c1,
             "c1_all": all_ or [c1] * 6, "ttft_short_ms": 50, "startup_s": 40}
        d.update(kw)
        return d

    def test_compare_verdicts(self):
        r = self.row
        ctl = [100, 101, 99, 100, 100, 102]
        rows = [r("ctl", 100, ctl), r("fast", 110, [110, 111, 112, 109, 110, 111]), r("ctl", 100, ctl),
                r("noisy", 101, [100, 104, 96, 101, 99, 103]), r("ctl", 100, ctl),
                r("slowttft", 112, [112] * 6, ttft_short_ms=80), r("ctl", 100, ctl),
                r("broken", 150, ok=False), r("ctl", 100, ctl), r("lonely", 120)]
        v = {x["name"]: x["verdict"] for x in bench.compare(rows, "ctl")}
        self.assertEqual(v, {"fast": "ACCEPT", "noisy": "REPEAT", "slowttft": "REJECT", "broken": "VOID",
                             "lonely": "NO_CONTROLS"})

    def test_compare_must_beat_both_neighbours(self):
        r = self.row
        rows = [r("ctl", 100), r("cand", 105), r("ctl", 106)]  # drifted control: not a win
        self.assertEqual(bench.compare(rows, "ctl")[0]["verdict"], "REJECT")

    def test_select_rows_latest_group(self):
        rows = [{"name": "a", "group": "g1", "pos": 1}, {"name": "b", "group": "g2", "pos": 2},
                {"name": "c", "group": "g2", "pos": 1}]
        self.assertEqual([d["name"] for d in bench.select_rows(rows)], ["c", "b"])


class MemProbeTests(unittest.TestCase):
    def test_units_and_dedup(self):
        root = tempfile.mkdtemp()
        try:
            fd = os.path.join(root, "4242", "fdinfo")
            os.makedirs(fd)
            write(os.path.join(fd, "5"), "drm-client-id:\t7\ndrm-resident-system:\t30720 MiB\ndrm-resident-vram0:\t512 KiB\n")
            write(os.path.join(fd, "6"), "drm-client-id:\t7\ndrm-resident-system:\t30720 MiB\n")  # same client
            write(os.path.join(fd, "7"), "drm-client-id:\t8\ndrm-resident-system:\t4096\n")       # bytes
            write(os.path.join(root, "4242", "status"), "RssAnon:\t1024 kB\n")
            env = dict(os.environ, PROC_ROOT=root)
            out = subprocess.run(["bash", os.path.join(SCRIPTS, "mem.sh"), "--pid", "4242", "unified"],
                                 capture_output=True, text=True, env=env, check=True).stdout.strip()
            self.assertEqual(int(out), 30720 * 2**20 + 512 * 1024 + 4096 + 1024 * 1024)
        finally:
            shutil.rmtree(root)


class FakeServer:
    def __init__(self, *args):
        self.srv = fake_server.serve(fake_server.parse(["--port", "0", *args]))
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "configs"))
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.close()
        shutil.rmtree(self.dir)

    def config(self, name, *server_args, **extra):
        s = FakeServer(*server_args)
        self.servers.append(s)
        cfg = {"name": name, "base_url": s.url, "model": "fake", "concurrency": 3, "prefill_lengths": [1024, 4096],
               "temperature": 0.6}
        cfg.update(extra)
        write(os.path.join(self.dir, "configs", name + ".json"), json.dumps(cfg))
        return os.path.join("configs", name + ".json")

    def bench(self, *args):
        return subprocess.run([sys.executable, "-I", os.path.join(SCRIPTS, "bench.py"), *args], cwd=self.dir,
                              capture_output=True, text=True)

    def ledger(self):
        with open(os.path.join(self.dir, "ledger.jsonl")) as f:
            return [json.loads(l) for l in f]

    def test_run_records_everything(self):
        p = self.bench("run", self.config("base", "--tps", "2000"), "--long", "8000", "--reps", "3")
        self.assertEqual(p.returncode, 0, p.stderr)
        d = self.ledger()[-1]
        self.assertTrue(d["ok"], d)
        self.assertEqual(d["flags"], [])
        self.assertGreater(d["c1_decode"], 500)
        self.assertEqual(len(d["c1_all"]), 3)
        self.assertGreater(d["c3_overlap"], 0.8)
        self.assertEqual(d["long"]["retrieval"], "3/3")
        for k in ("run_id", "config_hash", "panel", "env_start", "env_end", "prefill_1024_tokens"):
            self.assertIn(k, d)

    def test_flags_void_bad_runs(self):
        self.assertEqual(self.bench("run", self.config("loop", "--tps", "4000", "--loop"), "--reps", "2").returncode, 1)
        self.assertIn("LOOP", self.ledger()[-1]["flags"])
        self.assertEqual(self.bench("run", self.config("nousage", "--tps", "4000", "--no-usage"), "--reps", "2").returncode, 1)
        self.assertIn("NO_USAGE", self.ledger()[-1]["flags"])
        self.assertEqual(self.bench("run", self.config("serial", "--tps", "4000", "--serial"), "--reps", "2").returncode, 0)
        self.assertIn("NOT_CONCURRENT", self.ledger()[-1]["flags"])

    def test_profile_guard_refuses(self):
        p = self.bench("run", self.config("guard", require_profile="definitely-not-a-profile"))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refusing to run", p.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "ledger.jsonl")))

    def test_soak(self):
        p = self.bench("soak", self.config("soak", "--tps", "4000"), "--n", "3", "--tokens", "6000")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.ledger()[-1]["soak_found"], "3/3")

    def test_greedy_equiv(self):
        a = self.config("a", "--tps", "8000")
        self.assertEqual(self.bench("greedy", a, "--out", "g/a.json", "--max-tokens", "32").returncode, 0)
        self.assertEqual(self.bench("greedy", a, "--out", "g/b.json", "--max-tokens", "32").returncode, 0)
        p = self.bench("equiv", "g/a.json", "g/b.json")
        self.assertEqual(p.returncode, 0, p.stdout)
        b = read_json(os.path.join(self.dir, "g/b.json"))
        b["outputs"][2] = b["outputs"][2][:10] + "X" + b["outputs"][2][11:]
        write(os.path.join(self.dir, "g/b.json"), json.dumps(b))
        p = self.bench("equiv", "g/a.json", "g/b.json")
        self.assertEqual(p.returncode, 1)
        self.assertIn("DIVERGES at char 10", p.stdout)

    def test_queue_and_compare_accepts_a_real_speedup(self):
        if not shutil.which("flock"):
            self.skipTest("flock not installed")
        self.config("ctl", "--tps", "700")
        self.config("cand", "--tps", "1400")
        env = dict(os.environ, QUEUE_LOCK=os.path.join(self.dir, "lock"), QUEUE_GROUP="g1",
                   BENCH_FLAGS="--reps 6 --skip-cN")
        p = subprocess.run(["bash", os.path.join(SCRIPTS, "queue.sh"), "ctl", "cand", "ctl"], cwd=self.dir,
                           capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("failed=0", p.stdout)
        ids = [d["run_id"] for d in self.ledger()]
        self.assertEqual(ids, ["ctl@g1#1", "cand@g1#2", "ctl@g1#3"])
        self.assertEqual(len(os.listdir(os.path.join(self.dir, "runs"))), 3)  # no log overwritten
        out = json.loads(self.bench("compare", "--control", "ctl", "--json").stdout)
        self.assertEqual(out[0]["verdict"], "ACCEPT", out)
        self.assertGreater(out[0]["ci95_ratio"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
