import contextlib
import importlib.util
import io
import json
import math
import socket
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "inference-optimisation" / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_script("preflight")
ceiling = load_script("ceiling")


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        nbytes = math.prod(shape) * preflight.DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return offset


@contextlib.contextmanager
def darwin(chip="Apple M2 Pro", memory=16 * 2 ** 30, battery=False):
    outputs = {
        ("sysctl", "-n", "machdep.cpu.brand_string"): chip + "\n",
        ("sysctl", "-n", "hw.model"): "Mac15,6\n",
        ("sysctl", "-n", "hw.memsize"): str(memory) + "\n",
        ("pmset", "-g", "batt"): "Now drawing from 'Battery Power'\n" if battery else "Now drawing from 'AC Power'\n",
    }

    def run(args, **kwargs):
        if tuple(args) not in outputs:
            raise AssertionError(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=outputs[tuple(args)], stderr="")

    with mock.patch.object(preflight.platform, "system", return_value="Darwin"), \
            mock.patch.object(preflight.subprocess, "run", side_effect=run):
        yield


def call_main(args, chip="Apple M2 Pro", battery=False, port=18000):
    stdout, stderr = io.StringIO(), io.StringIO()
    with darwin(chip=chip, battery=battery), \
            mock.patch.object(preflight, "choose_port", return_value=port), \
            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = preflight.main(args)
    return code, stdout.getvalue(), stderr.getvalue()


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def dense_model(self, name="dense-model"):
        root = self.base / name
        root.mkdir()
        write_json(root / "config.json", {
            "architectures": ["LlamaForCausalLM"], "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": 64, "tie_word_embeddings": False, "torch_dtype": "float32",
        })
        tensors = {
            "model.embed_tokens.weight": ("F32", [64, 32]),
            "model.layers.0.self_attn.q_proj.weight": ("F32", [32, 32]),
            "model.layers.0.self_attn.k_proj.weight": ("F32", [16, 32]),
            "model.layers.0.self_attn.v_proj.weight": ("F32", [16, 32]),
            "model.layers.0.self_attn.o_proj.weight": ("F32", [32, 32]),
            "model.layers.0.mlp.gate_proj.weight": ("F32", [64, 32]),
            "model.layers.0.mlp.up_proj.weight": ("F32", [64, 32]),
            "model.layers.0.mlp.down_proj.weight": ("F32", [32, 64]),
            "lm_head.weight": ("F32", [64, 32]),
        }
        write_safetensors(root / "model.safetensors", tensors)
        return root

    def moe_model(self):
        root = self.base / "moe-model"
        root.mkdir()
        write_json(root / "config.json", {
            "architectures": ["Qwen2MoEForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "vocab_size": 256, "tie_word_embeddings": True,
            "num_experts": 8, "num_experts_per_tok": 2, "torch_dtype": "bfloat16",
        })
        tensors = {"model.embed_tokens.weight": ("BF16", [256, 64]),
                   "lm_head.weight": ("BF16", [256, 64])}
        for layer in range(2):
            prefix = f"model.layers.{layer}"
            tensors.update({
                f"{prefix}.self_attn.q_proj.weight": ("BF16", [64, 64]),
                f"{prefix}.self_attn.k_proj.weight": ("BF16", [32, 64]),
                f"{prefix}.self_attn.v_proj.weight": ("BF16", [32, 64]),
                f"{prefix}.self_attn.o_proj.weight": ("BF16", [64, 64]),
                f"{prefix}.mlp.gate_proj.weight": ("BF16", [128, 64]),
                f"{prefix}.mlp.up_proj.weight": ("BF16", [128, 64]),
                f"{prefix}.mlp.down_proj.weight": ("BF16", [64, 128]),
                f"{prefix}.mlp.gate.weight": ("BF16", [8, 64]),
                f"{prefix}.mlp.experts.gate_up_proj": ("BF16", [8, 256, 64]),
                f"{prefix}.mlp.experts.down_proj": ("BF16", [8, 64, 128]),
            })
        write_safetensors(root / "model.safetensors", tensors)
        return root

    def assert_ceiling_usable(self, data):
        required = ("bandwidth_gbs", "peak_tflops", "mfu", "dense_gb", "experts_gb", "top_k", "n_experts",
                    "lm_head_gb", "active_params_b", "attn_layers", "attn_heads", "head_dim", "kv_bytes_per_token")
        for key in required:
            self.assertIsInstance(data[key], (int, float), key)
            self.assertTrue(math.isfinite(data[key]), key)
        for n in data["concurrency"]:
            for spec in data["spec"]:
                for ctx in data["contexts"]:
                    value = ceiling.decode_ceiling(data, spec["drafts"], spec["accept"], ctx, n)
                    self.assertTrue(math.isfinite(value) and value > 0)
        for ctx in data["contexts"]:
            value = ceiling.prefill_ceiling(data, ctx)
            self.assertTrue(math.isfinite(value) and value > 0)

    def test_dense_model_generates_valid_files_and_provenance(self):
        model = self.dense_model()
        out = self.base / "run"
        code, stdout, stderr = call_main([
            "--model-path", str(model), "--model", "served-dense", "--out", str(out), "--port", "18000",
            "--context", "4096,8192", "--context", "16384", "--concurrency", "2",
        ])
        self.assertEqual(code, 0, stderr)
        data = json.loads((out / "ceiling.json").read_text())
        config = json.loads((out / "config.json").read_text())
        self.assert_ceiling_usable(data)
        self.assertEqual(data["model"], "served-dense")
        self.assertEqual(data["contexts"], [4096, 8192, 16384])
        self.assertEqual(data["concurrency"], [1, 2])
        self.assertEqual(data["experts_gb"], 0)
        self.assertEqual(data["top_k"], 1)
        self.assertEqual(data["n_experts"], 1)
        self.assertAlmostEqual(data["total_weight_gb"], 53248 / 1e9)
        self.assertAlmostEqual(data["dense_gb"], 45056 / 1e9)
        self.assertAlmostEqual(data["lm_head_gb"], 8192 / 1e9)
        self.assertAlmostEqual(data["total_params_b"], 13312 / 1e9)
        self.assertEqual(data["active_params_b"], data["total_params_b"])
        self.assertEqual(data["attn_layers"], 1)
        self.assertEqual(data["attn_heads"], 4)
        self.assertEqual(data["head_dim"], 8)
        self.assertEqual(data["kv_bytes_per_token"], 128)
        self.assertEqual(data["memory_cap_gib"], 12.0)
        self.assertEqual(data["inventory"]["hardware"]["observed"]["chip"]["value"], "Apple M2 Pro")
        self.assertEqual(data["inventory"]["hardware"]["observed"]["memory_bytes"]["value"], 16 * 2 ** 30)
        self.assertEqual(data["inventory"]["power"]["status"], "ac")
        self.assertIn("pmset", data["inventory"]["power"]["sources"][0])
        self.assertEqual(data["inventory"]["model"]["weight_bytes"]["status"], "inferred")
        self.assertEqual(data["inventory"]["bandwidth_gbs"]["status"], "assumed")
        self.assertEqual(preflight.parse_args([])[1].port, 8000)
        self.assertIn("not benchmark", stdout)
        self.assertEqual(config["name"], "served-dense")
        self.assertEqual(config["base_url"], "http://127.0.0.1:18000")
        self.assertEqual(config["model"], "served-dense")
        self.assertEqual(config["concurrency"], 2)
        self.assertEqual(config["prefill_lengths"], [4096, 8192, 16384])
        self.assertEqual(config["temperature"], 0.6)
        self.assertTrue(config["require_ac"])
        self.assertEqual(config["suggested_memory_cap_gib"], 12.0)
        self.assertNotIn("memory", config)
        self.assertTrue(any("not enforced" in w for w in data["assumptions"]))
        self.assertEqual(config["model_path"], str(model))
        self.assertIn("Record rejected changes", (out / "DEAD-ENDS.md").read_text())
        self.assertIn("| Date | Change |", (out / "DEAD-ENDS.md").read_text())

    def test_moe_components_active_params_and_tied_head(self):
        model = self.moe_model()
        out = self.base / "moe-run"
        code, _, stderr = call_main(["--model-path", str(model), "--out", str(out)])
        self.assertEqual(code, 0, stderr)
        data = json.loads((out / "ceiling.json").read_text())
        self.assert_ceiling_usable(data)
        self.assertAlmostEqual(data["total_weight_gb"], 1001472 / 1e9)
        self.assertAlmostEqual(data["dense_gb"], 182272 / 1e9)
        self.assertAlmostEqual(data["experts_gb"], 786432 / 1e9)
        self.assertEqual(data["lm_head_gb"], 0)
        self.assertEqual(data["top_k"], 2)
        self.assertEqual(data["n_experts"], 8)
        self.assertAlmostEqual(data["total_params_b"], 484352 / 1e9)
        self.assertAlmostEqual(data["active_params_b"], 189440 / 1e9)
        self.assertLess(data["active_params_b"], data["total_params_b"])
        self.assertEqual(data["attn_layers"], 2)
        self.assertEqual(data["kv_bytes_per_token"], 256)
        self.assertTrue(data["inventory"]["model"]["tied_embeddings"])
        self.assertEqual(data["inventory"]["model"]["tied_duplicate_bytes"]["value"], 32768)
        self.assertIn("tied embeddings", " ".join(data["assumptions"]))

    def test_sharded_index_and_bin_sizes(self):
        root = self.base / "sharded"
        root.mkdir()
        write_json(root / "config.json", {
            "architectures": ["LlamaForCausalLM"], "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "vocab_size": 64, "tie_word_embeddings": False,
        })
        (root / "model-00001-of-00002.safetensors").write_bytes(b"one")
        (root / "model-00002-of-00002.safetensors").write_bytes(b"two")
        write_json(root / "model.safetensors.index.json", {
            "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
        })
        args = preflight.parse_args(["--model-path", str(root), "--out", str(self.base / "out")])[1]
        with self.assertRaises(preflight.PreflightError):
            preflight.model_summary(args)

        bins = self.base / "bins"
        bins.mkdir()
        write_json(bins / "config.json", {"tie_word_embeddings": False, "num_hidden_layers": 2})
        (bins / "pytorch_model-00001-of-00002.bin").write_bytes(b"x" * 101)
        (bins / "pytorch_model-00002-of-00002.bin").write_bytes(b"y" * 203)
        write_json(bins / "pytorch_model.bin.index.json", {
            "weight_map": {"a": "pytorch_model-00001-of-00002.bin", "b": "pytorch_model-00002-of-00002.bin"},
        })
        args = preflight.parse_args([
            "--model-path", str(bins), "--params-b", "3", "--out", str(self.base / "bin-out"),
        ])[1]
        data = preflight.model_summary(args)
        self.assertEqual(data["weight_bytes"]["value"], 304)
        self.assertAlmostEqual(data["total_params_b"]["value"], 3.0)
        self.assertAlmostEqual(data["active_params_b"]["value"], 3.0)
        self.assertEqual(data["lm_head_bytes"]["value"], 30)
        self.assertEqual(data["dense_bytes"]["value"], 274)
        self.assertTrue(any(".bin tensor metadata" in w for w in data["warnings"]))

    def test_cli_only_models_and_overrides(self):
        for extra, expected_gb, expected_params in ((["--weights-gb", "4"], 4.0, 8.0),
                                                     (["--params-b", "3"], 1.5, 3.0)):
            with self.subTest(extra=extra):
                out = self.base / extra[0][2:]
                code, _, stderr = call_main([
                    "--model", "cli-model", "--out", str(out), *extra,
                    "--bandwidth-gbs", "123", "--peak-tflops", "17", "--mfu", "0.4",
                    "--memory-cap-gib", "20",
                ])
                self.assertEqual(code, 0, stderr)
                data = json.loads((out / "ceiling.json").read_text())
                config = json.loads((out / "config.json").read_text())
                self.assert_ceiling_usable(data)
                self.assertEqual(data["model"], "cli-model")
                self.assertAlmostEqual(data["total_weight_gb"], expected_gb)
                self.assertAlmostEqual(data["total_params_b"], expected_params)
                self.assertEqual(data["bandwidth_gbs"], 123)
                self.assertEqual(data["peak_tflops"], 17)
                self.assertEqual(data["mfu"], 0.4)
                self.assertEqual(data["memory_cap_gib"], 20)
                self.assertEqual(data["inventory"]["bandwidth_gbs"]["status"], "override")
                self.assertEqual(data["inventory"]["model"]["weight_bytes"]["status"],
                                 "override" if "--weights-gb" in extra else "assumed")
                self.assertEqual(data["inventory"]["model"]["attn_layers"]["status"], "assumed")
                self.assertEqual(data["inventory"]["model"]["kv_bytes_per_token"]["status"], "assumed")
                self.assertNotIn("model_path", config)

    def test_memory_probe_enables_the_generated_watchdog(self):
        model = self.dense_model()
        out = self.base / "probe-run"
        code, _, stderr = call_main(["--model-path", str(model), "--out", str(out),
                                     "--memory-probe", "bash mem.sh srv"])
        self.assertEqual(code, 0, stderr)
        config = json.loads((out / "config.json").read_text())
        self.assertEqual(config["memory"]["cap_gib"], 12.0)
        self.assertEqual(config["memory"]["probe"], ["bash", "mem.sh", "srv"])

    def test_battery_warns_but_config_requires_ac(self):
        model = self.dense_model()
        out = self.base / "battery"
        code, _, stderr = call_main(["--model-path", str(model), "--out", str(out)], battery=True)
        self.assertEqual(code, 0, stderr)
        self.assertIn("battery", stderr.lower())
        data = json.loads((out / "ceiling.json").read_text())
        config = json.loads((out / "config.json").read_text())
        self.assertEqual(data["inventory"]["power"]["status"], "battery")
        self.assertFalse(data["inventory"]["power"]["on_ac"])
        self.assertTrue(config["require_ac"])

    def test_linux_power_supply(self):
        mains = self.base / "mains-online"
        battery = self.base / "battery-online"
        mains.write_text("1\n")
        battery.write_text("0\n")
        with mock.patch.object(preflight.platform, "system", return_value="Linux"), \
                mock.patch.object(preflight.sys, "platform", "linux"), \
                mock.patch.object(preflight.glob, "glob", return_value=[str(battery), str(mains)]):
            self.assertEqual(preflight.power_status()["status"], "ac")
        battery.write_text("0\n")
        with mock.patch.object(preflight.platform, "system", return_value="Linux"), \
                mock.patch.object(preflight.sys, "platform", "linux"), \
                mock.patch.object(preflight.glob, "glob", return_value=[str(battery)]):
            self.assertEqual(preflight.power_status()["status"], "battery")
        with mock.patch.object(preflight.platform, "system", return_value="Linux"), \
                mock.patch.object(preflight.sys, "platform", "linux"), \
                mock.patch.object(preflight.glob, "glob", return_value=[]):
            self.assertEqual(preflight.power_status()["status"], "unknown")

    def test_port_selection_binds_without_starting_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            free = sock.getsockname()[1]
        self.assertEqual(preflight.choose_port(free), free)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            requested = occupied.getsockname()[1]
            chosen = preflight.choose_port(requested)
            self.assertGreater(chosen, requested)
            self.assertLessEqual(chosen, requested + preflight.PORT_WINDOW - 1)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", chosen))
        with mock.patch.object(preflight, "port_available", return_value=False):
            with self.assertRaises(preflight.PreflightError):
                preflight.choose_port(8000)

    def test_occupied_requested_port_is_reported_and_used(self):
        model = self.dense_model()
        out = self.base / "port-run"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            requested = occupied.getsockname()[1]
            stdout, stderr = io.StringIO(), io.StringIO()
            with darwin(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = preflight.main(["--model-path", str(model), "--out", str(out), "--port", str(requested)])
            self.assertEqual(code, 0, stderr.getvalue())
            config = json.loads((out / "config.json").read_text())
            selected = int(config["base_url"].rsplit(":", 1)[1])
            self.assertGreater(selected, requested)
            self.assertIn(f"port: {selected}", stdout.getvalue())
            self.assertIn(f"requested {requested}", stdout.getvalue())
            self.assertEqual(config["base_url"], f"http://127.0.0.1:{selected}")

    def test_no_overwrite_and_force_preserves_dead_ends(self):
        model = self.dense_model()
        out = self.base / "existing"
        args = ["--model-path", str(model), "--out", str(out), "--context", "4096"]
        self.assertEqual(call_main(args)[0], 0)
        ceiling_path, config_path = out / "ceiling.json", out / "config.json"
        ceiling_path.write_text("sentinel ceiling")
        config_path.write_text("sentinel config")
        code, _, stderr = call_main(args)
        self.assertEqual(code, 2)
        self.assertIn("--force", stderr)
        self.assertEqual(ceiling_path.read_text(), "sentinel ceiling")
        self.assertEqual(config_path.read_text(), "sentinel config")
        dead_ends = out / "DEAD-ENDS.md"
        dead_ends.write_text(dead_ends.read_text() + "\nKeep this rejected experiment.\n")
        before = dead_ends.read_text()
        code, _, stderr = call_main(args + ["--force", "--context", "8192"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(ceiling_path.read_text())["contexts"], [4096, 8192])
        self.assertEqual(dead_ends.read_text(), before)
        self.assertFalse(list(out.glob(".*.json.*")))

    def test_invalid_paths_arguments_and_model_files(self):
        code, _, stderr = call_main([])
        self.assertEqual(code, 2)
        self.assertIn("--model-path", stderr)
        code, _, stderr = call_main(["--model-path", str(self.base / "missing"), "--model", "x"])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", stderr)
        code, _, stderr = call_main(["--model", "x", "--weights-gb", "1", "--mfu", "1.1"])
        self.assertEqual(code, 2)
        self.assertIn("--mfu", stderr)
        for option in ("--weights-gb", "--params-b"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    preflight.parse_args(["--model", "x", option, "1e308"])
            self.assertEqual(caught.exception.code, 2)
        code, _, stderr = call_main(["--model", "x", "--weights-gb", "1", "--context", str(10 ** 400)])
        self.assertEqual(code, 2)
        self.assertIn("context", stderr)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                preflight.parse_args(["--model", "x", "--weights-gb", "1", "--concurrency", str(10 ** 400)])
        self.assertEqual(caught.exception.code, 2)
        code, _, stderr = call_main(["--model", "x", "--weights-gb", "1", "--context", "0"])
        self.assertEqual(code, 2)
        self.assertIn("context", stderr)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                preflight.parse_args(["--model", "x", "--weights-gb", "1", "--port", "70000"])
        self.assertEqual(caught.exception.code, 2)

        empty = self.base / "empty"
        empty.mkdir()
        code, _, stderr = call_main(["--model-path", str(empty)])
        self.assertEqual(code, 2)
        self.assertIn("no safetensors/.bin weights", stderr)

        corrupt = self.base / "corrupt"
        corrupt.mkdir()
        write_json(corrupt / "config.json", {"num_hidden_layers": 1})
        (corrupt / "model.safetensors").write_bytes(b"short")
        code, _, stderr = call_main(["--model-path", str(corrupt)])
        self.assertEqual(code, 2)
        self.assertIn("truncated safetensors", stderr)

        for name, shape in (("bad-span", [1000]), ("huge-span", [10 ** 400])):
            root = self.base / name
            root.mkdir()
            header = {"weight": {"dtype": "F32", "shape": shape, "data_offsets": [0, 4]}}
            raw = json.dumps(header, separators=(",", ":")).encode()
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
            code, _, stderr = call_main(["--model-path", str(root), "--model", "x"])
            self.assertEqual(code, 2, stderr)
            self.assertIn("byte count", stderr)

        bad_index = self.base / "bad-index"
        bad_index.mkdir()
        write_json(bad_index / "model.safetensors.index.json", {"weight_map": {"a": 17}})
        code, _, stderr = call_main(["--model-path", str(bad_index)])
        self.assertEqual(code, 2)
        self.assertIn("filenames", stderr)

        output_file = self.base / "not-a-directory"
        output_file.write_text("keep")
        code, _, stderr = call_main(["--model", "x", "--weights-gb", "1", "--out", str(output_file)])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", stderr)
        self.assertEqual(output_file.read_text(), "keep")

    def test_system_profiler_fallback_for_apple_inventory(self):
        profile = """Hardware:

    Model Name: MacBook Pro
    Model Identifier: Mac15,7
    Chip: Apple M3 Max
    Memory: 16 GB
"""

        def run(args, **kwargs):
            if args[0] == "system_profiler":
                return subprocess.CompletedProcess(args, 0, stdout=profile, stderr="")
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="unavailable")

        with mock.patch.object(preflight.platform, "system", return_value="Darwin"), \
                mock.patch.object(preflight.subprocess, "run", side_effect=run):
            hardware = preflight.apple_inventory()
        self.assertEqual(hardware["observed"]["chip"]["value"], "Apple M3 Max")
        self.assertEqual(hardware["observed"]["model"]["value"], "Mac15,7")
        self.assertEqual(hardware["observed"]["memory_bytes"]["value"], 16_000_000_000)
        self.assertIn("system_profiler", hardware["observed"]["chip"]["source"])
        self.assertEqual(hardware["starter"]["bandwidth_gbs"]["value"], 300)

    def test_generic_linux_fallback(self):
        with mock.patch.object(preflight.platform, "system", return_value="Linux"), \
                mock.patch.object(preflight.sys, "platform", "linux"), \
                mock.patch.object(preflight.shutil, "which", return_value=None), \
                mock.patch.object(preflight, "linux_cpu", return_value="Generic CPU"), \
                mock.patch.object(preflight, "linux_memory_bytes", return_value=32 * 2 ** 30), \
                mock.patch.object(preflight.glob, "glob", return_value=[]):
            hardware = preflight.inventory_hardware()
        self.assertEqual(hardware["observed"]["cpu"]["value"], "Generic CPU")
        self.assertEqual(hardware["starter"]["bandwidth_gbs"]["value"], 50.0)
        self.assertEqual(hardware["starter"]["peak_tflops"]["value"], 2.0)
        self.assertEqual(hardware["starter"]["memory_cap_gib"]["value"], 24.0)
        self.assertTrue(any("generic Linux fallback" in w for w in hardware["warnings"]))

    def test_hybrid_attention_and_deepseek_mla_kv(self):
        cfg = {
            "num_hidden_layers": 4, "layer_types": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
            "num_attention_heads": 16, "head_dim": 192, "torch_dtype": "bfloat16",
            "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
        }
        warnings = []
        details = preflight.attention_details(cfg, [], warnings)
        self.assertEqual(details["attn_layers"]["value"], 2)
        self.assertEqual(details["head_dim"]["value"], 192)
        self.assertEqual(details["kv_bytes_per_token"]["value"], 2 * (512 + 64) * 2)
        self.assertEqual(details["kv_bytes_per_token"]["status"], "inferred")
        self.assertIn("MLA", details["kv_bytes_per_token"]["source"])
        self.assertFalse(warnings)

        tensors = [
            {"name": "model.layers.0.self_attn.q_proj.weight", "shape": [256, 64], "dtype": "BF16"},
            {"name": "model.layers.0.self_attn.k_proj.weight", "shape": [128, 64], "dtype": "BF16"},
        ]
        warnings = []
        details = preflight.attention_details({}, tensors, warnings)
        self.assertEqual(details["attn_layers"]["value"], 1)
        self.assertEqual(details["attn_layers"]["status"], "inferred")
        self.assertEqual(details["attn_heads"]["value"], 2)
        self.assertEqual(details["head_dim"]["status"], "assumed")
        self.assertEqual(details["kv_bytes_per_token"]["value"], 512)
        self.assertTrue(warnings)

    def test_quantized_weight_dtype_does_not_shrink_kv_cache_estimate(self):
        cfg = {"num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
               "head_dim": 8, "torch_dtype": "bfloat16"}
        tensors = [{"name": "model.layers.0.self_attn.k_proj.weight", "shape": [16, 32], "dtype": "U8"}]
        details = preflight.attention_details(cfg, tensors, [])
        self.assertEqual(details["kv_cache_element_bytes"]["value"], 2)
        self.assertEqual(details["kv_bytes_per_token"]["value"], 128)

    def test_unknown_apple_hardware_is_labelled_and_warns(self):
        with darwin(chip="Apple M9 Future"):
            hardware = preflight.inventory_hardware()
        self.assertEqual(hardware["starter"]["bandwidth_gbs"]["value"], 50.0)
        self.assertEqual(hardware["starter"]["peak_tflops"]["value"], 2.0)
        self.assertEqual(hardware["starter"]["bandwidth_gbs"]["status"], "assumed")
        self.assertTrue(any("unknown Apple chip" in w for w in hardware["warnings"]))

    def test_nvidia_inventory_is_observed_not_benchmarked(self):
        row = "0, NVIDIA RTX 4090, 24564, 24000, 564, 550.1, 00000000:01:00.0\n"

        def run(args, **kwargs):
            self.assertTrue(any(arg.startswith("--query-gpu=index,name,memory.total") for arg in args))
            return subprocess.CompletedProcess(args, 0, stdout=row, stderr="")

        with mock.patch.object(preflight.platform, "system", return_value="Linux"), \
                mock.patch.object(preflight.sys, "platform", "linux"), \
                mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(preflight.subprocess, "run", side_effect=run):
            hardware = preflight.inventory_hardware()
        gpu = hardware["observed"]["nvidia_gpus"]["value"][0]
        self.assertEqual(gpu["name"], "NVIDIA RTX 4090")
        self.assertEqual(gpu["memory_bytes"], 24564 * 2 ** 20)
        self.assertEqual(hardware["observed"]["memory_bytes"]["value"], 24564 * 2 ** 20)
        self.assertEqual(hardware["starter"]["bandwidth_gbs"]["status"], "assumed")
        self.assertTrue(any("not sustained bandwidth" in w for w in hardware["warnings"]))

    def test_multi_gpu_cap_defaults_to_the_largest_single_device(self):
        rows = "0, GPU A, 8192, 8000, 192, 550.1, 0000:01:00.0\n1, GPU B, 8192, 8000, 192, 550.1, 0000:02:00.0\n"

        def run(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=rows, stderr="")

        with mock.patch.object(preflight.platform, "system", return_value="Linux"), \
                mock.patch.object(preflight.sys, "platform", "linux"), \
                mock.patch.object(preflight.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(preflight.subprocess, "run", side_effect=run):
            hardware = preflight.inventory_hardware()
        self.assertEqual(hardware["observed"]["memory_bytes"]["value"], 8192 * 2 ** 20)
        self.assertEqual(hardware["starter"]["memory_cap_gib"]["value"], 6.0)
        self.assertEqual(hardware["observed"]["aggregate_gpu_memory_bytes"]["value"], 16384 * 2 ** 20)
        self.assertTrue(any("largest single GPU" in w for w in hardware["warnings"]))

    def test_help_documents_required_options(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                preflight.parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        for option in ("--model", "--model-path", "--weights-gb", "--params-b", "--out", "--port",
                       "--concurrency", "--context", "--memory-cap-gib", "--memory-probe", "--bandwidth-gbs",
                       "--peak-tflops", "--mfu", "--force"):
            self.assertIn(option, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
