#!/usr/bin/env python3
"""Write starter ceiling/config files from a read-only local inventory. No benchmark is run.

  python3 preflight.py --model-path /path/to/model --out runs/my-model
  python3 preflight.py --model my-model --weights-gb 8 --out runs/my-model

Hardware peaks and missing model details are labelled starter assumptions, not measurements.
Replace them with measured bandwidth/FP32 before treating ceiling.py output as evidence.
"""
import argparse
import glob
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

GIB = 2 ** 30
MAX_HEADER = 100_000_000
MAX_NUMBER = 1e12
MAX_CONTEXT = 16 * 1024 * 1024
MAX_CONCURRENCY = 1024
PORT_WINDOW = 100
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
# Theoretical memory bandwidth; FP32 values are deliberately conservative core-count estimates.
APPLE_STARTERS = {
    (1, ""): (68.25, 2.6), (1, "Pro"): (204.8, 5.2), (1, "Max"): (409.6, 10.4), (1, "Ultra"): (819.2, 20.8),
    (2, ""): (100, 3.5), (2, "Pro"): (200, 6.8), (2, "Max"): (400, 13.4), (2, "Ultra"): (800, 26.8),
    (3, ""): (100, 4.0), (3, "Pro"): (150, 7.0), (3, "Max"): (300, 14.0), (3, "Ultra"): (600, 28.0),
    (4, ""): (120, 5.0), (4, "Pro"): (273, 10.0), (4, "Max"): (546, 20.0), (4, "Ultra"): (1092, 40.0),
}
DEAD_ENDS = """# Dead ends

Record rejected changes here so they are not retried without a new reason.

| Date | Change | Result | Why rejected / next hypothesis |
| --- | --- | --- | --- |
"""


class PreflightError(Exception):
    pass


def item(value, status, source):
    return {"value": value, "status": status, "source": source}


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def sysctl(name):
    src = shlex.join(["sysctl", "-n", name])
    out = command(["sysctl", "-n", name])
    return (out, src) if out else (None, src)


def hardware_profile():
    text = command(["system_profiler", "SPHardwareDataType"]) or ""
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.strip().split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def memory_gib(value):
    return round(value / GIB, 3)


def apple_inventory():
    observed, warnings = {}, []
    chip, chip_src = sysctl("machdep.cpu.brand_string")
    model, model_src = sysctl("hw.model")
    mem, mem_src = sysctl("hw.memsize")
    mem_status = "observed"
    if not chip or not model or not mem:
        profile = hardware_profile()
        prof_src = "system_profiler SPHardwareDataType"
        chip = chip or profile.get("Chip")
        model = model or profile.get("Model Identifier") or profile.get("Model Name")
        if chip:
            chip_src = prof_src + " (Chip)"
        if model:
            model_src = prof_src + " (Model Identifier/Model Name)"
        if not mem and profile.get("Memory"):
            match = re.search(r"([\d.]+)\s*(GB|GiB)", profile["Memory"], re.I)
            if match:
                mem = str(int(float(match[1]) * (GIB if match[2].lower() == "gib" else 1e9)))
                mem_src = prof_src + " (Memory; converted to bytes)"
                mem_status = "inferred"
    if chip:
        observed["chip"] = item(chip, "observed", chip_src)
    if model:
        observed["model"] = item(model, "observed", model_src)
    if mem:
        try:
            memory = int(mem)
        except ValueError:
            memory = 0
        if memory > 0:
            observed["memory_bytes"] = item(memory, mem_status, mem_src)
    memory = observed.get("memory_bytes", {}).get("value", 0)
    if not memory:
        warnings.append("could not determine unified-memory size; memory cap falls back to 8 GiB")
    match = re.search(r"Apple\s+M([1-4])\s*(Pro|Max|Ultra)?", chip or "", re.I)
    if match:
        family, variant = int(match[1]), (match[2] or "").title()
        bandwidth, peak = APPLE_STARTERS[(family, variant)]
        source = f"theoretical Apple M{family} {variant or 'base'} starter table (not measured)"
    else:
        bandwidth, peak = 50.0, 2.0
        source = "unknown-hardware conservative starter: 50 GB/s, 2 TFLOPS FP32 (not measured)"
        warnings.append("unknown Apple chip: starter defaults are 50 GB/s / 2 TFLOPS; "
                        "supply --bandwidth-gbs and --peak-tflops or measure them")
    starter = {
        "bandwidth_gbs": item(bandwidth, "assumed", source),
        "peak_tflops": item(peak, "assumed", source),
        "memory_cap_gib": item(memory_gib(memory * 0.75) if memory else 8.0, "assumed",
                               "75% of unified memory, leaving OS/runtime headroom"),
    }
    return {"platform": "darwin", "observed": observed, "starter": starter, "warnings": warnings}


def linux_memory_bytes():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def linux_cpu():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return None


def linux_inventory():
    observed, warnings = {}, []
    gpu_src = "nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,driver_version,pci.bus_id"
    gpu_text = command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,driver_version,pci.bus_id",
                        "--format=csv,noheader,nounits"]) if shutil.which("nvidia-smi") else None
    gpus, gpu_memory, largest_gpu_memory = [], 0, 0
    for line in (gpu_text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            total, free, used = (int(parts[i]) * 2 ** 20 for i in (2, 3, 4))
        except ValueError:
            continue
        gpus.append({"index": parts[0], "name": parts[1], "memory_bytes": total,
                     "free_memory_bytes": free, "used_memory_bytes": used,
                     "driver_version": parts[5], "pci_bus_id": parts[6]})
        gpu_memory += total
        largest_gpu_memory = max(largest_gpu_memory, total)
    if gpus:
        observed["nvidia_gpus"] = item(gpus, "observed", gpu_src)
        observed["aggregate_gpu_memory_bytes"] = item(gpu_memory, "inferred", "sum of nvidia-smi memory.total across GPUs")
    else:
        cpu = linux_cpu()
        if cpu:
            observed["cpu"] = item(cpu, "observed", "/proc/cpuinfo (model name)")
        cards = []
        for device in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            try:
                vendor_path, uevent_path = Path(device, "vendor"), Path(device, "uevent")
                vendor = vendor_path.read_text().strip() if vendor_path.exists() else ""
                name = uevent_path.read_text() if uevent_path.exists() else ""
            except OSError:
                continue
            cards.append({"device": device, "vendor": vendor, "uevent": name.strip()})
        if cards:
            observed["drm_devices"] = item(cards, "observed", "/sys/class/drm/card*/device")
        warnings.append("no NVIDIA GPU inventory available; inspect other accelerators with system tools")
    memory = largest_gpu_memory or linux_memory_bytes()
    if memory:
        observed["memory_bytes"] = item(memory, "inferred", "largest single nvidia-smi memory.total" if gpus else "/proc/meminfo (MemTotal; converted to bytes)")
    else:
        warnings.append("could not determine system/GPU memory; memory cap falls back to 8 GiB")
    if gpus:
        warnings.append("nvidia-smi reports memory, not sustained bandwidth/FP32; starter defaults are "
                        "100 GB/s / 10 TFLOPS -- override or measure them")
        bandwidth, peak = 100.0, 10.0
        if len(gpus) > 1:
            warnings.append("multiple NVIDIA GPUs: memory cap defaults to 75% of the largest single GPU; "
                            "override only after configuring an explicit multi-GPU serving mode")
    else:
        warnings.append("generic Linux fallback: starter defaults are 50 GB/s / 2 TFLOPS; "
                        "supply --bandwidth-gbs and --peak-tflops or measure them")
        bandwidth, peak = 50.0, 2.0
    starter = {
        "bandwidth_gbs": item(bandwidth, "assumed", "conservative starter constant (not measured)"),
        "peak_tflops": item(peak, "assumed", "conservative starter constant (not measured)"),
        "memory_cap_gib": item(memory_gib(memory * 0.75) if memory else 8.0, "assumed",
                               "75% of the largest GPU/system memory, leaving runtime headroom"),
    }
    return {"platform": "linux", "observed": observed, "starter": starter, "warnings": warnings}


def is_macos():
    return sys.platform == "darwin" or platform.system() == "Darwin"


def inventory_hardware():
    return apple_inventory() if is_macos() else linux_inventory()


def power_status():
    if is_macos():
        src = "pmset -g batt"
        text = command(["pmset", "-g", "batt"]) or ""
        if "AC Power" in text:
            return {"status": "ac", "on_ac": True, "sources": [src]}
        if "Battery Power" in text:
            return {"status": "battery", "on_ac": False, "sources": [src]}
        return {"status": "unknown", "on_ac": None, "sources": [src]}
    sources = []
    for path in sorted(glob.glob("/sys/class/power_supply/*/online")):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        sources.append(path)
        if value == "1":
            return {"status": "ac", "on_ac": True, "sources": sources}
    return ({"status": "battery", "on_ac": False, "sources": sources} if sources else
            {"status": "unknown", "on_ac": None, "sources": ["/sys/class/power_supply/*/online"]})


def port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port(requested):
    for port in range(requested, min(requested + PORT_WINDOW, 65536)):
        if port_available(port):
            return port
    raise PreflightError(f"no free localhost port in {requested}..{min(requested + PORT_WINDOW - 1, 65535)}")


def positive(value):
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    if number > MAX_NUMBER:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_NUMBER:g}")
    return number


def port_number(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    if number > MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_CONCURRENCY}")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="served model name (otherwise inferred from --model-path basename)")
    p.add_argument("--model-path", help="local Hugging Face-style model directory; nothing is downloaded")
    p.add_argument("--weights-gb", type=positive, help="stored weight size in decimal GB (overrides checkpoint size)")
    p.add_argument("--params-b", type=positive, help="total parameters in billions (overrides inferred count)")
    p.add_argument("--out", default=".", help="output directory (default: current directory)")
    p.add_argument("--port", type=port_number, default=8000, help="preferred localhost port (default: 8000)")
    p.add_argument("--concurrency", type=positive_int, default=4)
    p.add_argument("--context", dest="contexts", action="append", help="context tokens; repeat or comma-separate")
    p.add_argument("--memory-cap-gib", type=positive, help="benchmark memory cap (default: 75%% of inventory)")
    p.add_argument("--memory-probe", help="command that prints real memory bytes; enables the cap watchdog")
    p.add_argument("--bandwidth-gbs", type=positive, help="measured bandwidth in decimal GB/s")
    p.add_argument("--peak-tflops", type=positive, help="measured FP32 peak in TFLOPS")
    p.add_argument("--mfu", type=positive, help="practical compute fraction (default: 0.5)")
    p.add_argument("--force", action="store_true", help="replace existing ceiling.json/config.json")
    return p, p.parse_args(argv)


def read_json(path):
    try:
        with path.open() as f:
            value = json.load(f)
    except (OSError, ValueError) as e:
        raise PreflightError(f"cannot read {path}: {e}")
    if not isinstance(value, dict):
        raise PreflightError(f"{path} must contain a JSON object")
    return value


def safe_member(root, name):
    if not isinstance(name, str) or not name:
        raise PreflightError("model index weight_map values must be non-empty filenames")
    path = (root / name).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise PreflightError(f"model index references a file outside the model directory: {name}")
    return path


def read_safetensors(path):
    tensors = []
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise PreflightError(f"truncated safetensors header: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= MAX_HEADER:
                raise PreflightError(f"invalid safetensors header length: {path}")
            raw = f.read(length)
            if len(raw) != length:
                raise PreflightError(f"truncated safetensors header: {path}")
            header = json.loads(raw)
        if not isinstance(header, dict):
            raise PreflightError(f"safetensors header is not an object: {path}")
        spans = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if (not isinstance(info, dict) or not isinstance(info.get("dtype"), str) or
                    info.get("dtype") not in DTYPE_BYTES):
                raise PreflightError(f"unsupported or missing tensor dtype for {name} in {path}")
            shape, offsets = info.get("shape"), info.get("data_offsets")
            if (not isinstance(shape, list) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape) or
                    not isinstance(offsets, list) or len(offsets) != 2 or
                    any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in offsets) or offsets[1] < offsets[0]):
                raise PreflightError(f"invalid tensor shape/data_offsets for {name} in {path}")
            start, end = offsets
            expected = math.prod(shape) * DTYPE_BYTES[info["dtype"]]
            if end - start != expected:
                raise PreflightError(f"tensor byte count does not match shape/dtype for {name} in {path}")
            if 8 + length + end > size:
                raise PreflightError(f"truncated tensor data for {name} in {path}")
            spans.append((start, end, name))
            tensors.append({"name": name, "dtype": info["dtype"], "shape": shape,
                            "bytes": end - start, "elements": math.prod(shape), "file": path.name})
        cursor = 0
        for start, end, name in sorted(spans):
            if start < cursor:
                raise PreflightError(f"overlapping tensor data for {name} in {path}")
            cursor = end
    except PreflightError:
        raise
    except (OSError, ValueError, struct.error) as e:
        raise PreflightError(f"cannot read safetensors {path}: {e}")
    if not tensors:
        raise PreflightError(f"no tensors in safetensors file: {path}")
    return tensors


def checkpoint_files(root):
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise PreflightError(f"empty or invalid weight_map in {index}")
        return sorted({safe_member(root, name) for name in weight_map.values()})
    files = sorted(root.glob("*.safetensors"))
    if files:
        return files
    bin_index = root / "pytorch_model.bin.index.json"
    if bin_index.exists():
        weight_map = read_json(bin_index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise PreflightError(f"empty or invalid weight_map in {bin_index}")
        return sorted({safe_member(root, name) for name in weight_map.values()})
    weight_files = sorted(root.glob("*.bin"))
    preferred = [f for f in weight_files if f.name.startswith(("pytorch_model", "model", "weights"))]
    return preferred or weight_files


def is_expert(name):
    parts = name.lower().split(".")
    return any(part in ("experts", "local_experts") for part in parts) and "shared_experts" not in parts


def is_head(name):
    parts = name.lower().split(".")
    return parts[0] in ("lm_head", "output", "embed_out") or parts[-1] in ("lm_head", "embed_out")


def text_config(config):
    cfg = dict(config)
    if isinstance(config.get("text_config"), dict):
        cfg.update(config["text_config"])
    return cfg


def config_int(cfg, keys, default=None):
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def routing(cfg):
    n = config_int(cfg, ("num_experts", "n_routed_experts", "num_local_experts", "n_experts"), 1)
    k = config_int(cfg, ("num_experts_per_tok", "top_k", "topk", "moe_top_k", "num_experts_per_token"), 1)
    if k > n:
        raise PreflightError(f"invalid MoE routing: top-k {k} exceeds {n} experts")
    arch = cfg.get("architectures") or []
    if isinstance(arch, str):
        arch = arch.lower()
    elif isinstance(arch, (list, tuple)):
        arch = " ".join(str(x) for x in arch).lower()
    else:
        arch = ""
    return n, k, n > 1 or "moe" in arch


def parameter_estimate(cfg, n_experts, top_k, moe):
    h = config_int(cfg, ("hidden_size", "n_embd", "d_model"))
    inter = config_int(cfg, ("intermediate_size", "ffn_dim", "n_inner"))
    expert_inter = config_int(cfg, ("moe_intermediate_size",), inter)
    layers = config_int(cfg, ("num_hidden_layers", "n_layer", "num_layers"))
    vocab = config_int(cfg, ("vocab_size",))
    heads = config_int(cfg, ("num_attention_heads", "n_head"))
    kv_heads = config_int(cfg, ("num_key_value_heads", "num_kv_heads"), heads)
    head_dim = config_int(cfg, ("head_dim", "qk_head_dim"))
    if not head_dim and h and heads:
        head_dim = h // heads
    if not all((h, inter, layers, vocab, heads, kv_heads, head_dim)):
        return None
    tied = bool(cfg.get("tie_word_embeddings", False))
    embed = vocab * h
    head = 0 if tied else embed
    attn = h * (heads * head_dim + 2 * kv_heads * head_dim) + heads * head_dim * h
    base = layers * (attn + 2 * h) + h + embed + head
    if not moe:
        total = base + layers * 3 * h * inter
        return total, total
    shared = config_int(cfg, ("n_shared_experts", "num_shared_experts"), 0) or 0
    shared_inter = config_int(cfg, ("shared_expert_intermediate_size",), inter)
    dense_layers = min(config_int(cfg, ("first_k_dense_replace",), 0) or 0, layers)
    moe_layers = layers - dense_layers
    dense = base + dense_layers * 3 * h * inter + moe_layers * (h * n_experts + shared * 3 * h * shared_inter)
    total = dense + moe_layers * n_experts * 3 * h * expert_inter
    active = dense + moe_layers * top_k * 3 * h * expert_inter
    return total, active


def tensor_parameters(tensors, n_experts, top_k, tied=False):
    if not tensors or any(t["dtype"] not in FLOAT_DTYPES for t in tensors):
        return None
    unique = [t for t in tensors if not (tied and is_head(t["name"]))]
    total = sum(t["elements"] for t in unique)
    experts = sum(t["elements"] for t in unique if is_expert(t["name"]))
    return total, total - experts + experts * top_k / n_experts


def attention_details(cfg, tensors, warnings):
    layer_types = cfg.get("layer_types")
    layers = config_int(cfg, ("num_hidden_layers", "n_layer", "num_layers"))
    layer_status = "inferred"
    if isinstance(layer_types, list) and layer_types:
        attn_layers = sum(t not in ("linear_attention", "mamba", "ssm") for t in layer_types)
        layer_src = "config.json layer_types (full/sliding attention layers only)"
    elif layers:
        interval = config_int(cfg, ("full_attention_interval",))
        attn_layers = math.ceil(layers / interval) if interval else layers
        layer_src = "config.json layer count/interval"
    else:
        seen = set()
        for t in tensors:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", t["name"])
            if match and any(x in t["name"] for x in ("q_proj", "k_proj", "v_proj", "self_attn")):
                seen.add(int(match[1]))
        attn_layers = len(seen)
        layer_src = "safetensors attention tensor names"
    h = config_int(cfg, ("hidden_size", "n_embd", "d_model"))
    heads = config_int(cfg, ("num_attention_heads", "n_head"))
    head_dim = config_int(cfg, ("head_dim", "qk_head_dim"))
    head_src = "config.json attention dimensions"
    if not head_dim and h and heads:
        head_dim = h // heads
        head_src = "config.json hidden_size / num_attention_heads"
    q = next((t for t in tensors if t["name"].endswith("q_proj.weight")), None)
    k = next((t for t in tensors if t["name"].endswith("k_proj.weight")), None)
    missing = []
    if not head_dim:
        head_dim, head_src = 128, "starter default: head_dim 128"
        missing.append("head_dim")
    if not heads:
        if q and len(q["shape"]) == 2:
            heads = max(1, math.ceil(q["shape"][0] / head_dim))
            head_src = "q_proj output width / head_dim (rounded up)"
        else:
            heads, head_src = 32, "starter default: 32 heads"
            missing.append("heads")
    if not attn_layers:
        attn_layers, layer_status, layer_src = 32, "assumed", "starter default: 32 layers"
        missing.append("layers")
    if missing:
        warnings.append("attention shape incomplete; starter assumes 32 layers, 32 heads, head_dim 128 where missing")
    head_status = "assumed" if "heads" in missing or "head_dim" in missing else "inferred"
    kv_heads = config_int(cfg, ("num_key_value_heads", "num_kv_heads"), heads)
    kv_head_src = "config.json num_key_value_heads"
    if k and len(k["shape"]) == 2 and not config_int(cfg, ("num_key_value_heads", "num_kv_heads")):
        kv_heads = max(1, k["shape"][0] // head_dim)
        kv_head_src = "safetensors k_proj shape"
    kv_head_status = "assumed" if head_status == "assumed" and kv_heads == heads else "inferred"
    dtype = str(cfg.get("torch_dtype", "")).lower()
    elem = 4 if dtype in ("float32", "fp32", "float") else 1 if "float8" in dtype or "fp8" in dtype else 2
    kv_elem_src, kv_elem_status = "config.json torch_dtype", "inferred"
    if not dtype:
        kv_elem_src, kv_elem_status = "starter default: 2-byte FP16/BF16 KV", "assumed"
    if k and k["dtype"] in FLOAT_DTYPES and not dtype:
        elem, kv_elem_src, kv_elem_status = DTYPE_BYTES[k["dtype"]], "safetensors k_proj dtype", "inferred"
    kv_lora = config_int(cfg, ("kv_lora_rank",))
    rope = config_int(cfg, ("qk_rope_head_dim",), 0) or 0
    if kv_lora:
        nope = config_int(cfg, ("qk_nope_head_dim",), 0) or 0
        head_dim = nope + rope or head_dim
        head_src = "DeepSeek MLA qk_nope_head_dim + qk_rope_head_dim"
        kv = attn_layers * (kv_lora + rope) * elem
        kv_src = "DeepSeek MLA: attention layers x (kv_lora_rank + qk_rope_head_dim) x cache element bytes"
    else:
        kv = 2 * attn_layers * kv_heads * head_dim * elem
        kv_src = "2 x attention layers x KV heads x head_dim x cache element bytes"
    kv_status = "assumed" if "assumed" in (layer_status, head_status, kv_head_status, kv_elem_status) else "inferred"
    return {
        "attn_layers": item(attn_layers, layer_status, layer_src),
        "attn_heads": item(heads, head_status, head_src),
        "head_dim": item(head_dim, head_status, head_src),
        "kv_heads": item(kv_heads, kv_head_status, kv_head_src),
        "kv_cache_element_bytes": item(elem, kv_elem_status, kv_elem_src),
        "kv_bytes_per_token": item(kv, kv_status, kv_src + "; recurrent/SSM state excluded"),
    }


def model_summary(args):
    path = Path(args.model_path).expanduser().resolve() if args.model_path else None
    name = args.model or (path.name if path else None)
    if not name:
        raise PreflightError("supply --model when --model-path is omitted")
    if path and (not path.exists() or not path.is_dir()):
        raise PreflightError(f"model path is not a directory: {path}")
    warnings, tensors, config = [], [], {}
    raw_bytes, raw_dense, raw_experts, raw_head, raw_duplicate = 0, 0, 0, 0, 0
    weight_status, weight_src = "assumed", "starter assumption"
    component_status, component_src = "assumed", "conservative 90% dense / 10% LM-head split"
    if path:
        cfg_path = path / "config.json"
        if cfg_path.exists():
            config = read_json(cfg_path)
        cfg = text_config(config)
        files = checkpoint_files(path)
        if files and all(f.suffix == ".safetensors" for f in files):
            for f in files:
                tensors.extend(read_safetensors(f))
            raw_bytes = sum(t["bytes"] for t in tensors)
            raw_experts = sum(t["bytes"] for t in tensors if is_expert(t["name"]))
            raw_head = sum(t["bytes"] for t in tensors if is_head(t["name"]))
            if cfg.get("tie_word_embeddings") is True:
                raw_duplicate, raw_head = raw_head, 0
            raw_dense = raw_bytes - raw_experts - raw_head - raw_duplicate
            weight_status, weight_src = "inferred", "safetensors header data_offsets and tensor names"
            component_status = "inferred"
            component_src = "safetensors tensor names: routed experts, LM head, and all remaining tensors"
        elif files:
            try:
                raw_bytes = sum(f.stat().st_size for f in files)
            except OSError as e:
                raise PreflightError(f"cannot size model files: {e}")
            tied = bool(cfg.get("tie_word_embeddings", False))
            raw_head = 0 if tied else round(raw_bytes * 0.1)
            raw_dense = raw_bytes - raw_head
            component_src = ("config.json tied embeddings; remaining bytes treated as dense" if tied else
                             "conservative 90% dense / 10% LM-head split")
            weight_status, weight_src = "inferred", ".bin file sizes (includes pickle/container overhead)"
            warnings.append(".bin tensor metadata unavailable; component split and parameter count are estimates")
        elif args.weights_gb is None and args.params_b is None:
            raise PreflightError(f"no safetensors/.bin weights in {path}; supply --weights-gb")
    if args.weights_gb is not None:
        target = round(args.weights_gb * 1e9)
        if raw_bytes:
            scale = target / raw_bytes
            raw_head = min(target, round(raw_head * scale))
            raw_experts = min(target - raw_head, round(raw_experts * scale))
            raw_duplicate = min(target - raw_head - raw_experts, round(raw_duplicate * scale))
            raw_dense = target - raw_head - raw_experts - raw_duplicate
            if tensors:
                component_status, component_src = "inferred", "safetensors component split scaled to CLI --weights-gb"
            else:
                component_status, component_src = "assumed", "estimated component ratios scaled to CLI --weights-gb"
        else:
            raw_head, raw_dense = round(target * 0.1), target - round(target * 0.1)
            component_status, component_src = "assumed", "CLI --weights-gb with conservative 90/10 dense/head split"
        raw_bytes = target
        weight_status, weight_src = "override", "CLI --weights-gb"
    elif not raw_bytes and args.params_b is not None:
        raw_bytes = round(args.params_b * 1e9 * 4 / 8)
        raw_head, raw_dense = round(raw_bytes * 0.1), raw_bytes - round(raw_bytes * 0.1)
        weight_status, weight_src = "assumed", "CLI --params-b at an assumed 4 bits/weight"
        component_status, component_src = "assumed", "conservative 90% dense / 10% LM-head split"
    if raw_bytes <= 0:
        raise PreflightError("supply --model-path with weights, --weights-gb, or --params-b")
    cfg = text_config(config)
    n_experts, top_k, moe = routing(cfg)
    tied = cfg.get("tie_word_embeddings")
    if not tensors and tied is None:
        warnings.append("weight tying unknown; conservatively reserves 10% of weights for a separate LM head")
    if tensors and tied is True and not raw_head:
        warnings.append("tied embeddings: no separate LM-head read is counted")
    params = tensor_parameters(tensors, n_experts, top_k, tied=tied is True)
    estimate = parameter_estimate(cfg, n_experts, top_k, moe)
    if args.params_b is not None:
        total_params = args.params_b * 1e9
        active_fraction = (raw_dense + raw_experts * top_k / n_experts + raw_head) / raw_bytes
        active_params = total_params * active_fraction
        total_param_status = "override"
        active_param_status = "inferred" if component_status == "inferred" else "assumed"
        param_src = "CLI --params-b; active count scaled by active stored-byte fraction"
    elif params:
        total_params, active_params = params
        total_param_status = active_param_status = "inferred"
        param_src = "safetensors shapes and routing fraction"
    elif estimate:
        total_params, active_params = estimate
        total_param_status = active_param_status = "inferred"
        param_src = "config.json architecture estimate (approximate for quantized/hybrid models)"
    else:
        total_params = raw_bytes * 8 / 4
        active_params = total_params * (raw_dense + raw_experts * top_k / n_experts + raw_head) / raw_bytes
        total_param_status = active_param_status = "assumed"
        param_src = "stored bytes at an assumed 4 bits/weight"
        warnings.append("parameter count unavailable; assumes 4 bits/weight")
    details = attention_details(cfg, tensors, warnings)
    n_keys = ("num_experts", "n_routed_experts", "num_local_experts", "n_experts")
    k_keys = ("num_experts_per_tok", "top_k", "topk", "moe_top_k", "num_experts_per_token")
    return {
        "name": name, "path": str(path) if path else None,
        "weight_bytes": item(raw_bytes, weight_status, weight_src),
        "dense_bytes": item(raw_dense, component_status, component_src),
        "experts_bytes": item(raw_experts, component_status, component_src),
        "lm_head_bytes": item(raw_head, component_status, component_src),
        "tied_duplicate_bytes": item(raw_duplicate, component_status,
                                     "stored tied LM-head duplicate; excluded from per-token reads and parameter count"),
        "total_params_b": item(total_params / 1e9, total_param_status, param_src),
        "active_params_b": item(active_params / 1e9, active_param_status, param_src),
        "top_k": item(top_k, "inferred" if any(k in cfg for k in k_keys) else "assumed",
                      "config.json routing" if any(k in cfg for k in k_keys) else "dense default: top-1"),
        "n_experts": item(n_experts, "inferred" if any(k in cfg for k in n_keys) else "assumed",
                          "config.json routing" if any(k in cfg for k in n_keys) else "dense default: 1"),
        "tied_embeddings": tied if isinstance(tied, bool) else "unknown",
        **details,
        "warnings": warnings,
    }


def parse_contexts(values):
    if not values:
        return [4096, 32768], "assumed", "starter defaults"
    contexts = []
    for value in values:
        for part in value.split(","):
            try:
                number = int(part)
            except ValueError:
                raise PreflightError(f"invalid context length: {part!r}")
            if number <= 0:
                raise PreflightError("context lengths must be greater than zero")
            if number > MAX_CONTEXT:
                raise PreflightError(f"context lengths must be at most {MAX_CONTEXT}")
            contexts.append(number)
    return sorted(set(contexts)), "override", "CLI --context"


def override_or_starter(value, starter, source):
    return item(value, "override", source) if value is not None else starter


def build(args):
    path = Path(args.model_path).expanduser().resolve() if args.model_path else None
    name = args.model or (path.name if path else None)
    if not name:
        raise PreflightError("supply --model when --model-path is omitted")
    if path and (not path.exists() or not path.is_dir()):
        raise PreflightError(f"model path is not a directory: {path}")
    if args.model_path is None and args.weights_gb is None and args.params_b is None:
        raise PreflightError("supply --model-path, --weights-gb, or --params-b")
    if args.mfu is not None and args.mfu > 1:
        raise PreflightError("--mfu must be at most 1")
    out = Path(args.out).expanduser().resolve()
    if out.exists() and not out.is_dir():
        raise PreflightError(f"output path is not a directory: {out}")
    for filename in ("ceiling.json", "config.json"):
        if os.path.lexists(out / filename) and not args.force:
            raise PreflightError(f"{out / filename} exists; use --force to replace it")
    if os.path.lexists(out / "DEAD-ENDS.md") and (out / "DEAD-ENDS.md").is_dir():
        raise PreflightError(f"output DEAD-ENDS.md is a directory: {out}")
    contexts, context_status, context_src = parse_contexts(args.contexts)
    hardware = inventory_hardware()
    power = power_status()
    port = choose_port(args.port)
    model = model_summary(args)
    concurrency = sorted({1, args.concurrency})
    bandwidth = override_or_starter(args.bandwidth_gbs, hardware["starter"]["bandwidth_gbs"], "CLI --bandwidth-gbs")
    peak = override_or_starter(args.peak_tflops, hardware["starter"]["peak_tflops"], "CLI --peak-tflops")
    mfu = item(args.mfu if args.mfu is not None else 0.5, "override" if args.mfu is not None else "assumed",
               "CLI --mfu" if args.mfu is not None else "references/ceilings.md practical ~50% starter")
    cap = override_or_starter(args.memory_cap_gib, hardware["starter"]["memory_cap_gib"], "CLI --memory-cap-gib")
    probe = None
    if args.memory_probe:
        try:
            probe = shlex.split(args.memory_probe)
        except ValueError as e:
            raise PreflightError(f"invalid --memory-probe command: {e}")
        if not probe:
            raise PreflightError("--memory-probe must not be empty")
    warnings = hardware["warnings"] + model["warnings"]
    if not probe:
        warnings.append("memory cap is a suggestion only and is not enforced; pass --memory-probe to enable the watchdog")
    if power["status"] == "battery":
        warnings.append("on battery power: files were generated, but bench.py will refuse to run (require_ac)")
    elif power["status"] == "unknown":
        warnings.append("AC power status unknown; bench.py config still sets require_ac: true")
    weight_gib = model["weight_bytes"]["value"] / GIB
    if weight_gib > cap["value"]:
        warnings.append(f"weights ({weight_gib:.1f} GiB) exceed the memory cap ({cap['value']:.1f} GiB)")
    kv_gib = model["kv_bytes_per_token"]["value"] * max(contexts) * max(concurrency) / GIB
    if weight_gib + kv_gib > cap["value"]:
        warnings.append("weights plus KV at the largest context/concurrency exceed the memory cap")
    ceiling = {
        "bandwidth_gbs": bandwidth["value"], "peak_tflops": peak["value"], "mfu": mfu["value"],
        "dense_gb": model["dense_bytes"]["value"] / 1e9,
        "experts_gb": model["experts_bytes"]["value"] / 1e9,
        "top_k": model["top_k"]["value"], "n_experts": model["n_experts"]["value"],
        "lm_head_gb": model["lm_head_bytes"]["value"] / 1e9,
        "active_params_b": model["active_params_b"]["value"],
        "attn_layers": model["attn_layers"]["value"], "attn_heads": model["attn_heads"]["value"],
        "head_dim": model["head_dim"]["value"], "kv_bytes_per_token": model["kv_bytes_per_token"]["value"],
        "draft_layer_gb": 0.0, "draft_head_gb": 0.0,
        "spec": [{"drafts": 0, "accept": 1.0}], "contexts": contexts, "concurrency": concurrency,
        "total_weight_gb": model["weight_bytes"]["value"] / 1e9,
        "total_params_b": model["total_params_b"]["value"], "memory_cap_gib": cap["value"], "model": name,
        "inventory": {
            "hardware": hardware, "power": power, "model": {k: v for k, v in model.items() if k != "path"},
            "port": item(port, "observed", f"TCP bind probe on 127.0.0.1:{port}; no server started"),
            "requested_port": item(args.port, "override", "CLI --port/default 8000"),
            "bandwidth_gbs": bandwidth, "peak_tflops": peak, "mfu": mfu, "memory_cap_gib": cap,
            "memory_probe": item(probe, "override", "CLI --memory-probe") if probe else
                            item(None, "assumed", "not configured; generated memory cap is unenforced"),
            "contexts": item(contexts, context_status, context_src),
            "concurrency": item(concurrency, "override", "CLI --concurrency/default 4 plus c1 control"),
            "spec": item([{"drafts": 0, "accept": 1.0}], "assumed", "no speculative-decoding starter"),
            "draft_layer_gb": item(0.0, "assumed", "speculative decoding disabled"),
            "draft_head_gb": item(0.0, "assumed", "speculative decoding disabled"),
            "bench_temperature": item(0.6, "assumed", "bench.py default starter"),
        },
        "assumptions": ["Provenance: observed = read from system/files/port probe; inferred = calculated; "
                        "assumed = starter; override = user supplied, not measured.",
                        "No benchmark was run; hardware peaks are theoretical/conservative starters unless overridden.",
                        "Hybrid recurrent/SSM state, runtime overhead, and engine-specific KV formats are excluded.",
                        "Speculative decoding starts disabled; draft sizes are zero placeholders."] + warnings,
    }
    config = {
        "name": name, "base_url": f"http://127.0.0.1:{port}", "model": name,
        "concurrency": args.concurrency, "prefill_lengths": contexts, "temperature": 0.6,
        "suggested_memory_cap_gib": cap["value"], "require_ac": True,
    }
    if probe:
        config["memory"] = {"cap_gib": cap["value"], "probe": probe}
    if model["path"]:
        config["model_path"] = model["path"]
    return ceiling, config, out, port, warnings


def write_outputs(out, ceiling, config):
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for filename, data in (("ceiling.json", ceiling), ("config.json", config)):
            with tempfile.NamedTemporaryFile("w", dir=out, prefix=f".{filename}.", delete=False) as f:
                json.dump(data, f, indent=2, allow_nan=False)
                f.write("\n")
                pending.append((Path(f.name), out / filename))
        for temp, target in pending:
            os.replace(temp, target)
        dead_ends = out / "DEAD-ENDS.md"
        if not os.path.lexists(dead_ends):
            with dead_ends.open("x") as f:
                f.write(DEAD_ENDS)
    finally:
        for temp, _ in pending:
            if temp.exists():
                temp.unlink()


def report(ceiling, config, out, port, warnings):
    hw = ceiling["inventory"]["hardware"]["observed"]
    device = hw.get("chip", hw.get("cpu", {})).get("value")
    if not device and hw.get("nvidia_gpus"):
        device = ", ".join(gpu["name"] for gpu in hw["nvidia_gpus"]["value"])
    device = device or "unknown device"
    memory = hw.get("memory_bytes", {}).get("value")
    mem_text = f", {memory / GIB:.1f} GiB" if memory else ""
    print(f"hardware: {device}{mem_text} (read-only inventory)")
    print(f"power: {ceiling['inventory']['power']['status']}")
    print(f"port: {port} free at bind check; requested {ceiling['inventory']['requested_port']['value']}")
    print(f"model: {ceiling['model']}, {ceiling['total_weight_gb']:.3f} GB stored weights")
    print(f"starter ceiling: {ceiling['bandwidth_gbs']:g} GB/s, {ceiling['peak_tflops']:g} TFLOPS, "
          f"MFU {ceiling['mfu']:g} (not benchmark measurements)")
    print(f"wrote: {out / 'ceiling.json'}, {out / 'config.json'}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        ceiling, config, out, port, warnings = build(args)
        write_outputs(out, ceiling, config)
    except (PreflightError, OSError) as e:
        print(f"preflight: {e}", file=sys.stderr)
        return 2
    except OverflowError as e:
        print(f"preflight: numeric value is too large: {e}", file=sys.stderr)
        return 2
    report(ceiling, config, out, port, warnings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
