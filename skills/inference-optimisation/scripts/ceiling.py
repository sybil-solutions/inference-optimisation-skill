#!/usr/bin/env python3
"""Theoretical ceilings for decode / prefill / KV from a small JSON of measured hardware and model sizes.

  python3 ceiling.py model.json                 # human-readable tables
  python3 ceiling.py model.json --json          # machine-readable (for THEORY.md / scripts)
  python3 ceiling.py model.json --target 100    # is 100 tok/s single-user decode physically reachable?

Example model.json (Ornith-1.5-35B-A3B EXL3 4bpw on an Arc B390 iGPU; sizes in GB from the checkpoint):
{
  "bandwidth_gbs": 115, "peak_tflops": 36, "mfu": 0.5,
  "dense_gb": 0.758, "experts_gb": 16.26, "top_k": 8, "n_experts": 256, "lm_head_gb": 0.382,
  "active_params_b": 2.43, "attn_layers": 10, "attn_heads": 16, "head_dim": 256,
  "kv_bytes_per_token": 10240, "draft_layer_gb": 0.03, "draft_head_gb": 0.05,
  "spec": [{"drafts": 0, "accept": 1.0}, {"drafts": 1, "accept": 1.74}, {"drafts": 2, "accept": 2.5}],
  "contexts": [4096, 32768, 131072, 262144], "concurrency": [1, 4],
  "memory_cap_gb": 49.4, "resident_gb": 38.0
}

Field notes
  dense_gb         every byte read per target forward that is NOT a routed expert or the LM head.
  experts_gb       total routed-expert bytes (all experts); 0 or absent for dense models.
  draft_head_gb    the draft LM head as read per draft step (prune it!). Defaults to lm_head_gb.
  accept           accepted length per stream per verify step (tokens emitted per step, >= 1).
  kv_per_verify_row  true if the engine re-reads KV once per verify row (prefill-style verify).
  recurrent_state_gb  SSM/GDN state per stream (read + write each step is counted as 2x).
  memory_cap_gb, resident_gb  optional: cap and everything that is not KV (weights, state, runtime);
                   enables the "full-length streams that fit" table.

Model (aggregate over N streams, one engine step):
  verify = dense + lm_head + experts * P(expert used by any of (drafts+1)*N rows)
           + N * (kv_bytes_per_token * ctx [* (drafts+1)] + 2 * recurrent_state)
  draft  = drafts * (draft_layer + draft_head)          # weights are read once per step for all streams
  tok/s  = bandwidth * accept * N / (verify + draft)
"""
import json, sys

REQUIRED_DECODE = ("bandwidth_gbs", "dense_gb", "lm_head_gb")
REQUIRED_PREFILL = ("peak_tflops", "mfu", "active_params_b")


class ConfigError(ValueError):
    pass


def validate(m):
    missing = [k for k in REQUIRED_DECODE if k not in m]
    if missing:
        raise ConfigError(f"missing required field(s): {', '.join(missing)}")
    if m.get("experts_gb"):
        for k in ("top_k", "n_experts"):
            if not m.get(k):
                raise ConfigError(f"experts_gb is set, so {k} is required")
        if m["top_k"] > m["n_experts"]:
            raise ConfigError("top_k cannot exceed n_experts")
    if m["bandwidth_gbs"] <= 0:
        raise ConfigError("bandwidth_gbs must be > 0")
    if not 0 < m.get("mfu", 0.5) <= 1:
        raise ConfigError("mfu must be in (0, 1]")
    for s in m.get("spec", []):
        if s.get("drafts", 0) < 0:
            raise ConfigError("drafts must be >= 0")
        if not 1 <= s.get("accept", 1.0) <= s.get("drafts", 0) + 1:
            raise ConfigError(f"accept must be between 1 and drafts+1, got {s}")
    return m


def expert_union(m, rows):
    """Fraction of routed experts touched by `rows` independent top-k routings (uniform-routing assumption)."""
    if not m.get("experts_gb"):
        return 0.0
    return 1 - (1 - m["top_k"] / m["n_experts"]) ** rows


def step_bytes_gb(m, drafts, ctx=0, n=1):
    """GB read from memory per engine step serving `n` streams with `drafts` speculative tokens each."""
    rows = (drafts + 1) * n
    verify = m["dense_gb"] + m["lm_head_gb"] + m.get("experts_gb", 0) * expert_union(m, rows)
    kv_reads = drafts + 1 if m.get("kv_per_verify_row") else 1
    verify += n * (m.get("kv_bytes_per_token", 0) * ctx * kv_reads / 1e9 + 2 * m.get("recurrent_state_gb", 0))
    draft = drafts * (m.get("draft_layer_gb", 0) + m.get("draft_head_gb", m["lm_head_gb"]))
    return verify + draft


def decode_ceiling(m, drafts, accept, ctx=0, n=1):
    """Aggregate decode tok/s across `n` streams (divide by n for per-stream)."""
    return m["bandwidth_gbs"] * accept * n / step_bytes_gb(m, drafts, ctx, n)


def prefill_ceiling(m, L):
    linear = 2 * m["active_params_b"] * 1e9
    attn = 4 * m.get("attn_layers", 0) * m.get("attn_heads", 0) * m.get("head_dim", 0) * (L / 2)
    return m["mfu"] * m["peak_tflops"] * 1e12 / (linear + attn + m.get("extra_flops_per_token", 0))


def kv_fit(m, ctx):
    """Full-length streams of `ctx` tokens that fit in memory_cap_gb - resident_gb."""
    if not (m.get("kv_bytes_per_token") and m.get("memory_cap_gb") is not None and m.get("resident_gb") is not None):
        return None
    free = (m["memory_cap_gb"] - m["resident_gb"]) * 1e9
    return max(0, int(free // (m["kv_bytes_per_token"] * ctx)))


def compute(m):
    validate(m)
    ctxs = m.get("contexts", [0])
    specs = m.get("spec", [{"drafts": 0, "accept": 1.0}])
    out = {"decode": [], "prefill": {}, "kv": {}}
    for n in m.get("concurrency", [1]):
        for s in specs:
            for c in ctxs:
                agg = decode_ceiling(m, s["drafts"], s["accept"], c, n)
                out["decode"].append({"concurrency": n, "drafts": s["drafts"], "accept": s["accept"], "context": c,
                                      "aggregate_tok_s": round(agg, 1), "per_stream_tok_s": round(agg / n, 1),
                                      "gb_per_token": round(step_bytes_gb(m, s["drafts"], c, n) / (s["accept"] * n), 4)})
    if all(k in m for k in REQUIRED_PREFILL):
        out["prefill"] = {str(L): round(prefill_ceiling(m, L), 1) for L in ctxs if L}
    if m.get("kv_bytes_per_token"):
        for c in ctxs:
            if c:
                out["kv"][str(c)] = {"gb_per_stream": round(m["kv_bytes_per_token"] * c / 1e9, 3), "streams_fit": kv_fit(m, c)}
    return out


def target_verdict(m, res, target):
    """Best single-user decode ceiling at the shortest context vs a user target."""
    ctx = min(m.get("contexts", [0]))
    rows = [r for r in res["decode"] if r["concurrency"] == 1 and r["context"] == ctx]
    best = max(rows, key=lambda r: r["aggregate_tok_s"])
    ok = best["aggregate_tok_s"] >= target
    return {"target": target, "best_ceiling": best["aggregate_tok_s"], "drafts": best["drafts"], "accept": best["accept"],
            "context": ctx, "reachable": ok,
            "note": ("reachable in theory; practical results usually land at 50-80% of the ceiling" if ok else
                     "ABOVE the ceiling: not reachable with these sizes/bandwidth/acceptance")}


def print_tables(m, res):
    print(f"bandwidth {m['bandwidth_gbs']} GB/s, peak {m.get('peak_tflops', '?')} TFLOPS, MFU {m.get('mfu', 0.5)}")
    ctxs = m.get("contexts", [0])
    for n in m.get("concurrency", [1]):
        print(f"\nDecode ceiling, concurrency {n} (aggregate tok/s; per-stream = aggregate / {n}):")
        print("  drafts accept" + "".join(f"{c // 1024:>8}K" for c in ctxs))
        for s in m.get("spec", [{"drafts": 0, "accept": 1.0}]):
            vals = [r["aggregate_tok_s"] for r in res["decode"]
                    if r["concurrency"] == n and r["drafts"] == s["drafts"] and r["accept"] == s["accept"]]
            print(f"  {s['drafts']:>6} {s['accept']:>6}" + "".join(f"{v:9.0f}" for v in vals))
    if res["prefill"]:
        print("\nPrefill ceiling (tok/s, practical):")
        for L, v in res["prefill"].items():
            print(f"  {int(L) // 1024:>4}K {v:8.0f}")
    if res["kv"]:
        print("\nKV cache:")
        for c, v in res["kv"].items():
            fit = "" if v["streams_fit"] is None else f", {v['streams_fit']} full-length stream(s) fit under the cap"
            print(f"  {int(c) // 1024:>4}K: {v['gb_per_stream']:.2f} GB per stream{fit}")


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        sys.exit(__doc__)
    m = json.load(open(argv[0]))
    try:
        res = compute(m)
    except ConfigError as e:
        sys.exit(f"ceiling.py: {e}")
    if "--target" in argv:
        res["target"] = target_verdict(m, res, float(argv[argv.index("--target") + 1]))
    if "--json" in argv:
        print(json.dumps(res, indent=1))
        return
    print_tables(m, res)
    if "target" in res:
        t = res["target"]
        print(f"\nTarget {t['target']:g} tok/s c1 at {t['context'] // 1024}K: best ceiling {t['best_ceiling']:.0f} "
              f"({t['drafts']} drafts, accept {t['accept']}) -> {t['note']}")


if __name__ == "__main__":
    main(sys.argv[1:])
