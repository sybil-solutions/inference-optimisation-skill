#!/usr/bin/env python3
"""A/A noise floor: how much does the SAME config move from run to run, and how often would the
'beats both neighbouring controls' rule keep a change that does nothing? Standard library only.

  for i in $(seq 12); do python3 bench.py run configs/ctl.json --skip-cN; done    # same config, back to back
  python3 aa.py [ledger.jsonl] [--name ctl] [--metric c1_decode]

--name keeps the ledger rows whose name starts with that prefix (default: every row). Rows are taken in
ledger order, so run the series back to back with nothing else in between. Use at least 10 runs.

With nothing changed, the middle run of any three is the fastest about one time in three, so 'beats both
controls' on its own keeps about a third of useless changes. The floor printed here is the largest win a
no-change run showed over its better neighbour: require a candidate to beat its better control by more.
"""
import json, math, sys


def quantile(v, p):
    s = sorted(v); k = (len(s) - 1) * p; f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def analyse(vals):
    """vals: one metric from consecutive runs of an unchanged config."""
    n = len(vals)
    med = quantile(vals, 0.5)
    mid = range(1, n - 1)
    wins = [i for i in mid if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]]
    over = [vals[i] / max(vals[i - 1], vals[i + 1]) - 1 for i in mid]
    return {"runs": n, "median": med, "min": min(vals), "max": max(vals),
            "spread_p10_p90_pct": 100 * (quantile(vals, 0.9) - quantile(vals, 0.1)) / med,
            "triplets": len(mid), "false_wins": len(wins), "false_win_runs": [i + 1 for i in wins],
            "floor_pct": 100 * max(over) if over else None}


def main(argv):
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    flagged = {i for k in ("--name", "--metric") if k in argv for i in (argv.index(k), argv.index(k) + 1)}
    pos = [a for i, a in enumerate(argv) if i not in flagged]
    path, name, metric = (pos[0] if pos else "ledger.jsonl"), opt("--name", ""), opt("--metric", "c1_decode")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("name", "").startswith(name) and r.get("ok") and r.get(metric) is not None]
    if len(rows) < 5:
        sys.exit(f"need at least 5 ok runs with {metric} (found {len(rows)}); 10 or more is better")
    a = analyse([r[metric] for r in rows])
    print(f"{a['runs']} runs of the same config, {metric}: median {a['median']:.2f} "
          f"(min {a['min']:.2f}, max {a['max']:.2f}, p10-p90 spread {a['spread_p10_p90_pct']:.1f}%)")
    print(f"no-change runs that beat both neighbours: {a['false_wins']} of {a['triplets']} "
          f"({100 * a['false_wins'] / a['triplets']:.0f}%; chance alone gives about 33%)")
    print(f"noise floor (largest no-change win over the better neighbour): {a['floor_pct']:.2f}%")
    print(f"=> accept a candidate only if it beats its better control by more than {math.ceil(a['floor_pct'] * 10) / 10:.1f}%, "
          f"and repeat any win that is close to that")
    return a


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        sys.exit(__doc__)
    main(sys.argv[1:])
