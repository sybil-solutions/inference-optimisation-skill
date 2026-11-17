# Changelog

## Unreleased

### Fixed
- `ceiling.py`: concurrent decode ceilings were multiplied by N twice (4 users showed ~4x too high, e.g. 567 instead of ~142 tok/s for the B390 example). Draft weights are now counted once per engine step instead of once per stream.
- `mem.sh`: DRM fdinfo sizes are printed by the kernel in B, KiB or MiB; they were all read as KiB, so MiB-aligned allocations were undercounted 1024x and the memory cap could never trip. cgroup v1 (`rss`) is now supported.
- `queue.sh`: `ctl cand ctl` overwrote the first control's log and both controls had the same ledger name.

### Added
- `bench.py compare`: mechanical accept rule (beat both neighbouring controls, no regression > 3%, bootstrap 95% CI) with ACCEPT / REPEAT / REJECT / VOID verdicts.
- `bench.py soak`: N concurrent full-length needle requests (full-KV stability gate).
- `bench.py greedy` / `equiv`: greedy-output capture and divergence report for lossless claims.
- Environment guard: `require_profile`, `max_load`; environment snapshot at start and end; `ENV_CHANGED` voids a run.
- New flags: `NO_USAGE`, `NEEDLE_MISS`, `NOT_CONCURRENT` (with `cN_overlap`), `GPU_IDLE` (optional `gpu_busy_probe`), `SHORT_OUTPUT`; loop detection on every request, not only c1.
- Ledger: unique `run_id`, `group`/`pos`, `config_hash`, prompt-panel id, temperature and request extras.
- `ceiling.py`: input validation, `--json`, `--target`, recurrent-state bytes, "streams that fit" under a memory cap.
- `mem.sh --pid` for bare-metal servers.
- Tests (unit + end-to-end against a fake OpenAI-compatible server) and a GitHub Actions workflow with shellcheck.
