#!/usr/bin/env bash
# Regression tests for queue.sh's portable lock. Run from anywhere: bash tests/test_queue_lock.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
Q="$ROOT/skills/inference-optimisation/scripts/queue.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/inference-queue-test.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
WORK="$TMP/work"
mkdir -p "$WORK/configs" "$TMP/bin"
printf '{}\n' > "$WORK/configs/one.json"
printf '{}\n' > "$WORK/configs/two.json"
printf '{}\n' > "$WORK/configs/fail.json"

# A restricted PATH proves locking does not depend on the flock(1) executable.
for cmd in bash cat cut dirname kill mkdir python3 rm rmdir sed sleep tail tr; do
  ln -s "$(command -v "$cmd")" "$TMP/bin/$cmd"
done
cat > "$TMP/bench.py" <<'PY'
import os, sys, time
with open(os.environ["TRACE"], "a") as f:
    f.write(f"start {sys.argv[1]} {sys.argv[2]} {time.time()} fd={os.environ.get('QUEUE_LOCK_FD', '-')}\n")
    f.flush()
if os.environ.get("FAIL_CONFIG") == sys.argv[2]:
    sys.exit(7)
time.sleep(0.4)
with open(os.environ["TRACE"], "a") as f:
    f.write(f"end {sys.argv[1]} {sys.argv[2]} {time.time()}\n")
PY

cat > "$TMP/spoof.py" <<'PY'
import fcntl, os, sys
unrelated, canonical, queue, work, trace, bench = sys.argv[1:]
fd = os.open(unrelated, os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
os.set_inheritable(fd, True)
env = dict(os.environ, QUEUE_LOCK=canonical, QUEUE_LOCK_FD=str(fd), QUEUE_LOCK_PATH=unrelated,
           TRACE=trace, BENCH=bench, BENCH_MODE="run")
os.chdir(work)
os.execvpe("bash", ["bash", queue, "one"], env)
PY

export TRACE="$TMP/trace"
export BENCH="$TMP/bench.py"
export BENCH_MODE=screen
fail=0

check_overlap() {
  python3 - "$TRACE" <<'PY'
import sys
end = 0.0
for line in open(sys.argv[1]):
    parts = line.split()
    kind, stamp = parts[0], float(parts[3])
    if kind == "start" and stamp < end:
        sys.exit("FAIL: two benches overlapped; queue did not serialize")
    if kind == "end":
        end = stamp
PY
}

run_fallback() {
  (cd "$WORK" && exec /usr/bin/env PATH="$TMP/bin" QUEUE_LOCK="$TMP/lock" TRACE="$TRACE" BENCH="$BENCH" \
    BENCH_MODE="$BENCH_MODE" /bin/bash "$Q" "$@" > "$TMP/$1.log" 2>&1)
}

echo "1. portable locking serializes concurrent queues and forwards screen mode"
run_fallback one two &
p1=$!
run_fallback two one &
p2=$!
wait "$p1" || fail=1
wait "$p2" || fail=1
check_overlap || fail=1
grep -q '^start screen configs/one.json ' "$TRACE" || { echo "FAIL: BENCH_MODE was not forwarded"; fail=1; }
grep -q QUEUE_DONE "$TMP/one.log" "$TMP/two.log" || { echo "FAIL: a queue did not finish"; fail=1; }
if grep -q ' fd=[0-9]' "$TRACE"; then echo "FAIL: internal lock descriptor leaked into benchmarks"; fail=1; fi

echo "2. a spoofed lock descriptor cannot bypass serialization"
: > "$TRACE"
(cd "$WORK" && exec /usr/bin/env PATH="$TMP/bin" QUEUE_LOCK="$TMP/spoof.lock" QUEUE_LOCK_FD=999 \
  QUEUE_LOCK_PATH="$TMP/spoof.lock" TRACE="$TRACE" BENCH="$BENCH" BENCH_MODE=run \
  /bin/bash "$Q" one > "$TMP/spoof1.log" 2>&1) &
p6=$!
(cd "$WORK" && exec /usr/bin/env PATH="$TMP/bin" QUEUE_LOCK="$TMP/spoof.lock" QUEUE_LOCK_FD=999 \
  QUEUE_LOCK_PATH="$TMP/spoof.lock" TRACE="$TRACE" BENCH="$BENCH" BENCH_MODE=run \
  /bin/bash "$Q" two > "$TMP/spoof2.log" 2>&1) &
p7=$!
wait "$p6" 2>/dev/null || true
wait "$p7" 2>/dev/null || true
[ ! -s "$TRACE" ] || { echo "FAIL: spoofed descriptor allowed benchmarks to start"; fail=1; }

echo "3. valid unrelated descriptors cannot bypass the canonical lock"
: > "$TRACE"
python3 "$TMP/spoof.py" "$TMP/unrelated0.lock" "$TMP/canonical.lock" "$Q" "$WORK" "$TRACE" "$BENCH" \
  > "$TMP/valid-spoof1.log" 2>&1 &
p8=$!
python3 "$TMP/spoof.py" "$TMP/unrelated1.lock" "$TMP/canonical.lock" "$Q" "$WORK" "$TRACE" "$BENCH" \
  > "$TMP/valid-spoof2.log" 2>&1 &
p9=$!
wait "$p8" 2>/dev/null || true
wait "$p9" 2>/dev/null || true
[ ! -s "$TRACE" ] || { echo "FAIL: unrelated locked descriptors bypassed the canonical lock"; fail=1; }

echo "4. a failed bench makes the queue fail"
(cd "$WORK" && exec /usr/bin/env PATH="$TMP/bin" QUEUE_LOCK="$TMP/fail.lock" TRACE="$TRACE" BENCH="$BENCH" \
  BENCH_MODE=screen FAIL_CONFIG=configs/fail.json /bin/bash "$Q" fail > "$TMP/fail.log" 2>&1)
status=$?
[ "$status" -ne 0 ] || { echo "FAIL: queue swallowed bench exit status"; fail=1; }
grep -q 'VOID' "$TMP/fail.log" || { echo "FAIL: failed bench was printed as done"; fail=1; }

echo "5. portable lock reuses stale files and rejects symlinks"
printf '99999999\n' > "$TMP/lock"
run_fallback one || fail=1
grep -q QUEUE_DONE "$TMP/one.log" || { echo "FAIL: stale lock content wedged the queue"; fail=1; }
: > "$TMP/lock"
run_fallback one || fail=1
grep -q QUEUE_DONE "$TMP/one.log" || { echo "FAIL: empty lock file wedged the queue"; fail=1; }
printf 'do not truncate\n' > "$TMP/victim"
ln -s "$TMP/victim" "$TMP/symlink.lock"
(cd "$WORK" && QUEUE_LOCK="$TMP/symlink.lock" TRACE="$TRACE" BENCH="$BENCH" BENCH_MODE=run \
  bash "$Q" one > "$TMP/symlink.log" 2>&1)
status=$?
[ "$status" -ne 0 ] || { echo "FAIL: queue accepted a symlink lock"; fail=1; }
[ "$(cat "$TMP/victim")" = "do not truncate" ] || { echo "FAIL: queue truncated the symlink target"; fail=1; }

echo "6. portable lock is released after SIGTERM"
: > "$TRACE"
(cd "$WORK" && exec /usr/bin/env PATH="$TMP/bin" QUEUE_LOCK="$TMP/lock" TRACE="$TRACE" BENCH="$BENCH" \
  BENCH_MODE="$BENCH_MODE" /bin/bash "$Q" one two > "$TMP/signal.log" 2>&1) &
p3=$!
for ((i=0; i<50; i++)); do grep -q '^start ' "$TRACE" 2>/dev/null && break; sleep 0.05; done
kill -TERM "$p3" 2>/dev/null || true
wait "$p3" 2>/dev/null || true
run_fallback one || fail=1
grep -q QUEUE_DONE "$TMP/one.log" || { echo "FAIL: SIGTERM left the lock held"; fail=1; }

echo "7. normal environment serializes without the flock executable"
: > "$TRACE"
(cd "$WORK" && QUEUE_LOCK="$TMP/flock.lock" TRACE="$TRACE" BENCH="$BENCH" BENCH_MODE=run \
  bash "$Q" one > "$TMP/flock1.log" 2>&1) &
p4=$!
(cd "$WORK" && QUEUE_LOCK="$TMP/flock.lock" TRACE="$TRACE" BENCH="$BENCH" BENCH_MODE=run \
  bash "$Q" two > "$TMP/flock2.log" 2>&1) &
p5=$!
wait "$p4" || fail=1
wait "$p5" || fail=1
check_overlap || fail=1
grep -q '^start run configs/one.json ' "$TRACE" || { echo "FAIL: run mode was not forwarded"; fail=1; }

echo "8. default lock lives in a private per-user runtime directory"
mkdir -p "$TMP/runtime"
(cd "$WORK" && exec /usr/bin/env PATH="$TMP/bin" QUEUE_LOCK="" XDG_RUNTIME_DIR="" TMPDIR="$TMP/runtime" TRACE="$TRACE" BENCH="$BENCH" \
  BENCH_MODE=run /bin/bash "$Q" one > "$TMP/private.log" 2>&1) || fail=1
python3 - "$TMP/runtime" <<'PY'
import os, stat, sys
base = sys.argv[1]
dirs = [os.path.join(base, n) for n in os.listdir(base)
        if n.startswith("inference-hillclimb-") and os.path.isdir(os.path.join(base, n))]
assert len(dirs) == 1, dirs
st = os.stat(dirs[0], follow_symlinks=False)
assert stat.S_IMODE(st.st_mode) == 0o700, oct(stat.S_IMODE(st.st_mode))
lock = os.path.join(dirs[0], "queue.lock")
assert os.path.isfile(lock) and not os.path.islink(lock)
PY
[ "$?" -eq 0 ] || { echo "FAIL: default lock directory is not private"; fail=1; }

[ "$fail" -eq 0 ] && echo "ALL QUEUE LOCK TESTS PASSED" || exit 1
