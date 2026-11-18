#!/usr/bin/env bash
# Regression test for queue.sh's lock: two concurrent queues must not overlap, a dead
# holder's lock must be taken over, and an interrupted queue must release it.
# Issue #1: flock(1) is missing on macOS, so the fallback path runs there. Run from anywhere:
#   bash tests/test_queue_lock.sh
set -u
cd "$(dirname "$0")/.."
Q="$(pwd)/skills/inference-optimisation/scripts/queue.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export QUEUE_LOCK="$TMP/queue.lock"
export TRACE="$TMP/trace"
export BENCH="$TMP/bench.py"

mkdir -p "$TMP/configs"
echo '{}' > "$TMP/configs/one.json"
echo '{}' > "$TMP/configs/two.json"
# A stand-in bench that appends its start and end to a shared trace, slow enough to
# overlap visibly if the lock does not serialize.
cat > "$TMP/bench.py" <<'PY'
import os, sys, time
with open(os.environ["TRACE"], "a") as f:
    f.write(f"start {sys.argv[2]} {time.time()}\n")
    time.sleep(2)
    f.write(f"end {sys.argv[2]} {time.time()}\n")
PY

fail=0
overlap() {
  python3 - "$TRACE" <<'PY'
import sys
prev_end = 0.0
for line in open(sys.argv[1]):
    parts = line.split()
    if parts[0] == "start" and float(parts[2]) < prev_end:
        sys.exit("FAIL: two benches overlapped - the queue did not serialize")
    if parts[0] == "end":
        prev_end = float(parts[2])
PY
}

echo "1. two concurrent queues serialize"
(cd "$TMP" && bash "$Q" one > q1.log 2>&1) &
p1=$!
(cd "$TMP" && bash "$Q" two > q2.log 2>&1) &
p2=$!
wait $p1; wait $p2
overlap || fail=1
grep -q "QUEUE_DONE" "$TMP/q1.log" "$TMP/q2.log" || { echo "FAIL: a queue did not finish"; fail=1; }

if ! command -v flock >/dev/null 2>&1; then
  echo "2. a dead holder's directory lock is taken over"
  mkdir "$QUEUE_LOCK.d" && echo 99999999 > "$QUEUE_LOCK.d/pid"
  (cd "$TMP" && bash "$Q" one > q3.log 2>&1)
  grep -q "QUEUE_DONE" "$TMP/q3.log" || { echo "FAIL: the stale lock wedged the queue"; fail=1; }

  echo "3. an interrupted queue releases the lock"
  (cd "$TMP" && bash "$Q" one > q4.log 2>&1) &
  p4=$!
  sleep 1
  kill -INT $p4
  wait $p4 2>/dev/null
  for i in $(seq 1 10); do [ -d "$QUEUE_LOCK.d" ] || break; sleep 0.5; done
  [ -d "$QUEUE_LOCK.d" ] && { echo "FAIL: the lock survived SIGINT"; fail=1; }
fi

[ "$fail" -eq 0 ] && echo "ALL LOCK TESTS PASSED" || exit 1
