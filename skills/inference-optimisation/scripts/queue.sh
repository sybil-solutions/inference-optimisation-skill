#!/usr/bin/env bash
# Serialized GPU queue: runs bench.py on each config in order, one GPU job at a time across all callers.
#   queue.sh ctl-a cand ctl-b            # interleave controls around each candidate
#   BENCH_FLAGS="--long 131072" queue.sh cand
# Detach long queues so they survive agent/terminal restarts:
#   nohup bash queue.sh ctl-a cand ctl-b > queue.log 2>&1 < /dev/null & disown
# Run from your experiment directory (the one holding configs/ and ledger.jsonl).
LOCK=${QUEUE_LOCK:-/tmp/inference-hillclimb.lock}
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK"; flock 9
else
  # flock(1) is Linux-only (it is missing on macOS); a directory lock is atomic on both.
  until mkdir "$LOCK.d" 2>/dev/null; do
    pid=$(cat "$LOCK.d/pid" 2>/dev/null)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      rm -rf "$LOCK.d"; continue    # the holder died without cleaning up: take over
    fi
    sleep 5
  done
  echo $$ > "$LOCK.d/pid"
  trap 'rm -rf "$LOCK.d"; exit' INT TERM
  trap 'rm -rf "$LOCK.d"' EXIT
fi
for n in "$@"; do
  python3 "${BENCH:-$(dirname "$0")/bench.py}" run "configs/$n.json" ${BENCH_FLAGS:-} > "runs-$n.log" 2>&1
  echo "done $n: $(tail -c 300 "runs-$n.log" | tr '\n' ' ' | cut -c1-200)"
done
echo QUEUE_DONE "$@"
