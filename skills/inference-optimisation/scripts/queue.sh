#!/usr/bin/env bash
# Serialized GPU queue: runs bench.py on each config in order, one GPU job at a time across all callers.
#   queue.sh ctl-a cand ctl-b            # interleave controls around each candidate
#   BENCH_FLAGS="--long 131072" queue.sh cand
#   QUEUE_LOCK="$HOME/gpu.lock" queue.sh ctl-a cand ctl-b  # use the host's agreed lock
# Detach long queues so they survive agent/terminal restarts:
#   setsid nohup bash queue.sh ctl-a cand ctl-b > queue.log 2>&1 < /dev/null & disown
# Run from your experiment directory (the one holding configs/ and ledger.jsonl).
lock=${QUEUE_LOCK:-/tmp/inference-hillclimb.lock}
exec 9>"$lock" || exit 1
flock -w "${QUEUE_LOCK_TIMEOUT_S:-7200}" 9 || exit 1
status=0
for n in "$@"; do
  if python3 "${BENCH:-$(dirname "$0")/bench.py}" run "configs/$n.json" ${BENCH_FLAGS:-} > "runs-$n.log" 2>&1; then
    echo "passed $n"
  else
    echo "failed $n (exit $?). See runs-$n.log" >&2
    status=1
  fi
done
exit "$status"
