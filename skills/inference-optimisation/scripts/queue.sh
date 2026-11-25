#!/usr/bin/env bash
# Serialized GPU queue: runs bench.py on each config in order, one GPU job at a time across all callers.
#   queue.sh ctl-a cand ctl-b            # interleave controls around each candidate
#   BENCH_FLAGS="--long 131072" queue.sh cand
# Run long queues as a harness background task or a service-manager unit, so they survive agent restarts
# and stay visible and stoppable (never nohup/setsid):
#   systemd-run --user --unit=hillclimb --same-dir --collect bash queue.sh ctl-a cand ctl-b   # journalctl --user -u hillclimb
# Run from your experiment directory (the one holding configs/ and ledger.jsonl).
# Exits 1 if any run failed (a server that never came up included); later runs still go, so controls stay paired.
LOCK=${QUEUE_LOCK:-/tmp/inference-hillclimb.lock}
exec 9>"$LOCK"; flock 9
rc=0
for n in "$@"; do
  if python3 "${BENCH:-$(dirname "$0")/bench.py}" run "configs/$n.json" ${BENCH_FLAGS:-} > "runs-$n.log" 2>&1; then s=done; else s=FAILED; rc=1; fi
  echo "$s $n: $(tail -c 300 "runs-$n.log" | tr '\n' ' ' | cut -c1-200)"
done
echo QUEUE_DONE "$@"
exit $rc
