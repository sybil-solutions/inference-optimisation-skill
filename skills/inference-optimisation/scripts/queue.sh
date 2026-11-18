#!/usr/bin/env bash
# Serialized GPU queue: runs bench.py on each config in order, one GPU job at a time across all callers.
#   queue.sh ctl cand ctl                # interleave controls around each candidate
#   queue.sh ctl candA ctl candB ctl     # several candidates share neighbouring controls
#   BENCH_FLAGS="--long 131072" queue.sh cand
# Every invocation is one "group"; each run gets a unique run_id (name@group#pos), its own log in runs/,
# and its position, so `bench.py compare --control ctl` can judge each candidate against its neighbours.
# Detach long queues so they survive agent/terminal restarts:
#   setsid nohup bash queue.sh ctl cand ctl > queue.log 2>&1 < /dev/null & disown
# Run from your experiment directory (the one holding configs/ and ledger.jsonl).
set -uo pipefail
[ $# -ge 1 ] || { sed -n '2,4p' "$0" >&2; exit 2; }

for n in "$@"; do
  [ -f "configs/$n.json" ] || { echo "queue.sh: configs/$n.json not found (run from the experiment directory)" >&2; exit 2; }
done

BENCH=${BENCH:-$(dirname "$0")/bench.py}
LOCK=${QUEUE_LOCK:-/tmp/inference-hillclimb.lock}
GROUP=${QUEUE_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)}
read -r -a flags <<<"${BENCH_FLAGS:-}"
mkdir -p runs

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "queue.sh: waiting for another GPU job to finish (lock $LOCK)" >&2
  flock 9
fi
echo "group $GROUP: $*"

pos=0 failed=0
for n in "$@"; do
  pos=$((pos + 1))
  id="$n@$GROUP#$pos"
  logf="runs/${GROUP}-${pos}-${n}.log"
  if python3 "$BENCH" run "configs/$n.json" --run-id "$id" --group "$GROUP" --pos "$pos" "${flags[@]}" >"$logf" 2>&1; then
    status=ok
  else
    status=VOID; failed=$((failed + 1))
  fi
  echo "done $id [$status]: $(grep -E '"(c1_decode|flags)"' "$logf" | tr -d ' \n' | cut -c1-160)"
done
echo "QUEUE_DONE group=$GROUP failed=$failed"
echo "next: python3 $BENCH compare --control $1 --group $GROUP"
