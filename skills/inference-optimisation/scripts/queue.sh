#!/usr/bin/env bash
# Serialized GPU queue: runs bench.py on each config in order, one GPU job at a time across all callers.
#   queue.sh ctl-a cand ctl-b            # interleave controls around each candidate
#   BENCH_FLAGS="--long 131072" queue.sh cand
#   BENCH_MODE=screen queue.sh cand     # ~90-second onboarding screen before a full bake-off
# Detach long queues so they survive agent/terminal restarts:
#   nohup bash queue.sh ctl-a cand ctl-b > queue.log 2>&1 < /dev/null & disown
# Run from your experiment directory (the one holding configs/ and ledger.jsonl).
set -o pipefail

# Python's fcntl.flock is present on both Linux and macOS, unlike the flock(1) executable.
# Acquire it before exec'ing the real queue so the same process holds the lock for its whole lifetime.
if [ -z "${QUEUE_LOCK_FD:-}" ]; then
  exec python3 - "$0" "$@" <<'PY'
import fcntl, os, stat, sys, tempfile

script, args = os.path.abspath(sys.argv[1]), sys.argv[2:]
uid = os.getuid()
custom = os.environ.get("QUEUE_LOCK")
if custom:
    path = os.path.abspath(custom)
else:
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(tempfile.gettempdir(), f"inference-hillclimb-{uid}")
    try:
        os.makedirs(base, mode=0o700, exist_ok=True)
        st = os.stat(base, follow_symlinks=False)
    except OSError as e:
        sys.exit(f"queue.sh: cannot create private lock directory {base}: {e}")
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != uid or stat.S_IMODE(st.st_mode) & 0o077:
        sys.exit(f"queue.sh: lock directory is not private and owned by this user: {base}")
    path = os.path.join(base, "queue.lock")

flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
try:
    fd = os.open(path, flags, 0o600)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != uid:
        raise OSError("lock must be a regular file owned by this user")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"queue.sh: waiting for another GPU job to finish (lock {path})", file=sys.stderr, flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
except OSError as e:
    sys.exit(f"queue.sh: cannot acquire lock {path}: {e}")

os.set_inheritable(fd, True)
env = dict(os.environ, QUEUE_LOCK_FD=str(fd))
os.execvpe("bash", ["bash", script, *args], env)
PY
fi

# Validate the inherited descriptor against the canonical path, not a caller-supplied path.
if ! python3 - "$QUEUE_LOCK_FD" <<'PY'
import fcntl, os, stat, sys, tempfile
try:
    fd = int(sys.argv[1])
    uid = os.getuid()
    custom = os.environ.get("QUEUE_LOCK")
    if custom:
        expected = os.path.abspath(custom)
    else:
        base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(tempfile.gettempdir(), f"inference-hillclimb-{uid}")
        expected = os.path.join(base, "queue.lock")
    st, pst = os.fstat(fd), os.stat(expected, follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != uid or (st.st_dev, st.st_ino) != (pst.st_dev, pst.st_ino):
        raise OSError("descriptor does not refer to the canonical user-owned lock file")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except (OSError, ValueError) as e:
    sys.exit(f"queue.sh: invalid inherited lock descriptor: {e}")
PY
then
  exit 1
fi
unset QUEUE_LOCK_FD

[ $# -ge 1 ] || { sed -n '2,5p' "$0" >&2; exit 2; }

BENCH=${BENCH:-$(dirname "$0")/bench.py}
BENCH_MODE=${BENCH_MODE:-run}
read -r -a flags <<<"${BENCH_FLAGS:-}"

for n in "$@"; do
  [ -f "configs/$n.json" ] || { echo "queue.sh: configs/$n.json not found (run from the experiment directory)" >&2; exit 2; }
done

failed=0
for n in "$@"; do
  if python3 "$BENCH" "$BENCH_MODE" "configs/$n.json" "${flags[@]}" > "runs-$n.log" 2>&1; then
    status=done
  else
    status=VOID
    failed=$((failed + 1))
  fi
  echo "$status $n: $(tail -c 300 "runs-$n.log" | tr '\n' ' ' | cut -c1-200)"
done
echo QUEUE_DONE "$@" failed=$failed
[ "$failed" -eq 0 ]
