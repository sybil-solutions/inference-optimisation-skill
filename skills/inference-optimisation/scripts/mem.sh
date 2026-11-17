#!/usr/bin/env bash
# Print the real memory footprint (bytes) of a model server, for bench.py's memory watchdog.
#   mem.sh CONTAINER [auto|unified|nvidia]     # Docker container
#   mem.sh --pid PID[,PID...] [auto|unified|nvidia]   # bare-metal server process(es)
#
# Discrete NVIDIA GPU: nvidia-smi used memory of the processes + anonymous memory.
# Unified memory (Intel/AMD iGPU, Linux DRM drivers): drm-resident-* from /proc/<pid>/fdinfo, deduplicated by
# drm-client-id, + anonymous memory. The kernel prints these sizes in B, KiB or MiB depending on alignment, so
# units are parsed per line (assuming KiB undercounts MiB-aligned allocations 1024x and the cap never trips).
# Anonymous memory: cgroup v2 `anon`, cgroup v1 `rss`, or (for --pid) RssAnon from /proc/<pid>/status.
# Reading other users' fdinfo needs root: run with passwordless sudo for this script, or as root.
# PROC_ROOT (default /proc) exists so tests can point the probe at a fake tree.
# Never embed a password here. On any probe failure this prints a number on stdout and a reason on stderr.
set -uo pipefail

usage() { sed -n '2,4p' "$0" >&2; exit 2; }
[ $# -ge 1 ] || usage

# Convert "<value> <unit>" from fdinfo into bytes. Exposed for tests via `mem.sh --to-bytes 12 MiB`.
to_bytes() {
  local v=$1 u=${2:-}
  case "$u" in
    ""|B) echo "$v" ;;
    KiB|kB|KB) echo $((v * 1024)) ;;
    MiB|MB) echo $((v * 1048576)) ;;
    GiB|GB) echo $((v * 1073741824)) ;;
    *) echo "mem.sh: unknown unit '$u'" >&2; echo 0 ;;
  esac
}

# Sum drm-resident-* across fdinfo files of the given pids, deduplicated by drm-client-id.
drm_bytes() {
  local total=0 p f id line val unit
  declare -A seen=()
  for p in "$@"; do
    for f in "${PROC_ROOT:-/proc}"/"$p"/fdinfo/*; do
      [ -r "$f" ] || continue
      id=$(awk '/^drm-client-id:/{print $2; exit}' "$f" 2>/dev/null)
      [ -z "$id" ] && continue
      [ -n "${seen[$id]:-}" ] && continue
      seen[$id]=1
      while IFS= read -r line; do
        # e.g. "drm-resident-system:\t30720 MiB"  or "drm-resident-vram0:\t4096"
        read -r _ val unit <<<"$line"
        [ -n "${val:-}" ] || continue
        total=$((total + $(to_bytes "$val" "${unit:-}")))
      done < <(grep -E '^drm-resident-' "$f" 2>/dev/null)
    done
  done
  echo "$total"
}

nvidia_bytes() {
  local total=0 p m
  local table
  table=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null) || { echo 0; return; }
  for p in "$@"; do
    m=$(awk -F', ' -v p="$p" '$1==p{s+=$2} END{print s+0}' <<<"$table")
    total=$((total + m * 1048576))
  done
  echo "$total"
}

pid_anon_bytes() {
  local total=0 p kb
  for p in "$@"; do
    kb=$(awk '/^RssAnon:/{print $2}' "${PROC_ROOT:-/proc}"/"$p"/status 2>/dev/null)
    total=$((total + ${kb:-0} * 1024))
  done
  echo "$total"
}

if [ "$1" = "--to-bytes" ]; then to_bytes "${2:?}" "${3:-}"; exit 0; fi

pids=() anon=0
if [ "$1" = "--pid" ]; then
  IFS=',' read -r -a pids <<<"${2:?pid list}"; mode=${3:-auto}
  # include children (engine workers) of each pid
  for p in "${pids[@]}"; do
    while read -r c; do [ -n "$c" ] && pids+=("$c"); done < <(pgrep -P "$p" 2>/dev/null)
  done
  anon=$(pid_anon_bytes "${pids[@]}")
else
  c=$1; mode=${2:-auto}
  mapfile -t pids < <(docker top "$c" -eo pid 2>/dev/null | tail -n +2 | tr -d ' ')
  if [ ${#pids[@]} -eq 0 ]; then echo "mem.sh: container '$c' not found or not running" >&2; echo 0; exit 0; fi
  cid=$(docker inspect -f '{{.Id}}' "$c")
  for cg in /sys/fs/cgroup/system.slice/docker-"$cid".scope /sys/fs/cgroup/docker/"$cid"; do      # cgroup v2
    [ -r "$cg/memory.stat" ] && { anon=$(awk '/^anon /{print $2}' "$cg/memory.stat"); break; }
  done
  if [ "${anon:-0}" = 0 ]; then                                                                      # cgroup v1
    for cg in /sys/fs/cgroup/memory/docker/"$cid" /sys/fs/cgroup/memory/system.slice/docker-"$cid".scope; do
      [ -r "$cg/memory.stat" ] && { anon=$(awk '/^(total_)?rss /{print $2; exit}' "$cg/memory.stat"); break; }
    done
  fi
  [ "${anon:-0}" = 0 ] && anon=$(pid_anon_bytes "${pids[@]}")                                         # fallback
fi

case "$mode" in
  nvidia) gpu=$(nvidia_bytes "${pids[@]}") ;;
  unified) gpu=$(drm_bytes "${pids[@]}") ;;
  auto) if command -v nvidia-smi >/dev/null 2>&1; then gpu=$(nvidia_bytes "${pids[@]}"); else gpu=$(drm_bytes "${pids[@]}"); fi ;;
  *) usage ;;
esac
echo $((gpu + ${anon:-0}))
