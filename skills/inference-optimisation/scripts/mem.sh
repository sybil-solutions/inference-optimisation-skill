#!/usr/bin/env bash
# Print the real memory footprint (bytes) of a containerised or local model server, for bench.py's memory watchdog.
#   mem.sh TARGET [unified|metal]
# macOS / Apple Silicon (Darwin):
#   - TARGET="metal" or mode="metal": prints active Apple Silicon Metal GPU residency ("In use system memory" from ioreg).
#   - TARGET=<PID or process-name>: sums process RSS (bytes); if mode="unified", returns max(RSS, Metal In-Use bytes).
# Discrete NVIDIA GPU: sums nvidia-smi used memory of the container's processes + container anon RSS.
# Unified memory (Intel/AMD iGPU, Linux DRM drivers): sums drm-resident-* (deduplicated by drm-client-id)
# over the container's processes + cgroup anon. Compatible with Bash 3.2+ (no associative arrays).
set -euo pipefail
c=${1:?target container, PID, process name, or metal}; mode=${2:-auto}

if [ "$(uname -s)" = "Darwin" ]; then
  metal_bytes=$(ioreg -r -c IOAccelerator 2>/dev/null | sed -n 's/.*"In use system memory"=\([0-9][0-9]*\).*/\1/p' | awk '{s+=$1} END{print s+0}')
  if [ "$c" = "metal" ] || [ "$mode" = "metal" ]; then
    echo "${metal_bytes:-0}"
    exit 0
  fi
  rss_kb=0
  if [[ "$c" =~ ^[0-9]+$ ]]; then
    rss_kb=$(ps -o rss= -p "$c" 2>/dev/null | awk '{s+=$1} END{print s+0}')
  else
    pids=$(pgrep -f "$c" 2>/dev/null | tr '\n' ' ' || true)
    if [ -n "$pids" ]; then
      rss_kb=$(ps -o rss= -p $pids 2>/dev/null | awk '{s+=$1} END{print s+0}')
    fi
  fi
  rss_bytes=$(( ${rss_kb:-0} * 1024 ))
  if [ "$mode" = "unified" ] && [ "${metal_bytes:-0}" -gt "$rss_bytes" ]; then
    echo "$metal_bytes"
  else
    echo "$rss_bytes"
  fi
  exit 0
fi

pids=$(docker top "$c" -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ')
[ -z "$pids" ] && { echo 0; exit 0; }
cid=$(docker inspect -f '{{.Id}}' "$c")
cg=$(ls -d /sys/fs/cgroup/system.slice/docker-"$cid".scope /sys/fs/cgroup/docker/"$cid" 2>/dev/null | head -1 || true)
anon=$(awk '/^anon /{print $2}' "$cg/memory.stat" 2>/dev/null || echo 0)
gpu=0
if [ "$mode" != unified ] && command -v nvidia-smi >/dev/null; then
  for p in $pids; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$p" '$1==p{print $2}')
    gpu=$((gpu + ${m:-0} * 1048576))
  done
else
  seen=" "
  for p in $pids; do
    for f in /proc/"$p"/fdinfo/*; do
      id=$(awk '/drm-client-id/{print $2}' "$f" 2>/dev/null) || continue
      [ -z "$id" ] && continue
      case "$seen" in *" $id "*) continue ;; esac
      seen="$seen$id "
      kb=$(awk '/drm-resident-/{s+=$2} END{print s+0}' "$f"); gpu=$((gpu + kb * 1024))
    done
  done
fi
echo $((gpu + ${anon:-0}))
