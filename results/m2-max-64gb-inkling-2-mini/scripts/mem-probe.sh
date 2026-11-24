#!/bin/zsh
# Prints the physical footprint in bytes for the process listening on $1.
# macOS only: this is the unified-memory figure the harness compares to the cap.
PORT="${1:-8123}"
P=$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -1)
if [ -z "$P" ]; then
  echo 0
  exit 0
fi
footprint -p "$P" 2>/dev/null | awk '/phys_footprint:/ {v=$2; u=$3; if(u=="GB") v*=1073741824; else if(u=="MB") v*=1048576; else if(u=="KB") v*=1024; printf "%d\n", v; exit}'
