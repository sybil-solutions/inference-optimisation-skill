#!/bin/zsh
# Three profiles, exactly as benchmarked. Set MODEL_DIR and VOCAB before running.
#   MODEL_DIR  directory holding the Inkling-2-Mini weights (thinkingmachines/Inkling-2-Mini @ 7a67ae2)
#   VOCAB      path to the pruned 32k draft vocabulary, ids.npy; omit to run without it
set -e

PROFILE="agent"
PORT="8123"
MODEL="${MODEL_DIR:?set MODEL_DIR to the Inkling-2-Mini weights directory}"
VOCAB="${VOCAB:-}"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

export PYTHONPATH="${MLX_DSPARK_SRC:?set MLX_DSPARK_SRC to the mlx-dspark fork src/ directory}"
export MLX_DSPARK_INKLING_FP16=1
export MLX_DSPARK_INKLING_FP16_FUSED=1
export MLX_DSPARK_NVFP4_FP16_SCALE=1
export MLX_DSPARK_NVFP4_NIBBLE=half2
export MLX_DSPARK_BOUNDARY_SNAPSHOT=1
if [ -n "$VOCAB" ] && [ -f "$VOCAB" ]; then
  export MLX_DSPARK_INKLING_DRAFT_VOCAB="$VOCAB"
fi

ARGS=(--model "$MODEL" --port "$PORT" --target-bits 8 --max-batch 4)
case "$PROFILE" in
  plain) ARGS+=(--mode baseline) ;;
  multi) export MLX_DSPARK_SHARED_PREFIX_KV=1; ARGS+=(--lookup-drafts) ;;
  agent|*) ARGS+=(--lookup-drafts) ;;
esac

exec "${MLX_PYTHON:-python3}" -m mlx_dspark serve "${ARGS[@]}" "${EXTRA[@]}"
