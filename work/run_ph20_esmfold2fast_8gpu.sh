#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/codex_ph20_20260820
PYTHON=/home/ubuntu/esmfold2fast_env/bin/python
SCRIPT="$ROOT/work/fold_esmfold2fast_shard.py"
FASTA="${1:?pass a FASTA path}"
OUT_DIR="${2:?pass an output directory}"
mkdir -p "$OUT_DIR/logs"

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" "$SCRIPT" \
    --fasta "$FASTA" \
    --out-dir "$OUT_DIR" \
    --shard-index "$shard" \
    --num-shards 8 \
    >"$OUT_DIR/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
