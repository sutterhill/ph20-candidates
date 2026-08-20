#!/usr/bin/env bash
set -euo pipefail

AF2_DIR="${1:?pass AF2 work directory}"
IMAGE="ghcr.io/sokrypton/colabfold:1.6.2-cuda12"
CACHE="/home/ubuntu/colabfold_cache"

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  docker run --rm \
    --gpus "device=${shard}" \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -v "${AF2_DIR}:/work:rw" \
    -v "${CACHE}:/cache:ro" \
    "${IMAGE}" \
    colabfold_batch \
      "/work/input_shards/${shard}" \
      "/work/result_shards/${shard}" \
      --data /cache \
      --model-type alphafold2_ptm \
      --num-models 1 \
      --model-order 1 \
      --num-recycle 3 \
      --num-ensemble 1 \
      --num-seeds 1 \
      --random-seed 0 \
      --rank ptm \
      --num-relax 0 \
      --stop-at-score 100 \
      --sort-queries-by none \
      --compile-mode fast \
      --skip-output msa,plots \
      >"${AF2_DIR}/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
