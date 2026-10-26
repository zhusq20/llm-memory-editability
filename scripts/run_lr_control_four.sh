#!/usr/bin/env bash
# Run ONLY the four transferred arms, sequentially on one chosen GPU.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: bash $0 BATCH_DIRECTORY GPU_INDEX FOUR_CPU_IDS WANDB_TOKEN_FILE" >&2
  echo "Example CPU list: 4,5,6,7. Default is a plan; add EXECUTE=1 to run." >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_flags=()
if [[ "${EXECUTE:-0}" == 1 ]]; then run_flags+=(--execute); fi
for job in loop-lr5e-4 loop-lr1e-3 discoloop-lr2e-4 discoloop-lr5e-4; do
  LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "${PYTHON_BIN:-python3}" "$script_dir/continue_lr_control_handoff.py" \
    --batch "$1" --job "$job" --gpu "$2" --cpus "$3" \
    --wandb-token-file "$4" --docker-context "${DOCKER_CONTEXT:-lm-memory}" \
    --min-total-gpu-mib "${MIN_TOTAL_GPU_MIB:-90000}" \
    --min-free-gpu-mib "${MIN_FREE_GPU_MIB:-80000}" "${run_flags[@]}"
done
