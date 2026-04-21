#!/bin/sh

### Job Name:
#BSUB -J eval_only_new_01_7d_g

### Queue Name:
#BSUB -q "gpua10 gpua40 gpul40s gpuh100 gpua100"

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Runtime limit for eval
#BSUB -W 6:00

### Output and error files
#BSUB -o batch_jobs/logs/eval_only_%J.out
#BSUB -e batch_jobs/logs/eval_only_%J.err

# Exit on error and undefined variables
set -eu

# Allow overriding the repo location from submit environment.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/advanced-ba}"
cd "$PROJECT_ROOT"

mkdir -p batch_jobs/logs outputs/adapters

module swap cuda/12.6.3

. .venv/bin/activate

echo "JobID: ${LSB_JOBID:-local}"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
nvidia-smi || true

if [ "${RUN_UV_SYNC:-1}" = "1" ]; then
  echo "Running uv sync..."
  uv sync
fi

# Checkpoint path (can be overridden by submit environment)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/final_7d_g/best_hypernet.pt}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found at ${CHECKPOINT_PATH}"
  exit 1
fi

RUN_ID="eval_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
ORCH_RUN_NAME="${RUN_ID}"

echo "Starting standalone evaluation run: ${ORCH_RUN_NAME}"
uv run python -m src.orchestration.main \
  "orchestration.checkpoint_path=${CHECKPOINT_PATH}" \
  "orchestration.run_id=${RUN_ID}" \
  orchestration.long_history_length_steps=2016 \
  orchestration.short_context_length_steps=288 \
  stride_steps=1 \
  start_test_day=null \
  n_test_days=null \
  proportion_test=0.2 \
  evaluation.use_dynamic_lora=true \
  wandb.enabled=true \
  "wandb.run_name=${ORCH_RUN_NAME}" \
  "wandb.tags=[hpc,fullscale,orchestration,evaluation]"

echo "Eval-only job finished successfully."
