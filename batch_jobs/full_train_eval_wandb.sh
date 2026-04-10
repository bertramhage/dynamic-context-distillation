#!/bin/sh

### Job Name:
#BSUB -J full_train_eval_01

### Queue Name:
#BSUB -q gpua100

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 8 CPU cores, 8GB memory per core
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"

### Runtime limit for full training + evaluation
#BSUB -W 24:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o batch_jobs/logs/full_train_eval_%J.out
#BSUB -e batch_jobs/logs/full_train_eval_%J.err

# Exit on error and undefined variables
set -eu

# Allow overriding the repo location from submit environment.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/advanced-ba}"
cd "$PROJECT_ROOT"

mkdir -p batch_jobs/logs checkpoints outputs/adapters

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

RUN_ID="full_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_RUN_NAME="training_${RUN_ID}"
ORCH_RUN_NAME="orchestration_eval_${RUN_ID}"
CHECKPOINT_PATH="checkpoints/best_hypernet.pt"

echo "Starting full-scale training run: ${TRAIN_RUN_NAME}"
uv run python -m src.training.main \
  training.long_context_steps=4032 \
  training.short_context_steps=256 \
  training.query_stride_steps=72 \
  wandb.enabled=true \
  "wandb.run_name=${TRAIN_RUN_NAME}" \
  "wandb.tags=[hpc,fullscale,training]"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Expected checkpoint not found at ${CHECKPOINT_PATH}"
  exit 1
fi

echo "Starting full-scale orchestration + evaluation run: ${ORCH_RUN_NAME}"
uv run python -m src.orchestration.main \
  "orchestration.checkpoint_path=${CHECKPOINT_PATH}" \
  "orchestration.run_id=${RUN_ID}" \
  orchestration.long_history_length_steps=4032 \
  orchestration.short_context_length_steps=256 \
  stride_steps=1 \
  start_test_day=null \
  n_test_days=null \
  proportion_test=0.2 \
  wandb.enabled=true \
  "wandb.run_name=${ORCH_RUN_NAME}" \
  "wandb.tags=[hpc,fullscale,orchestration,evaluation]"

echo "Job finished successfully."