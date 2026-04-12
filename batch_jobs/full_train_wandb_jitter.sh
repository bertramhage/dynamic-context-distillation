#!/bin/sh

### Job Name:
#BSUB -J full_train_jitter_01

### Queue Name:
#BSUB -q "gpua40 gpul40s gpuh100 gpua100"

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Runtime limit for full training
#BSUB -W 24:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o batch_jobs/logs/full_train_jitter_%J.out
#BSUB -e batch_jobs/logs/full_train_jitter_%J.err

# Exit on error and undefined variables
set -eu

# Allow overriding the repo location from submit environment.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/advanced-ba}"
cd "$PROJECT_ROOT"

mkdir -p batch_jobs/logs checkpoints

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

RUN_ID="train_jitter_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_RUN_NAME="training_jitter_${RUN_ID}"

echo "Starting full-scale training run with jitter: ${TRAIN_RUN_NAME}"
uv run python -m src.training.main \
  training.long_context_steps=4032 \
  training.short_context_steps=256 \
  training.query_stride_steps=72 \
  training.train_batch_size=64 \
  training.num_workers=4 \
  training_loop.gradient_accumulation_steps=1 \
  training.length_jitter.enabled=true \
  wandb.enabled=true \
  training_loop.checkpoint_dir=checkpoints/jitter_01 \
  "wandb.run_name=${TRAIN_RUN_NAME}" \
  "wandb.tags=[hpc,fullscale,training,jitter]"

echo "Training jitter job finished successfully."
