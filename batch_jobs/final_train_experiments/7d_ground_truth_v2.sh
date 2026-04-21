#!/bin/sh

### Job Name:
#BSUB -J train_only_new_02_7d_g_v2

### Queue Name:
#BSUB -q "gpua100"

### Ensure 80GB GPU used
#BSUB -R "select[gpu80gb]"

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Runtime limit for full training
#BSUB -W 24:00

### Output and error files
#BSUB -o batch_jobs/logs/full_train_jitter_v2_%J.out
#BSUB -e batch_jobs/logs/full_train_jitter_v2_%J.err

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

RUN_ID="train_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_RUN_NAME="${RUN_ID}"

echo "Starting full-scale training run with jitter: ${TRAIN_RUN_NAME}"
uv run python -m src.training.main \
  training.long_context_steps=2016 \
  training.short_context_steps=288 \
  training.train_batch_size=64 \
  training.num_workers=4 \
  training_loop.gradient_accumulation_steps=2 \
  optimizer.lr=2.5e-5 \
  training_loop.warmup_steps=200 \
  training_loop.max_epochs=140 \
  training_loop.patience=20 \
  training.length_jitter.enabled=true \
  training.length_jitter.apply_in_val=false \
  training.length_jitter.long_sigma_outer=320.0 \
  training.length_jitter.long_sigma_inner=96.0 \
  training.length_jitter.short_sigma_outer=40.0 \
  training.length_jitter.short_sigma_inner=16.0 \
  training.length_jitter.quantize_steps=24 \
  training.length_jitter.long_min_steps=1008 \
  training.length_jitter.long_max_steps=4032 \
  training.length_jitter.short_min_steps=144 \
  training.length_jitter.short_max_steps=432 \
  wandb.enabled=true \
  training_loop.target_mode=ground_truth \
  training_loop.checkpoint_dir=checkpoints/final_7d_g_v2 \
  "wandb.run_name=${TRAIN_RUN_NAME}" \
  "wandb.tags=[hpc,fullscale,training,v2,ground_truth]"

echo "Training jitter job finished successfully."

echo "Submitting eval jobs for best and final checkpoints..."
CHECKPOINT_PATH="checkpoints/final_7d_g_v2/best_hypernet.pt" \
  bsub < batch_jobs/final_train_experiments/evals/eval_7d_ground_truth.sh
CHECKPOINT_PATH="checkpoints/final_7d_g_v2/final_hypernet.pt" \
  bsub < batch_jobs/final_train_experiments/evals/eval_7d_ground_truth.sh
