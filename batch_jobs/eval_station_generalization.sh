#!/bin/sh

### Job Name:
#BSUB -J station_generalization

### Queue Name:
#BSUB -q "gpua10 gpua40 gpul40s gpuh100 gpua100"

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Runtime limit
#BSUB -W 24:00

### Output and error files
#BSUB -o batch_jobs/logs/station_generalization_%J.out
#BSUB -e batch_jobs/logs/station_generalization_%J.err

set -eu

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

RUN_ID="rq2_station_gen_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="checkpoints/${RUN_ID}"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_hypernet.pt"
SPLIT_PATH="${CHECKPOINT_DIR}/station_split.json"

echo "Starting RQ2 station-holdout training: ${RUN_ID}"
uv run python -m src.training.main \
  training.long_context_steps=2016 \
  training.short_context_steps=288 \
  training.train_batch_size=64 \
  training.num_workers=4 \
  training_loop.gradient_accumulation_steps=2 \
  training.length_jitter.enabled=true \
  training.length_jitter.long_sigma_outer=320.0 \
  training.length_jitter.long_sigma_inner=96.0 \
  training.length_jitter.short_sigma_outer=40.0 \
  training.length_jitter.short_sigma_inner=16.0 \
  training.length_jitter.quantize_steps=24 \
  training.length_jitter.long_min_steps=1008 \
  training.length_jitter.long_max_steps=4032 \
  training.length_jitter.short_min_steps=144 \
  training.length_jitter.short_max_steps=432 \
  training_loop.target_mode=teacher \
  training.station_holdout.enabled=true \
  training.station_holdout.train_fraction=0.2 \
  training.station_holdout.station_split_seed=42 \
  training.station_holdout.stratify_by_mean=true \
  training_loop.checkpoint_dir="${CHECKPOINT_DIR}" \
  wandb.enabled=true \
  "wandb.run_name=${RUN_ID}_train" \
  "wandb.tags=[hpc,rq2,station-generalization,training]"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found at ${CHECKPOINT_PATH}"
  exit 1
fi

if [ ! -f "$SPLIT_PATH" ]; then
  echo "Station split not found at ${SPLIT_PATH}"
  exit 1
fi

echo "Evaluating on train stations"
uv run python -m src.orchestration.main \
  "orchestration.checkpoint_path=${CHECKPOINT_PATH}" \
  "orchestration.station_split_path=${SPLIT_PATH}" \
  orchestration.station_eval_set=train \
  "orchestration.run_id=${RUN_ID}_trainset" \
  orchestration.long_history_length_steps=2016 \
  orchestration.short_context_length_steps=288 \
  stride_steps=1 \
  start_test_day=null \
  n_test_days=null \
  proportion_test=0.2 \
  evaluation.use_dynamic_lora=true \
  wandb.enabled=true \
  "wandb.run_name=${RUN_ID}_eval_train" \
  "wandb.tags=[hpc,rq2,station-generalization,orchestration,evaluation,station_set_train]"

echo "Evaluating on holdout stations"
uv run python -m src.orchestration.main \
  "orchestration.checkpoint_path=${CHECKPOINT_PATH}" \
  "orchestration.station_split_path=${SPLIT_PATH}" \
  orchestration.station_eval_set=holdout \
  "orchestration.run_id=${RUN_ID}_holdoutset" \
  orchestration.long_history_length_steps=2016 \
  orchestration.short_context_length_steps=288 \
  stride_steps=1 \
  start_test_day=null \
  n_test_days=null \
  proportion_test=0.2 \
  evaluation.use_dynamic_lora=true \
  wandb.enabled=true \
  "wandb.run_name=${RUN_ID}_eval_holdout" \
  "wandb.tags=[hpc,rq2,station-generalization,orchestration,evaluation,station_set_holdout]"

echo "RQ2 station generalization job finished successfully."
