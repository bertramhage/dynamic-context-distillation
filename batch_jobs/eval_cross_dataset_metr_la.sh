#!/bin/sh

### Job Name:
#BSUB -J rq5_metr_la

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
#BSUB -W 12:00

### Output and error files
#BSUB -o batch_jobs/logs/rq5_metr_la_%J.out
#BSUB -e batch_jobs/logs/rq5_metr_la_%J.err

set -eu

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

RUN_STAMP="rq5_metr_la_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/final_7d_t_v2/best_hypernet.pt}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found at ${CHECKPOINT_PATH}"
  exit 1
fi

run_baseline() {
  LABEL="$1"
  HISTORY_STEPS="$2"
  RUN_NAME="${RUN_STAMP}_baseline_${LABEL}"

  echo "Running METR-LA baseline: ${LABEL} (history=${HISTORY_STEPS} steps)"
  uv run python -m src.evaluation.main \
    --config-name experiment_baseline \
    dataset_cfg=dataset/METR-LA \
    cross_learning=false \
    +evaluation.history_length_steps="${HISTORY_STEPS}" \
    wandb.enabled=true \
    "wandb.run_name=${RUN_NAME}" \
    "wandb.tags=[hpc,rq5,cross_dataset,metr_la,baseline,context_${LABEL}]"
}

run_baseline 24h 288
run_baseline 2d 576
run_baseline 7d 2016

ADAPTED_RUN_ID="${RUN_STAMP}_adapted"
ADAPTED_RUN_NAME="${RUN_STAMP}_adapted_pems_bay_hypernet"

echo "Running METR-LA adapted evaluation with PEMS-BAY trained hypernetwork"
uv run python -m src.orchestration.main \
  dataset_cfg=dataset/PEMS-BAY \
  orchestration.eval_dataset_cfg=dataset/METR-LA \
  "orchestration.checkpoint_path=${CHECKPOINT_PATH}" \
  "orchestration.run_id=${ADAPTED_RUN_ID}" \
  orchestration.long_history_length_steps=2016 \
  orchestration.short_context_length_steps=288 \
  stride_steps=1 \
  start_test_day=null \
  n_test_days=null \
  proportion_test=0.2 \
  evaluation.use_dynamic_lora=true \
  wandb.enabled=true \
  "wandb.run_name=${ADAPTED_RUN_NAME}" \
  "wandb.tags=[hpc,rq5,cross_dataset,metr_la,adapted,pems_bay_hypernet]"

echo "RQ5 METR-LA cross-dataset evaluation job finished successfully."
