#!/bin/sh

### Job Name:
#BSUB -J baseline_ctx_sweep

### Queue Name:
#BSUB -q "gpua10 gpua40 gpul40s gpuh100 gpuv100 gpua100"

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Runtime limit
#BSUB -W 24:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o batch_jobs/logs/baseline_ctx_sweep_%J.out
#BSUB -e batch_jobs/logs/baseline_ctx_sweep_%J.err

# exit on error and undefined variables
set -eu

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/advanced-ba}"
cd "$PROJECT_ROOT"

mkdir -p batch_jobs/logs

module swap cuda/12.6.3

. .venv/bin/activate

echo "JobID: ${LSB_JOBID:-local}"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
nvidia-smi || true

if [ "${RUN_UV_SYNC:-0}" = "1" ]; then
  echo "Running uv sync..."
  uv sync
fi

SUBSET_FRACTION="${SUBSET_FRACTION:-0.4}"
SUBSET_SEED="${SUBSET_SEED:-42}"
BASE_SEED="${BASE_SEED:-42}"

RUN_PREFIX="baseline_subset40_${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"

echo "Using station subset fraction: ${SUBSET_FRACTION}"
echo "Using station subset seed: ${SUBSET_SEED}"
echo "Using run seed: ${BASE_SEED}"

for spec in 12h:144 24h:288 2d:576 7d:2016 14d:4032; do
  label="${spec%%:*}"
  history_steps="${spec##*:}"
  run_name="${RUN_PREFIX}_ctx_${label}"

  echo "Running baseline context sweep for ${label} (${history_steps} steps)"

  uv run python -m src.evaluation.main \
    --config-name experiment_baseline_context_sweep \
    dataset_cfg=dataset/PEMS-BAY \
    seed="${BASE_SEED}" \
    evaluation.history_length_steps="${history_steps}" \
    evaluation.station_subset_fraction="${SUBSET_FRACTION}" \
    evaluation.station_subset_seed="${SUBSET_SEED}" \
    wandb.enabled=true \
    "wandb.run_name=${run_name}"
done

echo "Context sweep baseline job finished successfully."
