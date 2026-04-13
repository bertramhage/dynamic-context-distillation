#!/bin/sh

### Job Name (overridden by submit script):
#BSUB -J eval_hypernet_perm

### Queue Name:
#BSUB -q gpua40

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"

### Runtime limit for packed eval sweep
#BSUB -W 24:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o batch_jobs/logs/eval_matrix_%J.out
#BSUB -e batch_jobs/logs/eval_matrix_%J.err

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/advanced-ba}"
TRAIN_PERM="${TRAIN_PERM:-}"

if [ -z "$TRAIN_PERM" ]; then
  echo "TRAIN_PERM must be set (lc2016_teacher | lc2016_ground_truth | lc4032_teacher | lc4032_ground_truth)."
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p batch_jobs/logs outputs/adapters/hpc_matrix

module swap cuda/12.6.3

. .venv/bin/activate

echo "JobID: ${LSB_JOBID:-local}"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Eval permutation: $TRAIN_PERM"
nvidia-smi || true

if [ "${RUN_UV_SYNC:-0}" = "1" ]; then
  echo "Running uv sync..."
  uv sync
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/hpc_matrix/${TRAIN_PERM}/best_hypernet.pt}"
SHORT_CONTEXT="${SHORT_CONTEXT:-288}"
EVAL_LONG_CONTEXTS="${EVAL_LONG_CONTEXTS:-2016 4032 8064 16128}"
CONTEXT_ENCODER_MODEL="${CONTEXT_ENCODER_MODEL:-amazon/chronos-bolt-mini}"
STATION_SUBSET_FRACTION="${STATION_SUBSET_FRACTION:-0.4}"
STATION_SUBSET_SEED="${STATION_SUBSET_SEED:-42}"
DYNAMIC_BATCH_SIZE="${DYNAMIC_BATCH_SIZE:-null}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-8}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found at ${CHECKPOINT_PATH}"
  exit 1
fi

RUN_STAMP="${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Station subset: fraction=${STATION_SUBSET_FRACTION}, seed=${STATION_SUBSET_SEED}"

for LONG_HISTORY in ${EVAL_LONG_CONTEXTS}; do
  ORCH_RUN_ID="mx_${TRAIN_PERM}_lh${LONG_HISTORY}_${RUN_STAMP}"
  RUN_NAME="eval_${TRAIN_PERM}_lh${LONG_HISTORY}_sc${SHORT_CONTEXT}_${RUN_STAMP}"

  echo "Running eval for ${TRAIN_PERM}: long_history=${LONG_HISTORY}, short_context=${SHORT_CONTEXT}"

  uv run python -m src.orchestration.main \
    "orchestration.checkpoint_path=${CHECKPOINT_PATH}" \
    "orchestration.run_id=${ORCH_RUN_ID}" \
    "orchestration.adapter_output_dir=outputs/adapters/hpc_matrix" \
    "orchestration.long_history_length_steps=${LONG_HISTORY}" \
    "orchestration.short_context_length_steps=${SHORT_CONTEXT}" \
    "orchestration.encode_batch_size=${ENCODE_BATCH_SIZE}" \
    "context_encoder_model=${CONTEXT_ENCODER_MODEL}" \
    stride_steps=1 \
    start_test_day=null \
    n_test_days=null \
    proportion_test=0.2 \
    "evaluation.station_subset_fraction=${STATION_SUBSET_FRACTION}" \
    "evaluation.station_subset_seed=${STATION_SUBSET_SEED}" \
    evaluation.use_dynamic_lora=true \
    "evaluation.dynamic_batch_size=${DYNAMIC_BATCH_SIZE}" \
    wandb.enabled=true \
    "wandb.run_name=${RUN_NAME}" \
    "wandb.tags=[hpc,eval_matrix,${TRAIN_PERM},lh${LONG_HISTORY},sc${SHORT_CONTEXT},subset40,dynamic_lora]"
done

echo "Packed eval job finished successfully for ${TRAIN_PERM}."