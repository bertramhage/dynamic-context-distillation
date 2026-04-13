#!/bin/sh

### Job Name (overridden by submit script):
#BSUB -J train_hypernet_perm

### Queue Name:
#BSUB -q gpul40s

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
#BSUB -o batch_jobs/logs/train_matrix_%J.out
#BSUB -e batch_jobs/logs/train_matrix_%J.err

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/advanced-ba}"
TRAIN_PERM="${TRAIN_PERM:-}"

if [ -z "$TRAIN_PERM" ]; then
  echo "TRAIN_PERM must be set (lc2016_teacher | lc2016_ground_truth | lc4032_teacher | lc4032_ground_truth)."
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p batch_jobs/logs checkpoints/hpc_matrix

module swap cuda/12.6.3

. .venv/bin/activate

echo "JobID: ${LSB_JOBID:-local}"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Training permutation: $TRAIN_PERM"
nvidia-smi || true

if [ "${RUN_UV_SYNC:-0}" = "1" ]; then
  echo "Running uv sync..."
  uv sync
fi

SHORT_CONTEXT=288

case "$TRAIN_PERM" in
  lc2016_teacher)
    LONG_CONTEXT=2016
    TARGET_MODE=teacher
    LONG_SIGMA_OUTER=160.0
    LONG_SIGMA_INNER=48.0
    LONG_MIN_STEPS=1008
    LONG_MAX_STEPS=4032
    ;;
  lc2016_ground_truth)
    LONG_CONTEXT=2016
    TARGET_MODE=ground_truth
    LONG_SIGMA_OUTER=160.0
    LONG_SIGMA_INNER=48.0
    LONG_MIN_STEPS=1008
    LONG_MAX_STEPS=4032
    ;;
  lc4032_teacher)
    LONG_CONTEXT=4032
    TARGET_MODE=teacher
    LONG_SIGMA_OUTER=320.0
    LONG_SIGMA_INNER=96.0
    LONG_MIN_STEPS=2016
    LONG_MAX_STEPS=8064
    ;;
  lc4032_ground_truth)
    LONG_CONTEXT=4032
    TARGET_MODE=ground_truth
    LONG_SIGMA_OUTER=320.0
    LONG_SIGMA_INNER=96.0
    LONG_MIN_STEPS=2016
    LONG_MAX_STEPS=8064
    ;;
  *)
    echo "Unknown TRAIN_PERM: $TRAIN_PERM"
    exit 1
    ;;
esac

CHECKPOINT_DIR="checkpoints/hpc_matrix/${TRAIN_PERM}"
RUN_STAMP="${LSB_JOBID:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_RUN_NAME="train_${TRAIN_PERM}_${RUN_STAMP}"

echo "Long context steps: ${LONG_CONTEXT}"
echo "Short context steps: ${SHORT_CONTEXT}"
echo "Target mode: ${TARGET_MODE}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Length jitter (long): sigma_outer=${LONG_SIGMA_OUTER}, sigma_inner=${LONG_SIGMA_INNER}, min=${LONG_MIN_STEPS}, max=${LONG_MAX_STEPS}"

uv run python -m src.training.main \
  seed=42 \
  training.long_context_steps="${LONG_CONTEXT}" \
  training.short_context_steps="${SHORT_CONTEXT}" \
  training.query_stride_steps=72 \
  training.train_batch_size=64 \
  training.num_workers=4 \
  training_loop.gradient_accumulation_steps=2 \
  training_loop.target_mode="${TARGET_MODE}" \
  training.length_jitter.enabled=true \
  training.length_jitter.quantize_steps=24 \
  training.length_jitter.long_sigma_outer="${LONG_SIGMA_OUTER}" \
  training.length_jitter.long_sigma_inner="${LONG_SIGMA_INNER}" \
  training.length_jitter.long_min_steps="${LONG_MIN_STEPS}" \
  training.length_jitter.long_max_steps="${LONG_MAX_STEPS}" \
  training.length_jitter.short_sigma_outer=40.0 \
  training.length_jitter.short_sigma_inner=16.0 \
  training.length_jitter.short_min_steps=144 \
  training.length_jitter.short_max_steps=576 \
  teacher_cache.jitter_seed=42 \
  wandb.enabled=true \
  training_loop.checkpoint_dir="${CHECKPOINT_DIR}" \
  "wandb.run_name=${TRAIN_RUN_NAME}" \
  "wandb.tags=[hpc,train_matrix,${TRAIN_PERM},${TARGET_MODE},jitter]"

if [ ! -f "${CHECKPOINT_DIR}/best_hypernet.pt" ]; then
  echo "Training finished but checkpoint missing: ${CHECKPOINT_DIR}/best_hypernet.pt"
  exit 1
fi

echo "Training job finished successfully: ${CHECKPOINT_DIR}/best_hypernet.pt"