#!/bin/sh

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/advanced-ba}"
RUN_UV_SYNC="${RUN_UV_SYNC:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
STATION_SUBSET_FRACTION="${STATION_SUBSET_FRACTION:-0.4}"
STATION_SUBSET_SEED="${STATION_SUBSET_SEED:-42}"
CONTEXT_ENCODER_MODEL="${CONTEXT_ENCODER_MODEL:-amazon/chronos-bolt-mini}"
DYNAMIC_BATCH_SIZE="${DYNAMIC_BATCH_SIZE:-null}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-8}"

cd "$PROJECT_ROOT"

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub command not found in PATH. Run this script on an LSF login node."
  exit 1
fi

extract_job_id() {
  echo "$1" | sed -n 's/.*<\([0-9][0-9]*\)>.*/\1/p'
}

submit_train_job() {
  perm="$1"
  dep_expr="$2"
  env_spec="all,PROJECT_ROOT=${PROJECT_ROOT},RUN_UV_SYNC=${RUN_UV_SYNC},TRAIN_PERM=${perm}"

  if [ -n "$dep_expr" ]; then
    out=$(bsub -env "$env_spec" -w "$dep_expr" -J "train_${perm}" < batch_jobs/train_hypernet_permutation.sh)
  else
    out=$(bsub -env "$env_spec" -J "train_${perm}" < batch_jobs/train_hypernet_permutation.sh)
  fi

  job_id=$(extract_job_id "$out")
  if [ -z "$job_id" ]; then
    echo "Failed to parse training job id from: $out"
    exit 1
  fi

  echo "$job_id"
}

submit_eval_job() {
  perm="$1"
  dep_expr="$2"
  env_spec="all,PROJECT_ROOT=${PROJECT_ROOT},RUN_UV_SYNC=${RUN_UV_SYNC},TRAIN_PERM=${perm},STATION_SUBSET_FRACTION=${STATION_SUBSET_FRACTION},STATION_SUBSET_SEED=${STATION_SUBSET_SEED},CONTEXT_ENCODER_MODEL=${CONTEXT_ENCODER_MODEL},DYNAMIC_BATCH_SIZE=${DYNAMIC_BATCH_SIZE},ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE}"

  out=$(bsub -env "$env_spec" -w "$dep_expr" -J "eval_${perm}" < batch_jobs/eval_hypernet_permutation_matrix.sh)

  job_id=$(extract_job_id "$out")
  if [ -z "$job_id" ]; then
    echo "Failed to parse eval job id from: $out"
    exit 1
  fi

  echo "$job_id"
}

submit_lane() {
  lane_name="$1"
  shift

  prev_dep=""
  echo "Submitting lane ${lane_name}: $*"

  for perm in "$@"; do
    train_job_id=$(submit_train_job "$perm" "$prev_dep")
    eval_job_id=$(submit_eval_job "$perm" "done(${train_job_id})")

    echo "  ${perm}: train job ${train_job_id} -> eval job ${eval_job_id}"

    # Keep one active job per lane to enforce global parallelism by lane count.
    prev_dep="done(${eval_job_id})"
  done
}

echo "Submitting full train+eval matrix"
echo "Project root: ${PROJECT_ROOT}"
echo "MAX_PARALLEL: ${MAX_PARALLEL}"
echo "Station subset: fraction=${STATION_SUBSET_FRACTION}, seed=${STATION_SUBSET_SEED}"

case "$MAX_PARALLEL" in
  2)
    submit_lane lane1 lc4032_teacher lc2016_teacher
    submit_lane lane2 lc4032_ground_truth lc2016_ground_truth
    ;;
  3)
    submit_lane lane1 lc4032_teacher
    submit_lane lane2 lc4032_ground_truth
    submit_lane lane3 lc2016_teacher lc2016_ground_truth
    ;;
  *)
    echo "MAX_PARALLEL must be 2 or 3"
    exit 1
    ;;
esac

echo "All jobs submitted."