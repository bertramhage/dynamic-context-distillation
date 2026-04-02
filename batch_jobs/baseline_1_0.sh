#!/bin/sh

### Job Name:
#BSUB -J baseline_1_0

### Queue Name:
#BSUB -q gpua100

### Requesting one host
#BSUB -R "span[hosts=1]"

### Requesting one GPU in exclusive process mode
#BSUB -gpu "num=1:mode=exclusive_process"

### Requesting 4 CPU cores, 4GB memory per core (min 4 cores pr gpu)
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"

### Setting a runtime limit of 12 hours
#BSUB -W 12:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o batch_jobs/logs/Output_%J.out
#BSUB -e batch_jobs/logs/Output_%J.err

# exit on error and undefined variables
set -eu

### cd to repo dir
cd ~/advanced-ba

# ensure local log directory exists for any script-side logging
mkdir -p batch_jobs/logs

### load cuda module
module swap cuda/12.6.3

### activate environment
. .venv/bin/activate

# print info about the job and environment
echo "JobID: $LSB_JOBID"
echo "Host: $(hostname)"
nvidia-smi || true

### run script
uv run python -m src.evaluation.main --config-name experiment_baseline dataset_cfg=dataset/PEMS-BAY