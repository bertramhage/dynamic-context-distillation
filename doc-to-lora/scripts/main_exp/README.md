# D2L pipeline
### Data
You can either download the generated data (recommended, ~100 GB for each model) or generate them by youself.
Please see [`0-download_data.sh`](0-download_data.sh) for how to do model-specific data download.
```bash
# download training data for all three models (328GB)
uv run bash scripts/main_exp/0-download_data.sh
```

Generating data from scratch can take very long if not parallelized across multiple gpus.
```bash
# generate training data (takes very long if not parallelized across multiple gpus)
# optional: use the command below for generating data from scratch
# uv run bash scripts/main_exp/gen_data.sh
```

### Training
Simply run the training script once the data is ready.
```bash
# train
uv run bash scripts/main_exp/1-train.sh
```

### Evaluation
All evaluation scripts for reproducing the main results in the paper are included in [eval](eval/) directory.
