# Solution Design (Current Implementation)

## 1. Overall Summary
This project is being built as a 3-layer system for time-series adapter distillation on Chronos-2:
- Layer 1 trains a hypernetwork that produces LoRA adapters.
- Layer 2 orchestrates experiment execution and prepares evaluation inputs.
- Layer 3 performs deterministic evaluation with Chronos-2, metrics, and runtime reporting.



This document describes what is actually implemented now, not the full future scope from `IMPLEMENTATION_PLAN.md`.

---

## 2. Three-Layer Structure

### Layer 1: LoRA Training (not implemented yet)
**Receives:** training dataset/windows, model/hypernetwork config.

**Outputs:** trained hypernetwork checkpoint and/or generated adapter artifacts.

### Layer 2: Orchestration (not implemented yet)
**Receives:** experiment config, checkpoints/artifacts, dataset splits.

**Outputs:** in-memory evaluation dataset + in-memory adapter assignment mapping (prediction task -> adapter_id), then calls Layer 3.

### Layer 3: Evaluation
The implementation is intentionally minimal and strict:
- Adapter assignment is treated as an orchestrator-provided input.
- LoRA adapters are applied to Chronos-2 using PEFT.
- Evaluation reports forecasting metrics and runtime/memory metrics.

**Receives:**
- merged config object (`OmegaConf`),
- loaded `Chronos2Pipeline`,
- evaluation dataframe (`df_long`),
- in-memory adapter assignments dataframe (optional parameter, required in strict adapter mode).

**Outputs:**
- forecasting metrics per horizon (MAE, MAPE, RMSE, COVERAGE, IQR stats),
- runtime stats (inference latency per task + memory stats),
- printed metric summaries in CLI flow.

---

## 3. Config Setup and How to Use It

### Config files currently used
- `conf/experiment_eval_mvp.yaml`: evaluation defaults for quick smoke runs.
- `conf/dataset/PEMS-BAY.yaml`: dataset-specific fields (target/id/timestamp/horizons/etc.).
- Hydra entrypoint in evaluation uses `conf/experiment` by default in `src/evaluation/main.py`.

### Key evaluation config conventions
- `dataset_cfg`: selects dataset config under `conf/dataset/...`.
- `stride_steps`, `start_test_day`, `n_test_days` / `proportion_test`: evaluation slicing.
- `prediction_length`, `horizons`, `quantile_levels`: forecasting behavior.
- `evaluation.history_length_steps`: explicit context length override.
- `adapter.*`: LoRA application settings (`adapter_root`, rank/alpha/targets, etc.).

### Environment and run command
This project uses **uv** for environment management.

Use:
- `uv sync` to install dependencies
- `uv run ...` for commands

Examples:
- `uv run python -m compileall src/evaluation src/utils`
- `uv run python scripts/random_lora_smoke_test.py`

---

## 4. `src/` Folder Structure

```text
src/
  __init__.py
  evaluation/
    __init__.py
    adapter_runtime.py
    main.py
  orchestration/
    __init__.py
  training/
    __init__.py
  utils/
    __init__.py
    metrics.py
    utils.py
    wandb_utils.py
```

---

## 5. Layer Details

## Layer 1: Training
_Not implemented yet._

### Current state
- Placeholder package only (`src/training/__init__.py`).

### Notes
- No training loop, context encoder, or hypernetwork code has been added yet.

---

## Layer 2: Orchestration
_Not implemented yet._

### Current state
- Placeholder package only (`src/orchestration/__init__.py`).

### Notes
- No orchestration runner has been added yet.
- Current smoke testing script directly calls evaluation layer APIs.

---

## Layer 3: Evaluation

### Files
- `src/evaluation/main.py`
- `src/evaluation/adapter_runtime.py`
- Shared metric/data helpers in `src/utils/metrics.py` and `src/utils/utils.py`

### Exact coding choices implemented
1. **Single loaded Chronos model**
- Chronos-2 pipeline is created once via `BaseChronosPipeline.from_pretrained(...)`.

2. **In-memory adapter assignment mapping**
- Assignment mapping is built from an in-memory dataframe with:
  - `item_id` (or configured id column),
  - `prediction_time`,
  - `adapter_id`.
- Mapping key: `(item_id, prediction_time)`.

3. **Strict assignment behavior**
- Missing assignment for any prediction task raises an error.
- No silent fallback for missing task mappings.

4. **PEFT LoRA application**
- `ensure_peft_model(...)` wraps the base model with LoRA config only when needed.
- `apply_adapter(...)` loads adapter from `adapter_root/<adapter_id>` if needed and activates it.
- `adapter_id == "__none__"` triggers base-model path.

5. **Grouped inference by adapter_id**
- Prediction tasks are grouped by adapter id per rolling step.
- Each group runs one `predict_df` call over its station subset.
- This avoids mixing different adapters in one forward pass.

6. **Base model path under PEFT wrapper**
- For `__none__` when model is PEFT-wrapped, inference runs inside `with pipeline.model.disable_adapter():`.

7. **Metrics preserved from baseline style**
- MAE/MAPE/RMSE/COVERAGE/IQR aggregation per horizon remains the same structure.

8. **Runtime and memory metrics included**
- Latency is measured **per prediction task** (group call time divided by tasks in group), then aggregated.
- Reported runtime fields:
  - `predict_backend_calls`
  - `prediction_tasks`
  - `total_inference_seconds`
  - `avg_task_inference_seconds`
  - `p95_task_inference_seconds`
- Memory fields:
  - `cuda_peak_memory_mb` (CUDA peak allocated)
  - `mps_peak_memory_mb` (sampled current allocated peak)
  - `cpu_peak_rss_mb` (platform-correct `ru_maxrss` handling)

### Public evaluation API (current)
- `run_evaluation(cfg, pipeline, df_long, assignments_df=None, return_runtime=False)`
  - Returns metrics dict by default.
  - Returns `(metrics, runtime_stats)` when `return_runtime=True`.

### CLI entrypoint (current)
- `src/evaluation/main.py` main function:
  - loads Hydra config,
  - initializes WandB run (if enabled),
  - loads Chronos pipeline and dataset,
  - runs evaluation,
  - prints forecast + runtime summaries,
  - logs metrics and runtime stats to WandB (if enabled).

---

## 6. Experiment Tracking (WandB)

### Module
- `src/utils/wandb_utils.py` provides `init_wandb(cfg, group)` and `finish_wandb()`.

### Run ownership — CLI vs API
- **CLI entrypoint** (e.g. `src/evaluation/main.py main()`): calls `init_wandb(cfg, group="evaluation")` and `finish_wandb()`. Owns the full run lifecycle.
- **API call** (e.g. calling `run_evaluation()` from orchestration or a script): the caller is responsible for `init_wandb` / `finish_wandb`. `run_evaluation()` logs per-step metrics if `wandb.run` is active — it never inits or finishes.

### Per-step logging
`run_evaluation()` checks `wandb.run is not None` after each rolling-window step and logs:
- `eval/h{horizon}_{metric}` — metric value for the current step
- `eval/h{horizon}_{metric}_running_avg` — running average up to the current step
- `eval/progress` — fraction of steps completed

This gives live visibility into long evaluation runs.

### Final summary
`main()` writes cross-horizon averages and runtime stats to `wandb.summary` after the run completes.

### Project and group
- **Project** is hardcoded to `advanced-ba` in `wandb_utils.py`.
- **Group** is hardcoded per layer (e.g. `"evaluation"`, `"training"`), passed as an argument to `init_wandb`.
- **Run name** comes from `cfg.wandb.run_name`.

### Config
All experiment configs include a `wandb:` section (disabled by default):
```yaml
wandb:
  enabled: false
  entity: null
  run_name: null
  tags: []
```
Enable via CLI override: `wandb.enabled=true`.

### Metric namespace convention
| Prefix | Layer | Examples |
|--------|-------|---------|
| `eval/` | Evaluation | `eval/h15_MAE`, `eval/h15_MAE_running_avg`, `eval/avg_RMSE` |
| `runtime/` | Evaluation | `runtime/avg_task_inference_seconds` |
| `train/` | Training (future) | `train/loss`, `train/lr` |

### Adding WandB to a new layer
1. Call `init_wandb(cfg, group="your_layer")` at the top of your CLI entrypoint.
2. Inside your loop, check `if wandb.run is not None:` and call `wandb.log(...)`.
3. Call `finish_wandb()` at the end of your CLI entrypoint.
4. If the function may be called from an outer layer, do **not** init/finish inside it — just log when `wandb.run` is active.

---