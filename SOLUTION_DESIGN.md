# Solution Design (Current Implementation)

## 1. Overall Summary
This project is being built as a 3-layer system for time-series adapter distillation on Chronos-2:
- Layer 1 trains a hypernetwork that produces LoRA adapters.
- Layer 2 orchestrates experiment execution and prepares evaluation inputs.
- Layer 3 performs deterministic evaluation with Chronos-2, metrics, and runtime reporting.



This document describes what is actually implemented now, not the full future scope from `IMPLEMENTATION_PLAN.md`.

---

## 2. Three-Layer Structure

### Layer 1: LoRA Training
**Receives:** training dataset/windows, model/hypernetwork config.

**Outputs:** trained hypernetwork checkpoint (`best_hypernet.pt`, `final_hypernet.pt`).

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
    context_encoder.py
    lora_saver.py
    main.py
    run.py
  training/
    __init__.py
    dataset.py
    hypernet.py
    lora_injection.py
    main.py
    perceiver.py
    trainer.py
  utils/
    __init__.py
    metrics.py
    utils.py
    wandb_utils.py
```

---

## 5. Layer Details

## Layer 1: Training

### Files
- `src/training/main.py` — Hydra CLI entrypoint
- `src/training/perceiver.py` — Perceiver aggregator (cross-attention based context compression)
- `src/training/hypernet.py` — HyperLoRA generator (Perceiver + EinMix projection heads)
- `src/training/lora_injection.py` — Runtime LoRA monkey-patching for training forward passes
- `src/training/dataset.py` — Rolling-window dataset with multi-query support
- `src/training/trainer.py` — Teacher-student distillation training loop
- `conf/experiment_training.yaml` — Default training config

### How it works

The training layer implements teacher-student distillation to train a hypernetwork that produces LoRA adapters for Chronos-2.

1. **Dataset** (`HypernetTrainingDataset`): constructs rolling-window samples from the long-format DataFrame. Each sample contains a long context window and multiple (short_context, forecast_target) query pairs. The long context is what the hypernetwork sees; the short contexts are what the LoRA-adapted student model sees during inference. The collate path now supports optional hierarchical length jitter for both long and short contexts: one batch-level mean is sampled with `sigma_outer`, then each sample draws from an inner Gaussian (`sigma_inner`) and is clamped/quantized to configured bounds.

2. **Context encoding**: a separate frozen context encoder model (default: Chronos-Bolt-Mini, configurable via `context_encoder_model`) encodes the long context and returns its last hidden state `[batch, num_patches, d_model]`. This is cheaper than the original approach of capturing per-layer intermediates from Chronos-2, and avoids tying the context encoder to the student/teacher model. `ChronosContextEncoder` supports both Chronos-Bolt (T5-based `encode()`) and Chronos-2 (manual block loop) backends.

3. **Hypernetwork** (`HyperLoRA`): a Perceiver aggregator compresses the context encoder output into fixed-size latent queries, which are processed by ResMLPBlock layers and projected via EinMix heads to produce LoRA A/B matrices for each target module across all 12 encoder layers. The EinMix head is initialized with a custom small std (`0.5 / sqrt(d_latent + d_lora * rank)`) to prevent wild initial LoRA outputs.

4. **LoRA injection** (`apply_lora_to_model`): monkey-patches `nn.Linear.forward` on Chronos-2's TimeSelfAttention modules (q, k, v, o) to add the LoRA delta. Gradients flow through the LoRA weights back to the hypernetwork. Patches are cleaned up after each forward pass.

5. **Training loop** (`HypernetTrainer`):
   - Teacher: full-context Chronos-2 (frozen) produces quantile predictions.
   - Student: short-context Chronos-2 + LoRA from hypernetwork.
   - Loss: smooth L1 on quantile predictions (teacher vs student).
   - **Gradient accumulation**: loss is scaled by `1/grad_accum_steps`, optimizer steps every N batches (default 8), giving an effective batch size of `train_batch_size × grad_accum_steps`.
   - **LR schedule**: cosine annealing with linear warmup (`SequentialLR`). Linear warmup over configurable steps (default 100), then cosine decay to `eta_min=1e-7`. Current LR is logged to wandb.
   - Early stopping on validation loss with configurable patience.
   - Optional L1 regularization on generated LoRA weights.

### Data flow
```
Long history → Frozen context encoder (Chronos-Bolt-Mini) → last hidden [B, S, 384]
              → Perceiver → [B, L*M*r, d_latent]
              → ResMLPBlocks → EinMix heads
              → LoRA dict {q/k/v/o: {A, B}}

Short history + LoRA → Chronos-2 (monkey-patched) → student quantile preds
Full history         → Chronos-2 (frozen)          → teacher quantile preds
                                                   → Loss(student, teacher)
```

Note: two separate models are loaded — Chronos-Bolt-Mini for context encoding (embeddings for the hypernetwork) and Chronos-2 for teacher/student predictions. The context encoder model is configurable via `context_encoder_model` in the training config.

### Hypernetwork output format
```python
# lora_dict[module_short]["A"]  -> [batch, 12, r, 768]
# lora_dict[module_short]["B"]  -> [batch, 12, 768, r]
# module_short ∈ {"q", "k", "v", "o"}
```

This is the same format expected by the orchestration layer's `save_adapter_to_disk`.

### Architecture dimensions (current defaults)
- Context encoder: Chronos-Bolt-Mini (d_model=384)
- Perceiver latent dim: 128 (was 256)
- Perceiver bottleneck: n_latent_queries=32 (was 64)
- Perceiver blocks: 1 (was 2)
- Pre-head ResMLPBlock layers: 1 (was 2)
- Output queries: num_layers × num_modules × lora_rank = 12 × 4 × 8 = 384
- EinMix head output: d_model + d_model = 1536 per query (LoRA targets are still Chronos-2 Base)
- Total hypernetwork params: ~11M (was ~51M with previous defaults)

### Public training API
- `main()` — Hydra CLI entrypoint, manages full lifecycle.
- `HypernetTrainer.train()` → returns best checkpoint path.

### CLI entrypoint
```bash
uv run python -m src.training.main
uv run python -m src.training.main wandb.enabled=true optimizer.lr=5e-5
uv run python -m src.training.main training.train_batch_size=8 training_loop.max_epochs=100
```

---

## Layer 2: Orchestration

### Files
- `src/orchestration/main.py` — Hydra CLI entrypoint
- `src/orchestration/run.py` — core orchestration logic (`run_orchestration`)
- `src/orchestration/context_encoder.py` — frozen Chronos-2 encoder wrapper
- `src/orchestration/lora_saver.py` — hypernetwork output → PEFT adapter on disk
- `conf/experiment_orchestration.yaml` — default orchestration config

### How it works

The orchestration layer is a middleman between a trained hypernetwork (Layer 1 output) and the evaluation layer (Layer 3). Its job:

1. **Load** dataset (`shared_utils.load_dataset`), Chronos-2 pipeline, and hypernetwork checkpoint.
2. **Generate LoRA adapters**: for each sensor, extract the long-history window from the dataset, encode it through the frozen Chronos-2 encoder to get hidden states `[num_patches, 768]`, run the hypernetwork to produce LoRA weight dicts, and save each adapter to disk as a PEFT-compatible directory.
3. **Build assignment_df**: a DataFrame mapping `(item_id, prediction_time) → adapter_id` for every evaluation step.
4. **Call `run_evaluation`** with the short-context config, the generated assignment_df, and the pipeline.

### Time-window layout
```
|---- long history ----|-- short context --|-- forecast horizon --|
                       ^                   ^
                       short_start         prediction_time (forecast origin)
```

- **Long history** is consumed only by the hypernetwork; it is NOT passed to evaluation.
- **Short context** is what Chronos-2 sees during inference (set via `evaluation.history_length_steps`).
- Short context may overlap with long history (controlled by `long_history_end_offset_steps`).

### Fixed vs. rolling long history
- **Fixed** (`rolling_long_history: false`): one absolute time window for all prediction steps → one adapter per sensor (efficient: hypernetwork runs once per sensor).
- **Rolling** (`rolling_long_history: true`): the long-history window moves with the evaluation rolling window → one adapter per sensor per step.

### Context encoder
`ChronosContextEncoder` wraps a frozen Chronos model (Chronos-2 or Chronos-Bolt). It provides two methods:
- `encode_last_hidden(context)`: returns only the last hidden state `[batch, num_patches, d_model]`. Used by the training loop. Supports both Chronos-Bolt (uses built-in `encode()`) and Chronos-2 (manual block loop).
- `encode_intermediates(context)`: returns per-layer hidden states `[batch, num_layers, num_patches, d_model]`. Retained for backward compatibility (orchestration layer). Only works with Chronos-2.

Supports batched encoding to control GPU memory.

### Hypernetwork interface (expected)
```python
# Input:  context_hidden_states [batch, num_patches, 768]
# Output: dict[module_short_name, {"A": [batch, 12, r, d_in], "B": [batch, 12, d_out, r]}]
lora_dict = hypernetwork(hidden_states)
```
Module short names: `"q"`, `"k"`, `"v"`, `"o"`.

### Adapter saving
`save_adapter_to_disk` converts one sensor's slice of the hypernetwork output into a PEFT adapter directory containing `adapter_model.safetensors` and `adapter_config.json`. Modules not covered by the hypernetwork (GroupSelfAttention layer.1, output_patch_embedding) are filled with zero weights.

### Public orchestration API
- `run_orchestration(cfg)` → `(horizon_metrics, runtime_stats)`

### CLI entrypoint
```bash
uv run python -m src.orchestration.main \
    orchestration.checkpoint_path=checkpoints/my_hypernet.pt \
    orchestration.rolling_long_history=false \
    orchestration.long_history_start_date="2017-04-01"
```

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
| `train/` | Training | `train/loss`, `train/lr`, `train/epoch_loss`, `train/epoch_time` |

### Adding WandB to a new layer
1. Call `init_wandb(cfg, group="your_layer")` at the top of your CLI entrypoint.
2. Inside your loop, check `if wandb.run is not None:` and call `wandb.log(...)`.
3. Call `finish_wandb()` at the end of your CLI entrypoint.
4. If the function may be called from an outer layer, do **not** init/finish inside it — just log when `wandb.run` is active.

---