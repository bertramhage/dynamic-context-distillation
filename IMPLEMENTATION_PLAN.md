# Implementation Plan: Distilling Context into Parameters for Chronos-2

## Project Overview

We're adapting the Doc-to-LoRA (D2L) hypernetwork framework to time-series: a Perceiver-based hypernetwork reads a long historical context window and produces a LoRA adapter for Chronos-2 in one forward pass, so the model can forecast with a much shorter context window while retaining (or even exceeding) the accuracy of the full-context baseline.

Repository reference note: the full Doc-to-LoRA repository is included in this codebase under `doc-to-lora/` for direct implementation reference during reuse/adaptation steps.
The markdown implementation design and methodology reference is available at `doc-to-lora/methodology.md`.

---

## Critical Discovery: Chronos-2 Architecture

**In practice, Chronos-2 is encoder-only.** Each of its 12 blocks (for Base/120M) contains:

1. **TimeSelfAttention** — attends along the time axis (uses RoPE)
2. **GroupSelfAttention** — attends across time-series within the same group (no RoPE)
3. **FeedForward** — standard MLP

It takes patch-based input (8192 context values → ~512 patches), and produces quantile-based probabilistic forecasts (21 quantiles by default) through an output patch embedding head.

**LoRA support is already built into Chronos-2's codebase** via HuggingFace PEFT. Default targets:
- `self_attention.q`, `self_attention.k`, `self_attention.v`, `self_attention.o` (in all 12 blocks)
- `output_patch_embedding.output_layer`

This is excellent news — we don't need to figure out where to inject LoRA; the Chronos team already chose the modules.

---

## What We Reuse vs. What We Build

### From D2L (reuse heavily)
The full D2L codebase is available locally under `doc-to-lora/`, so reuse/adaptation in this plan can reference concrete source files directly.

| Component | Reuse? | Adaptation needed |
|-----------|--------|-------------------|
| Perceiver aggregator | **Yes** — core architecture | Change input dim from text embedding dim → Chronos hidden dim (768) |
| HyperLoRA generator (ResMLPBlocks + EinMix heads) | **Yes** — core architecture | Retarget to Chronos-2 module names and dimensions |
| LoRA forward injection (`lora_layer.py`) | **Partially** | Rewrite for Chronos-2's `nn.Linear` layers (different naming: `self_attention.q` not `self_attn.q_proj`) |
| LoRA merger | **Maybe later** | Only needed if we chunk the history window |
| Text context encoder | **No** | Replace entirely with time-series context encoder |
| Training loop (distillation) | **Partially** | Keep KL divergence (on output quantile vectors); adapt data pipeline |
| Config system | **Partially** | Simplify for our use case |

### From mobility-baselines (reuse heavily)
| Component | Reuse? | Adaptation needed |
|-----------|--------|-------------------|
| PEMS-BAY data loading (`utils.py`) | **Yes** | Minor: also expose raw numpy arrays, not just Chronos DataFrames |
| Hydra config system | **Yes** | Add our own configs for hypernetwork |
| Evaluation metrics (MAE, MAPE, RMSE, coverage) | **Yes** |  |
| Chronos-2 inference pipeline | **Yes** | Extend to support LoRA-injected model |
| Sliding window evaluation loop | **Yes** | Adapt to compare teacher vs. student |

### New code we build
| Component | Description |
|-----------|-------------|
| **TimeSeriesContextEncoder** | Encodes long history into feature representations using the frozen Chronos-2 encoder |
| **Adapted HyperLoRA** | Wired to Chronos-2's architecture: 12 layers × 4 attention modules + output head |
| **LoRA injection for Chronos-2** | Map generated weights → monkey-patch Chronos-2 Linear layers |
| **Teacher-student training loop** | Teacher = full-context Chronos-2; Student = short-context + LoRA Chronos-2 |
| **Experiment scripts** | Orchestrate training, evaluation, ablations |

---

## Architecture Dimensions (Chronos-2-Base)

These numbers drive the hypernetwork's output dimensions:

| Parameter | Value |
|-----------|-------|
| d_model (hidden) | 768 |
| num_heads | 12 |
| d_kv (per head) | 64 |
| d_ff | 3072 |
| num_layers | 12 |
| LoRA rank r | 8 (default) |

For each attention module (q, k, v, o) with LoRA rank 8:
- **A matrix**: (r × d_model) = (8 × 768) = 6,144 params
- **B matrix**: (d_model × r) = (768 × 8) = 6,144 params
- **Per layer**: 4 modules × 2 matrices × 6,144 = 49,152 params
- **All 12 layers**: 12 × 49,152 = **589,824 params**
- **Plus output head**: 2 × (8 × 768) ≈ 12,288 params
- **Total LoRA params**: ~602K (very lightweight)

The hypernetwork needs to produce these ~600K params in a single forward pass.

---

## RQ1 & RQ2 Implementation Plan

### Phase 0: Environment & Data Setup

**Goal**: Get everything running — data loaded, Chronos-2 producing predictions, metrics computed.

1. **Set up the project repository**
   - Create a clean repo structure merging relevant code from D2L and mobility-baselines
   - Install dependencies: `chronos-forecasting`, `peft`, `einops`, `hydra`, etc.

2. **Data pipeline**
  - Reuse `mobility-baselines/Chronos-2-evaluation/utils.py` for PEMS-BAY loading
  - Create a `PemsBayDataset` class that provides:
     - Long context window (teacher input): e.g. 2016 timesteps (1 week)
     - Short context window (student input): e.g. 256 or 512 timesteps
     - Ground truth future values for evaluation
  - Use the benchmark split for PEMS-BAY (5-minute granularity, ~6 months of data, 325 sensors), and reserve the tail of the training period as validation for early stopping and hyperparameter tuning.
  - **Rolling-window sample construction (explicit)**:
    - Training data is constructed from overlapping long-context windows per station, rather than one static window per station.
    - Long-context windows are advanced through time with a controlled stride to balance sample diversity and temporal redundancy.
    - For each long-context window, multiple forecast origins are evaluated by shifting the short-context/forecast pair forward within the admissible time range.
    - Chronology is strictly causal: forecast targets occur strictly after the short context and after the information used to generate the adapter (no overlap/data leakage).
  - **D2L-style multi-query analogue**:
    - One adapter is generated from a given long context and optimized to perform across multiple forecast origins associated with that same context.
    - This is the time-series analogue of multiple queries per context: the adapter must encode regime-level structure, not a single forecast moment.
  - PEMS-BAY provides enough temporal coverage to yield a large effective sample pool across stations, window placements, and forecast origins, so training should use densely overlapping windows rather than a tiny fixed set.

3. **Reproduce the baseline**
  - Run Chronos-2-Base zero-shot on PEMS-BAY with the full context window
   - Verify metrics match the benchmark paper (MAE, MAPE, RMSE, coverage)
   - This becomes the **teacher baseline** and our RQ1 target to match

---

### Phase 1: Time-Series Context Encoder

**Goal**: Build the module that converts a long history window into a sequence of embeddings the Perceiver can cross-attend to.

**Design decision**:

Run the long history through the frozen Chronos-2 encoder itself and extract the hidden states from the last layer. This gives us a sequence of (num_patches, 768) embeddings that already capture temporal patterns. Analogous to D2L's "EarlyExit" context encoder strategy.

```
Long history (2016 steps) → Chronos-2 patch embedding → Chronos-2 encoder (frozen)
                          → hidden states [num_patches × 768]
                          → input to Perceiver aggregator
```

**Implementation**:
- Add a `TimeSeriesContextEncoder` class that wraps a frozen Chronos-2 model
- Forward pass: tokenize/patch the long context, run through encoder, return hidden states
- No gradients through this encoder (fully frozen)

---

### Phase 2: Adapt Perceiver + HyperLoRA for Chronos-2 (Day 4–7)

**Goal**: Wire the Perceiver aggregator and LoRA generation heads to produce Chronos-2-compatible LoRA weights.

**2a. Perceiver Aggregator (mostly reuse D2L)**
- Input: context encoder output `[num_patches, 768]`
- The Perceiver cross-attends between learned latent queries and the context
- Output: compressed latent representation
- **Changes from D2L**: Input dimension changes from text embedding dim to 768. The aggregator architecture itself is model-agnostic — it just compresses variable-length input to fixed-size latents.

**2b. HyperLoRA Generator (adapt D2L)**
- Input: Perceiver output latents
- Output: dictionary of LoRA A and B matrices for each target module
- **Changes from D2L**:
  - Target modules: `self_attention.q`, `self_attention.k`, `self_attention.v`, `self_attention.o` (instead of `q_proj`, `k_proj`, etc.)
  - Dimensions: d_model=768, num_layers=12
  - D2L targets 32 layers × 7 modules; we target 12 layers × 4–5 modules (much smaller)

**2c. LoRA Injection for Chronos-2 (new code, inspired by D2L)**
- D2L monkey-patches `nn.Linear.forward` with a partial function that adds the LoRA delta
- We do the same, but navigate Chronos-2's module tree:
  - `model.encoder.block[i].layer[0].self_attention.q` (TimeSelfAttention)
  - `model.encoder.block[i].layer[1].self_attention.q` (GroupSelfAttention)
  - Start with targeting TimeSelfAttention only (`layer[0]`) — this is where temporal context is processed. Both attention types share the same module names (`self_attention.q/k/v/o`), so the injection code must explicitly index `layer[0]` to avoid also patching GroupSelfAttention.

**Key implementation detail**: D2L's `lora_forward` expects shape `[n_ctx, r, d_in]` for A and `[n_ctx, r, d_out]` for B, with `n_ctx` being the batch dimension of different contexts. For our use case, each context is a station's history, so `n_ctx` = batch of stations.

#### What to keep (and modify) from D2L
1. Reuse (Mostly As-Is)
These files operate purely on PyTorch tensors and don't care whether the data came from text or time-series.
- **lora_layer.py**: The most important file to keep. This contains the custom standard Linear overrides (DynamicLoraLinear, etc.) that allow you to inject LoRA A and B weights at runtime during the forward pass. You can use this directly to wrap Chronos-2's internal transformer layers.
- **aggregator.py**: If your time-series context encoder outputs a sequence of embeddings (e.g., [batch, sequence_length, hidden_dim]), this file contains the logic (mean pooling, attention blocks) to distill that sequence down into a single [batch, hidden_dim] conditioning vector.

2. Reuse With Modifications
These files contain the right mathematical ideas but are currently tangled up with HuggingFace text-model assumptions (like AutoModelForCausalLM).
**hypernet.py**:
- Keep: The projection heads (the actual nn.Linear layers that scale up the conditioning vector into the massive flattened LoRA matrices).
- Modify: Strip out ModulatedPretrainedModel. Replace it with a simpler generator class that just takes a tensor and returns a dictionary of weights.
**utils.py**:
Keep: Helpful PyTorch utilities like compile_linear, log_num_train_params, or basic YAML loading.

---

### Phase 3: Teacher-Student Training Loop (Day 7–11)

**Goal**: Train the hypernetwork to produce LoRA adapters that make the short-context student match the full-context teacher.

**Teacher pipeline** (run once per batch, cache results):
```
Long history (2016 steps) → Chronos-2 (frozen, full context) → teacher_output
```

**Student pipeline** (trained via hypernetwork):
```
Long history → Context Encoder (frozen Chronos-2) → Perceiver → HyperLoRA → LoRA weights
Short history (256 steps) + LoRA weights → Chronos-2 (frozen + LoRA) → student_output
```

**Loss function**:
- **KL divergence** between teacher and student output quantile vectors (21 quantiles per forecast step). Chronos-2's output is already a distribution over quantiles, so KL is a natural fit and preserves the probabilistic nature of the predictions.

**Training details**:
- Only the hypernetwork parameters are trained (Perceiver + ResMLPBlocks + EinMix heads)
- Batches are organized around long-context instances, with each instance contributing multiple forecast-origin losses.
- The training objective aggregates KL divergence across forecast origins for the same context, so learning pressure is placed on context-level generalization.
- This design reduces over-specialization to any single origin and promotes query-independent pattern encoding.
- Adam optimizer.
- Estimated hypernetwork size: ~5–15M params (much smaller than the 120M base model)

**What to watch for**:
- Teacher outputs should be cached to avoid running the full-context model every step
- Gradient only flows through: student Chronos-2 (LoRA path only) → HyperLoRA → Perceiver
- The base Chronos-2 weights are frozen; only the LoRA delta path carries gradients

---

### Phase 4: Evaluation & RQ1/RQ2 Analysis (Day 11–14)

**Goal**: Rigorously evaluate whether the student matches or exceeds the teacher.

**RQ1 — Context compression**:
- Compare student (short context + LoRA) vs. teacher (full context) vs. baseline (short context, no LoRA)
- Metrics: MAE, MAPE, RMSE, CRPS, coverage, IQR width
- Report inference latency and memory usage
- Ablations:
  - Vary short context length
  - Vary long context length
  - Vary LoRA rank
  - Vary which modules get LoRA: q/v only vs. q/k/v/o vs. + output head
  - TimeSelfAttention only vs. both attention types

**RQ2 — Student exceeding teacher**:
- The D2L paper shows this can happen when long contexts contain noise
- Traffic data has repetitive diurnal/weekly patterns — a LoRA adapter might compress these more efficiently than raw attention
- Compare on different forecast horizons (15, 30, 45 min)
- Analyze: on which stations/time periods does the student beat the teacher? Is it correlated with history noise level?

**Evaluation reuses mobility-baselines' sliding window approach** — same horizons, same metrics. We can try different horizons as long as we compare the same horizons (full context) between the benchmark and our approach.

---

### Phase 5: RQ3 — Ground-Truth Supervision Instead of Teacher Distillation

**RQ3**:
"If we replace the teacher target in the KL divergence with
ground-truth labels, does performance improve beyond the
baseline?"

**Goal**: Evaluate whether training the hypernetwork directly against observed future values (instead of teacher outputs) improves downstream forecasting quality.

**Design change**:
- Keep the same architecture and data construction (long context -> hypernetwork -> LoRA, short context -> student Chronos-2).
- Replace the teacher-target objective with a ground-truth-target objective.
- Preserve the same train/validation split and rolling-window multi-query setup to keep comparisons fair.

**Training pipeline (RQ3 variant)**:
```
Long history -> Context Encoder (frozen) -> Perceiver -> HyperLoRA -> LoRA weights
Short history + LoRA weights -> Chronos-2 (frozen + LoRA) -> student_output
student_output vs ground_truth_targets -> supervised loss
```

**Loss**:
- Use **CRPS (Continuous Ranked Probability Score)** between student predictive distributions and ground-truth targets.

**Implementation tasks**:
- Add a training-mode switch for objective target:
  - `target_mode: teacher | ground_truth`
- In the trainer loop:
  - Skip teacher forward pass when `target_mode=ground_truth`.
  - Compute supervised **CRPS** loss from `batch.targets` and student quantile predictions.
- Remove teacher-context construction in the RQ3 path to reduce compute and memory overhead.

**Evaluation protocol for RQ3**:
- Compare three systems under identical evaluation settings:
  1. Short-context baseline (no LoRA).
  2. Distilled student (RQ1/RQ2 teacher-target training).
  3. Ground-truth-supervised student (RQ3).
- Report MAE, MAPE, RMSE, CRPS, coverage, IQR width, and runtime/memory.
- Use the same horizons and test windows as RQ1/RQ2.

**Primary success criterion**:
- RQ3 student outperforms the short-context baseline on core metrics (MAE/RMSE) without unacceptable degradation of probabilistic calibration (coverage/IQR/CRPS).

**Analysis focus**:
- Where does direct supervision help most (horizon length, station type, traffic regime)?
- Does RQ3 beat distillation consistently, or only in specific regimes?
- Does CRPS-trained supervision improve both point accuracy and uncertainty calibration relative to distillation?

---

### Phase 6: RQ2 — Station Generalization (Unseen Sensors)

**RQ2**:
"Does the trained hypernetwork generalize to producing effective LoRA weights for stations not seen during training?"

**Goal**: Evaluate whether the hypernetwork learns a general time-series-to-adapter mapping rather than memorizing station-specific patterns.

**Why this matters**:
If the hypernetwork only works for stations it trained on, it is just an expensive lookup table. The interesting result is whether a hypernetwork trained on a small subset of stations produces useful adapters for the remaining majority — i.e. whether the mapping from "temporal context → LoRA weights" generalizes across spatial locations.

#### Design

**Station split**:
PEMS-BAY has 325 sensors. We train on a small subset and evaluate on the rest:
- **Train stations** (~20%, ~65 sensors): used during hypernetwork training.
- **Held-out stations** (~80%, ~260 sensors): never seen during training, used only for evaluation.

Training on 20% and evaluating on 80% is a deliberately harsh test. If the hypernetwork generalizes under this regime, it demonstrates that a small representative sample of stations is sufficient to learn the adapter-generation function for the full network.

The temporal train/val/test split remains unchanged — this is an orthogonal axis.

**Split construction**:
- Deterministic split using a configurable seed (default: `station_split_seed=42`).
- Stratified by mean traffic volume: bin sensors into quartiles by average speed, sample the train fraction proportionally from each bin. This prevents the train set from being accidentally all-highway or all-ramp.
- The split is logged to WandB and saved alongside the checkpoint as `station_split.json` for reproducibility.

#### Implementation tasks

**6a. Station split infrastructure**

1. Add config keys under `training:`:
   ```yaml
   training:
     station_holdout:
       enabled: false                 # default off for backward compatibility
       train_fraction: 0.2            # fraction of stations used for training
       station_split_seed: 42
       stratify_by_mean: true
   ```

2. Add a `split_stations()` utility in `src/utils/utils.py`:
   - Input: list of sensor IDs, DataFrame, train fraction, seed, stratify flag.
   - Output: `(train_station_ids, holdout_station_ids)`.
   - When `stratify_by_mean=true`: compute per-station mean target value, assign to quartile bins, sample train fraction from each bin.
   - When `stratify_by_mean=false`: simple random split.

3. Wire into `src/training/main.py`:
   - When `station_holdout.enabled=true`, call `split_stations()` before building datasets.
   - Pass `sensor_ids=train_station_ids` to `_build_dataset()` for both train and val datasets (`HypernetTrainingDataset` already accepts `sensor_ids`).
   - Save the split to `{checkpoint_dir}/station_split.json`.
   - Log `n_train_stations`, `n_holdout_stations` to WandB config.

**6b. Evaluation on held-out stations**

4. Extend orchestration to support station-scoped evaluation:
   - Add config key `orchestration.station_split_path` (optional path to `station_split.json`).
   - When provided, `run_orchestration` loads the split and runs evaluation twice:
     1. On train-station subset only.
     2. On held-out-station subset only.
   - Both runs use the same hypernetwork checkpoint.

5. Add a batch job script `batch_jobs/eval_station_generalization.sh`:
   - Trains with `station_holdout.enabled=true`.
   - Evaluates on train stations and held-out stations separately.
   - Logs both metric sets to WandB under distinct run tags (`station_set=train`, `station_set=holdout`).

**6c. Analysis**

- **Primary comparison**: held-out station metrics vs. train station metrics (paired by horizon). If the gap is small, generalization holds.
- **Secondary comparison**: held-out station metrics vs. short-context baseline (no LoRA) on the same held-out stations — does the hypernetwork still help on unseen stations?
- **Metric degradation ratio**: `(held_out_MAE - train_MAE) / train_MAE` — quantifies the generalization gap.
- **Per-station analysis**: scatter plot of per-station MAE improvement (LoRA vs. baseline) for train vs. held-out stations. Overlapping distributions = strong generalization.

**Experimental matrix**:

| Experiment | Train stations | Eval stations | Target mode |
|------------|---------------|---------------|-------------|
| RQ2-a      | 20% (65)      | 20% (65)      | best from RQ1/RQ3 |
| RQ2-b      | 20% (65)      | 80% (260)     | best from RQ1/RQ3 |
| RQ2-c (ref)| 100% (325)    | 100% (325)    | best from RQ1/RQ3 |

RQ2-a vs. RQ2-b is the core generalization test. RQ2-c provides the ceiling.

---

### Phase 7: RQ5 — Cross-Dataset Generalization

**RQ5**:
"Does the hypernetwork generalize to generating LoRA weights for other datasets without retraining?"

**Goal**: Test whether a hypernetwork trained on PEMS-BAY produces useful LoRA adapters when applied to entirely different traffic datasets, zero-shot (no retraining, no fine-tuning).

**Why this matters**:
RQ2 tests spatial generalization (unseen stations, same network). RQ5 tests distributional generalization: can the learned mapping from "temporal context → LoRA weights" transfer across different cities, road networks, and traffic regimes? If yes, this suggests the hypernetwork captures something fundamental about compressing temporal patterns into adapter weights, not just PEMS-BAY-specific structure.

#### Dataset selection

We restrict to same-domain (traffic) datasets to isolate distributional transfer from domain transfer. Candidate target datasets, ordered by compatibility:

| Dataset | Location | Measure | Period | Freq | Sensors | Source |
|---------|----------|---------|--------|------|---------|--------|
| **METR-LA** | Los Angeles | traffic speeds | 2012-03 to 2012-06 | 5 min | 207 | Li et al., 2018 |
| **PeMSD4** | California | traffic volumes | 2018-01 to 2018-02 | 5 min | 307 | Choi et al., 2022 |
| **PeMSD7(M)** | California | traffic speeds | 2016-04 to 2016-06 | 5 min | 228 | Yu et al., 2018 |
| **Seattle loop** | Seattle | traffic speeds | 2015-11 to 2015-12 | 5 min | 323 | Yang et al., 2021 |

**Primary transfer target: METR-LA**
- Same measurement (traffic speeds), same frequency (5 min), same country, different city.
- Closest distributional match to PEMS-BAY — the fairest first test.

**Secondary transfer target: PeMSD4**
- Same frequency and similar sensor count, but measures *volume* instead of *speed*.
- Tests whether the hypernetwork's context-compression is robust to a different traffic variable.

**Tertiary (if time permits): PeMSD7(M) or Seattle loop**
- Further geographical diversity (different California regions, different state).

#### Implementation tasks

**7a. Multi-dataset support**

1. Add dataset configs:
   - `conf/dataset/METR-LA.yaml`:
     ```yaml
     dataset: "METR-LA"
     start_date: "2012-03-01"
     start_test_day: null
     n_test_days: null
     proportion_test: 0.2
     freq: 5
     target: "target"
     future_covariates: null
     past_covariates: null
     prediction_length: 12
     id_column: "item_id"
     timestamp_column: "timestamp"
     horizons: [15, 30, 60]
     decimal_precision: 2
     metrics_to_show: [MAE, MAPE, RMSE, COVERAGE, IQR_MEAN]
     ```
   - `conf/dataset/PeMSD4.yaml`, `conf/dataset/PeMSD7M.yaml`, `conf/dataset/Seattle-Loop.yaml` similarly.

2. Extend `src/utils/utils.py` `load_dataset()`:
   - METR-LA uses the same HDF5 format as PEMS-BAY (both from the DCRNN paper). The loader should be nearly identical with a different file path and date range.
   - PeMSD4/PeMSD7(M): typically distributed as `.npz` with adjacency matrix. Add a loading path that reads the `.npz`, pivots to long-format `(item_id, timestamp, target)`.
   - Seattle loop: CSV format, straightforward column mapping.
   - All loaders must produce the same long-format DataFrame schema.

3. Add a `data/` download script or instructions for obtaining each dataset.

**7b. Zero-shot cross-dataset evaluation**

4. Add an orchestration mode for cross-dataset inference:
   - Config key: `orchestration.eval_dataset_cfg` (optional, defaults to the training dataset config).
   - When set to a different dataset (e.g., `dataset/METR-LA`), orchestration:
     - Loads evaluation data from the specified dataset config.
     - Loads the hypernetwork checkpoint trained on PEMS-BAY.
     - The frozen context encoder processes the new dataset's time series normally — it's a pretrained model, input-agnostic.
     - The hypernetwork generates LoRA weights from the context encoder output — no retraining.

5. Add a batch job script `batch_jobs/eval_cross_dataset.sh`:
   - Takes the best PEMS-BAY checkpoint.
   - Runs orchestration + evaluation on each target dataset.
   - Logs metrics to WandB with `dataset` as a tag.

**7c. Analysis**

- **Primary comparison (per target dataset)**:

  | System | Description |
  |--------|-------------|
  | Baseline | Chronos-2, short context, no LoRA |
  | Full context | Chronos-2, full context, no LoRA |
  | PEMS-BAY HyperNet (zero-shot) | Short context + LoRA from PEMS-BAY-trained hypernetwork |

  The zero-shot hypernetwork must beat the short-context baseline to claim positive transfer. Matching the full-context baseline would be a strong result.

- **Metric degradation**: compare the improvement margin (LoRA vs. short-context baseline) on each target dataset vs. PEMS-BAY. Smaller margin is expected — the question is sign, not magnitude.

- **Speed vs. volume**: if METR-LA (speeds) transfers well but PeMSD4 (volumes) does not, the hypernetwork's learned representation is speed-specific rather than general-traffic.

- **Per-station breakdown**: on METR-LA, do the LoRA adapters help uniformly, or only for stations whose patterns resemble PEMS-BAY?

---

### Phase 8: RQ4 — Encoding Non-Time-Series Data into LoRA

**RQ4**:
"Can we encode non-time-series data into the LoRA — such as weather or neighbouring station data — to improve TS-FM performance?"

**Goal**: Extend the hypernetwork's context encoder to incorporate non-time-series auxiliary data (weather covariates), so that the generated LoRA adapter internalizes information that Chronos-2 cannot natively access from the short context alone.

**Why this matters**:
Traffic patterns are influenced by weather (rain → slower speeds, snow → closures). The current hypernetwork sees only the target station's historical speed values. If we feed weather data into the context encoder, the hypernetwork can encode weather-conditioned patterns into the LoRA weights. At inference time, the student model runs with just the short time-series context + LoRA — the weather information is "baked into" the adapter parameters. This is a form of cross-modal knowledge internalization.

#### Key insight: the D2L VLM analogy

D2L (Section 5.2) demonstrated that a VLM (gemma-3-4b-it) could serve as the context encoder for a text-only target LLM (gemma-2-2b-it). The VLM processes both text and images; the Perceiver maps VLM activations to LoRA weights; the text-only LLM + LoRA then achieves 75% on image classification — despite never seeing images itself. The visual information was internalized into the LoRA weights via the VLM encoder bridge.

Our analogy is direct:

| D2L VLM experiment | Our RQ4 |
|---|---|
| VLM sees text + images | Covariate-aware encoder sees time series + weather |
| Perceiver maps VLM activations → LoRA | Perceiver maps covariate-aware activations → LoRA |
| Text-only LLM + LoRA classifies images | Chronos-2 (univariate) + LoRA reflects weather effects |
| VLM and LLM are different models | Covariate-aware encoder and student share same architecture family |

The student model never sees weather data at inference — it's a standard univariate Chronos-2 forward pass. But the LoRA weights carry the weather signal because they were generated from a covariate-aware context representation.

#### Architecture design

Chronos-2 natively supports future covariates through its `_prepare_patched_future()` pipeline:
1. Covariate values are normalized using the context's instance-norm statistics (loc/scale).
2. Patched into `[batch, num_patches, patch_size]`.
3. Concatenated as `[time_encoding, covariate_values, covariate_mask]`.
4. Embedded through the same `input_patch_embedding` (ResidualBlock) as context patches.
5. Context and covariate embeddings are concatenated and jointly processed by the encoder's TimeSelfAttention and GroupSelfAttention layers.

We exploit this by using **Chronos-2 itself as a covariate-aware context encoder**: feed it both the long history and the weather covariates, run the full encoder, and extract hidden states that now encode both temporal and weather information. The Perceiver then compresses these richer hidden states into LoRA weights.

```
Training (covariate-aware context encoding):

Long history + Weather covariates
    → Chronos-2 encoder (frozen, with future_covariates)
    → hidden_states [B, num_context_patches + num_covariate_patches, d_model]
    → Perceiver → HyperLoRA → LoRA weights

Inference (standard univariate):

Short history + LoRA weights → Chronos-2 (frozen + LoRA) → forecast
```

**Key design choices**:

- **Context encoder model**: use Chronos-2 (not Chronos-Bolt-Mini) as the context encoder when covariates are involved, since Chronos-2 has native covariate support (`_supports_future_covariates = True`). This is a different model than the current default (`amazon/chronos-bolt-mini`). The Perceiver input dimension changes from 384 to 768 — a config change, not an architectural one.

- **Which hidden states**: extract the full encoder output including both context patches and covariate patches. The covariate patch hidden states carry weather-conditioned representations that the Perceiver should attend to. Alternatively, extract only context-patch hidden states (which already attended to covariate patches via TimeSelfAttention) — this keeps the Perceiver input length unchanged.

- **Weather as "future covariates"**: Chronos-2 treats covariates as known future values. For our purpose, weather during the forecast horizon is "known" (weather forecasts are available) and weather during the historical period is observed. We can encode historical weather as covariates aligned to the long context, and forecast-horizon weather as future covariates.

#### Implementation tasks

**8a. Weather data pipeline**

1. **Obtain and preprocess historical weather data**:
   - Source: NOAA ISD (Integrated Surface Database) or Visual Crossing for the San Francisco Bay Area, covering the PEMS-BAY period (2017-01 to 2017-05).
   - Variables: temperature, precipitation, wind speed, visibility.
   - Temporal alignment: resample to 5-minute frequency (or hourly, then broadcast to 5-min).
   - Output: `data/weather/bay_area_weather.parquet` with columns `(timestamp, temperature, precipitation, wind_speed, visibility)`.

2. **Weather loading utility** (`src/utils/weather.py`):
   - `load_weather(path, start, end) → pd.DataFrame` — loads and time-slices.
   - `align_weather_to_context(weather_df, context_timestamps) → torch.Tensor` — aligns weather features to a time-series context window, returning `[T, n_features]`.

3. **Config keys**:
   ```yaml
   training:
     weather_context:
       enabled: false
       weather_data_path: "data/weather/bay_area_weather.parquet"
       features: [temperature, precipitation, wind_speed, visibility]
   ```

**8b. Covariate-aware context encoder**

4. **Extend `ChronosContextEncoder`** to accept covariates:
   - Add method `encode_last_hidden_with_covariates(context_tensor, covariate_tensor, covariate_mask=None)`:
     - Calls Chronos-2's `encode()` with `future_covariates=covariate_tensor`.
     - Returns hidden states `[B, num_context_patches + num_covariate_patches, d_model]`.
   - This reuses Chronos-2's native covariate encoding: instance normalization, patching, embedding through `input_patch_embedding`, and joint attention with context patches.
   - Only supported when the underlying model is Chronos-2 (not Bolt), since only Chronos-2 has `_supports_future_covariates = True`.

5. **Handle the encoder model switch**:
   - When `weather_context.enabled=true`, override `context_encoder_model` to `amazon/chronos-2` (or allow explicit config).
   - The Perceiver `d_input` changes from 384 (Bolt-Mini) to 768 (Chronos-2). This is already a config parameter in `HyperLoRA`.
   - Note: Chronos-2 as context encoder is heavier than Bolt-Mini (~120M vs ~20M params). This is the cost of covariate support. Both are frozen, so no training memory impact beyond activation memory.

**8c. Dataset and training loop changes**

6. **Extend `HypernetTrainingDataset`**:
   - When `weather_context.enabled=true`, each sample also extracts the weather features aligned to the long-context window.
   - New output field: `weather_context: torch.Tensor  # [long_ctx_len, n_features]` (or the forecast-horizon slice, depending on how we frame it for Chronos-2's covariate API).
   - Collation pads weather tensors identically to time-series contexts.

7. **Extend training loop** (`trainer.py`):
   - Context encoding step changes from:
     ```python
     ctx_features = self.context_encoder.encode_last_hidden(long_ctx)
     ```
     to:
     ```python
     if self.weather_enabled:
         ctx_features = self.context_encoder.encode_last_hidden_with_covariates(
             long_ctx, weather_covariates
         )
     else:
         ctx_features = self.context_encoder.encode_last_hidden(long_ctx)
     ```
   - The rest of the pipeline (Perceiver → HyperLoRA → LoRA injection → student forward) is unchanged. The Perceiver handles variable-length input natively via cross-attention.

**8d. Orchestration changes**

8. **Extend `run_orchestration`**:
   - When weather context is enabled, the orchestration layer must also extract weather data for the long-context window when generating adapters.
   - Load weather data once at startup, then slice per evaluation step.
   - Pass weather covariates to the context encoder before the hypernetwork call.

**8e. Evaluation protocol**

- **Ablation matrix**:

  | Experiment | Context encoder | Weather covariates | Notes |
  |------------|----------------|-------------------|-------|
  | RQ4-base   | Chronos-Bolt-Mini (d=384) | No  | Baseline (RQ1/RQ3 best) |
  | RQ4-c2enc  | Chronos-2 (d=768)         | No  | Isolate effect of larger context encoder |
  | RQ4-weather| Chronos-2 (d=768)         | Yes | Full weather-aware context |

  RQ4-c2enc is critical: it controls for the context encoder upgrade. The weather signal is only credible if RQ4-weather beats RQ4-c2enc, not just RQ4-base.

- **Weather-conditioned analysis**:
  - Stratify evaluation by weather regime: clear days vs. rainy days vs. extreme events.
  - The hypothesis: weather-aware LoRA helps most during adverse weather, when the univariate baseline has no signal for the disruption.
  - Per-station analysis: stations in weather-exposed locations (bridges, mountain passes) should benefit more.

- **Computational cost**: report the additional latency from using Chronos-2 as context encoder vs. Bolt-Mini (context encoding happens once per adapter, not per inference step, so the cost is amortized).

**8f. Batch job**

9. Add `batch_jobs/eval_weather_context.sh` that sweeps the ablation matrix above.
