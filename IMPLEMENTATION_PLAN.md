# Implementation Plan: Distilling Context into Parameters for Chronos-2

## Project Overview

We're adapting the Doc-to-LoRA (D2L) hypernetwork framework to time-series: a Perceiver-based hypernetwork reads a long historical context window and produces a LoRA adapter for Chronos-2 in one forward pass, so the model can forecast with a much shorter context window while retaining (or even exceeding) the accuracy of the full-context baseline.

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
| Component | Reuse? | Adaptation needed |
|-----------|--------|-------------------|
| Perceiver aggregator | **Yes** — core architecture | Change input dim from text embedding dim → Chronos hidden dim (768) |
| HyperLoRA generator (ResMLPBlocks + EinMix heads) | **Yes** — core architecture | Retarget to Chronos-2 module names and dimensions |
| LoRA forward injection (`lora_layer.py`) | **Partially** | Rewrite for Chronos-2's `nn.Linear` layers (different naming: `self_attention.q` not `self_attn.q_proj`) |
| LoRA merger | **Maybe later** | Only needed if we chunk the history window |
| Text context encoder | **No** | Replace entirely with time-series context encoder |
| Training loop (distillation) | **Partially** | Swap KL divergence for MSE; adapt data pipeline |
| Config system | **Partially** | Simplify for our use case |

### From mobility-baselines (reuse heavily)
| Component | Reuse? | Adaptation needed |
|-----------|--------|-------------------|
| PeMSD7 data loading (`utils.py`) | **Yes** | Minor: also expose raw numpy arrays, not just Chronos DataFrames |
| Hydra config system | **Yes** | Add our own configs for hypernetwork |
| Evaluation metrics (MAE, MAPE, RMSE, coverage) | **Yes** |  |
| Chronos-2 inference pipeline | **Yes** | Extend to support LoRA-injected model |
| Sliding window evaluation loop | **Yes** | Adapt to compare teacher vs. student |

### New code we build
| Component | Description |
|-----------|-------------|
| **TimeSeriesContextEncoder** | Encodes long history into feature representations (using frozen Chronos-2 encoder itself or a simpler MLP/Conv approach) |
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
   - Reuse `mobility-baselines/Chronos-2-evaluation/utils.py` for PeMSD7 loading
   - Create a `PeMSD7Dataset` class that provides:
     - Long context window (teacher input): e.g. 2016 timesteps (1 week)
     - Short context window (student input): e.g. 256 or 512 timesteps
     - Ground truth future values for evaluation
   - Implement train/val/test splits (by time, following the benchmark paper)

3. **Reproduce the baseline**
   - Run Chronos-2-Base zero-shot on PeMSD7 with the full context window
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
  - Start with targeting TimeSelfAttention only (this is where temporal context is processed).

**Key implementation detail**: D2L's `lora_forward` expects shape `[n_ctx, r, d_in]` for A and `[n_ctx, r, d_out]` for B, with `n_ctx` being the batch dimension of different contexts. For our use case, each context is a station's history, so `n_ctx` = batch of stations.

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
- We start with **MSE** between teacher and student output logits/quantile predictions, as it's simple.

**Training details**:
- Only the hypernetwork parameters are trained (Perceiver + ResMLPBlocks + EinMix heads)
- Batch = multiple stations' histories from PeMSD7
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
