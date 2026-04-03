from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from omegaconf import OmegaConf

from src.evaluation.main import run_evaluation
from src.orchestration.context_encoder import ChronosContextEncoder
from src.orchestration.lora_saver import save_adapter_to_disk
from src.utils import utils as shared_utils


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _extract_sensor_tensor(
    df_long: pd.DataFrame,
    sensor_ids: list[str],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    id_col: str,
    ts_col: str,
    target_col: str,
) -> tuple[torch.Tensor, list[str]]:
    """Extract a [n_sensors, seq_len] tensor from *df_long* for a time window.

    Returns the tensor and the ordered list of sensor IDs that appear as rows
    (same order as the batch dimension).
    """
    mask = (
        df_long[ts_col].ge(start_time)
        & df_long[ts_col].lt(end_time)
        & df_long[id_col].isin(sensor_ids)
    )
    window_df = df_long.loc[mask]

    wide = window_df.pivot(index=ts_col, columns=id_col, values=target_col)
    # Ensure consistent sensor ordering.
    ordered_ids = sorted(wide.columns.tolist())
    wide = wide[ordered_ids]

    tensor = torch.tensor(wide.values.T, dtype=torch.float32)  # [n_sensors, seq_len]
    return tensor, ordered_ids


# ---------------------------------------------------------------------------
# Prediction-time schedule (mirrors evaluation layer logic)
# ---------------------------------------------------------------------------

def _compute_prediction_times(
    cfg,
    df_long: pd.DataFrame,
) -> list[pd.Timestamp]:
    """Compute the list of prediction times that the evaluation layer will use."""
    freq = cfg.freq
    id_column = cfg.id_column
    number_of_sensors = df_long[id_column].nunique()
    day_slot = int(24 * 60 / freq)

    if cfg.start_test_day is not None and cfg.n_test_days is not None:
        start_test_day = cfg.start_test_day
        n_test_days = cfg.n_test_days
    elif cfg.proportion_test is not None:
        proportion_test = cfg.proportion_test
        start_test_day = int(
            df_long.shape[0] * (1 - proportion_test) / (day_slot * number_of_sensors)
        )
        n_test_days = int(
            df_long.shape[0] * proportion_test / (day_slot * number_of_sensors)
        )
    else:
        raise ValueError(
            "Either start_test_day/n_test_days or proportion_test must be set."
        )

    start_date = pd.Timestamp(cfg.start_date)
    experiment_start = start_date + pd.Timedelta(days=start_test_day)
    experiment_end = experiment_start + pd.Timedelta(days=n_test_days)

    prediction_length = cfg.prediction_length + 1
    stride_steps = cfg.stride_steps
    stride_delta = pd.Timedelta(minutes=stride_steps * freq)
    max_horizon_delta = pd.Timedelta(minutes=max(cfg.horizons))
    safe_end = experiment_end - max_horizon_delta

    times: list[pd.Timestamp] = []
    t = experiment_start
    while t <= safe_end:
        times.append(t)
        t += stride_delta
    return times


# ---------------------------------------------------------------------------
# Long-history window helpers
# ---------------------------------------------------------------------------

def _long_history_window_fixed(
    cfg,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the absolute (start, end) of the fixed long-history window."""
    orch = cfg.orchestration
    start = pd.Timestamp(orch.long_history_start_date)
    freq_delta = pd.Timedelta(minutes=cfg.freq)
    end = start + int(orch.long_history_length_steps) * freq_delta
    return start, end


def _long_history_window_rolling(
    cfg,
    prediction_time: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) of the rolling long-history window for a given prediction time.

    Layout:
        |--- long history ---|--- short context ---|--- forecast ---|
                             ^                     ^
                             short_start           prediction_time

    ``long_history_end_offset_steps`` shifts the end of the long history
    relative to the short-context start:
        0  -> no overlap / no gap (default)
        >0 -> overlap into the short context
        <0 -> gap between long history and short context
    """
    orch = cfg.orchestration
    freq_delta = pd.Timedelta(minutes=cfg.freq)
    short_len = int(orch.short_context_length_steps)
    long_len = int(orch.long_history_length_steps)
    offset = int(getattr(orch, "long_history_end_offset_steps", 0))

    short_start = prediction_time - short_len * freq_delta
    long_end = short_start + offset * freq_delta
    long_start = long_end - long_len * freq_delta
    return long_start, long_end


def _infer_output_layer_lora_dims(pipeline: Chronos2Pipeline) -> tuple[int, int]:
    """Infer LoRA (d_in, d_out) for output_patch_embedding.output_layer.

    Returns the Linear layer ``(in_features, out_features)`` used to size
    zero-filled LoRA tensors for the output head.
    """
    layer = pipeline.model.output_patch_embedding.output_layer
    if not hasattr(layer, "in_features") or not hasattr(layer, "out_features"):
        raise ValueError(
            "Could not infer output layer dimensions: "
            "missing in_features/out_features on output_layer"
        )
    return int(layer.in_features), int(layer.out_features)


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def run_orchestration(cfg) -> tuple[dict, dict]:
    """Generate LoRA adapters from a trained hypernetwork and run evaluation.

    Returns ``(horizon_metrics, runtime_stats)`` from the evaluation layer.
    """
    orch = cfg.orchestration
    shared_utils.set_seed(cfg.seed)

    # ---- device ----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ---- load dataset ----
    df_long = shared_utils.load_dataset(cfg)

    # ---- load Chronos-2 pipeline ----
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map=device,
    )

    # ---- context encoder (frozen Chronos-2) ----
    context_encoder = ChronosContextEncoder(pipeline)

    # ---- load hypernetwork checkpoint ----
    checkpoint_path = str(orch.checkpoint_path)
    print(f"Loading hypernetwork from {checkpoint_path}")
    hypernetwork = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hypernetwork.eval()
    for p in hypernetwork.parameters():
        p.requires_grad = False

    # ---- adapter output directory ----
    run_id = getattr(orch, "run_id", None) or str(uuid.uuid4())[:8]
    adapter_root = Path(str(orch.adapter_output_dir)) / run_id
    adapter_root.mkdir(parents=True, exist_ok=True)
    print(f"Adapters will be saved to {adapter_root}")

    # ---- gather sensor ids ----
    id_col = cfg.id_column
    ts_col = cfg.timestamp_column
    target_col = cfg.target
    sensor_ids = sorted(df_long[id_col].dropna().astype(str).unique().tolist())
    print(f"Number of sensors: {len(sensor_ids)}")

    # ---- adapter config from cfg ----
    adapter_cfg = getattr(cfg, "adapter", None)
    rank = 8 if adapter_cfg is None else int(adapter_cfg.get("rank", 8))
    alpha = 16 if adapter_cfg is None else int(adapter_cfg.get("alpha", 16))
    output_lora_d_in, output_lora_d_out = _infer_output_layer_lora_dims(pipeline)
    target_modules = None
    if adapter_cfg is not None and adapter_cfg.get("target_modules") is not None:
        target_modules = list(adapter_cfg.target_modules)

    # ---- compute prediction times ----
    prediction_times = _compute_prediction_times(cfg, df_long)
    print(f"Prediction steps: {len(prediction_times)}")

    rolling = bool(getattr(orch, "rolling_long_history", False))
    encode_batch_size = int(getattr(orch, "encode_batch_size", 32))

    assignment_rows: list[dict] = []

    if not rolling:
        # -----------------------------------------------------------
        # FIXED long history: one adapter per sensor (shared across
        # all prediction times).
        # -----------------------------------------------------------
        long_start, long_end = _long_history_window_fixed(cfg)
        print(f"Fixed long history: {long_start} -> {long_end}")

        context_tensor, ordered_ids = _extract_sensor_tensor(
            df_long, sensor_ids, long_start, long_end, id_col, ts_col, target_col,
        )
        print(f"Context tensor shape: {context_tensor.shape}")

        # Encode through frozen Chronos-2, collecting per-layer intermediates.
        all_z = context_encoder.encode_intermediates_batched(
            context_tensor, batch_size=encode_batch_size,
        )
        # all_z: [n_sensors, num_layers=12, num_context_patches, d_model=768]
        # all_z[:, l, :, :] = Z_l = input to block l, used to generate LoRA for block l.

        # Run hypernetwork: shared Perceiver applied per-layer internally.
        lora_dict = hypernetwork(all_z)
        # Expected: {"q": {"A": [n_sensors, 12, r, d_in], "B": [n_sensors, 12, d_out, r]}, ...}

        # Save one adapter per sensor.
        for sensor_batch_idx, sensor_id in enumerate(ordered_ids):
            adapter_id = sensor_id
            adapter_dir = adapter_root / adapter_id
            save_adapter_to_disk(
                lora_dict,
                sensor_idx=sensor_batch_idx,
                adapter_dir=adapter_dir,
                rank=rank,
                alpha=alpha,
                output_lora_d_in=output_lora_d_in,
                output_lora_d_out=output_lora_d_out,
                target_modules=target_modules,
            )

            # Same adapter for every prediction time.
            for pt in prediction_times:
                assignment_rows.append(
                    {
                        id_col: sensor_id,
                        "prediction_time": pt,
                        "adapter_id": adapter_id,
                    }
                )

        print(f"Saved {len(ordered_ids)} adapters (fixed long history)")

    else:
        # -----------------------------------------------------------
        # ROLLING long history: one adapter per sensor per step.
        # -----------------------------------------------------------
        for step_idx, pt in enumerate(prediction_times):
            long_start, long_end = _long_history_window_rolling(cfg, pt)

            context_tensor, ordered_ids = _extract_sensor_tensor(
                df_long, sensor_ids, long_start, long_end, id_col, ts_col, target_col,
            )

            all_z = context_encoder.encode_intermediates_batched(
                context_tensor, batch_size=encode_batch_size,
            )
            # all_z: [n_sensors, num_layers=12, num_context_patches, d_model=768]

            lora_dict = hypernetwork(all_z)

            for sensor_batch_idx, sensor_id in enumerate(ordered_ids):
                adapter_id = f"{sensor_id}_step{step_idx}"
                adapter_dir = adapter_root / adapter_id
                save_adapter_to_disk(
                    lora_dict,
                    sensor_idx=sensor_batch_idx,
                    adapter_dir=adapter_dir,
                    rank=rank,
                    alpha=alpha,
                    output_lora_d_in=output_lora_d_in,
                    output_lora_d_out=output_lora_d_out,
                    target_modules=target_modules,
                )

                assignment_rows.append(
                    {
                        id_col: sensor_id,
                        "prediction_time": pt,
                        "adapter_id": adapter_id,
                    }
                )

            if (step_idx + 1) % 10 == 0 or step_idx == 0:
                print(
                    f"Step {step_idx + 1}/{len(prediction_times)}: "
                    f"generated {len(ordered_ids)} adapters"
                )

        print(
            f"Saved {len(prediction_times) * len(sensor_ids)} adapters "
            f"(rolling long history)"
        )

    assignments_df = pd.DataFrame(assignment_rows)

    # ---- configure evaluation ----
    # Override history_length to the short context length.
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    OmegaConf.update(
        cfg,
        "evaluation.history_length_steps",
        int(orch.short_context_length_steps),
    )
    # Point the adapter root to our generated adapters.
    OmegaConf.update(cfg, "adapter.adapter_root", str(adapter_root))

    print(
        f"\nStarting evaluation with short context = "
        f"{orch.short_context_length_steps} steps, "
        f"{len(assignments_df)} assignment entries"
    )

    # ---- run evaluation ----
    horizon_metrics, runtime_stats = run_evaluation(
        cfg=cfg,
        pipeline=pipeline,
        df_long=df_long,
        assignments_df=assignments_df,
        return_runtime=True,
    )

    return horizon_metrics, runtime_stats
