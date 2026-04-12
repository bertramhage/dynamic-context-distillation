"""Smoke test for dynamic in-memory LoRA evaluation runtime.

This script validates two cases on a tiny synthetic dataset:
1) Dynamic runtime with mixed per-sample adapters (one real adapter + one __none__).
2) Dynamic runtime with base-only (__none__) behavior and no provider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from omegaconf import OmegaConf

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.main import run_evaluation


class RandomDynamicLoraProvider:
    """Simple random LoRA provider for smoke testing dynamic injection."""

    def __init__(self, rank: int = 8, d_model: int = 768, num_layers: int = 12):
        self.rank = int(rank)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.call_count = 0

    def get_step_lora_batch(
        self,
        prediction_time: pd.Timestamp,
        item_ids: list[str],
        adapter_ids: list[str],
        default_adapter_id: str,
        device: torch.device,
    ) -> dict[str, dict[str, torch.Tensor]]:
        self.call_count += 1
        batch_size = len(item_ids)
        scale = 1e-4

        lora_batch = {}
        for short_name in ("q", "k", "v", "o"):
            A = torch.randn(
                batch_size,
                self.num_layers,
                self.rank,
                self.d_model,
                device=device,
                dtype=torch.float32,
            ) * scale
            B = torch.randn(
                batch_size,
                self.num_layers,
                self.d_model,
                self.rank,
                device=device,
                dtype=torch.float32,
            ) * scale

            for idx, adapter_id in enumerate(adapter_ids):
                if str(adapter_id) == str(default_adapter_id):
                    A[idx].zero_()
                    B[idx].zero_()

            lora_batch[short_name] = {"A": A, "B": B}

        return lora_batch


def _build_synthetic_df(start_date: str, freq_minutes: int = 60) -> pd.DataFrame:
    """Build a tiny long-format dataframe with two sensors and 3 days of data."""
    timestamps = pd.date_range(start=start_date, periods=72, freq=f"{freq_minutes}min")

    rows = []
    for sensor_idx, sensor_id in enumerate(("s0", "s1")):
        values = np.sin(np.linspace(0, 4 * np.pi, len(timestamps)) + sensor_idx)
        for ts, val in zip(timestamps, values):
            rows.append(
                {
                    "item_id": sensor_id,
                    "timestamp": ts,
                    "target": float(val),
                }
            )

    return pd.DataFrame(rows)


def _build_cfg() -> OmegaConf:
    """Build a minimal evaluation config required by run_evaluation."""
    return OmegaConf.create(
        {
            "dataset": "PEMS-BAY",
            "dataset_cfg": "dataset/PEMS-BAY",
            "seed": 42,
            "cross_learning": False,
            "batch_size": None,
            "id_column": "item_id",
            "timestamp_column": "timestamp",
            "target": "target",
            "freq": 60,
            "prediction_length": 2,
            "horizons": [60],
            "quantile_levels": [0.1, 0.5, 0.9],
            "start_date": "2020-01-01",
            "start_test_day": 1,
            "n_test_days": 1,
            "proportion_test": None,
            "stride_steps": 12,
            "past_covariates": None,
            "future_covariates": None,
            "decimal_precision": 4,
            "evaluation": {
                "history_length_steps": 8,
                "use_dynamic_lora": True,
                "dynamic_batch_size": 8,
            },
            "adapter": {
                "default_id": "__none__",
                "rank": 8,
                "alpha": 16,
                "adapter_root": "outputs/adapters",
                "target_modules": [
                    "self_attention.q",
                    "self_attention.k",
                    "self_attention.v",
                    "self_attention.o",
                    "output_patch_embedding.output_layer",
                ],
            },
        }
    )


def main() -> None:
    """Run smoke checks for dynamic adapter and base-only paths."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading Chronos-2 on {device}...")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
    )

    cfg = _build_cfg()
    df_long = _build_synthetic_df(start_date=str(cfg.start_date), freq_minutes=int(cfg.freq))

    prediction_time = pd.Timestamp(cfg.start_date) + pd.Timedelta(days=int(cfg.start_test_day))
    assignments_df = pd.DataFrame(
        [
            {"item_id": "s0", "prediction_time": prediction_time, "adapter_id": "dyn_a"},
            {"item_id": "s1", "prediction_time": prediction_time, "adapter_id": "__none__"},
        ]
    )

    provider = RandomDynamicLoraProvider(rank=8, d_model=768, num_layers=12)
    metrics_dyn, runtime_dyn = run_evaluation(
        cfg=cfg,
        pipeline=pipeline,
        df_long=df_long,
        assignments_df=assignments_df,
        dynamic_lora_provider=provider,
        return_runtime=True,
    )

    assert provider.call_count > 0, "Dynamic provider was not used."
    assert runtime_dyn["prediction_tasks"] > 0, "No tasks evaluated in dynamic mode."
    assert runtime_dyn["predict_backend_calls"] > 0, "No backend calls in dynamic mode."
    assert len(metrics_dyn) > 0, "No horizon metrics produced in dynamic mode."
    print("PASS: dynamic mixed-adapter path")

    metrics_base, runtime_base = run_evaluation(
        cfg=cfg,
        pipeline=pipeline,
        df_long=df_long,
        assignments_df=None,
        dynamic_lora_provider=None,
        return_runtime=True,
    )

    assert runtime_base["prediction_tasks"] > 0, "No tasks evaluated in base-only mode."
    assert runtime_base["predict_backend_calls"] > 0, "No backend calls in base-only mode."
    assert len(metrics_base) > 0, "No horizon metrics produced in base-only mode."
    print("PASS: dynamic base-only (__none__) path")


if __name__ == "__main__":
    main()
