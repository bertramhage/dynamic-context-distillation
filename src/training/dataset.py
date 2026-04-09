"""Training dataset for the hypernetwork distillation loop.

Constructs rolling-window samples from the PEMS-BAY (or similar) long-format
DataFrame. Each sample consists of:
    - A long context window (fed to context encoder -> hypernetwork)
    - Multiple (short_context, forecast_target) pairs as "queries" for that context
      (the time-series analogue of D2L's multi-query per context)

Chronology is strictly causal:
    |---- long context ----|---- short context ----|---- forecast horizon ----|
                           ^                       ^
                           short_start              forecast_origin
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TrainingSample:
    """One training sample: long context + multiple forecast queries."""
    sensor_id: str
    long_context: torch.Tensor       # [long_ctx_len]
    short_contexts: torch.Tensor     # [n_queries, short_ctx_len]
    teacher_contexts: torch.Tensor   # [n_queries, teacher_ctx_len]  (full context for teacher)
    targets: torch.Tensor            # [n_queries, prediction_length]


class HypernetTrainingDataset(Dataset):
    """Rolling-window dataset for hypernetwork training.

    For each (sensor, long_context_start) pair, extracts the long context
    and generates multiple forecast-origin queries within the admissible range.
    """

    def __init__(
        self,
        df_long: pd.DataFrame,
        long_context_steps: int,
        short_context_steps: int,
        prediction_length: int,
        n_queries_per_context: int = 4,
        query_stride_steps: int = 12,
        long_context_stride_steps: int = 288,
        sensor_ids: list[str] | None = None,
        train_start: pd.Timestamp | None = None,
        train_end: pd.Timestamp | None = None,
        freq_minutes: int = 5,
        id_col: str = "item_id",
        ts_col: str = "timestamp",
        target_col: str = "target",
    ):
        super().__init__()
        self.long_context_steps = long_context_steps
        self.short_context_steps = short_context_steps
        self.prediction_length = prediction_length
        self.n_queries = n_queries_per_context
        self.query_stride_steps = query_stride_steps
        self.freq_minutes = freq_minutes
        self.id_col = id_col
        self.ts_col = ts_col
        self.target_col = target_col

        # Filter to training window
        df = df_long.copy()
        if train_start is not None:
            df = df[df[ts_col] >= train_start]
        if train_end is not None:
            df = df[df[ts_col] < train_end]

        if sensor_ids is None:
            sensor_ids = sorted(df[id_col].unique().tolist())
        self.sensor_ids = sensor_ids

        # Build per-sensor time-indexed arrays for fast slicing
        self._sensor_data: dict[str, np.ndarray] = {}
        self._sensor_timestamps: dict[str, np.ndarray] = {}
        for sid in sensor_ids:
            s_df = df[df[id_col] == sid].sort_values(ts_col)
            self._sensor_data[sid] = s_df[target_col].values.astype(np.float32)
            self._sensor_timestamps[sid] = s_df[ts_col].values

        # Build sample index: (sensor_id, long_context_start_idx)
        self._samples: list[tuple[str, int]] = []
        total_needed = (
            long_context_steps + short_context_steps
            + prediction_length
            + (n_queries_per_context - 1) * query_stride_steps
        )

        for sid in sensor_ids:
            n_steps = len(self._sensor_data[sid])
            start = 0
            while start + total_needed <= n_steps:
                self._samples.append((sid, start))
                start += long_context_stride_steps

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sid, lc_start = self._samples[idx]
        data = self._sensor_data[sid]

        long_ctx = data[lc_start : lc_start + self.long_context_steps]

        short_contexts = []
        teacher_contexts = []
        targets = []

        for q in range(self.n_queries):
            offset = q * self.query_stride_steps
            # Short context starts right after long context + offset
            sc_start = lc_start + self.long_context_steps + offset
            sc_end = sc_start + self.short_context_steps
            t_end = sc_end + self.prediction_length

            short_contexts.append(data[sc_start:sc_end])

            # Teacher gets the full window: from long_context_start to sc_end
            teacher_contexts.append(data[lc_start:sc_end])

            targets.append(data[sc_end:t_end])

        return {
            "long_context": torch.from_numpy(long_ctx),
            "short_contexts": torch.from_numpy(np.stack(short_contexts)),
            "teacher_contexts": torch.from_numpy(np.stack(teacher_contexts)),
            "targets": torch.from_numpy(np.stack(targets)),
        }


def collate_training_batch(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a batch of training samples.

    Returns dict with:
        long_context: [batch_size, long_ctx_len]
        short_contexts: [batch_size, n_queries, short_ctx_len]
        teacher_contexts: [batch_size, n_queries, teacher_ctx_len]
        targets: [batch_size, n_queries, prediction_length]
    """
    return {
        "long_context": torch.stack([s["long_context"] for s in batch]),
        "short_contexts": torch.stack([s["short_contexts"] for s in batch]),
        "teacher_contexts": torch.stack([s["teacher_contexts"] for s in batch]),
        "targets": torch.stack([s["targets"] for s in batch]),
    }
