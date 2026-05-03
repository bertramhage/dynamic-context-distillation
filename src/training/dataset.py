from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class TrainingSample:
    """One training sample: long context + multiple forecast queries."""
    sensor_id: str
    long_context: torch.Tensor       # [long_ctx_len]
    short_contexts: torch.Tensor     # [n_queries, short_ctx_len]
    teacher_contexts: torch.Tensor | None  # [n_queries, teacher_ctx_len] when teacher mode is used
    targets: torch.Tensor            # [n_queries, prediction_length]


def _to_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _sample_hierarchical_lengths(
    batch_size: int,
    base_steps: int,
    sigma_outer: float,
    sigma_inner: float,
    min_steps: int,
    max_steps: int,
    quantize_steps: int,
) -> torch.Tensor:
    if quantize_steps <= 0:
        raise ValueError("quantize_steps must be >= 1")

    outer_mean = float(base_steps)
    if sigma_outer > 0:
        outer_mean += float(torch.randn(1).item()) * float(sigma_outer)

    lengths = torch.full((batch_size,), outer_mean, dtype=torch.float32)
    if sigma_inner > 0:
        lengths = lengths + torch.randn(batch_size) * float(sigma_inner)

    lengths = torch.round(lengths)
    if quantize_steps > 1:
        q = float(quantize_steps)
        lengths = torch.round(lengths / q) * q

    lengths = torch.clamp(lengths, min=float(min_steps), max=float(max_steps))
    return lengths.to(torch.int64)


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
        long_context_min_steps: int | None = None,
        long_context_max_steps: int | None = None,
        short_context_min_steps: int | None = None,
        short_context_max_steps: int | None = None,
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
        self.long_context_steps = int(long_context_steps)
        self.short_context_steps = int(short_context_steps)
        self.prediction_length = int(prediction_length)

        self.long_context_min_steps = _to_optional_int(long_context_min_steps)
        self.long_context_max_steps = _to_optional_int(long_context_max_steps)
        self.short_context_min_steps = _to_optional_int(short_context_min_steps)
        self.short_context_max_steps = _to_optional_int(short_context_max_steps)

        if self.long_context_min_steps is None:
            self.long_context_min_steps = self.long_context_steps
        if self.long_context_max_steps is None:
            self.long_context_max_steps = self.long_context_steps
        if self.short_context_min_steps is None:
            self.short_context_min_steps = self.short_context_steps
        if self.short_context_max_steps is None:
            self.short_context_max_steps = self.short_context_steps

        if self.long_context_min_steps <= 0 or self.short_context_min_steps <= 0:
            raise ValueError("Minimum context lengths must be positive")
        if self.long_context_min_steps > self.long_context_max_steps:
            raise ValueError("long_context_min_steps must be <= long_context_max_steps")
        if self.short_context_min_steps > self.short_context_max_steps:
            raise ValueError("short_context_min_steps must be <= short_context_max_steps")

        self.n_queries = n_queries_per_context
        self.query_stride_steps = query_stride_steps
        self.freq_minutes = freq_minutes
        self.id_col = id_col
        self.ts_col = ts_col
        self.target_col = target_col

        self.max_query_offset = (self.n_queries - 1) * self.query_stride_steps
        self.window_len = (
            self.long_context_max_steps
            + self.short_context_max_steps
            + self.max_query_offset
            + self.prediction_length
        )

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
        total_needed = self.window_len

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

        series_window = data[lc_start : lc_start + self.window_len]
        return {
            "series_window": torch.from_numpy(series_window),
            "sample_idx": idx,
        }


class TrainingBatchCollator:
    """Collate callable with optional hierarchical context-length jitter."""

    def __init__(
        self,
        *,
        long_context_steps: int,
        short_context_steps: int,
        prediction_length: int,
        n_queries_per_context: int,
        query_stride_steps: int,
        length_jitter: dict | None = None,
        fixed_long_steps: torch.Tensor | None = None,
        fixed_short_steps: torch.Tensor | None = None,
        build_teacher_contexts: bool = True,
    ):
        self.base_long_steps = int(long_context_steps)
        self.base_short_steps = int(short_context_steps)
        self.pred_steps = int(prediction_length)
        self.n_queries = int(n_queries_per_context)
        self.query_stride = int(query_stride_steps)
        self.max_offset = (self.n_queries - 1) * self.query_stride

        jitter = length_jitter or {}
        self.jitter_enabled = bool(jitter.get("enabled", False))
        self.quantize_steps = int(jitter.get("quantize_steps", 1))

        self.long_sigma_outer = float(jitter.get("long_sigma_outer", 0.0))
        self.long_sigma_inner = float(jitter.get("long_sigma_inner", 0.0))
        self.short_sigma_outer = float(jitter.get("short_sigma_outer", 0.0))
        self.short_sigma_inner = float(jitter.get("short_sigma_inner", 0.0))

        self.long_min_steps = int(jitter.get("long_min_steps", self.base_long_steps))
        self.long_max_steps = int(jitter.get("long_max_steps", self.base_long_steps))
        self.short_min_steps = int(jitter.get("short_min_steps", self.base_short_steps))
        self.short_max_steps = int(jitter.get("short_max_steps", self.base_short_steps))
        self.build_teacher_contexts = bool(build_teacher_contexts)

        if self.long_min_steps > self.long_max_steps:
            raise ValueError("long_min_steps must be <= long_max_steps")
        if self.short_min_steps > self.short_max_steps:
            raise ValueError("short_min_steps must be <= short_max_steps")

        self.fixed_long_steps = None
        self.fixed_short_steps = None
        if (fixed_long_steps is None) != (fixed_short_steps is None):
            raise ValueError(
                "fixed_long_steps and fixed_short_steps must both be provided or both be None"
            )

        if fixed_long_steps is not None and fixed_short_steps is not None:
            self.fixed_long_steps = fixed_long_steps.to(dtype=torch.int64, device="cpu")
            self.fixed_short_steps = fixed_short_steps.to(dtype=torch.int64, device="cpu")
            if self.fixed_long_steps.shape != self.fixed_short_steps.shape:
                raise ValueError("Fixed long/short length tensors must have identical shape")

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        if batch_size == 0:
            raise ValueError("Empty batch is not supported")

        sample_indices = torch.tensor(
            [int(sample["sample_idx"]) for sample in batch], dtype=torch.int64
        )

        if self.fixed_long_steps is not None and self.fixed_short_steps is not None:
            sampled_long_steps = self.fixed_long_steps.index_select(0, sample_indices)
            sampled_short_steps = self.fixed_short_steps.index_select(0, sample_indices)

        elif self.jitter_enabled:
            sampled_long_steps = _sample_hierarchical_lengths(
                batch_size=batch_size,
                base_steps=self.base_long_steps,
                sigma_outer=self.long_sigma_outer,
                sigma_inner=self.long_sigma_inner,
                min_steps=self.long_min_steps,
                max_steps=self.long_max_steps,
                quantize_steps=self.quantize_steps,
            )
            sampled_short_steps = _sample_hierarchical_lengths(
                batch_size=batch_size,
                base_steps=self.base_short_steps,
                sigma_outer=self.short_sigma_outer,
                sigma_inner=self.short_sigma_inner,
                min_steps=self.short_min_steps,
                max_steps=self.short_max_steps,
                quantize_steps=self.quantize_steps,
            )
        else:
            sampled_long_steps = torch.full(
                (batch_size,), self.base_long_steps, dtype=torch.int64
            )
            sampled_short_steps = torch.full(
                (batch_size,), self.base_short_steps, dtype=torch.int64
            )

        long_contexts = []
        short_contexts_per_sample = []
        teacher_contexts_per_sample = [] if self.build_teacher_contexts else None
        targets_per_sample = []

        for i, sample in enumerate(batch):
            series_window = sample["series_window"]
            long_steps = int(sampled_long_steps[i].item())
            short_steps = int(sampled_short_steps[i].item())

            long_contexts.append(series_window[:long_steps])

            teacher_len = long_steps + short_steps + self.max_offset
            sample_short_contexts = []
            sample_teacher_contexts = [] if self.build_teacher_contexts else None
            sample_targets = []

            for q in range(self.n_queries):
                offset = q * self.query_stride
                sc_start = long_steps + offset
                sc_end = sc_start + short_steps
                t_end = sc_end + self.pred_steps

                sample_short_contexts.append(series_window[sc_start:sc_end])
                sample_targets.append(series_window[sc_end:t_end])

                if self.build_teacher_contexts:
                    teacher_raw = series_window[:sc_end]
                    pad_len = teacher_len - int(teacher_raw.shape[-1])
                    if pad_len > 0:
                        teacher_raw = F.pad(teacher_raw, (pad_len, 0), value=float("nan"))
                    sample_teacher_contexts.append(teacher_raw)

            short_contexts_per_sample.append(torch.stack(sample_short_contexts))
            if self.build_teacher_contexts:
                teacher_contexts_per_sample.append(torch.stack(sample_teacher_contexts))
            targets_per_sample.append(torch.stack(sample_targets))

        max_long_len = max(int(t.shape[-1]) for t in long_contexts)
        max_short_len = max(int(t.shape[-1]) for t in short_contexts_per_sample)

        padded_long_contexts = [
            F.pad(t, (max_long_len - int(t.shape[-1]), 0), value=float("nan"))
            for t in long_contexts
        ]
        padded_short_contexts = [
            F.pad(t, (max_short_len - int(t.shape[-1]), 0), value=float("nan"))
            for t in short_contexts_per_sample
        ]

        output = {
            "long_context": torch.stack(padded_long_contexts),
            "short_contexts": torch.stack(padded_short_contexts),
            "targets": torch.stack(targets_per_sample),
            "sample_indices": sample_indices,
            "sampled_long_context_steps": sampled_long_steps,
            "sampled_short_context_steps": sampled_short_steps,
        }

        if self.build_teacher_contexts:
            max_teacher_len = max(int(t.shape[-1]) for t in teacher_contexts_per_sample)
            padded_teacher_contexts = [
                F.pad(t, (max_teacher_len - int(t.shape[-1]), 0), value=float("nan"))
                for t in teacher_contexts_per_sample
            ]
            output["teacher_contexts"] = torch.stack(padded_teacher_contexts)

        return output


def make_training_collate_fn(
    *,
    long_context_steps: int,
    short_context_steps: int,
    prediction_length: int,
    n_queries_per_context: int,
    query_stride_steps: int,
    length_jitter: dict | None = None,
    fixed_long_steps: torch.Tensor | None = None,
    fixed_short_steps: torch.Tensor | None = None,
    build_teacher_contexts: bool = True,
) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    return TrainingBatchCollator(
        long_context_steps=long_context_steps,
        short_context_steps=short_context_steps,
        prediction_length=prediction_length,
        n_queries_per_context=n_queries_per_context,
        query_stride_steps=query_stride_steps,
        length_jitter=length_jitter,
        fixed_long_steps=fixed_long_steps,
        fixed_short_steps=fixed_short_steps,
        build_teacher_contexts=build_teacher_contexts,
    )
