"""Teacher input-output cache for distillation training.

Builds a deterministic, jitter-aware cache of teacher predictions so the
frozen Chronos teacher does not run inside every training epoch.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


@dataclass
class TeacherCacheData:
    """In-memory representation of cached teacher supervision."""

    split_name: str
    cache_key: str
    cache_path: Path
    sampled_long_steps: torch.Tensor
    sampled_short_steps: torch.Tensor
    teacher_preds: torch.Tensor


def _compute_num_output_patches(prediction_length: int, output_patch_size: int) -> int:
    """Compute number of output patches needed for a prediction length."""
    return math.ceil(prediction_length / output_patch_size)


def _normalize_for_hash(value: Any) -> Any:
    """Convert objects to JSON-hashable primitives."""
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _hash_dataset_samples(dataset) -> str:
    """Hash sample identity tuples to detect dataset-window changes."""
    samples = getattr(dataset, "_samples", None)
    if samples is None:
        return "no-sample-index"

    digest = hashlib.sha256()
    for sensor_id, start_idx in samples:
        digest.update(f"{sensor_id}:{int(start_idx)}|".encode("utf-8"))
    return digest.hexdigest()


def _resolve_cache_dtype(dtype_name: str) -> torch.dtype:
    """Map config dtype name to torch dtype used for persisted outputs."""
    name = str(dtype_name).lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported teacher cache dtype: {dtype_name}")


def _sample_hierarchical_lengths_for_cache(
    *,
    num_samples: int,
    base_steps: int,
    sigma_outer: float,
    sigma_inner: float,
    min_steps: int,
    max_steps: int,
    quantize_steps: int,
    group_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample deterministic per-sample lengths using hierarchical jitter.

    The sampler follows the same hierarchy as online collation:
    one outer draw per pseudo-batch and inner per-sample draws.
    """
    if quantize_steps <= 0:
        raise ValueError("quantize_steps must be >= 1")
    if group_size <= 0:
        raise ValueError("group_size must be >= 1")

    lengths = torch.empty(num_samples, dtype=torch.int64)

    for start in range(0, num_samples, group_size):
        end = min(start + group_size, num_samples)
        batch_size = end - start

        outer_mean = float(base_steps)
        if sigma_outer > 0:
            outer_mean += float(torch.randn(1, generator=generator).item()) * sigma_outer

        batch_lengths = torch.full((batch_size,), outer_mean, dtype=torch.float32)
        if sigma_inner > 0:
            batch_lengths = batch_lengths + torch.randn(
                batch_size, generator=generator
            ) * sigma_inner

        batch_lengths = torch.round(batch_lengths)
        if quantize_steps > 1:
            q = float(quantize_steps)
            batch_lengths = torch.round(batch_lengths / q) * q

        batch_lengths = torch.clamp(
            batch_lengths, min=float(min_steps), max=float(max_steps)
        )
        lengths[start:end] = batch_lengths.to(torch.int64)

    return lengths


def _resolve_sampled_lengths(
    *,
    dataset,
    jitter_enabled: bool,
    jitter_cfg: dict[str, Any] | None,
    jitter_seed: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve deterministic per-sample long/short lengths for cache builds."""
    n_samples = len(dataset)
    if not jitter_enabled:
        return (
            torch.full((n_samples,), int(dataset.long_context_steps), dtype=torch.int64),
            torch.full((n_samples,), int(dataset.short_context_steps), dtype=torch.int64),
        )

    cfg = jitter_cfg or {}
    quantize_steps = int(cfg.get("quantize_steps", 1))

    long_sigma_outer = float(cfg.get("long_sigma_outer", 0.0))
    long_sigma_inner = float(cfg.get("long_sigma_inner", 0.0))
    short_sigma_outer = float(cfg.get("short_sigma_outer", 0.0))
    short_sigma_inner = float(cfg.get("short_sigma_inner", 0.0))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(jitter_seed))

    sampled_long = _sample_hierarchical_lengths_for_cache(
        num_samples=n_samples,
        base_steps=int(dataset.long_context_steps),
        sigma_outer=long_sigma_outer,
        sigma_inner=long_sigma_inner,
        min_steps=int(dataset.long_context_min_steps),
        max_steps=int(dataset.long_context_max_steps),
        quantize_steps=quantize_steps,
        group_size=group_size,
        generator=generator,
    )
    sampled_short = _sample_hierarchical_lengths_for_cache(
        num_samples=n_samples,
        base_steps=int(dataset.short_context_steps),
        sigma_outer=short_sigma_outer,
        sigma_inner=short_sigma_inner,
        min_steps=int(dataset.short_context_min_steps),
        max_steps=int(dataset.short_context_max_steps),
        quantize_steps=quantize_steps,
        group_size=group_size,
        generator=generator,
    )
    return sampled_long, sampled_short


def _build_cache_key(
    *,
    split_name: str,
    cfg: DictConfig,
    dataset,
    jitter_enabled: bool,
    jitter_cfg: dict[str, Any] | None,
    jitter_seed: int,
    teacher_model_name: str,
    cache_dtype: str,
) -> tuple[str, dict[str, Any]]:
    """Build stable cache key from all parameters affecting teacher I/O."""
    t = cfg.training

    if split_name == "train":
        split_start = t.train_start_date
        split_end = t.train_end_date
    elif split_name == "val":
        split_start = t.val_start_date
        split_end = t.val_end_date
    else:
        split_start = None
        split_end = None

    teacher_cfg = getattr(cfg, "teacher_model", None)
    teacher_model_revision = (
        str(getattr(teacher_cfg, "revision", None)) if teacher_cfg is not None else None
    )

    payload = {
        "split": split_name,
        "split_window": {
            "start": str(split_start) if split_start is not None else None,
            "end": str(split_end) if split_end is not None else None,
        },
        "dataset_cfg": cfg.dataset_cfg,
        "freq": int(cfg.freq),
        "id_column": cfg.id_column,
        "timestamp_column": cfg.timestamp_column,
        "target": cfg.target,
        "train_window": {
            "long_context_steps": int(t.long_context_steps),
            "short_context_steps": int(t.short_context_steps),
            "prediction_length": int(t.prediction_length),
            "n_queries_per_context": int(t.n_queries_per_context),
            "query_stride_steps": int(t.query_stride_steps),
            "long_context_stride_steps": int(t.long_context_stride_steps),
        },
        "dataset_sample_count": int(len(dataset)),
        "dataset_sample_hash": _hash_dataset_samples(dataset),
        "jitter": {
            "enabled": bool(jitter_enabled),
            "config": _normalize_for_hash(jitter_cfg or {}),
            "seed": int(jitter_seed),
            "train_batch_size": int(t.train_batch_size),
        },
        "teacher_model": {
            "name": teacher_model_name,
            "revision": teacher_model_revision,
        },
        "teacher_cache_dtype": cache_dtype,
    }

    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return key, payload


@torch.no_grad()
def _run_teacher_forward(
    *,
    teacher_model,
    teacher_contexts: torch.Tensor,
    prediction_length: int,
    output_patch_size: int,
    device: str,
) -> torch.Tensor:
    """Run frozen teacher over batched contexts and return quantile predictions."""
    batch_size, n_queries, context_len = teacher_contexts.shape
    flat_ctx = teacher_contexts.reshape(batch_size * n_queries, context_len).to(device)

    num_patches = _compute_num_output_patches(prediction_length, output_patch_size)
    out = teacher_model.forward(flat_ctx, num_output_patches=num_patches)
    quantile_preds = out.quantile_preds[:, :, :prediction_length]
    n_quantiles = quantile_preds.shape[1]

    return quantile_preds.reshape(batch_size, n_queries, n_quantiles, prediction_length).cpu()


@torch.no_grad()
def _build_teacher_preds(
    *,
    dataset,
    teacher_model,
    sampled_long_steps: torch.Tensor,
    sampled_short_steps: torch.Tensor,
    n_queries: int,
    query_stride: int,
    prediction_length: int,
    build_batch_size: int,
    output_dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Build teacher predictions for all dataset samples once."""
    if build_batch_size <= 0:
        raise ValueError("build_batch_size must be >= 1")

    n_samples = len(dataset)
    max_offset = (n_queries - 1) * query_stride
    output_patch_size = int(teacher_model.chronos_config.output_patch_size)

    teacher_preds = None

    for start in range(0, n_samples, build_batch_size):
        end = min(start + build_batch_size, n_samples)

        contexts_batch = []
        max_teacher_len_in_batch = 0
        for sample_idx in range(start, end):
            sample = dataset[sample_idx]
            series_window = sample["series_window"].to(torch.float32)

            long_steps = int(sampled_long_steps[sample_idx].item())
            short_steps = int(sampled_short_steps[sample_idx].item())
            teacher_len = long_steps + short_steps + max_offset

            query_contexts = []
            for q in range(n_queries):
                offset = q * query_stride
                short_end = long_steps + offset + short_steps

                teacher_raw = series_window[:short_end]
                pad_len = teacher_len - int(teacher_raw.shape[-1])
                if pad_len > 0:
                    teacher_raw = F.pad(teacher_raw, (pad_len, 0), value=float("nan"))
                query_contexts.append(teacher_raw)

            sample_teacher_contexts = torch.stack(query_contexts)
            contexts_batch.append(sample_teacher_contexts)
            max_teacher_len_in_batch = max(
                max_teacher_len_in_batch, int(sample_teacher_contexts.shape[-1])
            )

        padded_contexts_batch = []
        for sample_teacher_contexts in contexts_batch:
            pad_len = max_teacher_len_in_batch - int(sample_teacher_contexts.shape[-1])
            if pad_len > 0:
                sample_teacher_contexts = F.pad(
                    sample_teacher_contexts, (pad_len, 0), value=float("nan")
                )
            padded_contexts_batch.append(sample_teacher_contexts)

        teacher_contexts = torch.stack(padded_contexts_batch)
        batch_preds = _run_teacher_forward(
            teacher_model=teacher_model,
            teacher_contexts=teacher_contexts,
            prediction_length=prediction_length,
            output_patch_size=output_patch_size,
            device=device,
        ).to(output_dtype)

        if teacher_preds is None:
            n_quantiles = int(batch_preds.shape[2])
            teacher_preds = torch.empty(
                (n_samples, n_queries, n_quantiles, prediction_length),
                dtype=output_dtype,
            )

        teacher_preds[start:end] = batch_preds

        if ((end // build_batch_size) % 25 == 0) or end == n_samples:
            print(f"  Teacher cache progress: {end}/{n_samples} samples")

    if teacher_preds is None:
        raise RuntimeError("Teacher cache build produced no predictions")

    return teacher_preds


def _load_cache(path: Path) -> TeacherCacheData:
    """Load persisted teacher cache tensors from disk."""
    payload = torch.load(path, map_location="cpu")
    return TeacherCacheData(
        split_name=str(payload["split_name"]),
        cache_key=str(payload["cache_key"]),
        cache_path=path,
        sampled_long_steps=payload["sampled_long_steps"].to(torch.int64),
        sampled_short_steps=payload["sampled_short_steps"].to(torch.int64),
        teacher_preds=payload["teacher_preds"],
    )


def _save_cache(
    *,
    path: Path,
    split_name: str,
    cache_key: str,
    metadata: dict[str, Any],
    sampled_long_steps: torch.Tensor,
    sampled_short_steps: torch.Tensor,
    teacher_preds: torch.Tensor,
) -> None:
    """Persist teacher cache tensors and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "split_name": split_name,
            "cache_key": cache_key,
            "metadata": metadata,
            "sampled_long_steps": sampled_long_steps.cpu().to(torch.int64),
            "sampled_short_steps": sampled_short_steps.cpu().to(torch.int64),
            "teacher_preds": teacher_preds.cpu(),
        },
        path,
    )


def load_or_build_teacher_cache(
    *,
    split_name: str,
    cfg: DictConfig,
    dataset,
    teacher_model,
    teacher_model_name: str,
    device: str,
    jitter_enabled: bool,
    jitter_cfg: dict[str, Any] | None,
) -> TeacherCacheData | None:
    """Load existing teacher cache or build it before training starts."""
    cache_cfg = cfg.get("teacher_cache", None)
    if cache_cfg is None or not bool(cache_cfg.get("enabled", True)):
        return None

    cache_dir = Path(str(cache_cfg.get("cache_dir", "outputs/teacher_cache")))
    cache_dtype_name = str(cache_cfg.get("dtype", "float16"))
    cache_dtype = _resolve_cache_dtype(cache_dtype_name)
    force_rebuild = bool(cache_cfg.get("rebuild", False))

    group_size = int(cfg.training.train_batch_size)
    build_batch_size = int(cache_cfg.get("build_batch_size", cfg.training.train_batch_size))

    jitter_seed = cache_cfg.get("jitter_seed", None)
    if jitter_seed is None:
        jitter_seed = cfg.seed
    jitter_seed = int(jitter_seed)

    cache_key, metadata = _build_cache_key(
        split_name=split_name,
        cfg=cfg,
        dataset=dataset,
        jitter_enabled=jitter_enabled,
        jitter_cfg=jitter_cfg,
        jitter_seed=jitter_seed,
        teacher_model_name=teacher_model_name,
        cache_dtype=cache_dtype_name,
    )
    cache_path = cache_dir / f"teacher_{split_name}_{cache_key}.pt"

    if cache_path.exists() and not force_rebuild:
        print(f"Loading teacher cache: {cache_path}")
        return _load_cache(cache_path)

    if force_rebuild and cache_path.exists():
        print(f"Rebuilding teacher cache despite existing file: {cache_path}")

    sampled_long_steps, sampled_short_steps = _resolve_sampled_lengths(
        dataset=dataset,
        jitter_enabled=jitter_enabled,
        jitter_cfg=jitter_cfg,
        jitter_seed=jitter_seed,
        group_size=group_size,
    )

    print(
        f"Building teacher cache for '{split_name}' split "
        f"({len(dataset)} samples, key={cache_key})"
    )

    teacher_preds = _build_teacher_preds(
        dataset=dataset,
        teacher_model=teacher_model,
        sampled_long_steps=sampled_long_steps,
        sampled_short_steps=sampled_short_steps,
        n_queries=int(cfg.training.n_queries_per_context),
        query_stride=int(cfg.training.query_stride_steps),
        prediction_length=int(cfg.training.prediction_length),
        build_batch_size=build_batch_size,
        output_dtype=cache_dtype,
        device=device,
    )

    _save_cache(
        path=cache_path,
        split_name=split_name,
        cache_key=cache_key,
        metadata=metadata,
        sampled_long_steps=sampled_long_steps,
        sampled_short_steps=sampled_short_steps,
        teacher_preds=teacher_preds,
    )
    print(f"Saved teacher cache: {cache_path}")

    return TeacherCacheData(
        split_name=split_name,
        cache_key=cache_key,
        cache_path=cache_path,
        sampled_long_steps=sampled_long_steps,
        sampled_short_steps=sampled_short_steps,
        teacher_preds=teacher_preds,
    )
