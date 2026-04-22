"""CLI entrypoint for hypernetwork training.

Usage:
    uv run python -m src.training.main
    uv run python -m src.training.main wandb.enabled=true
    uv run python -m src.training.main training.train_batch_size=8 optimizer.lr=5e-5
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import pandas as pd
import torch
import wandb
from chronos import BaseChronosPipeline, Chronos2Pipeline
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.training.dataset import HypernetTrainingDataset, make_training_collate_fn
from src.training.hypernet import HyperLoRA
from src.training.teacher_cache import load_or_build_teacher_cache
from src.training.trainer import HypernetTrainer
from src.orchestration.context_encoder import ChronosContextEncoder
from src.utils import utils as shared_utils
from src.utils.wandb_utils import init_wandb, finish_wandb


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_dataset(
    df_long: pd.DataFrame,
    cfg: DictConfig,
    start_date: str,
    end_date: str,
    sensor_ids: list[str] | None = None,
    long_context_min_steps: int | None = None,
    long_context_max_steps: int | None = None,
    short_context_min_steps: int | None = None,
    short_context_max_steps: int | None = None,
) -> HypernetTrainingDataset:
    """Build a HypernetTrainingDataset from config."""
    t = cfg.training
    return HypernetTrainingDataset(
        df_long=df_long,
        long_context_steps=t.long_context_steps,
        short_context_steps=t.short_context_steps,
        prediction_length=t.prediction_length,
        long_context_min_steps=long_context_min_steps,
        long_context_max_steps=long_context_max_steps,
        short_context_min_steps=short_context_min_steps,
        short_context_max_steps=short_context_max_steps,
        n_queries_per_context=t.n_queries_per_context,
        query_stride_steps=t.query_stride_steps,
        long_context_stride_steps=t.long_context_stride_steps,
        train_start=pd.Timestamp(start_date),
        train_end=pd.Timestamp(end_date),
        freq_minutes=cfg.freq,
        id_col=cfg.id_column,
        ts_col=cfg.timestamp_column,
        target_col=cfg.target,
        sensor_ids=sensor_ids,
    )


def _resolve_bounds(
    *,
    base_steps: int,
    sigma_outer: float,
    sigma_inner: float,
    min_steps: int | None,
    max_steps: int | None,
) -> tuple[int, int]:
    """Resolve min/max bounds for length sampling.

    If explicit bounds are not provided, uses +/- 3 sigma_total where
    sigma_total = sqrt(sigma_outer^2 + sigma_inner^2).
    """
    sigma_total = math.sqrt(max(0.0, sigma_outer ** 2 + sigma_inner ** 2))
    spread = int(round(3.0 * sigma_total))

    if min_steps is None:
        min_steps = max(1, int(base_steps) - spread)
    if max_steps is None:
        max_steps = max(int(base_steps), int(base_steps) + spread)

    min_steps = int(min_steps)
    max_steps = int(max_steps)
    if min_steps <= 0:
        raise ValueError("Resolved min_steps must be > 0")
    if min_steps > max_steps:
        raise ValueError("Resolved min_steps must be <= max_steps")
    return min_steps, max_steps


def _build_length_jitter_config(
    length_jitter_cfg,
    train_bounds: dict[str, int] | None,
) -> dict | None:
    """Build collate jitter config with resolved bounds."""
    if length_jitter_cfg is None or train_bounds is None:
        return None

    return {
        "enabled": True,
        "quantize_steps": int(length_jitter_cfg.get("quantize_steps", 1)),
        "long_sigma_outer": float(length_jitter_cfg.get("long_sigma_outer", 0.0)),
        "long_sigma_inner": float(length_jitter_cfg.get("long_sigma_inner", 0.0)),
        "short_sigma_outer": float(length_jitter_cfg.get("short_sigma_outer", 0.0)),
        "short_sigma_inner": float(length_jitter_cfg.get("short_sigma_inner", 0.0)),
        "long_min_steps": train_bounds["long_context_min_steps"],
        "long_max_steps": train_bounds["long_context_max_steps"],
        "short_min_steps": train_bounds["short_context_min_steps"],
        "short_max_steps": train_bounds["short_context_max_steps"],
    }


def _save_station_split(
    checkpoint_dir: str | Path,
    train_station_ids: list[str],
    holdout_station_ids: list[str],
    train_fraction: float,
    station_split_seed: int,
    stratify_by_mean: bool,
) -> Path:
    """Persist train/holdout station IDs and split metadata to JSON.

    Returns the path to the written ``station_split.json`` file.
    """
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "station_split.json"

    payload = {
        "train_station_ids": list(train_station_ids),
        "holdout_station_ids": list(holdout_station_ids),
        "n_train_stations": int(len(train_station_ids)),
        "n_holdout_stations": int(len(holdout_station_ids)),
        "train_fraction": float(train_fraction),
        "station_split_seed": int(station_split_seed),
        "stratify_by_mean": bool(stratify_by_mean),
    }
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return split_path


@hydra.main(
    config_path="../../conf",
    config_name="experiment_training",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Merge dataset config
    dataset_cfg_path = cfg.dataset_cfg
    dataset_cfg = OmegaConf.load(f"conf/{dataset_cfg_path}.yaml")
    cfg = OmegaConf.merge(dataset_cfg, cfg)

    print("=== Hypernetwork Training ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    shared_utils.set_seed(cfg.seed)

    target_mode = str(cfg.training_loop.get("target_mode", "teacher")).lower()
    if target_mode not in {"teacher", "ground_truth"}:
        raise ValueError(
            f"Unsupported training_loop.target_mode={target_mode!r}. "
            "Expected 'teacher' or 'ground_truth'."
        )
    print(f"Target supervision mode: {target_mode}")

    # --- WandB ---
    init_wandb(cfg, group="training")

    # --- Device ---
    device = _resolve_device()
    print(f"Device: {device}")

    # --- Load dataset ---
    df_long = shared_utils.load_dataset(cfg)
    print(f"Dataset loaded: {len(df_long)} rows")

    # --- Build train/val datasets ---
    t = cfg.training
    station_holdout_cfg = t.get("station_holdout", None)
    station_holdout_enabled = bool(
        station_holdout_cfg and station_holdout_cfg.get("enabled", False)
    )

    train_station_ids: list[str] | None = None
    holdout_station_ids: list[str] = []
    if station_holdout_enabled:
        all_station_ids = sorted(df_long[cfg.id_column].dropna().astype(str).unique().tolist())
        train_fraction = float(station_holdout_cfg.get("train_fraction", 0.2))
        split_seed_cfg = station_holdout_cfg.get("station_split_seed")
        split_seed = int(cfg.seed if split_seed_cfg is None else split_seed_cfg)
        stratify_by_mean = bool(station_holdout_cfg.get("stratify_by_mean", True))

        train_station_ids, holdout_station_ids = shared_utils.split_stations(
            sensor_ids=all_station_ids,
            df_long=df_long,
            train_fraction=train_fraction,
            seed=split_seed,
            stratify_by_mean=stratify_by_mean,
            id_col=cfg.id_column,
            target_col=cfg.target,
        )

        split_path = _save_station_split(
            checkpoint_dir=cfg.training_loop.checkpoint_dir,
            train_station_ids=train_station_ids,
            holdout_station_ids=holdout_station_ids,
            train_fraction=train_fraction,
            station_split_seed=split_seed,
            stratify_by_mean=stratify_by_mean,
        )

        print(
            "Station holdout enabled: "
            f"train={len(train_station_ids)}, holdout={len(holdout_station_ids)}"
        )
        print(f"Saved station split to: {split_path}")

        if wandb.run is not None:
            wandb.config.update(
                {
                    "n_train_stations": int(len(train_station_ids)),
                    "n_holdout_stations": int(len(holdout_station_ids)),
                    "station_split_path": str(split_path),
                },
                allow_val_change=True,
            )

    length_jitter_cfg = t.get("length_jitter", None)
    jitter_enabled = bool(length_jitter_cfg and length_jitter_cfg.get("enabled", False))
    jitter_apply_in_val = bool(
        length_jitter_cfg and length_jitter_cfg.get("apply_in_val", False)
    )

    train_bounds = None
    if jitter_enabled:
        long_sigma_outer = float(length_jitter_cfg.get("long_sigma_outer", 0.0))
        long_sigma_inner = float(length_jitter_cfg.get("long_sigma_inner", 0.0))
        short_sigma_outer = float(length_jitter_cfg.get("short_sigma_outer", 0.0))
        short_sigma_inner = float(length_jitter_cfg.get("short_sigma_inner", 0.0))

        train_long_min, train_long_max = _resolve_bounds(
            base_steps=int(t.long_context_steps),
            sigma_outer=long_sigma_outer,
            sigma_inner=long_sigma_inner,
            min_steps=length_jitter_cfg.get("long_min_steps", None),
            max_steps=length_jitter_cfg.get("long_max_steps", None),
        )
        train_short_min, train_short_max = _resolve_bounds(
            base_steps=int(t.short_context_steps),
            sigma_outer=short_sigma_outer,
            sigma_inner=short_sigma_inner,
            min_steps=length_jitter_cfg.get("short_min_steps", None),
            max_steps=length_jitter_cfg.get("short_max_steps", None),
        )
        train_bounds = {
            "long_context_min_steps": train_long_min,
            "long_context_max_steps": train_long_max,
            "short_context_min_steps": train_short_min,
            "short_context_max_steps": train_short_max,
        }

    train_dataset = _build_dataset(
        df_long,
        cfg,
        t.train_start_date,
        t.train_end_date,
        sensor_ids=train_station_ids,
        **(train_bounds or {}),
    )

    val_bounds = train_bounds if (jitter_enabled and jitter_apply_in_val) else None
    val_dataset = _build_dataset(
        df_long,
        cfg,
        t.val_start_date,
        t.val_end_date,
        sensor_ids=train_station_ids,
        **(val_bounds or {}),
    )

    # --- Load Chronos-2 (teacher/student predictions) ---
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map=device,
    )
    print("Chronos-2 loaded")

    teacher_model_name = str(getattr(pipeline.model.config, "_name_or_path", "amazon/chronos-2"))
    train_jitter_cfg = _build_length_jitter_config(length_jitter_cfg, train_bounds)
    val_jitter_cfg = (
        _build_length_jitter_config(length_jitter_cfg, train_bounds)
        if (jitter_enabled and jitter_apply_in_val)
        else None
    )

    if target_mode == "teacher":
        train_teacher_cache = load_or_build_teacher_cache(
            split_name="train",
            cfg=cfg,
            dataset=train_dataset,
            teacher_model=pipeline.model,
            teacher_model_name=teacher_model_name,
            device=device,
            jitter_enabled=jitter_enabled,
            jitter_cfg=train_jitter_cfg,
        )
        val_teacher_cache = load_or_build_teacher_cache(
            split_name="val",
            cfg=cfg,
            dataset=val_dataset,
            teacher_model=pipeline.model,
            teacher_model_name=teacher_model_name,
            device=device,
            jitter_enabled=(jitter_enabled and jitter_apply_in_val),
            jitter_cfg=val_jitter_cfg,
        )
    else:
        train_teacher_cache = None
        val_teacher_cache = None
        print("Teacher cache skipped (ground_truth supervision mode)")

    train_build_teacher_contexts = (target_mode == "teacher") and (train_teacher_cache is None)
    val_build_teacher_contexts = (target_mode == "teacher") and (val_teacher_cache is None)
    train_collate_jitter = (
        train_jitter_cfg if (target_mode == "ground_truth" or train_teacher_cache is None) else None
    )
    val_collate_jitter = (
        val_jitter_cfg if (target_mode == "ground_truth" or val_teacher_cache is None) else None
    )

    train_collate_fn = make_training_collate_fn(
        long_context_steps=t.long_context_steps,
        short_context_steps=t.short_context_steps,
        prediction_length=t.prediction_length,
        n_queries_per_context=t.n_queries_per_context,
        query_stride_steps=t.query_stride_steps,
        length_jitter=train_collate_jitter,
        fixed_long_steps=(
            train_teacher_cache.sampled_long_steps if train_teacher_cache is not None else None
        ),
        fixed_short_steps=(
            train_teacher_cache.sampled_short_steps if train_teacher_cache is not None else None
        ),
        build_teacher_contexts=train_build_teacher_contexts,
    )
    val_collate_fn = make_training_collate_fn(
        long_context_steps=t.long_context_steps,
        short_context_steps=t.short_context_steps,
        prediction_length=t.prediction_length,
        n_queries_per_context=t.n_queries_per_context,
        query_stride_steps=t.query_stride_steps,
        length_jitter=val_collate_jitter,
        fixed_long_steps=(
            val_teacher_cache.sampled_long_steps if val_teacher_cache is not None else None
        ),
        fixed_short_steps=(
            val_teacher_cache.sampled_short_steps if val_teacher_cache is not None else None
        ),
        build_teacher_contexts=val_build_teacher_contexts,
    )

    if jitter_enabled:
        print("Length jitter enabled (hierarchical batch sampling):")
        print(
            "  long steps bounds: "
            f"[{train_bounds['long_context_min_steps']}, {train_bounds['long_context_max_steps']}]"
        )
        print(
            "  short steps bounds: "
            f"[{train_bounds['short_context_min_steps']}, {train_bounds['short_context_max_steps']}]"
        )
        print(f"  apply in validation: {jitter_apply_in_val}")

    if train_teacher_cache is not None:
        print(
            "Teacher cache (train): "
            f"{train_teacher_cache.cache_path} (key={train_teacher_cache.cache_key})"
        )
    if val_teacher_cache is not None:
        print(
            "Teacher cache (val): "
            f"{val_teacher_cache.cache_path} (key={val_teacher_cache.cache_key})"
        )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=t.train_batch_size,
        shuffle=True,
        num_workers=t.num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=t.train_batch_size,
        shuffle=False,
        num_workers=t.num_workers,
        collate_fn=val_collate_fn,
        pin_memory=pin_memory,
    )

    # --- Load context encoder model (embeddings for hypernetwork) ---
    context_model_name = cfg.get("context_encoder_model", "amazon/chronos-2")
    context_pipeline = BaseChronosPipeline.from_pretrained(
        context_model_name, device_map=device,
    )
    num_encoder_layers = cfg.get("context_encoder_num_layers", None)
    if num_encoder_layers is not None:
        num_encoder_layers = int(num_encoder_layers)
    context_encoder = ChronosContextEncoder(
        context_pipeline, num_encoder_layers=num_encoder_layers,
    )
    context_d_model = context_pipeline.model.config.d_model
    max_ctx = context_encoder.max_context_length
    print(
        f"Context encoder loaded: {context_model_name} "
        f"(d_model={context_d_model}, max_context_length={max_ctx}"
        f", num_encoder_layers={context_encoder.num_encoder_layers})"
    )

    # Validate that no context length can exceed the encoder's capacity.
    max_long = int(t.long_context_steps)
    if jitter_enabled and length_jitter_cfg.get("long_max_steps") is not None:
        max_long = max(max_long, int(length_jitter_cfg.long_max_steps))
    if max_long > max_ctx:
        raise ValueError(
            f"long_context_steps (or jitter long_max_steps) = {max_long} exceeds "
            f"context encoder capacity ({max_ctx}) of {context_model_name}. "
            f"Either reduce long_context_steps / long_max_steps or choose a "
            f"model with a larger context window (e.g. amazon/chronos-2 "
            f"supports 8192)."
        )

    # --- Hypernetwork ---
    h = cfg.hypernet
    hypernetwork = HyperLoRA(
        d_input=context_d_model,
        d_latent=h.d_latent,
        lora_rank=h.lora_rank,
        n_latent_queries=h.n_latent_queries,
        num_perceiver_blocks=h.num_perceiver_blocks,
        n_self_attn_per_block=h.n_self_attn_per_block,
        n_heads=h.n_heads,
        num_pre_head_layers=h.num_pre_head_layers,
        dropout=h.dropout,
    )
    n_params = sum(p.numel() for p in hypernetwork.parameters())
    print(f"Hypernetwork params: {n_params:,}")

    # --- Trainer ---
    loop_cfg = cfg.training_loop
    trainer = HypernetTrainer(
        hypernetwork=hypernetwork,
        pipeline=pipeline,
        context_encoder=context_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        lora_scaling=loop_cfg.lora_scaling,
        max_epochs=loop_cfg.max_epochs,
        patience=loop_cfg.patience,
        checkpoint_dir=loop_cfg.checkpoint_dir,
        l1_reg_coef=loop_cfg.l1_reg_coef,
        grad_clip=loop_cfg.grad_clip,
        log_every=loop_cfg.log_every,
        device=device,
        gradient_accumulation_steps=loop_cfg.gradient_accumulation_steps,
        warmup_steps=loop_cfg.warmup_steps,
        train_teacher_cache=(train_teacher_cache.teacher_preds if train_teacher_cache else None),
        val_teacher_cache=(val_teacher_cache.teacher_preds if val_teacher_cache else None),
        target_mode=target_mode,
        quantile_levels=cfg.get("quantile_levels", None),
    )

    # --- Train ---
    best_path = trainer.train()
    print(f"\nTraining complete. Best checkpoint: {best_path}")

    finish_wandb()


if __name__ == "__main__":
    main()
