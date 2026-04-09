"""CLI entrypoint for hypernetwork training.

Usage:
    uv run python -m src.training.main
    uv run python -m src.training.main wandb.enabled=true
    uv run python -m src.training.main training.train_batch_size=8 optimizer.lr=5e-5
"""

from __future__ import annotations

import hydra
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.training.dataset import HypernetTrainingDataset, collate_training_batch
from src.training.hypernet import HyperLoRA
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
) -> HypernetTrainingDataset:
    """Build a HypernetTrainingDataset from config."""
    t = cfg.training
    return HypernetTrainingDataset(
        df_long=df_long,
        long_context_steps=t.long_context_steps,
        short_context_steps=t.short_context_steps,
        prediction_length=t.prediction_length,
        n_queries_per_context=t.n_queries_per_context,
        query_stride_steps=t.query_stride_steps,
        long_context_stride_steps=t.long_context_stride_steps,
        train_start=pd.Timestamp(start_date),
        train_end=pd.Timestamp(end_date),
        freq_minutes=cfg.freq,
        id_col=cfg.id_column,
        ts_col=cfg.timestamp_column,
        target_col=cfg.target,
    )


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
    train_dataset = _build_dataset(df_long, cfg, t.train_start_date, t.train_end_date)
    val_dataset = _build_dataset(df_long, cfg, t.val_start_date, t.val_end_date)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=t.train_batch_size,
        shuffle=True,
        num_workers=t.num_workers,
        collate_fn=collate_training_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=t.train_batch_size,
        shuffle=False,
        num_workers=t.num_workers,
        collate_fn=collate_training_batch,
    )

    # --- Load Chronos-2 ---
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map=device,
    )
    print("Chronos-2 loaded")

    # --- Context encoder ---
    context_encoder = ChronosContextEncoder(pipeline)

    # --- Hypernetwork ---
    h = cfg.hypernet
    hypernetwork = HyperLoRA(
        d_input=768,  # Chronos-2 Base hidden dim
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
    )

    # --- Train ---
    best_path = trainer.train()
    print(f"\nTraining complete. Best checkpoint: {best_path}")

    finish_wandb()


if __name__ == "__main__":
    main()
