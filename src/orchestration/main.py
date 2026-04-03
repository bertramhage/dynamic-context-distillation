"""Hydra CLI entrypoint for the orchestration layer."""

import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluation.main import _print_final_metrics, _print_runtime_metrics
from src.orchestration.run import run_orchestration
from src.utils.wandb_utils import init_wandb, finish_wandb


@hydra.main(version_base=None, config_path="../../conf", config_name="experiment_orchestration")
def main(cfg: DictConfig):
    dataset_cfg_path = cfg.dataset_cfg
    dataset_cfg = OmegaConf.load(f"conf/{dataset_cfg_path}.yaml")
    cfg = OmegaConf.merge(dataset_cfg, cfg)
    print(OmegaConf.to_yaml(cfg))

    wb_run = init_wandb(cfg, group="orchestration")

    horizon_metrics, runtime_stats = run_orchestration(cfg)

    _print_final_metrics(cfg, horizon_metrics)
    _print_runtime_metrics(runtime_stats)

    if wb_run is not None:
        all_horizons = list(horizon_metrics.keys())
        if all_horizons:
            sample_metrics = next(iter(horizon_metrics.values()))
            for metric_name in sample_metrics.keys():
                per_horizon_avgs = []
                for h in all_horizons:
                    vals = horizon_metrics[h][metric_name]
                    if isinstance(vals, list) and vals:
                        per_horizon_avgs.append(sum(vals) / len(vals))
                if per_horizon_avgs:
                    wandb.summary[f"eval/avg_{metric_name}"] = (
                        sum(per_horizon_avgs) / len(per_horizon_avgs)
                    )

        for k, v in runtime_stats.items():
            if v is not None:
                wandb.summary[f"runtime/{k}"] = v

        finish_wandb()


if __name__ == "__main__":
    main()
