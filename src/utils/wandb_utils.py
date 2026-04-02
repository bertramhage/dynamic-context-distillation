import wandb
from omegaconf import OmegaConf

PROJECT = "advanced-ba"


def init_wandb(cfg, group: str):
    """Initialize a WandB run from Hydra config.

    Args:
        cfg: Hydra config with an optional ``wandb`` section.
        group: Hard-coded group name for the layer (e.g. "evaluation", "training").

    Returns the run, or None if disabled.
    If a run is already active (e.g. started by an outer layer), returns it as-is.
    """
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None or not wandb_cfg.get("enabled", False):
        return None
    if wandb.run is not None:
        return wandb.run
    return wandb.init(
        project=PROJECT,
        entity=wandb_cfg.get("entity", None),
        group=group,
        name=wandb_cfg.get("run_name", None),
        tags=list(wandb_cfg.get("tags", [])),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def finish_wandb():
    """Finish the active WandB run, if any."""
    if wandb.run is not None:
        wandb.finish()
