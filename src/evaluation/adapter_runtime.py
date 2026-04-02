from __future__ import annotations

from pathlib import Path

import pandas as pd
from peft import LoraConfig, PeftModel, get_peft_model

DEFAULT_TARGET_MODULES = [
    "self_attention.q",
    "self_attention.k",
    "self_attention.v",
    "self_attention.o",
    "output_patch_embedding.output_layer",
]

def build_assignment_mapping(
    assignments_df: pd.DataFrame | None,
    item_id_col: str = "item_id",
    prediction_time_col: str = "prediction_time",
    adapter_id_col: str = "adapter_id",
) -> dict[tuple[str, pd.Timestamp], str]:
    """Build in-memory task-to-adapter mapping from orchestrator assignments.

    Returns a dictionary keyed by ``(item_id, prediction_time)`` so evaluation
    can resolve which adapter ID to activate for each prediction task.
    """
    if assignments_df is None:
        return {}

    missing = {item_id_col, prediction_time_col, adapter_id_col}.difference(
        assignments_df.columns
    )
    if missing:
        raise ValueError(f"Assignments dataframe missing required columns: {sorted(missing)}")

    df = assignments_df[[item_id_col, prediction_time_col, adapter_id_col]].copy()
    df[prediction_time_col] = pd.to_datetime(df[prediction_time_col])

    return {
        (str(row[item_id_col]), pd.Timestamp(row[prediction_time_col])): str(
            row[adapter_id_col]
        )
        for _, row in df.iterrows()
    }


def resolve_adapter_id(
    mapping: dict[tuple[str, pd.Timestamp], str],
    item_id: str,
    prediction_time: pd.Timestamp,
    default_adapter_id: str = "__none__",
) -> str:
    """Resolve adapter ID for a single prediction task.

    Falls back to ``default_adapter_id`` when no explicit assignment exists.
    """
    return mapping.get((str(item_id), pd.Timestamp(prediction_time)), default_adapter_id)


def ensure_peft_model(pipeline, cfg) -> PeftModel:
    """Ensure the pipeline model is PEFT-wrapped and ready for adapter switching.

    If the model is still a plain Chronos model, this wraps it with a LoRA
    configuration derived from ``cfg.adapter`` and returns the resulting
    ``PeftModel``.
    """
    if isinstance(pipeline.model, PeftModel):
        return pipeline.model

    adapter_cfg = getattr(cfg, "adapter", None)
    rank = 8 if adapter_cfg is None else int(adapter_cfg.get("rank", 8))
    alpha = 16 if adapter_cfg is None else int(adapter_cfg.get("alpha", 16))
    targets = DEFAULT_TARGET_MODULES
    if adapter_cfg is not None and adapter_cfg.get("target_modules") is not None:
        targets = list(adapter_cfg.target_modules)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        inference_mode=True,
    )
    pipeline.model = get_peft_model(pipeline.model, lora_config)
    pipeline.model.eval()
    return pipeline.model


def apply_adapter(pipeline, adapter_id: str, cfg, loaded_adapters: set[str]) -> bool:
    """Activate an adapter on the pipeline model for inference.

    Returns ``True`` when the caller should run the base model path
    (``adapter_id == '__none__'``). Otherwise loads and activates the requested
    PEFT adapter and returns ``False``.
    """
    if adapter_id == "__none__":
        return True

    peft_model = ensure_peft_model(pipeline, cfg)

    adapter_cfg = getattr(cfg, "adapter", None)
    adapter_root = "outputs/adapters" if adapter_cfg is None else str(
        adapter_cfg.get("adapter_root", "outputs/adapters")
    )
    adapter_path = Path(adapter_root) / adapter_id
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found for id '{adapter_id}': {adapter_path}"
        )

    if adapter_id not in loaded_adapters:
        peft_model.load_adapter(
            str(adapter_path),
            adapter_name=adapter_id,
            is_trainable=False,
        )
        loaded_adapters.add(adapter_id)

    peft_model.set_adapter(adapter_id)
    return False
