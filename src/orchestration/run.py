from __future__ import annotations

import json
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


# Data helpers

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


# Prediction-time schedule (mirrors evaluation layer logic)

def _compute_prediction_times(
    cfg,
    df_long: pd.DataFrame,
) -> list[pd.Timestamp]:
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


# Long-history window helpers

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


def _resolve_orchestration_dataset_cfg(cfg):
    """Return runtime cfg with optional dataset override for cross-dataset eval.

    If ``orchestration.eval_dataset_cfg`` is set (for example
    ``dataset/METR-LA``), the referenced dataset config is merged on top of the
    current config so dataset/time-range fields come from the evaluation dataset
    while orchestration/hypernetwork settings remain unchanged.
    """
    eval_dataset_cfg = getattr(cfg.orchestration, "eval_dataset_cfg", None)
    if eval_dataset_cfg is None:
        return cfg

    eval_dataset_cfg = str(eval_dataset_cfg).strip()
    if eval_dataset_cfg == "" or eval_dataset_cfg.lower() == "null":
        return cfg

    dataset_cfg_path = Path("conf") / f"{eval_dataset_cfg}.yaml"
    if not dataset_cfg_path.exists():
        raise FileNotFoundError(
            "orchestration.eval_dataset_cfg not found: "
            f"{dataset_cfg_path}"
        )

    eval_dataset_cfg_obj = OmegaConf.load(str(dataset_cfg_path))
    runtime_cfg = OmegaConf.merge(cfg, eval_dataset_cfg_obj)
    print(
        "Cross-dataset evaluation enabled: "
        f"using {eval_dataset_cfg} "
        f"(dataset={runtime_cfg.dataset})"
    )
    return runtime_cfg


class _DynamicLoraProvider:
    """On-demand batched LoRA provider for dynamic evaluation runtime."""

    def __init__(
        self,
        cfg,
        df_long: pd.DataFrame,
        hypernetwork,
        context_encoder: ChronosContextEncoder,
        sensor_ids: list[str],
        encode_batch_size: int,
    ) -> None:
        self.cfg = cfg
        self.df_long = df_long
        self.hypernetwork = hypernetwork
        self.context_encoder = context_encoder
        self.sensor_ids = sensor_ids
        self.encode_batch_size = int(encode_batch_size)
        self.id_col = cfg.id_column
        self.ts_col = cfg.timestamp_column
        self.target_col = cfg.target
        self.rolling = bool(getattr(cfg.orchestration, "rolling_long_history", False))

        self._fixed_cache: dict | None = None
        self._rolling_cache_time: pd.Timestamp | None = None
        self._rolling_cache: dict | None = None

    def _compute_cache(self, prediction_time: pd.Timestamp | None) -> dict:
        """Compute (or recompute) cached LoRA tensors for a prediction step."""
        if self.rolling:
            if prediction_time is None:
                raise ValueError("prediction_time must be provided for rolling dynamic LoRA.")
            start_time, end_time = _long_history_window_rolling(self.cfg, prediction_time)
        else:
            start_time, end_time = _long_history_window_fixed(self.cfg)

        context_tensor, ordered_ids = _extract_sensor_tensor(
            self.df_long,
            self.sensor_ids,
            start_time,
            end_time,
            self.id_col,
            self.ts_col,
            self.target_col,
        )

        all_z = self.context_encoder.encode_last_hidden_batched(
            context_tensor,
            batch_size=self.encode_batch_size,
        )
        with torch.no_grad():
            lora_dict = self.hypernetwork(all_z)

        cpu_lora_dict = {}
        for short_name, weights in lora_dict.items():
            cpu_lora_dict[short_name] = {
                "A": weights["A"].detach().to(device="cpu", dtype=torch.float32),
                "B": weights["B"].detach().to(device="cpu", dtype=torch.float32),
            }

        id_to_idx = {str(sensor_id): idx for idx, sensor_id in enumerate(ordered_ids)}
        return {
            "id_to_idx": id_to_idx,
            "lora_dict": cpu_lora_dict,
        }

    def _get_cache(self, prediction_time: pd.Timestamp) -> dict:
        """Get current cache for fixed/rolling long-history mode."""
        if not self.rolling:
            if self._fixed_cache is None:
                self._fixed_cache = self._compute_cache(prediction_time=None)
            return self._fixed_cache

        prediction_time = pd.Timestamp(prediction_time)
        if self._rolling_cache_time != prediction_time:
            self._rolling_cache = self._compute_cache(prediction_time=prediction_time)
            self._rolling_cache_time = prediction_time
        return self._rolling_cache

    def get_step_lora_batch(
        self,
        prediction_time: pd.Timestamp,
        item_ids: list[str],
        adapter_ids: list[str],
        default_adapter_id: str,
        device: torch.device,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Build batched LoRA tensors for one evaluation step.

        For `default_adapter_id` tasks, this returns zero-LoRA rows so base
        behavior is preserved for those samples.
        """
        if len(item_ids) != len(adapter_ids):
            raise ValueError("item_ids and adapter_ids must have identical lengths.")

        cache = self._get_cache(prediction_time=prediction_time)
        id_to_idx = cache["id_to_idx"]
        lora_dict = cache["lora_dict"]

        out = {
            short_name: {"A": [], "B": []}
            for short_name in lora_dict.keys()
        }
        zero_cache = {
            short_name: {
                "A": torch.zeros_like(weights["A"][0]),
                "B": torch.zeros_like(weights["B"][0]),
            }
            for short_name, weights in lora_dict.items()
        }

        for item_id, adapter_id in zip(item_ids, adapter_ids):
            item_key = str(item_id)
            if item_key not in id_to_idx:
                raise ValueError(
                    f"Missing dynamic LoRA tensors for item_id '{item_key}' at step {prediction_time}."
                )
            src_idx = id_to_idx[item_key]
            use_base = str(adapter_id) == str(default_adapter_id)

            for short_name, weights in lora_dict.items():
                if use_base:
                    out[short_name]["A"].append(zero_cache[short_name]["A"])
                    out[short_name]["B"].append(zero_cache[short_name]["B"])
                else:
                    out[short_name]["A"].append(weights["A"][src_idx])
                    out[short_name]["B"].append(weights["B"][src_idx])

        batched = {}
        for short_name, tensors in out.items():
            batched[short_name] = {
                "A": torch.stack(tensors["A"], dim=0).to(device=device),
                "B": torch.stack(tensors["B"], dim=0).to(device=device),
            }
        return batched


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def _load_station_split(split_path: str | Path) -> tuple[list[str], list[str]]:
    """Load train/holdout station IDs from a saved split JSON file."""
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"station_split_path not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    train_station_ids = payload.get("train_station_ids")
    holdout_station_ids = payload.get("holdout_station_ids")

    if not isinstance(train_station_ids, list) or not isinstance(holdout_station_ids, list):
        raise ValueError(
            "station_split.json must contain list fields 'train_station_ids' and "
            "'holdout_station_ids'."
        )

    train_ids = sorted({str(sid) for sid in train_station_ids})
    holdout_ids = sorted({str(sid) for sid in holdout_station_ids})

    overlap = set(train_ids).intersection(holdout_ids)
    if overlap:
        raise ValueError(
            "Station split has overlapping train/holdout IDs, e.g.: "
            f"{sorted(overlap)[:5]}"
        )
    if not train_ids:
        raise ValueError("station_split.json contains an empty train_station_ids list")
    if not holdout_ids:
        raise ValueError("station_split.json contains an empty holdout_station_ids list")

    return train_ids, holdout_ids


def _resolve_station_eval_sets(
    cfg,
    all_sensor_ids: list[str],
) -> list[tuple[str, list[str]]]:
    """Resolve which station subsets should be evaluated in orchestration."""
    orch = cfg.orchestration
    split_path = getattr(orch, "station_split_path", None)
    eval_set = str(getattr(orch, "station_eval_set", "all")).lower()

    if eval_set not in {"all", "train", "holdout"}:
        raise ValueError(
            "orchestration.station_eval_set must be one of: all, train, holdout"
        )

    if split_path in {None, "", "null"}:
        if eval_set != "all":
            raise ValueError(
                "orchestration.station_eval_set requires orchestration.station_split_path"
            )
        return [("all", all_sensor_ids)]

    split_train_ids, split_holdout_ids = _load_station_split(str(split_path))
    available_ids = set(all_sensor_ids)

    train_ids = sorted(available_ids.intersection(split_train_ids))
    holdout_ids = sorted(available_ids.intersection(split_holdout_ids))

    if not train_ids:
        raise ValueError("No train stations from station_split_path are present in dataset")
    if not holdout_ids:
        raise ValueError("No holdout stations from station_split_path are present in dataset")

    missing_ids = set(split_train_ids).union(split_holdout_ids).difference(available_ids)
    if missing_ids:
        print(
            "Warning: station_split.json contains IDs absent from dataset, "
            f"count={len(missing_ids)}"
        )

    if eval_set == "train":
        return [("train", train_ids)]
    if eval_set == "holdout":
        return [("holdout", holdout_ids)]
    return [("train", train_ids), ("holdout", holdout_ids)]


def _run_single_station_set(
    *,
    cfg,
    pipeline: Chronos2Pipeline,
    hypernetwork,
    context_encoder: ChronosContextEncoder,
    df_long: pd.DataFrame,
    sensor_ids: list[str],
    station_set_name: str,
    use_dynamic_lora: bool,
    rank: int,
    alpha: int,
    output_lora_d_in: int,
    output_lora_d_out: int,
    target_modules: list[str] | None,
) -> tuple[dict, dict]:
    """Run adapter generation + evaluation for one station subset."""
    orch = cfg.orchestration
    id_col = cfg.id_column
    ts_col = cfg.timestamp_column
    target_col = cfg.target

    subset_df = df_long[df_long[id_col].astype(str).isin(sensor_ids)].copy()
    if subset_df.empty:
        raise ValueError(f"No rows found for station set '{station_set_name}'")

    prediction_times = _compute_prediction_times(cfg, subset_df)
    print(
        f"Station set '{station_set_name}': sensors={len(sensor_ids)}, "
        f"prediction_steps={len(prediction_times)}"
    )

    rolling = bool(getattr(orch, "rolling_long_history", False))
    encode_batch_size = int(getattr(orch, "encode_batch_size", 32))

    dynamic_lora_provider = None
    adapter_root = None
    if use_dynamic_lora:
        print(
            f"Dynamic LoRA runtime enabled for station set '{station_set_name}'."
        )
        dynamic_lora_provider = _DynamicLoraProvider(
            cfg=cfg,
            df_long=subset_df,
            hypernetwork=hypernetwork,
            context_encoder=context_encoder,
            sensor_ids=sensor_ids,
            encode_batch_size=encode_batch_size,
        )
    else:
        run_id = getattr(orch, "run_id", None) or str(uuid.uuid4())[:8]
        if station_set_name != "all":
            run_id = f"{run_id}_{station_set_name}"
        adapter_root = Path(str(orch.adapter_output_dir)) / run_id
        adapter_root.mkdir(parents=True, exist_ok=True)
        print(f"Adapters for '{station_set_name}' saved under: {adapter_root}")

    assignment_rows: list[dict] = []

    if not rolling:
        if use_dynamic_lora:
            for sensor_id in sensor_ids:
                for pt in prediction_times:
                    assignment_rows.append(
                        {
                            id_col: sensor_id,
                            "prediction_time": pt,
                            "adapter_id": sensor_id,
                        }
                    )
            print(
                f"Prepared {len(sensor_ids)} dynamic adapters "
                f"(fixed long history, station set '{station_set_name}')"
            )
        else:
            long_start, long_end = _long_history_window_fixed(cfg)
            context_tensor, ordered_ids = _extract_sensor_tensor(
                subset_df,
                sensor_ids,
                long_start,
                long_end,
                id_col,
                ts_col,
                target_col,
            )

            all_z = context_encoder.encode_last_hidden_batched(
                context_tensor,
                batch_size=encode_batch_size,
            )
            lora_dict = hypernetwork(all_z)

            for sensor_batch_idx, sensor_id in enumerate(ordered_ids):
                adapter_dir = adapter_root / sensor_id
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

                for pt in prediction_times:
                    assignment_rows.append(
                        {
                            id_col: sensor_id,
                            "prediction_time": pt,
                            "adapter_id": sensor_id,
                        }
                    )

            print(
                f"Saved {len(ordered_ids)} adapters "
                f"(fixed long history, station set '{station_set_name}')"
            )

    else:
        if use_dynamic_lora:
            for step_idx, pt in enumerate(prediction_times):
                for sensor_id in sensor_ids:
                    assignment_rows.append(
                        {
                            id_col: sensor_id,
                            "prediction_time": pt,
                            "adapter_id": f"{sensor_id}_step{step_idx}",
                        }
                    )
            print(
                f"Prepared {len(prediction_times) * len(sensor_ids)} dynamic adapters "
                f"(rolling long history, station set '{station_set_name}')"
            )
        else:
            for step_idx, pt in enumerate(prediction_times):
                long_start, long_end = _long_history_window_rolling(cfg, pt)
                context_tensor, ordered_ids = _extract_sensor_tensor(
                    subset_df,
                    sensor_ids,
                    long_start,
                    long_end,
                    id_col,
                    ts_col,
                    target_col,
                )

                all_z = context_encoder.encode_last_hidden_batched(
                    context_tensor,
                    batch_size=encode_batch_size,
                )
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
                        f"[{station_set_name}] Step {step_idx + 1}/{len(prediction_times)}: "
                        f"generated {len(ordered_ids)} adapters"
                    )

            print(
                f"Saved {len(prediction_times) * len(sensor_ids)} adapters "
                f"(rolling long history, station set '{station_set_name}')"
            )

    assignments_df = pd.DataFrame(assignment_rows)

    eval_cfg = OmegaConf.to_container(cfg, resolve=True)
    eval_cfg = OmegaConf.create(eval_cfg)
    OmegaConf.update(
        eval_cfg,
        "evaluation.history_length_steps",
        int(orch.short_context_length_steps),
    )
    if not use_dynamic_lora:
        OmegaConf.update(eval_cfg, "adapter.adapter_root", str(adapter_root))

    print(
        f"\nStarting evaluation for station set '{station_set_name}' with "
        f"short context = {orch.short_context_length_steps} steps, "
        f"assignments = {len(assignments_df)}"
    )

    horizon_metrics, runtime_stats = run_evaluation(
        cfg=eval_cfg,
        pipeline=pipeline,
        df_long=subset_df,
        assignments_df=assignments_df,
        dynamic_lora_provider=dynamic_lora_provider,
        return_runtime=True,
    )
    return horizon_metrics, runtime_stats


def run_orchestration(cfg) -> tuple[dict, dict]:
    """Generate LoRA adapters from a trained hypernetwork and run evaluation.

    Without a station split this returns one ``(horizon_metrics, runtime_stats)`` pair.
    With ``orchestration.station_split_path`` and ``station_eval_set=all``, this
    returns two dictionaries keyed by ``train`` and ``holdout``.
    """
    cfg = _resolve_orchestration_dataset_cfg(cfg)
    orch = cfg.orchestration
    shared_utils.set_seed(cfg.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    df_long = shared_utils.load_dataset(cfg)

    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map=device,
    )

    checkpoint_path = str(orch.checkpoint_path)
    print(f"Loading hypernetwork from {checkpoint_path}")
    hypernetwork = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hypernetwork.eval()
    for p in hypernetwork.parameters():
        p.requires_grad = False

    expected_context_d = None
    if hasattr(hypernetwork, "perceiver") and hasattr(hypernetwork.perceiver, "d_input"):
        expected_context_d = int(hypernetwork.perceiver.d_input)

    context_model_name = str(cfg.get("context_encoder_model", "amazon/chronos-2"))
    context_pipeline = BaseChronosPipeline.from_pretrained(
        context_model_name, device_map=device,
    )
    num_encoder_layers = cfg.get("context_encoder_num_layers", None)
    if num_encoder_layers is not None:
        num_encoder_layers = int(num_encoder_layers)
    context_encoder = ChronosContextEncoder(
        context_pipeline, num_encoder_layers=num_encoder_layers,
    )
    context_d_model = int(context_pipeline.model.config.d_model)
    max_ctx = context_encoder.max_context_length
    print(
        f"Context encoder loaded: {context_model_name} "
        f"(d_model={context_d_model}, max_context_length={max_ctx}"
        f", num_encoder_layers={context_encoder.num_encoder_layers})"
    )

    long_steps = int(cfg.orchestration.long_history_length_steps)
    if long_steps > max_ctx:
        raise ValueError(
            f"long_history_length_steps={long_steps} exceeds context encoder "
            f"capacity ({max_ctx}) of {context_model_name}. "
            f"Either reduce long_history_length_steps or choose a model with "
            f"a larger context window (e.g. amazon/chronos-2 supports 8192)."
        )

    if expected_context_d is not None and context_d_model != expected_context_d:
        raise ValueError(
            "Context encoder dimension mismatch with checkpoint: "
            f"hypernetwork expects d_input={expected_context_d}, "
            f"but {context_model_name} provides d_model={context_d_model}. "
            "Set context_encoder_model to the model used during training."
        )

    id_col = cfg.id_column
    sensor_ids = sorted(df_long[id_col].dropna().astype(str).unique().tolist())
    print(f"Number of sensors: {len(sensor_ids)}")

    evaluation_cfg = getattr(cfg, "evaluation", None)
    use_dynamic_lora = bool(
        evaluation_cfg is not None and evaluation_cfg.get("use_dynamic_lora", False)
    )

    adapter_cfg = getattr(cfg, "adapter", None)
    rank = 8 if adapter_cfg is None else int(adapter_cfg.get("rank", 8))
    alpha = 16 if adapter_cfg is None else int(adapter_cfg.get("alpha", 16))
    output_lora_d_in, output_lora_d_out = _infer_output_layer_lora_dims(pipeline)
    target_modules = None
    if adapter_cfg is not None and adapter_cfg.get("target_modules") is not None:
        target_modules = list(adapter_cfg.target_modules)

    station_sets = _resolve_station_eval_sets(cfg, sensor_ids)
    if len(station_sets) == 1:
        station_set_name, station_sensor_ids = station_sets[0]
        return _run_single_station_set(
            cfg=cfg,
            pipeline=pipeline,
            hypernetwork=hypernetwork,
            context_encoder=context_encoder,
            df_long=df_long,
            sensor_ids=station_sensor_ids,
            station_set_name=station_set_name,
            use_dynamic_lora=use_dynamic_lora,
            rank=rank,
            alpha=alpha,
            output_lora_d_in=output_lora_d_in,
            output_lora_d_out=output_lora_d_out,
            target_modules=target_modules,
        )

    all_horizon_metrics: dict[str, dict] = {}
    all_runtime_stats: dict[str, dict] = {}

    for station_set_name, station_sensor_ids in station_sets:
        horizon_metrics, runtime_stats = _run_single_station_set(
            cfg=cfg,
            pipeline=pipeline,
            hypernetwork=hypernetwork,
            context_encoder=context_encoder,
            df_long=df_long,
            sensor_ids=station_sensor_ids,
            station_set_name=station_set_name,
            use_dynamic_lora=use_dynamic_lora,
            rank=rank,
            alpha=alpha,
            output_lora_d_in=output_lora_d_in,
            output_lora_d_out=output_lora_d_out,
            target_modules=target_modules,
        )
        all_horizon_metrics[station_set_name] = horizon_metrics
        all_runtime_stats[station_set_name] = runtime_stats

    return all_horizon_metrics, all_runtime_stats
