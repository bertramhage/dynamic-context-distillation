import pandas as pd
import numpy as np
import torch
import time
import resource
import sys
from peft import PeftModel
from chronos import BaseChronosPipeline, Chronos2Pipeline

import hydra
from omegaconf import DictConfig, OmegaConf
from src.evaluation.adapter_runtime import (
    apply_adapter,
    build_assignment_mapping,
)
from src.utils.metrics import evaluation
from src.utils import utils as shared_utils
from src.utils.utils import calculate_metrics, metrics, probabilistic_metrics


def _print_final_metrics(cfg, horizon_metrics: dict) -> None:
    print("Final Average Metrics per Horizon:")

    overall = {"MAE": [], "MAPE": [], "RMSE": [], "COVERAGE": [], "IQR_MEAN": []}

    for h in cfg.horizons:
        print(f"\nHorizon {h} minutes:")
        avg_mae = np.mean(horizon_metrics[h]["MAE"])
        avg_mape = np.mean(horizon_metrics[h]["MAPE"])
        avg_rmse = np.mean(horizon_metrics[h]["RMSE"])
        avg_coverage = np.mean(horizon_metrics[h]["COVERAGE"])
        avg_iqr = np.mean(horizon_metrics[h]["IQR_MEAN"])

        metric_values = {
            "MAE": avg_mae,
            "MAPE": avg_mape,
            "RMSE": avg_rmse,
            "COVERAGE": avg_coverage,
            "IQR_MEAN": avg_iqr,
        }

        for k, v in metric_values.items():
            overall[k].append(v)

        for name in getattr(cfg, "metrics_to_show", metric_values.keys()):
            if name in metric_values:
                val = metric_values[name]
                if name == "MAPE" or name == "COVERAGE":
                    val *= 100
                    print(f"  {name}: {np.round(val, cfg.decimal_precision)}%")
                else:
                    print(f"  {name}: {np.round(val, cfg.decimal_precision)}")

    print("\nAverage Across Horizons:")
    for name in getattr(cfg, "metrics_to_show", overall.keys()):
        if name in overall and len(overall[name]) > 0:
            val = np.mean(overall[name])
            if name == "MAPE" or name == "COVERAGE":
                val *= 100
                print(f"  {name}: {np.round(val, cfg.decimal_precision)}%")
            else:
                print(f"  {name}: {np.round(val, cfg.decimal_precision)}")


def _print_runtime_metrics(runtime_stats: dict) -> None:
    print("\nRuntime Metrics:")
    print(f"  predict_backend_calls: {runtime_stats['predict_backend_calls']}")
    print(f"  prediction_tasks: {runtime_stats['prediction_tasks']}")
    print(f"  total_inference_seconds: {runtime_stats['total_inference_seconds']:.4f}")
    print(
        f"  avg_task_inference_seconds: {runtime_stats['avg_task_inference_seconds']:.6f}"
    )
    print(
        f"  p95_task_inference_seconds: {runtime_stats['p95_task_inference_seconds']:.6f}"
    )
    if runtime_stats.get("cuda_peak_memory_mb") is not None:
        print(f"  cuda_peak_memory_mb: {runtime_stats['cuda_peak_memory_mb']:.2f}")
    if runtime_stats.get("mps_peak_memory_mb") is not None:
        print(f"  mps_peak_memory_mb: {runtime_stats['mps_peak_memory_mb']:.2f}")
    if runtime_stats.get("cpu_peak_rss_mb") is not None:
        print(f"  cpu_peak_rss_mb: {runtime_stats['cpu_peak_rss_mb']:.2f}")


def run_evaluation(
    cfg,
    pipeline: Chronos2Pipeline,
    df_long: pd.DataFrame,
    assignments_df: pd.DataFrame | None = None,
    return_runtime: bool = False,
) -> dict | tuple[dict, dict]:
    prediction_length = cfg.prediction_length + 1
    id_column = cfg.id_column
    number_of_sensors = df_long[id_column].nunique()
    freq = cfg.freq
    day_slot = int(24 * 60 / freq)
    stride_steps = cfg.stride_steps

    history_length = int(10080 / freq)
    if getattr(cfg, "evaluation", None) is not None and cfg.evaluation.get(
        "history_length_steps"
    ) is not None:
        history_length = int(cfg.evaluation.history_length_steps)

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
            "Either start_test_day and n_test_days or proportion_test must be provided in the config."
        )

    start_date = pd.Timestamp(cfg.start_date)
    experiment_start_time = start_date + pd.Timedelta(days=start_test_day)
    experiment_end_time = experiment_start_time + pd.Timedelta(days=n_test_days)
    pred_delta = pd.Timedelta(minutes=prediction_length * freq)
    stride_delta = pd.Timedelta(minutes=stride_steps * freq)

    print(
        f"Testing from day {start_test_day} to day {start_test_day + n_test_days} (total {n_test_days} days)"
    )

    adapter_cfg = getattr(cfg, "adapter", None)
    if adapter_cfg is not None:
        _ = str(adapter_cfg.get("default_id", "__none__"))

    assignment_mapping = build_assignment_mapping(
        assignments_df=assignments_df,
        item_id_col=id_column,
        prediction_time_col="prediction_time",
        adapter_id_col="adapter_id",
    )

    timestamp_column = cfg.timestamp_column
    horizons = cfg.horizons
    target = cfg.target
    current_pred_time = experiment_start_time
    max_horizon_delta = pd.Timedelta(minutes=max(horizons))
    safe_experiment_end = experiment_end_time - max_horizon_delta
    total_steps = int((safe_experiment_end - experiment_start_time) / stride_delta)

    loaded_adapters: set[str] = set()
    per_task_times: list[float] = []
    predict_backend_calls = 0
    prediction_tasks = 0
    mps_peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    horizon_metrics = {
        h: {
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "COVERAGE": [],
            "IQR_MEAN": [],
            "IQR_MEDIAN": [],
            "IQR_STD": [],
        }
        for h in horizons
    }
    all_preds = {h: [] for h in horizons}
    all_trues = {h: [] for h in horizons}

    context_columns = [id_column, timestamp_column, target]
    if getattr(cfg, "past_covariates", None) is not None:
        for col in cfg.past_covariates:
            if col not in context_columns:
                context_columns.append(col)

    for step_idx in range(1, total_steps + 1):
        context_start_time = current_pred_time - pd.Timedelta(minutes=history_length * freq)
        context_df = df_long[
            (df_long[timestamp_column] >= context_start_time)
            & (df_long[timestamp_column] < current_pred_time)
        ][context_columns].copy()

        test_slice_end = current_pred_time + pred_delta
        test_df = df_long[
            (df_long[timestamp_column] >= current_pred_time)
            & (df_long[timestamp_column] < test_slice_end)
        ].copy()

        future_df = None
        if cfg.future_covariates is not None:
            future_df = df_long[
                (df_long[timestamp_column] >= current_pred_time)
                & (df_long[timestamp_column] < test_slice_end)
            ][[id_column, timestamp_column] + cfg.future_covariates].copy()

        item_ids = (
            context_df[id_column].dropna().astype(str).drop_duplicates().sort_values().tolist()
        )

        assignment_rows = []
        missing_assignments: list[str] = []
        for item in item_ids:
            key = (str(item), pd.Timestamp(current_pred_time))
            adapter_id = assignment_mapping.get(key)
            if adapter_id is None:
                missing_assignments.append(str(item))
                continue

            assignment_rows.append(
                {
                    id_column: item,
                    "adapter_id": str(adapter_id),
                }
            )

        if missing_assignments:
            sample = ", ".join(missing_assignments[:5])
            raise ValueError(
                "Missing adapter assignment for prediction tasks at "
                f"prediction_time={current_pred_time}. "
                f"Missing count={len(missing_assignments)}. Sample item_ids=[{sample}]"
            )

        step_assignments = pd.DataFrame(assignment_rows)

        forecast_chunks = []
        for adapter_id, group_df in step_assignments.groupby("adapter_id", sort=True):
            group_item_ids = set(group_df[id_column].astype(str).tolist())
            context_chunk = context_df[context_df[id_column].astype(str).isin(group_item_ids)]
            if context_chunk.empty:
                continue

            future_chunk = None
            if future_df is not None:
                future_chunk = future_df[future_df[id_column].astype(str).isin(group_item_ids)]

            use_base_model = apply_adapter(
                pipeline=pipeline,
                adapter_id=str(adapter_id),
                cfg=cfg,
                loaded_adapters=loaded_adapters,
            )

            if use_base_model and isinstance(pipeline.model, PeftModel):
                with pipeline.model.disable_adapter():
                    start_t = time.perf_counter()
                    forecast_chunk = pipeline.predict_df(
                        context_chunk,
                        future_df=future_chunk,
                        prediction_length=prediction_length,
                        quantile_levels=cfg.quantile_levels,
                        id_column=cfg.id_column,
                        timestamp_column=cfg.timestamp_column,
                        target=cfg.target,
                        cross_learning=cfg.cross_learning,
                    )
                    elapsed = time.perf_counter() - start_t
            else:
                start_t = time.perf_counter()
                forecast_chunk = pipeline.predict_df(
                    context_chunk,
                    future_df=future_chunk,
                    prediction_length=prediction_length,
                    quantile_levels=cfg.quantile_levels,
                    id_column=cfg.id_column,
                    timestamp_column=cfg.timestamp_column,
                    target=cfg.target,
                    cross_learning=cfg.cross_learning,
                )
                elapsed = time.perf_counter() - start_t

            group_tasks = len(group_item_ids)
            if group_tasks > 0:
                per_task_times.extend([elapsed / group_tasks] * group_tasks)
                prediction_tasks += group_tasks
            predict_backend_calls += 1

            if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
                mps_now_mb = torch.mps.current_allocated_memory() / (1024**2)
                mps_peak_memory_mb = max(mps_peak_memory_mb, float(mps_now_mb))

            forecast_chunks.append(forecast_chunk)

        if not forecast_chunks:
            raise ValueError(f"No forecasts generated at step {step_idx}")
        forecast_df = pd.concat(forecast_chunks, ignore_index=True)

        preds = forecast_df[[id_column, timestamp_column, "predictions"]]
        true_vals = test_df[[id_column, timestamp_column, target]]
        merged_df = pd.merge(true_vals, preds, on=[id_column, timestamp_column], how="inner")

        if cfg.dataset == "UrbanEV":
            for h in horizons:
                target_ts = current_pred_time + pd.Timedelta(minutes=h)
                step_df = merged_df[merged_df[timestamp_column] == target_ts]
                step_true = step_df[target].values
                step_pred = step_df["predictions"].values

                if len(step_true) > 0:
                    all_trues[h].append(step_true)
                    all_preds[h].append(step_pred)

                    forecast_h = forecast_df[forecast_df[timestamp_column] == target_ts]
                    true_h = test_df[test_df[timestamp_column] == target_ts]
                    prob = probabilistic_metrics(
                        forecast_df=forecast_h,
                        true_df=true_h,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        target_column=target,
                    )

                    horizon_metrics[h]["COVERAGE"].append(prob.get("coverage"))
                    horizon_metrics[h]["IQR_MEAN"].append(prob.get("iqr_mean"))
                    horizon_metrics[h]["IQR_MEDIAN"].append(prob.get("iqr_median"))
                    horizon_metrics[h]["IQR_STD"].append(prob.get("iqr_std"))
        else:
            for h in horizons:
                target_ts = current_pred_time + pd.Timedelta(minutes=h)
                step_df = merged_df[merged_df[timestamp_column] == target_ts]
                step_true = step_df[target].values
                step_pred = step_df["predictions"].values

                if len(step_true) > 0:
                    if cfg.dataset == "METR-LA":
                        step_mape, step_mae, step_rmse = calculate_metrics(
                            pd.DataFrame(step_pred), pd.DataFrame(step_true), null_val=0
                        )
                    else:
                        step_mape, step_mae, step_rmse = evaluation(step_true, step_pred)

                    horizon_metrics[h]["MAE"].append(step_mae)
                    horizon_metrics[h]["MAPE"].append(step_mape)
                    horizon_metrics[h]["RMSE"].append(step_rmse)

                    forecast_h = forecast_df[forecast_df[timestamp_column] == target_ts]
                    true_h = test_df[test_df[timestamp_column] == target_ts]
                    prob = probabilistic_metrics(
                        forecast_df=forecast_h,
                        true_df=true_h,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        target_column=target,
                    )

                    horizon_metrics[h]["COVERAGE"].append(prob.get("coverage"))
                    horizon_metrics[h]["IQR_MEAN"].append(prob.get("iqr_mean"))
                    horizon_metrics[h]["IQR_MEDIAN"].append(prob.get("iqr_median"))
                    horizon_metrics[h]["IQR_STD"].append(prob.get("iqr_std"))
                else:
                    raise ValueError(
                        f"No data to evaluate for horizon {h} at step {step_idx} and {current_pred_time}"
                    )

        current_pred_time += stride_delta

    if cfg.dataset == "UrbanEV":
        horizon_metrics = {}
        for h in horizons:
            y_true = np.concatenate(all_trues[h])
            y_pred = np.concatenate(all_preds[h])
            step_mape, step_mae, step_rmse = metrics(y_pred, y_true)
            horizon_metrics[h] = {"MAE": step_mae, "MAPE": step_mape, "RMSE": step_rmse}

    total_inference_seconds = float(np.sum(per_task_times)) if per_task_times else 0.0
    avg_task_inference_seconds = float(np.mean(per_task_times)) if per_task_times else 0.0
    p95_task_inference_seconds = (
        float(np.percentile(per_task_times, 95)) if per_task_times else 0.0
    )

    cuda_peak_memory_mb = None
    if torch.cuda.is_available():
        cuda_peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    mps_memory_out = None
    if torch.backends.mps.is_available() and mps_peak_memory_mb > 0:
        mps_memory_out = float(mps_peak_memory_mb)

    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        cpu_peak_rss_mb = float(ru_maxrss / (1024.0 * 1024.0))
    else:
        cpu_peak_rss_mb = float(ru_maxrss / 1024.0)

    runtime_stats = {
        "predict_backend_calls": int(predict_backend_calls),
        "prediction_tasks": int(prediction_tasks),
        "total_inference_seconds": total_inference_seconds,
        "avg_task_inference_seconds": avg_task_inference_seconds,
        "p95_task_inference_seconds": p95_task_inference_seconds,
        "cuda_peak_memory_mb": cuda_peak_memory_mb,
        "mps_peak_memory_mb": mps_memory_out,
        "cpu_peak_rss_mb": cpu_peak_rss_mb,
    }

    if return_runtime:
        return horizon_metrics, runtime_stats
    return horizon_metrics


@hydra.main(version_base=None, config_path="../../conf", config_name="experiment")
def main(cfg: DictConfig):
    dataset_cfg_path = cfg.dataset_cfg
    dataset_cfg = OmegaConf.load(f"conf/{dataset_cfg_path}.yaml")
    cfg = OmegaConf.merge(dataset_cfg, cfg)
    print(OmegaConf.to_yaml(cfg))
    shared_utils.set_seed(cfg.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map=device
    )
    df_long = shared_utils.load_dataset(cfg)

    horizon_metrics, runtime_stats = run_evaluation(
        cfg=cfg,
        pipeline=pipeline,
        df_long=df_long,
        assignments_df=None,
        return_runtime=True,
    )
    _print_final_metrics(cfg, horizon_metrics)
    _print_runtime_metrics(runtime_stats)


if __name__ == "__main__":
    main()
