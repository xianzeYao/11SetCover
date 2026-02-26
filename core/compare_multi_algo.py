from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.utils import ensure_dir, timestamp_id

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


DEFAULT_CLASS_AXIS_MAP = {
    "set_scale_small": "set_count",
    "set_scale_large": "set_count",
    "item_scale_small": "item_count",
    "item_scale_large": "item_count",
    "small_scale": "set_count",
    "large_scale": "set_count",
    "low_density": "density",
    "high_density": "density",
    "special_clustered": "sample_id",
    "special_hub": "sample_id",
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def _require_columns(df: pd.DataFrame, required: Iterable[str], run_dir: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{run_dir} 缺少必要列: {missing}")


def _load_single_run(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_path = Path(run_dir)
    csv_path = run_path / "results" / "runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 runs.csv: {csv_path}")

    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        required=[
            "algorithm_id",
            "algorithm_module",
            "dataset_id",
            "class_id",
            "sample_id",
            "instance_id",
            "set_count",
            "item_count",
            "density",
            "seed",
            "repeat_idx",
            "param_signature",
            "runtime_sec",
            "objective",
            "feasible",
        ],
        run_dir=run_path,
    )

    algorithm_values = sorted(df["algorithm_id"].dropna().astype(str).unique().tolist())
    if len(algorithm_values) != 1:
        raise ValueError(f"{run_path} 必须只包含一个 algorithm_id，实际={algorithm_values}")

    signature_values = sorted(df["param_signature"].dropna().astype(str).unique().tolist())
    if len(signature_values) != 1:
        raise ValueError(
            f"{run_path} 包含多个 param_signature={signature_values}，请使用固定参数单组实验后再 compare"
        )

    algorithm_id = algorithm_values[0]
    algorithm_module = str(df["algorithm_module"].dropna().astype(str).iloc[0])
    dataset_id = str(df["dataset_id"].dropna().astype(str).iloc[0])
    param_signature = signature_values[0]

    df = df.copy()
    df["algorithm_id"] = algorithm_id
    df["algorithm_module"] = algorithm_module
    df["dataset_id"] = dataset_id
    df["param_signature"] = param_signature
    df["run_dir"] = str(run_path)
    df["feasible"] = _as_bool(df["feasible"])

    meta = {
        "algorithm_id": algorithm_id,
        "algorithm_module": algorithm_module,
        "dataset_id": dataset_id,
        "param_signature": param_signature,
        "run_dir": str(run_path),
        "run_id": str(df["run_id"].iloc[0]) if "run_id" in df.columns and not df.empty else "",
    }
    return df, meta


def _validate_aligned_instances(dfs: list[pd.DataFrame]) -> None:
    if not dfs:
        raise ValueError("run_dirs 为空，无法 compare")

    key_cols = [
        "dataset_id",
        "instance_id",
        "class_id",
        "sample_id",
        "set_count",
        "item_count",
        "density",
        "pattern",
        "seed",
        "repeat_idx",
    ]
    for idx, df in enumerate(dfs):
        _require_columns(df, required=key_cols, run_dir=Path(f"run_dirs[{idx}]"))

    baseline = dfs[0][key_cols].sort_values(key_cols).reset_index(drop=True)
    for idx, df in enumerate(dfs[1:], start=1):
        candidate = df[key_cols].sort_values(key_cols).reset_index(drop=True)
        if len(candidate) != len(baseline):
            raise ValueError(f"run_dirs[{idx}] 样本数不一致，无法做严格配对比较")
        if not candidate.equals(baseline):
            raise ValueError(f"run_dirs[{idx}] 与基准数据键不一致，无法做严格配对比较")


def _recompute_gap_to_best(runs_df: pd.DataFrame) -> pd.DataFrame:
    df = runs_df.copy()
    for col in ["best_objective", "gap_to_best_pct"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    best = (
        df[df["feasible"] == True]
        .groupby(["instance_id", "seed", "repeat_idx"], as_index=False)["objective"]
        .min()
        .rename(columns={"objective": "best_objective"})
    )
    df = df.merge(best, on=["instance_id", "seed", "repeat_idx"], how="left")
    df["gap_to_best_pct"] = np.where(
        df["best_objective"].isna(),
        np.nan,
        np.where(
            df["best_objective"] == 0.0,
            0.0,
            (df["objective"] - df["best_objective"]) / df["best_objective"] * 100.0,
        ),
    )
    df.loc[df["feasible"] == False, "gap_to_best_pct"] = np.nan
    return df.drop(columns=["best_objective"])


def _aggregate_rows(group: pd.DataFrame) -> dict[str, Any]:
    feasible_group = group[group["feasible"] == True]
    return {
        "run_count": int(len(group)),
        "feasible_count": int(feasible_group.shape[0]),
        "feasible_rate": float(group["feasible"].mean()),
        "runtime_sec_mean": float(group["runtime_sec"].mean()),
        "runtime_sec_std": float(group["runtime_sec"].std()),
        "objective_mean": float(feasible_group["objective"].mean()) if not feasible_group.empty else np.nan,
        "objective_std": float(feasible_group["objective"].std()) if not feasible_group.empty else np.nan,
        "gap_to_best_pct_mean": float(feasible_group["gap_to_best_pct"].mean()) if not feasible_group.empty else np.nan,
        "gap_to_best_pct_std": float(feasible_group["gap_to_best_pct"].std()) if not feasible_group.empty else np.nan,
    }


def _summary_by_algo_class(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (algorithm_id, class_id), group in runs_df.groupby(["algorithm_id", "class_id"], sort=True):
        row = {
            "algorithm_id": algorithm_id,
            "class_id": class_id,
        }
        row.update(_aggregate_rows(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_by_algo_class_axis(runs_df: pd.DataFrame, class_axis_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_id, class_group in runs_df.groupby("class_id", sort=True):
        axis_name = class_axis_map.get(class_id)
        if not axis_name:
            raise ValueError(f"class_id={class_id} 未在 class_axis_map 中定义横轴字段")
        if axis_name not in class_group.columns:
            raise ValueError(f"class_id={class_id} 的横轴字段不存在: {axis_name}")

        work = class_group.copy()
        work["axis_name"] = axis_name
        work["axis_value"] = work[axis_name]
        for (algorithm_id, axis_value), group in work.groupby(["algorithm_id", "axis_value"], sort=True):
            row = {
                "algorithm_id": algorithm_id,
                "class_id": class_id,
                "axis_name": axis_name,
                "axis_value": axis_value,
            }
            row.update(_aggregate_rows(group))
            rows.append(row)

    return pd.DataFrame(rows)


def _paired_significance_by_class(
    runs_df: pd.DataFrame,
    metrics: list[str],
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_method = method.lower()

    for class_id, class_group in runs_df.groupby("class_id", sort=True):
        algs = sorted(class_group["algorithm_id"].dropna().astype(str).unique().tolist())
        for algorithm_a, algorithm_b in combinations(algs, 2):
            pair_group = class_group[class_group["algorithm_id"].isin([algorithm_a, algorithm_b])]
            for metric in metrics:
                if metric not in pair_group.columns:
                    continue

                filtered = pair_group
                if metric in {"objective", "gap_to_best_pct"}:
                    filtered = filtered[filtered["feasible"] == True]
                    filtered = filtered[np.isfinite(filtered[metric])]

                pivot = (
                    filtered.pivot_table(
                        index=["instance_id", "seed", "repeat_idx"],
                        columns="algorithm_id",
                        values=metric,
                        aggfunc="mean",
                    )
                    .dropna()
                    .reset_index(drop=True)
                )

                if len(pivot) < 2:
                    rows.append(
                        {
                            "class_id": class_id,
                            "algorithm_a": algorithm_a,
                            "algorithm_b": algorithm_b,
                            "metric": metric,
                            "test": "insufficient_pairs",
                            "n_pairs": int(len(pivot)),
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "mean_diff_a_minus_b": np.nan,
                        }
                    )
                    continue

                x = pivot[algorithm_a].to_numpy(dtype=float)
                y = pivot[algorithm_b].to_numpy(dtype=float)
                if np.allclose(x, y):
                    stat, p_value, test_name = 0.0, 1.0, "identical_samples"
                elif stats is None:
                    stat, p_value, test_name = np.nan, np.nan, "scipy_missing"
                elif test_method == "ttest":
                    stat, p_value = stats.ttest_rel(x, y)
                    test_name = "ttest_rel"
                else:
                    try:
                        stat, p_value = stats.wilcoxon(x, y)
                        test_name = "wilcoxon"
                    except ValueError:
                        stat, p_value = np.nan, np.nan
                        test_name = "wilcoxon_failed"

                rows.append(
                    {
                        "class_id": class_id,
                        "algorithm_a": algorithm_a,
                        "algorithm_b": algorithm_b,
                        "metric": metric,
                        "test": test_name,
                        "n_pairs": int(len(pivot)),
                        "statistic": float(stat) if stat == stat else np.nan,
                        "p_value": float(p_value) if p_value == p_value else np.nan,
                        "mean_diff_a_minus_b": float(np.mean(x) - np.mean(y)),
                    }
                )

    return pd.DataFrame(rows)


def compare_experiments(
    run_dirs: list[str | Path],
    output_dir: str | Path | None = None,
    class_axis_map: dict[str, str] | None = None,
    with_plots: bool = True,
    significance_method: str = "wilcoxon",
    significance_metrics: list[str] | None = None,
) -> Path:
    if not run_dirs:
        raise ValueError("run_dirs 不能为空")

    loaded: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    seen_algorithm_ids: set[str] = set()
    for run_dir in run_dirs:
        df, meta = _load_single_run(run_dir)
        algorithm_id = str(meta["algorithm_id"])
        if algorithm_id in seen_algorithm_ids:
            raise ValueError(f"重复的 algorithm_id: {algorithm_id}。compare 期望每个算法只给一个 run_dir")
        seen_algorithm_ids.add(algorithm_id)
        loaded.append(df)
        manifest_rows.append(meta)

    _validate_aligned_instances(loaded)

    merged = pd.concat(loaded, ignore_index=True)
    merged = _recompute_gap_to_best(merged)

    axis_map = dict(DEFAULT_CLASS_AXIS_MAP)
    if class_axis_map:
        axis_map.update(class_axis_map)

    summary_class_df = _summary_by_algo_class(merged)
    summary_axis_df = _summary_by_algo_class_axis(merged, axis_map)

    metrics = significance_metrics or ["objective"]
    significance_df = _paired_significance_by_class(
        runs_df=merged,
        metrics=metrics,
        method=significance_method,
    )

    if output_dir is None:
        output_root = Path("outputs/experiments")
        out = ensure_dir(output_root / timestamp_id("cmp"))
    else:
        out = ensure_dir(output_dir)
    results_dir = ensure_dir(out / "results")
    figures_dir = ensure_dir(out / "figures")

    merged.to_csv(results_dir / "runs_merged.csv", index=False)
    summary_class_df.to_csv(results_dir / "summary_algo_class.csv", index=False)
    summary_axis_df.to_csv(results_dir / "summary_algo_class_axis.csv", index=False)
    significance_df.to_csv(results_dir / "significance_algo.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(results_dir / "manifest.csv", index=False)

    if with_plots:
        from core.visualize_compare import generate_compare_plots

        generate_compare_plots(
            summary_class_df=summary_class_df,
            summary_axis_df=summary_axis_df,
            class_axis_map=axis_map,
            out_dir=figures_dir,
        )

    return out
