from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RESULT = {
    "objective": float("inf"),
    "runtime_sec": 0.0,
    "is_feasible": False,
    "selected_sets": [],
    "convergence_curve": [],
    "meta": {},
}


def normalize_result(raw_result: Any, elapsed: float | None = None) -> dict[str, Any]:
    elapsed_sec = elapsed if elapsed is not None else 0.0
    result = dict(DEFAULT_RESULT)

    if isinstance(raw_result, dict):
        result.update(raw_result)
    else:
        for key in result:
            if hasattr(raw_result, key):
                result[key] = getattr(raw_result, key)

    result["objective"] = float(result.get("objective", float("inf")))
    result["runtime_sec"] = float(result.get("runtime_sec", elapsed_sec))
    result["is_feasible"] = bool(result.get("is_feasible", False))
    result["selected_sets"] = list(result.get("selected_sets", []) or [])

    curve = result.get("convergence_curve", None)
    if curve is None and isinstance(result.get("meta"), dict):
        curve = result["meta"].get("convergence_curve")
    result["convergence_curve"] = [float(x) for x in (curve or [])]

    meta = result.get("meta", {})
    result["meta"] = meta if isinstance(meta, dict) else {"raw_meta": meta}
    return result


def convergence_speed(curve: list[float]) -> float:
    if len(curve) < 2:
        return math.nan

    start = float(curve[0])
    final = float(min(curve))
    if start <= final:
        return 0.0

    target = final + (start - final) * 0.05
    for idx, value in enumerate(curve):
        if value <= target:
            return float(idx)
    return float(len(curve) - 1)


def add_gap_to_best(runs_df: pd.DataFrame) -> pd.DataFrame:
    df = runs_df.copy()
    feasible = df[df["feasible"] == True]
    best = feasible.groupby("instance_id", as_index=False)["objective"].min()
    best = best.rename(columns={"objective": "best_objective"})
    df = df.merge(best, on="instance_id", how="left")

    df["gap_to_best_pct"] = np.where(
        df["best_objective"].isna(),
        np.nan,
        np.where(
            df["best_objective"] == 0,
            0.0,
            (df["objective"] - df["best_objective"]) / df["best_objective"] * 100.0,
        ),
    )
    return df


def add_stability_var(runs_df: pd.DataFrame) -> pd.DataFrame:
    df = runs_df.copy()
    grouped = (
        df.groupby(["param_signature", "class_id", "sample_id"], as_index=False)["objective"]
        .var(ddof=0)
        .rename(columns={"objective": "stability_var"})
    )
    return df.merge(grouped, on=["param_signature", "class_id", "sample_id"], how="left")


def summarize_by_param(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()

    agg_map: dict[str, tuple[str, str]] = {
        "run_count": ("objective", "count"),
        "feasible_rate": ("feasible", "mean"),
        "runtime_sec_mean": ("runtime_sec", "mean"),
        "runtime_sec_std": ("runtime_sec", "std"),
        "objective_mean": ("objective", "mean"),
        "objective_std": ("objective", "std"),
        "gap_to_best_pct_mean": ("gap_to_best_pct", "mean"),
        "gap_to_best_pct_std": ("gap_to_best_pct", "std"),
        "stability_var_mean": ("stability_var", "mean"),
        "convergence_speed_mean": ("convergence_speed", "mean"),
    }
    optional_metrics = [
        ("pareto_size", "pareto_size_mean", "mean"),
        ("pareto_size", "pareto_size_std", "std"),
        ("front_cost_span", "front_cost_span_mean", "mean"),
        ("front_cost_span", "front_cost_span_std", "std"),
        ("front_hv", "front_hv_mean", "mean"),
        ("front_hv", "front_hv_std", "std"),
        ("front_hv_norm", "front_hv_norm_mean", "mean"),
        ("front_hv_norm", "front_hv_norm_std", "std"),
        ("front_feasible_ratio", "front_feasible_ratio_mean", "mean"),
    ]
    for src, dst, fn in optional_metrics:
        if src in runs_df.columns:
            agg_map[dst] = (src, fn)

    agg = (
        runs_df.groupby(["algorithm_id", "param_signature", "param_key", "param_value"], as_index=False)
        .agg(**agg_map)
        .sort_values(["param_signature"])
        .reset_index(drop=True)
    )
    return agg


def summarize_by_class(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()

    agg_map: dict[str, tuple[str, str]] = {
        "run_count": ("objective", "count"),
        "feasible_rate": ("feasible", "mean"),
        "runtime_sec_mean": ("runtime_sec", "mean"),
        "runtime_sec_std": ("runtime_sec", "std"),
        "objective_mean": ("objective", "mean"),
        "objective_std": ("objective", "std"),
        "gap_to_best_pct_mean": ("gap_to_best_pct", "mean"),
        "gap_to_best_pct_std": ("gap_to_best_pct", "std"),
    }
    optional_metrics = [
        ("pareto_size", "pareto_size_mean", "mean"),
        ("pareto_size", "pareto_size_std", "std"),
        ("front_cost_span", "front_cost_span_mean", "mean"),
        ("front_cost_span", "front_cost_span_std", "std"),
        ("front_hv", "front_hv_mean", "mean"),
        ("front_hv", "front_hv_std", "std"),
        ("front_hv_norm", "front_hv_norm_mean", "mean"),
        ("front_hv_norm", "front_hv_norm_std", "std"),
        ("front_feasible_ratio", "front_feasible_ratio_mean", "mean"),
    ]
    for src, dst, fn in optional_metrics:
        if src in runs_df.columns:
            agg_map[dst] = (src, fn)

    agg = (
        runs_df.groupby(["algorithm_id", "class_id", "param_signature"], as_index=False)
        .agg(**agg_map)
        .sort_values(["class_id", "param_signature"])
        .reset_index(drop=True)
    )
    return agg
