from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _maybe_save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_ofat_lines(summary_df: pd.DataFrame, sweep_param: str, out_dir: Path) -> list[str]:
    if summary_df.empty:
        return []

    df = summary_df.copy()
    df["x_value"] = _to_numeric(df["param_value"])
    if df["x_value"].isna().all():
        df["x_value"] = np.arange(len(df), dtype=float)

    df = df.sort_values("x_value")

    figures: list[str] = []
    for metric_mean, metric_std, title, ylabel, filename in [
        ("runtime_sec_mean", "runtime_sec_std", "OFAT: Runtime Trend", "runtime_sec", "line_runtime.png"),
        ("objective_mean", "objective_std", "OFAT: Objective Trend", "objective", "line_objective.png"),
        ("gap_to_best_pct_mean", "gap_to_best_pct_std", "OFAT: Gap-to-Best Trend", "gap_to_best_pct", "line_gap.png"),
        ("pareto_size_mean", "pareto_size_std", "OFAT: Pareto Size Trend", "pareto_size", "line_pareto_size.png"),
        ("front_hv_norm_mean", "front_hv_norm_std", "OFAT: Normalized Hypervolume Trend", "front_hv_norm", "line_front_hv_norm.png"),
        ("front_cost_span_mean", "front_cost_span_std", "OFAT: Front Cost Span Trend", "front_cost_span", "line_front_cost_span.png"),
    ]:
        if metric_mean not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        x = df["x_value"].to_numpy(dtype=float)
        y = df[metric_mean].to_numpy(dtype=float)
        yerr = df[metric_std].fillna(0.0).to_numpy(dtype=float) if metric_std in df.columns else np.zeros_like(y)

        ax.plot(x, y, marker="o", label=metric_mean)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

        figures.append(_maybe_save(fig, out_dir / filename))

    return figures


def _plot_class_bar(summary_class_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if summary_class_df.empty:
        return []

    figures: list[str] = []
    for metric, title, ylabel, filename in [
        ("objective_mean", "Class vs Parameter Setting: Objective Mean", "objective_mean", "bar_class_objective.png"),
        ("pareto_size_mean", "Class vs Parameter Setting: Pareto Size Mean", "pareto_size_mean", "bar_class_pareto_size.png"),
        ("front_hv_norm_mean", "Class vs Parameter Setting: Front HV(norm) Mean", "front_hv_norm_mean", "bar_class_front_hv_norm.png"),
    ]:
        if metric not in summary_class_df.columns:
            continue

        pivot = summary_class_df.pivot_table(
            index="class_id",
            columns="param_signature",
            values=metric,
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("class_id")
        ax.set_ylabel(ylabel)
        ax.legend(title="param_signature", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        figures.append(_maybe_save(fig, out_dir / filename))

    return figures


def _plot_grid_heatmaps(runs_df: pd.DataFrame, grid_params: list[str], out_dir: Path) -> list[str]:
    if len(grid_params) < 2 or runs_df.empty:
        return []

    p1 = f"param__{grid_params[0]}"
    p2 = f"param__{grid_params[1]}"
    if p1 not in runs_df.columns or p2 not in runs_df.columns:
        return []

    figures: list[str] = []
    for metric, title, filename in [
        ("runtime_sec", "Grid Heatmap: runtime_sec", "heatmap_runtime.png"),
        ("objective", "Grid Heatmap: objective", "heatmap_objective.png"),
        ("gap_to_best_pct", "Grid Heatmap: gap_to_best_pct", "heatmap_gap.png"),
    ]:
        if metric not in runs_df.columns:
            continue

        heat = (
            runs_df.groupby([p1, p2], as_index=False)[metric]
            .mean()
            .pivot(index=p1, columns=p2, values=metric)
            .sort_index()
            .sort_index(axis=1)
        )
        if heat.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(heat.values, aspect="auto", origin="lower")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels([str(x) for x in heat.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels([str(y) for y in heat.index])
        ax.set_xlabel(grid_params[1])
        ax.set_ylabel(grid_params[0])
        ax.set_title(title)
        fig.colorbar(cax, ax=ax)

        figures.append(_maybe_save(fig, out_dir / filename))

    return figures


def _plot_complexity(runs_df: pd.DataFrame, fit_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if runs_df.empty:
        return []

    fig, ax = plt.subplots(figsize=(8, 5))
    for signature, group in runs_df.groupby("param_signature"):
        ax.scatter(group["set_count"], group["runtime_sec"], s=12, alpha=0.6, label=signature)

        row = fit_df[fit_df["param_signature"] == signature]
        if row.empty:
            continue

        slope = row.iloc[0]["slope"]
        intercept = row.iloc[0]["intercept"]
        if pd.notna(slope) and pd.notna(intercept):
            x_line = np.linspace(group["set_count"].min(), group["set_count"].max(), 50)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linewidth=1.5)

    ax.set_title("Theory Consistency Check: set_count vs runtime_sec")
    ax.set_xlabel("set_count")
    ax.set_ylabel("runtime_sec")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    return [_maybe_save(fig, out_dir / "complexity_trend.png")]


def _parse_front_points(cell: Any) -> list[tuple[float, float]]:
    if isinstance(cell, list):
        raw = cell
    elif isinstance(cell, str) and cell.strip():
        try:
            raw = json.loads(cell)
        except json.JSONDecodeError:
            return []
    else:
        return []

    points: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return points


def _plot_pareto_front_by_class(runs_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if runs_df.empty or "front_points_json" not in runs_df.columns:
        return []

    work = runs_df.copy()
    if "feasible" in work.columns and work["feasible"].dtype == object:
        work["feasible"] = work["feasible"].astype(str).str.lower().isin({"1", "true", "yes"})
    work["front_points"] = work["front_points_json"].apply(_parse_front_points)
    work = work[work["front_points"].map(len) > 0]
    if work.empty:
        return []

    reps = []
    for class_id, group in work.groupby("class_id", sort=True):
        feasible_group = group[group["feasible"] == True]
        candidate = feasible_group if not feasible_group.empty else group
        row = candidate.sort_values(["objective", "runtime_sec"]).iloc[0]
        reps.append((class_id, row))

    if not reps:
        return []

    n = len(reps)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (class_id, row) in enumerate(reps):
        ax = axes_flat[idx]
        points = row["front_points"]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.scatter(x, y, s=20, alpha=0.8)
        ax.set_title(f"class={class_id}")
        ax.set_xlabel("cost (f1)")
        ax.set_ylabel("selected_set_count (f2)")
        ax.grid(alpha=0.25)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    return [_maybe_save(fig, out_dir / "pareto_front_by_class.png")]


def generate_plots(
    runs_df: pd.DataFrame,
    summary_param_df: pd.DataFrame,
    summary_class_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    out_dir: str | Path,
    mode: str,
    sweep_param: str,
    grid_params: list[str],
) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    if mode == "ofat" and sweep_param:
        generated.extend(_plot_ofat_lines(summary_param_df, sweep_param, out))
    generated.extend(_plot_class_bar(summary_class_df, out))
    if mode == "grid":
        generated.extend(_plot_grid_heatmaps(runs_df, grid_params, out))
    generated.extend(_plot_complexity(runs_df, fit_df, out))
    generated.extend(_plot_pareto_front_by_class(runs_df, out))
    return generated


def plot_from_experiment_dir(experiment_dir: str | Path) -> list[str]:
    exp = Path(experiment_dir)
    runs_path = exp / "results" / "runs.csv"
    summary_param_path = exp / "results" / "summary_by_param.csv"
    summary_class_path = exp / "results" / "summary_by_class.csv"
    fit_path = exp / "results" / "complexity_fit.csv"
    meta_path = exp / "results" / "run_meta.csv"

    if not runs_path.exists():
        raise FileNotFoundError(f"未找到 runs.csv: {runs_path}")

    runs_df = pd.read_csv(runs_path)
    summary_param_df = pd.read_csv(summary_param_path) if summary_param_path.exists() else pd.DataFrame()
    summary_class_df = pd.read_csv(summary_class_path) if summary_class_path.exists() else pd.DataFrame()
    fit_df = pd.read_csv(fit_path) if fit_path.exists() else pd.DataFrame()

    mode = "ofat"
    sweep_param = "param"
    grid_params: list[str] = []
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        meta = {str(r["key"]): str(r["value"]) for _, r in meta_df.iterrows()}
        mode = meta.get("mode", "ofat")
        sweep_param = meta.get("sweep_param", "param")
        grid_params = [x for x in meta.get("grid_params", "").split(",") if x]

    return generate_plots(
        runs_df=runs_df,
        summary_param_df=summary_param_df,
        summary_class_df=summary_class_df,
        fit_df=fit_df,
        out_dir=exp / "figures",
        mode=mode,
        sweep_param=sweep_param,
        grid_params=grid_params,
    )
