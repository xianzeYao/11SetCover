from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPECIAL_CLASSES = ("special_clustered", "special_hub")


def _maybe_save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _safe_name(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text))
    safe = safe.strip("_")
    return safe or "unknown"


def _sort_axis_index(index_values: pd.Index) -> pd.Index:
    raw = pd.Series(index_values.astype(str), dtype=str)
    nums = pd.to_numeric(raw, errors="coerce")
    if nums.notna().all():
        order = np.argsort(nums.to_numpy(dtype=float))
        return pd.Index(raw.iloc[order].tolist())
    return pd.Index(sorted(raw.tolist()))


def _plot_bar_by_class(summary_class_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if summary_class_df.empty:
        return []

    figures: list[str] = []
    metric_specs = [
        ("runtime_sec_mean", "Runtime by Class", "runtime_sec", "compare_bar_runtime_by_class.png"),
        ("objective_mean", "Objective by Class (Feasible Only)", "objective", "compare_bar_objective_by_class.png"),
        ("gap_to_best_pct_mean", "Gap-to-Best by Class (Feasible Only)", "gap_to_best_pct", "compare_bar_gap_by_class.png"),
    ]

    for metric_col, title, ylabel, filename in metric_specs:
        if metric_col not in summary_class_df.columns:
            continue

        pivot = summary_class_df.pivot_table(
            index="class_id",
            columns="algorithm_id",
            values=metric_col,
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("class_id")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="algorithm_id", fontsize=8)
        figures.append(_maybe_save(fig, out_dir / filename))

    return figures


def _plot_lines_by_class(summary_axis_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if summary_axis_df.empty:
        return []

    figures: list[str] = []
    metric_specs = [
        ("runtime_sec_mean", "runtime_sec_std", "runtime", "runtime_sec"),
        ("objective_mean", "objective_std", "objective", "objective"),
        ("gap_to_best_pct_mean", "gap_to_best_pct_std", "gap", "gap_to_best_pct"),
    ]

    for class_id, class_df in summary_axis_df.groupby("class_id", sort=True):
        if class_id in SPECIAL_CLASSES:
            continue
        if "axis_name" not in class_df.columns or "axis_value" not in class_df.columns:
            continue
        axis_name = str(class_df["axis_name"].iloc[0])

        for metric_mean, metric_std, suffix, ylabel in metric_specs:
            if metric_mean not in class_df.columns:
                continue

            pivot_mean = class_df.pivot_table(
                index="axis_value",
                columns="algorithm_id",
                values=metric_mean,
                aggfunc="mean",
            )
            if pivot_mean.empty:
                continue

            pivot_std = (
                class_df.pivot_table(
                    index="axis_value",
                    columns="algorithm_id",
                    values=metric_std,
                    aggfunc="mean",
                )
                if metric_std in class_df.columns
                else pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns)
            )

            ordered_index = _sort_axis_index(pivot_mean.index)
            pivot_mean = pivot_mean.reindex(ordered_index)
            pivot_std = pivot_std.reindex(index=ordered_index, columns=pivot_mean.columns).fillna(0.0)

            x = np.arange(len(pivot_mean.index))
            fig, ax = plt.subplots(figsize=(10, 4))
            for algorithm_id in pivot_mean.columns:
                y = pivot_mean[algorithm_id].to_numpy(dtype=float)
                yerr = pivot_std[algorithm_id].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", label=str(algorithm_id))
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)

            ax.set_title(f"Class={class_id}: {suffix} by algorithm")
            ax.set_xlabel(axis_name)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in pivot_mean.index], rotation=45, ha="right")
            ax.grid(alpha=0.25)
            ax.legend(title="algorithm_id", fontsize=8)

            filename = f"compare_line_{_safe_name(class_id)}_{suffix}.png"
            figures.append(_maybe_save(fig, out_dir / filename))

    return figures


def _plot_lines_special_combined(summary_axis_df: pd.DataFrame, out_dir: Path) -> list[str]:
    if summary_axis_df.empty:
        return []

    work = summary_axis_df[summary_axis_df["class_id"].isin(SPECIAL_CLASSES)].copy()
    if work.empty:
        return []

    figures: list[str] = []
    metric_specs = [
        ("runtime_sec_mean", "runtime_sec_std", "runtime", "runtime_sec"),
        ("objective_mean", "objective_std", "objective", "objective"),
        ("gap_to_best_pct_mean", "gap_to_best_pct_std", "gap", "gap_to_best_pct"),
    ]

    for metric_mean, metric_std, suffix, ylabel in metric_specs:
        if metric_mean not in work.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
        plotted = 0
        for i, class_id in enumerate(SPECIAL_CLASSES):
            class_df = work[work["class_id"] == class_id]
            ax = axes[0, i]
            if class_df.empty:
                ax.axis("off")
                continue

            axis_name = str(class_df["axis_name"].iloc[0])
            pivot_mean = class_df.pivot_table(
                index="axis_value",
                columns="algorithm_id",
                values=metric_mean,
                aggfunc="mean",
            )
            if pivot_mean.empty:
                ax.axis("off")
                continue

            pivot_std = (
                class_df.pivot_table(
                    index="axis_value",
                    columns="algorithm_id",
                    values=metric_std,
                    aggfunc="mean",
                )
                if metric_std in class_df.columns
                else pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns)
            )

            ordered_index = _sort_axis_index(pivot_mean.index)
            pivot_mean = pivot_mean.reindex(ordered_index)
            pivot_std = pivot_std.reindex(index=ordered_index, columns=pivot_mean.columns).fillna(0.0)

            x = np.arange(len(pivot_mean.index))
            for algorithm_id in pivot_mean.columns:
                y = pivot_mean[algorithm_id].to_numpy(dtype=float)
                yerr = pivot_std[algorithm_id].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", label=str(algorithm_id))
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)

            ax.set_title(f"class={class_id}")
            ax.set_xlabel(axis_name)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in pivot_mean.index], rotation=45, ha="right")
            ax.grid(alpha=0.25)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="algorithm_id", loc="upper center", ncol=min(4, len(labels)))
        fig.suptitle(f"Special Classes Combined: {suffix}")
        figures.append(_maybe_save(fig, out_dir / f"compare_line_special_combined_{suffix}.png"))

    return figures


def generate_compare_plots(
    summary_class_df: pd.DataFrame,
    summary_axis_df: pd.DataFrame,
    class_axis_map: dict[str, str],
    out_dir: str | Path,
) -> list[str]:
    _ = class_axis_map  # reserved for future validation/debug output
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    generated.extend(_plot_bar_by_class(summary_class_df, out))
    generated.extend(_plot_lines_by_class(summary_axis_df, out))
    generated.extend(_plot_lines_special_combined(summary_axis_df, out))
    return generated
