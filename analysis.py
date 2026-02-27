from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core.io_dataset import read_instance
try:
    import seaborn as sns
except Exception as exc:  # pragma: no cover
    sns = None
    _SEABORN_IMPORT_ERROR = exc
else:
    _SEABORN_IMPORT_ERROR = None

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


DEFAULT_CLASS_AXIS_MAP: dict[str, str] = {
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

DEFAULT_CLASS_COLOR_MAP: dict[str, str] = {
    "set_scale_small": "#1f77b4",
    "set_scale_large": "#ff7f0e",
    "item_scale_small": "#2ca02c",
    "item_scale_large": "#d62728",
    "low_density": "#9467bd",
    "high_density": "#8c564b",
    "special_clustered": "#e377c2",
    "special_hub": "#17becf",
    # Backward compatibility with old class ids.
    "small_scale": "#1f77b4",
    "large_scale": "#ff7f0e",
}

REQUIRED_COLUMNS = [
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
    "runtime_sec",
    "objective",
    "feasible",
]

ROBUST_DROP_PCTS = [1, 3, 5, 7]
ROBUST_TRIALS = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set Cover runs.csv analysis and plotting")
    parser.add_argument("--runs-root", required=True,
                        help="Path to runs root folder")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory, default: <runs-root>/analysis")
    parser.add_argument(
        "--ilp-id",
        default="auto",
        help="ILP algorithm_id used as gold standard; default auto-detects from ilp_ortools/ilp_pulp",
    )
    parser.add_argument(
        "--pareto-algo-id",
        default="moea_nsga2",
        help="algorithm_id used for per-sample Pareto plots; set empty string to include all algorithms with front_points_json",
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance alpha threshold")
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Optional comma-separated algorithm_id filter, e.g. ga,greedy_001,ilp_pulp; when set, must include at least one ILP algorithm",
    )
    parser.add_argument(
        "--test",
        choices=["wilcoxon", "ttest"],
        default="wilcoxon",
        help="Paired significance test method",
    )
    return parser.parse_args()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _parse_algorithm_filter(raw: Any) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    tokens = [x.strip() for x in text.split(",")]
    dedup: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        dedup.append(token)
    return dedup


def _parse_solver_status(meta_json: Any) -> str:
    if not isinstance(meta_json, str) or not meta_json:
        return ""
    try:
        payload = json.loads(meta_json)
    except Exception:
        return ""
    status = payload.get("solver_status", "")
    return str(status).strip().lower()


def _discover_csv_candidates(algo_dir: Path) -> list[Path]:
    candidates: dict[Path, None] = {}

    direct = algo_dir / "runs.csv"
    if direct.exists():
        candidates[direct.resolve()] = None

    for old_path in algo_dir.glob("*/results/runs.csv"):
        if old_path.exists():
            candidates[old_path.resolve()] = None

    return sorted(candidates.keys(), key=lambda p: p.stat().st_mtime)


def _load_runs_from_root(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not runs_root.exists() or not runs_root.is_dir():
        raise FileNotFoundError(f"runs_root 不存在或不是目录: {runs_root}")

    all_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    seen_algorithm_ids: set[str] = set()

    for algo_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        candidates = _discover_csv_candidates(algo_dir)
        if not candidates:
            continue
        selected = candidates[-1]  # latest mtime

        df = pd.read_csv(selected)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{selected} 缺少必要列: {missing}")

        algorithm_values = sorted(
            df["algorithm_id"].dropna().astype(str).unique().tolist())
        if len(algorithm_values) != 1:
            raise ValueError(
                f"{selected} 必须只包含一个 algorithm_id，实际={algorithm_values}")
        algorithm_id = algorithm_values[0]
        if algorithm_id in seen_algorithm_ids:
            raise ValueError(
                f"重复 algorithm_id: {algorithm_id}，请保证每个算法只对应一个 runs.csv")
        seen_algorithm_ids.add(algorithm_id)

        work = df.copy()
        work["source_runs_csv"] = str(selected)
        work["feasible"] = _as_bool(work["feasible"])
        work["runtime_sec"] = pd.to_numeric(
            work["runtime_sec"], errors="coerce")
        work["objective"] = pd.to_numeric(work["objective"], errors="coerce")
        work["set_count"] = pd.to_numeric(work["set_count"], errors="coerce")
        work["item_count"] = pd.to_numeric(work["item_count"], errors="coerce")
        work["density"] = pd.to_numeric(work["density"], errors="coerce")
        work["seed"] = pd.to_numeric(
            work["seed"], errors="coerce").astype("Int64")
        work["repeat_idx"] = pd.to_numeric(
            work["repeat_idx"], errors="coerce").astype("Int64")
        work["solver_status_norm"] = work.get(
            "meta_json", "").map(_parse_solver_status)

        all_frames.append(work)
        manifest_rows.append(
            {
                "algorithm_dir": str(algo_dir),
                "algorithm_id": algorithm_id,
                "selected_runs_csv": str(selected),
                "candidate_count": len(candidates),
                "auto_selected_latest": bool(len(candidates) > 1),
                "candidate_paths": " | ".join(str(x) for x in candidates),
            }
        )

    if not all_frames:
        raise ValueError(f"在 {runs_root} 下未发现可用 runs.csv")

    merged = pd.concat(all_frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(
        "algorithm_id").reset_index(drop=True)
    return merged, manifest


def _add_ilp_baseline_gap(runs_df: pd.DataFrame, ilp_id: str) -> pd.DataFrame:
    df = runs_df.copy()
    baseline_key = ["instance_id", "seed", "repeat_idx"]

    ilp_rows = df[
        (df["algorithm_id"] == ilp_id)
        & (df["feasible"] == True)
        & np.isfinite(df["objective"])
        & (df["objective"] > 0)
        & (df["solver_status_norm"] == "optimal")
    ]

    if ilp_rows.empty:
        raise ValueError(f"未找到可用 ILP Optimal 基线数据，请检查 --ilp-id={ilp_id}")

    ilp_best = (
        ilp_rows.groupby(baseline_key, as_index=False)["objective"]
        .min()
        .rename(columns={"objective": "ilp_opt_objective"})
    )

    df = df.merge(ilp_best, on=baseline_key, how="left")
    df["has_ilp_baseline"] = df["ilp_opt_objective"].notna()

    valid_gap = (
        (df["feasible"] == True)
        & (df["has_ilp_baseline"] == True)
        & np.isfinite(df["objective"])
        & np.isfinite(df["ilp_opt_objective"])
        & (df["ilp_opt_objective"] > 0)
    )
    df["gap_to_ilp_opt_pct"] = np.nan
    df.loc[valid_gap, "gap_to_ilp_opt_pct"] = (
        (df.loc[valid_gap, "objective"] -
         df.loc[valid_gap, "ilp_opt_objective"])
        / df.loc[valid_gap, "ilp_opt_objective"]
        * 100.0
    )
    return df


def _normalize_ilp_id(raw: Any) -> str:
    text = str(raw or "").strip()
    if text.endswith("\\"):
        text = text.rstrip("\\").strip()
    lowered = text.lower()
    if lowered in {"", "auto", "none", "null"}:
        return ""
    return text


def _count_ilp_optimal_rows(runs_df: pd.DataFrame, ilp_id: str) -> int:
    mask = (
        (runs_df["algorithm_id"] == ilp_id)
        & (runs_df["feasible"] == True)
        & np.isfinite(runs_df["objective"])
        & (runs_df["objective"] > 0)
        & (runs_df["solver_status_norm"] == "optimal")
    )
    return int(mask.sum())


def _resolve_ilp_id(runs_df: pd.DataFrame, requested_ilp_id: Any) -> str:
    requested = _normalize_ilp_id(requested_ilp_id)
    algorithm_ids = sorted(
        runs_df["algorithm_id"].dropna().astype(str).unique().tolist())
    available = set(algorithm_ids)

    if requested:
        if requested in available:
            n_opt = _count_ilp_optimal_rows(runs_df, requested)
            if n_opt > 0:
                return requested
            print(
                f"[analysis] --ilp-id={requested} 存在但无 optimal 记录，尝试自动回退到可用 ILP 基线。"
            )
        else:
            print(
                f"[analysis] --ilp-id={requested} 不在当前 runs 中，尝试自动回退到可用 ILP 基线。"
            )

    preferred_ids = ["ilp_ortools", "ilp_pulp"]
    candidate_ids: list[str] = []
    for algorithm_id in preferred_ids:
        if algorithm_id in available and algorithm_id not in candidate_ids:
            candidate_ids.append(algorithm_id)
    for algorithm_id in algorithm_ids:
        if "ilp" in algorithm_id.lower() and algorithm_id not in candidate_ids:
            candidate_ids.append(algorithm_id)

    for algorithm_id in candidate_ids:
        n_opt = _count_ilp_optimal_rows(runs_df, algorithm_id)
        if n_opt <= 0:
            continue
        if algorithm_id != requested:
            print(
                f"[analysis] Auto-selected ILP baseline: {algorithm_id} (optimal rows={n_opt})")
        return algorithm_id

    if requested:
        raise ValueError(
            f"未找到可用 ILP Optimal 基线：--ilp-id={requested} 无法使用，且自动回退失败；"
            f"已检测算法={algorithm_ids}"
        )
    raise ValueError(
        f"未找到可用 ILP Optimal 基线：请检查 runs 中是否存在 ILP 算法且含 optimal 记录；"
        f"已检测算法={algorithm_ids}"
    )


def _aggregate_group(group: pd.DataFrame) -> dict[str, Any]:
    feasible = group[group["feasible"] == True]
    gap_valid = group[np.isfinite(group["gap_to_ilp_opt_pct"])]
    stability_valid = pd.Series(dtype=float)
    if "stability_var" in group.columns:
        stability_valid = pd.to_numeric(
            group["stability_var"], errors="coerce")
        stability_valid = stability_valid[np.isfinite(stability_valid)]
    return {
        "run_count": int(len(group)),
        "feasible_count": int(len(feasible)),
        "feasible_rate": float(group["feasible"].mean()),
        "runtime_sec_mean": float(group["runtime_sec"].mean()),
        "runtime_sec_median": float(group["runtime_sec"].median()),
        "runtime_sec_std": float(group["runtime_sec"].std()),
        "objective_mean": float(feasible["objective"].mean()) if not feasible.empty else np.nan,
        "objective_median": float(feasible["objective"].median()) if not feasible.empty else np.nan,
        "objective_std": float(feasible["objective"].std()) if not feasible.empty else np.nan,
        "gap_to_ilp_opt_pct_mean": float(gap_valid["gap_to_ilp_opt_pct"].mean()) if not gap_valid.empty else np.nan,
        "gap_to_ilp_opt_pct_median": float(gap_valid["gap_to_ilp_opt_pct"].median()) if not gap_valid.empty else np.nan,
        "gap_to_ilp_opt_pct_std": float(gap_valid["gap_to_ilp_opt_pct"].std()) if not gap_valid.empty else np.nan,
        "stability_var_mean": float(stability_valid.mean()) if not stability_valid.empty else np.nan,
        "stability_var_std": float(stability_valid.std()) if not stability_valid.empty else np.nan,
    }


def _summary_algo_class(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (algorithm_id, class_id), group in runs_df.groupby(["algorithm_id", "class_id"], sort=True):
        row = {
            "algorithm_id": algorithm_id,
            "class_id": class_id,
        }
        row.update(_aggregate_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _add_axis_columns(runs_df: pd.DataFrame, class_axis_map: dict[str, str]) -> pd.DataFrame:
    df = runs_df.copy()
    axis_names: list[str] = []
    axis_values: list[Any] = []

    for _, row in df.iterrows():
        class_id = str(row["class_id"])
        axis_name = class_axis_map.get(class_id)
        if axis_name is None:
            raise ValueError(f"class_id={class_id} 未配置横轴字段")
        if axis_name not in df.columns:
            raise ValueError(f"class_id={class_id} 的横轴字段不存在: {axis_name}")
        axis_names.append(axis_name)
        axis_values.append(row[axis_name])

    df["axis_name"] = axis_names
    df["axis_value"] = axis_values
    return df


def _summary_algo_class_axis(runs_with_axis_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (algorithm_id, class_id, axis_name, axis_value), group in runs_with_axis_df.groupby(
        ["algorithm_id", "class_id", "axis_name", "axis_value"], sort=True
    ):
        row = {
            "algorithm_id": algorithm_id,
            "class_id": class_id,
            "axis_name": axis_name,
            "axis_value": axis_value,
        }
        row.update(_aggregate_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_significance(
    runs_df: pd.DataFrame,
    metrics: list[str],
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_method = method.lower()
    pair_key = ["instance_id", "seed", "repeat_idx"]

    for class_id, class_group in runs_df.groupby("class_id", sort=True):
        algorithms = sorted(
            class_group["algorithm_id"].dropna().astype(str).unique().tolist())
        for algorithm_a, algorithm_b in combinations(algorithms, 2):
            pair_group = class_group[class_group["algorithm_id"].isin(
                [algorithm_a, algorithm_b])]
            for metric in metrics:
                if metric not in pair_group.columns:
                    continue
                filtered = pair_group.copy()
                if metric in {"objective"}:
                    filtered = filtered[filtered["feasible"] == True]
                if metric in {"gap_to_ilp_opt_pct"}:
                    filtered = filtered[np.isfinite(
                        filtered["gap_to_ilp_opt_pct"])]
                filtered = filtered[np.isfinite(filtered[metric])]

                pivot = (
                    filtered.pivot_table(
                        index=pair_key,
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


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _set_zero_baseline_with_padding(ax: plt.Axes, values: pd.Series, pad_ratio: float = 0.03) -> None:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return
    vmax = float(vals.max())
    # Keep y=0 visible with a small margin below axis floor for readability.
    pad = max(1e-4, vmax * float(pad_ratio))
    current_top = float(ax.get_ylim()[1])
    target_top = max(current_top, vmax * 1.05 + pad)
    ax.set_ylim(bottom=-pad, top=target_top)


def _build_class_color_map(class_order: list[str]) -> dict[str, str]:
    color_map = dict(DEFAULT_CLASS_COLOR_MAP)
    missing = [c for c in class_order if c not in color_map]
    if missing:
        extra = sns.color_palette("tab20", n_colors=max(3, len(missing)))
        for idx, class_id in enumerate(missing):
            color_map[class_id] = extra[idx % len(extra)]
    return color_map


def _colorize_class_xticklabels(ax: plt.Axes, class_color_map: dict[str, str] | None) -> None:
    for tick in ax.get_xticklabels():
        tick.set_color("black")


def _colorize_class_yticklabels(ax: plt.Axes, class_color_map: dict[str, str] | None) -> None:
    for tick in ax.get_yticklabels():
        tick.set_color("black")


def _write_class_color_map(class_order: list[str], class_color_map: dict[str, str], path: Path) -> None:
    rows = [{"class_id": c, "color_hex": str(
        class_color_map.get(c, ""))} for c in class_order]
    pd.DataFrame(rows).to_csv(path, index=False)


def _parse_front_points_payload(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []

    payload: Any
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:
            return []
    else:
        payload = value

    if not isinstance(payload, list):
        return []

    points: list[tuple[float, float]] = []
    for item in payload:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                cost = float(item[0])
                card = float(item[1])
            except Exception:
                continue
            if np.isfinite(cost) and np.isfinite(card):
                points.append((cost, card))
            continue
        if isinstance(item, dict):
            try:
                cost = float(item.get("cost", item.get("objective")))
                card = float(item.get("set_count", item.get(
                    "selected_set_count", item.get("cardinality"))))
            except Exception:
                continue
            if np.isfinite(cost) and np.isfinite(card):
                points.append((cost, card))
    return points


def _nondominated_front_2d_min(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return strict non-dominated frontier for 2D minimization (f1, f2)."""
    if not points:
        return []
    unique = sorted(set(points), key=lambda xy: (float(xy[0]), float(xy[1])))
    front: list[tuple[float, float]] = []
    best_y = float("inf")
    for x, y in unique:
        fy = float(y)
        if fy < best_y:
            front.append((float(x), float(y)))
            best_y = fy
    return front


def _safe_sample_id(sample_id: Any) -> str:
    if isinstance(sample_id, (int, np.integer)):
        return f"{int(sample_id):03d}"
    text = str(sample_id)
    if text.isdigit():
        return f"{int(text):03d}"
    return text.replace("/", "_")


def _plot_pareto_per_sample(
    runs_df: pd.DataFrame,
    figures_dir: Path,
    results_dir: Path,
    pareto_algo_id: str,
) -> None:
    if "front_points_json" not in runs_df.columns:
        return

    root = figures_dir / "pareto_per_sample"
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.rglob("*.png"):
        stale.unlink(missing_ok=True)

    algo_id = str(pareto_algo_id).strip()
    work = runs_df.copy()
    if algo_id:
        work = work[work["algorithm_id"].astype(str) == algo_id].copy()
    if work.empty:
        pd.DataFrame(
            [
                {
                    "class_id": "",
                    "sample_id": "",
                    "algorithm_filter": algo_id,
                    "run_count": 0,
                    "points_total": 0,
                    "points_unique": 0,
                    "plot_path": "",
                    "status": "empty_after_filter",
                }
            ]
        ).to_csv(results_dir / "pareto_per_sample_index.csv", index=False)
        return

    summary_rows: list[dict[str, Any]] = []
    for (class_id, sample_id), group in work.groupby(["class_id", "sample_id"], sort=True):
        all_points: list[tuple[float, float]] = []
        for _, row in group.iterrows():
            pts = _parse_front_points_payload(row.get("front_points_json"))
            if pts:
                all_points.extend(pts)

        if not all_points:
            summary_rows.append(
                {
                    "class_id": class_id,
                    "sample_id": sample_id,
                    "algorithm_filter": algo_id,
                    "run_count": int(len(group)),
                    "points_total": 0,
                    "points_unique": 0,
                    "plot_path": "",
                    "status": "no_points",
                }
            )
            continue

        counter = Counter(all_points)
        unique_points = sorted(counter.keys(), key=lambda xy: (xy[0], xy[1]))
        strict_front = _nondominated_front_2d_min(unique_points)
        xs_front = [p[1] for p in strict_front]
        ys_front = [p[0] for p in strict_front]

        class_dir = root / str(class_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        sid = _safe_sample_id(sample_id)
        fig_path = class_dir / f"pareto_{class_id}_{sid}.png"

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        x_all = [p[1] for p in all_points]
        y_all = [p[0] for p in all_points]
        ax.scatter(
            x_all,
            y_all,
            s=10,
            c="#9a9a9a",
            alpha=0.25,
            edgecolors="none",
            label="All points from 5 repeats",
        )
        if strict_front:
            ax.scatter(
                xs_front,
                ys_front,
                s=34,
                c="#d62728",
                alpha=0.95,
                edgecolors="white",
                linewidths=0.3,
                label="Strict non-dominated front",
            )
            ax.plot(xs_front, ys_front, color="#d62728",
                    linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Vulnerability (f2, singly-covered items)")
        ax.set_ylabel("Total Cost (f1)")
        ax.set_title(f"Pareto Front | class={class_id} sample={sid}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", frameon=True)
        note = f"runs={len(group)} raw={len(all_points)} unique={len(unique_points)} front={len(strict_front)}"
        ax.text(0.02, 0.98, note, transform=ax.transAxes,
                va="top", ha="left", fontsize=8)
        _save_fig(fig, fig_path)

        summary_rows.append(
            {
                "class_id": class_id,
                "sample_id": sample_id,
                "algorithm_filter": algo_id,
                "run_count": int(len(group)),
                "points_total": int(len(all_points)),
                "points_unique": int(len(unique_points)),
                "points_frontier": int(len(strict_front)),
                "plot_path": str(fig_path),
                "status": "ok",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["class_id", "sample_id"])
    summary_df.to_csv(results_dir / "pareto_per_sample_index.csv", index=False)


def _parse_selected_sets_payload(selected_sets_json: Any, meta_json: Any) -> list[int] | None:
    payload: Any = None
    if isinstance(selected_sets_json, str) and selected_sets_json.strip():
        try:
            payload = json.loads(selected_sets_json)
        except Exception:
            payload = None
    elif isinstance(selected_sets_json, list):
        payload = selected_sets_json

    if payload is None and isinstance(meta_json, str) and meta_json.strip():
        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = None
        if isinstance(meta, dict) and isinstance(meta.get("selected_sets"), list):
            payload = meta.get("selected_sets")

    if not isinstance(payload, list):
        return None

    out: list[int] = []
    for x in payload:
        try:
            v = int(x)
        except Exception:
            continue
        if v >= 0:
            out.append(v)
    return sorted(set(out))


def _stable_trial_seed(*parts: Any) -> int:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _compute_drop_robustness(
    runs_df: pd.DataFrame,
    drop_pcts: list[int],
    trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_cols = [
        "algorithm_id",
        "class_id",
        "instance_id",
        "seed",
        "repeat_idx",
        "drop_pct",
        "trial_count",
        "feasible_rate_after_drop",
    ]
    summary_cols = [
        "algorithm_id",
        "class_id",
        "drop_pct",
        "sample_count",
        "feasible_rate_mean",
        "feasible_rate_std",
    ]
    required = {"algorithm_id", "class_id", "instance_id",
                "source_path", "seed", "repeat_idx", "feasible"}
    if not required.issubset(set(runs_df.columns)):
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    detail_rows: list[dict[str, Any]] = []
    instance_cache: dict[str, tuple[int, list[np.ndarray]]] = {}
    work = runs_df[runs_df["feasible"] == True].copy()
    if work.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    for _, row in work.iterrows():
        source_path = str(row.get("source_path", "")).strip()
        if not source_path:
            continue

        selected = _parse_selected_sets_payload(
            row.get("selected_sets_json"), row.get("meta_json"))
        if selected is None:
            continue
        if source_path not in instance_cache:
            try:
                inst = read_instance(
                    source_path, dataset_id=str(row.get("dataset_id", "")))
                set_arrays = [np.asarray(rec.items, dtype=int)
                              for rec in inst.sets]
                instance_cache[source_path] = (int(inst.n_items), set_arrays)
            except Exception:
                continue
        n_items, set_arrays = instance_cache[source_path]

        for drop_pct in drop_pcts:
            if not selected:
                detail_rows.append(
                    {
                        "algorithm_id": str(row["algorithm_id"]),
                        "class_id": str(row["class_id"]),
                        "instance_id": str(row["instance_id"]),
                        "seed": int(row["seed"]) if pd.notna(row["seed"]) else -1,
                        "repeat_idx": int(row["repeat_idx"]) if pd.notna(row["repeat_idx"]) else -1,
                        "drop_pct": int(drop_pct),
                        "trial_count": int(trials),
                        "feasible_rate_after_drop": 0.0,
                    }
                )
                continue

            drop_k = min(len(selected), max(
                1, int(math.ceil(len(selected) * float(drop_pct) / 100.0))))
            success = 0
            for t in range(trials):
                rng = random.Random(
                    _stable_trial_seed(
                        row.get("algorithm_id", ""),
                        row.get("instance_id", ""),
                        row.get("seed", ""),
                        row.get("repeat_idx", ""),
                        drop_pct,
                        t,
                    )
                )
                dropped = set(rng.sample(selected, drop_k))
                remain = [j for j in selected if j not in dropped]

                cover = np.zeros(n_items, dtype=bool)
                for j in remain:
                    if 0 <= j < len(set_arrays):
                        items = set_arrays[j]
                        if items.size:
                            cover[items] = True
                if bool(np.all(cover)):
                    success += 1

            detail_rows.append(
                {
                    "algorithm_id": str(row["algorithm_id"]),
                    "class_id": str(row["class_id"]),
                    "instance_id": str(row["instance_id"]),
                    "seed": int(row["seed"]) if pd.notna(row["seed"]) else -1,
                    "repeat_idx": int(row["repeat_idx"]) if pd.notna(row["repeat_idx"]) else -1,
                    "drop_pct": int(drop_pct),
                    "trial_count": int(trials),
                    "feasible_rate_after_drop": float(success / float(trials)),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    summary_df = (
        detail_df.groupby(["algorithm_id", "class_id",
                          "drop_pct"], as_index=False)
        .agg(
            sample_count=("feasible_rate_after_drop", "count"),
            feasible_rate_mean=("feasible_rate_after_drop", "mean"),
            feasible_rate_std=("feasible_rate_after_drop", "std"),
        )
        .sort_values(["class_id", "drop_pct", "algorithm_id"])
        .reset_index(drop=True)
    )
    return detail_df, summary_df


def _parse_curve_payload(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[float] = []
        for x in value:
            try:
                fx = float(x)
            except Exception:
                continue
            if np.isfinite(fx):
                out.append(fx)
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:
            return []
        return _parse_curve_payload(payload)
    return []


def _extract_convergence_curve(row: pd.Series) -> list[float]:
    if "convergence_curve_json" in row.index:
        curve = _parse_curve_payload(row.get("convergence_curve_json"))
        if curve:
            return curve

    meta_raw = row.get("meta_json", "")
    if isinstance(meta_raw, str) and meta_raw.strip():
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}
        if isinstance(meta, dict):
            return _parse_curve_payload(meta.get("convergence_curve"))

    return []


def _mean_curve(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray] | None:
    valid = [c for c in curves if c]
    if not valid:
        return None
    max_len = max(len(c) for c in valid)
    mat = np.empty((len(valid), max_len), dtype=float)
    for i, curve in enumerate(valid):
        arr = np.asarray(curve, dtype=float)
        mat[i, : len(arr)] = arr
        if len(arr) < max_len:
            mat[i, len(arr):] = arr[-1]

    y = np.mean(mat, axis=0)
    if not np.any(np.isfinite(y)):
        return None
    x = np.arange(1, max_len + 1, dtype=int)
    return x, y


def _plot_convergence_curves_by_class(
    runs_df: pd.DataFrame,
    path: Path,
    class_order: list[str],
    algo_order: list[str],
    ilp_id: str,
    class_color_map: dict[str, str] | None = None,
) -> None:
    classes = [c for c in class_order if c in set(
        runs_df["class_id"].astype(str).unique())]
    if not classes:
        return

    preferred_algos = ["ga", "hgasa"]
    plot_algos = [a for a in preferred_algos if a in set(algo_order)]
    if not plot_algos:
        return
    ncols = 3
    nrows = int(math.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5.2 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    palette = sns.color_palette("tab10", n_colors=max(3, len(plot_algos)))
    color_map = {alg: palette[idx % len(palette)]
                 for idx, alg in enumerate(plot_algos)}
    label_map = {"ga": "GA", "hgasa": "HGASA"}

    for i, class_id in enumerate(classes):
        ax = axes_flat[i]
        sub = runs_df[runs_df["class_id"] == class_id].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue

        plotted = 0
        for alg in plot_algos:
            alg_sub = sub[sub["algorithm_id"] == alg]
            curves: list[list[float]] = []
            for _, row in alg_sub.iterrows():
                curve = _extract_convergence_curve(row)
                if curve:
                    curves.append(curve)

            mean_xy = _mean_curve(curves)
            if mean_xy is None:
                continue
            x, y = mean_xy
            ax.plot(x, y, label=label_map.get(alg, alg),
                    linewidth=1.8, color=color_map.get(alg))
            plotted += 1

        ilp_sub = sub[(sub["algorithm_id"] == ilp_id)
                      & (sub["feasible"] == True)]
        ilp_sub = ilp_sub[np.isfinite(ilp_sub["objective"])]
        if not ilp_sub.empty:
            ilp_cost = float(ilp_sub["objective"].mean())
            ax.axhline(ilp_cost, color="orange", linestyle=":",
                       linewidth=2.0, label="ILP mean cost")

        ax.set_title(f"class: {class_id}", color="black")
        ax.set_xlabel("Iteration / Generation Step")
        ax.set_ylabel("Solution Cost")
        if plotted == 0:
            ax.text(0.5, 0.5, "No convergence curve data",
                    ha="center", va="center", transform=ax.transAxes)
        ax.legend(loc="best", fontsize=8)

    for j in range(len(classes), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Convergence Curves by Class (Orange: ILP Mean Cost)", y=1.02)
    _save_fig(fig, path)


def _plot_bar(
    data: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    path: Path,
    class_order: list[str],
    algo_order: list[str],
    force_y_min_zero: bool = False,
    class_color_map: dict[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=data,
        x="class_id",
        y=y,
        hue="algorithm_id",
        order=class_order,
        hue_order=algo_order,
        estimator="mean",
        errorbar="sd",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("class_id")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    _colorize_class_xticklabels(ax, class_color_map)
    if force_y_min_zero:
        _set_zero_baseline_with_padding(ax, data[y])
    _save_fig(fig, path)


def _plot_ci_point(
    data: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    path: Path,
    class_order: list[str],
    algo_order: list[str],
    force_y_min_zero: bool = False,
    class_color_map: dict[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.pointplot(
        data=data,
        x="class_id",
        y=y,
        hue="algorithm_id",
        order=class_order,
        hue_order=algo_order,
        estimator="mean",
        errorbar=("ci", 95),
        dodge=0.35,
        markers="o",
        linestyles="",
        capsize=0.12,
        err_kws={"linewidth": 1.2},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("class_id")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    _colorize_class_xticklabels(ax, class_color_map)
    if force_y_min_zero:
        _set_zero_baseline_with_padding(ax, data[y])
    _save_fig(fig, path)


def _sort_axis_values(values: pd.Series) -> list[Any]:
    vals = list(pd.Series(values).dropna().unique())
    numeric = pd.to_numeric(pd.Series(vals), errors="coerce")
    if numeric.notna().all():
        zipped = sorted(zip(vals, numeric.tolist()), key=lambda x: x[1])
        return [v for v, _ in zipped]
    return sorted(vals, key=lambda x: str(x))


def _plot_line_by_axis(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    class_order: list[str],
    algo_order: list[str],
    class_color_map: dict[str, str] | None = None,
) -> None:
    classes = [c for c in class_order if c in set(
        data["class_id"].astype(str).unique())]
    if not classes:
        return

    ncols = 3
    nrows = int(math.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    legend_handles = None
    legend_labels = None

    for i, class_id in enumerate(classes):
        ax = axes_flat[i]
        sub = data[data["class_id"] == class_id].copy()
        if metric == "objective":
            sub = sub[sub["feasible"] == True]
        if metric == "gap_to_ilp_opt_pct":
            sub = sub[np.isfinite(sub["gap_to_ilp_opt_pct"])]
        sub = sub[np.isfinite(sub[metric])]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(class_id, color="black")
            ax.axis("off")
            continue

        axis_name = str(sub["axis_name"].iloc[0])
        axis_order = _sort_axis_values(sub["axis_value"])
        sub["axis_value"] = pd.Categorical(
            sub["axis_value"], categories=axis_order, ordered=True)

        sns.lineplot(
            data=sub,
            x="axis_value",
            y=metric,
            hue="algorithm_id",
            hue_order=algo_order,
            estimator="mean",
            errorbar=("ci", 95),
            marker="o",
            ax=ax,
        )
        ax.set_title(class_id, color="black")
        ax.set_xlabel(axis_name)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        if metric == "runtime_sec":
            _set_zero_baseline_with_padding(ax, sub[metric])

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()

    for j in range(len(classes), len(axes_flat)):
        axes_flat[j].axis("off")

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels,
                   loc="upper center", ncol=min(4, len(legend_labels)))
    fig.suptitle(title, y=1.02)
    _save_fig(fig, path)


def _plot_rank_heatmap_by_metric(
    summary_class_df: pd.DataFrame,
    metric_col: str,
    title: str,
    path: Path,
    class_color_map: dict[str, str] | None = None,
) -> None:
    if metric_col not in summary_class_df.columns:
        return

    work = summary_class_df.copy()
    rows: list[dict[str, Any]] = []
    for class_id, group in work.groupby("class_id", sort=True):
        g = group.copy()
        metric = pd.to_numeric(g[metric_col], errors="coerce")
        rank = metric.rank(method="min", ascending=True, na_option="bottom")
        g["rank_value"] = rank
        for _, row in g.iterrows():
            rows.append(
                {
                    "class_id": class_id,
                    "algorithm_id": row["algorithm_id"],
                    "rank_value": float(row["rank_value"]),
                }
            )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return

    pivot = score_df.pivot(
        index="class_id", columns="algorithm_id", values="rank_value")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("algorithm_id")
    ax.set_ylabel("class_id")
    _colorize_class_yticklabels(ax, class_color_map)
    _save_fig(fig, path)


def _plot_rank_heatmap_weighted(
    summary_class_df: pd.DataFrame,
    quality_metric_col: str,
    runtime_metric_col: str,
    quality_weight: float,
    runtime_weight: float,
    title: str,
    path: Path,
    class_color_map: dict[str, str] | None = None,
) -> None:
    if quality_metric_col not in summary_class_df.columns or runtime_metric_col not in summary_class_df.columns:
        return

    wq = float(quality_weight)
    wr = float(runtime_weight)
    if wq < 0 or wr < 0 or (wq + wr) <= 0:
        return
    scale = wq + wr
    wq /= scale
    wr /= scale

    rows: list[dict[str, Any]] = []
    for class_id, group in summary_class_df.groupby("class_id", sort=True):
        g = group.copy()
        q_metric = pd.to_numeric(g[quality_metric_col], errors="coerce")
        t_metric = pd.to_numeric(g[runtime_metric_col], errors="coerce")
        q_rank = q_metric.rank(
            method="min", ascending=True, na_option="bottom")
        t_rank = t_metric.rank(
            method="min", ascending=True, na_option="bottom")
        score = wq * q_rank + wr * t_rank
        g["rank_value"] = score
        for _, row in g.iterrows():
            rows.append(
                {
                    "class_id": class_id,
                    "algorithm_id": row["algorithm_id"],
                    "rank_value": float(row["rank_value"]),
                }
            )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return

    pivot = score_df.pivot(
        index="class_id", columns="algorithm_id", values="rank_value")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("algorithm_id")
    ax.set_ylabel("class_id")
    _colorize_class_yticklabels(ax, class_color_map)
    _save_fig(fig, path)


def _resolve_stability_metric_col(summary_class_df: pd.DataFrame) -> str | None:
    candidates = [
        "stability_var_mean",
        "gap_to_ilp_opt_pct_std",
        "objective_std",
    ]
    for metric_col in candidates:
        if metric_col not in summary_class_df.columns:
            continue
        values = pd.to_numeric(summary_class_df[metric_col], errors="coerce")
        if np.isfinite(values).any():
            return metric_col
    return None


def _plot_rank_heatmap_weighted_3d(
    summary_class_df: pd.DataFrame,
    quality_metric_col: str,
    runtime_metric_col: str,
    stability_metric_col: str,
    quality_weight: float,
    runtime_weight: float,
    stability_weight: float,
    title: str,
    path: Path,
    class_color_map: dict[str, str] | None = None,
) -> None:
    required = [quality_metric_col, runtime_metric_col, stability_metric_col]
    if any(metric_col not in summary_class_df.columns for metric_col in required):
        return

    wq = float(quality_weight)
    wr = float(runtime_weight)
    ws = float(stability_weight)
    if wq < 0 or wr < 0 or ws < 0 or (wq + wr + ws) <= 0:
        return
    scale = wq + wr + ws
    wq /= scale
    wr /= scale
    ws /= scale

    rows: list[dict[str, Any]] = []
    for class_id, group in summary_class_df.groupby("class_id", sort=True):
        g = group.copy()
        q_metric = pd.to_numeric(g[quality_metric_col], errors="coerce")
        t_metric = pd.to_numeric(g[runtime_metric_col], errors="coerce")
        s_metric = pd.to_numeric(g[stability_metric_col], errors="coerce")
        q_rank = q_metric.rank(
            method="min", ascending=True, na_option="bottom")
        t_rank = t_metric.rank(
            method="min", ascending=True, na_option="bottom")
        s_rank = s_metric.rank(
            method="min", ascending=True, na_option="bottom")
        score = wq * q_rank + wr * t_rank + ws * s_rank
        g["rank_value"] = score
        for _, row in g.iterrows():
            rows.append(
                {
                    "class_id": class_id,
                    "algorithm_id": row["algorithm_id"],
                    "rank_value": float(row["rank_value"]),
                }
            )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return

    pivot = score_df.pivot(
        index="class_id", columns="algorithm_id", values="rank_value")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("algorithm_id")
    ax.set_ylabel("class_id")
    _colorize_class_yticklabels(ax, class_color_map)
    _save_fig(fig, path)


def _plot_quality_runtime_distribution(
    runs_df: pd.DataFrame,
    path: Path,
    ilp_id: str,
    class_order: list[str],
    algo_order: list[str],
    class_color_map: dict[str, str] | None = None,
) -> None:
    algo_order_non_ilp = [a for a in algo_order if a != ilp_id]
    if not algo_order_non_ilp:
        return

    quality = runs_df[
        (runs_df["algorithm_id"] != ilp_id)
        & (runs_df["feasible"] == True)
        & np.isfinite(runs_df["gap_to_ilp_opt_pct"])
    ].copy()
    runtime = runs_df[
        (runs_df["algorithm_id"] != ilp_id)
        & np.isfinite(runs_df["runtime_sec"])
        & (runs_df["runtime_sec"] > 0)
    ].copy()

    if quality.empty and runtime.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    meanprops = {
        "marker": "D",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": 4,
    }

    if quality.empty:
        axes[0].text(0.5, 0.5, "No quality data", ha="center", va="center")
        axes[0].axis("off")
    else:
        sns.boxplot(
            data=quality,
            x="class_id",
            y="gap_to_ilp_opt_pct",
            hue="algorithm_id",
            order=class_order,
            hue_order=algo_order_non_ilp,
            showmeans=True,
            meanprops=meanprops,
            ax=axes[0],
        )
        axes[0].set_title("Solution Quality Distribution (Lower is Better)")
        axes[0].set_xlabel("class_id")
        axes[0].set_ylabel("Gap to ILP Opt (%)")
        axes[0].set_ylim(bottom=0.0)
        axes[0].tick_params(axis="x", rotation=20)
        _colorize_class_xticklabels(axes[0], class_color_map)

    if runtime.empty:
        axes[1].text(0.5, 0.5, "No runtime data", ha="center", va="center")
        axes[1].axis("off")
    else:
        sns.boxplot(
            data=runtime,
            x="class_id",
            y="runtime_sec",
            hue="algorithm_id",
            order=class_order,
            hue_order=algo_order_non_ilp,
            showmeans=True,
            meanprops=meanprops,
            ax=axes[1],
        )
        axes[1].set_title("Runtime Distribution by Category and Algorithm")
        axes[1].set_xlabel("class_id")
        axes[1].set_ylabel("runtime_sec")
        axes[1].set_ylim(bottom=0.0)
        axes[1].tick_params(axis="x", rotation=20)
        _colorize_class_xticklabels(axes[1], class_color_map)

    handles = None
    labels = None
    for ax in axes:
        if ax.legend_ is not None:
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(4, len(labels)))

    _save_fig(fig, path)


def _plot_tradeoff_by_class(
    runs_df: pd.DataFrame,
    path: Path,
    ilp_id: str,
    class_order: list[str],
    algo_order: list[str],
    class_color_map: dict[str, str],
) -> None:
    work = runs_df[
        (runs_df["algorithm_id"] != ilp_id)
        & (runs_df["feasible"] == True)
        & np.isfinite(runs_df["runtime_sec"])
        & (runs_df["runtime_sec"] > 0)
        & np.isfinite(runs_df["gap_to_ilp_opt_pct"])
    ].copy()
    if work.empty:
        return

    agg = (
        work.groupby(["class_id", "algorithm_id"], as_index=False)
        .agg(
            runtime_sec_mean=("runtime_sec", "mean"),
            gap_to_ilp_opt_pct_mean=("gap_to_ilp_opt_pct", "mean"),
        )
        .copy()
    )
    if agg.empty:
        return

    agg["class_id"] = pd.Categorical(
        agg["class_id"], categories=class_order, ordered=True)
    agg = agg.sort_values(["class_id", "algorithm_id"])

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sns.scatterplot(
        data=agg,
        x="runtime_sec_mean",
        y="gap_to_ilp_opt_pct_mean",
        hue="class_id",
        hue_order=class_order,
        style="algorithm_id",
        style_order=algo_order,
        palette=class_color_map,
        s=85,
        alpha=0.95,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("runtime_sec_mean (log scale)")
    ax.set_ylabel("gap_to_ilp_opt_pct_mean (%)")
    ax.set_title("Quality-Runtime Tradeoff (Color=Class, Marker=Algorithm)")
    ax.grid(True, linestyle="--", alpha=0.3)
    _save_fig(fig, path)


def _plot_drop_robustness_by_class(
    robustness_detail_df: pd.DataFrame,
    path: Path,
    class_order: list[str],
    algo_order: list[str],
    class_color_map: dict[str, str] | None = None,
) -> None:
    if robustness_detail_df.empty:
        return

    classes = [c for c in class_order if c in set(
        robustness_detail_df["class_id"].astype(str).unique())]
    if not classes:
        return

    ncols = 3
    nrows = int(math.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5.2 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    x_order = ROBUST_DROP_PCTS

    legend_handles = None
    legend_labels = None

    for i, class_id in enumerate(classes):
        ax = axes_flat[i]
        sub = robustness_detail_df[robustness_detail_df["class_id"] == class_id].copy(
        )
        sub = sub.sort_values("drop_pct")
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue

        sns.lineplot(
            data=sub,
            x="drop_pct",
            y="feasible_rate_after_drop",
            hue="algorithm_id",
            hue_order=algo_order,
            estimator="mean",
            errorbar=("ci", 95),
            marker="o",
            sort=False,
            ax=ax,
        )
        ax.set_title(f"class: {class_id}", color="black")
        ax.set_xlabel("Removed selected sets (%)")
        ax.set_ylabel("Feasible rate after drop")
        ax.set_xticks(x_order)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.3)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()

    for j in range(len(classes), len(axes_flat)):
        axes_flat[j].axis("off")

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels,
                   loc="upper center", ncol=min(4, len(legend_labels)))
    fig.suptitle(
        "Robustness Test: Feasible Rate After Random Set Removal", y=1.02)
    _save_fig(fig, path)


def _complexity_axis_class_mask(
    class_ids: pd.Series,
    axis_name: str,
    class_axis_map: dict[str, str],
) -> pd.Series:
    mapped_axis = class_ids.astype(str).map(class_axis_map)
    if axis_name == "problem_size":
        return mapped_axis.isin({"set_count", "item_count", "problem_size"})
    return mapped_axis == axis_name


def _complexity_axes_frames(
    runs_df: pd.DataFrame,
    class_axis_map: dict[str, str] | None = None,
    axis_matched_only: bool = True,
) -> dict[str, pd.DataFrame]:
    axis_map = class_axis_map or DEFAULT_CLASS_AXIS_MAP
    work = runs_df.copy()
    work["class_id"] = work["class_id"].astype(str)
    work["problem_size"] = work["set_count"].to_numpy(
        dtype=float) * work["item_count"].to_numpy(dtype=float)
    out: dict[str, pd.DataFrame] = {}
    for axis_name in ["set_count", "item_count", "problem_size"]:
        sub = work[(work["runtime_sec"] > 0) & (work[axis_name] > 0)].copy()
        if axis_matched_only and not sub.empty:
            mask = _complexity_axis_class_mask(
                sub["class_id"], axis_name, axis_map)
            if bool(mask.any()):
                sub = sub[mask].copy()
        if sub.empty:
            out[axis_name] = sub
            continue
        sub["log_x"] = np.log10(sub[axis_name].to_numpy(dtype=float))
        sub["log_runtime"] = np.log10(sub["runtime_sec"].to_numpy(dtype=float))
        out[axis_name] = sub
    return out


def _complexity_fit_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    axis_frames = _complexity_axes_frames(
        runs_df,
        class_axis_map=DEFAULT_CLASS_AXIS_MAP,
        axis_matched_only=True,
    )
    for axis_name, axis_df in axis_frames.items():
        for algorithm_id, group in axis_df.groupby("algorithm_id", sort=True):
            if len(group) < 2:
                rows.append(
                    {
                        "axis_filter": "axis_matched",
                        "axis_name": axis_name,
                        "algorithm_id": algorithm_id,
                        "slope": np.nan,
                        "intercept": np.nan,
                        "r2": np.nan,
                        "sample_size": int(len(group)),
                        "class_count": int(group["class_id"].nunique()),
                    }
                )
                continue
            x = group["log_x"].to_numpy(dtype=float)
            y = group["log_runtime"].to_numpy(dtype=float)
            if float(np.var(x)) == 0.0:
                rows.append(
                    {
                        "axis_filter": "axis_matched",
                        "axis_name": axis_name,
                        "algorithm_id": algorithm_id,
                        "slope": np.nan,
                        "intercept": np.nan,
                        "r2": np.nan,
                        "sample_size": int(len(group)),
                        "class_count": int(group["class_id"].nunique()),
                    }
                )
                continue
            slope, intercept = np.polyfit(x, y, deg=1)
            y_pred = slope * x + intercept
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
            rows.append(
                {
                    "axis_filter": "axis_matched",
                    "axis_name": axis_name,
                    "algorithm_id": algorithm_id,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r2": float(r2),
                    "sample_size": int(len(group)),
                    "class_count": int(group["class_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _plot_complexity_loglog(runs_df: pd.DataFrame, fit_df: pd.DataFrame, path: Path) -> None:
    axis_frames = _complexity_axes_frames(
        runs_df,
        class_axis_map=DEFAULT_CLASS_AXIS_MAP,
        axis_matched_only=True,
    )
    axis_order = ["set_count", "item_count", "problem_size"]
    axis_labels = {
        "set_count": "log10(set_count)",
        "item_count": "log10(item_count)",
        "problem_size": "log10(problem_size=n_items*n_sets)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), squeeze=False)
    axes_flat = axes.flatten()
    algo_order = sorted(
        runs_df["algorithm_id"].dropna().astype(str).unique().tolist())
    palette = sns.color_palette(n_colors=max(3, len(algo_order)))
    color_map = {alg: palette[idx % len(palette)]
                 for idx, alg in enumerate(algo_order)}

    for i, axis_name in enumerate(axis_order):
        ax = axes_flat[i]
        sub = axis_frames.get(axis_name, pd.DataFrame())
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        sns.scatterplot(
            data=sub,
            x="log_x",
            y="log_runtime",
            hue="algorithm_id",
            hue_order=algo_order,
            alpha=0.35,
            s=24,
            ax=ax,
        )

        fit_sub = fit_df[fit_df["axis_name"] ==
                         axis_name] if "axis_name" in fit_df.columns else fit_df
        for _, row in fit_sub.iterrows():
            alg = row["algorithm_id"]
            slope = row["slope"]
            intercept = row["intercept"]
            if not np.isfinite(slope) or not np.isfinite(intercept):
                continue
            alg_sub = sub[sub["algorithm_id"] == alg]
            if alg_sub.empty:
                continue
            xs = np.linspace(float(alg_sub["log_x"].min()), float(
                alg_sub["log_x"].max()), 100)
            ys = slope * xs + intercept
            fit_line = pd.DataFrame({"x": xs, "y": ys})
            sns.lineplot(
                data=fit_line,
                x="x",
                y="y",
                color=color_map.get(str(alg)),
                linewidth=2.0,
                ax=ax,
                legend=False,
            )

        ax.set_title(f"log({axis_name}) vs log(runtime_sec)")
        ax.set_xlabel(axis_labels.get(axis_name, f"log10({axis_name})"))
        ax.set_ylabel("log10(runtime_sec)")
        used_classes = sorted(sub["class_id"].astype(str).unique().tolist())
        if used_classes:
            shown = ", ".join(used_classes[:4])
            if len(used_classes) > 4:
                shown += ", ..."
            ax.text(
                0.02,
                0.02,
                f"classes: {shown}",
                transform=ax.transAxes,
                fontsize=8,
                alpha=0.85,
                ha="left",
                va="bottom",
            )
        if ax.legend_ is not None:
            ax.legend_.remove()

    legend_handles = [
        plt.Line2D([0], [0], color=color_map[alg], marker="o",
                   linestyle="", markersize=6, alpha=0.8)
        for alg in algo_order
    ]
    fig.legend(legend_handles, algo_order, loc="upper center",
               ncol=min(6, len(algo_order)))
    fig.suptitle("Complexity Trend (Multi-axis Log-Log Fit)", y=1.02)
    _save_fig(fig, path)


def _plot_significance_heatmaps(
    significance_df: pd.DataFrame,
    figures_dir: Path,
    alpha: float,
    metric: str = "gap_to_ilp_opt_pct",
) -> None:
    sub = significance_df[significance_df["metric"] == metric].copy()
    if sub.empty:
        return

    for class_id, class_group in sub.groupby("class_id", sort=True):
        algorithms = sorted(
            set(class_group["algorithm_a"].dropna().astype(str).tolist())
            | set(class_group["algorithm_b"].dropna().astype(str).tolist())
        )
        if not algorithms:
            continue
        mat = pd.DataFrame(np.nan, index=algorithms,
                           columns=algorithms, dtype=float)
        for _, row in class_group.iterrows():
            a = str(row["algorithm_a"])
            b = str(row["algorithm_b"])
            p = row["p_value"]
            if np.isfinite(p):
                mat.loc[a, b] = float(p)
                mat.loc[b, a] = float(p)

        annot = pd.DataFrame("", index=algorithms,
                             columns=algorithms, dtype=object)
        for a in algorithms:
            for b in algorithms:
                if a == b:
                    annot.loc[a, b] = ""
                    continue
                p = mat.loc[a, b]
                if np.isfinite(p):
                    annot.loc[a, b] = f"{p:.3g}{'*' if p < alpha else ''}"
                else:
                    annot.loc[a, b] = ""

        mask = np.eye(len(algorithms), dtype=bool)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            mat,
            annot=annot,
            fmt="",
            cmap="YlOrRd_r",
            vmin=0.0,
            vmax=1.0,
            mask=mask,
            ax=ax,
            cbar_kws={"label": "p-value"},
        )
        ax.set_title(
            f"Significance p-value Heatmap ({class_id})  *: p<{alpha}")
        ax.set_xlabel("algorithm_id")
        ax.set_ylabel("algorithm_id")
        _save_fig(fig, figures_dir / f"sig_heatmap_{class_id}.png")


def _generate_figures(
    runs_df: pd.DataFrame,
    runs_with_axis_df: pd.DataFrame,
    summary_class_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    robustness_detail_df: pd.DataFrame,
    figures_dir: Path,
    alpha: float,
    ilp_id: str,
    class_color_map: dict[str, str],
) -> None:
    sns.set_theme(style="whitegrid")
    class_order = sorted(
        runs_df["class_id"].dropna().astype(str).unique().tolist())
    special_classes = {"special_clustered", "special_hub"}
    line_class_order = [c for c in class_order if c not in special_classes]
    special_class_order = [c for c in class_order if c in special_classes]
    algo_order = sorted(
        runs_df["algorithm_id"].dropna().astype(str).unique().tolist())

    _plot_bar(
        data=runs_df,
        y="runtime_sec",
        ylabel="runtime_sec",
        title="Runtime by Class",
        path=figures_dir / "bar_runtime_by_class.png",
        class_order=class_order,
        algo_order=algo_order,
        force_y_min_zero=True,
        class_color_map=class_color_map,
    )
    _plot_bar(
        data=runs_df[np.isfinite(runs_df["gap_to_ilp_opt_pct"])],
        y="gap_to_ilp_opt_pct",
        ylabel="gap_to_ilp_opt_pct (%)",
        title="Gap to ILP Opt by Class",
        path=figures_dir / "bar_gap_ilp_by_class.png",
        class_order=class_order,
        algo_order=algo_order,
        class_color_map=class_color_map,
    )
    _plot_ci_point(
        data=runs_df,
        y="runtime_sec",
        ylabel="runtime_sec (95% CI)",
        title="Runtime by Class (Mean with 95% CI)",
        path=figures_dir / "ci_runtime_by_class.png",
        class_order=class_order,
        algo_order=algo_order,
        force_y_min_zero=True,
        class_color_map=class_color_map,
    )
    _plot_ci_point(
        data=runs_df[np.isfinite(runs_df["gap_to_ilp_opt_pct"])],
        y="gap_to_ilp_opt_pct",
        ylabel="gap_to_ilp_opt_pct (%) (95% CI)",
        title="Gap to ILP Opt by Class (Mean with 95% CI)",
        path=figures_dir / "ci_gap_ilp_by_class.png",
        class_order=class_order,
        algo_order=algo_order,
        class_color_map=class_color_map,
    )

    _plot_line_by_axis(
        data=runs_with_axis_df,
        metric="runtime_sec",
        ylabel="runtime_sec",
        title="Runtime vs Class Axis",
        path=figures_dir / "line_runtime_by_axis.png",
        class_order=line_class_order,
        algo_order=algo_order,
        class_color_map=class_color_map,
    )
    _plot_line_by_axis(
        data=runs_with_axis_df,
        metric="gap_to_ilp_opt_pct",
        ylabel="gap_to_ilp_opt_pct (%)",
        title="Gap to ILP Opt vs Class Axis",
        path=figures_dir / "line_gap_ilp_by_axis.png",
        class_order=line_class_order,
        algo_order=algo_order,
        class_color_map=class_color_map,
    )

    _plot_quality_runtime_distribution(
        runs_df=runs_df,
        path=figures_dir / "box_quality_runtime.png",
        ilp_id=ilp_id,
        class_order=class_order,
        algo_order=algo_order,
        class_color_map=class_color_map,
    )
    if special_class_order:
        _plot_quality_runtime_distribution(
            runs_df=runs_df[runs_df["class_id"].isin(
                special_class_order)].copy(),
            path=figures_dir / "box_quality_runtime_special.png",
            ilp_id=ilp_id,
            class_order=special_class_order,
            algo_order=algo_order,
            class_color_map=class_color_map,
        )

    _plot_tradeoff_by_class(
        runs_df=runs_df,
        path=figures_dir / "scatter_tradeoff_by_class.png",
        ilp_id=ilp_id,
        class_order=class_order,
        algo_order=algo_order,
        class_color_map=class_color_map,
    )

    _plot_rank_heatmap_by_metric(
        summary_class_df=summary_class_df,
        metric_col="gap_to_ilp_opt_pct_mean",
        title="Quality Rank Heatmap by Class (Gap to ILP mean, lower is better)",
        path=figures_dir / "heatmap_rank_quality_by_class.png",
        class_color_map=class_color_map,
    )
    _plot_rank_heatmap_by_metric(
        summary_class_df=summary_class_df,
        metric_col="runtime_sec_mean",
        title="Runtime Rank Heatmap by Class (mean, lower is better)",
        path=figures_dir / "heatmap_rank_runtime_by_class.png",
        class_color_map=class_color_map,
    )
    for quality_weight, runtime_weight, tag in [
        (0.5, 0.5, "5050"),
        (0.7, 0.3, "7030"),
        (0.3, 0.7, "3070"),
    ]:
        _plot_rank_heatmap_weighted(
            summary_class_df=summary_class_df,
            quality_metric_col="gap_to_ilp_opt_pct_mean",
            runtime_metric_col="runtime_sec_mean",
            quality_weight=quality_weight,
            runtime_weight=runtime_weight,
            title=(
                f"Weighted Rank Heatmap by Class (Quality mean {quality_weight:.1f} + "
                f"Runtime mean {runtime_weight:.1f}, lower is better)"
            ),
            path=figures_dir / f"heatmap_rank_weighted_{tag}_by_class.png",
            class_color_map=class_color_map,
        )
    _plot_complexity_loglog(runs_df, fit_df, figures_dir /
                            "complexity_loglog_runtime_fit.png")
    _plot_convergence_curves_by_class(
        runs_df=runs_df,
        path=figures_dir / "convergence_curves_by_class.png",
        class_order=class_order,
        algo_order=algo_order,
        ilp_id=ilp_id,
        class_color_map=class_color_map,
    )
    _plot_significance_heatmaps(
        significance_df, figures_dir, alpha=alpha, metric="gap_to_ilp_opt_pct")
    _plot_drop_robustness_by_class(
        robustness_detail_df=robustness_detail_df,
        path=figures_dir / "robust_feasible_rate_drop_by_class.png",
        class_order=class_order,
        algo_order=algo_order,
        class_color_map=class_color_map,
    )


def main() -> None:
    if sns is None:
        raise RuntimeError(
            "Missing dependency 'seaborn'. Please install dependencies first (e.g. `uv sync` or pip install seaborn)."
        ) from _SEABORN_IMPORT_ERROR

    args = _parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "analysis"
    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in figures_dir.glob("*.png"):
        stale_png.unlink(missing_ok=True)

    merged, manifest = _load_runs_from_root(runs_root)
    algorithm_filter = _parse_algorithm_filter(args.algorithms)
    if algorithm_filter:
        available = sorted(
            merged["algorithm_id"].dropna().astype(str).unique().tolist())
        missing = [
            algorithm_id for algorithm_id in algorithm_filter if algorithm_id not in available]
        if missing:
            raise ValueError(
                f"--algorithms 包含未找到的 algorithm_id: {missing}; 可选={available}")

        has_ilp = any("ilp" in algorithm_id.lower()
                      for algorithm_id in algorithm_filter)
        if not has_ilp:
            raise ValueError(
                "--algorithms 必须至少包含一个 ILP 算法（如 ilp_pulp 或 ilp_ortools）"
            )

        keep = set(algorithm_filter)
        merged = merged[merged["algorithm_id"].astype(
            str).isin(keep)].copy()
        if "algorithm_id" in manifest.columns:
            manifest = manifest[manifest["algorithm_id"].astype(
                str).isin(keep)].copy()

    resolved_ilp_id = _resolve_ilp_id(merged, requested_ilp_id=args.ilp_id)
    print(f"[analysis] ILP baseline id: {resolved_ilp_id}")
    merged = _add_ilp_baseline_gap(merged, ilp_id=resolved_ilp_id)
    class_order = sorted(
        merged["class_id"].dropna().astype(str).unique().tolist())
    class_color_map = _build_class_color_map(class_order)

    runs_with_axis = _add_axis_columns(merged, DEFAULT_CLASS_AXIS_MAP)
    summary_class = _summary_algo_class(merged)
    summary_axis = _summary_algo_class_axis(runs_with_axis)
    significance = _paired_significance(
        runs_df=merged,
        metrics=["gap_to_ilp_opt_pct", "runtime_sec"],
        method=str(args.test),
    )
    fit_df = _complexity_fit_table(merged)
    robustness_detail_df, robustness_summary_df = _compute_drop_robustness(
        merged,
        drop_pcts=ROBUST_DROP_PCTS,
        trials=ROBUST_TRIALS,
    )

    merged.to_csv(results_dir / "runs_merged.csv", index=False)
    summary_class.to_csv(results_dir / "summary_algo_class.csv", index=False)
    summary_axis.to_csv(
        results_dir / "summary_algo_class_axis.csv", index=False)
    significance.to_csv(results_dir / "significance_algo.csv", index=False)
    manifest.to_csv(results_dir / "manifest.csv", index=False)
    fit_df.to_csv(results_dir / "complexity_fit.csv", index=False)
    robustness_detail_df.to_csv(
        results_dir / "robustness_drop_feasible_detail.csv", index=False)
    robustness_summary_df.to_csv(
        results_dir / "robustness_drop_feasible_summary.csv", index=False)
    _write_class_color_map(class_order, class_color_map,
                           results_dir / "class_color_map.csv")

    _generate_figures(
        runs_df=merged,
        runs_with_axis_df=runs_with_axis,
        summary_class_df=summary_class,
        fit_df=fit_df,
        significance_df=significance,
        robustness_detail_df=robustness_detail_df,
        figures_dir=figures_dir,
        alpha=float(args.alpha),
        ilp_id=resolved_ilp_id,
        class_color_map=class_color_map,
    )
    _plot_pareto_per_sample(
        runs_df=merged,
        figures_dir=figures_dir,
        results_dir=results_dir,
        pareto_algo_id=str(args.pareto_algo_id),
    )

    print(f"Analysis done: {out_dir}")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
