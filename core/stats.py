from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


def complexity_fit(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    if runs_df.empty:
        return pd.DataFrame(rows)

    for signature, group in runs_df.groupby("param_signature"):
        x = group["set_count"].to_numpy(dtype=float)
        y = group["runtime_sec"].to_numpy(dtype=float)

        if len(x) < 2 or float(np.var(x)) == 0.0:
            rows.append(
                {
                    "param_signature": signature,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "sample_size": int(len(x)),
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
                "param_signature": signature,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "sample_size": int(len(x)),
            }
        )

    return pd.DataFrame(rows)


def paired_significance(
    runs_df: pd.DataFrame,
    metric: str = "objective",
    method: str = "wilcoxon",
) -> pd.DataFrame:
    columns = [
        "param_a",
        "param_b",
        "metric",
        "test",
        "n_pairs",
        "statistic",
        "p_value",
        "mean_diff_a_minus_b",
    ]

    if runs_df.empty or metric not in runs_df.columns:
        return pd.DataFrame(columns=columns)

    settings = sorted(runs_df["param_signature"].dropna().unique().tolist())
    if len(settings) < 2:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, float | int | str]] = []
    for a, b in combinations(settings, 2):
        subset = runs_df[runs_df["param_signature"].isin([a, b])]
        pivot = (
            subset.pivot_table(
                index=["instance_id", "seed", "repeat_idx"],
                columns="param_signature",
                values=metric,
                aggfunc="mean",
            )
            .dropna()
            .reset_index(drop=True)
        )

        if len(pivot) < 2:
            continue

        x = pivot[a].to_numpy(dtype=float)
        y = pivot[b].to_numpy(dtype=float)

        if np.allclose(x, y):
            stat = 0.0
            p_value = 1.0
            test_name = "identical_samples"
        elif stats is None:
            stat = np.nan
            p_value = np.nan
            test_name = "scipy_missing"
        elif method.lower() == "ttest":
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
                "param_a": a,
                "param_b": b,
                "metric": metric,
                "test": test_name,
                "n_pairs": int(len(pivot)),
                "statistic": float(stat) if stat == stat else np.nan,
                "p_value": float(p_value) if p_value == p_value else np.nan,
                "mean_diff_a_minus_b": float(np.mean(x) - np.mean(y)),
            }
        )

    return pd.DataFrame(rows, columns=columns)
