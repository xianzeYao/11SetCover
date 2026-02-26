from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from core.io_dataset import read_all_instances
from core.metrics import (
    add_gap_to_best,
    add_stability_var,
    convergence_speed,
    normalize_result,
    summarize_by_class,
    summarize_by_param,
)
from core.param_space import expand_grid, expand_ofat, param_signature
from core.stats import complexity_fit, paired_significance
from core.utils import ensure_dir, timestamp_id


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = config
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = value


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return config
    for key, value in overrides.items():
        if value is None:
            continue
        _set_nested(config, key, value)
    return config


def _load_solver(module_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, "solve"):
        raise AttributeError(f"算法模块缺少 solve(): {module_name}")
    return module.solve


def _build_param_points(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str, list[str]]:
    mode = str(cfg["experiment"]["mode"]).lower()
    if mode == "ofat":
        base_params = dict(cfg["experiment"]["ofat"].get("base_params", {}))
        sweep_param = str(cfg["experiment"]["ofat"].get("sweep_param", ""))
        sweep_values = list(cfg["experiment"]["ofat"].get("sweep_values", []))
        points = expand_ofat(base_params=base_params, sweep_param=sweep_param, sweep_values=sweep_values)
        param_names = [sweep_param] if sweep_param else []
        return points, mode, sweep_param, param_names

    if mode == "grid":
        grid_params = dict(cfg["experiment"]["grid"].get("params", {}))
        points = expand_grid(grid_params=grid_params)
        return points, mode, "", sorted(grid_params.keys())

    raise ValueError(f"不支持的实验模式: {mode}")


def run_single_experiment(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    cfg_obj = OmegaConf.load(str(config_path))
    cfg = OmegaConf.to_container(cfg_obj, resolve=True)
    assert isinstance(cfg, dict)
    cfg = _apply_overrides(cfg, overrides)

    algorithm_id = str(cfg["algorithm"]["id"])
    algorithm_module = str(cfg["algorithm"]["module"])
    solver = _load_solver(algorithm_module)

    run_prefix = str(cfg["output"].get("run_id_prefix", "exp"))
    run_id = timestamp_id(run_prefix)
    output_root = Path(cfg["output"].get("root", "outputs/experiments"))
    run_dir = ensure_dir(output_root / run_id)
    results_dir = ensure_dir(run_dir / "results")
    figures_dir = ensure_dir(run_dir / "figures")

    dataset_root = Path(cfg["dataset"]["root"])
    class_filter = cfg["dataset"].get("class_filter", [])
    file_prefix = str(cfg["dataset"].get("file_prefix", "sc_"))

    instances = read_all_instances(dataset_root=dataset_root, class_filter=class_filter, file_prefix=file_prefix)
    if not instances:
        raise ValueError(f"未找到可用实例: dataset_root={dataset_root}")

    repeats = int(cfg["experiment"].get("repeats", 5))
    seeds = [int(x) for x in cfg["experiment"].get("seeds", list(range(repeats)))]
    if repeats < 1:
        raise ValueError("repeats 必须 >= 1")
    if repeats > len(seeds):
        seeds = seeds + [seeds[-1] + i + 1 for i in range(repeats - len(seeds))]

    param_points, mode, sweep_param, grid_params = _build_param_points(cfg)
    all_param_keys = sorted({k for params in param_points for k in params.keys()})

    rows: list[dict[str, Any]] = []

    for point_idx, params in enumerate(param_points):
        signature = param_signature(params)
        point_param_key = sweep_param if mode == "ofat" else ""
        point_param_value = params.get(sweep_param, "") if mode == "ofat" else ""

        for repeat_idx in range(repeats):
            seed = int(seeds[repeat_idx])
            for inst in instances:
                start = time.perf_counter()
                raw = solver(inst, seed=seed, **params)
                elapsed = time.perf_counter() - start
                result = normalize_result(raw, elapsed)
                meta = result.get("meta", {})
                if not isinstance(meta, dict):
                    meta = {"raw_meta": meta}

                row = {
                    "run_id": run_id,
                    "algorithm_id": algorithm_id,
                    "algorithm_module": algorithm_module,
                    "dataset_id": inst.dataset_id,
                    "class_id": inst.class_id,
                    "sample_id": inst.sample_id,
                    "instance_id": inst.instance_id,
                    "source_path": inst.path,
                    "set_count": inst.n_sets,
                    "item_count": inst.n_items,
                    "density": inst.density,
                    "pattern": inst.pattern,
                    "seed": seed,
                    "repeat_idx": repeat_idx,
                    "param_point_idx": point_idx,
                    "param_signature": signature,
                    "param_key": point_param_key,
                    "param_value": point_param_value,
                    "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "runtime_sec": float(result["runtime_sec"]),
                    "objective": float(result["objective"]),
                    "feasible": bool(result["is_feasible"]),
                    "selected_set_count": int(len(result["selected_sets"])),
                    "selected_sets_json": json.dumps([int(x) for x in result["selected_sets"]], ensure_ascii=False),
                    "convergence_speed": convergence_speed(result["convergence_curve"]),
                    "convergence_curve_json": json.dumps(result["convergence_curve"], ensure_ascii=False),
                    "meta_json": json.dumps(meta, ensure_ascii=False, sort_keys=True),
                    "pareto_size": meta.get("pareto_size"),
                    "front_cost_span": meta.get("front_cost_span"),
                    "front_hv": meta.get("front_hv"),
                    "front_hv_norm": meta.get("front_hv_norm"),
                    "front_feasible_ratio": meta.get("feasible_ratio"),
                    "front_points_json": json.dumps(meta.get("pareto_front_points", []), ensure_ascii=False),
                }
                for key in all_param_keys:
                    row[f"param__{key}"] = params.get(key)
                rows.append(row)

    runs_df = pd.DataFrame(rows)
    runs_df = add_gap_to_best(runs_df)
    runs_df = add_stability_var(runs_df)

    summary_param_df = summarize_by_param(runs_df)
    summary_class_df = summarize_by_class(runs_df)
    fit_df = complexity_fit(runs_df)

    significance_method = str(cfg["metrics"].get("significance_method", "wilcoxon"))
    significance_metric = str(cfg["metrics"].get("significance_metric", "objective"))
    significance_df = paired_significance(
        runs_df=runs_df,
        metric=significance_metric,
        method=significance_method,
    )

    runs_df.to_csv(results_dir / "runs.csv", index=False)
    summary_param_df.to_csv(results_dir / "summary_by_param.csv", index=False)
    summary_class_df.to_csv(results_dir / "summary_by_class.csv", index=False)
    fit_df.to_csv(results_dir / "complexity_fit.csv", index=False)
    significance_df.to_csv(results_dir / "significance.csv", index=False)

    meta_rows = [
        {"key": "run_id", "value": run_id},
        {"key": "algorithm_id", "value": algorithm_id},
        {"key": "algorithm_module", "value": algorithm_module},
        {"key": "mode", "value": mode},
        {"key": "sweep_param", "value": sweep_param},
        {"key": "grid_params", "value": ",".join(grid_params)},
        {"key": "dataset_root", "value": str(dataset_root)},
        {"key": "repeats", "value": repeats},
    ]
    pd.DataFrame(meta_rows).to_csv(results_dir / "run_meta.csv", index=False)

    pd.DataFrame(
        [
            {
                "key": "algorithm_id",
                "value": algorithm_id,
            },
            {
                "key": "algorithm_module",
                "value": algorithm_module,
            },
            {
                "key": "dataset_root",
                "value": str(dataset_root),
            },
            {
                "key": "mode",
                "value": mode,
            },
            {
                "key": "repeats",
                "value": repeats,
            },
        ]
    ).to_csv(results_dir / "resolved_config.csv", index=False)

    if bool(cfg["output"].get("generate_plots", True)):
        from core.visualize import generate_plots

        generate_plots(
            runs_df=runs_df,
            summary_param_df=summary_param_df,
            summary_class_df=summary_class_df,
            fit_df=fit_df,
            out_dir=figures_dir,
            mode=mode,
            sweep_param=sweep_param,
            grid_params=grid_params,
        )

    return run_dir
