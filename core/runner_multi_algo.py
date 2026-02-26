from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from core.compare_multi_algo import compare_experiments
from core.runner_single_algo import run_single_experiment
from core.utils import ensure_dir, timestamp_id


def _safe_name(text: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_")
    return name or "algo"


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


def run_multi_experiment(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    cfg_obj = OmegaConf.load(str(config_path))
    cfg = OmegaConf.to_container(cfg_obj, resolve=True)
    assert isinstance(cfg, dict)
    cfg = _apply_overrides(cfg, overrides)

    algorithms = list(cfg.get("algorithms", []))
    if not algorithms:
        raise ValueError("algorithms 不能为空")
    algorithm_filter_raw = cfg.get("experiment", {}).get("algorithm_filter", [])
    if isinstance(algorithm_filter_raw, str):
        algorithm_filter_values = [token.strip() for token in algorithm_filter_raw.split(",")]
    else:
        algorithm_filter_values = list(algorithm_filter_raw)
    algorithm_filter = [str(x).strip() for x in algorithm_filter_values if str(x).strip()]
    if algorithm_filter:
        configured_ids = [str(alg.get("id", "")).strip() for alg in algorithms if isinstance(alg, dict)]
        missing = [algorithm_id for algorithm_id in algorithm_filter if algorithm_id not in configured_ids]
        if missing:
            raise ValueError(
                f"algorithm_filter 包含未配置的 algorithm_id: {missing}; 可选={sorted(set(configured_ids))}"
            )
        filter_set = set(algorithm_filter)
        algorithms = [alg for alg in algorithms if isinstance(alg, dict) and str(alg.get("id", "")).strip() in filter_set]

    dataset_root = str(cfg["dataset"]["root"])
    class_filter = list(cfg["dataset"].get("class_filter", []))
    file_prefix = str(cfg["dataset"].get("file_prefix", "sc_"))

    repeats = int(cfg["experiment"].get("repeats", 5))
    seeds = [int(x) for x in cfg["experiment"].get("seeds", list(range(repeats)))]
    if repeats < 1:
        raise ValueError("repeats 必须 >= 1")
    if repeats > len(seeds):
        seeds = seeds + [seeds[-1] + i + 1 for i in range(repeats - len(seeds))]

    output_cfg = dict(cfg.get("output", {}))
    output_root = Path(output_cfg.get("root", "outputs/experiments"))
    single_plots = bool(output_cfg.get("generate_single_plots", True))
    compare_plots = bool(output_cfg.get("generate_compare_plots", True))

    batch_dir = ensure_dir(output_root / timestamp_id("batch"))
    runs_root = ensure_dir(batch_dir / "runs")
    compare_root = ensure_dir(batch_dir / "compare")

    single_config_path = str(cfg.get("single_config", "configs/experiment_single.yaml"))

    run_dirs: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for alg in algorithms:
        if not isinstance(alg, dict):
            raise ValueError(f"algorithm 定义必须是对象: {alg}")
        algorithm_id = str(alg.get("id", "")).strip()
        algorithm_module = str(alg.get("module", "")).strip()
        params = dict(alg.get("params", {}))
        if not algorithm_id or not algorithm_module:
            raise ValueError(f"algorithm 需要 id/module: {alg}")
        safe_algorithm_id = _safe_name(algorithm_id)
        algorithm_output_root = ensure_dir(runs_root / safe_algorithm_id)

        alg_repeats = int(alg.get("repeats", repeats))
        if alg_repeats < 1:
            raise ValueError(f"{algorithm_id}: repeats 必须 >= 1")

        if alg.get("seeds", None) is not None:
            alg_seeds = [int(x) for x in list(alg.get("seeds", []))]
        else:
            alg_seeds = list(seeds)

        if not alg_seeds:
            alg_seeds = list(range(alg_repeats))
        if alg_repeats > len(alg_seeds):
            alg_seeds = alg_seeds + [alg_seeds[-1] + i + 1 for i in range(alg_repeats - len(alg_seeds))]
        elif alg_repeats < len(alg_seeds):
            alg_seeds = alg_seeds[:alg_repeats]

        single_overrides = {
            "algorithm.id": algorithm_id,
            "algorithm.module": algorithm_module,
            "dataset.root": dataset_root,
            "dataset.class_filter": class_filter,
            "dataset.file_prefix": file_prefix,
            "experiment.mode": "ofat",
            "experiment.repeats": alg_repeats,
            "experiment.seeds": alg_seeds,
            "experiment.ofat.base_params": params,
            "experiment.ofat.sweep_param": "",
            "experiment.ofat.sweep_values": [],
            "experiment.grid.params": {},
            "output.root": str(algorithm_output_root),
            "output.run_id_prefix": f"exp_{safe_algorithm_id}",
            "output.generate_plots": single_plots,
        }

        run_dir = run_single_experiment(config_path=single_config_path, overrides=single_overrides)
        run_dirs.append(run_dir)
        manifest_rows.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_module": algorithm_module,
                "params": str(params),
                "repeats": alg_repeats,
                "seeds": str(alg_seeds),
                "run_dir": str(run_dir),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(batch_dir / "manifest.csv", index=False)

    compare_cfg = dict(cfg.get("compare", {}))
    compare_axis_map = dict(compare_cfg.get("class_axis_map", {}))
    significance_method = str(compare_cfg.get("significance_method", "wilcoxon"))
    significance_metrics = list(compare_cfg.get("significance_metrics", ["objective"]))

    compare_experiments(
        run_dirs=run_dirs,
        output_dir=compare_root,
        class_axis_map=compare_axis_map,
        with_plots=compare_plots,
        significance_method=significance_method,
        significance_metrics=significance_metrics,
    )

    return batch_dir
