from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from core.io_dataset import write_instance_file
from core.utils import ensure_dir


@dataclass(frozen=True)
class ClassConfig:
    class_id: str
    set_range: tuple[int, int]
    item_range: tuple[int, int] | None
    density_range: tuple[float, float]
    pattern: str
    samples: int
    cost_mode: str
    cluster_bias: float
    hub_bias: float
    hub_ratio: float
    cluster_count_factor: float


def _as_class_config(raw: dict[str, Any], default_samples: int, default_cost_mode: str) -> ClassConfig:
    item_raw = raw.get("item_range")
    item_range: tuple[int, int] | None = None
    if item_raw is not None:
        item_range = (int(item_raw[0]), int(item_raw[1]))

    return ClassConfig(
        class_id=str(raw["class_id"]),
        set_range=(int(raw["set_range"][0]), int(raw["set_range"][1])),
        item_range=item_range,
        density_range=(float(raw["density_range"][0]),
                       float(raw["density_range"][1])),
        pattern=str(raw.get("pattern", "random")),
        samples=int(raw.get("samples", default_samples)),
        cost_mode=str(raw.get("cost_mode", default_cost_mode)),
        cluster_bias=float(raw.get("cluster_bias", 0.82)),
        hub_bias=float(raw.get("hub_bias", 0.78)),
        hub_ratio=float(raw.get("hub_ratio", 0.12)),
        cluster_count_factor=float(raw.get("cluster_count_factor", 0.5)),
    )


def _clusters_for_items(n_items: int, n_clusters: int) -> list[list[int]]:
    n_clusters = max(2, min(n_clusters, n_items))
    clusters = [[] for _ in range(n_clusters)]
    for item in range(n_items):
        clusters[item % n_clusters].append(item)
    return clusters


def _sample_item(
    rng: random.Random,
    pattern: str,
    clusters: list[list[int]],
    set_cluster_idx: int,
    n_items: int,
    hub_items: list[int],
    cluster_bias: float,
    hub_bias: float,
) -> int:
    cluster_bias = min(max(cluster_bias, 0.0), 1.0)
    hub_bias = min(max(hub_bias, 0.0), 1.0)
    if pattern == "clustered" and clusters:
        if rng.random() < cluster_bias:
            return rng.choice(clusters[set_cluster_idx])
        return rng.randrange(n_items)

    if pattern == "hub" and hub_items:
        if rng.random() < hub_bias:
            return rng.choice(hub_items)
        return rng.randrange(n_items)

    return rng.randrange(n_items)


def _build_set_items(
    n_items: int,
    n_sets: int,
    target_density: float,
    pattern: str,
    rng: random.Random,
    cluster_bias: float = 0.82,
    hub_bias: float = 0.78,
    hub_ratio: float = 0.12,
    cluster_count_factor: float = 0.5,
) -> list[list[int]]:
    total_pairs = n_items * n_sets
    target_nonzeros = int(round(total_pairs * target_density))
    target_nonzeros = max(max(n_items, n_sets),
                          min(total_pairs, target_nonzeros))

    set_items = [set() for _ in range(n_sets)]

    cluster_count_factor = max(0.05, float(cluster_count_factor))
    cluster_count = max(
        2, int(round(math.sqrt(max(4, n_items)) * cluster_count_factor)))
    clusters = _clusters_for_items(
        n_items, cluster_count) if pattern == "clustered" else []
    set_cluster = [rng.randrange(len(clusters))
                   if clusters else 0 for _ in range(n_sets)]
    hub_items: list[int] = []
    if pattern == "hub":
        hub_ratio = min(max(float(hub_ratio), 0.01), 1.0)
        hub_count = max(1, min(n_items, int(round(n_items * hub_ratio))))
        hub_items = rng.sample(list(range(n_items)), k=hub_count)

    for s in range(n_sets):
        item = _sample_item(
            rng, pattern, clusters, set_cluster[s], n_items, hub_items,
            cluster_bias=cluster_bias, hub_bias=hub_bias,
        )
        set_items[s].add(item)

    for item in range(n_items):
        assigned = any(item in row for row in set_items)
        if assigned:
            continue
        if pattern == "clustered" and clusters:
            cluster_idx = next(i for i, c in enumerate(clusters) if item in c)
            candidates = [s for s in range(
                n_sets) if set_cluster[s] == cluster_idx]
            s = rng.choice(candidates) if candidates else rng.randrange(n_sets)
        else:
            s = rng.randrange(n_sets)
        set_items[s].add(item)

    used = sum(len(x) for x in set_items)
    if used > target_nonzeros:
        target_nonzeros = used

    attempts = 0
    max_attempts = max(10_000, target_nonzeros * 40)
    while used < target_nonzeros and attempts < max_attempts:
        s = rng.randrange(n_sets)
        item = _sample_item(
            rng, pattern, clusters, set_cluster[s], n_items, hub_items,
            cluster_bias=cluster_bias, hub_bias=hub_bias,
        )
        if item not in set_items[s]:
            set_items[s].add(item)
            used += 1
        attempts += 1

    if used < target_nonzeros:
        for s in range(n_sets):
            if used >= target_nonzeros:
                break
            for item in range(n_items):
                if item in set_items[s]:
                    continue
                set_items[s].add(item)
                used += 1
                if used >= target_nonzeros:
                    break

    return [sorted(row) for row in set_items]


def _build_costs(
    set_items: list[list[int]],
    n_items: int,
    cost_mode: str,
    cost_range: tuple[float, float],
    rng: random.Random,
) -> list[int]:
    lo, hi = cost_range
    lo_i = max(1, int(round(lo)))
    hi_i = max(lo_i, int(round(hi)))
    costs: list[int] = []

    for items in set_items:
        if cost_mode == "skewed":
            coverage_ratio = len(items) / max(1, n_items)
            base = lo + (hi - lo) * (0.2 + coverage_ratio * 0.8)
            noise = 0.85 + 0.3 * rng.random()
            value = int(round(base * noise))
            value = min(max(value, lo_i), hi_i)
            costs.append(value)
        else:
            costs.append(rng.randint(lo_i, hi_i))

    return costs


def _build_int_schedule(low: int, high: int, count: int, rng: random.Random) -> list[int]:
    if count <= 0:
        return []
    if low >= high:
        return [int(low)] * count

    if count == 1:
        return [int(low)]

    step = (high - low) / float(count - 1)
    values = [int(round(low + step * i)) for i in range(count)]
    # enforce monotonic non-decreasing and clamp
    values = [min(max(v, low), high) for v in values]
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]
    return values


def _build_float_schedule(low: float, high: float, count: int, rng: random.Random) -> list[float]:
    if count <= 0:
        return []
    if low >= high:
        return [float(low)] * count

    if count == 1:
        return [float(low)]

    step = (high - low) / float(count - 1)
    return [float(low + step * i) for i in range(count)]


def generate_dataset(
    config_path: str | Path,
    output_root: str | Path | None = None,
    dataset_id: str | None = None,
    samples_per_class_override: int | None = None,
    seed_override: int | None = None,
) -> Path:
    cfg = OmegaConf.to_container(
        OmegaConf.load(str(config_path)), resolve=True)
    assert isinstance(cfg, dict)

    seed = int(
        seed_override if seed_override is not None else cfg.get("seed", 2026))
    rng = random.Random(seed)

    target_dataset_id = str(
        dataset_id if dataset_id is not None else cfg.get("dataset_id", "default"))
    base_output = Path(output_root if output_root is not None else cfg.get(
        "output_root", "outputs/datasets"))
    dataset_dir = ensure_dir(base_output / target_dataset_id)

    if bool(cfg.get("clean_output", True)) and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

    item_ratio_range = tuple(float(x)
                             for x in cfg.get("item_ratio_range", [0.4, 0.8]))
    cost_range = tuple(float(x) for x in cfg.get("cost_range", [1.0, 10.0]))
    default_samples = int(samples_per_class_override if samples_per_class_override is not None else cfg.get(
        "samples_per_class", 20))
    default_cost_mode = str(cfg.get("cost_mode", "uniform"))

    class_rows = cfg.get("classes", [])
    classes = [_as_class_config(
        row, default_samples, default_cost_mode) for row in class_rows]

    index_rows: list[dict[str, Any]] = []
    for class_cfg in classes:
        class_dir = ensure_dir(dataset_dir / class_cfg.class_id)
        set_schedule = _build_int_schedule(
            low=class_cfg.set_range[0],
            high=class_cfg.set_range[1],
            count=class_cfg.samples,
            rng=rng,
        )
        density_schedule = _build_float_schedule(
            low=class_cfg.density_range[0],
            high=class_cfg.density_range[1],
            count=class_cfg.samples,
            rng=rng,
        )
        item_schedule: list[int] | None = None
        if class_cfg.item_range is not None:
            item_schedule = _build_int_schedule(
                low=class_cfg.item_range[0],
                high=class_cfg.item_range[1],
                count=class_cfg.samples,
                rng=rng,
            )

        for idx in range(class_cfg.samples):
            sample_id = f"{idx:03d}"
            n_sets = set_schedule[idx]
            if item_schedule is not None:
                n_items = max(2, int(item_schedule[idx]))
            else:
                ratio = rng.uniform(item_ratio_range[0], item_ratio_range[1])
                n_items = max(2, int(round(n_sets * ratio)))

            target_density = density_schedule[idx]
            set_items = _build_set_items(
                n_items=n_items,
                n_sets=n_sets,
                target_density=target_density,
                pattern=class_cfg.pattern,
                rng=rng,
                cluster_bias=class_cfg.cluster_bias,
                hub_bias=class_cfg.hub_bias,
                hub_ratio=class_cfg.hub_ratio,
                cluster_count_factor=class_cfg.cluster_count_factor,
            )

            nonzeros = sum(len(row) for row in set_items)
            density = nonzeros / \
                float(n_items * n_sets) if n_items > 0 and n_sets > 0 else 0.0

            costs = _build_costs(
                set_items=set_items,
                n_items=n_items,
                cost_mode=class_cfg.cost_mode,
                cost_range=cost_range,
                rng=rng,
            )

            file_name = f"sc_{class_cfg.class_id}_{sample_id}"
            file_path = class_dir / file_name
            write_instance_file(
                path=file_path,
                class_id=class_cfg.class_id,
                sample_id=sample_id,
                n_items=n_items,
                n_sets=n_sets,
                density=density,
                pattern=class_cfg.pattern,
                seed=seed,
                costs=costs,
                set_items=set_items,
            )

            index_rows.append(
                {
                    "dataset_id": target_dataset_id,
                    "class_id": class_cfg.class_id,
                    "sample_id": sample_id,
                    "path": str(file_path),
                    "n_items": n_items,
                    "n_sets": n_sets,
                    "density": density,
                    "pattern": class_cfg.pattern,
                    "seed": seed,
                }
            )

    pd.DataFrame(index_rows).to_csv(
        dataset_dir / "dataset_index.csv", index=False)
    return dataset_dir
