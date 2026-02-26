from __future__ import annotations

import itertools
import json
from typing import Any, Mapping


def parse_scalar(text: str) -> Any:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_params_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("参数 JSON 必须是对象，例如: '{\"alpha\":1.0}'")
    return parsed


def parse_sweep_values(text: str) -> list[Any]:
    if not text:
        return []
    return [parse_scalar(x) for x in text.split(",") if x.strip()]


def parse_grid_spec(text: str) -> dict[str, list[Any]]:
    """解析形如 alpha=0.1,0.2;beta=1,2 的字符串。"""

    spec: dict[str, list[Any]] = {}
    if not text:
        return spec

    for block in text.split(";"):
        block = block.strip()
        if not block:
            continue
        if "=" not in block:
            raise ValueError(f"网格参数片段缺少 '=': {block}")
        key, values_text = block.split("=", 1)
        key = key.strip()
        values = [parse_scalar(x) for x in values_text.split(",") if x.strip()]
        if not values:
            raise ValueError(f"网格参数 {key} 没有可用取值")
        spec[key] = values

    return spec


def expand_ofat(
    base_params: Mapping[str, Any],
    sweep_param: str,
    sweep_values: list[Any],
) -> list[dict[str, Any]]:
    if not sweep_param or not sweep_values:
        return [dict(base_params)]

    points: list[dict[str, Any]] = []
    for value in sweep_values:
        params = dict(base_params)
        params[sweep_param] = value
        points.append(params)
    return points


def expand_grid(grid_params: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid_params:
        return [{}]
    keys = sorted(grid_params.keys())
    values_product = itertools.product(*(grid_params[k] for k in keys))
    return [dict(zip(keys, combo, strict=True)) for combo in values_product]


def param_signature(params: Mapping[str, Any]) -> str:
    if not params:
        return "default"
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(parts)
