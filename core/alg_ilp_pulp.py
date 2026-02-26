from __future__ import annotations

import time
from typing import Any


def solve(instance, seed: int, time_limit_sec: int = 60, msg: int = 0, **kwargs: Any) -> dict[str, Any]:
    """ILP solver using PuLP + CBC."""

    try:
        import pulp as pl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'pulp'. Install it to run ILP algorithm.") from exc

    start = time.perf_counter()

    n_items = instance.n_items
    n_sets = instance.n_sets
    costs = [float(rec.cost) for rec in instance.sets]

    item_to_sets: list[list[int]] = [[] for _ in range(n_items)]
    for j, rec in enumerate(instance.sets):
        for item in rec.items:
            item_to_sets[item].append(j)

    for i in range(n_items):
        if not item_to_sets[i]:
            runtime_sec = time.perf_counter() - start
            return {
                "objective": float("inf"),
                "runtime_sec": float(runtime_sec),
                "is_feasible": False,
                "selected_sets": [],
                "convergence_curve": [],
                "meta": {
                    "solver_status": "infeasible_input",
                    "reason": f"item_{i}_has_no_covering_set",
                },
            }

    model = pl.LpProblem("SetCover", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", range(n_sets), lowBound=0, upBound=1, cat=pl.LpBinary)

    model += pl.lpSum(costs[j] * x[j] for j in range(n_sets))
    for i in range(n_items):
        model += pl.lpSum(x[j] for j in item_to_sets[i]) >= 1, f"cover_{i}"

    solver = pl.PULP_CBC_CMD(timeLimit=int(time_limit_sec), msg=int(msg))
    status_code = model.solve(solver)
    solver_status = pl.LpStatus.get(status_code, str(status_code))

    selected = [j for j in range(n_sets) if (x[j].value() is not None and x[j].value() > 0.5)]

    covered = set()
    for j in selected:
        covered.update(instance.sets[j].items)
    is_feasible = len(covered) == n_items

    objective = sum(costs[j] for j in selected) if is_feasible else float("inf")
    runtime_sec = time.perf_counter() - start

    return {
        "objective": float(objective),
        "runtime_sec": float(runtime_sec),
        "is_feasible": bool(is_feasible),
        "selected_sets": selected,
        "convergence_curve": [],
        "meta": {
            "solver": "pulp_cbc",
            "solver_status": solver_status,
            "time_limit_sec": int(time_limit_sec),
            "msg": int(msg),
            "n_vars": int(n_sets),
            "n_constraints": int(n_items),
        },
    }
