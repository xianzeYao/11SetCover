from __future__ import annotations

import time
from typing import Any


def solve(
    instance,
    seed: int,
    time_limit_sec: int = 60,
    msg: int = 0,
    num_search_workers: int = 8,
    **kwargs: Any,
) -> dict[str, Any]:
    """ILP-like baseline via OR-Tools CP-SAT."""

    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'ortools'. Install it to run OR-Tools algorithm.") from exc

    start = time.perf_counter()

    n_items = instance.n_items
    n_sets = instance.n_sets
    costs = [int(round(float(rec.cost))) for rec in instance.sets]

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
                    "solver": "ortools_cp_sat",
                    "solver_status": "infeasible_input",
                    "reason": f"item_{i}_has_no_covering_set",
                },
            }

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{j}") for j in range(n_sets)]

    for i in range(n_items):
        model.Add(sum(x[j] for j in item_to_sets[i]) >= 1)
    model.Minimize(sum(costs[j] * x[j] for j in range(n_sets)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.log_search_progress = bool(msg)
    solver.parameters.random_seed = int(seed)
    workers = int(num_search_workers)
    if workers > 0:
        solver.parameters.num_search_workers = workers

    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code)

    feasible_status = {cp_model.OPTIMAL, cp_model.FEASIBLE}
    if status_code in feasible_status:
        selected = [j for j in range(n_sets) if solver.BooleanValue(x[j])]
        covered = set()
        for j in selected:
            covered.update(instance.sets[j].items)
        is_feasible = len(covered) == n_items
        objective = sum(float(instance.sets[j].cost)
                        for j in selected) if is_feasible else float("inf")
    else:
        selected = []
        is_feasible = False
        objective = float("inf")

    runtime_sec = time.perf_counter() - start

    best_bound = float("nan")
    objective_value = float("nan")
    try:
        best_bound = float(solver.BestObjectiveBound())
    except Exception:
        pass
    try:
        objective_value = float(solver.ObjectiveValue())
    except Exception:
        pass

    return {
        "objective": float(objective),
        "runtime_sec": float(runtime_sec),
        "is_feasible": bool(is_feasible),
        "selected_sets": selected,
        "convergence_curve": [],
        "meta": {
            "solver": "ortools_cp_sat",
            "solver_status": str(status_name).lower(),
            "time_limit_sec": int(time_limit_sec),
            "msg": int(msg),
            "n_vars": int(n_sets),
            "n_constraints": int(n_items),
            "num_search_workers": int(max(0, workers)),
            "objective_value_raw": objective_value,
            "best_objective_bound_raw": best_bound,
        },
    }
