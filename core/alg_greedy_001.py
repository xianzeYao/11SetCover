from __future__ import annotations

import random
import time
from typing import Any


def solve(instance, seed: int, epsilon: float = 1e-12, **kwargs: Any) -> dict[str, Any]:
    """Greedy_001: maximize new_covered / cost at each step."""

    start = time.perf_counter()
    rng = random.Random(seed)

    uncovered = set(range(instance.n_items))
    set_items = [set(rec.items) for rec in instance.sets]
    available = set(range(instance.n_sets))

    selected: list[int] = []
    newly_covered_trace: list[int] = []

    while uncovered and available:
        best_score = float("-inf")
        candidates: list[int] = []

        for idx in available:
            rec = instance.sets[idx]
            new_cover = len(uncovered.intersection(set_items[idx]))
            if new_cover <= 0:
                continue

            score = float("inf") if rec.cost <= 0 else (new_cover / rec.cost)
            if score > best_score + epsilon:
                best_score = score
                candidates = [idx]
            elif abs(score - best_score) <= epsilon:
                candidates.append(idx)

        if not candidates:
            break

        chosen = rng.choice(candidates)
        available.remove(chosen)
        selected.append(chosen)

        newly_covered = uncovered.intersection(set_items[chosen])
        newly_covered_trace.append(len(newly_covered))
        uncovered.difference_update(set_items[chosen])

    is_feasible = len(uncovered) == 0
    objective = sum(float(instance.sets[i].cost) for i in selected)
    runtime_sec = time.perf_counter() - start

    if not is_feasible:
        objective = float("inf")

    return {
        "objective": float(objective),
        "runtime_sec": float(runtime_sec),
        "is_feasible": bool(is_feasible),
        "selected_sets": selected,
        "convergence_curve": [],
        "meta": {
            "iterations": len(selected),
            "uncovered_count": len(uncovered),
            "newly_covered_trace": newly_covered_trace,
        },
    }
