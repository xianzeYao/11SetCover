from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np


def _hypervolume_2d_min(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """2D hypervolume for minimization with rectangular decomposition."""
    if not points:
        return float("nan")

    ref_x, ref_y = float(reference[0]), float(reference[1])
    # Sort by f1 ascending; keep best f2 seen so far to avoid overlap.
    sorted_points = sorted((float(x), float(y))
                           for x, y in points if np.isfinite(x) and np.isfinite(y))
    hv = 0.0
    current_y = ref_y

    for x, y in sorted_points:
        width = ref_x - x
        height = current_y - y
        if width > 0 and height > 0:
            hv += width * height
            current_y = min(current_y, y)
    return float(hv)


def _relative_improved(previous_best: float, current_best: float, threshold: float) -> bool:
    if not np.isfinite(current_best):
        return False
    if not np.isfinite(previous_best):
        return True
    if current_best >= previous_best:
        return False
    denom = max(abs(previous_best), 1e-12)
    rel = (previous_best - current_best) / denom
    return bool(rel >= threshold)


def _build_incidence(instance):
    n_items = instance.n_items
    n_sets = instance.n_sets
    a = np.zeros((n_items, n_sets), dtype=np.int16)
    for j, rec in enumerate(instance.sets):
        for item in rec.items:
            a[item, j] = 1
    costs = np.array([float(rec.cost) for rec in instance.sets], dtype=float)
    item_to_sets = [np.flatnonzero(a[i]).tolist() for i in range(n_items)]
    return a, costs, item_to_sets


def _greedy_seed(instance) -> np.ndarray:
    uncovered = set(range(instance.n_items))
    selected = np.zeros(instance.n_sets, dtype=np.int8)
    set_items = [set(rec.items) for rec in instance.sets]

    while uncovered:
        best = -1.0
        best_idx = -1
        for j, rec in enumerate(instance.sets):
            if selected[j] == 1:
                continue
            gain = len(uncovered.intersection(set_items[j]))
            if gain <= 0:
                continue
            score = float("inf") if rec.cost <= 0 else gain / rec.cost
            if score > best:
                best = score
                best_idx = j
        if best_idx < 0:
            break
        selected[best_idx] = 1
        uncovered.difference_update(set_items[best_idx])

    return selected


def _repair(ind: np.ndarray, a: np.ndarray, costs: np.ndarray, item_to_sets: list[list[int]], rng: random.Random) -> np.ndarray:
    repaired = ind.copy()
    n_items = a.shape[0]
    coverage = a @ repaired
    for i in range(n_items):
        if coverage[i] > 0:
            continue
        candidates = item_to_sets[i]
        if not candidates:
            continue
        min_cost = min(costs[j] for j in candidates)
        ties = [j for j in candidates if costs[j] == min_cost]
        chosen = rng.choice(ties)
        repaired[chosen] = 1
        coverage += a[:, chosen]
    return repaired


def _prune(ind: np.ndarray, a: np.ndarray, costs: np.ndarray) -> np.ndarray:
    pruned = ind.copy().astype(np.int8)
    if pruned.sum() <= 1:
        return pruned

    coverage = a @ pruned
    selected = np.flatnonzero(pruned)
    remove_order = sorted(
        (int(idx) for idx in selected),
        key=lambda idx: (float(costs[idx]), int(np.sum(a[:, idx]))),
        reverse=True,
    )

    for idx in remove_order:
        if pruned[idx] == 0:
            continue
        covered_items = np.flatnonzero(a[:, idx] > 0)
        if covered_items.size == 0:
            pruned[idx] = 0
            continue
        if np.all(coverage[covered_items] >= 2):
            pruned[idx] = 0
            coverage -= a[:, idx]

    if pruned.sum() == 0 and pruned.size > 0:
        pruned[int(np.argmin(costs))] = 1
    return pruned


def _dedup_population(population: np.ndarray) -> np.ndarray:
    if population.size == 0:
        return population.astype(np.int8)
    return np.unique(population.astype(np.int8), axis=0).astype(np.int8)


def _evaluate_population(pop: np.ndarray, a: np.ndarray, costs: np.ndarray):
    coverage = a @ pop.T  # (n_items, pop_size)
    uncovered = (coverage == 0).sum(axis=0).astype(int)
    vulnerability = (coverage == 1).sum(axis=0).astype(float)
    feasible = uncovered == 0
    obj_cost = (pop * costs.reshape(1, -1)).sum(axis=1)
    return obj_cost.astype(float), vulnerability, uncovered, feasible


def _dominates(i: int, j: int, obj_cost, obj_vuln, uncovered, feasible) -> bool:
    fi, fj = bool(feasible[i]), bool(feasible[j])

    if fi and not fj:
        return True
    if not fi and fj:
        return False

    if fi and fj:
        no_worse = obj_cost[i] <= obj_cost[j] and obj_vuln[i] <= obj_vuln[j]
        strictly_better = obj_cost[i] < obj_cost[j] or obj_vuln[i] < obj_vuln[j]
        return bool(no_worse and strictly_better)

    if uncovered[i] < uncovered[j]:
        return True
    if uncovered[i] > uncovered[j]:
        return False

    no_worse = obj_cost[i] <= obj_cost[j] and obj_vuln[i] <= obj_vuln[j]
    strictly_better = obj_cost[i] < obj_cost[j] or obj_vuln[i] < obj_vuln[j]
    return bool(no_worse and strictly_better)


def _fast_non_dominated_sort(obj_cost, obj_vuln, uncovered, feasible):
    n = len(obj_cost)
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    dominated_count = [0] * n
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(i, j, obj_cost, obj_vuln, uncovered, feasible):
                dominates_list[i].append(j)
            elif _dominates(j, i, obj_cost, obj_vuln, uncovered, feasible):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    rank = np.full(n, fill_value=np.inf)
    rank[np.array(fronts[0], dtype=int)] = 0

    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    rank[j] = current + 1
                    next_front.append(j)
        current += 1
        if next_front:
            fronts.append(next_front)

    return fronts, rank.astype(int)


def _crowding_distance(front: list[int], obj_cost, obj_vuln) -> dict[int, float]:
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    for values in (obj_cost, obj_vuln):
        sorted_idx = sorted(front, key=lambda i: values[i])
        v_min = values[sorted_idx[0]]
        v_max = values[sorted_idx[-1]]

        distance[sorted_idx[0]] = float("inf")
        distance[sorted_idx[-1]] = float("inf")

        denom = v_max - v_min
        if denom == 0:
            continue

        for k in range(1, len(sorted_idx) - 1):
            i_prev = sorted_idx[k - 1]
            i_next = sorted_idx[k + 1]
            i_cur = sorted_idx[k]
            if math.isinf(distance[i_cur]):
                continue
            distance[i_cur] += (values[i_next] - values[i_prev]) / denom

    return distance


def _tournament(
    candidates: list[int],
    rank: np.ndarray,
    crowding: dict[int, float],
    uncovered: np.ndarray,
    obj_cost: np.ndarray,
    rng: random.Random,
) -> int:
    a, b = rng.sample(candidates, 2)
    if rank[a] < rank[b]:
        return a
    if rank[b] < rank[a]:
        return b

    ca = crowding.get(a, 0.0)
    cb = crowding.get(b, 0.0)
    if ca > cb:
        return a
    if cb > ca:
        return b

    if uncovered[a] < uncovered[b]:
        return a
    if uncovered[b] < uncovered[a]:
        return b

    if obj_cost[a] < obj_cost[b]:
        return a
    if obj_cost[b] < obj_cost[a]:
        return b

    return a


def _select_survivors(population: np.ndarray, keep_size: int, a: np.ndarray, costs: np.ndarray) -> np.ndarray:
    if population.shape[0] <= keep_size:
        return population.astype(np.int8)

    obj_cost, obj_vuln, uncovered, feasible = _evaluate_population(
        population, a, costs)
    fronts, _ = _fast_non_dominated_sort(
        obj_cost, obj_vuln, uncovered, feasible)

    next_indices: list[int] = []
    for front in fronts:
        if len(next_indices) + len(front) <= keep_size:
            next_indices.extend(front)
        else:
            dist = _crowding_distance(front, obj_cost, obj_vuln)
            remain = keep_size - len(next_indices)
            sorted_front = sorted(
                front, key=lambda i: dist.get(i, 0.0), reverse=True)
            next_indices.extend(sorted_front[:remain])
            break

    return population[np.array(next_indices, dtype=int)].astype(np.int8)


def solve(
    instance,
    seed: int,
    pop_size: int = 80,
    generations: int = 120,
    crossover_rate: float = 0.9,
    mutation_rate: float | None = None,
    elite_archive_size: int = 200,
    init_on_prob: float = 0.08,
    repair_probability: float = 0.5,
    init_strategy: str = "random",
    early_stop_patience: int = 20,
    early_stop_min_generations: int = 20,
    min_rel_improve: float = 1e-4,
    **kwargs: Any,
) -> dict[str, Any]:
    """NSGA-II style MOEA for Set Cover with two objectives.

    f1: total cost (min)
    f2: vulnerability = count of singly-covered elements (min)
    Constraint handling: feasible-first, then uncovered count.
    """

    start = time.perf_counter()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    a, costs, item_to_sets = _build_incidence(instance)
    n_items, n_sets = a.shape

    population_size = max(4, int(pop_size))
    n_gen = max(1, int(generations))
    mut_rate = (1.0 / max(1, n_sets)
                ) if mutation_rate is None else float(mutation_rate)
    archive_size = max(1, int(elite_archive_size))
    init_mode = str(init_strategy).strip().lower()
    if init_mode not in {"random", "hybrid", "greedy"}:
        raise ValueError(
            f"Unsupported init_strategy={init_strategy}, expected random|hybrid|greedy")

    greedy = _greedy_seed(instance)
    population = np.zeros((population_size, n_sets), dtype=np.int8)

    for i in range(population_size):
        if init_mode == "random":
            ind = (np_rng.random(n_sets) < init_on_prob).astype(np.int8)
        elif init_mode == "greedy":
            ind = greedy.copy()
            if i > 0:
                flips = rng.randint(1, 3)
                for _ in range(flips):
                    idx = rng.randrange(n_sets)
                    ind[idx] = 1 - ind[idx]
        else:
            if i == 0:
                ind = greedy.copy()
            elif i < max(2, population_size // 5):
                ind = greedy.copy()
                flips = rng.randint(1, 3)
                for _ in range(flips):
                    idx = rng.randrange(n_sets)
                    ind[idx] = 1 - ind[idx]
            else:
                ind = (np_rng.random(n_sets) < init_on_prob).astype(np.int8)

        if np.sum(ind) == 0 and n_sets > 0:
            ind[rng.randrange(n_sets)] = 1

        if rng.random() < float(repair_probability):
            ind = _repair(ind, a, costs, item_to_sets, rng)
        ind = _prune(ind, a, costs)

        population[i] = ind

    archive = _select_survivors(_dedup_population(
        population.copy()), archive_size, a, costs)

    convergence_curve: list[float] = []
    patience = max(0, int(early_stop_patience))
    min_gens = max(0, int(early_stop_min_generations))
    rel_threshold = max(0.0, float(min_rel_improve))
    best_metric = float("inf")
    no_improve_generations = 0
    early_stopped = False
    stop_reason = "max_generations"
    effective_generations = 0

    for gen in range(n_gen):
        obj_cost, obj_vuln, uncovered, feasible = _evaluate_population(
            population, a, costs)
        fronts, rank = _fast_non_dominated_sort(
            obj_cost, obj_vuln, uncovered, feasible)

        crowding: dict[int, float] = {}
        for front in fronts:
            crowding.update(_crowding_distance(front, obj_cost, obj_vuln))

        if np.any(feasible):
            generation_metric = float(np.min(obj_cost[feasible]))
            convergence_curve.append(generation_metric)
        else:
            penalty = float(np.max(costs) * max(1, n_items))
            generation_metric = float(np.min(obj_cost + uncovered * penalty))
            convergence_curve.append(generation_metric)

        effective_generations = gen + 1
        if _relative_improved(best_metric, generation_metric, rel_threshold):
            best_metric = generation_metric
            no_improve_generations = 0
        else:
            no_improve_generations += 1

        if patience > 0 and effective_generations >= min_gens and no_improve_generations >= patience:
            early_stopped = True
            stop_reason = "no_improve_patience"
            break

        children = np.zeros_like(population)
        all_indices = list(range(population_size))

        for i in range(0, population_size, 2):
            p1_idx = _tournament(all_indices, rank, crowding,
                                 uncovered, obj_cost, rng)
            p2_idx = _tournament(all_indices, rank, crowding,
                                 uncovered, obj_cost, rng)
            p1 = population[p1_idx]
            p2 = population[p2_idx]

            c1 = p1.copy()
            c2 = p2.copy()

            if rng.random() < float(crossover_rate):
                mask = (np_rng.random(n_sets) < 0.5).astype(np.int8)
                c1 = np.where(mask == 1, p1, p2)
                c2 = np.where(mask == 1, p2, p1)

            m1 = np_rng.random(n_sets) < mut_rate
            m2 = np_rng.random(n_sets) < mut_rate
            c1[m1] = 1 - c1[m1]
            c2[m2] = 1 - c2[m2]

            if rng.random() < float(repair_probability):
                c1 = _repair(c1.astype(np.int8), a, costs, item_to_sets, rng)
            if rng.random() < float(repair_probability):
                c2 = _repair(c2.astype(np.int8), a, costs, item_to_sets, rng)
            c1 = _prune(c1, a, costs)
            c2 = _prune(c2, a, costs)

            children[i] = c1.astype(np.int8)
            if i + 1 < population_size:
                children[i + 1] = c2.astype(np.int8)

        combined = np.vstack([population, children])
        population = _select_survivors(combined, population_size, a, costs)
        archive_pool = _dedup_population(np.vstack([archive, combined]))
        archive = _select_survivors(archive_pool, archive_size, a, costs)

    candidate_pool = _dedup_population(np.vstack([archive, population]))
    obj_cost, obj_vuln, uncovered, feasible = _evaluate_population(
        candidate_pool, a, costs)
    fronts, _ = _fast_non_dominated_sort(
        obj_cost, obj_vuln, uncovered, feasible)

    front0 = fronts[0] if fronts else list(range(candidate_pool.shape[0]))

    if np.any(feasible):
        feasible_idx = np.where(feasible)[0].tolist()
        rep_idx = min(feasible_idx, key=lambda i: (obj_cost[i], obj_vuln[i]))
    else:
        rep_idx = min(range(candidate_pool.shape[0]), key=lambda i: (
            uncovered[i], obj_cost[i], obj_vuln[i]))

    selected_bits = candidate_pool[rep_idx]
    rep_objective = float(obj_cost[rep_idx])
    rep_feasible = bool(feasible[rep_idx])
    if rep_feasible:
        selected_bits = _prune(selected_bits, a, costs)
        coverage = a @ selected_bits
        rep_feasible = bool(np.all(coverage > 0))
        rep_objective = float(np.dot(costs, selected_bits)
                              ) if rep_feasible else float("inf")
    selected_sets = [int(i) for i in np.flatnonzero(selected_bits)]

    front0_feasible = [i for i in front0 if feasible[i]]
    pareto_front_points: list[tuple[float, float]] = []
    front_hv = float("nan")
    front_hv_norm = float("nan")

    if front0_feasible:
        pareto_front_points = [
            (float(obj_cost[i]), float(obj_vuln[i])) for i in front0_feasible]
        front_cost_values = [x for x, _ in pareto_front_points]
        front_vuln_values = [y for _, y in pareto_front_points]
        cost_span = max(front_cost_values) - min(front_cost_values)
        vuln_span = max(front_vuln_values) - min(front_vuln_values)

        ref_x = max(front_cost_values) * 1.05 + 1e-9
        ref_y = max(front_vuln_values) * 1.05 + 1e-9
        front_hv = _hypervolume_2d_min(
            pareto_front_points, reference=(ref_x, ref_y))

        box = (ref_x - min(front_cost_values)) * \
            (ref_y - min(front_vuln_values))
        if box > 0 and np.isfinite(front_hv):
            front_hv_norm = float(front_hv / box)
    else:
        cost_span = float("nan")
        vuln_span = float("nan")

    runtime_sec = time.perf_counter() - start

    return {
        "objective": float(rep_objective),
        "runtime_sec": float(runtime_sec),
        "is_feasible": bool(rep_feasible),
        "selected_sets": selected_sets,
        "convergence_curve": convergence_curve,
        "meta": {
            "algorithm": "nsga2",
            "pareto_size": int(len(front0)),
            "feasible_ratio": float(np.mean(feasible.astype(float))),
            "front_cost_span": float(cost_span),
            "front_vulnerability_span": float(vuln_span),
            "front_hv": float(front_hv),
            "front_hv_norm": float(front_hv_norm),
            "pareto_front_points": pareto_front_points,
            "best_uncovered": int(np.min(uncovered)),
            "best_vulnerability": float(np.min(obj_vuln)),
            "population_size": int(population_size),
            "archive_size": int(archive.shape[0]),
            "generations": int(n_gen),
            "effective_generations": int(effective_generations),
            "early_stopped": bool(early_stopped),
            "stop_reason": stop_reason,
            "early_stop_patience": int(patience),
            "early_stop_min_generations": int(min_gens),
            "min_rel_improve": float(rel_threshold),
            "no_improve_generations": int(no_improve_generations),
            "init_strategy": init_mode,
            "elite_archive_size": int(elite_archive_size),
        },
    }
