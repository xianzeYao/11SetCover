from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np


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


def _build_problem(instance):
    n_items = instance.n_items
    n_sets = instance.n_sets
    a = np.zeros((n_items, n_sets), dtype=np.int8)
    set_items: list[tuple[int, ...]] = []
    item_to_sets: list[list[int]] = [[] for _ in range(n_items)]

    for j, rec in enumerate(instance.sets):
        items = tuple(int(i) for i in rec.items)
        set_items.append(items)
        for item in items:
            a[item, j] = 1
            item_to_sets[item].append(j)

    costs = np.array([float(rec.cost) for rec in instance.sets], dtype=float)
    return a, costs, set_items, item_to_sets


def _evaluate_solution(solution: np.ndarray, a: np.ndarray, costs: np.ndarray, penalty: float) -> tuple[float, float, int, bool]:
    coverage = a @ solution
    uncovered = int(np.sum(coverage == 0))
    cost = float(np.dot(costs, solution))
    feasible = uncovered == 0
    penalized = float(cost + uncovered * penalty)
    return penalized, cost, uncovered, feasible


def _evaluate_population(population: np.ndarray, a: np.ndarray, costs: np.ndarray, penalty: float):
    penalized = np.zeros(population.shape[0], dtype=float)
    raw_cost = np.zeros(population.shape[0], dtype=float)
    uncovered = np.zeros(population.shape[0], dtype=int)
    feasible = np.zeros(population.shape[0], dtype=bool)

    for i in range(population.shape[0]):
        p, c, u, f = _evaluate_solution(population[i], a, costs, penalty)
        penalized[i] = p
        raw_cost[i] = c
        uncovered[i] = u
        feasible[i] = f
    return penalized, raw_cost, uncovered, feasible


def _repair_solution(
    solution: np.ndarray,
    a: np.ndarray,
    costs: np.ndarray,
    item_to_sets: list[list[int]],
    rng: random.Random,
) -> np.ndarray:
    repaired = solution.copy()
    if repaired.sum() == 0 and repaired.size > 0:
        repaired[rng.randrange(repaired.size)] = 1

    coverage = a @ repaired
    for item, cov in enumerate(coverage):
        if cov > 0:
            continue
        candidates = item_to_sets[item]
        if not candidates:
            continue
        min_cost = min(costs[j] for j in candidates)
        ties = [j for j in candidates if costs[j] == min_cost]
        chosen = rng.choice(ties)
        if repaired[chosen] == 1:
            continue
        repaired[chosen] = 1
        coverage += a[:, chosen]
    return repaired


def _prune_solution(solution: np.ndarray, a: np.ndarray, costs: np.ndarray) -> np.ndarray:
    pruned = solution.copy()
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

    if pruned.sum() == 0 and solution.size > 0:
        keep = int(np.argmin(costs))
        pruned[keep] = 1
    return pruned


def _greedy_seed(instance, rng: random.Random) -> np.ndarray:
    uncovered = set(range(instance.n_items))
    selected = np.zeros(instance.n_sets, dtype=np.int8)
    set_items = [set(rec.items) for rec in instance.sets]
    available = set(range(instance.n_sets))

    while uncovered and available:
        best_score = float("-inf")
        candidates: list[int] = []
        for idx in available:
            rec = instance.sets[idx]
            gain = len(uncovered.intersection(set_items[idx]))
            if gain <= 0:
                continue
            score = float("inf") if rec.cost <= 0 else gain / rec.cost
            if score > best_score:
                best_score = score
                candidates = [idx]
            elif score == best_score:
                candidates.append(idx)
        if not candidates:
            break
        chosen = rng.choice(candidates)
        selected[chosen] = 1
        available.remove(chosen)
        uncovered.difference_update(set_items[chosen])

    if selected.sum() == 0 and instance.n_sets > 0:
        selected[rng.randrange(instance.n_sets)] = 1
    return selected


def _initialize_population(
    instance,
    pop_size: int,
    init_strategy: str,
    init_on_prob: float,
    repair_init: bool,
    rng: random.Random,
    np_rng: np.random.Generator,
    a: np.ndarray,
    costs: np.ndarray,
    item_to_sets: list[list[int]],
) -> np.ndarray:
    n_sets = instance.n_sets
    population = np.zeros((pop_size, n_sets), dtype=np.int8)
    init_mode = str(init_strategy).strip().lower()
    if init_mode not in {"random", "hybrid", "greedy"}:
        raise ValueError(
            f"Unsupported init_strategy={init_strategy}, expected random|hybrid|greedy")

    greedy = _greedy_seed(instance, rng)
    p_on = min(max(float(init_on_prob), 0.0), 1.0)

    half = pop_size // 2
    for i in range(pop_size):
        if init_mode == "random":
            ind = (np_rng.random(n_sets) < p_on).astype(np.int8)
        elif init_mode == "greedy":
            ind = greedy.copy()
            if i > 0 and rng.random() < 0.3 and n_sets > 0:
                idx = rng.randrange(n_sets)
                ind[idx] = 1 - ind[idx]
        else:
            if i < half:
                ind = greedy.copy()
                if rng.random() < 0.3 and n_sets > 0:
                    idx = rng.randrange(n_sets)
                    ind[idx] = 1 - ind[idx]
            else:
                ind = (np_rng.random(n_sets) < p_on).astype(np.int8)

        if ind.sum() == 0 and n_sets > 0:
            ind[rng.randrange(n_sets)] = 1
        if repair_init:
            ind = _repair_solution(ind, a, costs, item_to_sets, rng)
            ind = _prune_solution(ind, a, costs)
        population[i] = ind

    return population


def _tournament_select(population: np.ndarray, fitness: np.ndarray, rng: random.Random, k: int = 3) -> np.ndarray:
    n = population.shape[0]
    k_eff = min(max(2, int(k)), n)
    idxs = rng.sample(range(n), k_eff)
    winner = max(idxs, key=lambda idx: fitness[idx])
    return population[winner].copy()


def _uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float,
    np_rng: np.random.Generator,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() > float(crossover_rate):
        return parent1.copy(), parent2.copy()
    mask = np_rng.random(parent1.size) < 0.5
    child1 = np.where(mask, parent1, parent2).astype(np.int8)
    child2 = np.where(mask, parent2, parent1).astype(np.int8)
    return child1, child2


def _bitflip_mutation(solution: np.ndarray, mutation_rate: float, np_rng: np.random.Generator, rng: random.Random) -> np.ndarray:
    mutated = solution.copy()
    p = min(max(float(mutation_rate), 0.0), 1.0)
    flips = np_rng.random(mutated.size) < p
    if np.any(flips):
        mutated[flips] = 1 - mutated[flips]
    if mutated.sum() == 0 and mutated.size > 0:
        mutated[rng.randrange(mutated.size)] = 1
    return mutated


def solve(
    instance,
    seed: int,
    population_size: int = 50,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    elitism: int = 2,
    tournament_size: int = 3,
    init_strategy: str = "random",
    init_on_prob: float = 0.1,
    repair_init: bool = True,
    infeasible_penalty: float | None = None,
    early_stop_patience: int = 30,
    early_stop_min_generations: int = 40,
    min_rel_improve: float = 1e-4,
    **kwargs: Any,
) -> dict[str, Any]:
    """Genetic algorithm adapted from source/meta GeneticAlgorithm core."""

    start = time.perf_counter()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    a, costs, _set_items, item_to_sets = _build_problem(instance)
    n_items, n_sets = a.shape
    pop_size = max(4, int(population_size))
    n_gen = max(1, int(generations))
    elite_n = min(max(1, int(elitism)), pop_size)

    penalty = (
        float(infeasible_penalty)
        if infeasible_penalty is not None
        else float(max(1.0, float(np.max(costs)) * max(1, n_items)))
    )

    population = _initialize_population(
        instance=instance,
        pop_size=pop_size,
        init_strategy=init_strategy,
        init_on_prob=float(init_on_prob),
        repair_init=bool(repair_init),
        rng=rng,
        np_rng=np_rng,
        a=a,
        costs=costs,
        item_to_sets=item_to_sets,
    )

    best_feasible_cost = float("inf")
    best_feasible_solution = population[0].copy()
    best_any_value = float("inf")
    best_any_cost = float("inf")
    best_any_uncovered = n_items
    best_any_solution = population[0].copy()

    convergence_curve: list[float] = []
    best_metric = float("inf")
    no_improve_generations = 0
    early_stopped = False
    stop_reason = "max_generations"
    patience = max(0, int(early_stop_patience))
    min_gens = max(0, int(early_stop_min_generations))
    rel_threshold = max(0.0, float(min_rel_improve))
    effective_generations = 0

    for gen in range(n_gen):
        penalized, raw_cost, uncovered, feasible = _evaluate_population(
            population, a, costs, penalty)
        fitness = 1.0 / (penalized + 1.0)

        idx_best_pen = int(np.argmin(penalized))
        if feasible[idx_best_pen] and raw_cost[idx_best_pen] < best_feasible_cost:
            best_feasible_cost = float(raw_cost[idx_best_pen])
            best_feasible_solution = population[idx_best_pen].copy()
        if penalized[idx_best_pen] < best_any_value:
            best_any_value = float(penalized[idx_best_pen])
            best_any_cost = float(raw_cost[idx_best_pen])
            best_any_uncovered = int(uncovered[idx_best_pen])
            best_any_solution = population[idx_best_pen].copy()

        generation_metric = float(best_feasible_cost) if math.isfinite(
            best_feasible_cost) else float(best_any_value)
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

        sorted_idx = np.argsort(penalized)
        new_population = [population[int(sorted_idx[i])].copy()
                          for i in range(elite_n)]

        while len(new_population) < pop_size:
            p1 = _tournament_select(
                population, fitness, rng=rng, k=tournament_size)
            p2 = _tournament_select(
                population, fitness, rng=rng, k=tournament_size)
            c1, c2 = _uniform_crossover(
                p1, p2, crossover_rate=crossover_rate, np_rng=np_rng, rng=rng)
            c1 = _bitflip_mutation(
                c1, mutation_rate=mutation_rate, np_rng=np_rng, rng=rng)
            c2 = _bitflip_mutation(
                c2, mutation_rate=mutation_rate, np_rng=np_rng, rng=rng)
            c1 = _repair_solution(c1, a, costs, item_to_sets, rng)
            c2 = _repair_solution(c2, a, costs, item_to_sets, rng)
            c1 = _prune_solution(c1, a, costs)
            c2 = _prune_solution(c2, a, costs)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = np.array(new_population, dtype=np.int8)

    if math.isfinite(best_feasible_cost):
        final_solution = best_feasible_solution
        objective = float(best_feasible_cost)
        is_feasible = True
        uncovered_count = 0
    else:
        final_solution = best_any_solution
        objective = float("inf")
        is_feasible = False
        uncovered_count = int(best_any_uncovered)

    selected_sets = [int(i) for i in np.flatnonzero(final_solution)]
    runtime_sec = time.perf_counter() - start

    return {
        "objective": float(objective),
        "runtime_sec": float(runtime_sec),
        "is_feasible": bool(is_feasible),
        "selected_sets": selected_sets,
        "convergence_curve": convergence_curve,
        "meta": {
            "algorithm": "ga",
            "population_size": int(pop_size),
            "generations": int(n_gen),
            "effective_generations": int(effective_generations),
            "early_stopped": bool(early_stopped),
            "stop_reason": stop_reason,
            "early_stop_patience": int(patience),
            "early_stop_min_generations": int(min_gens),
            "min_rel_improve": float(rel_threshold),
            "no_improve_generations": int(no_improve_generations),
            "crossover_rate": float(crossover_rate),
            "mutation_rate": float(mutation_rate),
            "elitism": int(elite_n),
            "init_strategy": str(init_strategy).strip().lower(),
            "init_on_prob": float(init_on_prob),
            "repair_init": bool(repair_init),
            "best_any_penalized": float(best_any_value),
            "best_any_cost": float(best_any_cost),
            "uncovered_count": int(uncovered_count),
        },
    }
