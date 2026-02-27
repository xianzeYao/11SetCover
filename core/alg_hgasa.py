from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np

from core.alg_ga import (
    _NumpyRNGAdapter,
    _bitflip_mutation,
    _build_problem,
    _evaluate_population,
    _evaluate_solution,
    _initialize_population,
    _relative_improved,
    _prune_solution,
    _repair_solution,
    _tournament_select,
    _uniform_crossover,
)


def _build_neighbors_k1(solution: np.ndarray) -> list[np.ndarray]:
    neighbors: list[np.ndarray] = []
    for idx, flag in enumerate(solution.tolist()):
        neighbor = solution.copy().astype(np.int8)
        neighbor[int(idx)] = np.int8(0 if int(flag) == 1 else 1)
        neighbors.append(neighbor)
    return neighbors


def _sa_local_search(
    initial_solution: np.ndarray,
    a: np.ndarray,
    costs: np.ndarray,
    set_items: list[tuple[int, ...]],
    item_to_sets: list[list[int]],
    rng: random.Random,
    initial_temp: float,
    cooling_rate: float,
    min_temp: float,
    max_iter: int,
    moves_per_temp: int,
    penalty: float,
) -> tuple[np.ndarray, float, float, int, bool, dict[str, Any]]:
    selected = initial_solution.copy().astype(np.int8)

    current_value, _current_raw_cost, _current_uncovered, _ = _evaluate_solution(
        solution=selected,
        a=a,
        costs=costs,
        penalty=penalty,
    )

    best_value = float(current_value)
    best_solution = selected.copy()

    temp = float(initial_temp)
    iter_count = 0
    temp_steps = 0
    accepted = 0

    while temp > float(min_temp) and iter_count < int(max_iter):
        for _ in range(int(moves_per_temp)):
            if iter_count >= int(max_iter):
                break
            neighbors = _build_neighbors_k1(solution=selected)
            if not neighbors:
                break
            next_solution = neighbors[rng.randrange(len(neighbors))]
            new_value, _new_cost, _new_uncovered, _new_feasible = _evaluate_solution(
                solution=next_solution,
                a=a,
                costs=costs,
                penalty=penalty,
            )
            delta = new_value - current_value

            accept = delta < 0
            if not accept:
                prob = math.exp(-delta / max(temp, 1e-12))
                accept = rng.random() < prob

            if accept:
                selected = next_solution
                accepted += 1
                current_value = new_value

                if current_value < best_value:
                    best_value = float(current_value)
                    best_solution = selected.copy()

            iter_count += 1

        temp *= float(cooling_rate)
        temp_steps += 1

    final_solution = best_solution
    final_penalized, final_cost, final_uncovered, final_feasible = _evaluate_solution(
        solution=final_solution,
        a=a,
        costs=costs,
        penalty=penalty,
    )

    meta = {
        "iterations": int(iter_count),
        "temp_steps": int(temp_steps),
        "accepted_moves": int(accepted),
        "acceptance_rate": float(accepted / max(1, iter_count)),
        "final_temp": float(temp),
        "uncovered_count": int(final_uncovered),
        "best_penalized": float(final_penalized),
        "neighbor_mode": "full_enumeration_k1",
        "evaluation_mode": "full_recompute",
    }
    return final_solution, float(final_cost), float(final_penalized), int(final_uncovered), bool(final_feasible), meta


def solve(
    instance,
    seed: int,
    population_size: int = 50,
    generations: int = 120,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    elitism: int = 2,
    tournament_size: int = 3,
    init_strategy: str = "random",
    init_on_prob: float = 0.1,
    repair_init: bool = True,
    hybrid_frequency: int = 10,
    elite_fraction: float = 0.2,
    sa_initial_temp: float = 1000.0,
    sa_cooling_rate: float = 0.99,
    sa_min_temp: float = 0.1,
    sa_max_iter: int = 5000,
    sa_moves_per_temp: int = 50,
    infeasible_penalty: float | None = None,
    early_stop_patience: int = 20,
    early_stop_min_generations: int = 30,
    min_rel_improve: float = 1e-4,
    **kwargs: Any,
) -> dict[str, Any]:
    """Hybrid GA-SA adapted from source/meta HybridGASA core."""

    start = time.perf_counter()
    rng = _NumpyRNGAdapter(seed=seed)
    np_rng = rng.np_rng

    a, costs, set_items, item_to_sets = _build_problem(instance)
    n_items, _n_sets = a.shape
    pop_size = max(4, int(population_size))
    n_gen = max(1, int(generations))
    elite_n = min(max(1, int(elitism)), pop_size)
    hybrid_freq = max(1, int(hybrid_frequency))
    elite_count = max(1, int(round(pop_size * float(elite_fraction))))

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

    sa_applied_generations = 0
    sa_calls = 0
    sa_improved = 0
    sa_total_time = 0.0

    for gen in range(n_gen):
        penalized, raw_cost, uncovered, feasible = _evaluate_population(
            population, a, costs, penalty)

        if gen > 0 and gen % hybrid_freq == 0:
            sa_applied_generations += 1
            sorted_idx = np.argsort(penalized)[:elite_count]
            sa_start = time.perf_counter()
            for idx in sorted_idx:
                idx_int = int(idx)
                sa_calls += 1
                improved, _imp_cost, imp_pen, _imp_uncovered, _imp_feasible, _imp_meta = _sa_local_search(
                    initial_solution=population[idx_int],
                    a=a,
                    costs=costs,
                    set_items=set_items,
                    item_to_sets=item_to_sets,
                    rng=rng,
                    initial_temp=float(sa_initial_temp),
                    cooling_rate=float(sa_cooling_rate),
                    min_temp=float(sa_min_temp),
                    max_iter=int(sa_max_iter),
                    moves_per_temp=int(sa_moves_per_temp),
                    penalty=penalty,
                )
                if imp_pen + 1e-12 < float(penalized[idx_int]):
                    population[idx_int] = improved
                    sa_improved += 1
            sa_total_time += float(time.perf_counter() - sa_start)
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
            "algorithm": "hgasa",
            "population_size": int(pop_size),
            "generations": int(n_gen),
            "effective_generations": int(effective_generations),
            "early_stopped": bool(early_stopped),
            "stop_reason": stop_reason,
            "early_stop_patience": int(patience),
            "early_stop_min_generations": int(min_gens),
            "min_rel_improve": float(rel_threshold),
            "no_improve_generations": int(no_improve_generations),
            "hybrid_frequency": int(hybrid_freq),
            "elite_fraction": float(elite_fraction),
            "sa_applied_generations": int(sa_applied_generations),
            "sa_calls": int(sa_calls),
            "sa_improved": int(sa_improved),
            "sa_total_time_sec": float(sa_total_time),
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
