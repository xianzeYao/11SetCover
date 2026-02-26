#!/usr/bin/env python3
"""
Set Cover问题元启发式算法研究 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from typing import Tuple, List, Dict
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SetCoverProblem:
    """Set Cover问题定义"""
    
    def __init__(self, A: np.ndarray, c: np.ndarray, name: str = ""):
        self.A = A
        self.c = c
        self.m, self.n = A.shape
        self.name = name
    
    def evaluate(self, solution: np.ndarray) -> Tuple[float, bool]:
        coverage = np.dot(self.A, solution)
        is_feasible = np.all(coverage >= 1)
        cost = np.dot(self.c, solution)
        if not is_feasible:
            uncovered = np.sum(coverage == 0)
            cost += uncovered * np.max(self.c) * self.m
        return cost, is_feasible
    
    def greedy_solution(self) -> np.ndarray:
        solution = np.zeros(self.n, dtype=int)
        uncovered = set(range(self.m))
        while uncovered:
            best_col, best_ratio = -1, -1
            for j in range(self.n):
                if solution[j] == 1:
                    continue
                covered = np.sum(self.A[list(uncovered), j])
                if covered > 0:
                    ratio = covered / self.c[j]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_col = j
            if best_col == -1:
                break
            solution[best_col] = 1
            new_covered = set(np.where(self.A[:, best_col] == 1)[0])
            uncovered -= new_covered
        return solution
    
    def get_neighborhood(self, solution: np.ndarray, k: int = 1) -> List[np.ndarray]:
        neighbors = []
        indices = np.where(solution == 1)[0]
        zero_indices = np.where(solution == 0)[0]
        
        for idx in indices:
            neighbor = solution.copy()
            neighbor[idx] = 0
            neighbors.append(neighbor)
        
        for idx in zero_indices:
            neighbor = solution.copy()
            neighbor[idx] = 1
            neighbors.append(neighbor)
        
        if k >= 2 and len(indices) > 0 and len(zero_indices) > 0:
            max_swaps = min(20, len(indices) * len(zero_indices))
            for _ in range(max_swaps):
                i = np.random.choice(indices)
                j = np.random.choice(zero_indices)
                neighbor = solution.copy()
                neighbor[i] = 0
                neighbor[j] = 1
                neighbors.append(neighbor)
        
        return neighbors

class SimulatedAnnealing:
    """模拟退火算法"""
    
    def __init__(self, problem: SetCoverProblem, 
                 initial_temp: float = 1000,
                 cooling_rate: float = 0.995,
                 min_temp: float = 0.01,
                 max_iter: int = 10000,
                 max_iter_per_temp: int = 100):
        self.problem = problem
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iter = max_iter
        self.max_iter_per_temp = max_iter_per_temp
        self.history = []
    
    def solve(self, initial_solution: np.ndarray = None) -> Tuple[np.ndarray, float, Dict]:
        start_time = time.time()
        
        if initial_solution is None:
            current = self.problem.greedy_solution()
        else:
            current = initial_solution.copy()
        
        current_cost, _ = self.problem.evaluate(current)
        best_solution = current.copy()
        best_cost = current_cost
        
        temp = self.initial_temp
        iteration = 0
        
        while temp > self.min_temp and iteration < self.max_iter:
            for _ in range(self.max_iter_per_temp):
                neighbors = self.problem.get_neighborhood(current, k=1)
                if not neighbors:
                    break
                
                next_solution = neighbors[np.random.randint(len(neighbors))]
                next_cost, _ = self.problem.evaluate(next_solution)
                
                delta = next_cost - current_cost
                
                if delta < 0:
                    current = next_solution
                    current_cost = next_cost
                    if current_cost < best_cost:
                        best_solution = current.copy()
                        best_cost = current_cost
                else:
                    prob = np.exp(-delta / temp)
                    if np.random.random() < prob:
                        current = next_solution
                        current_cost = next_cost
                
                iteration += 1
                if iteration % 100 == 0:
                    self.history.append({
                        'iteration': iteration,
                        'temperature': temp,
                        'current_cost': current_cost,
                        'best_cost': best_cost
                    })
            
            temp *= self.cooling_rate
        
        runtime = time.time() - start_time
        
        return best_solution, best_cost, {
            'runtime': runtime,
            'iterations': iteration,
            'final_temp': temp,
            'history': self.history
        }

class GeneticAlgorithm:
    """遗传算法"""
    
    def __init__(self, problem: SetCoverProblem,
                 population_size: int = 50,
                 generations: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.05,
                 elitism: int = 2):
        self.problem = problem
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.elitism = elitism
        self.history = []
    
    def initialize_population(self) -> List[np.ndarray]:
        population = []
        for _ in range(self.pop_size // 2):
            sol = self.problem.greedy_solution()
            if np.random.random() < 0.3:
                idx = np.random.randint(self.problem.n)
                sol[idx] = 1 - sol[idx]
            population.append(sol)
        
        for _ in range(self.pop_size - len(population)):
            sol = np.random.choice([0, 1], size=self.problem.n, p=[0.9, 0.1])
            coverage = np.dot(self.problem.A, sol)
            for i in range(self.problem.m):
                if coverage[i] == 0:
                    valid_cols = np.where(self.problem.A[i] == 1)[0]
                    if len(valid_cols) > 0:
                        sol[np.random.choice(valid_cols)] = 1
                        coverage = np.dot(self.problem.A, sol)
            population.append(sol)
        
        return population
    
    def fitness(self, solution: np.ndarray) -> float:
        cost, feasible = self.problem.evaluate(solution)
        if not feasible:
            cost += 1e6
        return 1.0 / (cost + 1)
    
    def selection(self, population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        tournament_size = 3
        selected = []
        for _ in range(2):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitnesses[i] for i in tournament]
            winner_idx = tournament[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected[0], selected[1]
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > self.cx_rate:
            return parent1.copy(), parent2.copy()
        mask = np.random.choice([0, 1], size=self.problem.n)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def mutation(self, solution: np.ndarray) -> np.ndarray:
        mutated = solution.copy()
        for i in range(self.problem.n):
            if np.random.random() < self.mut_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def repair(self, solution: np.ndarray) -> np.ndarray:
        coverage = np.dot(self.problem.A, solution)
        repaired = solution.copy()
        for i in range(self.problem.m):
            if coverage[i] == 0:
                valid_cols = np.where(self.problem.A[i] == 1)[0]
                if len(valid_cols) > 0:
                    best_col = min(valid_cols, key=lambda j: self.problem.c[j])
                    repaired[best_col] = 1
                    coverage = np.dot(self.problem.A, repaired)
        return repaired
    
    def solve(self) -> Tuple[np.ndarray, float, Dict]:
        start_time = time.time()
        population = self.initialize_population()
        best_solution = None
        best_cost = float('inf')
        
        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            costs = [self.problem.evaluate(ind)[0] for ind in population]
            
            gen_best_idx = np.argmin(costs)
            gen_best_cost = costs[gen_best_idx]
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_solution = population[gen_best_idx].copy()
            
            self.history.append({
                'generation': gen,
                'best_cost': best_cost,
                'avg_cost': np.mean(costs),
                'diversity': np.std(costs)
            })
            
            new_population = [population[gen_best_idx].copy()]
            if self.elitism > 1:
                sorted_indices = np.argsort(costs)
                for i in range(1, min(self.elitism, len(sorted_indices))):
                    new_population.append(population[sorted_indices[i]].copy())
            
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.selection(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                child1 = self.repair(child1)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    child2 = self.repair(child2)
                    new_population.append(child2)
            
            population = new_population
        
        runtime = time.time() - start_time
        
        return best_solution, best_cost, {
            'runtime': runtime,
            'generations': self.generations,
            'history': self.history
        }

class HybridGASA:
    """混合遗传-模拟退火算法"""
    
    def __init__(self, problem: SetCoverProblem,
                 population_size: int = 50,
                 generations: int = 100,
                 sa_initial_temp: float = 1000,
                 sa_cooling_rate: float = 0.99,
                 sa_min_temp: float = 0.1,
                 sa_max_iter: int = 5000,
                 hybrid_frequency: int = 10):
        self.problem = problem
        self.pop_size = population_size
        self.generations = generations
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_min_temp = sa_min_temp
        self.sa_max_iter = sa_max_iter
        self.hybrid_frequency = hybrid_frequency
        self.ga = GeneticAlgorithm(problem, population_size, generations)
        self.history = []
    
    def local_search_sa(self, solution: np.ndarray) -> Tuple[np.ndarray, float]:
        sa = SimulatedAnnealing(
            self.problem,
            initial_temp=self.sa_initial_temp,
            cooling_rate=self.sa_cooling_rate,
            min_temp=self.sa_min_temp,
            max_iter=self.sa_max_iter,
            max_iter_per_temp=50
        )
        best_sol, best_cost, _ = sa.solve(solution)
        return best_sol, best_cost
    
    def solve(self) -> Tuple[np.ndarray, float, Dict]:
        start_time = time.time()
        population = self.ga.initialize_population()
        best_solution = None
        best_cost = float('inf')
        total_sa_time = 0
        
        for gen in range(self.generations):
            fitnesses = [self.ga.fitness(ind) for ind in population]
            costs = [self.problem.evaluate(ind)[0] for ind in population]
            
            gen_best_idx = np.argmin(costs)
            gen_best_cost = costs[gen_best_idx]
            
            if gen % self.hybrid_frequency == 0 and gen > 0:
                sa_start = time.time()
                elite_count = max(1, self.pop_size // 5)
                sorted_indices = np.argsort(costs)
                
                for i in range(elite_count):
                    idx = sorted_indices[i]
                    improved_sol, improved_cost = self.local_search_sa(population[idx])
                    if improved_cost < costs[idx]:
                        population[idx] = improved_sol
                        costs[idx] = improved_cost
                
                total_sa_time += time.time() - sa_start
            
            current_best_idx = np.argmin(costs)
            current_best_cost = costs[current_best_idx]
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_solution = population[current_best_idx].copy()
            
            self.history.append({
                'generation': gen,
                'best_cost': best_cost,
                'avg_cost': np.mean(costs),
                'diversity': np.std(costs),
                'sa_applied': gen % self.hybrid_frequency == 0 and gen > 0
            })
            
            new_population = [population[current_best_idx].copy()]
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.ga.selection(population, fitnesses)
                child1, child2 = self.ga.crossover(parent1, parent2)
                child1 = self.ga.mutation(child1)
                child2 = self.ga.mutation(child2)
                child1 = self.ga.repair(child1)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    child2 = self.ga.repair(child2)
                    new_population.append(child2)
            
            population = new_population
        
        total_time = time.time() - start_time
        
        return best_solution, best_cost, {
            'runtime': total_time,
            'ga_generations': self.generations,
            'sa_total_time': total_sa_time,
            'hybrid_frequency': self.hybrid_frequency,
            'history': self.history
        }

class ExperimentFramework:
    """完整实验框架"""
    
    def __init__(self, instances: Dict, config: Dict):
        self.instances = instances
        self.config = config
        self.results = []
    
    def run_experiment(self):
        for name, data in self.instances.items():
            print(f"\n处理实例: {name}")
            problem = SetCoverProblem(data['A'], data['c'], name)
            category = name.split('_')[0]
            
            for alg_name in self.config['algorithms']:
                print(f"  - 算法: {alg_name}")
                runtimes, costs = [], []
                
                for r in range(self.config['repeat']):
                    if alg_name == 'SA':
                        solver = SimulatedAnnealing(problem)
                    elif alg_name == 'GA':
                        solver = GeneticAlgorithm(problem)
                    elif alg_name == 'HGASA':
                        solver = HybridGASA(problem)
                    
                    _, cost, stats = solver.solve()
                    runtimes.append(stats['runtime'])
                    costs.append(cost)
                
                self.results.append({
                    'instance': name,
                    'algorithm': alg_name,
                    'category': category,
                    'm': data['m'],
                    'n': data['n'],
                    'runtime_mean': np.mean(runtimes),
                    'runtime_std': np.std(runtimes),
                    'cost_mean': np.mean(costs),
                    'cost_std': np.std(costs),
                    'cost_min': np.min(costs),
                    'cost_max': np.max(costs)
                })
        
        return pd.DataFrame(self.results)
    
    def analyze(self):
        df = pd.DataFrame(self.results)
        
        # 统计检验
        print("\n=== 显著性统计分析 ===")
        for cat in df['category'].unique():
            cat_data = df[df['category'] == cat]
            sa_costs = cat_data[cat_data['algorithm'] == 'SA']['cost_mean'].values
            ga_costs = cat_data[cat_data['algorithm'] == 'GA']['cost_mean'].values
            hgasa_costs = cat_data[cat_data['algorithm'] == 'HGASA']['cost_mean'].values
            
            if len(sa_costs) > 0 and len(hgasa_costs) > 0:
                stat, p = stats.wilcoxon(sa_costs, hgasa_costs)
                print(f"{cat}: SA vs HGASA p-value = {p:.4f}")
            
            if len(ga_costs) > 0 and len(hgasa_costs) > 0:
                stat, p = stats.wilcoxon(ga_costs, hgasa_costs)
                print(f"{cat}: GA vs HGASA p-value = {p:.4f}")
        
        # 可视化
        self._plot_results(df)
        
        return df
    
    def _plot_results(self, df: pd.DataFrame):
        colors = {'SA': '#FF6B6B', 'GA': '#4ECDC4', 'HGASA': '#45B7D1'}
        
        # 箱线图
        plt.figure(figsize=(12, 6))
        categories = sorted(df['category'].unique())
        algorithms = ['SA', 'GA', 'HGASA']
        
        data_to_plot, positions, colors_list, labels = [], [], [], []
        pos = 1
        for cat in categories:
            for alg in algorithms:
                values = df[(df['category'] == cat) & (df['algorithm'] == alg)]['cost_mean'].values
                if len(values) > 0:
                    data_to_plot.append(values)
                    positions.append(pos)
                    colors_list.append(colors[alg])
                    labels.append(f"{cat}\n{alg}")
                    pos += 1
            pos += 1
        
        bp = plt.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        plt.xticks(positions, labels, rotation=45, ha='right')
        plt.ylabel('Cost')
        plt.title('Solution Quality by Category and Algorithm')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('results_boxplot.png', dpi=300)
        plt.close()
        
        # 可扩展性分析
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        df['problem_size'] = df['m'] * df['n']
        
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg]
            size_groups = alg_data.groupby('problem_size').agg({
                'runtime_mean': ['mean', 'std'],
                'cost_mean': ['mean', 'std']
            }).reset_index()
            
            sizes = size_groups['problem_size'].values
            time_means = size_groups['runtime_mean']['mean'].values
            time_stds = size_groups['runtime_mean']['std'].values
            cost_means = size_groups['cost_mean']['mean'].values
            
            ax1.errorbar(sizes, time_means, yerr=time_stds, label=alg, color=colors[alg], marker='o', capsize=5)
            ax2.plot(sizes, cost_means, label=alg, color=colors[alg], marker='s')
        
        ax1.set_xlabel('Problem Size (m × n)')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.set_title('Scalability: Runtime vs Problem Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        ax2.set_xlabel('Problem Size (m × n)')
        ax2.set_ylabel('Average Cost')
        ax2.set_title('Scalability: Solution Quality vs Problem Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('scalability_analysis.png', dpi=300)
        plt.close()

def main():
    """主程序"""
    print("=" * 60)
    print("Set Cover问题元启发式算法研究")
    print("=" * 60)
    
    # 生成实例
    print("\n[1/3] 生成实验实例...")
    generator = SetCoverInstanceGenerator(seed=42)
    instances = generator.generate_all("./instances")
    
    # 配置
    config = {
        'algorithms': ['SA', 'GA', 'HGASA'],
        'repeat': 5
    }
    
    # 运行实验
    print("\n[2/3] 运行算法实验...")
    framework = ExperimentFramework(instances, config)
    results = framework.run_experiment()
    
    # 分析
    print("\n[3/3] 数据分析与可视化...")
    framework.analyze()
    
    # 保存结果
    results.to_csv("experiment_results.csv", index=False)
    print("\n✓ 实验完成！结果已保存至 experiment_results.csv")

if __name__ == "__main__":
    main()
