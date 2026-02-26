import os
import time
import csv
import random
from typing import Tuple, Set, List, Optional


class SetCoverProblem:
    """集合覆盖问题实例封装"""
    def __init__(self, all_elements: set, sets: list[set], costs: list[float]):
        """
        参数说明：
        - all_elements: 所有需要覆盖的元素集合（如 {0,1,2,3}）
        - sets: 候选集合列表（如 [{0,1}, {1,2}, {3}]）
        - costs: 每个候选集合的成本（长度与sets一致，如 [2.0, 3.0, 1.0]）
        """
        self.all_elements = all_elements  # 总元素集合
        self.sets = sets                  # 候选集合列表
        self.costs = costs                # 集合成本列表
        self.n_sets = len(sets)           # 候选集合总数
        self.n_elements = len(all_elements)  # 总元素数


class SetCoverSolution:
    """集合覆盖解决方案封装"""
    def __init__(self):
        self.selected_sets = []  # 选中的集合索引列表
        self.total_cost = 0.0    # 总成本
        self.covered_elements = set()  # 已覆盖的元素集合

    def __str__(self):
        return (f"选中集合索引：{self.selected_sets}\n"
                f"总成本：{self.total_cost:.2f}\n"
                f"已覆盖元素数：{len(self.covered_elements)}/{self.covered_elements if len(self.covered_elements)<=20 else '...'}")


def read_setcover_file(file_path: str) -> SetCoverProblem:
    """
    读取setcover/data目录下的集合覆盖数据文件，解析为SetCoverProblem实例
    :param file_path: 数据文件路径（如 "setcover/data/sc_1000_0"）
    :return: 封装后的问题实例
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    all_elements: Set[int] = set()
    sets: List[Set[int]] = []
    costs: List[float] = []

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # 过滤空行和空白

    # 解析第一行：总元素数、总集合数
    first_line = lines[0].split()
    n_elements = int(first_line[0])
    n_sets = int(first_line[1])

    # 解析后续行：每个集合的成本+元素
    for line_idx in range(1, n_sets + 1):
        parts = lines[line_idx].split()
        if not parts:
            raise ValueError(f"第{line_idx+1}行格式错误：无内容")
        
        # 提取成本（转为浮点数）
        cost = float(parts[0])
        # 提取元素（转为整数，注意元素编号从0开始/1开始均兼容）
        elements = set(int(x) for x in parts[1:])
        
        # 收集数据
        costs.append(cost)
        sets.append(elements)
        all_elements.update(elements)

    # 校验：解析的元素数/集合数与声明一致（容错，部分文件可能有编号偏移）
    if len(all_elements) != n_elements:
        print(f"警告：文件声明元素数{n_elements}，实际解析{len(all_elements)}个元素")
    if len(sets) != n_sets:
        raise ValueError(f"文件声明集合数{n_sets}，实际解析{len(sets)}个集合")

    return SetCoverProblem(all_elements, sets, costs)


def greedy_001(problem: SetCoverProblem) -> SetCoverSolution:
    """
    greedy_001：每次选「单位成本覆盖新元素数最多」的集合（效率优先）
    效率 = 未覆盖元素数 / 集合成本
    """
    solution = SetCoverSolution()
    remaining_elements = problem.all_elements.copy()
    available_set_indices = set(range(problem.n_sets))  # 未选中的集合索引

    while remaining_elements and available_set_indices:
        best_set_idx = -1
        max_efficiency = -1.0

        # 遍历所有可用集合，计算效率并选最优
        for set_idx in available_set_indices:
            current_set = problem.sets[set_idx]
            current_cost = problem.costs[set_idx]
            # 计算该集合能覆盖的未覆盖元素数
            uncovered_in_set = len(current_set & remaining_elements)
            
            if uncovered_in_set == 0:
                continue  # 无新覆盖元素，跳过
            
            # 计算效率（单位成本覆盖数）
            efficiency = uncovered_in_set / current_cost
            # 更新最优集合
            if efficiency > max_efficiency:
                max_efficiency = efficiency
                best_set_idx = set_idx

        if best_set_idx == -1:
            break  # 无可用集合（无法覆盖所有元素）

        # 选中最优集合，更新解决方案
        solution.selected_sets.append(best_set_idx)
        solution.total_cost += problem.costs[best_set_idx]
        # 更新已覆盖元素和剩余元素
        selected_set = problem.sets[best_set_idx]
        solution.covered_elements.update(selected_set)
        remaining_elements -= selected_set
        # 标记该集合为已选（从可用集合中移除）
        available_set_indices.remove(best_set_idx)

    return solution


def greedy_002(problem: SetCoverProblem) -> SetCoverSolution:
    """
    greedy_002：每次选「覆盖未覆盖元素数最多」的集合（覆盖数优先，忽略成本）
    """
    solution = SetCoverSolution()
    remaining_elements = problem.all_elements.copy()
    available_set_indices = set(range(problem.n_sets))

    while remaining_elements and available_set_indices:
        best_set_idx = -1
        max_uncovered = -1

        # 遍历可用集合，选覆盖未覆盖元素最多的
        for set_idx in available_set_indices:
            current_set = problem.sets[set_idx]
            uncovered_in_set = len(current_set & remaining_elements)
            
            if uncovered_in_set > max_uncovered:
                max_uncovered = uncovered_in_set
                best_set_idx = set_idx

        if best_set_idx == -1:
            break

        # 更新解决方案
        solution.selected_sets.append(best_set_idx)
        solution.total_cost += problem.costs[best_set_idx]
        selected_set = problem.sets[best_set_idx]
        solution.covered_elements.update(selected_set)
        remaining_elements -= selected_set
        available_set_indices.remove(best_set_idx)

    return solution


def greedy_003(problem: SetCoverProblem) -> SetCoverSolution:
    """
    greedy_003：每次选「成本最低且能覆盖未覆盖元素」的集合（成本优先，忽略覆盖数）
    """
    solution = SetCoverSolution()
    remaining_elements = problem.all_elements.copy()
    available_set_indices = set(range(problem.n_sets))

    while remaining_elements and available_set_indices:
        best_set_idx = -1
        min_cost = float('inf')

        # 遍历可用集合，选成本最低且能覆盖新元素的
        for set_idx in available_set_indices:
            current_set = problem.sets[set_idx]
            current_cost = problem.costs[set_idx]
            # 必须能覆盖至少一个未覆盖元素
            if len(current_set & remaining_elements) == 0:
                continue
            
            if current_cost < min_cost:
                min_cost = current_cost
                best_set_idx = set_idx

        if best_set_idx == -1:
            break

        # 更新解决方案
        solution.selected_sets.append(best_set_idx)
        solution.total_cost += problem.costs[best_set_idx]
        selected_set = problem.sets[best_set_idx]
        solution.covered_elements.update(selected_set)
        remaining_elements -= selected_set
        available_set_indices.remove(best_set_idx)

    return solution


def random_algorithm(problem: SetCoverProblem) -> SetCoverSolution:
    """
    random_algorithm：每次从「能够覆盖未覆盖元素」的集合中随机选择一个
    """
    solution = SetCoverSolution()
    remaining_elements = problem.all_elements.copy()
    available_set_indices = set(range(problem.n_sets))

    while remaining_elements and available_set_indices:
        # 找出所有能够覆盖至少一个未覆盖元素的集合
        candidate_sets = []
        for set_idx in available_set_indices:
            current_set = problem.sets[set_idx]
            if len(current_set & remaining_elements) > 0:
                candidate_sets.append(set_idx)
        
        if not candidate_sets:
            break
        
        # 从候选集合中随机选择一个
        selected_set_idx = random.choice(candidate_sets)

        # 更新解决方案
        solution.selected_sets.append(selected_set_idx)
        solution.total_cost += problem.costs[selected_set_idx]
        selected_set = problem.sets[selected_set_idx]
        solution.covered_elements.update(selected_set)
        remaining_elements -= selected_set
        available_set_indices.remove(selected_set_idx)

    return solution


if __name__ == "__main__":
    # ========== 配置数据文件路径 ==========
    # 替换为你的data文件夹实际路径
    DATA_DIR = "./data"
    # 需要测试的文件列表
    TEST_FILES = [
        "sc_6_1", "sc_9_0", "sc_15_0", "sc_25_0", "sc_27_0", "sc_45_0", "sc_81_0",
        "sc_135_0", "sc_157_0", "sc_192_0", "sc_243_0", "sc_330_0", "sc_405_0", "sc_448_0",
        "sc_450_0", "sc_450_1", "sc_450_2", "sc_450_3", "sc_450_4", "sc_495_0",
        "sc_500_0", "sc_500_1", "sc_500_2", "sc_500_3", "sc_500_4",
        "sc_595_0", "sc_595_1", "sc_595_2", "sc_595_3", "sc_595_4",
        "sc_715_0", "sc_729_0",
        "sc_760_0", "sc_760_1", "sc_760_2", "sc_760_3", "sc_760_4",
        "sc_945_0", "sc_945_1", "sc_945_2", "sc_945_3", "sc_945_4",
        "sc_1000_0", "sc_1000_1", "sc_1000_2", "sc_1000_3", "sc_1000_4", "sc_1000_5", "sc_1000_6", "sc_1000_7", "sc_1000_8", "sc_1000_9",
        "sc_1000_10", "sc_1000_11", "sc_1000_12", "sc_1000_13", "sc_1000_14",
        "sc_1024_0",
        "sc_1150_0", "sc_1150_1", "sc_1150_2", "sc_1150_3", "sc_1150_4",
        "sc_1165_0", "sc_1215_0",
        "sc_1272_0", "sc_1272_1", "sc_1272_2", "sc_1272_3", "sc_1272_4",
        "sc_1400_0", "sc_1400_1", "sc_1400_2", "sc_1400_3", "sc_1400_4",
        "sc_1534_0", "sc_1534_1", "sc_1534_2", "sc_1534_3", "sc_1534_4",
        "sc_2000_0", "sc_2000_1", "sc_2000_2", "sc_2000_3", "sc_2000_4", "sc_2000_5", "sc_2000_6", "sc_2000_7", "sc_2000_8", "sc_2000_9",
        "sc_2187_0", "sc_2241_0", "sc_2304_0",
        "sc_3000_0", "sc_3000_1", "sc_3000_2", "sc_3000_3", "sc_3000_4", "sc_3000_5", "sc_3000_6", "sc_3000_7", "sc_3000_8", "sc_3000_9",
        "sc_3095_0", "sc_3202_0", "sc_3314_0", "sc_3425_0", "sc_3558_0", "sc_3701_0", "sc_3868_0",
        "sc_4000_0", "sc_4000_1", "sc_4000_2", "sc_4000_3", "sc_4000_4", "sc_4000_5", "sc_4000_6", "sc_4000_7", "sc_4000_8", "sc_4000_9",
        "sc_4025_0", "sc_4208_0", "sc_4413_0",
        "sc_5000_0", "sc_5000_1", "sc_5000_2", "sc_5000_3", "sc_5000_4", "sc_5000_5", "sc_5000_6", "sc_5000_7", "sc_5000_8", "sc_5000_9",
        "sc_5120_0",
        "sc_6931_0", "sc_6951_0", "sc_7435_0", "sc_8002_0",
        "sc_8661_0", "sc_8661_1", "sc_9524_0",
        "sc_10000_0", "sc_10000_1", "sc_10000_2", "sc_10000_3", "sc_10000_4", "sc_10000_5", "sc_10000_6", "sc_10000_7", "sc_10000_8",
        "sc_10370_0", "sc_11264_0", "sc_18753_0", "sc_25032_0", "sc_47311_0", "sc_55515_0"
    ]  # 从sc_6_1到sc_55515_0的所有文件，排除了太大的文件
    
    # CSV文件路径
    csv_file = "setcover_results.csv"
    # 检查CSV文件是否存在，不存在则创建并写入表头
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["file_name", "elements_count", "sets_count", 
                         "greedy_001_time", "greedy_001_cost", 
                         "greedy_002_time", "greedy_002_cost", 
                         "greedy_003_time", "greedy_003_cost",
                         "random_algorithm_time", "random_algorithm_cost"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # ========== 读取数据并运行算法 ==========
    try:
        for TEST_FILE in TEST_FILES:
            file_path = os.path.join(DATA_DIR, TEST_FILE)
            print(f"\n======================================")
            print(f"正在处理文件：{TEST_FILE}")
            print(f"======================================")
            
            # 读取数据文件
            print(f"正在读取数据文件：{file_path}")
            problem = read_setcover_file(file_path)
            print(f"数据解析完成：总元素数={problem.n_elements}，总集合数={problem.n_sets}")

            # 运行三个贪心算法
            print("\n===== greedy_001（单位成本效率优先）=====")
            start_time = time.time()
            sol1 = greedy_001(problem)
            time1 = time.time() - start_time
            cost1 = sol1.total_cost
            print(sol1)
            print(f"运行时间：{time1:.4f}秒")

            print("\n===== greedy_002（覆盖元素数优先）=====")
            start_time = time.time()
            sol2 = greedy_002(problem)
            time2 = time.time() - start_time
            cost2 = sol2.total_cost
            print(sol2)
            print(f"运行时间：{time2:.4f}秒")

            print("\n===== greedy_003（成本优先）=====")
            start_time = time.time()
            sol3 = greedy_003(problem)
            time3 = time.time() - start_time
            cost3 = sol3.total_cost
            print(sol3)
            print(f"运行时间：{time3:.4f}秒")
            
            print("\n===== random_algorithm（随机选择）=====")
            start_time = time.time()
            sol4 = random_algorithm(problem)
            time4 = time.time() - start_time
            cost4 = sol4.total_cost
            print(sol4)
            print(f"运行时间：{time4:.4f}秒")
            
            # 构建结果数据
            result = {
                "file_name": TEST_FILE,
                "elements_count": problem.n_elements,
                "sets_count": problem.n_sets,
                "greedy_001_time": time1,
                "greedy_001_cost": cost1,
                "greedy_002_time": time2,
                "greedy_002_cost": cost2,
                "greedy_003_time": time3,
                "greedy_003_cost": cost3,
                "random_algorithm_time": time4,
                "random_algorithm_cost": cost4
            }
            
            # 追加写入CSV文件
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = ["file_name", "elements_count", "sets_count", 
                             "greedy_001_time", "greedy_001_cost", 
                             "greedy_002_time", "greedy_002_cost", 
                             "greedy_003_time", "greedy_003_cost",
                             "random_algorithm_time", "random_algorithm_cost"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)
            
            print(f"\n结果已追加到：{csv_file}")
        
        print(f"\n======================================")
        print(f"所有文件处理完成，结果已保存到：{csv_file}")
        print(f"======================================")

    except Exception as e:
        print(f"执行出错：{e}")