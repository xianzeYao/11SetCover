import numpy as np
import matplotlib.pyplot as plt

# ===================================== 1. 参数设置=====================================
# Set Cover问题参数
M = 20  # 总元素数
N = 30  # 总集合数
p = 0.4  # 每个集合包含元素的概率
cost_range = [1, 10]  # 集合成本范围

# NSGA-II算法参数
pop_size = 80  # 种群规模
max_iter = 100  # 最大迭代次数
pc = 0.8  # 交叉概率
pm = 0.1  # 变异概率
dim = N  # 决策变量维度=集合数，0=不选，1=选

# 随机种子
np.random.seed(42)

# ===================================== 2. 构造Set Cover数据集=====================================
# 生成集合-元素关联矩阵S: N×M，0/1表示是否包含
S = np.random.random((N, M)) < p
# 剔除全覆盖的集合
full_cover_idx = np.where(np.sum(S, axis=1) == M)[0]
S = np.delete(S, full_cover_idx, axis=0)
# 更新集合数和维度
N = S.shape[0]
dim = N
# 生成集合成本
cover_num = np.sum(S, axis=1)
cost = cost_range[0] + (cost_range[1] - cost_range[0]) * (cover_num / np.max(cover_num))
cost = np.round(cost).astype(int)  # 成本取整

# ===================================== 3. 定义目标函数：f1=总覆盖成本，f2=未覆盖元素数 =====================================
def cal_obj(pop, S, cost, M):
    """
    计算目标函数值
    :param pop: 种群矩阵 (Nind, dim)，Nind=种群数，dim=集合数
    :param S: 集合-元素关联矩阵 (N, M)
    :param cost: 集合成本数组 (N,)
    :param M: 总元素数
    :return: obj: 目标函数矩阵 (Nind, 2)，[:,0]=f1总成本，[:,1]=f2未覆盖数
    """
    Nind = pop.shape[0]
    obj = np.zeros((Nind, 2))
    for i in range(Nind):
        select = pop[i, :]  # 单个解的0-1选择向量
        obj[i, 0] = np.dot(select, cost)  # f1：总覆盖成本
        cover_flag = np.dot(select, S)    # 元素被覆盖的次数（≥1表示覆盖）
        obj[i, 1] = np.sum(cover_flag < 1)# f2：未覆盖元素数
    return obj

# ===================================== 4. NSGA-II核心算子=====================================
def non_dominated_sort(obj):
    """
    非支配排序：计算每个解的支配等级和拥挤度
    :param obj: 目标函数矩阵 (Nind, 2)
    :return: rank: 支配等级数组 (Nind,)，等级越小越优；cd: 拥挤度数组 (Nind,)
    """
    Nind = obj.shape[0]
    rank = np.ones(Nind, dtype=int)  # 初始等级为1（最优）
    cd = np.zeros(Nind)              # 初始拥挤度为0
    S = [[] for _ in range(Nind)]    # 每个解支配的解的索引列表
    n = np.zeros(Nind, dtype=int)    # 每个解被支配的解的数量

    # 第一步：计算支配关系
    for i in range(Nind):
        for j in range(Nind):
            if i != j:
                # 解i支配解j：f1和f2均更小，且不同时相等
                i_dom_j = (obj[i, 0] <= obj[j, 0]) and (obj[i, 1] <= obj[j, 1])
                not_eq = not (obj[i, 0] == obj[j, 0] and obj[i, 1] == obj[j, 1])
                if i_dom_j and not_eq:
                    S[i].append(j)
                # 解j支配解i
                j_dom_i = (obj[j, 0] <= obj[i, 0]) and (obj[j, 1] <= obj[i, 1])
                if j_dom_i and not_eq:
                    n[i] += 1
        # 被支配数为0的解，等级为1
        if n[i] == 0:
            rank[i] = 1

    # 第二步：分层排序（计算所有解的支配等级）
    k = 1
    while np.any(rank == k):
        Fk = np.where(rank == k)[0]  # 第k层的所有解索引
        # 更新下一层的支配等级
        for i in Fk:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    rank[j] = k + 1
        # 计算第k层的拥挤度
        if len(Fk) > 2:
            f1 = obj[Fk, 0]
            f2 = obj[Fk, 1]
            # 按f1、f2排序的索引
            idx1 = np.argsort(f1)
            idx2 = np.argsort(f2)
            # 边界解的拥挤度设为无穷大（保证被保留）
            cd[Fk[idx1[0]]] = np.inf
            cd[Fk[idx1[-1]]] = np.inf
            cd[Fk[idx2[0]]] = np.inf
            cd[Fk[idx2[-1]]] = np.inf
            # 计算中间解的拥挤度
            f1_range = np.max(f1) - np.min(f1)
            f2_range = np.max(f2) - np.min(f2)
            for i in range(1, len(Fk)-1):
                if f1_range != 0:
                    cd[Fk[idx1[i]]] += (f1[idx1[i+1]] - f1[idx1[i-1]]) / f1_range
                if f2_range != 0:
                    cd[Fk[idx2[i]]] += (f2[idx2[i+1]] - f2[idx2[i-1]]) / f2_range
        else:
            # 少于2个解，拥挤度设为无穷大
            cd[Fk] = np.inf
        k += 1
    return rank, cd

def selection(pop, rank, cd, pop_size):
    """
    锦标赛选择：基于支配等级+拥挤度，每次随机选2个解竞争
    :param pop: 种群矩阵 (Nind, dim)
    :param rank: 支配等级数组 (Nind,)
    :param cd: 拥挤度数组 (Nind,)
    :param pop_size: 待生成的父代种群规模
    :return: parent: 父代种群矩阵 (pop_size, dim)
    """
    Nind, dim = pop.shape
    parent = np.zeros((pop_size, dim), dtype=int)
    for i in range(pop_size):
        # 随机选2个不同的解
        idx = np.random.choice(Nind, 2, replace=False)
        idx1, idx2 = idx[0], idx[1]
        # 等级低的胜出；等级相同，拥挤度大的胜出
        if rank[idx1] < rank[idx2]:
            parent[i, :] = pop[idx1, :]
        elif rank[idx1] > rank[idx2]:
            parent[i, :] = pop[idx2, :]
        else:
            if cd[idx1] > cd[idx2]:
                parent[i, :] = pop[idx1, :]
            else:
                parent[i, :] = pop[idx2, :]
    return parent

def crossover(parent, pc):
    """
    二进制单点交叉：适配0-1离散变量
    :param parent: 父代种群矩阵 (pop_size, dim)
    :param pc: 交叉概率
    :return: offspring: 交叉后的子代种群 (pop_size, dim)
    """
    Nind, dim = parent.shape
    offspring = parent.copy()
    # 两两配对交叉，步长为2
    for i in range(0, Nind-1, 2):
        if np.random.random() < pc:
            # 随机选择交叉点（1~dim-1，对应Python的0~dim-2）
            cross_pt = np.random.randint(1, dim)
            # 交换交叉点后的基因
            offspring[i, cross_pt:] = parent[i+1, cross_pt:]
            offspring[i+1, cross_pt:] = parent[i, cross_pt:]
    return offspring

def mutation(offspring, pm):
    """
    位变异：0变1，1变0，适配0-1离散变量
    :param offspring: 子代种群矩阵 (pop_size, dim)
    :param pm: 变异概率
    :return: offspring: 变异后的子代种群 (pop_size, dim)
    """
    Nind, dim = offspring.shape
    # 生成变异掩码：小于pm的位置需要变异
    mut_mask = np.random.random((Nind, dim)) < pm
    # 0/1翻转
    offspring[mut_mask] = 1 - offspring[mut_mask]
    return offspring

# ===================================== 5. 算法主循环 =====================================
# 初始化种群：0-1随机矩阵
pop = np.random.randint(0, 2, size=(pop_size, dim))

# 迭代过程
for iter in range(max_iter):
    # 计算目标函数
    obj = cal_obj(pop, S, cost, M)
    # 非支配排序
    rank, cd = non_dominated_sort(obj)
    # 选择+交叉+变异
    parent = selection(pop, rank, cd, pop_size)
    offspring = crossover(parent, pc)
    offspring = mutation(offspring, pm)
    # 合并父代和子代
    pop_all = np.vstack((pop, offspring))
    # 重新计算目标函数并排序
    obj_all = cal_obj(pop_all, S, cost, M)
    rank_all, cd_all = non_dominated_sort(obj_all)
    # 按「等级升序、拥挤度降序」筛选新种群（取前pop_size个）
    # 先按rank排序，再按-cd排序（降序）
    sort_idx = np.lexsort((-cd_all, rank_all))
    pop = pop_all[sort_idx[:pop_size], :]

    # 每20次迭代打印进度（与Matlab一致）
    if (iter + 1) % 20 == 0:
        min_cost = np.min(obj_all[:, 0])
        min_uncover = np.min(obj_all[:, 1])
        print(f'迭代次数：{iter+1}，当前最优解成本：{min_cost:.0f}，未覆盖数：{min_uncover:.0f}')

# ===================================== 6. 结果分析与可视化=====================================
# 计算最终种群的目标函数
obj_final = cal_obj(pop, S, cost, M)
# 非支配排序，筛选帕累托非支配解（等级为1的解）
rank_final, _ = non_dominated_sort(obj_final)
pareto_idx = np.where(rank_final == 1)[0]
pareto_obj = obj_final[pareto_idx, :]
# 去重
pareto_obj = np.unique(pareto_obj, axis=0)

# 绘制帕累托前沿图
plt.rcParams['figure.facecolor'] = 'white'  # 画布背景为白色
plt.figure(figsize=(8, 6))
# 绘制帕累托非支配解
plt.scatter(pareto_obj[:, 1], pareto_obj[:, 0], s=50, c='red', marker='*', label='帕累托非支配解')
# 坐标轴与标题设置
plt.xlabel('未覆盖元素数（f2，越小越好）', fontsize=12)
plt.ylabel('总覆盖成本（f1，越小越好）', fontsize=12)
plt.title('多目标Set Cover帕累托前沿（NSGA-II）', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=10)
plt.box(True)
plt.show()

# 输出核心结果
print('='*50 + ' 验证结果 ' + '='*50)
print(f'帕累托非支配解的数量：{pareto_obj.shape[0]}')
# 最小成本对应的未覆盖数
min_cost_val = np.min(pareto_obj[:, 0])
min_cost_uncover = pareto_obj[np.where(pareto_obj[:, 0] == min_cost_val)[0][0], 1]
print(f'最小成本：{min_cost_val:.0f}，对应未覆盖数：{min_cost_uncover:.0f}')
# 最小未覆盖数对应的成本
min_uncover_val = np.min(pareto_obj[:, 1])
min_uncover_cost = pareto_obj[np.where(pareto_obj[:, 1] == min_uncover_val)[0][0], 0]
print(f'最小未覆盖数：{min_uncover_val:.0f}，对应成本：{min_uncover_cost:.0f}')