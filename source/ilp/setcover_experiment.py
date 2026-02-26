import numpy as np
import pulp as pl
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 实例生成函数
# ------------------------------
def generate_instance(n, m, density=None, structured=False):
    """
    生成集合覆盖实例
    :param n: 元素个数
    :param m: 集合个数
    :param density: 每个元素被覆盖的概率（大致），若 structured=True 则忽略
    :param structured: 是否生成结构化实例（每个元素被固定数量的集合覆盖）
    :return: 元素-集合包含关系的列表（每行是集合索引列表）
    """
    if structured:
        k = 3  # 每个元素出现在 k 个集合中
        sets = [[] for _ in range(m)]
        for elem in range(n):
            chosen = np.random.choice(m, size=k, replace=False)
            for s in chosen:
                sets[s].append(elem)
        # 确保每个集合非空
        for s in range(m):
            if not sets[s]:
                sets[s].append(np.random.randint(n))
    else:
        # 随机生成，每个元素以概率 density 被每个集合覆盖
        sets = [[] for _ in range(m)]
        for s in range(m):
            size = np.random.binomial(n, density)
            elems = np.random.choice(n, size=size, replace=False)
            sets[s].extend(elems)
    return sets

# ------------------------------
# 求解函数（增加节点数记录）
# ------------------------------
def solve_set_cover(sets, n, m, time_limit=60):
    """
    使用PuLP的CBC求解器求解集合覆盖整数规划，并尝试获取节点数。
    """
    prob = pl.LpProblem("SetCover", pl.LpMinimize)
    x = {j: pl.LpVariable(f"x_{j}", cat=pl.LpBinary) for j in range(m)}
    prob += pl.lpSum(x[j] for j in range(m))
    for i in range(n):
        prob += pl.lpSum(x[j] for j in range(m) if i in sets[j]) >= 1

    # 记录开始时间和内存
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024

    # 求解
    solver = pl.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
    prob.solve(solver)

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024

    # 获取求解器内部信息（CBC节点数需要从日志提取，此处简化）
    status = pl.LpStatus[prob.status]
    obj_val = pl.value(prob.objective) if prob.objective is not None else None

    # 尝试获取节点数（CBC 通过 solver 的接口可能不直接提供，这里用伪代码）
    # 实际可使用 prob.solverModel 获取底层对象，但 PuLP 封装较深，暂略
    nodes = 0  # 演示用

    return {
        'status': status,
        'objective': obj_val,
        'time': end_time - start_time,
        'memory': mem_after - mem_before,
        'nodes': nodes
    }

# ------------------------------
# 实验参数
# ------------------------------
instance_types = {
    'small': {'n': 10, 'm': 20, 'density': 0.3, 'structured': False, 'num_samples': 20},
    'large': {'n': 50, 'm': 100, 'density': 0.2, 'structured': False, 'num_samples': 20},
    'dense': {'n': 30, 'm': 60, 'density': 0.6, 'structured': False, 'num_samples': 20},
    'sparse': {'n': 30, 'm': 60, 'density': 0.1, 'structured': False, 'num_samples': 20},
    'structured': {'n': 30, 'm': 60, 'density': None, 'structured': True, 'num_samples': 20}
}
repeats = 5

# ------------------------------
# 运行实验
# ------------------------------
results = []
for type_name, params in instance_types.items():
    print(f"Processing {type_name}...")
    n, m = params['n'], params['m']
    density = params.get('density')
    structured = params['structured']
    num_samples = params['num_samples']

    for sample_id in range(num_samples):
        sets = generate_instance(n, m, density, structured)
        for rep in range(repeats):
            res = solve_set_cover(sets, n, m, time_limit=60)
            results.append({
                'type': type_name,
                'sample_id': sample_id,
                'repetition': rep,
                'n': n,
                'm': m,
                'density': density if density is not None else -1,
                'structured': structured,
                'status': res['status'],
                'objective': res['objective'],
                'time': res['time'],
                'memory': res['memory']
            })

df = pd.DataFrame(results)

# ------------------------------
# 数据汇总与可视化
# ------------------------------
# 1. 汇总表
summary = df.groupby('type').agg(
    mean_time=('time', 'mean'),
    std_time=('time', 'std'),
    mean_memory=('memory', 'mean'),
    std_memory=('memory', 'std'),
    optimal_rate=('status', lambda x: (x == 'Optimal').mean())
).reset_index()
print("\n=== 性能汇总 ===")
print(summary)

# 2. 置信区间
for typ in df['type'].unique():
    subset = df[df['type'] == typ]['time']
    mean = subset.mean()
    sem = subset.std() / np.sqrt(len(subset))
    ci = stats.t.interval(0.95, len(subset)-1, loc=mean, scale=sem)
    print(f"{typ}: 时间均值 {mean:.3f}s, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

# 3. 箱线图
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='type', y='time')
plt.title('Running Time by Instance Type')
plt.ylabel('Time (s)')
plt.savefig('time_boxplot.png', dpi=150)
plt.show()

# 4. 密度影响
dense_df = df[df['type'].isin(['dense', 'sparse', 'structured'])]
plt.figure()
sns.boxplot(data=dense_df, x='type', y='time')
plt.title('Comparison: Dense vs Sparse vs Structured')
plt.savefig('dense_sparse_struct.png', dpi=150)
plt.show()

# 5. 可扩展性（固定密度0.2，m=n，从10到40）
scale_n = list(range(10, 41, 5))
scale_times = []
for n in scale_n:
    m = n
    times = []
    for _ in range(10):
        sets = generate_instance(n, m, density=0.2, structured=False)
        res = solve_set_cover(sets, n, m, time_limit=60)
        times.append(res['time'])
    scale_times.append(np.mean(times))
plt.figure()
plt.plot(scale_n, scale_times, marker='o')
plt.xlabel('Number of elements (n=m)')
plt.ylabel('Average time (s)')
plt.title('Scalability with Problem Size')
plt.savefig('scalability.png', dpi=150)
plt.show()

# 6. 密度扫描
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
density_times = []
for d in densities:
    times = []
    for _ in range(20):
        sets = generate_instance(30, 60, density=d, structured=False)
        res = solve_set_cover(sets, 30, 60, time_limit=60)
        times.append(res['time'])
    density_times.append(np.mean(times))
plt.figure()
plt.plot(densities, density_times, marker='s')
plt.xlabel('Density')
plt.ylabel('Average time (s)')
plt.title('Impact of Density on Running Time')
plt.savefig('density_impact.png', dpi=150)
plt.show()

print("实验完成，图表已保存。")