# Set Cover 单算法参数实验框架 V2

## 快速开始

1. 生成数据（默认 8 类、每类 20）：

```bash
python main.py gen --dataset-id datagen
```

2. 运行单算法实验（默认 `greedy_001`）：

```bash
python main.py run-single --dataset-root outputs/datasets/datagen
```

3. 重绘图：

```bash
python main.py plot --experiment-dir outputs/experiments/<run_id>
```

4. 一键跑多算法并自动横向对比（固定基线参数）：

```bash
python main.py run-multi --dataset-root outputs/datasets/datagen
```

5. 对已有单算法实验目录做离线 compare：

```bash
python main.py compare --run-dirs outputs/experiments/exp_xxx,outputs/experiments/exp_yyy
```

## 算法来源映射

- `core.alg_ilp_pulp` <- `source/ilp/setcover_experiment.py`（PuLP + CBC 建模思路）
- `core.alg_ilp_ortools` <- OR-Tools CP-SAT（0-1 整数模型）
- `core.alg_greedy_001` <- `source/greedy/main.py`（`greedy_001` 核心评分）
- `core.alg_moea_nsga2` <- `source/cuttingedge/MOEAs.m`（借鉴非支配排序 + 拥挤距离思想，Python重写）

## 运行算法示例

- Greedy：

```bash
python main.py run-single --dataset-root outputs/datasets/datagen --algorithm-id greedy_001 --algorithm-module core.alg_greedy_001
```

- ILP（需要 `pulp`）：

```bash
python main.py run-single --dataset-root outputs/datasets/datagen --algorithm-id ilp_pulp --algorithm-module core.alg_ilp_pulp --base-params '{"time_limit_sec":60,"msg":0}'
```

- ILP（OR-Tools，需要 `ortools`）：

```bash
python main.py run-single --dataset-root outputs/datasets/datagen --algorithm-id ilp_ortools --algorithm-module core.alg_ilp_ortools --base-params '{"time_limit_sec":60,"msg":0,"num_search_workers":8}'
```

- MOEA (NSGA-II)：

```bash
python main.py run-single --dataset-root outputs/datasets/datagen --algorithm-id moea_nsga2 --algorithm-module core.alg_moea_nsga2 --base-params '{"pop_size":80,"generations":100,"crossover_rate":0.9}'
```

## 多算法横向比较（固定参数）

- `run-multi` 和 `compare` 只做“同数据下不同算法”比较，不做参数调优筛选。
- 每个输入 run 需为单参数组（仅一个 `param_signature`）；否则 compare 会报错。
- 类别拆分折线图横轴固定映射：
  - `set_scale_small` / `set_scale_large` -> `set_count`
  - `item_scale_small` / `item_scale_large` -> `item_count`
  - `low_density` / `high_density` -> `density`
  - `special_clustered` / `special_hub` -> `sample_id`
- `special_clustered` 和 `special_hub` 在 compare 中会额外输出“合并图”（同一张图两个子图）便于并排观察。

## NSGA 前沿输出（单算法）

- 当算法为 `core.alg_moea_nsga2` 时，`runs.csv` 会额外包含：
  - `pareto_size`
  - `front_cost_span`
  - `front_hv`
  - `front_hv_norm`
  - `front_feasible_ratio`
  - `front_points_json`
- `plot`/`run-single --with-plots` 会额外输出：
  - `figures/pareto_front_by_class.png`
  - `figures/bar_class_pareto_size.png`
  - `figures/bar_class_front_hv_norm.png`

## 默认 8 类数据（分维控制）

- `set_scale_small`：固定 `item_count`，只变 `set_count`
- `set_scale_large`：固定 `item_count`，只变 `set_count`
- `item_scale_small`：固定 `set_count`，只变 `item_count`
- `item_scale_large`：固定 `set_count`，只变 `item_count`
- `low_density`：固定 `set_count` 与 `item_count`，只变 `density`（低）
- `high_density`：固定 `set_count` 与 `item_count`，只变 `density`（高）
- `special_clustered`：结构类（`clustered`），固定 `set_count`/`item_count`/`density`
- `special_hub`：结构类（`hub`），固定 `set_count`/`item_count`/`density`
- 成本字段：默认全部为整数
- 生成顺序：样本按 `sample_id` 升序组织，区间变量采用等间隔均匀采样
- 复杂度拟合：`analysis.py` 会按“横轴匹配类”做 log-log 回归（`set_count` 用 set-scale 类，`item_count` 用 item-scale 类，`problem_size` 用两类规模类），减少不同实验维度混合造成的斜率偏差

## 输出目录

- 数据：`outputs/datasets/<dataset_id>/...`
- 实验：`outputs/experiments/<run_id>/`
  - `results/runs.csv`
  - `results/summary_by_param.csv`
  - `results/summary_by_class.csv`
  - `results/complexity_fit.csv`
  - `results/significance.csv`
  - `figures/*.png`
- 批量多算法：`outputs/experiments/batch_<ts>/`
  - `manifest.csv`
  - `runs/<algorithm_id>/exp_<algorithm_id>_<ts>/...`（按算法分目录，便于定位）
  - `compare/results/runs_merged.csv`
  - `compare/results/summary_algo_class.csv`
  - `compare/results/summary_algo_class_axis.csv`
  - `compare/results/significance_algo.csv`
  - `compare/results/manifest.csv`
  - `compare/figures/compare_bar_runtime_by_class.png`
  - `compare/figures/compare_bar_objective_by_class.png`
  - `compare/figures/compare_bar_gap_by_class.png`
  - `compare/figures/compare_line_<class_id>_{runtime|objective|gap}.png`
  - `compare/figures/compare_line_special_combined_{runtime|objective|gap}.png`
