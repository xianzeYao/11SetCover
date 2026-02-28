# Set Cover 单算法参数实验框架 V2

## 快速开始

1. 安装依赖：

```bash
uv sync
```

2. 生成数据（推荐当前主配置）：

```bash
uv run main.py gen --config configs/dataset_profiles_datagen2_plusplusplus.yaml --output-root outputs/datasets --dataset-id datagen2_plusplusplus
```

2.1 仅生成 `special_clustered`（20 样本）：

```bash
uv run main.py gen --config configs/dataset_profiles_special_clustered_only.yaml --output-root outputs/datasets --dataset-id datagen_special_clustered_only
```

3. 多算法批量运行：

```bash
uv run main.py run-multi --config configs/experiment_multi.yaml --dataset-root outputs/datasets/datagen2_plusplusplus
```

4. 对 batch 结果做统一分析与作图：

```bash
uv run analysis.py --runs-root outputs/experiments/<batch_id>/runs
```

推荐（只分析指定算法并显式指定 ILP 基线）：

```bash
uv run analysis.py --runs-root outputs/experiments/<batch_id>/runs --algorithms ilp_pulp,greedy_001,ga,moea_nsga2 --ilp-id ilp_pulp
```

## 常用命令

- 仅跑指定算法：

```bash
uv run main.py run-multi --config configs/experiment_multi.yaml --dataset-root outputs/datasets/datagen2_plusplusplus --algorithms ilp_pulp,greedy_001,ga --repeats 5 --seeds 0,11,22,33,44
```

- 后台运行（nohup）：

```bash
nohup uv run main.py run-multi --config configs/experiment_multi.yaml --dataset-root outputs/datasets/datagen2_plusplusplus --algorithms ilp_pulp,greedy_001,ga,hgasa,moea_nsga2 --repeats 5 --seeds 0,11,22,33,44 > run_all.log 2>&1 &
```

- 仅 `special_clustered` 数据集跑全算法（20 样本 × repeat5）：

```bash
nohup uv run main.py run-multi --config configs/experiment_multi.yaml --dataset-root outputs/datasets/datagen_special_clustered_only --algorithms ilp_pulp,greedy_001,ga,hgasa,moea_nsga2 --repeats 5 --seeds 0,11,22,33,44 > run_special_clustered_only.log 2>&1 &
```

## 数据配置文件

- `configs/dataset_profiles.yaml`：默认配置
- `configs/dataset_profiles_datagen2_plus.yaml`：中等加强版
- `configs/dataset_profiles_datagen2_plusplus.yaml`：加强版
- `configs/dataset_profiles_datagen2_plusplusplus.yaml`：当前 stress 版（high/large/special 更强）
- `configs/dataset_profiles_special_clustered_only.yaml`：仅 `special_clustered`，20 样本（用于结构类专项分析）

## 结构化生成参数（新增）

`special_clustered` / `special_hub` 支持以下可选参数（不写则用默认值）：

- `cluster_bias`：clustered 模式下，按簇采样概率（默认 `0.82`）
- `cluster_count_factor`：簇数量系数（默认 `0.5`）
- `hub_bias`：hub 模式下，按 hub 采样概率（默认 `0.78`）
- `hub_ratio`：hub 元素占比（默认 `0.12`）

这些参数在 `core/generator.py` 中生效，便于控制结构强度。

## 分析脚本说明

- `analysis.py` 默认自动识别 ILP 基线（`--ilp-id auto`，优先 `ilp_ortools` / `ilp_pulp`）。
- 支持 `--algorithms` 只分析指定算法（逗号分隔），且该参数启用时必须包含至少一个 ILP 算法。
- 质量排名使用 `gap_to_ilp_opt_pct_mean`，时间排名使用 `runtime_sec_mean`（都越小越好）。
- 会输出按类的质量/时间/加权排名热图、收敛曲线、显著性热图等。
## 算法来源映射

- `core.alg_ilp_pulp` <- `source/ilp/setcover_experiment.py`（PuLP + CBC）
- `core.alg_ilp_ortools` <- OR-Tools CP-SAT（0-1 ILP）
- `core.alg_greedy_001` <- `source/greedy/main.py`
- `core.alg_moea_nsga2` <- `source/cuttingedge/MOEAs.m`（思想借鉴，Python 重写）

## 输出目录

- 数据：`outputs/datasets/<dataset_id>/...`
- 单算法：`outputs/experiments/exp_<algo>_<ts>/`
  - `results/runs.csv`
  - `results/summary_by_param.csv`
  - `results/summary_by_class.csv`
  - `results/complexity_fit.csv`
  - `results/significance.csv`
  - `figures/*.png`
- 多算法 batch：`outputs/experiments/batch_<ts>/`
  - `manifest.csv`
  - `runs/<algorithm_id>/exp_<algorithm_id>_<ts>/...`
  - `compare/results/*`
  - `compare/figures/*`
  - `runs/analysis/results/*`（执行 `analysis.py` 后生成）
    - 包含 `robustness_drop_feasible_detail.csv` / `robustness_drop_feasible_summary.csv`
  - `runs/analysis/figures/*`（执行 `analysis.py` 后生成）
    - 包含 `robust_feasible_rate_drop_by_class.png`
