# Set Cover 实验报告（report_newest）
## 分析范围：`GA + Greedy + ILP + NSGA-II`

## 0. 复现信息

- 批次目录：`outputs/experiments/batch_20260226_160846_985493`
- 本次分析目录：`outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2`
- 结果 CSV：`outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/results`
- 图目录：`outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures`
- 分析算法清单校验：`ga`、`greedy_001`、`ilp_pulp`、`moea_nsga2`（无 `hgasa`）

---

## 1. 参数与运行设置

### 1.1 实验规模

- 算法：`greedy_001`、`ga`、`ilp_pulp`、`moea_nsga2`
- 数据类别：8 类（`set_scale_small`、`set_scale_large`、`item_scale_small`、`item_scale_large`、`low_density`、`high_density`、`special_clustered`、`special_hub`）
- 每类样本数：10
- 每样本重复：5（`seeds=[0,11,22,33,44]`）
- 总记录数：`8 × 10 × 5 × 4 = 1600`

### 1.2 数据参数（m, n, density）

| class_id | m(set_count) | n(item_count) | density | pattern |
|---|---:|---:|---:|---|
| set_scale_small | 60~360 | 320 | 0.060 | random |
| set_scale_large | 1000~1800 | 320 | 0.065 | random |
| item_scale_small | 520 | 140~320 | 0.060 | random |
| item_scale_large | 560 | 450~850 | 0.065 | random |
| low_density | 560 | 360 | 0.008~0.025 | random |
| high_density | 620 | 380 | 0.100~0.160 | random |
| special_clustered | 620 | 380 | 0.080 | clustered |
| special_hub | 620 | 380 | 0.080 | hub |

### 1.3 算法参数（批次 manifest）

- `ilp_pulp`：`time_limit_sec=300`
- `ga`：`population_size=80`，`generations=200`，`mutation_rate=0.006`，`init_strategy=hybrid`，早停 `min_gen=140,patience=30`
- `moea_nsga2`：`pop_size=100`，`generations=250`，`repair_probability=0.85`，早停 `min_gen=175,patience=30`
- `greedy_001`：无关键超参

---

## 2. 性能评估指标

本报告采用以下指标（满足“时间 + 质量 + 稳定性/收敛”要求）：

1. 运行时间：`runtime_sec`（越小越好）
2. 解质量：`gap_to_ilp_opt_pct`（相对 ILP 最优差距，越小越好）
3. 稳定性/收敛：
   - `stability_var_mean`（同一实例重复运行方差）
   - `convergence_curves_by_class.png`（收敛曲线）
   - `effective_generations`（实际执行代数）
4. 扩展指标：
   - 95% CI：`ci_runtime_by_class.png`、`ci_gap_ilp_by_class.png`
   - 可扩展性/复杂度：`complexity_loglog_runtime_fit.png` + `complexity_fit.csv`
   - 显著性检验：`significance_algo.csv`（Wilcoxon）

---

## 3. 数据分析与可视化

### 3.1 总体均值结果

| 算法 | runtime_mean(s) | gap_mean(%) | stability_var_mean | 累计运行时长(h) |
|---|---:|---:|---:|---:|
| greedy_001 | 0.046 | 11.591 | 4.823 | 0.01 |
| ga | 13.251 | 2.624 | 2.738 | 1.47 |
| ilp_pulp | 28.589 | 0.000 | 0.000 | 3.18 |
| moea_nsga2 | 83.243 | 1.818 | 1.472 | 9.25 |

结论（总体）：

- **质量排序**：`ilp_pulp` 最优，`moea_nsga2` 与 `ga` 明显好于 `greedy_001`。
- **时间排序**：`greedy_001` 最快，`ga` 次之，`ilp_pulp` 与 `moea_nsga2` 更慢。
- 单目标成本-时间平衡下，`ga` 仍是主线最合理算法。

关键对比指标表：

| 指标 | 数值 | 解读 |
|---|---:|---|
| GA 相对 Greedy 质量提升 | 8.967 个百分点 | `11.591% -> 2.624%` |
| GA 相对 Greedy 时间倍率 | 286.13x | 质量显著提升但付出时间成本 |
| NSGA-II 相对 GA 质量提升 | 0.807 个百分点 | 提升存在但幅度有限 |
| NSGA-II 相对 GA 时间倍率 | 6.28x | 时间代价明显 |
| ILP 贴近 300s 时间墙比例 | 10/400 = 2.5% | 出现在 `item_scale_large`、`set_scale_large` |

### 3.2 运行时间分析

![Runtime by Class](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/bar_runtime_by_class.png)
![Runtime by Axis](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/line_runtime_by_axis.png)

观察：

- `greedy_001` 在所有类别都是稳定毫秒级。
- `ga` 时间较平滑，且在多数类别显著低于 `moea_nsga2`。
- `ilp_pulp` 在 `item_scale_large`、`set_scale_large` 类显著上升。
- `ilp_pulp` 有 `10/400` 条记录接近 300 秒墙（`item_scale_large` 5 条、`set_scale_large` 5 条），但 `solver_status` 记录均为 `Optimal`。

各类运行时间均值表（秒）：

| class_id | greedy_001 | ga | moea_nsga2 | ilp_pulp |
|---|---:|---:|---:|---:|
| high_density | 0.045 | 10.313 | 77.467 | 21.354 |
| low_density | 0.035 | 19.757 | 89.821 | 0.676 |
| set_scale_large | 0.093 | 17.402 | 119.097 | 103.936 |
| set_scale_small | 0.013 | 7.995 | 53.282 | 1.704 |
| item_scale_large | 0.070 | 17.315 | 99.812 | 90.498 |
| item_scale_small | 0.019 | 8.771 | 62.651 | 1.860 |
| special_clustered | 0.024 | 10.032 | 80.293 | 0.099 |
| special_hub | 0.070 | 14.423 | 83.522 | 8.588 |

### 3.3 解质量分析

![Gap by Class](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/bar_gap_ilp_by_class.png)
![Gap by Axis](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/line_gap_ilp_by_axis.png)

关键量化：

- `ga` 相对 `greedy_001` 的平均 gap 改善约 **8.97** 个百分点（11.591% → 2.624%）。
- `moea_nsga2` 的平均 gap 低于 `ga`（1.818% vs 2.624%），但运行时间显著更高。
- `special_clustered` 上 `ga`、`greedy_001`、`moea_nsga2` 对 ILP 均为 0 gap，说明该类在当前参数下偏容易。

各类质量（gap_to_ilp_opt_pct 均值，%）：

| class_id | greedy_001 | ga | moea_nsga2 | ilp_pulp |
|---|---:|---:|---:|---:|
| high_density | 12.491 | 3.568 | 1.530 | 0.000 |
| low_density | 12.217 | 2.195 | 2.614 | 0.000 |
| set_scale_large | 15.424 | 5.217 | 3.479 | 0.000 |
| set_scale_small | 11.848 | 1.289 | 0.882 | 0.000 |
| item_scale_large | 14.330 | 3.238 | 2.466 | 0.000 |
| item_scale_small | 12.286 | 2.634 | 1.078 | 0.000 |
| special_clustered | 0.000 | 0.000 | 0.000 | 0.000 |
| special_hub | 14.135 | 2.855 | 2.495 | 0.000 |

### 3.4 稳定性与收敛速度

![Convergence](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/convergence_curves_by_class.png)
![Box Plot](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/box_quality_runtime.png)

统计要点：

- `ga`：`effective_generations` 均值 `140.69`（400/400 早停）
- `moea_nsga2`：`effective_generations` 均值 `175.80`（400/400 早停）
- `stability_var_mean`：`moea_nsga2(1.472) < ga(2.738) < greedy_001(4.823)`

解释：

- 在当前早停设置下，GA 与 NSGA-II 都在最小代数门槛后不久停止，说明后期改进有限。
- NSGA-II 更稳定，但其运行时间显著增加，适合“质量优先/多目标”而非“低时延”场景。

### 3.5 多维性能剖析与优势区间

#### A) 多维剖析（m / n / density / structure）

- `set_scale_large`（m 增长）对 ILP 时间放大最明显；GA 增长较平滑。
- `item_scale_large`（n 增长）同样让 ILP/NSGA2 负担上升明显。
- `low_density` 与 `high_density` 在质量曲线上表现不同，说明 density 影响搜索难度和剪枝效果。
- 结构类（`special_clustered`/`special_hub`）与随机类趋势不同，体现生成结构对性能的影响。

#### B) 性能边界总结（工程选型）

| 场景目标 | 推荐算法 | 原因 |
|---|---|---|
| 绝对最优成本（离线） | ilp_pulp | 质量基线最强（gap=0） |
| 极致时延（在线） | greedy_001 | 速度远快于其余算法 |
| 单目标平衡（主线） | ga | 相对 greedy 质量显著提升，且时间远低于 NSGA-II |
| 多目标决策（成本+鲁棒） | moea_nsga2 | 应按帕累托前沿选解，不建议只看单成本 |

非 ILP 质量优势区间（按 gap 最小）：

| class_id | 最优非 ILP 算法 | gap_mean(%) |
|---|---|---:|
| high_density | moea_nsga2 | 1.530 |
| low_density | ga | 2.195 |
| set_scale_large | moea_nsga2 | 3.479 |
| set_scale_small | moea_nsga2 | 0.882 |
| item_scale_large | moea_nsga2 | 2.466 |
| item_scale_small | moea_nsga2 | 1.078 |
| special_clustered | ga | 0.000 |
| special_hub | moea_nsga2 | 2.495 |

### 3.6 理论与实验一致性验证

![Complexity Fit](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/complexity_loglog_runtime_fit.png)

`complexity_fit.csv`（axis_matched）要点：

- `ilp_pulp` 斜率：`set_count=2.08`，`item_count=3.85`，`problem_size=2.41`
- `greedy_001` 斜率约 `1.06~1.27`（近线性）
- `ga` 与 `moea_nsga2` 斜率低于 ILP（受固定代数 + 早停约束）

一致性判断：

- 与理论一致：ILP 在规模增长下更敏感，启发式整体更平滑。
- 偏差来源：早停截断、时间墙截断、实例结构差异、实现常数项。

### 3.7 显著性统计（Wilcoxon）

![Sig Heatmap](../outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2/figures/sig_heatmap_item_scale_large.png)

基于 `significance_algo.csv`：

- `ga` vs `greedy_001`：质量上 **7/8 类显著更优**（`special_clustered` 无差异），时间上 `ga` 8/8 类显著更慢。
- `ga` vs `moea_nsga2`：质量上 `moea_nsga2` 在 4/8 类显著更优；时间上 `ga` 8/8 类显著更快。
- `ga` vs `ilp_pulp`：质量上 ILP 7/8 类显著更优；时间差异在 8/8 类显著，其中 GA 仅在 3/8 类更快（`high_density`、`item_scale_large`、`set_scale_large`）。

显著性汇总表（显著类数 / 8）：

| 算法对 | 质量（gap） | 时间（runtime） |
|---|---:|---:|
| ga vs greedy_001 | 7/8 | 8/8 |
| ga vs moea_nsga2 | 4/8 | 8/8 |
| ga vs ilp_pulp | 7/8 | 8/8 |
| moea_nsga2 vs greedy_001 | 7/8 | 8/8 |

---

## 4. 总结与展望

### 4.1 总结与分析

1. 本轮四算法分析结论清晰：  
   - `ilp_pulp` 保持最优质量；  
   - `greedy_001` 保持最快速度；  
   - `ga` 是当前单目标性价比最优解；  
   - `moea_nsga2` 适合多目标，不适合作为“低时延单目标”默认方案。  
2. 当前数据已能形成“不同类型数据对算法表现有影响”的分析框架，但若要更强统计结论，可继续扩大难类样本和范围。

### 4.2 改进与验证思路

- 主线报告建议固定为 `ilp_pulp + greedy_001 + ga`，NSGA-II 作为扩展章节（帕累托解读）。
- 后续增强：
  - 扩大 `set_scale_large / item_scale_large / high_density` 区间，增强性能边界分化；
  - 增加每类样本数（建议 20）并保持 repeats=5；
  - 补齐鲁棒性“删减集合后可行率”统计（当前 summary 文件为空）；
  - 对 NSGA-II 使用“阈值筛选 + 成本最小”口径选解，而不是只比较单一成本。

---

## 5. 本次分析命令

```bash
uv run analysis.py --runs-root outputs/experiments/batch_20260226_160846_985493/runs --algorithms ilp_pulp,greedy_001,ga,moea_nsga2 --ilp-id ilp_pulp --out-dir outputs/experiments/batch_20260226_160846_985493/runs/analysis_ga_greedy_ilp_nsga2
```
