# 半尺寸装配微调实验报告

本报告对应原始 KUKA + SHARPA，装配件和配合机构的线性尺寸均为原来的 1/2。
研究目标是完整装配及回撤成功率高于 40%。本轮有界实验已完成，结果见下表及本机原始 JSON。

**阅读说明：下文任务成功采用环境的关键点达标及回撤判据；抬升奖励标志不是实际抓起率。**
指标公式、12 次训练的逐项改动和 seed42 回放见[详细实验说明](half_scale_experiments_detailed.md)。
新增的 576 个起点、轨迹及初态穿透复查见[种子场景复查](seed_scene_review.md)；其中原尺寸参照也新增了严格几何检查，结果与环境 SUCCESS 不同。

**结果：12 次训练、约 1,229 万环境交互步后，未达到目标。**
筛选出的最佳半尺寸候选为官方 beam 权重，验证集 1/32，独立留出集 0/128
（Wilson 95% 区间 0%–2.9%）；没有得到值得替换部署权重的新模型。
原尺寸插接的两种目标模式均为 28/32，表明当前流程能够复现原任务的较高成功率。

## 评价协议

- 训练 seed=42；候选筛选 seed=43，每个条件 32 个随机起点；最终候选使用未参与筛选的 seed=49、50、51、52，各 32 个起点。
- 分母包括全部起点，早期失败和未完成回合均计失败。主指标同时要求装配成功和回撤完成。
- 半尺寸插接容差为 5 mm、回撤容差 2.5 mm，沿用任务的关键点定义；这不是机构的机械间隙。另用几何审计检查插接件就位，beam 尚未实现相同的独立几何审计。
- 默认超时窗口 10 秒，每完成子目标会清零计时；它不是严格整回合总时限。20 秒窗口条件单列，不能与 10 秒结果直接合并。采用确定性动作、固定 SAPG 条件 ID，评估时关闭域随机化。这里只验证这一仿真分布，不推断真实部署成功率。
- 最终候选仅在 1/2 尺寸任务中选择，0.75 倍和原尺寸任务作为课程、参照，不能算半尺寸达标。

早期评估存在两个已修复的问题：Runner 会用训练配置 seed=42 覆盖评价种子；并行播放器会在环境间分配未经训练的 SAPG 条件 ID。首轮历史 8 回合数据不能视为严格留出结果。本轮统一显式设置实际种子，且所有环境使用同一已训练条件 ID（0、10、20、30、40、50 中之一）。

## 已实施的路线

1. 原插接权重继续训练：保留六组 SAPG，使用 finalGoalOnly，恢复插接阶段原先被掩码屏蔽的抬升奖励，训练 1,500 轮；同时检查 best、last、其他条件 ID 和 preInsertAndFinal。
2. 调整统计先验与探索：保留 actor、critic 及归一化均值方差，把匹配的统计计数统一设为 100,000（并非每项都是降低），动作标准差设置为六组 0.35 到 0.10，恒定学习率 1e-5，再训练 500 轮。计数、探索和学习率方案共同改变，不能把效果归因于单一因素。
3. 质量/惯量：保持半尺寸几何，恢复原质量，惯量按质量乘长度平方匹配；和同密度半尺寸路线对照。
4. 起点课程：先从固定桌面位置学习，再迁移回标准随机起点；最终评价仍使用标准起点。
5. 真实尺寸课程：物理资产先缩至 0.75 倍训练，再迁移至 0.5 倍。不是只缩观测中的包围盒。
6. beam-assembly-step1：缩放装配件与整套配合机构，测试官方装配权重、1,500 轮微调、调整统计先验后继续训练及不同 SAPG 条件 ID。
7. Stage 1 抓取权重：测试零样本和继续微调，避免仅依赖已有插接策略的抓取习惯。
8. 抬升奖励参考：增加可选的回合内最低高度参考。原始物体从空中生成，使用生成高度会使桌面抓取后的正常抬升难以获得奖励；新参考不把初始自由下落当成抬升，要求之后再升高 25 mm。只在明确指定的训练路线启用。
9. 超时窗口与目标序列：对最佳筛选候选测试 20 秒超时窗口；原尺寸原权重单独比较 finalGoalOnly 与 preInsertAndFinal。

所有模型、初始模型来源、解析配置、训练命令、仿真读回质量/惯量及逐回合诊断保存在
`outputs/half_scale_round2/` 和 `evaluation_runs/half_scale_round2/`。这些是本机运行产物，按仓库规则不纳入 Git。

## 训练规模与解释限制

每次使用 96 个并行环境、16 步 rollout，六组各 16 个环境。500 轮约 76.8 万环境交互步，1,500 轮约 230.4 万步；不同路线不能当作同一个策略累积的样本量。GPU 为 RTX 4060 Ti（16 GB），Isaac Sim 实验串行运行。训练仅使用 seed=42，最终多种子评价只衡量该候选在不同初始状态下的表现，不衡量不同训练种子的差异。未执行 robust 阶段。此规模明显小于原项目大并行训练，负结果只表明本机这些有界尝试未奏效，不能证明该任务不可学习。

奖励上涨不等于任务成功。接触诊断中的 `stable_grasp_observed` 也不等于可靠抓持；`object_lifted_observed` 依赖环境内部抬升标志，不能作为独立物理测量。插接几何审计额外要求连续 10 个控制状态满足 1 mm、2° 以及碰撞盒重叠不超过 0.05 mm，但不验证接触力、摩擦稳定和实物公差。

## 复现方法

先生成资产，再选择独立实验名运行。每个实验保留原文件，不覆盖已有目录。

```bash
pixi run python tools/generate_half_scale_insertion.py
pixi run python tools/generate_half_scale_beam.py
pixi run python tools/run_half_scale_finetune.py \
  --problem tight_insertion_half \
  --checkpoint outputs/half_scale_round2/initializers/peg_adapted_priors.pth \
  --name example_minimum --epochs 500 \
  --lr-schedule constant --learning-rate 1e-5 --lift-reference minimum
```

`initializers/*.json` 记录初始化来源；可用 `tools/prepare_finetune_checkpoint.py --help` 重建。
使用新种子评价，主指标读取 `task_success_count_all_envs / num_envs`：

```bash
pixi run python evaluation/eval_offline.py \
  --problem tight_insertion_half --checkpoint-path /path/to/model.pth \
  --goal-mode finalGoalOnly --policy-coefficient-id 50 \
  --num-envs 32 --steps 3000 --min-episode-steps 0 --seed 49 \
  --episode-length-s 10 --diagnostics
```

同一候选的多种子日志可用 `tools/summarize_finetuning.py` 汇总；脚本检查任务、模型、目标模式、条件 ID 和时间上限一致，并给出 Wilson 95% 区间。

## 最终结果

独立留出测试：**0/128（0.0%）**，Wilson 95% 区间 0.0%–2.9%。

候选由验证集选择：`beam_baseline`；任务 `beam_assembly_step1_half`，条件 ID 50，阶段超时窗口 10 秒。

**未达到高于 40% 的目标。**

最终留出种子为 49、50、51、52。早期组合筛选中使用的 45–48 不纳入最终统计；同一验证集比较多种候选会产生选择偏差，因此验证集最高分不作为最终成功率。

| 条件（seed=43） | 任务 | 条件 ID | 超时窗口/秒 | 环境任务成功/32 | 诊断曾见抬升奖励标志/32 | 独立几何就位且任务成功/32 |
|---|---|---:|---:|---:|---:|---:|
| `baseline100_final` | `tight_insertion_half` | 50 | 10 | 0/32 | 0/32 | 0/32 |
| `final1500_final` | `tight_insertion_half` | 50 | 10 | 0/32 | 4/32 | 0/32 |
| `best1500_final` | `tight_insertion_half` | 50 | 10 | 0/32 | 3/32 | 0/32 |
| `final1500_pre` | `tight_insertion_half` | 50 | 10 | 0/32 | 14/32 | 0/32 |
| `mass_baseline` | `tight_insertion_half_original_mass` | 50 | 10 | 0/32 | 1/32 | 0/32 |
| `beam_baseline` | `beam_assembly_step1_half` | 50 | 10 | 1/32 | 17/32 | 未测 |
| `beam1500_final` | `beam_assembly_step1_half` | 50 | 10 | 0/32 | 31/32 | 未测 |
| `beam1500_best` | `beam_assembly_step1_half` | 50 | 10 | 0/32 | 20/32 | 未测 |
| `beam1500_policy0` | `beam_assembly_step1_half` | 0 | 10 | 0/32 | 31/32 | 未测 |
| `peg1500_policy0` | `tight_insertion_half` | 0 | 10 | 0/32 | 8/32 | 0/32 |
| `peg_priors_eval` | `tight_insertion_half` | 50 | 10 | 0/32 | 0/32 | 0/32 |
| `peg_mass_priors_eval` | `tight_insertion_half_original_mass` | 50 | 10 | 0/32 | 5/32 | 0/32 |
| `peg_table_eval` | `tight_insertion_half` | 50 | 10 | 0/32 | 0/32 | 0/32 |
| `peg_table_transfer_eval` | `tight_insertion_half` | 50 | 10 | 0/32 | 0/32 | 0/32 |
| `peg_scale075_eval` | `tight_insertion_075` | 50 | 10 | 0/32 | 13/32 | 0/32 |
| `peg_scale_transfer_eval` | `tight_insertion_half` | 50 | 10 | 0/32 | 2/32 | 0/32 |
| `beam_priors_eval` | `beam_assembly_step1_half` | 50 | 10 | 0/32 | 6/32 | 未测 |
| `beam_source_policy0` | `beam_assembly_step1_half` | 0 | 10 | 1/32 | 18/32 | 未测 |
| `beam_source_policy10` | `beam_assembly_step1_half` | 10 | 10 | 0/32 | 16/32 | 未测 |
| `beam_source_policy20` | `beam_assembly_step1_half` | 20 | 10 | 0/32 | 16/32 | 未测 |
| `beam_source_policy30` | `beam_assembly_step1_half` | 30 | 10 | 0/32 | 20/32 | 未测 |
| `beam_source_policy40` | `beam_assembly_step1_half` | 40 | 10 | 0/32 | 18/32 | 未测 |
| `play_source_policy0` | `tight_insertion_half` | 0 | 10 | 0/32 | 4/32 | 0/32 |
| `play_source_policy50` | `tight_insertion_half` | 50 | 10 | 0/32 | 6/32 | 0/32 |
| `peg_source_final` | `tight_insertion_half` | 50 | 10 | 0/32 | 2/32 | 0/32 |
| `peg_minimum_eval` | `tight_insertion_half` | 50 | 10 | 0/32 | 1/32 | 0/32 |
| `peg_play_eval` | `tight_insertion_half` | 50 | 10 | 0/32 | 5/32 | 0/32 |
| `beam_minimum_eval` | `beam_assembly_step1_half` | 50 | 10 | 0/32 | 6/32 | 未测 |
| `original_final` | `tight_insertion` | 50 | 10 | 28/32 | 9/32 | 未测 |
| `original_pre` | `tight_insertion` | 50 | 10 | 28/32 | 15/32 | 未测 |
| `best_long20` | `beam_assembly_step1_half` | 50 | 20 | 1/32 | 19/32 | 未测 |

逐条件 JSON、checkpoint 路径、逐回合事件及物理参数见本机 `evaluation_runs/half_scale_round2/extension.json`；最终留出汇总为 `final_summary.json`。原尺寸对照及 0.75 倍课程不参与半尺寸达标判断。

## 实际训练预算

| 实验目录 | 任务 | 轮数 | 约交互步数 | 抬升参考 |
|---|---|---:|---:|---|
| `final_lift` | `tight_insertion_half` | 1500 | 2,304,000 | initial |
| `beam_final_lift` | `beam_assembly_step1_half` | 1500 | 2,304,000 | initial |
| `beam_minimum` | `beam_assembly_step1_half` | 500 | 768,000 | minimum |
| `beam_priors` | `beam_assembly_step1_half` | 500 | 768,000 | initial |
| `peg_mass_priors` | `tight_insertion_half_original_mass` | 500 | 768,000 | initial |
| `peg_minimum` | `tight_insertion_half` | 500 | 768,000 | minimum |
| `peg_play` | `tight_insertion_half` | 500 | 768,000 | minimum |
| `peg_priors` | `tight_insertion_half` | 500 | 768,000 | initial |
| `peg_scale075` | `tight_insertion_075` | 500 | 768,000 | initial |
| `peg_scale_transfer` | `tight_insertion_half` | 500 | 768,000 | initial |
| `peg_table` | `tight_insertion_half` | 500 | 768,000 | initial |
| `peg_table_transfer` | `tight_insertion_half` | 500 | 768,000 | initial |

本轮共 12 次训练，跨路线合计约 12,288,000 个环境交互步；这是搜索预算，不是单一最终策略的训练量。

共记录 39 批评价、1,248 个初始环境，其中包括原尺寸参照、课程任务及探索性留出评价，不能把这些不同条件合并计算成功率。

## 失败分析与验证

- 半尺寸插接的 1,500 轮候选为 0/32，抬升标志只有 4/32；桌面固定起点、质量恢复、Stage 1 权重和新抬升参考均未产生装配成功。抓取与搬运适配仍是优先排查环节。
- 半尺寸 beam 的 1,500 轮候选已有 31/32 触发抬升标志，仍是 0/32 装配。抬升标志不是独立物理抓取验证，这一结果也说明不能只优化抬升：稳定搬运、姿态对齐及插入阶段仍未解决。
- 0.75 倍课程的训练成功标志曾短暂达到约 13%，后来降至零；这不是独立评价成功率，也不等于完整回撤成功率。半尺寸继续训练的奖励会上涨，而任务成功仍为零。需要防止继续训练丢失已有行为，不能仅按回合奖励选模型。见[训练曲线（本机产物）](../evaluation_runs/half_scale_round2/reward_vs_success.png)。
- 原尺寸两种目标模式均为 28/32，成功/失败的环境 ID 也完全一致；本批配对结果没有显示目标序列的优势，不足以否定其他场景或随机种子下的差异。半尺寸两种模式均失败，同样无法据此判断谁更好。
- 最佳半尺寸候选把阶段超时窗口从 10 秒延长到 20 秒后仍为 1/32，未观察到改善。原质量恢复对照也未改善；未扫描所有中间质量和惯量，不能概括为质量调整无效。

20 项 CPU 测试通过，28 个修改过的 Python 模块语法检查通过；GPU 训练、保存和评价已完成。
运行时另检查了四个留出种子产生不同初始位姿、原尺寸配对实验初始位姿一致，且最终 128 回合全部结束。
这些检查确认流程可执行，不替代任务成功率验证。

## 结论与后续建议

本轮未达标，保留原部署权重。继续投入时，建议先在半尺寸插接上建立可重复的抓取、抬升和搬运子任务，并收集少量专家成功轨迹；beam 则优先检查抓持后的姿态对齐与插入。随后考虑行为克隆初始化、约束策略偏离原成功行为，再逐步恢复完整装配，而不是继续加长当前奖励配置下的训练。上述建议尚未在本轮验证。

本轮覆盖可在现有接口内实现的继续训练、初始化、探索、质量/惯量、起点课程、尺寸课程、任务切换、奖励参考和时限路线。尚未覆盖专家抓取演示、行为克隆、抓取轨迹重定向、自动课程、大规模多训练种子搜索及控制结构改造，因此不构成“所有强化学习方法均无效”的证明。
