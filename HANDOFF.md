# 当前交付与历史实验交接

## 2026-11-13：5090 远端训练准备

当前交付分支为 `experiment/beam050-5090-official-budget-20261113`。
用户已明确本轮从官方 Stage-1 Play 直接微调 0.5 倍 beam，
以官方 beam 权重记录的 2,608,791,552 环境步为预算。
默认 1536 环境、六组 SAPG、106152 epochs；调整并行数时自动保持总预算。
安装、短测、启动、监控、停止、续训和验收见 `docs/beam_5090_training.md`，
统一 shell 入口为 `tools/train_beam_5090.sh`。
本机 GPU 短测与优化器续训验证通过，未在远端 5090 上实测，未启动完整预算训练。

以下保留此前 0.6→0.5 渐进研究的历史上下文，避免与本次直接微调路线混用。

## 2026-11-02：0.6 → 0.5 尺度渐进训练研究

当时分支：`experiment/beam060to050-study-20261102`。
实现基底：已验证的渐进训练交付树 `8c66541`；本分支仅新增本交接记录，尚未启动新实验。

## 当前实验已归档

官方 Play 直接微调的两轮实验已提交到本地分支
`archive/direct-play-beam-20261102`，提交 `037c522`。
其中保存训练器、配置、测试、官方代码定位、验收文档，以及
`archives/2026-11-02_direct-play-beam/` 下的状态快照、解析配置、命令与原始文件 SHA256 清单。
可用 `git show archive/direct-play-beam-20261102:archives/2026-11-02_direct-play-beam/README.md` 查阅。

两轮均从官方 Stage-1 Play 出发，96 环境、3000 epochs：

| 模型 | 独立集完整成功（含撤离） |
|---|---:|
| Play → 0.5 | 1/256（0.39%） |
| Play → 1.0 | 55/256（21.48%） |
| 官方 1.0 beam 成品参照 | 224/256（87.50%） |

结论是本地缩减配方尚未恢复官方训练成果，不能认定其配置可靠或将失败仅归因于尺度。
官方历史完整配置仍有不确定性；本地样本量约为官方发布 beam 权重记录的 1/566。
此配方不自动成为下一轮渐进实验默认值。

原始输出、全部 checkpoint 和验收报告继续保留在
`outputs/beam_from_play/20261102_direct050_v1/`、`20261102_direct100_v1/` 及对应 smoke 目录。
所有四轮均已结束，没有本次遗留的训练子进程。输出未删除、未迁移，仍被 Git 忽略。
归档分支尚未推送远端。

## 后续研究基线

重点转为 **0.60 → 0.59 → … → 0.50 的渐进训练过程**。

- 训练执行器：`tools/run_beam_fixed_curriculum.py`；公共训练/评测配置：`tools/beam_curriculum_common.py`。
- 尺度生成与注册：`tools/generate_scaled_beam.py`、`evaluation/beam_scaling.py`。
- 既有复现说明：`docs/beam_half_reproduction.md`。
- 0.6 已验收起点：`outputs/progressive_beam/20261027_scale060_via065/accepted.json`。
- 十段历史流程：`outputs/progressive_beam/20261027_fixed060to050_v1/status.json`。
- 最终模型及谱系：该目录的 `best_model.json`，以及 `releases/beam_half.json`。

历史后十段每段 500 epochs，96 环境、六组 SAPG，actor LR 5e-6、critic LR 5e-5，
训练时应用 `EVAL_DR_OFF_OVERRIDES` 关闭对应延迟、噪声和扰动力等随机化，
以 weights 初始化每段并重建优化器。
开发 seeds 101/102 选模，最终 seeds 601–608 独立报告；结果为 53/64 选模、191/256（74.61%）独立评测。
这是**放宽关卡的渐进协议**，不是逐段严格验收；中间段没有额外独立 heldout。

后续实验应从明确的 0.6 checkpoint 和固定开发评测协议出发，记录每段学习变化、失败类型与继承关系。
每次使用新 run-id，不覆盖已归档结果；若调整步长、预算、随机化或学习率，应显式记录，
并保留独立测试集，不把开发集选优分数当成泛化结果。

本次只完成归档和工作区整理，没有预定新实验参数，也没有自动启动训练。
