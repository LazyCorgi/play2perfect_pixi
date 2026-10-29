# Beam step 1：0.5 倍尺度模型交付与复现

本次提交基于 `main`（`a7d0b71`），提供完整任务资产、渐进微调流程、模型清单及本地演示/评测入口。旧的半尺度 peg、失败的桥接/BC 实验和训练日志保留在实验分支与本地输出目录，不属于本次代码交付。

## 环境与交付物

Linux x86-64、支持 CUDA 12.8 的 NVIDIA 驱动、Pixi；原实验使用 RTX 4060 Ti 16 GB。软件版本由原有 `pixi.lock` 固定：Python 3.11、Isaac Lab 2.3.2.post1、Isaac Sim 5.1、PyTorch 2.7.0+cu128。首次安装需要联网和约 20 GB 磁盘空间。脚本设置 `OMNI_KIT_ACCEPT_EULA=YES`，执行者需接受对应软件许可。

代码通过 Git 交付；`beam-half-artifacts.tar.gz` 单独传输，不加入普通 Git。归档含最终 `model.pth`、可选的 0.6 起点、相对路径推理配置和 0.6 验收元数据。包和文件的 SHA256 分别记录在 `releases/beam_half_artifacts.sha256` 与 `releases/beam_half.json`。官方 1.0 权重不依赖此包，完整训练入口会按需使用项目原下载器获取。

## 从官方 1.0 模型完整复现训练

在仓库根目录执行：

```bash
bash tools/reproduce_beam_half.sh train --run-id beam_half_full_v1
```

脚本先 `pixi install --locked`。缺少官方模型时，仅下载 `beam_assembly_step1` 的官方任务微调权重；已有文件则直接校验。起点必须匹配：

```text
pretrained_assembly/beam_assembly_step1/model.pth
be7e9f6c367e76a8f0697a65d3253cc8ccbdbeed83ad2dd21a126df5455481e0
```

整个流程不需要交付的最终模型或 0.6 权重：

```text
官方 1.00 → 0.90 → 0.80 → 0.70 → 0.65 → 0.60
         → 0.59 → 0.58 → 0.57 → 0.56 → 0.55
         → 0.54 → 0.53 → 0.52 → 0.51 → 0.50
```

共 15 段，每段 500 epochs；96 个环境、6 个 SAPG 组、组大小 16；actor LR 5e-6、critic LR 5e-5，均保持原配方。每段只继承模型、归一化和 sigma，重新初始化优化器。保留原奖励，不启用 BC、额外桥接或自动补救。机器人尺寸不变，物体和夹具几何按 s 缩放、质量按 s³、惯量按 s⁵ 缩放。

前五段：在 seeds 101、102 上选 checkpoint，再用 103、104 补充验证；每段另有 8×32 个 heldout 首次 episode，seeds 起点为 `201+(90-percent)*10`。验证/heldout 的基础阈值分别为 `max(0.8, 上一段对应分数−0.05, 117/128−0.10)`；0.9 的最后一项使用 `117/128−0.05`。验证还要求不比同尺寸源模型低 5 个百分点。任一关卡失败即停止。旧任务回归不参与验收。

0.9 保留历史候选 epochs 100、400、500；其余阶段候选为 100、300、500，同分取较早 epoch。每段训练均为 500 epochs，不强制复用历史选中的轮次。

后十段：在 seeds 101、102（64 首次 episodes）上选择最佳 checkpoint，达到 `max(0.50, 同尺寸源模型分数−0.25)` 即进入下一段。中间段没有额外 heldout 或补救；0.5 最后用 seeds 601–608 独立报告 256 次。此处是放宽后的推进协议，不等同于前五段的严格验收。

实时输出同时写入：

```text
outputs/progressive_beam/beam_half_full_v1/status.json
outputs/progressive_beam/beam_half_full_v1/course.log
```

`status.json` 的 `active_stage_status` 指向当前子任务。成功完成的唯一标志为顶层 `status == "completed"`；中止/拒绝/失败均不表示完成。最终模型路径记录在 `best_model.json`，并提供 `best_scale050.pth`。例如：

```bash
pixi run --locked python -c 'import json; print(json.load(open("outputs/progressive_beam/beam_half_full_v1/status.json"))["status"])'
tail -f outputs/progressive_beam/beam_half_full_v1/course.log
# 请求停止（不会自动恢复；再次执行需换 run-id）：
touch outputs/progressive_beam/beam_half_full_v1/STOP
```

目录拒绝覆盖；重复运行使用新的 `--run-id`。完整训练是长时任务，不是单次快速推理。锁文件固定软件，GPU/驱动和仿真非确定性仍可能改变选模轮次与分数，不承诺重训权重逐字节相同。

## 演示和独立评测交付模型

一次命令安装锁定环境、校验/解压权重包并开启演示：

```bash
bash tools/reproduce_beam_half.sh --artifact /absolute/path/beam-half-artifacts.tar.gz demo
```

浏览器打开 `http://localhost:8050`；可加 `--port 8051`。已解包后：

```bash
bash tools/reproduce_beam_half.sh verify
bash tools/reproduce_beam_half.sh demo
bash tools/reproduce_beam_half.sh smoke   # seed 601，32 个首次 episodes
bash tools/reproduce_beam_half.sh eval    # seeds 601–608，256 个首次 episodes
```

评测终端输出同步写入 `evaluation_runs/<run-id>/seed*.log`，汇总为 `results.json`。每个环境只评分首次 episode，包含不稳定起始状态，要求所有 32 环境完成；使用确定性 SAPG branch 50、preInsertAndFinal、XY 观测噪声 ±2 mm、每目标 10 秒可重置窗口、插入/撤离容差 5/2.5 mm。最终指标包含撤离；汇总读取 `task_success_count_all_envs`，不要将只完成装配的 success_rate 当成最终指标。

如果只需从附带的已验收 0.6 起点重跑后十段，可选：

```bash
bash tools/reproduce_beam_half.sh train-from060 --run-id beam_half_from060_v1
```

## 已交付模型的结果与验证范围

原实验模型选模为 53/64（82.81%）；独立报告为 191/256（74.61%），各 seed 成功数为 `25,22,27,26,24,23,25,19`。最终 checkpoint SHA256 为 `e097ac9f1d18ba3ab1da04c2e005857d65e903cbe10fad3a96981f29ea6a4741`。完整谱系、各段选中轮次与分数见 `releases/beam_half.json`。

整理后的 main 基底代码已通过 55 项自动化测试；从官方 1.0 权重以 weights-only 方式完成了 0.9 任务的一个训练 epoch（96 环境），并实际加载交付权重完成 seed 601 的 32 环境评测，结果 25/32，与历史同 seed 一致。整理过程复用了本机已有的锁定 Pixi 环境，尚未在同事机器上执行首次安装，也未重跑全部 7500 epochs。该结果属于仿真任务指标，不表示已完成实机部署验证。

## 后续维护

建议后续开发基于本次清理后的代码分支。可复用的边界为尺寸生成器、任务注册/配置、训练阶段执行器、评测和模型清单；新增实验使用新 run-id 和显式配置，结果继续写入被忽略的输出目录。保留版本化的训练配方和交付模型清单，不将这次实验的绝对路径、失败分支和大模型历史加入主线。

Git 暂存区（stage）里的文件会进入下一次提交，不是归档区；Git stash 只是本地临时保存，也不是共享归档。旧实验应保存在现有实验/归档分支；共享必要结论用摘要和元数据。若同事已有 main，只需获取本次新增提交，不必手工复制整个仓库，但运行仍依赖仓库中原有的机器人、网格和基础代码。
