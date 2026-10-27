# 最新 beam 半尺寸模型的人工核验

2026-10-27 归档时已关闭下述网页和仿真进程，端口不再保证可用。原始轨迹及模型保留；重新查看可启动 `evaluation/browse_seed_scenes.py --catalog <下文对应批次目录> --port 8049`，实时模型使用 `evaluation/eval_isaacsim.py` 加载下列权重。归档入口见[交接文档](../HANDOFF.md)。

本次核验模型是通宵训练**第 76,200 轮**权重，不是 `best_evaluated` 中第 5,000 轮的零分并列候选：

```text
outputs/overnight_beam_half/20261026_overnight_beam_v2/block_016/0_overnight_beam_half/last/model.pth
```

## 1. 看已经完成的标准测试

已启动两套 CPU 回放网页：

- 分支 50：http://localhost:8049
- 分支 0：http://localhost:8050

它们展示 seed65、32 环境批测导出的初始状态和轨迹。网页下方显示 checkpoint 路径、策略分支和目标模式，可据此确认来源。

操作步骤：

1. `Env ID` 选 0 或 1。这次只保存了这两个环境的完整动作轨迹；其余环境可检查初始状态及最终结果。
2. 点击 `Play rollout` 观看；取消勾选后拖动 `Frame (0 = reset)` 可逐帧检查。
3. 用 `Speed` 调整速度，用 `Back to initial state` 返回精确导出的 reset 状态。
4. 红色是实际物体，绿色半透明模型是目标，灰色是夹具。绿色只是目标标记，不是另一个参与碰撞的物体；可关闭 `Show target` 避免遮挡。

这两个网页播放保存的机器人关节、物体和夹具位姿，不重新计算物理或模型推理。网页中的 `FAILURE` 来自同一批次 `episodes.jsonl`，不是根据回放画面重新判定。

两分支均为 0/32；同 seed 同批量提供对应起点，便于比较动作。32 个回合全部完成。成功要求达到环境规定目标保持和回撤条件，单纯拿起或靠近不会计为成功。

## 2. 自己操作最新模型重新运行

实时测试界面：http://localhost:8048

1. 确认 `Problem` 是 `beam_assembly_step1_half`，`Policy` 是 `last`（名称来自权重父目录）。该页面只注册上面这一个最新权重。
2. 确认目标模式为 `finalGoalOnly`，装配/回撤参数分别为 0.005 / 0.0025。
3. 点击 **Load / reload env**，等待 Isaac Sim 加载完成。
4. 点击 **Run Episode** 观察，重置后可继续测试。

页面使用确定性策略、分支 50、单环境。首次点击加载才创建 GPU 仿真 worker。单环境随机采样与 seed65 的 32 环境批测不是相同初态，因此此界面用于观察动作，不用于逐帧复现上述计分。逐帧核验请看 8049/8050；严格批测复现请用下面命令。

## 3. 重跑原计分流程

在项目根目录执行，每个分支一次，串行运行：

```bash
pixi run python evaluation/eval_offline.py \
  --problem beam_assembly_step1_half \
  --checkpoint-path outputs/overnight_beam_half/20261026_overnight_beam_v2/block_016/0_overnight_beam_half/last/model.pth \
  --num-envs 32 --steps 3000 --seed 65 \
  --min-episode-steps 0 --goal-mode finalGoalOnly \
  --policy-coefficient-id 50 --episode-length-s 10 \
  --diagnostics --diagnostics-trace-envs 0,1 \
  --diagnostics-dir evaluation_runs/manual_latest_beam_p50
```

分支 0：把 `--policy-coefficient-id 50` 改为 `0`，并使用不同的输出目录，例如 `evaluation_runs/manual_latest_beam_p0`。如需保存全部 32 条轨迹，把轨迹参数改为 `--diagnostics-trace-envs 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31`。

该测试默认使用确定性动作，10 秒是可重置的超时窗口，3000 是整个批测循环上限；初始状态不稳或未完成也不能从所有起点成功率分母中直接消失。PhysX/GPU 数值可能产生差异，核对时应同时查看导出的初态和逐回合记录。

批测前先释放实时网页的仿真 worker，避免两套仿真争用 GPU。回放页仅占 CPU，可保留。

完成后核对控制台 `RESULT_JSON` 的 `task_success_count_all_envs`、`num_envs` 和 `n_unfinished`，并查输出目录下的 `episodes.jsonl`。逐回合记录按完成顺序写入，必须用 `env_id` 对应回放环境，不能把第几行当作第几个环境。

## 原始记录位置

```text
evaluation_runs/overnight_beam_half/20261026_overnight_beam_v2/
  acceptance_results.json
  acceptance_latest_p50_s65.log
  acceptance_latest_p0_s65.log
  acceptance/latest_p50_s65/20261027_135723/beam_assembly_step1_half/
  acceptance/latest_p0_s65/20261027_135827/beam_assembly_step1_half/
```

每个任务目录包含 `initial_scenes.json`、`metadata.json`、`episodes.jsonl` 和 `traces/`。启动网页的命令和 PID 另记在该日志目录 `acceptance_viewers.json`。
