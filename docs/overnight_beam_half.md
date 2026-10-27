# 2026-10-26 通宵 beam 半尺寸训练

2026-10-27 已结束：可恢复模型完成 76,200 轮，最终留出 0/128；最新模型次日补测两个分支也均为 0/32。**效果验收未通过。** 详见[验收报告](overnight_beam_half_results.md)。以下保留启动时的设定与操作说明。

已按用户授权启动，无需进一步确认。目标固定为 `beam_assembly_step1_half`，机器人仍为原 KUKA + SHARPA，实际物理零件及夹具均为项目发布资产的 1/2。

## 当前运行

- 有效运行 ID：`20261026_overnight_beam_v2`。
- 监控进程 PID：94502；防休眠包装进程 PID：94500。PID 是本次启动记录，不应在其他时间不核对就使用。
- 北京时间 2026-10-26 20:46 左右启动，上限 12 小时，预计最晚约 2026-10-27 08:46 结束；另有累计 120,000 轮上限，先到者生效。
- GPU：RTX 4060 Ti 16 GB；训练 192 环境，六组 SAPG 各 32 环境，rollout 16，minibatch 1024。
- 已核实实际训练更新、GPU 使用、每 500 轮的均匀保存及每 200 轮的滚动恢复文件。

最初的 `20261026_overnight_beam` 启动失败，因为框架解析实验名数字前缀；它完成了 0 轮。已修正为 `0_overnight_beam_half`，旧日志保留，当前有效进程是 v2。

## 训练设定

从官方 `pretrained_assembly/beam_assembly_step1/model.pth` 开始，不从此前未成功的半尺寸微调候选继续。

- 原 actor/critic 权重及归一化均值方差保留。仅观测统计计数设为 1,000,000，value 归一化计数保持原样，避免此前把 value.count=1 也改大的问题。
- 六组动作标准差初始化为 0.40 到 0.10；actor 学习率恒定 5e-6，critic 为 5e-5。
- `finalGoalOnly`，最低高度参考；从回合已见最低高度上升 25 mm 后触发抬升标志。
- 抬升一次性奖励从 300 降为 50，抬升进度系数从 20 降为 5，关键点进度系数从 200 调为 400。目的是减少只获取抬升奖励的偏好；效果尚未验证。
- 训练阶段从目标 XY 观测噪声 0 开始，其他观测/动作延迟与外力关闭。物体起点、朝向、孔位仍按标准配置随机采样，不使用固定桌面课程。
- 训练超时窗口为 20 秒；完成子目标会重置计时，不是严格整回合 20 秒截止。
- 未修改装配、回撤成功容差，未修改质量、惯量或重力，也没有将物体尺寸放大。

## 后台接力与评价

每段最多 5,000 轮。每段结束后，Isaac Sim 训练进程退出释放 GPU，再串行评价最后模型、中间快照和按训练回报保存的 best；先在 seed43 比较分支 50 与 0，再对排名靠前的两个候选用 seed44 确认。标准评价使用半尺寸任务原参数：10 秒可重置窗口、目标 XY 噪声 ±2 mm，所有起点均计分。

训练继续沿最后 checkpoint 推进；按环境任务成功率保存的最佳评价模型另存为 `best_evaluated/`，不自动替换部署权重。若验证达到 25% 或 40%，后续训练目标噪声分别增加到 0.5 mm 或 1 mm。这个门槛是训练课程决策，不是宣布目标达标。

跨段保留优化器、统计及轮数，清除旧仿真状态对应的 observation、RNN 状态和回合缓冲，以新物理场景开始。它是分段继续优化，不是逐帧恢复旧仿真。

结束前预留约 10 分钟，尝试用 seed61–64 对最佳评价候选做最终留出测试，各 32 环境。预算或停止请求可能使最终测试未完成；0 次测试不会被记作 0% 成功。beam 尚无独立几何就位审计，评价成功仍是环境判定。

后台脚本每约 5 秒更新状态，检测日志长时间停滞、进程失败、空间不足和停止请求；初始 OOM 可降到 96 环境，连续失败最多重试两次后结束。保存坏文件会尝试退回其他可读恢复文件。系统休眠和 idle 被临时 inhibitor 阻止，保护随后台程序结束释放；关机、断电或系统崩溃仍会中断。

本会话没有可用的反向唤醒入口，未配置会话自动唤醒，也未调用额外模型/API。结束会写入 `READY_FOR_REVIEW`、`completion.json` 和 `report.md`，由本地监控脚本完成接力与整理。

## 明天查看

运行目录：

```text
outputs/overnight_beam_half/20261026_overnight_beam_v2/
  plan.json                    整体预算与设定
  status.json                  状态、心跳、PID、最后完成段及评价记录
  block_NNN_command.json       每段精确命令
  block_NNN/config.yaml        每段解析后的完整配置
  block_NNN/0_overnight_beam_half/nn/  周期 checkpoint
  block_NNN/0_overnight_beam_half/last/ 滚动恢复文件
  best_evaluated/              最佳评价候选，实验用
  completion.json             结束时评价与统计
  report.md                   结束摘要
  READY_FOR_REVIEW             本地完成标志，不会唤醒聊天
```

日志：`evaluation_runs/overnight_beam_half/20261026_overnight_beam_v2/`。

```bash
# 心跳与当前子进程；epoch 字段是最后已结束训练段的轮数
cat outputs/overnight_beam_half/20261026_overnight_beam_v2/status.json

# 第一段的实时训练轮数；后续段替换为 train_002.log 等
 tail -n 35 evaluation_runs/overnight_beam_half/20261026_overnight_beam_v2/train_001.log

# 请求停止，只作用于这次训练；脚本会结束自己的子进程并保留已有 checkpoint
 touch outputs/overnight_beam_half/20261026_overnight_beam_v2/STOP
```

不要用广泛的 `pkill python` 停止，会误伤其他服务。已有 8046/8047 CPU 轨迹回放保留；先前交互式 8044 的仿真 worker 已释放 GPU，网页父服务保留。

重开新实验应使用新的运行 ID：

```bash
pixi run python tools/run_overnight_beam.py \
  --run-id my_new_beam_night --hours 12 --max-epochs 120000 \
  --block-epochs 5000 --num-envs 192
```

该命令本身前台运行；本次已用独立进程组和防休眠包装启动，退出聊天或终端不依赖前台会话继续存在。新运行前要检查是否还有活跃训练，避免同时占用 GPU。
