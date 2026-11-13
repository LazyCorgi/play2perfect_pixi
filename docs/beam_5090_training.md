# RTX 5090：官方样本预算的 Play → 0.5 beam 微调

本次路线明确为 **官方 Stage-1 Play → 0.5 倍 beam step 1，直接微调**。
不是从已微调 beam 或 0.6 权重出发，也不经过尺度课程。
原 0.6→0.5 实验与归档分支不变。

## 固定配方和预算

训练入口 `tools/beam_5090.py`，配方 `isaacsimenvs/cfg/train/PreciseAssemblyBeam050BudgetSAPG.yaml`。
以官方发布 beam step 1 权重记录的 `frame=2,608,791,552` 为训练预算基准。
[论文附录 H/I](https://arxiv.org/html/2606.26428) 报告 A6000、12288 环境、一天微调；
一天是硬件相关耗时，不将 5090 的墙钟 24 小时当作同等训练预算。

| 环境数 | SAPG 每组环境数（六组） | actor/critic minibatch | 总 epochs |
|---:|---:|---:|---:|
| 768 | 128 | 6144 | 212304 |
| **1536（默认）** | **256** | **12288** | **106152** |
| 3072 | 512 | 24576 | 53076 |
| 6144 | 1024 | 49152 | 26538 |
| 12288 | 2048 | 98304 | 13269 |

所有正式配置都满足 `环境数 × horizon 16 × epochs = 2,608,791,552`。
保留六组是为匹配公开 Play 的 sigma `[6,29]`、条件参数 `[6,32]`；
公开默认 block=4096 与发布 beam 权重的六组结构存在历史配置缺口。
因此本次匹配的是**样本预算和网络结构**，不宣称已经逐参数复原官方历史训练。
更小的 batch、并行数会改变更新统计，达到预算也不保证达到官方成功率。

actor/critic 初始 LR 均 1e-4，actor adaptive，KL 阈值 0.016，horizon/sequence 16、mini epochs 2；
训练 seed 42。保持通用官方 SAPG 优化参数及现有半尺度任务随机化，
不使用旧渐进路线的低学习率和 DR-off 覆盖项。
物体/夹具几何按 0.5 缩放，机器人不缩放，插入/撤离容差 5/2.5 mm。
固定 Play SHA256：`4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9`。

默认 1536 是尚待远端实测的保守起点；不是承诺该机器一定能放下。
建议先测试 1536，显存充裕可在**新 run-id** 下测试 3072/6144。
选定环境数后不可在同一实验续训时改动。不要仅照搬 13269 epochs 到更少环境。

## 安装环境

远端使用 Linux x86-64（如 Ubuntu 22.04/24.04）、NVIDIA 驱动和 Pixi。
核对 [Isaac Sim 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)：
Linux 验证驱动为 580.65.06；优先使用适配该卡且经运维验证的 R580 驱动。
不要在 Conda/其他 venv 中混装此环境。首次安装、仿真缓存及训练产物建议预留至少 100 GB SSD，
内存建议 64 GB 或更多；实际峰值由远端 smoke 验证。

若尚未安装 Pixi，按 [Pixi 官方安装说明](https://pixi.sh/latest/installation/) 安装并重新打开终端。

```bash
git clone --branch experiment/beam050-5090-official-budget-20261113 \
  git@github.com:LazyCorgi/play2perfect_pixi.git
cd play2perfect_pixi
# 已有仓库时：git fetch origin，然后切换到上面的分支并 git pull --ff-only
nvidia-smi
bash tools/train_beam_5090.sh install
```

`install` 执行锁定安装、包与 CUDA 检查、Isaac Sim headless 启动及官方 Play 下载。
脚本设置 `OMNI_KIT_ACCEPT_EULA=YES`，执行者需先接受 NVIDIA 对应许可。
已有 Play 文件会由下载器跳过；训练时仍强制校验哈希，错误文件需由操作者另存或移除后重新下载。
不需要传输本机 `outputs/`、旧 checkpoint 或 `.pixi/`。

`pixi.lock` 固定 Python 3.11、Isaac Lab 2.3.2.post1、Isaac Sim 5.1、PyTorch 2.7.0+cu128。
[PyTorch 2.7 官方说明](https://pytorch.org/blog/pytorch-2-7/) 确认 CUDA 12.8 wheel 支持 Blackwell。
`check` 会真实运行 CUDA 运算和反向传播，不只检查 CUDA 是否可见。

### 下载时报 SSL EOF / 无法访问 Hugging Face

若已经看到 CUDA 运算成功及 `Isaac Sim headless boot and project import: OK`，随后在下载时
出现 `SSL: UNEXPECTED_EOF_WHILE_READING`，失败点是模型网络下载，不需要重新安装环境。
旧版 snapshot 下载器会先打印无法访问 Hub，随后误报 `play/model.pth` 不存在。
现下载器按单文件下载、重试网络故障，校验官方哈希后才发布目标模型；已有文件也会校验。

仅重试权重步骤：

```bash
bash tools/train_beam_5090.sh weights --retries 3
```

如果网络要求代理，请在远端设置其实际可达的 `HTTPS_PROXY`/`HTTP_PROXY`；不要照搬另一台机器的
`127.0.0.1` 代理端口。也可以用 `weights --endpoint https://YOUR-SELECTED-HUB-ENDPOINT`，
只指定自己信任且能访问的 Hub 服务，不关闭 TLS 校验。参数依据
[Hugging Face 单文件下载接口](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download)。

完全离线的办法是在已有模型的电脑上，通过 scp/rsync 将 `pretrained_policy/model.pth`
复制到远端仓库相同路径（约 414 MB）；不要误传 `pretrained_assembly/` 下的任务模型。
远端核验：

```bash
mkdir -p pretrained_policy
# 将官方 Stage-1 model.pth 放入该目录后：
echo '4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9  pretrained_policy/model.pth' | sha256sum -c -
bash tools/train_beam_5090.sh smoke --run-id smoke5090_n1536_v1 --num-envs 1536
```

日志中的 Pixi `system-requirements` deprecated 和 Kit extension.toml 警告没有阻止本次安装。
`Failed to create directory .../Documents/Kit/apps/Isaac-Sim/scripts/new_stage` 也未阻止此次 headless
检查成功，但反映该目录权限或 Documents 路径异常；可检查 `ls -ld ~/Documents` 和
`namei -l ~/Documents/Kit/apps/Isaac-Sim/scripts/new_stage`，按实际所有权修复，避免直接 sudo 运行训练。
CPU powersave 提示影响吞吐，不是本次下载失败原因；正式测速前可按机器管理策略调整性能模式。

```bash
bash tools/train_beam_5090.sh check
bash tools/train_beam_5090.sh plan --num-envs 1536
```

## 先短测，再正式训练

先在前台运行相同规模、默认 256 epochs 的 smoke；首次 shader/URDF 构建可能较慢：

```bash
bash tools/train_beam_5090.sh smoke --run-id smoke5090_n1536_v1 --num-envs 1536
bash tools/train_beam_5090.sh status --run-id smoke5090_n1536_v1
```

仅 `status=smoke_completed` 且日志无 PhysX buffer overflow/CUDA OOM 时继续。
同时使用 `watch -n 2 nvidia-smi` 查看显存与利用率。
若 OOM，改为 768 并使用新的 smoke run-id；若 PhysX 报 contact/buffer 不足，应停止，
调整配置并另开实验，不能忽略错误后将结果当有效训练。
默认 smoke 覆盖 4096 控制步，超过六个默认超时窗口；仍不代表长时稳定或成功率验收。
原版 32 epochs 仅 512 步，未覆盖完整默认超时窗口，不足以检验大场景的后期接触峰值。
`--smoke-epochs 512` 可进一步延长压力检查（正式训练预算不变）。

### Contact / Patch buffer overflow

这种错误是 PhysX 预分配接触记录容量不足，即使显存、内存和利用率均较低也会发生。
[PhysX 官方 GPU Memory 说明](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html#gpu-memory)
指出部分缓冲区无法动态增长，溢出的 contacts/constraints/pairs 会被丢弃，影响物理行为。
溢出期间的奖励、接触和状态不能作为可靠训练数据。

早期入口仅扩大 num-envs，却沿用 96 环境任务 YAML 的容量（contact 524288、patch 131072）。
6144 环境实测日志要求至少约 contact 690201、patch 189806，说明旧容量确实不足。
修复后的入口按 num-envs 生成 2 的幂次容量并写进 recipe、训练命令及解析配置。
6144 环境的初始配置为：

| PhysX 字段 | 容量 |
|---|---:|
| gpu_max_rigid_contact_count | 4194304 |
| gpu_max_rigid_patch_count | 1048576 |
| gpu_found_lost_aggregate_pairs_capacity | 1048576 |
| gpu_total_aggregate_pairs_capacity | 2097152 |
| gpu_collision_stack_size | 536870912 bytes |

这些值为已有峰值留出余量，不保证所有后续策略状态都足够。
入口在训练或评测日志发现 `PhysX error:` / buffer overflow 后停止子进程、标记
`status=failed, invalid_physics=true`，即便 Isaac Sim 最终返回 0 也不接受结果；
同一 invalid_physics 运行禁止 resume。

先用旧版本的 STOP 机制停止已有溢出运行，保留日志和权重用于诊断；
不能将其标为正式完成，也不要直接从可能受影响的权重继续做有效实验。
同步修复后，使用新的 run-id 从 Play 重跑更长 smoke，再正式训练：

```bash
# 等旧 run 的 status=stopped 后再启动其他 Isaac Sim 进程。
bash tools/train_beam_5090.sh smoke --run-id smoke5090_n6144_buffers_v2 --num-envs 6144
bash tools/train_beam_5090.sh train --run-id beam050_5090_n6144_buffers_v2 --num-envs 6144
```

未更新入口时的临时修复：停止旧进程，在远端
`isaacsimenvs/cfg/task/PreciseAssemblyBeamScale050.yaml` 的 `sim.physx` 下填入上表五个数值，
再用新 run-id 启动。配置写入日志以便复核。
旧入口没有自动终止保护且 smoke 仍仅 32 epochs，应持续检查正式日志并优先同步入口修复。
增大容量不会改变 timestep、solver 迭代数或训练预算；修改后的随机轨迹仍可能与旧运行不同。
本轮修复已在本机 96 环境、32 epochs 验证显式容量能正确加载并完成 GPU 训练；
79 项项目测试通过，其中覆盖仿真退出码为 0 但日志溢出时仍判失败的回归场景。
未在 5090 上重新实测 6144 环境；远端须用默认 256 epochs 或更长 smoke 确认容量与显存余量。

建议用 tmux 保持 SSH 断线后运行；没有 tmux 时先由系统管理员安装：

```bash
tmux new -s beam050
bash tools/train_beam_5090.sh train --run-id beam050_5090_v1 --num-envs 1536
# Ctrl-b 然后 d：分离；tmux attach -t beam050：重新连接
```

这是长时任务，不能用本机 96 环境的耗时直接推算 5090 速度。
预算约 26.1 亿步：实际 10000/20000/30000 环境步每秒分别约 72.5/36.2/24.2 小时，
另加保存、选模和评测耗时。用 TensorBoard 的吞吐与进度实测估算，勿以 epochs/s 跨环境数比较。

## 监控、停止与续训

```bash
bash tools/train_beam_5090.sh status --run-id beam050_5090_v1
tail -f outputs/beam_5090/beam050_5090_v1/training_000.log
watch -n 2 nvidia-smi
pixi run --locked tensorboard \
  --logdir outputs/beam_5090/beam050_5090_v1/train --host 127.0.0.1 --port 6006
```

本地浏览器访问远端 TensorBoard，可先建立 SSH 隧道：
`ssh -L 6006:127.0.0.1:6006 user@remote-host`，再打开 `http://localhost:6006`。
关注 `info/kl`、actor/critic loss、成功/撤离指标和吞吐；训练 reward 不能代替首次 episode 成功率。

`status.json` 记录 recipe、当前 epoch/frame、预算进度、活动日志、源哈希、选模与独立结果。
epoch/frame 进度来自日志；`completed` 最终还会核验权重内部 frame，确保不是虚报预算。
训练约每 5% 预算保留周期权重，另有每 200 epochs 的滚动 checkpoint（上一个为 `.old`）。
重跑拒绝覆盖，停机最多可能丢失最近 200 epochs；不在有训练时启动第二个 Isaac Sim/demo。
独立工具进程间使用 GPU 锁，外部程序不受该锁约束。

```bash
touch outputs/beam_5090/beam050_5090_v1/STOP
# 等 status=stopped；停止不强制保存当前 epoch，从最近的完整 checkpoint 续训。
rm outputs/beam_5090/beam050_5090_v1/STOP
bash tools/train_beam_5090.sh resume --run-id beam050_5090_v1 --num-envs 1536
```

同一 Git commit、环境数/配方下，续训恢复 epoch/frame、模型、归一化、sigma、actor 和 critic 优化器，
总预算不重新计数；自动选取最新可读的完整 checkpoint。
新仿真进程重置 physics rollout、RNN、旧观测及 episode 累加状态，随机流和轨迹不会逐位连续。
该模式叫 `resume_fresh_rollout`，不是历史物理状态精确恢复。
原始官方 Play 只用于首轮 weights 初始化，不得用 resume 模式从其 epoch 218836 开始。

每次重启有独立 `attempt_XXX/` Hydra 配置、`training_XXX.log`，同一 `train/` 保存权重和 TensorBoard。
若最终训练已完成但评测中断，resume 会跳过训练并复用已完成的评测批次。

## 最终验收

正式训练前 seeds 101/102 各 32 次 Play 零样本评测。
达到完整预算后，从保存权重中取最接近预算 25%、50%、75%、100% 的候选，
在相同开发 seeds 上选优，同分取更早 epoch；选中模型用 seeds 601–608 共 256 次独立首次 episodes。
评测 DR-off、coefficient 50、目标 XY 噪声 ±2 mm、包含不稳定初始样本，要求所有环境完成。
使用 `task_success_count_all_envs`（含撤离），不能把仅装配成功当完整任务成功。

完成标志：`status=completed`、`frames=2608791552`、有 `selected_checkpoint`、`selected_sha256`、
`validation_rate`、`heldout_rate` 和 `heldout_trials=256`。smoke 预算不算正式完成。
`completed` 是流程完成；尚未预设成功率验收门槛。
保存整个 `outputs/beam_5090/<run-id>/` 以交付，不只复制最佳权重。

## 验证范围

本提交在当前 RTX 4060 Ti 上完成 96 环境、32 epochs 的实际 GPU smoke，
最终 frame=49152；随后在新仿真进程从 32 轮续训到 33 轮，frame=50688，
actor/critic 两个优化器的 step 均由 128 增至 132，确认恢复后继续更新。
项目 `tests/` 共 68 项测试通过，另有一项依赖弃用告警。
执行测试时使用 `pixi run --locked python -m pytest tests -q`，避免收集被忽略输出目录内的历史代码快照。
未在目标 RTX 5090 上实测，未执行 26.1 亿步完整训练，也没有重现官方成功率的保证。
历史失败与原尺度交叉检查留存在 `archive/direct-play-beam-20261102`（`037c522`），不将其结果覆盖。
