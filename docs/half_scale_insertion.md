# 半尺寸插接：运行与首轮实验

多路线微调、独立种子评价与训练预算见[微调实验报告](half_scale_finetuning_report.md)。

已实现 `tight_insertion_half`，使用原始 KUKA + SHARPA，插接件和配合孔均缩小为 1/2。
原始 `tight_insertion` 的资产、目标和默认评价参数保持不变。

## 当前结果（2026-10-26）

后续审计发现旧评估的种子会在创建 player 时被训练配置的 42 覆盖，且并行 SAPG
条件 ID 没有统一。下表为首轮历史记录，不能当作严格的独立留出结果。
新评估已修正这些问题，统一按全部初始环境计分，包含早期失败和未完成回合。
半尺寸任务当前默认使用 `finalGoalOnly` 并启用抬起奖励；原尺寸任务默认目标模式不变。

| 实验 | 配置 | 结果 |
|---|---|---|
| 几何/惯性/任务配置检查 | CPU 单元测试 | 通过 |
| 最小微调检查 | 48 环境，六组 SAPG，2 轮 | 加载 actor/critic、更新和保存通过 |
| 初始微调 | 48 环境，100 轮，学习率初值 1e-5 | 完成，约 7.68 万环境交互步 |
| 原权重零样本 | 半尺寸，seed=43，8 个首回合 | 装配 0/8，观察到抬起 0/8 |
| 100 轮微调后 | 相同半尺寸评估配置及 seed=43 | 装配 0/8，观察到抬起 1/8 |
| 独立几何就位检查 | 微调前/后，各 8 回合 | 均为 0/8 |
| 原尺寸回归 | 原权重，seed=42，1 个首回合 | 装配及回撤成功 |
| 导出模型推理 | CPU，140 维输入 → 29 维动作，LSTM 重置 | 通过，无 Isaac Sim 导入 |

仿真读回质量为 0.0072249998 kg，惯量张量与源 URDF 的 1/32 一致。
100 轮只验证初期训练流程，不代表收敛；8 个回合也不足以估计可靠成功率。
当前不能将此候选称为已完成装配学习或可实机部署的策略。
下一阶段应优先针对小物体的抓取/抬起适配，比较抓取课程或真实几何尺寸课程，再进行更长训练。
`stable_grasp_observed` 是诊断接触事件，不能单凭它推断抓取已稳定可靠。

实验日志和 HTML 诊断报告位于 `evaluation_runs/half_scale/`；模型和解析后配置位于
`outputs/half_scale/policies/tight_insertion_half/`。这些运行产物按仓库规则忽略，不提交进 Git。

## 重新生成资产

```bash
pixi run python tools/generate_half_scale_insertion.py
pixi run python -m unittest discover -s tests -v
```

生成器只读取原始 primitive URDF，写入 `assets/urdf/peg_in_hole/half_scale/`。
长度和 COM 乘 0.5，质量乘 0.125，惯量各项乘 0.03125，并记录源文件哈希。
质量采用同密度假设，打印实物后需要重新测量。

## 用官方原权重评估半尺寸任务

```bash
pixi run python evaluation/eval_offline.py \
  --problem tight_insertion_half \
  --checkpoint-path pretrained_assembly/tight_insertion/model.pth \
  --num-envs 8 --steps 700 --seed 43 \
  --diagnostics --diagnostics-dir evaluation_runs/half_scale/baseline
```

评估器自动选择独立的 `Isaacsimenvs-PreciseAssemblyHalf-Direct-v0` 配置，
目标接近容差为 5 mm，回撤容差为 2.5 mm；对应关键点判定仍会乘 keypoint_scale。
这两个训练/策略判定阈值不是机械装配间隙。
原尺寸任务仍使用原 10 mm / 5 mm 评价参数；命令行明确给定的容差优先。

半尺寸离线评估还执行独立 rigid-box 几何检查：与最终位姿相差不超过 1 mm、2°，
杆端高度误差不超过 1 mm、碰撞盒相交不超过 0.05 mm，连续满足 10 个采样控制状态。
结果字段为 `geometry_seated_10_steps_rate_all_envs`，分母是所有本批环境，
详细结果保存为 `geometry_audit.json`。状态在步进前采样，避免终止后的自动 reset 替换物体位姿。
该检查不验证真实接触力、摩擦稳定、弹性变形或制造公差。

## 重现短程微调

在本机可访问 GPU 的终端运行；同一 GPU 同时只启动一个 Isaac Sim worker。

```bash
OMNI_KIT_ACCEPT_EULA=YES pixi run python isaacsimenvs/train.py \
  --task Isaacsimenvs-PreciseAssemblyHalf-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --checkpoint pretrained_assembly/tight_insertion/model.pth \
  --checkpoint_load_mode weights --headless \
  env.precise_assembly.goal_mode=preInsertAndFinal \
  env.precise_assembly.force_lift_reward_active=false \
  hydra.run.dir=outputs/half_scale/finetune_run
```

默认 48 环境、每组 8 环境，保留官方权重六组 SAPG 条件参数；actor 和 critic 的
minibatch 均为 256，horizon 为 16。实际 actor/critic 权重加载及优化已在本机跑通。
该配置保留原传感器噪声、世界工作空间、控制周期和原始动作映射，
只调整任务几何、关键点固定评价尺寸、部分长度门槛及单环境规模所需的物理缓冲区。

训练目录会生成 `config.yaml`、`metadata.json` 和 `0_half_insertion_sapg/nn/*.pth`。
模型打包时必须同时保存配置和元数据，明确选择最终轮次或按什么验证指标选择 checkpoint；
训练奖励最高不等于物理装配成功率最高。

## 查看本次实验候选

```bash
pixi run python evaluation/eval_isaacsim.py \
  --policies-dir outputs/half_scale/policies \
  --problem tight_insertion_half --deterministic --port 8044
```

打开 http://localhost:8044，确认 Problem / Policy 都是 `tight_insertion_half`，
点击 **Load / reload env**，就绪后点击 **Run Episode**。
此处演示的是短程微调后的实验候选，尚未学会完成装配。
也可以将 `--policies-dir` 换成 `--checkpoint-path pretrained_assembly/tight_insertion/model.pth`
来观察原权重操纵小零件的基线。

## 实现位置

- 对象/任务注册：`evaluation/problems/peg/objects.py`、`evaluation/problems/__init__.py`。
- 半尺寸物理/奖励配置：`isaacsimenvs/cfg/task/PreciseAssemblyHalf.yaml`。
- 兼容微调配置：`isaacsimenvs/cfg/train/PreciseAssemblyHalfSAPG.yaml`。
- 原始 box 几何独立审计：`evaluation/peg_insertion_geometry.py`。
- CPU 部署配置加载已与 `isaacsimenvs` 解耦；硬件感知与控制接口仍需另行实现。
- 修复训练首次创建 best checkpoint 链接时，对不存在旧链接进行约 31 秒无效重试的问题。

## 扩展实验入口

```bash
# 保留六组 SAPG；默认 finalGoalOnly，显式启用抬起奖励
pixi run python tools/run_half_scale_finetune.py \
  --checkpoint pretrained_assembly/tight_insertion/model.pth \
  --name my_final_only_run --epochs 1500

# 归一化及探索先验实验：不修改几何或成功门槛
pixi run python tools/prepare_finetune_checkpoint.py \
  --checkpoint pretrained_assembly/tight_insertion/model.pth \
  --output outputs/my_initializer.pth
```

`--phase adaptation` 关闭训练的观测/动作延迟与外扰，保留真实几何和重力。
`--phase robust` 使用任务原始随机化；两者不能混为已验证的实机鲁棒性。
`--start-curriculum table_fixed` 仅在训练时使用无初始穿透的桌面固定姿态和分离的孔位，
标准评估仍使用随机起点及原孔位范围。

新增独立备选任务：

- `tight_insertion_half_original_mass`：几何 1/2，质量维持原始 57.8 g，惯量为原始 1/4。
- `beam_assembly_step1_half`：beam 零件及夹具均为 1/2，质量和惯量按同密度缩放。
- `tight_insertion_075`：真实 0.75 尺寸的中间课程；其结果不能算作半尺寸目标成功率。

运行记录位于 `evaluation_runs/half_scale_round2/`。评估可显式使用
`--policy-coefficient-id 0|10|20|30|40|50` 选择模型中实际存在的分支；该 ID 必须随模型元数据保存，
并同步用于部署的 `RlPlayer(policy_coefficient_id=...)`。
用 `tools/summarize_finetuning.py` 汇总同一候选的不同种子，会同时给出点估计和 Wilson 95% 区间。
