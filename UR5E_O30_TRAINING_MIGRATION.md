# UR5e + Linker O30 训练迁移指引

本文档用于指导后续 Codex 将 Play2Perfect 从原始的 **7-DoF KUKA IIWA +
22-DoF SHARPA** 机器人迁移到 **6-DoF UR5e + 20-DoF Linker O30**。

迁移目标是保留原机器人配置作为可运行的回归基线，同时增加一套独立的
UR5e-O30 配置。不要通过全局替换关节名或直接覆盖 KUKA 常量完成迁移。

## 1. 当前状态与事实

### 1.1 已导入资产

- 机器人 URDF：
  `assets/urdf/ur5e_linker_o30/ur5e_linker_o30_sim.urdf`
- UR5e visual mesh：
  `assets/urdf/ur5e_linker_o30/ur5e_visual_meshes/`
- UR5e collision mesh：
  `assets/urdf/ur5e_linker_o30/ur5e_collision_meshes/`
- O30 只有一套 mesh，visual 和 collision 使用相同模型：
  `assets/urdf/ur5e_linker_o30/o30_collision_meshes/`
- URDF 已经可以被 Isaac 导入并显示。

### 1.2 场景调姿和存储工具

项目内已有统一的场景状态和两个工具：

- `isaacsimenvs/cfg/robot/ur5e_o30_scene.json`
  - 是机器人基座、26 个关节、桌子、孔和插销位姿的唯一标定数据源；
  - 位姿包含 `position` 和 `rotation_wxyz`；
  - 后续训练环境应读取这个文件，不要复制另一套数值。
- `tools/ur5e_o30_scene_editor.py`
  - 使用 Viser 显示完整场景；
  - 可调节机器人基座、桌子、孔、插销的 `x/y/z/yaw`；
  - 可调节全部 26 个关节；
  - `Save scene as training default` 会原子更新上述 JSON。
- `tools/check_ur5e_o30_scene.py`
  - 使用 Isaac 加载同一 JSON；
  - 用于检查初始碰撞和接触力。
- `tools/ur5e_o30_scene_state.py`
  - 提供统一的 JSON 加载、校验和保存逻辑。

### 1.3 当前初始值

当前已保存的机器人基座状态为：

```text
base position = (0.0, 0.418, 0.0)
base rotation wxyz = (1.0, 0.0, 0.0, 0.0)
```

UR5e 初始关节角（第一个关节已经在用户原始数值上增加 `pi`）：

```text
shoulder_pan_joint  =  1.5080126535959464
shoulder_lift_joint = -0.6052352985178681
elbow_joint         = -1.4034392478161284
wrist_1_joint       = -4.306915475615747
wrist_2_joint       = -1.6335215572162032
wrist_3_joint       =  3.089
```

手部关节暂时全部为 `0.0`。以
`isaacsimenvs/cfg/robot/ur5e_o30_scene.json` 的版本控制内容为准。

当前场景参考位置：

```text
table = (0.0, 0.0, 0.38)       # tabletop z = 0.53 m
hole  = (-0.05, 0.08, 0.53)
peg   = (0.135, 0.08, 0.54)
```

这些值只是标定起点，不应在没有可达性和碰撞验证的情况下直接用于大规模训练。

## 2. 机器人结构

### 2.1 机械臂关节（6）

```text
shoulder_pan_joint
shoulder_lift_joint
elbow_joint
wrist_1_joint
wrist_2_joint
wrist_3_joint
```

### 2.2 手部关节（20）

```text
thumb_cmc_roll
thumb_cmc_yaw
thumb_mcp
thumb_ip
index_mcp_roll
index_mcp_pitch
index_pip
index_dip
middle_mcp_roll
middle_mcp_pitch
middle_pip
middle_dip
ring_mcp_roll
ring_mcp_pitch
ring_pip
ring_dip
pinky_mcp_roll
pinky_mcp_pitch
pinky_pip
pinky_dip
```

建议将上述顺序作为 UR5e-O30 的 canonical policy order。Isaac Lab 实际解析顺序
仍应通过名称建立 permutation，不能假定与 URDF 或字典顺序相同。

### 2.3 关键 body/link

```text
palm:         hand_base_link
thumb tip:    thumb_distal
index tip:    index_distal
middle tip:   middle_distal
ring tip:     ring_distal
pinky tip:    pinky_distal
```

`PALM_CENTER_OFFSET` 和 `FINGERTIP_OFFSET` 必须根据 O30 的 link 坐标和实际指腹
位置重新标定，不能沿用 SHARPA 的数值。

## 3. 已知阻塞问题

### 3.1 `ring_mcp_roll` 的限制无效

当前 URDF 中该关节仍为：

```xml
<limit lower="-0.1371" upper="0.18823" effort="0.0" velocity="0.0"/>
```

正式训练前必须根据厂家参数填写有效的 `effort` 和 `velocity`。如果暂时没有厂家
数值，应明确记录所采用的保守仿真假设，并在 sim-to-real 前替换；不要静默使用 0。

### 3.2 必须确认真实手的控制接口

URDF 表达了 20 个独立 revolute joint，但真实 O30 控制器未必提供 20 个独立位置
指令。如果真实硬件存在电机耦合、从动关节或腱驱动，则应：

1. 让策略动作对应真实可控电机，而不是盲目对应全部 URDF joint；
2. 在仿真动作层实现 motor-to-joint coupling；
3. 观测仍可包含全部关节状态，但动作维度必须和部署接口一致。

在控制接口未确认之前，不要开始长时间训练。

## 4. 代码迁移要求

### 4.1 新增机器人 profile，不覆盖原始配置

机器人 profile 至少应包含：

- URDF 路径；
- base position 和 base rotation (`wxyz`)；
- canonical arm/hand joint names；
- arm/hand joint regex 或显式名称；
- 默认关节位置；
- palm body 和五个 fingertip body；
- palm/fingertip local offsets；
- stiffness、damping、armature、friction；
- effort/velocity limits 或覆盖策略；
- arm/hand 动作划分信息。

初始状态已存放在 `isaacsimenvs/cfg/robot/ur5e_o30_scene.json`。训练接入时应直接
读取该文件，不能把其中的数值再次硬编码到 Python 或 YAML。

### 4.2 需要处理的主要文件

#### `isaacsimenvs/tasks/play/utils/scene_utils.py`

当前包含 KUKA/SHARPA 专用的：

- `ARM_JOINT_REGEX` / `HAND_JOINT_REGEX`；
- `JOINT_NAMES_CANONICAL`；
- `PALM_BODY_NAME` / fingertip names；
- PD gains、armature、friction；
- 默认关节角；
- 固定基座 `(0.0, 0.8, 0.0)`；
- 7+22 自由度断言。

应改为由所选 robot profile 提供，并保持原配置仍能运行。

#### `isaacsimenvs/tasks/play/utils/reset_utils.py`

当前假定 7 个 arm joint、22 个 hand joint 和 29 个 canonical joint。需要：

- 按 profile 查找 joint/body IDs；
- 对 exact joint-name set 做清晰的启动期校验；
- 为 26 个关节建立 Lab-order/canonical-order permutation；
- 保证 reset、action target buffer 和 joint limit tensor 尺寸一致。

#### `isaacsimenvs/tasks/play/utils/action_utils.py`

当前硬编码：

```python
arm_action = actions[:, :7]
hand_action = actions[:, 7:]
```

必须按 profile 中的 arm joint count/IDs 切分。保持现有语义时：

- arm：velocity-delta accumulator；
- hand：把 `[-1, 1]` 映射到位置上下限。

如果真实 O30 使用耦合控制，应在此处或独立映射层完成 motor-to-joint mapping。

#### `isaacsimenvs/tasks/play/utils/obs_utils.py`

当前 `NUM_JOINTS = 29`，并使用 SHARPA 专用 palm/fingertip offsets。UR5e-O30
需要 26 个 joint，并重新标定 offsets。

在保持当前 observation field list 不变时，预计维度为：

```text
action space      = 26
actor observation = 131
critic state      = 153
```

应继续由 field sizes 自动计算最终空间，避免在多个位置重复写死这些数字。

#### task YAML / configclass

目前 `Play.yaml`、`PreciseAssembly.yaml` 和 `PlayEnvCfg` 中存在
`action_space: 29` 以及 KUKA URDF 路径。应新增独立 UR5e-O30 task/profile 配置，
不要破坏原任务的 checkpoint 兼容性。

#### evaluation / viewer

`evaluation/_viser_demo.py` 当前硬编码：

- `N_ACT = 29`；
- KUKA URDF；
- 7 轴默认姿态。

训练环境跑通后，评估工具也应读取同一 robot profile。不要维护第二套独立关节
顺序或默认姿态。

#### deployment

部署端必须与训练保持完全相同的：

- canonical joint order；
- observation field 顺序和归一化；
- action scaling、smoothing 和限位；
- 控制频率；
- base/table/object 坐标约定；
- quaternion 顺序。

## 5. 坐标系要求

- 配置中的机器人基座应表示为相对于每个 `env_origin` 的局部位姿。
- 单环境中局部坐标看起来与世界坐标相同；多环境训练时不能把所有机器人写到
  同一个绝对世界坐标，否则各环境会重叠。
- 机器人、桌子、物体和孔的位置必须使用一致的环境局部坐标约定。
- 四元数在 Isaac 配置和本文档中统一使用 `wxyz`。
- Viser 只负责运动学可视化，不提供可靠的物理碰撞结论；碰撞必须在 Isaac 中
  验证。
- 移动机器人但不移动桌子/物体会改变任务几何和 policy observation，不能被视为
  无影响的显示变换。

## 6. 权重兼容性

原始 checkpoint 针对 29 维动作和 140 维 actor observation。UR5e-O30 预计为
26 维动作和 131 维 actor observation，因此：

- 不要直接加载原 KUKA/SHARPA checkpoint；
- 第一版应从头训练；
- 不要为了加载旧权重而虚构 3 个 dummy action 或 dummy observation；
- 如未来确实需要迁移学习，应单独实现并验证 feature transfer、teacher/student 或
  distillation，不要混入首版机器人迁移。

## 7. 推荐实施阶段和验收门槛

### 阶段 A：资产和机器人 profile

1. 固化用户最终保存的 base pose 和 26 个默认关节角；
2. 修正无效 joint limits；
3. 新增 UR5e-O30 profile；
4. 成功转换 URDF -> USD；
5. 启动时打印并核对所有 joint/body 名称与数量。

验收：1 个环境能够稳定加载，默认姿态无明显穿模，所有关节值有限且处于限位内。

### 阶段 B：动作、观测和 reset

1. 动作空间改为 6+20（或经确认后的真实电机维度）；
2. 移除 `:7`、`7:`、`29` 等机器人专用切片/尺寸；
3. 建立 canonical permutation；
4. 重新标定 palm/fingertip links 和 offsets；
5. reset 后 `_prev_targets`、`_cur_targets` 与真实 joint state 一致。

验收：逐关节小幅动作方向正确，无关节串位，无 target 越界，无 NaN/Inf。

### 阶段 C：单环境物理验证

暂时关闭或减小 domain randomization：

- 固定机器人基座；
- 固定桌子；
- 减小 arm/hand reset noise；
- 缩小物体初始位置范围；
- 暂时关闭大外力、延迟和强摩擦随机化。

验收：连续运行数千 physics steps，无爆炸速度、异常接触、物体穿透或关节失控；
机器人工作空间覆盖物体和插孔。

### 阶段 D：并行环境 smoke test

按顺序测试 1、16、64 个环境，再扩大规模。检查：

- 各机器人位于各自 `env_origin`；
- reset 后没有环境重叠；
- observation/action tensor shape 正确；
- GPU 内存、接触数量和仿真速度正常；
- 多次 reset 后无 NaN/Inf。

### 阶段 E：训练课程

建议从简到难：

1. 固定物体位置的 reach；
2. 小范围随机位置的 grasp/lift；
3. play pretraining；
4. precise assembly finetuning；
5. 最后逐步恢复摩擦、质量、观测噪声、延迟、外力和场景位置随机化。

不要一开始就使用完整随机化，否则难以区分机器人建模错误与强化学习问题。

## 8. 最小测试清单

后续 Codex 每完成一个迁移阶段，至少验证：

- [ ] 原 KUKA 配置仍能构建或现有相关测试仍通过；
- [ ] UR5e-O30 正好识别 6 个 arm joint 和 20 个 hand joint；
- [ ] 正好识别 1 个 palm body 和 5 个 fingertip bodies；
- [ ] canonical 和 Isaac Lab joint-name set 完全一致；
- [ ] default pose 全部处于 joint limits 内；
- [ ] action tensor 最后一维与配置一致；
- [ ] actor observation/state 维度与运行时实际 tensor 一致；
- [ ] 每个动作索引只驱动预期关节；
- [ ] 初始状态无机器人—桌子非预期碰撞；
- [ ] 多环境使用 `env_origin + local_position`；
- [ ] 数千步运行无 NaN/Inf；
- [ ] 保存的配置中没有 `/tmp` 依赖或绝对用户机器路径；
- [ ] 评估与部署使用训练时相同的关节顺序和归一化。

## 9. 对后续 Codex 的执行要求

1. 开始修改前先检查工作树，保留用户已有改动。
2. 先阅读本文件及上述相关源文件，不要只替换配置中的 URDF 路径。
3. 优先实现阶段 A/B 的单环境 smoke test，不要直接启动长时间训练。
4. 保留原 KUKA/SHARPA 路径作为回归基线。
5. 任何 joint/body 数量或名称不匹配都应启动失败并给出清晰错误，不要静默回退。
6. 未确认真实 O30 控制维度之前，不要假定 20 个 URDF joints 都是独立动作。
7. 不要宣称 Viser 可视化已经验证了碰撞、动力学或训练稳定性。
8. 完成每个阶段后报告实际运行命令、tensor shapes、识别到的 joint/body names
   和验证结果。

建议给后续 Codex 的首个任务是：

> 按 `UR5E_O30_TRAINING_MIGRATION.md` 完成阶段 A：在保留原 KUKA 配置的前提下，
> 新增 UR5e-O30 robot profile，固化已确认的初始状态，并提供 1-env、无训练的
> Isaac smoke test。先不要修改动作/观测网络或启动 PPO。
