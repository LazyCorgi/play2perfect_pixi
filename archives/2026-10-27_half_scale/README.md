# 半尺寸迁移实验归档（2026-10-27）

本归档封存截至 2026-10-27 的环境部署、半尺寸资产、训练和评价代码，以及插接/beam 的探索实验、种子审计和通宵训练结果。下一轮工作从[交接文档](../../HANDOFF.md)开始。

## 保存方式

- **代码**：在本地分支 `archive/half-scale-20261027` 建立 Git 提交，基线为 `a7d0b71`。保留当前全部改动，包括安装说明，不丢弃或反向覆盖工作区内容。没有推送远端。
- **代码离线包**：`outputs/archive/20261027_half_scale/code.bundle`，包含上述分支及其提交历史，可用 Git 独立恢复。
- **实验数据原位归档**：保留 `outputs/`、`evaluation_runs/`、`pretrained_assembly/`、`pretrained_policy/` 中全部现有文件，保持原路径，避免破坏配置和报告中的引用。没有删减 checkpoint，也没有额外复制整套约 28 GB 训练产物。
- **完整性清单**：[data_manifest.jsonl](data_manifest.jsonl) 记录每个数据文件的相对路径、字节数、修改时间和 SHA-256；符号链接单独记录链接目标。[data_summary.json](data_summary.json) 提供总量。[checkpoint_lineage.json](checkpoint_lineage.json) 收集训练输入权重和初始化元数据。
- **运行现场**：本轮遗留的 9 个网页/仿真进程已停止，记录在 `outputs/archive/20261027_half_scale/stopped_services.json`。没有新训练在后台接力。
- **检查**：归档前 `python -m pytest -q tests` 为 27 passed；仅有一条依赖弃用警告。

**原位归档不是异机备份。** 代码 bundle 不包含被 Git 忽略的模型和实验数据。迁移到另一台机器时，应同时复制清单涉及的原始目录；不要仅复制此 README 或 bundle 后删除数据。

## 结果与入口

| 内容 | 入口 |
| --- | --- |
| 可行性与修改位置 | [half_scale_feasibility.md](../../docs/half_scale_feasibility.md) |
| 十二条短训练路线及指标定义 | [half_scale_experiments_detailed.md](../../docs/half_scale_experiments_detailed.md) |
| 短训练汇总 | [half_scale_finetuning_report.md](../../docs/half_scale_finetuning_report.md) |
| 576 个起点和轨迹复查 | [seed_scene_review.md](../../docs/seed_scene_review.md) |
| 通宵设定 | [overnight_beam_half.md](../../docs/overnight_beam_half.md) |
| 通宵结果与模型评选局限 | [overnight_beam_half_results.md](../../docs/overnight_beam_half_results.md) |
| 最新模型观看与复现命令 | [latest_beam_model_verification.md](../../docs/latest_beam_model_verification.md) |

原始实验目录：

```text
outputs/half_scale/                         初始半尺寸短训练及实验候选
outputs/half_scale_round2/                  12 条后续路线及初始化权重
outputs/overnight_beam_half/20261026_overnight_beam_v2/
evaluation_runs/half_scale/                基线与回归测试
evaluation_runs/half_scale_round2/         命令清单和评价汇总
evaluation_runs/seed_scene_review/         576 个初态与轨迹
evaluation_runs/observation_review/        真实状态输入及对比图
evaluation_runs/source_verification/       论文/公开数据入口的核查材料
evaluation_runs/overnight_beam_half/20261026_overnight_beam_v2/
```

失败的通宵首启动 `20261026_overnight_beam` 也保留，但完成 0 轮；有效训练是带 `_v2` 的目录。

## 权重起点与原任务能力

项目提供两层权重：

1. `pretrained_policy/model.pth`：Stage 1 通用 Play 预训练。
2. `pretrained_assembly/{tight_insertion,beam_assembly_step1,beam_assembly_step2,screwing}/model.pth`：分别针对四个装配任务微调的 Stage 2 模型。它们是四份模型，不是一份同时完成四任务的通用模型。

我们绝大多数半尺寸路线以对应的 Stage 2 模型或其本地继续微调候选为起点。只有 `peg_play` 明确以 Stage 1 为起点。通宵 beam 实验重新从官方 Stage 2 `pretrained_assembly/beam_assembly_step1/model.pth` 出发，经初始化副本后连续优化，没有承接之前失败的半尺寸 beam 候选。精确来源链以本目录 JSON 和各实验命令为准。

只在半尺寸分布优化，且修改奖励、归一化统计和探索，没有原尺寸混合训练或保留约束，可能导致原任务能力下降，通常称为灾难性遗忘；小物体适应与原尺寸保留也可能存在取舍。**这不是成功迁移的证据，也不是必须接受的退化。** 此次新任务也没有成功，因此不能解释为已经取得有效取舍。

用户已观察到最新候选无法完成原尺寸 beam；目前没有同一批原尺寸起点、同一评价协议下官方权重与最新权重的完整配对统计，尚不能量化遗忘程度或把原因唯一归于遗忘。需排除权重与任务尺寸、目标模式、归一化统计、SAPG 分支、噪声和时限不匹配。官方权重保留在独立目录，本地微调输出未用作覆盖目标。

## 恢复代码和核查数据

```bash
# 在其他位置恢复代码；模型/日志需要另行复制
git clone /path/to/code.bundle restored_play2perfect

# 本仓库内以归档为起点开展新实验
git switch -c experiment/progressive-scale archive/half-scale-20261027
```

以下在项目根目录运行，核对某一个模型是否仍与封存时相同：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
target = 'pretrained_assembly/beam_assembly_step1/model.pth'
rows = map(json.loads, Path('archives/2026-10-27_half_scale/data_manifest.jsonl').read_text().splitlines())
row = next(r for r in rows if r['path'] == target)
with open(target, 'rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == row['sha256']
print('OK', target)
PY
```
