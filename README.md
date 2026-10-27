# Play2Perfect

本地半尺寸迁移实验已于 2026-10-27 [归档](archives/2026-10-27_half_scale/README.md)。下一轮逐步缩小工作请先读 [HANDOFF.md](HANDOFF.md)。

### What Matters in Dexterous Play Pretraining for Precise Assembly?

Official code release for **Play2Perfect**, a 2-stage reinforcement-learning pipeline for
contact-rich robotic assembly.

📄 **Paper:** https://arxiv.org/abs/2606.26428

https://github.com/user-attachments/assets/6bfafdbc-41f4-472a-aeaa-444f7e50076b

## Overview

Play2Perfect learns precise, contact-rich assembly in two stages:

1. **Play** — pretrain a policy by playing with random objects in free space.
2. **Finetune** — finetune the play policy on a contact-rich assembly task.

This repository provides:

- **Training** code for both stages in Isaac Sim (Isaac Lab).
- **Interactive (viser) evaluation** for four assembly tasks using our released checkpoints.
- A **minimal sim-to-real deployment** reference.

> This release targets **Isaac Sim only**.

## Project structure

```
play2perfect/
├── isaacsimenvs/          # Training: Stage-1 play env + Stage-2 precise-assembly env
│   ├── train.py           # Single training entry point (both stages)
│   ├── cfg/               # Hydra task + train configs
│   └── tasks/{play,precise_assembly}/
├── evaluation/         # The 4 assembly tasks: problem registry + viser evaluation
│   ├── problems/          # peg-in-hole, fabrica beam parts 0 & 2, furniture-bench
│   └── eval_isaacsim.py   # Interactive viser evaluation
├── rl_games/              # Vendored RL library (PPO + SAPG)
├── deployment/            # Minimal sim-to-real reference
├── assets/urdf/           # Robot + task assets
└── docs/                  # Installation + deployment guides
```

## The four assembly tasks

| Task | `problem` key |
|------|----------------|
| Tight insertion (L-peg, 0.5 mm tolerance) | `tight_insertion` |
| Beam assembly — step 1 | `beam_assembly_step1` |
| Beam assembly — step 2 | `beam_assembly_step2` |
| Screwing (furniture leg) | `screwing` |

## Installation

Prerequisites: Linux x86-64, a recent NVIDIA driver with CUDA 12.8 support, and
[Pixi](https://pixi.sh/).

```bash
pixi install --locked
pixi run verify
pixi run smoke-test
```

The first install downloads Isaac Sim and requires roughly 20 GB of disk space. See
[docs/isaacsim_installation.md](docs/isaacsim_installation.md) for details and troubleshooting.

## Quick start

### 1. Download checkpoints

```bash
# Stage-1 play policy + the 4 finetuned assembly policies
pixi run python download_checkpoints.py
```

### 2. Interactive evaluation (viser)

```bash
# One problem from a single checkpoint
pixi run python evaluation/eval_isaacsim.py \
  --checkpoint-path pretrained_assembly/tight_insertion/model.pth \
  --problem tight_insertion \
  --port 8044

# ...or load all four downloaded policies and switch between them in the GUI
pixi run python evaluation/eval_isaacsim.py --policies-dir pretrained_assembly --port 8044
```

Open `http://localhost:8044` in a browser to watch the policy roll out the insertion.

### Offline success-rate evaluation

Headless, batched eval that scores each problem over many parallel envs (one
i.i.d. first-episode trial per env) and prints a success-rate table:

```bash
pixi run python evaluation/eval_offline.py --policies-dir pretrained_assembly
```

### 3. Train Stage 1 (play)

```bash
pixi run python isaacsimenvs/train.py \
  --task Isaacsimenvs-Play-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --headless
```

### 4. Finetune Stage 2 (precise assembly)

```bash
pixi run python isaacsimenvs/train.py \
  --task Isaacsimenvs-PreciseAssembly-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --checkpoint pretrained_policy/model.pth \
  env.precise_assembly.problem=tight_insertion \
  --headless
```

## Deployment

For the experimental **half-size insertion** task on the original KUKA + SHARPA,
see [the run guide and initial results](docs/half_scale_insertion.md) and
[the fine-tuning experiment report](docs/half_scale_finetuning_report.md).
The original task remains available as the regression baseline.

See [docs/deployment.md](docs/deployment.md) for the minimal sim-to-real reference.

## Formatting

```bash
./format.sh   # ruff over the Python packages + xmllint over URDFs
```

## Acknowledgements

This codebase builds on excellent prior work:

- [**rl_games**](https://github.com/Denys88/rl_games) — the high-throughput RL library our
  training loop is built on (vendored here).
- [**SAPG**](https://sapg-rl.github.io/) — Split and Aggregate Policy Gradients, the
  large-scale PPO variant we use for training.
- [**SimToolReal**](https://github.com/tylerlum/simtoolreal) — our prior work on
  object-centric tool manipulation, which the Stage-1 "play" pretraining is based on.

## Citation

```bibtex
@misc{lum2026play2perfectmattersdexterousplay,
      title={Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?},
      author={Tyler Ga Wei Lum and Kushal Kedia and C. Karen Liu and Jeannette Bohg},
      year={2026},
      eprint={2606.26428},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.26428},
}
```

## Contact

- Kushal Kedia — kk837@cornell.edu
- Tyler Lum — tylerlum@stanford.edu
