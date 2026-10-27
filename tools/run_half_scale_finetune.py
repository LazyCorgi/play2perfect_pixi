"""Run a reproducible, bounded six-group SAPG adaptation experiment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.eval_isaacsim import EVAL_DR_OFF_OVERRIDES
from evaluation.problem_presets import HALF_PROBLEMS, task_for_problem


def build_command(args):
    if args.num_envs <= 0 or args.num_envs % 6:
        raise ValueError('Environment count must be a positive multiple of six')
    cmd = [sys.executable, '-u', 'isaacsimenvs/train.py', '--task', task_for_problem(args.problem),
           '--agent', 'rl_games_sapg_cfg_entry_point', '--checkpoint', args.checkpoint,
           '--checkpoint_load_mode', 'weights', '--headless']
    overrides = dict(EVAL_DR_OFF_OVERRIDES) if args.phase == 'adaptation' else {}
    if args.start_curriculum == 'table_fixed':
        from evaluation.problems import PROBLEM_REGISTRY
        from evaluation.objects import NAME_TO_OBJECT
        from evaluation._viser_demo import _load_mesh_for_viz
        obj = NAME_TO_OBJECT[PROBLEM_REGISTRY[args.problem].insertion_object_name]
        mesh = _load_mesh_for_viz(obj.urdf_path)
        # Original table top is 0.38 + 0.15 m; place the canonical object flat
        # just above it. This is a TRAINING curriculum, never an eval override.
        root_z = .53 - float(mesh.bounds[0, 2]) + .0003
        overrides.update({'env.reset.fixed_start_pose': [.08, 0., root_z, 1., 0., 0., 0.],
                          'env.precise_assembly.hole_x_range': [-.1, -.1],
                          'env.precise_assembly.hole_y_range': [.08, .08]})
    overrides.update({
        'env.precise_assembly.problem': args.problem,
        'env.precise_assembly.goal_mode': args.goal_mode,
        'env.precise_assembly.force_lift_reward_active': True,
        'env.reward.use_minimum_height_reference': args.lift_reference == 'minimum',
        'env.episode_length_s': args.episode_seconds,
        'env.termination.episode_length': int(round(args.episode_seconds * 60)),
        'env.scene.num_envs': args.num_envs,
        'agent.params.seed': args.seed,
        'agent.params.config.expl_coef_block_size': args.num_envs // 6,
        'agent.params.config.minibatch_size': args.num_envs * 16 // 3,
        'agent.params.config.central_value_config.minibatch_size': args.num_envs * 16 // 3,
        'agent.params.config.learning_rate': args.learning_rate,
        'agent.params.config.lr_schedule': args.lr_schedule,
        'agent.params.config.central_value_config.learning_rate': args.learning_rate,
        'agent.params.config.max_epochs': args.epochs,
        'agent.params.config.save_frequency': min(250, args.epochs),
        'agent.params.config.save_best_after': min(100, args.epochs),
        'hydra.run.dir': str(Path('outputs/half_scale_round2') / args.name),
    })
    for key, value in overrides.items():
        cmd.append(key + '=' + (value if isinstance(value, str) else json.dumps(value, separators=(',', ':'))))
    return cmd, overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', choices=sorted(HALF_PROBLEMS), default='tight_insertion_half')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--goal-mode', choices=['finalGoalOnly', 'preInsertAndFinal'], default='finalGoalOnly')
    parser.add_argument('--phase', choices=['adaptation', 'robust'], default='adaptation')
    parser.add_argument('--start-curriculum', choices=['random', 'table_fixed'], default='random')
    parser.add_argument('--lift-reference', choices=['initial', 'minimum'], default='initial')
    parser.add_argument('--episode-seconds', type=float, default=10., help='Per-goal timeout window; goal success resets the timer, so this is not a hard full-episode deadline.')
    parser.add_argument('--num-envs', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--lr-schedule', choices=['adaptive', 'constant'], default='adaptive')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a single directory name')
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    if args.episode_seconds <= 0:
        parser.error('--episode-seconds must be positive')
    cmd, overrides = build_command(args)
    manifest = {'arguments': vars(args), 'command': cmd, 'overrides': overrides}
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    if (ROOT / overrides['hydra.run.dir']).exists():
        raise FileExistsError('Choose a new experiment name; existing runs are preserved')
    if not (ROOT / args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    directory = ROOT / 'evaluation_runs/half_scale_round2' / args.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = os.environ.copy()
    env['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    with (directory / 'train.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
