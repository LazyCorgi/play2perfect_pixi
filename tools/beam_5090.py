"""Portable official-sample-budget Play -> half-scale beam experiment."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 2_608_791_552
NAME = '0_beam050_official_budget'
SOURCE = ROOT / 'pretrained_policy/model.pth'
SOURCE_SHA = '4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


class PhysicsCapacityError(RuntimeError):
    """A completed process can still have produced invalid physics."""


def check_physics_log(text):
    match = re.search(r'PhysX error:[^\n]*|(?:contact|patch|pair|collision stack)[^\n]*(?:overflow|capacity exceeded)[^\n]*',
                      text, re.IGNORECASE)
    if match:
        raise PhysicsCapacityError('Invalid physics: ' + match.group(0) + '; preserve this run and start a new run with sufficient buffers')


def physics_buffers(num_envs):
    # Explicit headroom, not a peak-capacity guarantee. Capacity must grow with
    # scene population even when CUDA's overall memory usage remains low.
    def capacity(minimum, per_env):
        return 1 << (max(minimum, num_envs * per_env) - 1).bit_length()
    return dict(gpu_max_rigid_contact_count=capacity(524288, 512),
                gpu_max_rigid_patch_count=capacity(131072, 128),
                gpu_found_lost_aggregate_pairs_capacity=capacity(131072, 128),
                gpu_total_aggregate_pairs_capacity=capacity(262144, 256),
                gpu_collision_stack_size=capacity(134217728, 65536))


def plan(num_envs, smoke=False, smoke_epochs=256):
    # Keep six coefficient-conditioned policy groups compatible with Play.
    if num_envs < 96 or num_envs % 96 or FRAMES % (num_envs * 16):
        raise ValueError('num-envs must be a multiple of 96 and divide the exact sample budget /16')
    if smoke_epochs < 32:
        raise ValueError('smoke-epochs must be at least 32; use 256 or more for capacity testing')
    epochs = smoke_epochs if smoke else FRAMES // (num_envs * 16)
    return dict(num_envs=num_envs, block_size=num_envs // 6,
                minibatch_size=num_envs * 8, epochs=epochs,
                budget_frames=num_envs * 16 * epochs,
                save_frequency=epochs if smoke else max(1, epochs // 20),
                smoke=smoke, seed=42, scale=.5, official_reference_frames=FRAMES,
                physics_buffers=physics_buffers(num_envs), protocol='scaled_physx_buffers_v2')


def train_command(directory, recipe, checkpoint=SOURCE, resume=False, attempt=0):
    # Reuse the same experiment dir for optimizer checkpoints/TensorBoard;
    # Hydra provenance for each restart is kept in a separate attempt directory.
    command = [sys.executable, '-u', 'isaacsimenvs/train.py',
            '--task', 'Isaacsimenvs-PreciseAssemblyBeamScale050-Direct-v0',
            '--agent', 'rl_games_budget_cfg_entry_point', '--checkpoint', str(checkpoint),
            '--checkpoint_load_mode', 'resume_fresh_rollout' if resume else 'weights', '--headless', '--env_seed', '42',
            f'env.scene.num_envs={recipe["num_envs"]}', 'agent.params.seed=42',
            f'agent.params.config.expl_coef_block_size={recipe["block_size"]}',
            f'agent.params.config.minibatch_size={recipe["minibatch_size"]}',
            f'agent.params.config.central_value_config.minibatch_size={recipe["minibatch_size"]}',
            f'agent.params.config.max_epochs={recipe["epochs"]}',
            f'agent.params.config.save_frequency={recipe["save_frequency"]}',
            f'+agent.params.config.train_dir={directory / "train"}',
            f'hydra.run.dir={directory / ("attempt_%03d" % attempt)}']
    command += [f'env.sim.physx.{key}={value}' for key,value in recipe['physics_buffers'].items()]
    return command


def latest_checkpoint(directory):
    # Avoid .old and best-reward files: only unambiguous periodic/final or
    # rolling checkpoints with their own saved epoch/frame will be accepted.
    paths = list((directory / 'train' / NAME / 'nn').glob('last_*_ep_*.pth'))
    rolling = directory / 'train' / NAME / 'last/model.pth'
    if rolling.is_file():
        paths.append(rolling)
    if not paths:
        raise ValueError('No saved checkpoint to resume')
    import torch
    candidates = []
    for path in paths:
        try:
            weights = torch.load(path, map_location='cpu', weights_only=False)[0]
            candidates.append((weights['epoch'], path))
        except (OSError, RuntimeError, EOFError):
            continue
    if not candidates:
        raise ValueError('No readable checkpoint to resume')
    return max(candidates, key=lambda row: (row[0], str(row[1])))[1]


def checkpoint_info(path, recipe):
    import torch
    weights = torch.load(path, map_location='cpu', weights_only=False)[0]
    epoch, frame = weights['epoch'], weights['frame']
    if frame != epoch * recipe['num_envs'] * 16 or not 0 < epoch <= recipe['epochs']:
        raise ValueError('Checkpoint counters do not match this run budget')
    if weights['current_rewards'].shape[0] != recipe['num_envs']:
        raise ValueError('Cannot change num-envs on resume')
    if 'central_value_optimizer' not in weights:
        raise ValueError('Resume requires a checkpoint generated by this runner version')
    return dict(epoch=epoch, frame=frame, sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'check', 'run', 'status'])
    parser.add_argument('--run-id', default='beam050_5090_v1')
    parser.add_argument('--num-envs', type=int, default=1536)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-epochs', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Use a single safe directory name as run-id')
    directory = ROOT / 'outputs/beam_5090' / args.run_id
    if args.mode == 'status':
        print((directory / 'status.json').read_text()); return
    if args.mode == 'check':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        a = torch.randn(128, 128, device='cuda', requires_grad=True)
        a.square().mean().backward(); torch.cuda.synchronize()
        print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                              gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                              memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30), indent=2))
        return
    recipe = plan(args.num_envs, args.smoke, args.smoke_epochs)
    if args.mode == 'plan':
        print(json.dumps(recipe, indent=2)); return
    if not SOURCE.is_file() or digest(SOURCE) != SOURCE_SHA:
        raise ValueError('Missing/incorrect official Play weights: run download_checkpoints.py --what play')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            state = json.loads((directory / 'status.json').read_text())
            if state.get('invalid_physics'):
                raise ValueError('This run recorded invalid physics and cannot be resumed; use a new run-id')
            if state['recipe'] != recipe or state['source_sha256'] != SOURCE_SHA:
                raise ValueError('Resume must keep run recipe unchanged')
            current_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            if state['git_head'] != current_head:
                raise ValueError('Resume requires the original Git commit; checkout the recorded git_head first')
            if state['status'] in ('completed', 'smoke_completed'):
                raise ValueError('Run already completed')
            if (directory / 'STOP').exists():
                raise ValueError('Remove STOP explicitly before resuming')
            checkpoint = latest_checkpoint(directory)
            info = checkpoint_info(checkpoint, recipe)
            state['resume_from'] = dict(path=str(checkpoint), **info)
            attempt = state['attempt'] + 1
        else:
            directory.mkdir(exist_ok=False)
            checkpoint, attempt = SOURCE, 0
            state = dict(recipe=recipe, source_checkpoint=str(SOURCE), source_sha256=SOURCE_SHA,
                         git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         evaluations=[], attempts=[], started_utc=datetime.now(timezone.utc).isoformat())
            (directory / 'source.diff').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True))
        stopped = [False]
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.__setitem__(0, True))

        def publish(**changes):
            state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
            save(directory / 'status.json', state)

        def run(command, label):
            log_path = directory / f'{label}.log'
            save(directory / f'{label}.command.json', command)
            with log_path.open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1'))
                publish(active_step=label, child_pid=child.pid, active_log=str(log_path))
                offset = 0
                tail = ''
                try:
                    while child.poll() is None:
                        if stopped[0] or (directory / 'STOP').exists():
                            raise InterruptedError('Stop requested; restart from latest saved checkpoint')
                        with log_path.open() as reader:
                            reader.seek(offset); chunk = reader.read(); offset = reader.tell()
                        check_physics_log(tail + chunk)
                        tail = (tail + chunk)[-512:]
                        epochs = re.findall(r'epoch\s*:\s*([\d,]+)\s*/', chunk)
                        if epochs and label.startswith('training'):
                            epoch = int(epochs[-1].replace(',', ''))
                            publish(epoch=epoch, frames=epoch * recipe['num_envs'] * 16,
                                    progress=epoch / recipe['epochs'])
                        else:
                            publish()
                        time.sleep(2)
                    # Catch errors emitted after the last poll, even if Kit
                    # reports exit code zero; smoke must never pass on overflow.
                    with log_path.open() as reader:
                        reader.seek(offset)
                        check_physics_log(tail + reader.read())
                    if child.returncode:
                        raise RuntimeError(f'{label} failed ({child.returncode}); see {log_path}')
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try: child.wait(timeout=30)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                    publish(child_pid=None)

        def evaluate(path, seeds, label):
            # Completed batches are reusable on process restart; no reselection
            # on heldout seeds, and no fallback to a partially completed batch.
            for seed in seeds:
                key = f'{label}_s{seed}'
                if any(r['label'] == key for r in state['evaluations']):
                    continue
                run([sys.executable, '-u', 'evaluation/eval_offline.py', '--worker',
                     '--problem', 'beam_assembly_step1_scale_050', '--checkpoint-path', str(path),
                     '--num-envs', '32', '--seed', str(seed), '--steps', '3000',
                     '--min-episode-steps', '0', '--policy-coefficient-id', '50',
                     '--goal-mode', 'preInsertAndFinal', '--insertion-success-tolerance', '.005',
                     '--retract-success-tolerance', '.0025'], key)
                records = [json.loads(line[12:]) for line in (directory / f'{key}.log').read_text().splitlines()
                           if line.startswith('RESULT_JSON ')]
                row = records[-1]
                if row['n_completed_first_episodes'] != 32 or row['n_unfinished'] or row['num_envs'] != 32:
                    raise RuntimeError(f'Incomplete evaluation {key}')
                state['evaluations'].append(dict(label=key, **row)); publish()
            rows = [r for r in state['evaluations'] if r['label'] in {f'{label}_s{s}' for s in seeds}]
            return sum(r['task_success_count_all_envs'] for r in rows) / (32 * len(rows))

        state['attempts'].append(dict(attempt=attempt, checkpoint=str(checkpoint),
                                      load_mode='resume_fresh_rollout' if args.resume else 'weights',
                                      started_utc=datetime.now(timezone.utc).isoformat()))
        publish(status='running', pid=os.getpid(), child_pid=None, attempt=attempt)
        try:
            if not args.smoke:
                publish(zero_shot_rate=evaluate(SOURCE, [101, 102], 'zero'))
            if not args.resume or info['epoch'] < recipe['epochs']:
                run(train_command(directory, recipe, checkpoint, args.resume, attempt), f'training_{attempt:03d}')
            paths = sorted((directory / 'train' / NAME / 'nn').glob('last_*_ep_*.pth'))
            epochs = {int(re.search(r'_ep_(\d+)', p.name)[1]): p for p in paths}
            if recipe['epochs'] not in epochs:
                raise RuntimeError('Final budget checkpoint missing')
            final = epochs[recipe['epochs']]
            final_info = checkpoint_info(final, recipe)
            if final_info['frame'] != recipe['budget_frames']:
                raise RuntimeError('Exact sample budget not reached')
            publish(final_checkpoint=str(final), frames=final_info['frame'], epoch=final_info['epoch'], progress=1.0)
            if args.smoke:
                publish(status='smoke_completed'); return
            # Predeclared sparse screen: approximately 25%, 50%, 75%, 100%.
            candidates = sorted({min(epochs, key=lambda e: abs(e-recipe['epochs']*f)) for f in (.25,.5,.75,1)})
            scores = [(evaluate(epochs[e], [101,102], f'epoch{e}'), e) for e in candidates]
            rate, epoch = max(scores, key=lambda r: (r[0], -r[1]))
            selected = epochs[epoch]
            publish(selected_checkpoint=str(selected), selected_epoch=epoch, validation_rate=rate,
                    selected_sha256=digest(selected), candidate_scores=[dict(rate=r, epoch=e) for r,e in scores])
            publish(heldout_rate=evaluate(selected, list(range(601,609)), 'heldout'), heldout_trials=256)
            publish(status='completed')
        except BaseException as exc:
            publish(status='stopped' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', error=str(exc))
            if isinstance(exc, PhysicsCapacityError):
                publish(invalid_physics=True)
            raise


if __name__ == '__main__':
    main()
