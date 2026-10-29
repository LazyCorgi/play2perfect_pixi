"""Shared checkpoint selection, training configuration and stage bookkeeping."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.beam_scaling import problem_name, scale_percent, task_name
from evaluation.eval_isaacsim import EVAL_DR_OFF_OVERRIDES
from evaluation.problem_presets import default_tolerances

NAME = '0_progressive_beam'


def screen_checkpoints(directory, all_saved=False):
    # rl_games can save two filenames for the same final epoch.
    epochs = {}
    for path in sorted(directory.glob('last_*_ep_*_rew_*.pth')):
        epochs[int(re.search(r'_ep_(\d+)_',path.name)[1])] = path
    ordered = [epochs[epoch] for epoch in sorted(epochs)]
    if not ordered:
        raise RuntimeError(f'No periodic checkpoints saved in {directory}')
    if all_saved:
        return ordered
    return list(dict.fromkeys([ordered[0], ordered[len(ordered)//2], ordered[-1]]))


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def training_command(args, output):
    training_checkpoint = getattr(args, 'training_initializer', None) or args.checkpoint
    command = [sys.executable, '-u', 'isaacsimenvs/train.py', '--task', task_name(args.percent),
               '--agent', 'rl_games_sapg_cfg_entry_point', '--checkpoint', str(training_checkpoint),
               '--checkpoint_load_mode', 'weights', '--headless']
    overrides = dict(EVAL_DR_OFF_OVERRIDES)
    overrides.update({
        'env.precise_assembly.problem': problem_name(args.percent),
        'env.precise_assembly.goal_mode': 'preInsertAndFinal',
        'env.precise_assembly.goal_xy_obs_noise': .002,
        'env.scene.num_envs': 96,
        'agent.params.seed': getattr(args, 'training_seed', 42),
        'agent.params.config.max_epochs': args.epochs,
        'agent.params.config.save_frequency': min(getattr(args, 'save_frequency', 100), args.epochs),
        'agent.params.config.uniform_checkpoints': True,
        'hydra.run.dir': str(output),
    })
    command += [key+'='+(value if isinstance(value,str) else json.dumps(value,separators=(',',':'))) for key,value in overrides.items()]
    return command, overrides


class Stage:
    def __init__(self, args):
        self.args = args
        self.directory = ROOT/'outputs/progressive_beam'/args.run_id
        self.logs = ROOT/'evaluation_runs/progressive_beam'/args.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.logs.mkdir(parents=True, exist_ok=False)
        self.lock = (self.directory.parent/'gpu0.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.deadline = time.monotonic()+args.minutes*60
        self.stopped = False
        self.child = None
        self.state = {'arguments': vars(args), 'status': 'initializing', 'pid': os.getpid(),
                      'started_utc': datetime.now(timezone.utc).isoformat(), 'evaluations': [],
                      'training_initialization': 'weights; original normalization and sigma preserved; fresh optimizer',
                      'protocol': 'fixed_relaxed_curriculum_v1'}
        with open(args.checkpoint, 'rb') as f:
            self.state['source_sha256'] = hashlib.file_digest(f,'sha256').hexdigest()
        self.state['git_head'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        self.state['git_branch'] = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        (self.logs/'source_changes.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
        changed=set(subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines())
        changed.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines())
        snapshot=[]
        for name in sorted(changed):
            source=ROOT/name
            if not source.is_file():continue
            destination=self.logs/'code_snapshot'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            snapshot.append({'path':name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        save(self.logs/'source_snapshot.json',snapshot)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self,'stopped',True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self,'stopped',True))
        self.publish()

    def publish(self, **values):
        self.state.update(values)
        self.state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save(self.directory/'status.json',self.state)
        print(json.dumps({k:self.state[k] for k in ('status','updated_utc')} | values, default=str),flush=True)

    @staticmethod
    def rate(rows):
        return sum(r['task_success_count_all_envs'] for r in rows)/sum(r['num_envs'] for r in rows)
