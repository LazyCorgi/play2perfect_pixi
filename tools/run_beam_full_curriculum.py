"""Reproduce the successful lineage from official 1.0 weights through 0.5."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.beam_curriculum_common import NAME, save, screen_checkpoints, training_command
from tools.run_beam_fixed_curriculum import FixedStage, digest, streamed_process, verified_source
from evaluation.beam_scaling import problem_name

COARSE_STAGES = [90, 80, 70, 65, 60]
OFFICIAL_SHA256 = 'be7e9f6c367e76a8f0697a65d3253cc8ccbdbeed83ad2dd21a126df5455481e0'

def candidates(directory, percent):
    paths = screen_checkpoints(directory, all_saved=True)
    # Preserve the historical 0.9 experiment's candidate set explicitly.
    epochs = (100, 400, 500) if percent == 90 else (100, 300, 500)
    found = {int(re.search(r'_ep_(\d+)_', p.name)[1]): p for p in paths}
    if not all(epoch in found for epoch in epochs):
        raise RuntimeError('Missing scheduled periodic checkpoint')
    return [found[epoch] for epoch in epochs]

def gate(previous, percent):
    return max(.8, previous - .05, 117/128 - (.05 if percent == 90 else .10))

class CoarseStage(FixedStage):
    def execute(self):
        self.publish(status='evaluating_source', protocol='official_to060_v1')
        baseline = [self.evaluate(self.args.checkpoint, s, f'zero_s{s}') for s in range(101,105)]
        baseline_rate = self.rate(baseline)
        self.publish(zero_shot_rate=baseline_rate)
        command, overrides = training_command(self.args, self.directory/'train')
        save(self.directory/'training.json', dict(command=command, overrides=overrides))
        self.publish(status='training'); self.process(command, 'train', seconds=2400)
        scores = []
        for checkpoint in candidates(self.directory/'train'/NAME/'nn', self.args.percent):
            epoch = int(re.search(r'_ep_(\d+)_', checkpoint.name)[1])
            rows = [self.evaluate(checkpoint, s, f'epoch{epoch}_s{s}') for s in (101,102)]
            scores.append((self.rate(rows), epoch, checkpoint, rows))
        _, epoch, checkpoint, rows = max(scores, key=lambda item:(item[0], -item[1]))
        rows += [self.evaluate(checkpoint, s, f'selected_epoch{epoch}_s{s}') for s in (103,104)]
        validation = self.rate(rows)
        required = max(gate(self.args.previous_validation, self.args.percent), baseline_rate-.05)
        self.publish(selected_checkpoint=str(checkpoint), selected_epoch=epoch,
                     validation_rate=validation, required_rate=required,
                     candidate_scores=[dict(rate=r,epoch=e,checkpoint=str(p)) for r,e,p,_ in scores])
        if validation < required:
            self.publish(status='rejected_validation'); return
        start = 201 + (90-self.args.percent)*10
        heldout = [self.evaluate(checkpoint, s, f'heldout_s{s}') for s in range(start,start+8)]
        held_rate = self.rate(heldout)
        held_gate = gate(self.args.previous_heldout, self.args.percent)
        unchanged = digest(self.args.checkpoint) == self.state['source_sha256']
        if not unchanged: raise RuntimeError('Source checkpoint changed during training')
        accepted = held_rate >= held_gate
        self.publish(status='accepted' if accepted else 'rejected_acceptance',
                     heldout_rate=held_rate, heldout_required_rate=held_gate, source_unchanged=True)
        if accepted:
            save(self.directory/'accepted.json', dict(checkpoint=str(checkpoint),
                checkpoint_sha256=digest(checkpoint), problem=problem_name(self.args.percent),
                policy_coefficient_id=50, validation_rate=validation, heldout_rate=held_rate,
                source_sha256=self.state['source_sha256'], regression_in_acceptance=False))

def run_course(args, runner=None):
    checkpoint = (ROOT / args.checkpoint).resolve()
    if digest(checkpoint) != OFFICIAL_SHA256:
        raise ValueError('Official beam-step1 starting checkpoint SHA256 mismatch')
    directory = ROOT/'outputs/progressive_beam'/args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    lock = (directory.parent/'full_curriculum.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='running', plan=COARSE_STAGES+list(range(59,49,-1)), stages=[],
                 source_sha256=OFFICIAL_SHA256, pid=os.getpid(), child_pid=None)
    stopped = [False]
    if runner is None:
        signal.signal(signal.SIGINT, lambda *_:stopped.__setitem__(0,True))
        signal.signal(signal.SIGTERM, lambda *_:stopped.__setitem__(0,True))
    def publish(**values):
        state.update(values); state['updated_utc']=datetime.now(timezone.utc).isoformat()
        save(directory/'status.json', state)
    def stop(): return stopped[0] or (directory/'STOP').exists()
    deadline = time.monotonic()+args.total_minutes*60
    previous_validation = previous_heldout = 117/128
    publish()
    try:
        with (directory/'course.log').open('ab') as log:
            def execute(command, label):
                save(directory/(label+'_command.json'), dict(command=command))
                if runner is not None: return runner(command)
                return streamed_process(command, [log], deadline,
                    stop, on_pid=lambda pid:publish(child_pid=pid))
            for percent in COARSE_STAGES:
                if stop(): raise InterruptedError('Course stopped')
                stage_id = args.run_id+f'_scale{percent:03d}'
                stage_dir = directory.parent/stage_id
                command = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
                    '--percent',str(percent),'--checkpoint',str(checkpoint),'--run-id',stage_id,
                    '--previous-validation',str(previous_validation),'--previous-heldout',str(previous_heldout)]
                publish(active_stage=percent, active_stage_status=str(stage_dir/'status.json'))
                code = execute(command, f'scale{percent:03d}')
                child = json.loads((stage_dir/'status.json').read_text()) if (stage_dir/'status.json').exists() else {}
                state['stages'].append(dict(percent=percent, status=child.get('status','failed'),returncode=code))
                publish(child_pid=None)
                if code or child.get('status')!='accepted':
                    publish(status='stopped_at_stage'); return state
                parent = verified_source(stage_dir/'accepted.json')
                checkpoint=Path(parent['checkpoint'])
                previous_validation=parent['validation_rate']; previous_heldout=parent['heldout_rate']
                publish(latest_checkpoint=str(checkpoint),latest_scale=percent/100)
            fine_id=args.run_id+'_fine'
            fine_dir=directory.parent/fine_id
            publish(active_stage='0.59→0.50',active_stage_status=str(fine_dir/'status.json'))
            command=[sys.executable,'-u','tools/run_beam_fixed_curriculum.py','--run-id',fine_id,
                     '--source-accepted',str(stage_dir/'accepted.json')]
            code=execute(command,'fine_course')
            fine=json.loads((fine_dir/'status.json').read_text()) if (fine_dir/'status.json').exists() else {}
            if code or fine.get('status')!='completed':
                publish(status='stopped_at_stage', child_pid=None, fine_status=fine.get('status')); return state
            best=json.loads((fine_dir/'best_model.json').read_text())
            save(directory/'best_model.json',best)
            if runner is None: os.link(best['checkpoint'],directory/'best_scale050.pth')
            publish(status='completed',child_pid=None,active_stage=None,
                    latest_checkpoint=best['checkpoint'],latest_scale=.5,
                    final_heldout_rate=best.get('heldout_rate'))
            return state
    except InterruptedError:
        publish(status='stopped',child_pid=None); return state
    except BaseException as exc:
        publish(status='failed',child_pid=None,error=repr(exc)); raise
    finally: lock.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='pretrained_assembly/beam_assembly_step1/model.pth')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--total-minutes',type=float,default=360)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--percent',type=int,choices=COARSE_STAGES)
    parser.add_argument('--previous-validation',type=float,default=117/128)
    parser.add_argument('--previous-heldout',type=float,default=117/128)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        parser.error('Use one run directory name')
    if not math.isfinite(args.total_minutes) or args.total_minutes <= 0:
        parser.error("Use a finite positive time budget")
    if args.worker:
        if args.percent is None: parser.error('Worker requires --percent')
        args.epochs=500; args.minutes=40
        stage=CoarseStage(args)
        try: stage.execute()
        except InterruptedError: stage.publish(status='stopped')
        except BaseException as exc:
            stage.publish(status='failed',error=repr(exc)); raise
        finally: stage.publish(child_pid=None)
    else: run_course(args)

if __name__=='__main__': main()
