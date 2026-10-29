"""Fixed 0.60 -> 0.50 curriculum with relaxed progression and live tee logs.

Each stage uses the same 500-epoch recipe. Select the best of three periodic
checkpoints on fixed development seeds and inherit it immediately. Only severe
regression or execution failure stops the course; no remedial branches/retries.
This protocol is separate from the historical strict acceptance protocol.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.beam_scaling import problem_name,scale_percent
from evaluation.problem_presets import default_tolerances
from tools.beam_curriculum_common import Stage,save,screen_checkpoints,training_command,NAME

STAGES=list(range(59,49,-1))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def progression_decision(candidate_rate,baseline_rate,min_rate=.5,max_loss=.25):
    return candidate_rate>=min_rate and candidate_rate>=baseline_rate-max_loss


def streamed_process(command,logfiles,deadline,stop,on_pid=None,on_tick=None,on_output=None):
    """Drain output while checking STOP/deadline; tee bytes to files and stdout."""
    env=os.environ.copy();env['OMNI_KIT_ACCEPT_EULA']='YES';env['PYTHONUNBUFFERED']='1'
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
    if on_pid:on_pid(child.pid)
    selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
    aborted=None;terminate_at=None;last_tick=0.
    try:
        while selector.get_map() or child.poll() is None:
            now=time.monotonic()
            if aborted is None and (stop() or now>=deadline):
                aborted='stop requested' if stop() else 'time budget exhausted'
                terminate_at=now
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            if terminate_at is not None and now-terminate_at>45 and child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL)
            if on_tick and now-last_tick>=15:
                on_tick();last_tick=now
            for key,_ in selector.select(.5):
                chunk=os.read(key.fileobj.fileno(),65536)
                if not chunk:
                    selector.unregister(key.fileobj);continue
                for stream in logfiles:
                    stream.write(chunk);stream.flush()
                sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
                if on_output:on_output(chunk)
        code=child.wait()
        if aborted:
            if aborted=='stop requested':raise InterruptedError(aborted)
            raise TimeoutError(aborted)
        return code
    finally:
        selector.close();child.stdout.close()
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


class FixedStage(Stage):
    def evaluate(self,checkpoint,seed,label):
        insertion,retract=default_tolerances(problem_name(self.args.percent))
        command=[sys.executable,'-u','evaluation/eval_offline.py',
            '--problem',problem_name(self.args.percent),'--checkpoint-path',str(checkpoint),
            '--num-envs','32','--steps','3000','--seed',str(seed),'--min-episode-steps','0',
            '--goal-mode','preInsertAndFinal','--policy-coefficient-id','50','--episode-length-s','10',
            '--insertion-success-tolerance',str(insertion),'--retract-success-tolerance',str(retract)]
        self.publish(status='evaluating')
        log=self.process(command,label)
        rows=[json.loads(line[12:]) for line in log.read_text().splitlines() if line.startswith('RESULT_JSON ')]
        if len(rows)!=1 or rows[0].get('n_unfinished',1) or rows[0].get('n_completed_first_episodes')!=32:
            raise RuntimeError(f'Incomplete or invalid task evaluation: {log}')
        row=rows[0];row['label']=label
        self.state['evaluations'].append(row)
        self.publish(last_result={'label':label,'successes':row['task_success_count_all_envs'],'trials':32})
        return row

    def process(self,command,label,seconds=600):
        if self.stopped or (self.directory/'STOP').exists():raise InterruptedError('Stage stopped')
        save(self.logs/(label+'_command.json'),{'command':command})
        path=self.logs/(label+'.log')
        with path.open('wb') as stream:
            code=streamed_process(command,[stream],min(self.deadline,time.monotonic()+seconds),
                lambda:self.stopped or (self.directory/'STOP').exists(),
                on_pid=lambda pid:self.publish(active_label=label,child_pid=pid,active_log=str(path)))
        self.publish(child_pid=None)
        if code:raise RuntimeError(f'{label} exited {code}: {path}')
        return path

    def execute(self):
        policy={'protocol':'fixed_relaxed_curriculum_v1','min_rate':self.args.min_rate,
                'max_same_size_source_loss':self.args.max_loss,'selection_seeds':[101,102],
                'selection_metric':'full task success, all first episodes, deterministic branch50',
                'old_strict_gates':False,'remedial_training':False,'regression_tests':False,
                'intermediate_heldout':False,'reward_lr_exploration':'unchanged'}
        self.publish(status='evaluating_source',acceptance=policy)
        baseline=[self.evaluate(self.args.checkpoint,seed,f'zero_s{seed}') for seed in (101,102)]
        baseline_rate=self.rate(baseline)
        self.publish(zero_shot_rate=baseline_rate)
        command,overrides=training_command(self.args,self.directory/'train')
        save(self.directory/'training.json',{'command':command,'overrides':overrides})
        self.publish(status='training');self.process(command,'train',seconds=2400)
        scores=[]
        for checkpoint in screen_checkpoints(self.directory/'train'/NAME/'nn'):
            epoch=int(re.search(r'_ep_(\d+)_',checkpoint.name)[1])
            rows=[self.evaluate(checkpoint,seed,f'epoch{epoch}_s{seed}') for seed in (101,102)]
            scores.append((self.rate(rows),epoch,checkpoint))
        # Highest complete-task rate; prefer the earlier epoch on ties.
        rate,epoch,checkpoint=max(scores,key=lambda item:(item[0],-item[1]))
        source_unchanged=digest(self.args.checkpoint)==self.state['source_sha256']
        if not source_unchanged:raise RuntimeError('Source checkpoint changed during stage')
        self.publish(selected_checkpoint=str(checkpoint),selected_epoch=epoch,validation_rate=rate,
                     candidate_scores=[{'rate':r,'epoch':e,'checkpoint':str(p)} for r,e,p in scores],
                     required_rate=max(self.args.min_rate,baseline_rate-self.args.max_loss),source_unchanged=True)
        if not progression_decision(rate,baseline_rate,self.args.min_rate,self.args.max_loss):
            self.publish(status='stopped_severe_regression',reason='Below relaxed floor or excessive same-size loss; no remedial actions')
            return
        promotion={'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
                   'problem':problem_name(self.args.percent),'policy_coefficient_id':50,
                   'validation_rate':rate,'selection_trials':64,'source_checkpoint':self.args.checkpoint,
                   'source_sha256':self.state['source_sha256'],'source_on_target_rate':baseline_rate,
                   'selected_epoch':epoch,'protocol':'fixed_relaxed_curriculum_v1',
                   'strict_acceptance':False,'heldout_report_only':True}
        save(self.directory/'promotion.json',promotion)
        if self.args.final_report:
            heldout=[self.evaluate(checkpoint,seed,f'final_heldout_s{seed}') for seed in range(601,609)]
            promotion['heldout_rate']=self.rate(heldout)
            promotion['heldout_trials']=256
            save(self.directory/'promotion.json',promotion)
            self.publish(heldout_rate=promotion['heldout_rate'])
        self.publish(status='promoted',reason='Best checkpoint progresses under the user-authorized relaxed protocol; no strict acceptance claim')
        (self.directory/'report.md').write_text('# Fixed curriculum stage\n\n'+
            f'Size: {self.args.percent/100:g}. Selected epoch: {epoch}. Development full-task rate: {rate:.1%}.\n\n'+
            'Relaxed progression only; development seeds used for selection. See promotion.json.\n')


def verified_source(path,expected_protocol=None):
    meta=json.loads(Path(path).read_text())
    state=json.loads((Path(path).parent/'status.json').read_text())
    expected='promoted' if expected_protocol else 'accepted'
    if state['status']!=expected or state.get('child_pid') is not None:
        raise ValueError('Source stage is not finished with the required progression status')
    if meta['checkpoint']!=state['selected_checkpoint']:raise ValueError('Source checkpoint/state mismatch')
    checkpoint=Path(meta['checkpoint'])
    if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
    sha=digest(checkpoint)
    if meta.get('checkpoint_sha256',sha)!=sha:raise ValueError('Source checksum mismatch')
    if expected_protocol and meta.get('protocol')!=expected_protocol:raise ValueError('Wrong source progression protocol')
    return dict(meta,checkpoint=str(checkpoint),checkpoint_sha256=sha)


def run_course(args,runner=None):
    source_path=(ROOT/args.source_accepted).resolve()
    source=verified_source(source_path)
    if scale_percent(source['problem'])!=60:raise ValueError('This fixed course requires accepted 0.60 source')
    # Generate missing physical sizes once; never overwrite existing assets.
    from tools.generate_scaled_beam import generate
    for percent in STAGES:
        config=ROOT/f'isaacsimenvs/cfg/task/PreciseAssemblyBeamScale{percent:03d}.yaml'
        if not config.exists():generate(percent)
    directory=ROOT/'outputs/progressive_beam'/args.run_id
    logs=ROOT/'evaluation_runs/progressive_beam'/args.run_id
    directory.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    lock=(directory.parent/'fixed_curriculum.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'running','pid':os.getpid(),'child_pid':None,'plan':STAGES,'stages':[],
           'started_utc':datetime.now(timezone.utc).isoformat(),'protocol':'fixed_relaxed_curriculum_v1',
           'source_accepted':str(source_path),'min_rate':args.min_rate,'max_loss':args.max_loss,
           'epochs_per_stage':args.epochs,'total_minutes':args.total_minutes,
           'live_log':str(directory/'course.log'),'strict_acceptance':False}
    stopped=[False]
    if runner is None:
        signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
        signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
    deadline=time.monotonic()+args.total_minutes*60
    with (directory/'course.log').open('ab') as course_log:
        def publish(**values):
            state.update(values);state['updated_utc']=datetime.now(timezone.utc).isoformat()
            save(directory/'status.json',state)
            text=json.dumps({'updated_utc':state['updated_utc'],**values},ensure_ascii=False)+'\n'
            course_log.write(text.encode());course_log.flush();print(text,end='',flush=True)
        def stop():return stopped[0] or (directory/'STOP').exists()
        publish(plan=STAGES,source_checkpoint=source['checkpoint'])
        try:
            for percent in STAGES:
                if stop():raise InterruptedError('Course stopped')
                if time.monotonic()>=deadline:raise TimeoutError('Course total budget exhausted')
                stage_id=args.run_id+f'_scale{percent:03d}'
                stage_dir=directory.parent/stage_id
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--percent',str(percent),
                    '--checkpoint',source['checkpoint'],'--run-id',stage_id,'--epochs',str(args.epochs),
                    '--minutes',str(args.minutes),'--min-rate',str(args.min_rate),'--max-loss',str(args.max_loss)]
                if percent==50:command.append('--final-report')
                record={'percent':percent,'run_id':stage_id,'source_metadata':str(source_path),
                        'source_checkpoint':source['checkpoint'],'source_sha256':source['checkpoint_sha256'],
                        'status':'running'}
                state['stages'].append(record)
                publish(active_stage=percent,active_stage_status=str(stage_dir/'status.json'),training_epoch=None)
                save(logs/f'scale{percent:03d}_command.json',{'command':command})
                if runner is not None:code=runner(command)
                else:
                    def tick():
                        p=stage_dir/'status.json'
                        child_state=json.loads(p.read_text()) if p.exists() else {}
                        publish(active_stage_phase=child_state.get('status','starting'),training_epoch=state.get('training_epoch'))
                    tail=['']
                    def output(chunk):
                        tail[0]=(tail[0]+chunk.decode(errors='replace'))[-8192:]
                        matches=re.findall(r'epoch\s+:\s*(\d+)\s*/',tail[0])
                        if matches:state['training_epoch']=int(matches[-1])
                    with (logs/f'scale{percent:03d}.log').open('wb') as stage_log:
                        code=streamed_process(command,[course_log,stage_log],deadline,stop,
                            on_pid=lambda pid:publish(child_pid=pid),on_tick=tick,on_output=output)
                child_state=json.loads((stage_dir/'status.json').read_text())
                record.update(status=child_state['status'],returncode=code,
                              validation_rate=child_state.get('validation_rate'),selected_epoch=child_state.get('selected_epoch'))
                publish(child_pid=None)
                if code:raise RuntimeError(f'Stage {percent} execution failed; no automatic retry')
                if child_state['status']=='stopped_severe_regression':
                    publish(status='stopped_severe_regression',reason=f'Severe regression at {percent/100:g}; later sizes not started')
                    return state
                if child_state['status']!='promoted':raise RuntimeError(f'Unexpected stage state: {child_state["status"]}')
                source_path=stage_dir/'promotion.json';source=verified_source(source_path,'fixed_relaxed_curriculum_v1')
                publish(latest_promotion=str(source_path),latest_checkpoint=source['checkpoint'],
                        latest_scale=percent/100,latest_validation_rate=source['validation_rate'])
                save(directory/'best_model.json',source)
            if runner is None:
                os.link(source['checkpoint'],directory/'best_scale050.pth')
            publish(status='completed',active_stage=None,active_stage_phase=None,
                    final_heldout_rate=source.get('heldout_rate'),
                    reason='All ten fixed sizes progressed under relaxed gates; final heldout is reported separately')
            return state
        except InterruptedError as exc:
            publish(status='stopped',child_pid=None,reason=str(exc));return state
        except BaseException as exc:
            publish(status='failed',child_pid=None,error=repr(exc));raise
        finally:
            (directory/'report.md').write_text('# Fixed beam curriculum\n\nStatus: '+state['status']+
                '\n\nSee status.json, course.log and best_model.json. This is relaxed curriculum progression, not historical strict acceptance.\n')
            lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-accepted',default='outputs/progressive_beam/20261027_scale060_via065/accepted.json')
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--minutes',type=float,default=30)
    parser.add_argument('--total-minutes',type=float,default=240)
    parser.add_argument('--min-rate',type=float,default=.5)
    parser.add_argument('--max-loss',type=float,default=.25)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--percent',type=int)
    parser.add_argument('--checkpoint')
    parser.add_argument('--final-report',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):parser.error('Use one run directory name')
    if args.epochs!=500:parser.error('This fixed v1 recipe uses exactly 500 epochs per size')
    if not all(math.isfinite(x) for x in (args.min_rate,args.max_loss,args.minutes,args.total_minutes)):
        parser.error('Use finite settings')
    if not (0<=args.min_rate<=1 and 0<=args.max_loss<=1 and args.minutes>0 and args.total_minutes>0):
        parser.error('Invalid gates or budgets')
    if args.worker:
        if args.percent not in STAGES or not args.checkpoint or not Path(args.checkpoint).is_file():
            parser.error('Worker requires a planned size and existing checkpoint')
        args.exploration_std_multiplier=1.;args.training_seed=42;args.save_frequency=100
        stage=FixedStage(args)
        try:stage.execute()
        except InterruptedError as exc:stage.publish(status='stopped',reason=str(exc))
        except BaseException as exc:
            stage.publish(status='failed',error=repr(exc));raise
        finally:stage.publish(child_pid=None)
    else:run_course(args)


if __name__=='__main__':main()
