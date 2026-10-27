"""Detached, bounded beam-half finetuning with serial evaluation and restart state.

Run inside the Pixi environment. STOP in the run directory requests shutdown.
No agent API, email, or external notification is invoked.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.eval_isaacsim import EVAL_DR_OFF_OVERRIDES

PROBLEM = 'beam_assembly_step1_half'
NAME = '0_overnight_beam_half'  # rl_games parses the numeric prefix as a rank.


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def initialize(source, destination):
    import torch
    from tools.prepare_finetune_checkpoint import prepare_weights
    raw = torch.load(source, map_location='cpu', weights_only=False)
    weights, changes = prepare_weights(raw, prior_count=None, sigma_range=(.4,.1))
    # Adapt observation moments gradually; preserve all value-normalizer counts.
    for section, state in weights.items():
        for key, value in state.items():
            if key.endswith('running_mean_std.count'):
                changes.append({'key':section+'/'+key, 'old_count':float(value), 'new_count':1000000.})
                value.fill_(1000000.)
    torch.save(weights, destination)
    atomic_json(destination.with_suffix('.json'), {'source':str(source),'changes':changes,'value_normalizer_counts':'preserved','weights_only':True})


def checkpoint_state(path):
    import torch
    raw = torch.load(path, map_location='cpu', weights_only=False)
    state = raw.get(0, raw)
    return raw, state


def resume_copy(source, destination):
    import torch
    raw, state = checkpoint_state(source)
    epoch = int(state['epoch'])
    # A new Isaac Sim process has new physical states. Preserve optimizer,
    # moments and counters, but never restore observations/RNN from an old scene.
    for key in ['obs','rnn_states','dones','current_rewards','current_shaped_rewards','current_lengths','env_state']:
        state.pop(key, None)
    torch.save(raw, destination)
    return epoch


def training_command(checkpoint, mode, output, target_epoch, num_envs, noise):
    command = [sys.executable,'-u','isaacsimenvs/train.py','--task','Isaacsimenvs-PreciseAssemblyHalf-Direct-v0',
               '--agent','rl_games_sapg_cfg_entry_point','--checkpoint',str(checkpoint),
               '--checkpoint_load_mode',mode,'--headless']
    overrides = dict(EVAL_DR_OFF_OVERRIDES)
    overrides.update({
        'env.precise_assembly.problem':PROBLEM,
        'env.precise_assembly.goal_mode':'finalGoalOnly',
        'env.precise_assembly.goal_xy_obs_noise':noise,
        'env.precise_assembly.goal_yaw_obs_noise_deg':0.,
        'env.precise_assembly.force_lift_reward_active':True,
        'env.reward.use_minimum_height_reference':True,
        'env.reward.lifting_bonus':50.,
        'env.reward.lifting_rew_scale':5.,
        'env.reward.keypoint_rew_scale':400.,
        'env.episode_length_s':20.,
        'env.termination.episode_length':1200,
        'env.scene.num_envs':num_envs,
        'agent.params.seed':42,
        'agent.params.config.name':NAME,
        'agent.params.config.full_experiment_name':NAME,
        'agent.params.config.expl_coef_block_size':num_envs//6,
        'agent.params.config.minibatch_size':num_envs*16//3,
        'agent.params.config.central_value_config.minibatch_size':num_envs*16//3,
        'agent.params.config.learning_rate':5e-6,
        'agent.params.config.lr_schedule':'constant',
        'agent.params.config.central_value_config.learning_rate':5e-5,
        'agent.params.config.max_epochs':target_epoch,
        'agent.params.config.save_frequency':500,
        'agent.params.config.uniform_checkpoints':True,
        'agent.params.config.save_best_after':0,
        'hydra.run.dir':str(output),
    })
    command += [key+'='+(value if isinstance(value,str) else json.dumps(value,separators=(',',':'))) for key,value in overrides.items()]
    return command, overrides


class Supervisor:
    def __init__(self, args):
        self.args=args
        self.directory=ROOT/'outputs/overnight_beam_half'/args.run_id
        self.logs=ROOT/'evaluation_runs/overnight_beam_half'/args.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.lock=(self.directory/'run.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.start=time.time()
        self.deadline=self.start+args.hours*3600
        self.state={'run_id':args.run_id,'pid':os.getpid(),'problem':PROBLEM,'started_utc':datetime.now(timezone.utc).isoformat(),
                    'deadline_utc':datetime.fromtimestamp(self.deadline,timezone.utc).isoformat(),
                    'status':'initializing','num_envs':args.num_envs,'epoch':0,'evaluations':[], 'restarts':0}
        self.stop_requested=False
        self.child=None
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop_requested',True))
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop_requested',True))
        self.publish()

    def publish(self, **values):
        self.state.update(values)
        self.state['updated_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(self.directory/'status.json',self.state)

    def stop(self):
        return self.stop_requested or (self.directory/'STOP').exists()

    def run_process(self, command, log_path, *, deadline):
        env=os.environ.copy();env['OMNI_KIT_ACCEPT_EULA']='YES';env['PYTHONUNBUFFERED']='1'
        with log_path.open('w') as log:
            self.child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.publish(child_pid=self.child.pid,active_log=str(log_path))
            reason=None
            while self.child.poll() is None:
                if self.stop():reason='stop_requested'
                elif time.time()>deadline:reason='time_limit'
                elif shutil.disk_usage(self.directory).free<8*1024**3:reason='low_disk'
                elif time.time()-log_path.stat().st_mtime>1200:reason='stalled_log'
                if reason:
                    os.killpg(self.child.pid,signal.SIGTERM)
                    try:self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
                    break
                self.publish()
                time.sleep(5)
            code=self.child.returncode
            self.child=None
            self.publish(child_pid=None,last_process_returncode=code,last_process_stop_reason=reason)
            return code,reason

    def evaluate(self, checkpoint, seed, policy, label):
        # Always evaluate the unmodified standard half task: 10-second resettable
        # window and original +/-2 mm goal noise. Training easing is not scoring.
        directory=self.logs/'evaluation'/label
        log=self.logs/(label+'.log')
        command=[sys.executable,'-u','evaluation/eval_offline.py','--problem',PROBLEM,'--checkpoint-path',str(checkpoint),
                 '--num-envs','32','--steps','3000','--seed',str(seed),'--min-episode-steps','0','--goal-mode','finalGoalOnly',
                 '--policy-coefficient-id',str(policy),'--episode-length-s','10','--diagnostics',
                 '--diagnostics-trace-envs','','--diagnostics-dir',str(directory)]
        self.publish(status='evaluating',checkpoint=str(checkpoint),evaluation_label=label)
        code,reason=self.run_process(command,log,deadline=min(self.deadline,time.time()+600))
        if code or reason:return None
        rows=[json.loads(line[12:]) for line in log.read_text().splitlines() if line.startswith('RESULT_JSON ')]
        if not rows:return None
        row=rows[-1];self.state['evaluations'].append(row);self.publish()
        return row

    def save_best(self, checkpoint, policy, rows, *, block):
        total=sum(r['num_envs'] for r in rows);success=sum(r['task_success_count_all_envs'] for r in rows)
        score=success/total
        if score <= self.state.get('best_validation_rate',-1):return
        best=self.directory/'best_evaluated';best.mkdir(exist_ok=True)
        # This is an experimental candidate, not an automatic deployment.
        shutil.copy2(checkpoint,best/'model.pth')
        output=Path(checkpoint).parents[2]
        if (output/'config.yaml').exists():shutil.copy2(output/'config.yaml',best/'config.yaml')
        meta={'problem':PROBLEM,'source_checkpoint':str(checkpoint),'policy_coefficient_id':policy,
              'validation_successes':success,'validation_trials':total,'seeds':[r['seed'] for r in rows],
              'status':'experimental_not_deployed','block':block}
        atomic_json(best/'metadata.json',meta)
        self.publish(best_validation_rate=score,best_checkpoint=str(checkpoint),best_policy=policy)

    def execute(self):
        source=ROOT/'pretrained_assembly/beam_assembly_step1/model.pth'
        initializer=self.directory/'initializer.pth'
        initialize(source,initializer)
        atomic_json(self.directory/'plan.json',{'arguments':vars(self.args),
            'training':'192 env (fallback 96 on initial OOM), true half geometry, official beam initialization, minimum-height lift reference, weaker lift bonus, constant small actor LR, 20-second resettable training window',
            'selection':'standard-task validation at seeds 43 and 44; fixed policy branch per rollout; reserve seeds 61-64 for final test',
            'continuation':'resume optimizer and epoch counters; clear stale simulator observations and recurrent state at process boundary',
            'wakeup':'no agent wakeup endpoint available; local supervisor handles training, evaluation and report',
            'deployment':'no automatic replacement of deployment weights'})
        checkpoint=initializer;epoch=0;noise=0.;block=0;mode='weights';failures=0
        while epoch<self.args.max_epochs and time.time()<self.deadline-600 and not self.stop():
            block+=1;target=min(epoch+self.args.block_epochs,self.args.max_epochs)
            output=self.directory/f'block_{block:03d}'
            log=self.logs/f'train_{block:03d}.log'
            if mode=='resume':
                resume=self.directory/f'resume_{block:03d}.pth'
                epoch=resume_copy(checkpoint,resume);checkpoint=resume
                target=min(epoch+self.args.block_epochs,self.args.max_epochs)
            command,overrides=training_command(checkpoint,mode,output,target,self.state['num_envs'],noise)
            atomic_json(self.directory/f'block_{block:03d}_command.json',{'command':command,'overrides':overrides})
            self.publish(status='training',block=block,epoch=epoch,target_epoch=target,training_goal_noise_m=noise,output_dir=str(output))
            code,reason=self.run_process(command,log,deadline=self.deadline-600)
            nn=output/NAME/'nn'
            candidates=list(nn.glob('*.pth')) if nn.exists() else []
            periodic=[p for p in candidates if re.search(r'_ep_\d+_',p.name)]
            periodic.sort(key=lambda p:int(re.search(r'_ep_(\d+)_',p.name).group(1)))
            rolling=output/NAME/'last/model.pth'
            available=periodic+([rolling] if rolling.exists() else [])
            valid_latest=None
            for candidate in sorted(available,key=lambda p:p.stat().st_mtime,reverse=True):
                try:
                    _,loaded=checkpoint_state(candidate)
                    loaded_epoch=int(loaded['epoch']);del loaded
                    valid_latest=(candidate,loaded_epoch);break
                except Exception as exc:
                    self.publish(checkpoint_warning=f'{candidate}: {exc!r}')
            if valid_latest and not periodic:periodic=[valid_latest[0]]
            if not periodic:
                text=log.read_text(errors='replace')
                if self.state['num_envs']==192 and epoch==0 and ('out of memory' in text.lower() or 'CUDA_ERROR_OUT_OF_MEMORY' in text):
                    self.publish(num_envs=96,restarts=self.state['restarts']+1);mode='weights';checkpoint=initializer;continue
                failures+=1;self.publish(restarts=self.state['restarts']+1)
                if failures>=3 or reason in ['time_limit','stop_requested','low_disk']:break
                continue
            if valid_latest is None:
                self.publish(status='failed',error='No readable restart checkpoint');break
            checkpoint,epoch=valid_latest
            self.publish(epoch=epoch,latest_checkpoint=str(checkpoint))
            if reason in ['time_limit','stop_requested','low_disk']:break
            if code:
                failures+=1;self.publish(restarts=self.state['restarts']+1)
                if failures>=3:break
            else:failures=0
            # Retain best reward, a middle periodic snapshot, and last; selection
            # itself uses task completion, not reward.
            choices=[checkpoint,periodic[len(periodic)//2]]
            best_reward=nn/(NAME+'.pth')
            if best_reward.exists():choices.append(best_reward)
            choices=list(dict.fromkeys(choices))
            screen=[]
            for j,candidate in enumerate(choices):
                for policy in [50,0]:
                    if self.stop() or time.time()>self.deadline-300:break
                    row=self.evaluate(candidate,43,policy,f'b{block:03d}_c{j}_p{policy}_s43')
                    if row:screen.append((candidate,policy,row))
            screen.sort(key=lambda item:item[2]['task_success_count_all_envs'],reverse=True)
            for j,(candidate,policy,row) in enumerate(screen[:2]):
                if self.stop() or time.time()>self.deadline-240:break
                other=self.evaluate(candidate,44,policy,f'b{block:03d}_confirm{j}_p{policy}_s44')
                if other:self.save_best(candidate,policy,[row,other],block=block)
            best_rate=self.state.get('best_validation_rate',0)
            noise=.001 if best_rate>=.4 else (.0005 if best_rate>=.25 else 0.)
            mode='resume'
        self.publish(status='final_evaluation',epoch=epoch)
        best=self.state.get('best_checkpoint')
        held=[]
        if best and not self.stop():
            for seed in [61,62,63,64]:
                if time.time()>self.deadline-60:break
                row=self.evaluate(best,seed,self.state['best_policy'],f'final_seed{seed}')
                if row:held.append(row)
        report={'run_id':self.args.run_id,'epoch':epoch,'heldout_runs':held,
                'heldout_successes':sum(r['task_success_count_all_envs'] for r in held),
                'heldout_trials':sum(r['num_envs'] for r in held),'state':self.state}
        atomic_json(self.directory/'completion.json',report)
        (self.directory/'report.md').write_text(
            f'# Overnight beam-half run\n\nRun: {self.args.run_id}\n\nCompleted epoch: {epoch}\n\n'
            f'Standard-task held-out successes: {report["heldout_successes"]}/{report["heldout_trials"]}\n\n'
            f'Best validation rate: {self.state.get("best_validation_rate","not available")}\n\n'
            'See status.json, block command manifests, resolved configs and evaluation logs. '
            'No policy was deployed. A missing final evaluation is not a zero-success result.\n')
        terminal = 'stopped' if self.stop() else ('failed' if failures>=3 or epoch==0 else 'completed')
        self.publish(status=terminal,epoch=epoch,completion_file=str(self.directory/'completion.json'))
        (self.directory/'READY_FOR_REVIEW').write_text('Training supervisor finished; read report.md and completion.json.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--hours',type=float,default=12.)
    parser.add_argument('--max-epochs',type=int,default=120000)
    parser.add_argument('--block-epochs',type=int,default=5000)
    parser.add_argument('--num-envs',type=int,choices=[96,192],default=192)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.hours<=0 or args.max_epochs<=0 or args.block_epochs<=0:
        parser.error('Use a simple run ID and positive limits')
    if args.dry_run:
        command,overrides=training_command('INITIALIZER','weights',Path('OUTPUT'),args.block_epochs,args.num_envs,0.)
        print(json.dumps({'arguments':vars(args),'command':command,'overrides':overrides},indent=2));return
    supervisor=Supervisor(args)
    try:supervisor.execute()
    except BaseException as exc:
        supervisor.publish(status='failed',error=repr(exc))
        if supervisor.child is not None and supervisor.child.poll() is None:
            os.killpg(supervisor.child.pid,signal.SIGTERM)
        raise


if __name__=='__main__':main()
