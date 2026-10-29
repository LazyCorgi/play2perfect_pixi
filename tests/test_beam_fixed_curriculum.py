from argparse import Namespace
import json
from pathlib import Path
import sys
import time

import pytest

from tools import run_beam_fixed_curriculum as module


@pytest.mark.parametrize('candidate,baseline,expected',[
    (.70,.88,True),(.50,.70,True),(.49,.50,False),(.55,.85,False),(.60,.85,True),
])
def test_only_severe_regression_blocks_relaxed_progression(candidate,baseline,expected):
    assert module.progression_decision(candidate,baseline)==expected


def test_stage_inherits_best_complete_task_candidate_under_new_gate(tmp_path,monkeypatch):
    source=tmp_path/'source.pth';source.write_bytes(b'source')
    stage=module.FixedStage.__new__(module.FixedStage)
    stage.directory=tmp_path
    stage.args=Namespace(percent=55,checkpoint=str(source),epochs=500,min_rate=.5,max_loss=.25,final_report=False)
    stage.state={'source_sha256':module.digest(source)}
    stage.publish=lambda **values:stage.state.update(values)
    def process(command,label,seconds):
        directory=tmp_path/'train'/module.NAME/'nn';directory.mkdir(parents=True)
        for epoch in (100,200,300,400,500):
            (directory/f'last_0_test_ep_{epoch}_rew_99.pth').write_bytes(str(epoch).encode())
    stage.process=process
    def evaluate(checkpoint,seed,label):
        successes=25 if label.startswith('zero') else {100:20,300:22,500:21}[int(label.split('_')[0][5:])]
        return {'task_success_count_all_envs':successes,'num_envs':32}
    stage.evaluate=evaluate
    stage.execute()
    assert stage.state['status']=='promoted'
    assert stage.state['selected_epoch']==300
    promotion=json.loads((tmp_path/'promotion.json').read_text())
    assert promotion['validation_rate']==44/64
    assert promotion['strict_acceptance'] is False
    assert not (tmp_path/'accepted.json').exists()


@pytest.mark.parametrize('severe_at',[None,58])
def test_course_hands_off_selected_weights_and_stops_without_remedial_branches(tmp_path,monkeypatch,severe_at):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    source_dir=tmp_path/'outputs/progressive_beam/source';source_dir.mkdir(parents=True)
    source=source_dir/'model.pth';source.write_bytes(b'source')
    module.save(source_dir/'accepted.json',{'checkpoint':str(source),'problem':'beam_assembly_step1_scale_060'})
    module.save(source_dir/'status.json',{'status':'accepted','child_pid':None,'selected_checkpoint':str(source)})
    for percent in module.STAGES:
        p=tmp_path/f'isaacsimenvs/cfg/task/PreciseAssemblyBeamScale{percent:03d}.yaml';p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    previous=[str(source)];calls=[]
    def runner(command):
        def value(flag):return command[command.index(flag)+1]
        percent=int(value('--percent'));calls.append(percent)
        assert value('--checkpoint')==previous[0]
        assert '--successful-demo-path' not in command
        directory=tmp_path/'outputs/progressive_beam'/value('--run-id');directory.mkdir()
        if percent==severe_at:
            module.save(directory/'status.json',{'status':'stopped_severe_regression','child_pid':None,'validation_rate':.1})
            return 0
        selected=directory/'selected.pth';selected.write_bytes(str(percent).encode());previous[0]=str(selected)
        module.save(directory/'status.json',{'status':'promoted','child_pid':None,'selected_checkpoint':str(selected),'validation_rate':.6,'selected_epoch':300})
        module.save(directory/'promotion.json',{'checkpoint':str(selected),'checkpoint_sha256':module.digest(selected),
            'problem':f'beam_assembly_step1_scale_{percent:03d}','protocol':'fixed_relaxed_curriculum_v1','validation_rate':.6})
        assert ('--final-report' in command)==(percent==50)
        return 0
    args=Namespace(source_accepted=str(source_dir/'accepted.json'),run_id='course',epochs=500,minutes=30,
                   total_minutes=240,min_rate=.5,max_loss=.25)
    state=module.run_course(args,runner=runner)
    assert calls==(module.STAGES if severe_at is None else [59,58])
    assert state['status']==('completed' if severe_at is None else 'stopped_severe_regression')
    if severe_at is None:
        best=json.loads((tmp_path/'outputs/progressive_beam/course/best_model.json').read_text())
        assert best['checkpoint']==previous[0]


def test_streaming_tees_stdout_and_stderr_without_losing_output(tmp_path,capfd):
    logfile=tmp_path/'live.log'
    with logfile.open('wb') as stream:
        code=module.streamed_process([sys.executable,'-c',
            'import sys; print("stdout-line",flush=True); print("stderr-line",file=sys.stderr,flush=True)'],
            [stream],time.monotonic()+10,lambda:False)
    assert code==0
    assert 'stdout-line' in logfile.read_text() and 'stderr-line' in logfile.read_text()
    assert 'stdout-line' in capfd.readouterr().out


def test_fixed_evaluation_uses_full_task_counts_without_extra_diagnostics(tmp_path):
    stage=module.FixedStage.__new__(module.FixedStage)
    stage.args=Namespace(percent=59);stage.state={'evaluations':[]}
    stage.publish=lambda **values:stage.state.update(values)
    def process(command,label):
        assert '--diagnostics' not in command and '--export-success-demos' not in command
        assert command[command.index('--min-episode-steps')+1]=='0'
        path=tmp_path/'eval.log';path.write_text('RESULT_JSON '+json.dumps({
            'n_unfinished':0,'n_completed_first_episodes':32,'task_success_count_all_envs':20,'num_envs':32})+'\n')
        return path
    stage.process=process
    assert stage.evaluate('checkpoint.pth',101,'zero_s101')['task_success_count_all_envs']==20
