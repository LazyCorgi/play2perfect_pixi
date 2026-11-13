from pathlib import Path

import pytest

from tools import beam_5090 as runner


@pytest.mark.parametrize('envs,epochs', [(768,212304),(1536,106152),(3072,53076),(6144,26538),(12288,13269)])
def test_exact_official_budget_and_six_groups(envs, epochs):
    p = runner.plan(envs)
    assert p['epochs'] == epochs
    assert envs * 16 * epochs == runner.FRAMES
    assert envs // p['block_size'] == 6
    assert p['minibatch_size'] == envs * 8


@pytest.mark.parametrize('envs', [0,95,128,1000,672])
def test_reject_invalid_budget_divisors(envs):
    with pytest.raises(ValueError):
        runner.plan(envs)


def test_commands_keep_randomization_and_restart_budget(tmp_path):
    recipe = runner.plan(1536)
    first = runner.train_command(tmp_path, recipe)
    resumed = runner.train_command(tmp_path, recipe, tmp_path/'last.pth', True, 1)
    assert first[first.index('--checkpoint')+1] == str(runner.SOURCE)
    assert first[first.index('--checkpoint_load_mode')+1] == 'weights'
    assert resumed[resumed.index('--checkpoint_load_mode')+1] == 'resume_fresh_rollout'
    assert 'agent.params.config.max_epochs=106152' in first
    assert 'agent.params.config.max_epochs=106152' in resumed
    assert not any('domain_randomization' in x for x in first)
    assert f'+agent.params.config.train_dir={tmp_path / "train"}' in resumed


def test_resume_clears_old_physics_rollout_but_keeps_training_state(monkeypatch):
    from rl_games import torch_runner
    original = dict(model={}, optimizer={'state': 'actor'}, central_value_optimizer={'state': 'critic'},
                    epoch=400, frame=400*1536*16, obs='stale', env_state='stale', rnn_states='stale',
                    dones='stale', current_rewards='stale', current_lengths='stale', current_shaped_rewards='stale')
    monkeypatch.setattr(torch_runner, '_load_checkpoint_weights', lambda *args: original)
    captured = {}
    class Agent:
        def set_full_state_weights(self, weights):
            captured.update(weights)
    torch_runner._restore(Agent(), dict(checkpoint='saved.pth', checkpoint_load_mode='resume_fresh_rollout'))
    assert captured['epoch'] == 400 and captured['optimizer'] == original['optimizer']
    assert captured['central_value_optimizer'] == original['central_value_optimizer']
    assert not any(k in captured for k in ['obs','env_state','dones','rnn_states','current_rewards','current_lengths'])
    assert 'obs' in original


def test_selected_checkpoint_counters_must_match_budget(tmp_path):
    import torch
    p = tmp_path/'checkpoint.pth'
    recipe = runner.plan(96, smoke=True, smoke_epochs=32)
    weights = dict(epoch=32, frame=32*96*16, current_rewards=torch.zeros(96,1),central_value_optimizer={})
    torch.save({0:weights},p)
    assert runner.checkpoint_info(p,recipe)['frame'] == recipe['budget_frames']
    weights['frame'] += 16
    torch.save({0:weights},p)
    with pytest.raises(ValueError,match='counters'):
        runner.checkpoint_info(p,recipe)


def test_6144_buffers_have_headroom_over_reported_overflow():
    recipe=runner.plan(6144)
    buffers=recipe['physics_buffers']
    assert buffers['gpu_max_rigid_contact_count']==4194304
    assert buffers['gpu_max_rigid_patch_count']==1048576
    assert buffers['gpu_max_rigid_contact_count'] > 690201 * 4
    assert buffers['gpu_max_rigid_patch_count'] > 189806 * 4
    command=runner.train_command(Path('/tmp/capacity-test'),recipe)
    for key,value in buffers.items():
        assert f'env.sim.physx.{key}={value}' in command
    assert runner.plan(1536)['physics_buffers']['gpu_max_rigid_contact_count'] < buffers['gpu_max_rigid_contact_count']


@pytest.mark.parametrize('line',[
    '[Error] [omni.physx.plugin] PhysX error: Contact buffer overflow detected, please increase its size to at least 690201',
    'PhysX error: Patch buffer overflow detected, please increase its size to at least 189806',
    'contact buffer overflow detected',
    'PhysX error: GPU collision stack exhausted',
])
def test_physics_errors_are_fatal_even_if_process_does_not_crash(line):
    with pytest.raises(runner.PhysicsCapacityError,match='Invalid physics'):
        runner.check_physics_log(line)


def test_harmless_kit_warnings_pass_and_smoke_spans_multiple_timeouts():
    runner.check_physics_log('[Warning] omni.isaac.dynamic_control is deprecated\nCPU profile powersave')
    recipe=runner.plan(6144,smoke=True)
    assert recipe['epochs']==256
    assert recipe['epochs']*16 > 600*6
    assert recipe['budget_frames']==6144*16*256


def test_zero_exit_smoke_with_overflow_is_recorded_as_invalid(tmp_path,monkeypatch):
    import hashlib
    import json
    import sys
    source=tmp_path/'play.pth';source.write_bytes(b'fixture')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'SOURCE',source)
    monkeypatch.setattr(runner,'SOURCE_SHA',hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'test-commit')
    monkeypatch.setattr(sys,'argv',['runner','run','--smoke','--num-envs','96','--run-id','bad-physics'])
    class ExitedProcess:
        pid=123
        returncode=0
        def __init__(self,command,stdout,**kwargs):
            stdout.write('PhysX error: Contact buffer overflow detected\n')
            stdout.flush()
        def poll(self):
            return 0
    monkeypatch.setattr(runner.subprocess,'Popen',ExitedProcess)
    with pytest.raises(runner.PhysicsCapacityError):
        runner.main()
    state=json.loads((tmp_path/'outputs/beam_5090/bad-physics/status.json').read_text())
    assert state['status']=='failed' and state['invalid_physics'] is True
    assert state['child_pid'] is None
