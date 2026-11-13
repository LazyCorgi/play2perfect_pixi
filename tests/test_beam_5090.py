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
    recipe = runner.plan(96, smoke=True)
    weights = dict(epoch=32, frame=32*96*16, current_rewards=torch.zeros(96,1),central_value_optimizer={})
    torch.save({0:weights},p)
    assert runner.checkpoint_info(p,recipe)['frame'] == recipe['budget_frames']
    weights['frame'] += 16
    torch.save({0:weights},p)
    with pytest.raises(ValueError,match='counters'):
        runner.checkpoint_info(p,recipe)
