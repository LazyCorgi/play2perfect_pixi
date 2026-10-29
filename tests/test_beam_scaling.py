"""Physical size, goals and train/eval consistency for the new beam course."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from evaluation.beam_scaling import problem_name, scale_percent, task_name
from evaluation.objects import NAME_TO_OBJECT
from evaluation.problems import PROBLEM_REGISTRY
from evaluation.problem_presets import default_goal_mode, default_tolerances, task_for_problem

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('percent', [90, 80, 70, 65, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50])
def test_beam_physical_geometry_and_all_link_inertias(percent):
    from evaluation._viser_demo import _asset_abs, _load_mesh_for_viz, _resolve_problem_assets

    s = percent / 100
    for full, small in zip(_resolve_problem_assets('beam_assembly_step1'), _resolve_problem_assets(problem_name(percent)), strict=True):
        a, b = _asset_abs(full), _asset_abs(small)
        np.testing.assert_allclose(_load_mesh_for_viz(b).bounds, s * _load_mesh_for_viz(a).bounds, atol=1e-9)
        source, scaled = ET.parse(a), ET.parse(b)
        for x, y in zip(source.findall('.//mass'), scaled.findall('.//mass'), strict=True):
            assert float(y.get('value')) == pytest.approx(float(x.get('value')) * s**3)
        for x, y in zip(source.findall('.//inertia'), scaled.findall('.//inertia'), strict=True):
            for key, value in x.attrib.items():
                assert float(y.get(key)) == pytest.approx(float(value) * s**5)
        # Both collision and visual geometry must scale, including link offsets.
        for tag, attr in [('origin','xyz'), ('mesh','scale'), ('box','size')]:
            for x, y in zip(source.findall(f'.//{tag}'), scaled.findall(f'.//{tag}'), strict=True):
                if attr not in x.attrib and tag != 'mesh':
                    continue
                np.testing.assert_allclose(np.fromstring(y.get(attr), sep=' '), s*np.fromstring(x.get(attr, '1 1 1'), sep=' '), atol=1e-12)


@pytest.mark.parametrize('percent', [90, 80, 70, 65, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50])
def test_goals_and_shared_profiles(percent):
    s = percent / 100
    full, small = [PROBLEM_REGISTRY[key] for key in ('beam_assembly_step1', problem_name(percent))]
    for a, b in zip(full.insert_pose_rel_receptive, small.insert_pose_rel_receptive, strict=True):
        np.testing.assert_allclose(b[:3], s*np.array(a[:3]))
        np.testing.assert_allclose(b[3:], a[3:])
    np.testing.assert_allclose(NAME_TO_OBJECT[small.insertion_object_name].scale, s*np.array(NAME_TO_OBJECT[full.insertion_object_name].scale))
    cfg = yaml.safe_load((ROOT/f'isaacsimenvs/cfg/task/PreciseAssemblyBeamScale{percent:03d}.yaml').read_text())
    original = yaml.safe_load((ROOT/'isaacsimenvs/cfg/task/PreciseAssembly.yaml').read_text())
    agent = yaml.safe_load((ROOT/'isaacsimenvs/cfg/train/PreciseAssemblyBeamSAPG.yaml').read_text())['params']['config']
    assert default_tolerances(small.name) == pytest.approx((cfg['precise_assembly']['insertion_success_tolerance'],cfg['precise_assembly']['retract_success_tolerance']))
    assert task_for_problem(small.name) == task_name(percent)
    assert default_goal_mode(small.name) == cfg['precise_assembly']['goal_mode'] == 'preInsertAndFinal'
    assert cfg['scene']['num_envs'] // agent['expl_coef_block_size'] == 6
    np.testing.assert_allclose(cfg['reward']['fixed_size'], s*np.array(original['reward']['fixed_size']))
    assert cfg['precise_assembly']['goal_xy_obs_noise'] == .002
    assert cfg['reset'] == original['reset']
    assert cfg['domain_randomization'] == original['domain_randomization']
    with pytest.raises(ValueError):
        task_for_problem(small.name, 'Isaacsimenvs-PreciseAssemblyHalf-Direct-v0')
    with pytest.raises(ValueError):
        scale_percent('beam_assembly_step1_scale_000')


@pytest.mark.parametrize('percent', [0, -1, 100, 1000, 90.5])
def test_invalid_size_cannot_create_unregistered_assets(percent):
    with pytest.raises(ValueError):
        problem_name(percent)


def test_first_training_segment_uses_source_weights_and_standard_noise():
    from argparse import Namespace
    from tools.beam_curriculum_common import training_command

    command, overrides = training_command(Namespace(percent=90, checkpoint='source.pth', epochs=500), Path('/tmp/beam_test_output'))
    assert command[command.index('--checkpoint_load_mode')+1] == 'weights'
    assert command[command.index('--checkpoint')+1] == 'source.pth'
    assert overrides['env.precise_assembly.goal_mode'] == 'preInsertAndFinal'
    assert overrides['env.precise_assembly.goal_xy_obs_noise'] == .002
    assert not any(key.startswith('env.reward.') for key in overrides)


def test_duplicate_final_save_does_not_bias_middle_checkpoint(tmp_path):
    from tools.beam_curriculum_common import screen_checkpoints
    for epoch in [100,200,300,400,500]:
        (tmp_path/f'last_0_beam_ep_{epoch}_rew_10.pth').touch()
    (tmp_path/'last_0_beam_ep_500_rew__10_.pth').touch()
    chosen=screen_checkpoints(tmp_path)
    assert [int(p.name.split('_ep_')[1].split('_')[0]) for p in chosen] == [100,300,500]
