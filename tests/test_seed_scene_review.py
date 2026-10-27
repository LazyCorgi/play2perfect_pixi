import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from evaluation.browse_seed_scenes import trace_pose
from evaluation.initial_scenes import save_initial_scenes
from evaluation.replay_assembly_trace import load_trace


class SeedSceneReviewTest(unittest.TestCase):
    def test_snapshot_and_replay_remove_world_origin_once(self):
        origin = torch.tensor([[12., -8., 0.]])
        state = torch.tensor([[12.1, -7.8, .63, 1., 0., 0., 0.]])
        asset = NS(data=NS(root_state_w=state, joint_pos=torch.zeros(1, 1), joint_names=['joint']))
        cfg = NS(precise_assembly=NS(problem='test', goal_mode='finalGoalOnly'), assets=NS(robot_urdf='robot.urdf'))
        env = NS(cfg=cfg, num_envs=1, scene=NS(env_origins=origin), robot=asset, table=asset, object=asset, goal_viz=asset,
                 hole_pos=torch.tensor([[.2, .1, .53]]), hole_quat_wxyz=torch.tensor([[1., 0., 0., 0.]]),
                 _pih_object_urdf_abs='object.urdf', _pih_receptive_urdf_abs='fixture.urdf')
        with tempfile.TemporaryDirectory() as directory:
            saved = save_initial_scenes(env, Path(directory)/'initial.json', seed=43, checkpoint='weights.pth', policy_id=50)
        np.testing.assert_allclose(saved['object_pose'][0][:3], [.1, .2, .63], atol=1e-6)
        np.testing.assert_allclose(saved['fixture_pose'][0][:3], [.2, .1, .53], atol=1e-6)
        cols = ['T_world_object_'+a+'_m' for a in 'xyz'] + ['T_world_object_q'+a for a in 'wxyz']
        replay = trace_pose(state[0].numpy(), dict(zip(cols, range(7))), 'object', origin[0].numpy())
        np.testing.assert_allclose(replay, saved['object_pose'][0])

    def test_trace_rejects_reversed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.npz'
            np.savez(path, trace=np.array([[1.], [.5]]), columns=np.array(['time_s']))
            with self.assertRaisesRegex(ValueError, 'increasing'):
                load_trace(path)

    def test_historical_seed42_trace_is_playable(self):
        path = Path(__file__).resolve().parents[1]/'evaluation_runs/half_scale/original_regression/20261026_160939/tight_insertion/traces/env0000_ep0000.npz'
        if not path.exists():
            self.skipTest('Historical local artifact is not distributed with Git')
        trace, index = load_trace(path)
        self.assertEqual(len(trace), 176)
        self.assertIn('T_world_object_qw', index)
        self.assertGreater(trace[-1, index['time_s']], 2.9)

    def test_strict_audit_accepts_the_original_asset_at_its_nominal_target(self):
        from evaluation.objects import NAME_TO_OBJECT
        from evaluation.problems import PROBLEM_REGISTRY
        from evaluation.peg_insertion_geometry import PegInsertionGeometry
        root = Path(__file__).resolve().parents[1]
        problem = PROBLEM_REGISTRY['tight_insertion']
        audit = PegInsertionGeometry(NAME_TO_OBJECT[problem.insertion_object_name].urdf_path,
                                     root/'assets'/problem.receptive_urdf, problem.final_insert_pose_rel_receptive)
        self.assertTrue(audit.evaluate(np.array(problem.final_insert_pose_rel_receptive))['seated'])


if __name__ == '__main__':
    unittest.main()
