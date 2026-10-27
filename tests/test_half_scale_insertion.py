"""Check physical scaling and independent insertion auditing without Isaac Sim."""

import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from evaluation.objects import NAME_TO_OBJECT
from evaluation.problems import PROBLEM_REGISTRY
from evaluation.peg_insertion_geometry import PegInsertionGeometry, collision_boxes
from evaluation.problem_presets import HALF_TASK, default_goal_mode, default_tolerances, task_for_problem

ROOT = Path(__file__).resolve().parents[1]


class HalfScaleInsertionTest(unittest.TestCase):
    def setUp(self):
        self.original = PROBLEM_REGISTRY["tight_insertion"]
        self.half = PROBLEM_REGISTRY["tight_insertion_half"]
        self.original_object = NAME_TO_OBJECT[self.original.insertion_object_name]
        self.half_object = NAME_TO_OBJECT[self.half.insertion_object_name]

    def test_geometry_and_inertia_are_scaled_together(self):
        for original, half in (
            (self.original_object.urdf_path, self.half_object.urdf_path),
            (ROOT / "assets" / self.original.receptive_urdf, ROOT / "assets" / self.half.receptive_urdf),
        ):
            for (center, rotation, size), (small_center, small_rotation, small_size) in zip(collision_boxes(original), collision_boxes(half), strict=True):
                np.testing.assert_allclose(small_center, center * .5)
                np.testing.assert_allclose(small_size, size * .5)
                np.testing.assert_allclose(small_rotation, rotation)
            source, scaled = ET.parse(original), ET.parse(half)
            self.assertAlmostEqual(float(scaled.find('.//mass').get('value')), float(source.find('.//mass').get('value')) / 8)
            for key, value in source.find('.//inertia').attrib.items():
                self.assertAlmostEqual(float(scaled.find('.//inertia').get(key)), float(value) / 32)
            for a, b in zip(source.findall('.//inertial/origin'), scaled.findall('.//inertial/origin'), strict=True):
                np.testing.assert_allclose(np.fromstring(b.get('xyz'), sep=' '), .5 * np.fromstring(a.get('xyz'), sep=' '))

    def test_baseline_and_scaled_goal_semantics(self):
        self.assertEqual(self.original.final_insert_pose_rel_receptive[2], .136)
        self.assertEqual(self.half.final_insert_pose_rel_receptive[2], .068)
        for original, half in zip(self.original.insert_pose_rel_receptive, self.half.insert_pose_rel_receptive, strict=True):
            np.testing.assert_allclose(half[:3], .5 * np.asarray(original[:3]))
            np.testing.assert_allclose(half[3:], original[3:])
        np.testing.assert_allclose(self.half_object.scale, .5 * np.asarray(self.original_object.scale))

    def test_geometry_rejects_wall_penetration_and_hovering(self):
        audit = PegInsertionGeometry(self.half_object.urdf_path, ROOT / "assets" / self.half.receptive_urdf, self.half.final_insert_pose_rel_receptive)
        pose = np.asarray(self.half.final_insert_pose_rel_receptive).copy()
        self.assertTrue(audit.evaluate(pose)['seated'])
        shifted = pose.copy(); shifted[0] += .001
        metrics = audit.evaluate(shifted)
        self.assertFalse(metrics['seated'])
        self.assertGreater(metrics['box_overlap_m'], .00005)
        shifted = pose.copy(); shifted[2] += .005
        self.assertFalse(audit.evaluate(shifted)['seated'])
        # The nominal 0.25 mm side clearance admits a 0.2 mm shift.
        shifted = pose.copy(); shifted[0] += .0002
        self.assertTrue(audit.evaluate(shifted)['seated'])

    def test_eval_and_training_profiles_agree(self):
        cfg = yaml.safe_load((ROOT / 'isaacsimenvs/cfg/task/PreciseAssemblyHalf.yaml').read_text())
        agent = yaml.safe_load((ROOT / 'isaacsimenvs/cfg/train/PreciseAssemblyHalfSAPG.yaml').read_text())['params']['config']
        self.assertEqual(task_for_problem('tight_insertion_half'), HALF_TASK)
        self.assertEqual(default_goal_mode('tight_insertion_half'), 'finalGoalOnly')
        self.assertEqual(default_goal_mode('tight_insertion'), 'preInsertAndFinal')
        with self.assertRaises(ValueError):
            task_for_problem('tight_insertion', HALF_TASK)
        self.assertEqual(default_tolerances('tight_insertion'), (.01, .005))
        self.assertEqual(default_tolerances('tight_insertion_half'), (cfg['precise_assembly']['insertion_success_tolerance'], cfg['precise_assembly']['retract_success_tolerance']))
        self.assertEqual(cfg['scene']['num_envs'] // agent['expl_coef_block_size'], 6)
        self.assertEqual((cfg['scene']['num_envs'] * agent['horizon_length']) % agent['minibatch_size'], 0)

    def test_original_mass_ablation_preserves_geometry_and_scales_inertia(self):
        heavy = NAME_TO_OBJECT['lpeg_matchedmass_half_original_mass']
        original = ET.parse(self.original_object.urdf_path)
        ablation = ET.parse(heavy.urdf_path)
        self.assertEqual(float(ablation.find('.//mass').get('value')), float(original.find('.//mass').get('value')))
        for key, value in original.find('.//inertia').attrib.items():
            self.assertAlmostEqual(float(ablation.find('.//inertia').get(key)), float(value) / 4)
        for a, b in zip(collision_boxes(heavy.urdf_path), collision_boxes(self.half_object.urdf_path), strict=True):
            for x, y in zip(a, b, strict=True):
                np.testing.assert_allclose(x, y)

    def test_beam_mesh_scale_matches_physical_half_size(self):
        from evaluation._viser_demo import _asset_abs, _load_mesh_for_viz, _resolve_problem_assets
        for original, half in zip(_resolve_problem_assets('beam_assembly_step1'), _resolve_problem_assets('beam_assembly_step1_half'), strict=True):
            full_mesh = _load_mesh_for_viz(_asset_abs(original))
            small_mesh = _load_mesh_for_viz(_asset_abs(half))
            np.testing.assert_allclose(small_mesh.bounds, full_mesh.bounds * .5, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
