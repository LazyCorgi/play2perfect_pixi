import importlib.util
from pathlib import Path
import unittest

import torch

path = Path(__file__).resolve().parents[1] / 'isaacsimenvs/tasks/play/utils/reward_utils.py'
spec = importlib.util.spec_from_file_location('standalone_rewards', path)
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)


class LiftReferenceTest(unittest.TestCase):
    def test_minimum_reference_rewards_lift_not_initial_free_fall(self):
        # Airborne reset at .63 m is not a grasp. At the minimum .535 m,
        # resting is not rewarded; a later 30 mm lift crosses 25 mm.
        for height in [.63, .58, .535]:
            reward, bonus, lifted = rewards.lifting_reward(
                torch.tensor([height]), torch.tensor([height]), torch.tensor([False]),
                .025, 300., 20., height_offset=0.)
            self.assertEqual(reward.item(), 0.)
            self.assertEqual(bonus.item(), 0.)
            self.assertFalse(lifted.item())
        _, bonus, lifted = rewards.lifting_reward(
            torch.tensor([.565]), torch.tensor([.535]), torch.tensor([False]),
            .025, 300., 20., height_offset=0.)
        self.assertTrue(lifted.item())
        self.assertEqual(bonus.item(), 300.)

    def test_legacy_reference_still_requires_rising_above_airborne_start(self):
        _, bonus, lifted = rewards.lifting_reward(
            torch.tensor([.565]), torch.tensor([.63]), torch.tensor([False]),
            .075, 300., 20.)
        self.assertFalse(lifted.item())
        self.assertEqual(bonus.item(), 0.)
