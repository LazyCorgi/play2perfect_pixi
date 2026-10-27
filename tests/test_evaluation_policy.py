import unittest
from types import SimpleNamespace

import torch

from evaluation.policy_evaluation import use_evaluation_policy


class EvaluationPolicyTest(unittest.TestCase):
    def test_all_parallel_envs_use_the_requested_learned_row(self):
        player = SimpleNamespace(
            expl_type='mixed_expl_learn_param',
            intr_reward_coef_embd=torch.linspace(50, 0, 32).reshape(-1, 1),
            model=SimpleNamespace(a2c_network=SimpleNamespace(param_ids=torch.linspace(50, 0, 6))),
        )
        use_evaluation_policy(player)
        self.assertTrue(torch.equal(player.intr_reward_coef_embd, torch.full((32, 1), 50.)))
        with self.assertRaises(ValueError):
            use_evaluation_policy(player, 12.)

    def test_ordinary_ppo_does_not_get_an_extra_id(self):
        player = SimpleNamespace(expl_type='none', intr_reward_coef_embd=None)
        use_evaluation_policy(player)
        self.assertIsNone(player.intr_reward_coef_embd)
