import unittest

import torch

from tools.prepare_finetune_checkpoint import prepare_weights


class FinetuneInitializerTest(unittest.TestCase):
    def test_changes_priors_without_changing_actor_or_source(self):
        model = {'running_mean_std.count': torch.tensor(1e11, dtype=torch.float64),
                 'running_mean_std.running_mean': torch.tensor([1., 2.]),
                 'a2c_network.sigma': torch.ones(6, 2),
                 'a2c_network.mu.weight': torch.ones(2, 3)}
        prepared, changes = prepare_weights({0: {'model': model, 'optimizer': {'unused': True}}})
        self.assertNotIn('optimizer', prepared)
        self.assertEqual(prepared['model']['running_mean_std.count'].item(), 100000.)
        self.assertEqual(model['running_mean_std.count'].item(), 1e11)
        self.assertTrue(torch.equal(model['a2c_network.mu.weight'], prepared['model']['a2c_network.mu.weight']))
        self.assertTrue(torch.equal(model['running_mean_std.running_mean'], prepared['model']['running_mean_std.running_mean']))
        self.assertAlmostEqual(prepared['model']['a2c_network.sigma'][0, 0].exp().item(), .35, places=6)
