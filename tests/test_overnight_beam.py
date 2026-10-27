import tempfile
from pathlib import Path
import unittest

import torch

from tools.run_overnight_beam import NAME, initialize, resume_copy, training_command


class OvernightBeamTest(unittest.TestCase):
    def test_initializer_preserves_value_counts_and_actor_weights(self):
        weights = {'model': {'a2c_network.sigma':torch.zeros(6,29),
                             'running_mean_std.count':torch.tensor(1e11,dtype=torch.float64),
                             'value_mean_std.count':torch.tensor(1.,dtype=torch.float64),
                             'actor.weight':torch.tensor([2.,3.])}}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.pth';dest=Path(directory)/'init.pth'
            torch.save({0:weights},source);initialize(source,dest)
            prepared=torch.load(dest,weights_only=False)['model']
        self.assertEqual(prepared['running_mean_std.count'].item(),1e6)
        self.assertEqual(prepared['value_mean_std.count'].item(),1.)
        torch.testing.assert_close(prepared['actor.weight'],weights['model']['actor.weight'])

    def test_resume_keeps_optimizer_but_clears_stale_physical_observation(self):
        state={'epoch':5000,'frame':123,'optimizer':{'param_groups':[{'lr':5e-6}]},
               'model':{'weight':torch.ones(1)},'obs':torch.ones(2,141),
               'rnn_states':torch.ones(1),'current_rewards':torch.ones(2),'env_state':{'old':True}}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.pth';dest=Path(directory)/'resume.pth'
            torch.save({0:state},source)
            self.assertEqual(resume_copy(source,dest),5000)
            restored=torch.load(dest,weights_only=False)[0]
            original=torch.load(source,weights_only=False)[0]
        self.assertIn('obs',original)
        for key in ['obs','rnn_states','current_rewards','env_state']:self.assertNotIn(key,restored)
        self.assertEqual(restored['optimizer'],state['optimizer'])
        self.assertEqual(restored['frame'],123)

    def test_command_retains_six_groups_and_numeric_experiment_prefix(self):
        _,cfg=training_command('weights.pth','weights',Path('/tmp/run'),5000,192,0.)
        self.assertEqual(int(NAME.split('_')[0]),0)
        self.assertEqual(192//cfg['agent.params.config.expl_coef_block_size'],6)
        self.assertEqual(cfg['env.precise_assembly.problem'],'beam_assembly_step1_half')
        self.assertTrue(cfg['agent.params.config.uniform_checkpoints'])
        self.assertNotIn('env.precise_assembly.insertion_success_tolerance',cfg)


if __name__=='__main__':unittest.main()
