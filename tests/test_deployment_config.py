import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DeploymentConfigTest(unittest.TestCase):
    def test_loading_config_does_not_import_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('train:\n  params:\n    config:\n      device: cuda:0\n      device_name: cuda:0\n')
            code = """
import sys
from deployment.rl_player_utils import read_cfg
cfg = read_cfg(sys.argv[1], device='cpu')
assert cfg['train']['params']['config']['device'] == 'cpu'
assert 'isaacsimenvs' not in sys.modules
assert 'isaacsim' not in sys.modules
"""
            env = os.environ.copy()
            env.pop('OMNI_KIT_ACCEPT_EULA', None)
            result = subprocess.run([sys.executable, '-c', code, str(config)], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
