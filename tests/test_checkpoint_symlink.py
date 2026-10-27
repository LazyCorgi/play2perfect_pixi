import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_games.algos_torch.torch_ext import safe_symlink


class CheckpointSymlinkTest(unittest.TestCase):
    def test_first_checkpoint_does_not_retry_missing_link(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            with patch('rl_games.algos_torch.torch_ext.time.sleep') as sleep:
                safe_symlink('checkpoint.pth', path)
                sleep.assert_not_called()
            self.assertTrue(path.is_symlink())
            self.assertEqual(path.readlink(), Path('checkpoint.pth'))

    def test_broken_checkpoint_link_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            path.symlink_to('missing.pth')
            safe_symlink('new.pth', path)
            self.assertEqual(path.readlink(), Path('new.pth'))
