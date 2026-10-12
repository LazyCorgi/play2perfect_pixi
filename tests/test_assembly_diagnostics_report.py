import tempfile
import unittest
from pathlib import Path

from evaluation.report_assembly_diagnostics import CATEGORIES, PHASES, _normalize_episode


def episode(problem="tight_insertion", attempts=1, recovery=0, success=True):
    return {
        "problem": problem,
        "env_id": 0,
        "episode": 0,
        "completed": True,
        "duration_s": 2.0,
        "task_success": success,
        "assembly_success": success,
        "insertion_attempt_count": attempts,
        "insertion_retry_count": max(0, attempts - 1) if attempts is not None else None,
        "drop_count": 0,
        "regrasp_count": 0,
        "recovery_event_count": recovery,
        "local_adjustment_count": 4,
        "contact_metrics": {
            phase: {category: {} for category in CATEGORIES} for phase in PHASES
        },
    }


class DiagnosticsClassificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metadata = {"schema_version": 3}

    def tearDown(self):
        self.temp.cleanup()

    def normalize(self, value):
        return _normalize_episode(value, [], self.metadata, self.root)

    def test_local_adjustment_stays_first_pass(self):
        row = self.normalize(episode())
        self.assertTrue(row["first_pass_success"])
        self.assertEqual(row["local_adjustment_count"], 4)

    def test_backtrack_retry_is_recovered_success(self):
        row = self.normalize(episode(attempts=2, recovery=1))
        self.assertFalse(row["first_pass_success"])
        self.assertEqual(row["result_code"], "recovered_success")

    def test_drop_and_regrasp_are_independent(self):
        value = episode(recovery=1)
        value.update(drop_count=1, regrasp_count=1)
        row = self.normalize(value)
        self.assertFalse(row["recovery_free_success"])
        self.assertEqual(row["drop_count"], 1)
        self.assertEqual(row["regrasp_count"], 1)

    def test_screwing_candidates_are_not_forced_into_attempts(self):
        value = episode(problem="screwing", attempts=None)
        value["screw_metrics"] = {
            "raw_backtrack_candidate_count": 6,
            "raw_reengagement_candidate_count": 6,
        }
        row = self.normalize(value)
        self.assertIsNone(row["first_pass_success"])
        self.assertIsNone(row["recovery_free_success"])
        self.assertEqual(row["result_code"], "successful_unclassified")


if __name__ == "__main__":
    unittest.main()
