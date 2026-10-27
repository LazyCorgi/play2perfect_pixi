import unittest

from tools.summarize_finetuning import summarize, wilson_interval


class FinetuningSummaryTest(unittest.TestCase):
    def test_unfinished_and_early_failures_stay_in_denominator(self):
        result = summarize([dict(num_envs=8, n_completed_first_episodes=4,
                                 task_success_count_all_envs=2, seed=43)])
        self.assertEqual(result['task_success_rate'], .25)
        self.assertFalse(result['observed_above_40_percent'])

    def test_point_estimate_and_confidence_are_separate(self):
        low, high = wilson_interval(41, 100)
        self.assertLess(low, .4)
        self.assertGreater(high, .4)
        self.assertGreater(wilson_interval(60, 100)[0], .4)
