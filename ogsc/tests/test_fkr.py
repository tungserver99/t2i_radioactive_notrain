import importlib.util
import unittest
from pathlib import Path

import numpy as np


class OriginalFKRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        path = root / 'BEST_TEACHER_32bits_refine/server21/files/var/lib/structure32_student12500/key_metrics.py'
        spec = importlib.util.spec_from_file_location('_test_original_key_metrics', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_exact_full_key_and_one_bit_failure(self):
        signs = np.ones(32)
        all_correct = self.module.aggregate_metrics(np.ones((256, 32)), signs)
        self.assertEqual([r['images_per_group'] for r in all_correct['rows']], [1, 8, 32, 64, 256])
        self.assertTrue(all(r['full_key_recovery_rate'] == 1 for r in all_correct['rows']))
        one_wrong = np.ones((256, 32)); one_wrong[:, 31] = -1
        failed = self.module.aggregate_metrics(one_wrong, signs)
        self.assertTrue(all(r['full_key_recovery_rate'] == 0 for r in failed['rows']))

    def test_group_mean_precedes_decode_and_fixed_protocol(self):
        signs = np.ones(32)
        q = np.ones((256, 32)); q[:128, 0] = -3
        metrics = self.module.aggregate_metrics(q, signs)
        row = next(x for x in metrics['rows'] if x['images_per_group'] == 8)
        self.assertEqual(row['partitions'], 100)
        self.assertEqual(row['groups_per_partition'], 32)
        self.assertEqual(metrics['grouping_seed'], 20261029)
        self.assertIn('mean normalized carrier coordinates', metrics['decode'])

    def test_group_uses_coordinate_mean_not_majority_vote(self):
        signs = np.ones(32)
        q = np.ones((8, 32)); q[0, 0] = -10
        report = self.module.aggregate_metrics(q, signs)
        row = next(x for x in report['rows'] if x['images_per_group'] == 8)
        self.assertEqual(row['bit_accuracy'], 31 / 32)
        self.assertEqual(row['full_key_recovery_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()
