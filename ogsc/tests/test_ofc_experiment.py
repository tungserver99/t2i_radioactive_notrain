import unittest
import tempfile
from pathlib import Path

import numpy as np

from ogsc.ofc_experiment import (_freeze_run_identity, _validate_paired_provenance,
                                 _variant_metrics)
from ogsc.state import write_json, read_json


class OFCExperimentTests(unittest.TestCase):
    def test_png_bit_accuracy_keeps_original_zero_tie_as_positive(self):
        calls = {}

        def aggregate(coordinates, signs, sizes, repeats, seed):
            calls.update(sizes=sizes, repeats=repeats, seed=seed,
                         coordinates=coordinates.copy(), signs=signs.copy())
            return {'rows': [{'images_per_group': size,
                              'full_key_recovery_rate': 0.0,
                              'groups_per_partition': 1}
                             for size in sizes]}

        raw = np.zeros((2, 32), dtype=np.float64)
        signs = np.asarray([1, -1] * 16, dtype=np.int8)
        result = _variant_metrics(raw, signs, aggregate)
        self.assertEqual(result['bit_accuracy_at_1'], 0.5)
        self.assertEqual(result['per_bit_accuracy_32'], [1.0, 0.0] * 16)
        self.assertTrue(np.array_equal(calls['coordinates'], raw))
        self.assertTrue(np.array_equal(calls['signs'], signs))
        self.assertEqual((calls['sizes'], calls['repeats'], calls['seed']),
                         ((1, 8, 32, 64, 256), 100, 20261029))

    def test_paired_provenance_rejects_different_carrier(self):
        with tempfile.TemporaryDirectory() as root:
            baseline = Path(root) / 'baseline'
            bitwise = Path(root) / 'bitwise'
            baseline.mkdir()
            bitwise.mkdir()
            common = {'source_run_sha256': {'prompts': 'same'},
                      'source_files_sha256': {'source.py': 'same'},
                      'carrier_artifact_sha256': {'key.json': 'baseline'},
                      'resolved_model_path': str(Path(root) / 'model')}
            write_json(baseline / 'provenance.json', common)
            bitwise_value = dict(common, carrier_artifact_sha256={'key.json': 'other'})
            write_json(bitwise / 'provenance.json', bitwise_value)
            with self.assertRaisesRegex(RuntimeError, 'carrier'):
                _validate_paired_provenance(baseline, bitwise)

    def test_output_identity_is_frozen_before_resume_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'run'
            first = {'gamma_budget': 0.15, 'carrier': 'a'}
            _freeze_run_identity(output, first)
            original = read_json(output / 'calibration/ofc_validation.json')
            self.assertEqual(_freeze_run_identity(output, first)['identity'], first)
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                _freeze_run_identity(output, {'gamma_budget': 0.15, 'carrier': 'b'})
            self.assertEqual(read_json(output / 'calibration/ofc_validation.json'), original)


if __name__ == '__main__':
    unittest.main()
