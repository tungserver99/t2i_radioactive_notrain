import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ogsc.run import _parser
from ogsc.direction_probe import (
    _accuracy_summary,
    _read_full_scores,
    _score_identity_matches,
    classify_direction_evidence,
    evenly_spaced_indices,
    interpolate_small_latent,
)


class DirectionProbeLogicTests(unittest.TestCase):
    def test_launcher_has_no_carriage_returns_and_one_python_command(self):
        launcher = Path(__file__).resolve().parents[2] / 'run_ogsc32_direction_probe.sh'
        contents = launcher.read_bytes()
        self.assertNotIn(b'\r', contents)
        self.assertEqual(contents.decode('utf-8').strip(), 'python -m ogsc.run direction-probe')

    def test_diagnostic_has_a_separate_cli_command(self):
        args = _parser().parse_args(["direction-probe", "--offload"])
        self.assertEqual(args.command, "direction-probe")
        self.assertTrue(args.offload)
        self.assertIsNone(args.baseline_run)

    def test_evenly_spaced_selection_is_unique_and_covers_endpoints(self):
        indices = evenly_spaced_indices(1000, 64)
        self.assertEqual(len(indices), 64)
        self.assertEqual(len(set(indices)), 64)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 999)
        gaps = [b - a for a, b in zip(indices, indices[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_small_gamma_interpolation_preserves_fp16_storage(self):
        clean = torch.zeros((1, 4, 2, 2), dtype=torch.float16)
        marked = torch.full_like(clean, 16.0)
        small = interpolate_small_latent(clean, marked, gamma_old=1.0, gamma_small=1.0 / 16)
        self.assertEqual(small.dtype, torch.float16)
        self.assertTrue(torch.equal(small, torch.ones_like(small)))

    def test_accuracy_summary_uses_original_zero_tie_break(self):
        summary = _accuracy_summary(np.zeros((1, 2)), np.array([1, -1]))
        self.assertEqual(summary["mean_correct_bits_per_image"], 1.0)

    def test_score_identity_allows_subsecond_drive_mtime_rounding(self):
        expected = {'label': 'teacher_clean_validation', 'metadata': [], 'generation': None,
                    'files': [['sample_0000.png', 123, 1700000000123000000]]}
        current = {'label': 'teacher_clean_validation', 'metadata': [], 'generation': None,
                   'files': [['sample_0000.png', 123, 1700000000987000000]]}
        self.assertTrue(_score_identity_matches(expected, current))
        current['files'][0][1] = 124
        self.assertFalse(_score_identity_matches(expected, current))

    def test_cached_scores_reject_a_stale_png_identity_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'images'
            image_dir.mkdir()
            (image_dir / 'sample_0000.png').write_bytes(b'not-hashed-test-png')
            score = root / 'teacher_clean_validation.json'
            score.write_text(json.dumps([{'file': 'sample_0000.png', 'coordinates': [0.0] * 32}]))
            score.with_suffix('.identity.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                _read_full_scores(score, [{'image_id': 1, 'text': 'p'}],
                                  'teacher_clean_validation', image_dir)

    def test_evidence_classifier_identifies_overshoot(self):
        failed = [3, 5, 13, 14, 17, 31, 32]
        diagnosis = classify_direction_evidence(
            failed_bits=failed,
            surrogate_shifts={bit: 0.2 for bit in failed},
            small_png_shifts={bit: 0.15 for bit in failed},
            old_png_shifts={bit: -0.1 for bit in failed},
        )
        self.assertEqual(diagnosis['conclusion'], 'overshoot')
        self.assertEqual(diagnosis['supporting_bits'], 7)


if __name__ == '__main__':
    unittest.main()
