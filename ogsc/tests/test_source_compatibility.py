import unittest
from pathlib import Path

import torch


class SourceCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.source = cls.root / 'BEST_TEACHER_32bits_refine/server21/files/var/lib/structure32_student12500/runs/structure32'
        cls.paths = {'weight32_repo': str(cls.root), 'oracle_repo': str(cls.root.parent.parent / 'Hmark_inmodel'),
                     'source_run': str(cls.source), 'carrier_dir': str(cls.source / 'carrier')}

    def test_original_carrier_identity_and_normalization(self):
        from ogsc.assets import verify_carrier

        carrier, hashes = verify_carrier(self.paths)
        key = carrier.metadata
        self.assertEqual(key['num_bits'], 32)
        self.assertEqual(carrier.signs.shape, (32,))
        self.assertTrue(torch.isin(carrier.signs, torch.tensor([-1., 1.])).all())
        self.assertIsNone(carrier.carrier_version)
        self.assertIsNone(carrier.medians)
        self.assertEqual(set(hashes), {'complete.json', 'key.json', 'pca.pt', 'stability.json', 'provenance.json'})
        self.assertEqual(carrier.selected_pc_indices, key['selected_pc_indices'])
        pca = torch.load(self.source / 'carrier/pca.pt', map_location='cpu', weights_only=True)
        torch.testing.assert_close(carrier.mean, pca['mean'])
        torch.testing.assert_close(carrier.basis, pca['basis'][:, key['selected_pc_indices']])
        torch.testing.assert_close(carrier.sigmas, torch.tensor(key['sigmas']))
        torch.testing.assert_close(carrier.signs, torch.tensor(key['signs'], dtype=torch.float32))

    def test_rgb8_ste_matches_oracle_forward_and_identity_gradient(self):
        import sys
        oracle = self.root.parent.parent / 'Hmark_inmodel/Malicious_quantization/bitmark_lora'
        harness = self.root / 'BEST_TEACHER_32bits_refine/harness'
        sys.path[:0] = [str(harness), str(oracle)]
        from semantic_oracle_math import fake_quantize_rgb8_ste

        value = torch.tensor([-1.2, -0.503, 0.003, 0.782, 1.2], requires_grad=True)
        output = fake_quantize_rgb8_ste(value)
        clipped = value.detach().clamp(-1, 1)
        expected = ((clipped / 2 + .5) * 255).round() / 127.5 - 1
        torch.testing.assert_close(output.detach(), expected)
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([0., 1., 1., 1., 0.]))

    def test_workflow_needs_no_old_threshold_files_and_uses_original_student_bundle(self):
        from ogsc.paths import validate_run_inputs

        inputs = validate_run_inputs(self.paths)
        self.assertTrue(inputs['assets']['teacher_source']['revision'])
        self.assertFalse((self.source / 'result/threshold_teacher.json').exists())
        self.assertFalse((self.source / 'scores/teacher_clean_calibration.json').exists())
        self.assertTrue((self.root / 'structure_student12500/campaign.py').is_file())
        self.assertFalse((self.root / 'structure_student12500/data/threshold_quality_20261022/structure_margin_20261026/structure_margin_native/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
