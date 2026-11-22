import unittest
from unittest.mock import patch

import torch


class SoftMinControllerTests(unittest.TestCase):
    def test_long_stage_progress_includes_completed_total_and_eta(self):
        from ogsc.quality import progress_message

        message = progress_message('clean validation', 250, 1000, elapsed_seconds=108,
                                   work_completed=40, work_total=790)
        self.assertIn('250/1000', message)
        self.assertIn('25%', message)
        self.assertIn('40/790 this run', message)
        self.assertIn('ETA 33.8 min', message)

    def test_progress_eta_warms_up_before_extrapolating(self):
        from ogsc.quality import progress_message

        message = progress_message('clean validation', 201, 1000, elapsed_seconds=2,
                                   work_completed=1, work_total=799)
        self.assertIn('ETA warming up', message)

    def test_quality_gamma_selection_checks_all_candidates_without_monotonicity(self):
        from ogsc.quality import select_gamma

        trials = [{'gamma': .1, 'passes': True}, {'gamma': .2, 'passes': False},
                  {'gamma': .3, 'passes': True}, {'gamma': 0., 'passes': True}]
        self.assertEqual(select_gamma(trials), .3)
        self.assertIsNone(select_gamma([{'gamma': .1, 'passes': False}]))

    def test_objective_is_stable_and_prioritizes_weak_coordinates(self):
        from ogsc.control import soft_min_loss

        signed = torch.tensor([[1000.0] * 31 + [-2.0]], requires_grad=True)
        loss = soft_min_loss(signed)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertLess(loss.item(), 0.0)

        weights, = torch.autograd.grad(loss.sum(), signed)
        self.assertLess(weights[0, -1].item(), weights[0, 0].item())

    def test_one_step_has_one_vjp_and_reduces_tiny_softmin(self):
        from ogsc.control import one_step, soft_min_loss

        torch.manual_seed(9)
        model = torch.nn.Linear(8, 32, bias=False).eval().requires_grad_(False)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        latent = torch.randn(1, 8)
        with patch('torch.autograd.grad', wraps=torch.autograd.grad) as grad_call:
            marked, diagnostics = one_step(latent, model, gamma=1e-3)
        self.assertEqual(grad_call.call_count, 1)
        self.assertEqual(marked.shape, latent.shape)
        self.assertTrue(torch.isfinite(marked).all())
        loss_after = soft_min_loss(model(marked)).item()
        self.assertLess(loss_after, diagnostics['loss_before'])
        for parameter, value in zip(model.parameters(), before):
            torch.testing.assert_close(parameter, value)
            self.assertIsNone(parameter.grad)

    def test_gamma_zero_bypasses_semantic_forward_and_preserves_latent(self):
        from ogsc.control import one_step

        latent = torch.randn(1, 4, 64, 64, dtype=torch.float16)
        def forbidden(_value):
            raise AssertionError('gamma zero must skip semantic extraction')

        result, diagnostics = one_step(latent, forbidden, gamma=0.0)
        self.assertIs(result, latent)
        self.assertEqual(diagnostics['loss_before'], None)


if __name__ == '__main__':
    unittest.main()
