import unittest
from unittest import mock

import torch

from ogsc.ofc_control import clean_anchor_target, ofc_control, repair_loss


class OFCControlTests(unittest.TestCase):
    def test_repair_loss_keeps_each_coordinate_and_clean_target_detached(self):
        clean = torch.tensor([[2.0, -1.0] + [0.5] * 30], requires_grad=True)
        target = clean_anchor_target(clean)
        self.assertFalse(target.requires_grad)
        self.assertEqual(target[0, 0].item(), 2.0)
        self.assertEqual(target[0, 1].item(), 0.0)
        current = torch.tensor([[1.0, -1.0] + [0.5] * 30], requires_grad=True)
        loss = repair_loss(current, target)
        self.assertAlmostEqual(loss.item(), (1.0 + 1.0) / 32.0, places=6)
        grad, = torch.autograd.grad(loss, current)
        self.assertLess(grad[0, 0].item(), 0.0)
        self.assertLess(grad[0, 1].item(), 0.0)
        self.assertEqual(grad[0, 2].item(), 0.0)

    def test_controller_uses_at_most_three_vjps_and_leaves_weights_frozen(self):
        weight = torch.nn.Parameter(torch.ones(32), requires_grad=False)
        clean = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        def signed_view(z):
            return (z.reshape(1, 1) - 0.1).expand(1, 32) * weight
        with mock.patch('torch.autograd.grad', wraps=torch.autograd.grad) as grad_call:
            _, diagnostics = ofc_control(clean, signed_view, gamma_budget=0.3)
        self.assertLessEqual(grad_call.call_count, 3)
        self.assertEqual(diagnostics['vjp_count'], grad_call.call_count)
        self.assertIsNone(weight.grad)
        self.assertLessEqual(diagnostics['updates_attempted'], 3)

    def test_backtracking_accepts_smaller_trial_and_preserves_fp16_dtype(self):
        clean = torch.zeros((1, 1, 1, 1), dtype=torch.float16)
        def signed_view(z):
            x = z.reshape(1, 1)
            return (x - x.square() - 0.1).expand(1, 32)
        marked, diagnostics = ofc_control(clean, signed_view, gamma_budget=3.0)
        self.assertEqual(marked.dtype, torch.float16)
        self.assertAlmostEqual(marked.float().item(), 0.5, places=3)
        self.assertEqual(diagnostics['backtracking_halvings'][0], 1)
        self.assertLessEqual(diagnostics['vjp_count'], 3)

    def test_margin_phase_never_accepts_trial_that_breaks_sign_feasibility(self):
        clean = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        def signed_view(z):
            x = z.reshape(1, 1)
            first = 0.1 + x
            second = 0.11 - x
            rest = torch.ones((1, 30), dtype=z.dtype, device=z.device)
            return torch.cat([first, second, rest], dim=1)
        marked, diagnostics = ofc_control(clean, signed_view, gamma_budget=0.6)
        self.assertTrue(torch.equal(marked, clean))
        self.assertEqual(diagnostics['updates_accepted'], 0)
        self.assertEqual(diagnostics['final_mode'], 'margin')

    def test_zero_gamma_returns_original_dtype_and_no_vjp(self):
        clean = torch.ones((1, 1, 1, 1), dtype=torch.float16)
        marked, diagnostics = ofc_control(clean, lambda z: z.reshape(1, 1).expand(1, 32), gamma_budget=0)
        self.assertTrue(torch.equal(marked, clean))
        self.assertEqual(marked.dtype, clean.dtype)
        self.assertEqual(diagnostics['vjp_count'], 0)


if __name__ == '__main__':
    unittest.main()
