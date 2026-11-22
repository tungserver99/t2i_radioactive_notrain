"""Soft minimum signed-margin controller with one final-latent VJP."""

import math

import torch

BITS = 32
GRAD_EPSILON = 1e-12


def soft_min_loss(signed_margins):
    """Stable per-image log-mean-exp objective over all 32 signed margins."""
    if signed_margins.ndim != 2 or signed_margins.shape[1] != BITS:
        raise ValueError(f'Expected [batch, {BITS}] signed margins')
    if not torch.isfinite(signed_margins).all():
        raise ValueError('Nonfinite signed margins')
    return torch.logsumexp(-signed_margins, dim=-1) - math.log(BITS)


@torch.enable_grad()
def one_step(latent, signed_view, gamma, epsilon=GRAD_EPSILON):
    """Apply normalized negative gradient of the soft-min loss to final latent."""
    if not math.isfinite(float(gamma)) or gamma < 0:
        raise ValueError('Gamma must be finite and nonnegative')
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError('Gradient epsilon must be finite and positive')
    if gamma == 0:
        return latent, {'loss_before': None, 'gradient_rms': None, 'direction_l2': None}

    z = latent.detach().float().clone().requires_grad_(True)
    signed = signed_view(z)
    if signed.shape != (z.shape[0], BITS) or not torch.isfinite(signed).all():
        raise ValueError(f'Expected {BITS} finite signed coordinates per image')
    losses = soft_min_loss(signed)
    gradient, = torch.autograd.grad(losses.sum(), z, retain_graph=False, create_graph=False)
    if not torch.isfinite(gradient).all():
        raise RuntimeError('OGSC soft-min latent gradient is nonfinite')
    rms = gradient.square().mean(dim=tuple(range(1, gradient.ndim)), keepdim=True).sqrt()
    if not torch.isfinite(rms).all() or (rms <= 0).any():
        raise RuntimeError('OGSC soft-min latent gradient is zero or nonfinite')
    direction = -gradient / (rms + epsilon)
    marked = (z.detach() + float(gamma) * direction.detach()).to(dtype=latent.dtype)
    if not torch.isfinite(marked).all():
        raise RuntimeError('OGSC marked latent is nonfinite')
    info = {
        'loss_before': float(losses.detach().mean().cpu()),
        'gradient_rms': rms.detach().flatten().cpu().tolist(),
        'direction_l2': direction.detach().flatten(1).norm(dim=1).cpu().tolist(),
        'signed_margins': signed.detach().cpu().tolist(),
    }
    return marked, info
