"""Online clean-anchored feedback control for one final latent at a time."""

import math

import torch


def _validate_coordinates(value, name):
    if not isinstance(value, torch.Tensor) or value.shape != (1, 32):
        raise ValueError(f'{name} must be a tensor with shape [1, 32]')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains nonfinite coordinates')


def clean_anchor_target(clean_coordinates):
    """Freeze the nonnegative clean signed margins as per-coordinate targets."""
    _validate_coordinates(clean_coordinates, 'clean_coordinates')
    return clean_coordinates.detach().clamp_min(0)


def repair_loss(coordinates, target):
    """Mean squared per-bit shortfall; fulfilled bits cannot cancel deficits."""
    _validate_coordinates(coordinates, 'coordinates')
    _validate_coordinates(target, 'target')
    if coordinates.device != target.device:
        raise ValueError('coordinates and target must be on the same device')
    return (target - coordinates).clamp_min(0).square().mean()


def _repair_key(coordinates, target):
    values = coordinates.detach()
    return (int((values <= 0).sum().item()),
            float((target - values).clamp_min(0).square().mean().item()),
            -float(values.min().item()))


def _rms(value):
    return value.square().mean().sqrt()


def _improves(current, trial, target, mode):
    if mode == 'repair':
        return _repair_key(trial, target) < _repair_key(current, target)
    return bool((trial > 0).all()) and float(trial.min().item()) > float(current.min().item())


@torch.enable_grad()
def ofc_control(clean_latent, signed_view, gamma_budget, max_updates=3, epsilon=1e-12):
    """Control one clean final latent using at most three online VJPs.

    ``signed_view`` must return signed semantic coordinates with shape [1, 32].
    Candidate scoring is forward-only and every candidate is quantized to the
    clean latent's storage dtype before evaluation.
    """
    if not isinstance(clean_latent, torch.Tensor) or clean_latent.ndim < 2 or clean_latent.shape[0] != 1:
        raise ValueError('clean_latent must be a batched tensor with batch size one')
    if clean_latent.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('clean_latent must use float16, bfloat16, or float32')
    if not torch.isfinite(clean_latent).all():
        raise ValueError('clean_latent contains nonfinite values')
    gamma_budget = float(gamma_budget)
    if not math.isfinite(gamma_budget) or gamma_budget < 0:
        raise ValueError('gamma_budget must be finite and nonnegative')
    if not isinstance(max_updates, int) or not 1 <= max_updates <= 3:
        raise ValueError('max_updates must be between 1 and 3')

    original_dtype = clean_latent.dtype
    z0 = clean_latent.detach().float()
    with torch.no_grad():
        r0 = signed_view(z0).detach()
    _validate_coordinates(r0, 'signed_view output')
    if r0.device != z0.device:
        raise ValueError('signed_view output must be on the latent device')
    target = clean_anchor_target(r0)
    current = z0
    best = current
    diagnostics = {
        'vjp_count': 0, 'forward_count': 1, 'updates_attempted': 0,
        'updates_accepted': 0, 'backtracking_halvings': [], 'accepted_modes': [],
        'initial_signed_margins': r0.cpu().reshape(32).tolist(),
        'gamma_budget': gamma_budget, 'final_mode': 'unchanged',
    }
    if gamma_budget == 0:
        diagnostics['final_signed_margins'] = diagnostics['initial_signed_margins']
        return clean_latent.detach().clone(), diagnostics

    for _ in range(max_updates):
        z = current.detach().float().requires_grad_(True)
        coordinates = signed_view(z)
        diagnostics['forward_count'] += 1
        _validate_coordinates(coordinates, 'signed_view output')
        detached = coordinates.detach()
        mode = 'margin' if bool((detached > 0).all()) else 'repair'
        loss = (-coordinates.min(dim=-1).values.mean() if mode == 'margin'
                else repair_loss(coordinates, target))
        diagnostics['final_mode'] = mode
        diagnostics['updates_attempted'] += 1
        gradient, = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)
        diagnostics['vjp_count'] += 1
        if not torch.isfinite(gradient).all():
            raise RuntimeError('Nonfinite OFC gradient')
        grad_rms = _rms(gradient)
        if float(grad_rms.item()) == 0:
            break
        direction = (-gradient / (grad_rms + epsilon)).detach()
        accepted = False
        initial_step = gamma_budget / max_updates
        for halve in range(3):
            step = initial_step / (2 ** halve)
            trial = z.detach() + step * direction
            displacement = trial - z0
            displacement_rms = _rms(displacement)
            if float(displacement_rms.item()) > gamma_budget:
                trial = z0 + displacement * (gamma_budget / float(displacement_rms.item()))
            trial = trial.to(original_dtype).float().detach()
            with torch.no_grad():
                trial_coordinates = signed_view(trial).detach()
            diagnostics['forward_count'] += 1
            _validate_coordinates(trial_coordinates, 'signed_view output')
            if _improves(detached, trial_coordinates, target, mode):
                current = trial
                best = current
                diagnostics['updates_accepted'] += 1
                diagnostics['backtracking_halvings'].append(halve)
                diagnostics['accepted_modes'].append(mode)
                accepted = True
                break
        if not accepted:
            break

    with torch.no_grad():
        final = signed_view(best).detach()
    diagnostics['forward_count'] += 1
    _validate_coordinates(final, 'signed_view output')
    diagnostics['final_signed_margins'] = final.cpu().reshape(32).tolist()
    return best.to(original_dtype).detach(), diagnostics
