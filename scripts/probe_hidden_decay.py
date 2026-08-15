"""Does a decay on the hidden recurrence bound the hidden_phase gradient?

The step-1000 decomposition traced 93% of the 121M run's squared gradient
norm to ``hidden_phase``, amplified by the horizon-index weight the cumsum
introduces: the angle at horizon h is ``h * theta``, so ``d/dtheta`` carries
weight ``h`` and the coherent sum over h costs ``sum_h h / H = (H+1)/2``.

With a decay ``lambda`` on the hidden transport the same sum becomes
``sum_h h*lambda^h / H``, which is bounded in H instead of linear in it.
This probe replaces the undamped rollout with

    c_h = (lambda R)^h z_0
    z_h = lambda R z_(h-1) + delta_h

on the trained 121M checkpoint and measures the resulting hidden_phase
gradient directly, so the predicted bound can be checked against a real
backward pass rather than assumed.
"""
from __future__ import annotations

import argparse
import math

import torch

from rotlm.models.complex_self_prediction import rotate_pairwise
from train_time_varying_scan import (
    configure_scan_backend,
    make_model,
    memmap,
    sampled_windows,
    scan_loss,
)

CHECKPOINT = "outputs/time_varying_scan/checkpoints/step01000.pt"


def decayed_rollout(transition, root, horizons, decay, *, collect=False):
    """``c_h = (decay*R)^h z_0`` then ``z_h = decay*R z_(h-1) + delta_h``."""
    angles = transition.hidden_phase.expand(*root.shape[:-1], horizons, -1)
    cumulative = angles.cumsum(dim=-2)
    expanded = root.unsqueeze(-2).expand(*root.shape[:-1], horizons, transition.width)

    steps = torch.arange(
        1, horizons + 1, device=root.device, dtype=root.dtype
    ).reshape(*(1,) * (root.ndim - 1), horizons, 1)
    schedule_states = rotate_pairwise(expanded, cumulative) * decay**steps

    forcing = transition.forcing_velocity_tape(
        schedule_states, collect_scan_diagnostics=collect
    )
    deltas = forcing.velocity

    local_angles = transition.hidden_phase.expand(*root.shape[:-1], -1)
    state = root
    states = []
    for index in range(horizons):
        state = decay * rotate_pairwise(state, local_angles) + deltas[..., index, :]
        states.append(state)
    return torch.stack(states, dim=-2), forcing


def patched_loss(model, window, args, decay):
    """Run the training objective with the decayed hidden rollout."""
    transition = model.complex_self_prediction
    original = transition.rollout_time_varying_scan

    def replacement(root, horizons, *, collect_scan_diagnostics=True, **kwargs):
        result = original(root, horizons, collect_scan_diagnostics=collect_scan_diagnostics, **kwargs)
        full_states, _ = decayed_rollout(transition, root, horizons, decay)
        return result.__class__(
            **{
                **{
                    field: getattr(result, field)
                    for field in result.__dataclass_fields__
                },
                "full_states": full_states,
                "read_states": transition._fixed_rms_normalize_hidden(full_states),
            }
        )

    transition.rollout_time_varying_scan = replacement
    try:
        return scan_loss(model, window, args)
    finally:
        transition.rollout_time_varying_scan = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--width", type=int, default=4352)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=31)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = make_model(args).cuda().train()
    configure_scan_backend(model, "eager")
    checkpoint = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    training = memmap("train")
    window = sampled_windows(
        training,
        args.microbatch,
        torch.Generator().manual_seed(args.seed),
        args.context + args.horizons,
    )

    horizons = args.horizons
    print(f"width={args.width} H={horizons} (121M step-1000 checkpoint)\n")
    print(
        f"{'decay':>7} | {'sum_h h*d^h/H':>13} | {'hidden_phase':>13} "
        f"{'rms':>9} | {'total gnorm':>11} | {'loss':>7}"
    )
    print("-" * 78)

    for decay in (1.0, 0.99, 0.95, 0.9, 0.8, 0.5):
        predicted = sum(
            h * decay**h for h in range(1, horizons + 1)
        ) / horizons

        model.zero_grad(set_to_none=True)
        loss = patched_loss(model, window, args, decay)
        loss_value = float(loss.detach())
        loss.backward()

        phase_grad = model.complex_self_prediction.hidden_phase.grad
        phase_norm = float(phase_grad.float().norm())
        total = math.sqrt(
            sum(
                float(p.grad.float().square().sum())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        print(
            f"{decay:7.2f} | {predicted:13.3f} | {phase_norm:13.4f} "
            f"{phase_norm / math.sqrt(phase_grad.numel()):9.2e} | "
            f"{total:11.4f} | {loss_value:7.4f}"
        )


if __name__ == "__main__":
    main()
