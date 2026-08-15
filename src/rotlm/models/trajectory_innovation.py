"""Noise-conditioned causal innovation tapes for latent trajectories."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def diagonal_affine_scan(
    multipliers: torch.Tensor,
    increments: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``s_t = a_t * s_(t-1) + b_t`` by associative scan.

    The horizon axis is the penultimate dimension.  The initial state is zero.
    Affine pairs compose as ``(a2, b2) o (a1, b1) =
    (a2*a1, b2 + a2*b1)``; Hillis--Steele doubling therefore gives every
    causal prefix in logarithmic dependency depth.
    """
    if multipliers.shape != increments.shape:
        raise ValueError("multipliers and increments must share shape")
    if multipliers.ndim < 2:
        raise ValueError("scan inputs need horizon and state dimensions")
    horizons = multipliers.shape[-2]
    if horizons < 1:
        raise ValueError("scan horizon must be positive")

    prefix_a = multipliers
    prefix_b = increments
    offset = 1
    while offset < horizons:
        old_a = prefix_a
        old_b = prefix_b
        composed_a = (
            old_a[..., offset:, :] * old_a[..., :-offset, :]
        )
        composed_b = (
            old_b[..., offset:, :]
            + old_a[..., offset:, :] * old_b[..., :-offset, :]
        )
        prefix_a = torch.cat(
            (old_a[..., :offset, :], composed_a),
            dim=-2,
        )
        prefix_b = torch.cat(
            (old_b[..., :offset, :], composed_b),
            dim=-2,
        )
        offset *= 2
    return prefix_b


def sequential_diagonal_affine(
    multipliers: torch.Tensor,
    increments: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation for :func:`diagonal_affine_scan`."""
    if multipliers.shape != increments.shape:
        raise ValueError("multipliers and increments must share shape")
    state = torch.zeros_like(increments[..., 0, :])
    states = []
    for horizon in range(increments.shape[-2]):
        state = (
            multipliers[..., horizon, :] * state
            + increments[..., horizon, :]
        )
        states.append(state)
    return torch.stack(states, dim=-2)


class SelectiveInnovationTrajectory(nn.Module):
    """Map sampled noise tapes to centered deterministic innovations.

    Every sampled candidate receives a full base-noise tape.  Context/noise
    features determine a diagonal affine recurrence whose causal states are
    computed by associative scan.  A shared output map turns those states into
    local additive innovations for ``u_(t+1) = K u_t + r_(t+1)``.
    """

    def __init__(
        self,
        width: int,
        trajectories: int,
        *,
        noise_width: int = 32,
        state_width: int = 64,
        init_scale: float = 0.05,
        min_scale: float = 1e-3,
        max_scale: float = 0.25,
        initial_memory: float = 0.75,
    ) -> None:
        super().__init__()
        if min(width, trajectories, noise_width, state_width) < 1:
            raise ValueError("all innovation sizes must be positive")
        if trajectories < 2:
            raise ValueError("at least two trajectories are required")
        if not 0 < min_scale < init_scale < max_scale:
            raise ValueError(
                "require 0 < min_scale < init_scale < max_scale"
            )
        if not 0 < initial_memory < 1:
            raise ValueError("initial_memory must be in (0,1)")

        self.width = int(width)
        self.trajectories = int(trajectories)
        self.noise_width = int(noise_width)
        self.state_width = int(state_width)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        self.context = nn.Linear(width, state_width, bias=False)
        self.noise = nn.Linear(noise_width, state_width, bias=False)
        self.decay = nn.Linear(state_width, state_width)
        self.drive = nn.Linear(state_width, state_width, bias=False)
        self.output = nn.Linear(state_width, width, bias=False)
        self.scale = nn.Linear(width, 1)
        self.prior = nn.Linear(state_width, 1)

        nn.init.normal_(self.context.weight, std=0.02)
        nn.init.normal_(self.noise.weight, std=0.02)
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(
            self.decay.bias,
            math.log(initial_memory / (1.0 - initial_memory)),
        )
        nn.init.normal_(self.drive.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, math.log(init_scale))
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    def forward(
        self,
        roots: torch.Tensor,
        horizons: int,
        *,
        noise_generator: torch.Generator | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return innovations, candidate priors, scales, noise, and scan data.

        ``roots`` has shape ``[batch, anchors, width]``. Innovations have shape
        ``[batch, trajectories, anchors, horizons, width]``.  Their
        prior-weighted centroid is zero independently at every horizon.
        """
        if roots.ndim != 3 or roots.shape[-1] != self.width:
            raise ValueError(
                "roots must have shape [batch,anchors,width], got "
                f"{tuple(roots.shape)}"
            )
        if horizons < 1:
            raise ValueError("horizons must be positive")

        detached = roots.detach().float()
        root_rms = detached.square().mean(dim=-1, keepdim=True).sqrt()
        normalized = detached / root_rms.clamp_min(1e-8)
        standard_noise = torch.randn(
            roots.shape[0],
            self.trajectories,
            roots.shape[1],
            horizons,
            self.noise_width,
            device=roots.device,
            dtype=torch.float32,
            generator=noise_generator,
        )

        context = self.context(normalized)[:, None, :, None]
        noise_features = self.noise(standard_noise)
        features = F.silu(context + noise_features)
        multipliers = torch.sigmoid(self.decay(features))
        candidates = torch.tanh(self.drive(features))
        increments = (1.0 - multipliers) * candidates
        scan_states = diagonal_affine_scan(multipliers, increments)

        raw = self.output(scan_states)
        raw = raw - raw.mean(dim=1, keepdim=True)
        direction = raw / raw.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-8)

        log_scale = self.scale(normalized).clamp(
            min=math.log(self.min_scale),
            max=math.log(self.max_scale),
        )
        residuals = (
            log_scale.exp()[:, None, :, None]
            * root_rms[:, None, :, None]
            * direction
        )

        prior_logits = self.prior(scan_states[..., -1, :]).squeeze(-1)
        prior_probs = torch.softmax(prior_logits, dim=1)
        residuals = residuals - (
            prior_probs.detach()[..., None, None] * residuals
        ).sum(dim=1, keepdim=True)

        observed_ratio = (
            residuals.square().mean(dim=-1).sqrt()
            / root_rms.squeeze(-1)[:, None, :, None].clamp_min(1e-8)
        )
        observed_log_scale = observed_ratio.clamp_min(1e-8).log()
        return (
            residuals.to(roots.dtype),
            prior_probs.to(roots.dtype),
            observed_log_scale,
            standard_noise,
            multipliers,
            increments,
        )


__all__ = [
    "SelectiveInnovationTrajectory",
    "diagonal_affine_scan",
    "sequential_diagonal_affine",
]
