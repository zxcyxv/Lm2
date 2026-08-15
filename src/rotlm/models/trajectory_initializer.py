"""Context-conditioned initial conditions for persistent latent trajectories."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class InitialConditionTrajectory(nn.Module):
    """Compile fixed branch codes into one centered residual per prefix state.

    The prefix state is detached before it enters the initializer and prior
    heads.  Token and orbit losses still reach the encoder through the clean
    ``K h_A`` path, while this module learns only how to place and weight
    branch-specific initial conditions around that clean center.
    """

    def __init__(
        self,
        width: int,
        trajectories: int,
        *,
        code_width: int = 32,
        bottleneck: int = 64,
        init_scale: float = 0.05,
        min_scale: float = 1e-3,
        max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if min(width, trajectories, code_width, bottleneck) < 1:
            raise ValueError("all trajectory initializer sizes must be positive")
        if trajectories < 2:
            raise ValueError("at least two trajectories are required")
        if not 0 < min_scale < init_scale < max_scale:
            raise ValueError(
                "require 0 < min_scale < init_scale < max_scale"
            )
        self.width = int(width)
        self.trajectories = int(trajectories)
        self.code_width = int(code_width)
        self.bottleneck = int(bottleneck)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        codes = torch.randn(trajectories, code_width)
        codes -= codes.mean(dim=0, keepdim=True)
        codes = F.normalize(codes, dim=-1)
        self.branch_codes = nn.Parameter(codes)
        self.context = nn.Linear(width, bottleneck, bias=False)
        self.code = nn.Linear(code_width, bottleneck, bias=False)
        self.output = nn.Linear(bottleneck, width, bias=False)
        self.scale = nn.Linear(width, 1)
        self.prior = nn.Linear(width, trajectories)

        nn.init.normal_(self.context.weight, std=0.02)
        nn.init.normal_(self.code.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, math.log(init_scale))
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    def forward(
        self,
        roots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return centered residuals, prefix priors, and observed log scales.

        ``roots`` has shape ``[batch, anchors, width]``. Residuals have shape
        ``[batch, trajectories, anchors, width]`` and their prior-weighted
        centroid is exactly zero up to floating-point error.
        """
        if roots.ndim != 3 or roots.shape[-1] != self.width:
            raise ValueError(
                "roots must have shape [batch,anchors,width], got "
                f"{tuple(roots.shape)}"
            )
        detached = roots.detach().float()
        root_rms = detached.square().mean(dim=-1, keepdim=True).sqrt()
        normalized = detached / root_rms.clamp_min(1e-8)

        context = self.context(normalized)[:, None]
        code = self.code(self.branch_codes.float())[None, :, None]
        features = F.silu(context + code)
        raw = self.output(features)
        raw = raw - raw.mean(dim=1, keepdim=True)
        direction = raw / raw.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-8)

        log_scale = self.scale(normalized).clamp(
            min=math.log(self.min_scale),
            max=math.log(self.max_scale),
        )
        residuals = (
            log_scale.exp()[:, None]
            * root_rms[:, None]
            * direction
        )
        prior_logits = self.prior(normalized).movedim(-1, 1)
        prior_probs = torch.softmax(prior_logits, dim=1)
        residuals = residuals - (
            prior_probs.detach()[..., None] * residuals
        ).sum(dim=1, keepdim=True)

        observed_ratio = (
            residuals.square().mean(dim=-1).sqrt()
            / root_rms[:, None, :, 0].clamp_min(1e-8)
        )
        observed_log_scale = observed_ratio.clamp_min(1e-8).log()
        return (
            residuals.to(roots.dtype),
            prior_probs.to(roots.dtype),
            observed_log_scale,
        )


__all__ = ["InitialConditionTrajectory"]
