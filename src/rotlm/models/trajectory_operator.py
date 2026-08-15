"""Context-conditioned operators for persistent latent trajectories."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class BranchPlaneRotationTrajectory(nn.Module):
    """Compile each prefix/branch into one reusable orthogonal rotation.

    The module emits a two-dimensional rotation plane and angle once from
    ``stopgrad(h_A)`` and a fixed branch code.  Applying that same rotation
    after every shared ``K`` step defines one branch operator
    ``T_i = Q_i K`` without a per-horizon network or branch reselection.
    """

    def __init__(
        self,
        width: int,
        trajectories: int,
        *,
        code_width: int = 32,
        bottleneck: int = 64,
        init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if min(width, trajectories, code_width, bottleneck) < 1:
            raise ValueError("all trajectory operator sizes must be positive")
        if trajectories < 2:
            raise ValueError("at least two trajectories are required")
        maximum_isotropic_scale = 2.0 * math.sqrt(2.0 / width)
        if not 0.0 < init_scale < maximum_isotropic_scale:
            raise ValueError(
                "init_scale must be below the one-plane isotropic maximum "
                f"{maximum_isotropic_scale:.6f}"
            )
        self.width = int(width)
        self.trajectories = int(trajectories)
        self.code_width = int(code_width)
        self.bottleneck = int(bottleneck)
        self.init_scale = float(init_scale)

        codes = torch.randn(trajectories, code_width)
        codes -= codes.mean(dim=0, keepdim=True)
        codes = F.normalize(codes, dim=-1)
        self.branch_codes = nn.Parameter(codes)
        self.context = nn.Linear(width, bottleneck, bias=False)
        self.code = nn.Linear(code_width, bottleneck, bias=False)
        self.plane = nn.Linear(bottleneck, 2 * width, bias=False)
        self.angle = nn.Linear(width, 1)
        self.prior = nn.Linear(width, trajectories)

        nn.init.normal_(self.context.weight, std=0.02)
        nn.init.normal_(self.code.weight, std=0.02)
        nn.init.normal_(self.plane.weight, std=0.02)
        nn.init.zeros_(self.angle.weight)
        initial_sine = (
            init_scale * math.sqrt(width / 2.0) / 2.0
        )
        initial_angle = 2.0 * math.asin(initial_sine)
        initial_fraction = initial_angle / math.pi
        nn.init.constant_(
            self.angle.bias,
            math.log(initial_fraction / (1.0 - initial_fraction)),
        )
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    def forward(
        self,
        roots: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return rotation planes, angles, and prefix-only branch priors.

        ``roots`` has shape ``[batch, anchors, width]``. Plane vectors have
        shape ``[batch, trajectories, anchors, width]`` and are orthonormal
        within each branch/anchor pair. Angles have a singleton branch-shared
        final dimension and are broadcast over trajectories.
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
        first_raw, second_raw = self.plane(features).chunk(2, dim=-1)
        first = F.normalize(first_raw, dim=-1)
        second_raw = second_raw - (
            second_raw * first
        ).sum(dim=-1, keepdim=True) * first
        second = F.normalize(second_raw, dim=-1)

        angles = (
            math.pi * torch.sigmoid(self.angle(normalized))
        )[:, None]
        prior_logits = self.prior(normalized).movedim(-1, 1)
        prior_probs = torch.softmax(prior_logits, dim=1)
        return (
            first.to(roots.dtype),
            second.to(roots.dtype),
            angles.to(roots.dtype),
            prior_probs.to(roots.dtype),
        )

    @staticmethod
    def rotate(
        states: torch.Tensor,
        first: torch.Tensor,
        second: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the represented plane rotation without materializing Q."""
        if states.shape != first.shape or states.shape != second.shape:
            raise ValueError("states and plane vectors must share shape")
        if angles.shape != states.shape[:-1] + (1,):
            try:
                angles = angles.expand(states.shape[:-1] + (1,))
            except RuntimeError as error:
                raise ValueError(
                    "angles must broadcast over state rows"
                ) from error

        first_coordinate = (states * first).sum(dim=-1, keepdim=True)
        second_coordinate = (states * second).sum(
            dim=-1, keepdim=True
        )
        cosine = angles.cos()
        sine = angles.sin()
        first_delta = (
            (cosine - 1.0) * first_coordinate
            - sine * second_coordinate
        )
        second_delta = (
            sine * first_coordinate
            + (cosine - 1.0) * second_coordinate
        )
        return states + first_delta * first + second_delta * second


__all__ = ["BranchPlaneRotationTrajectory"]
