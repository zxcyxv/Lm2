"""Prefix-conditioned orthogonal affine tapes with exact associative scan."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def rotate_pairwise(
    values: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Apply independent two-dimensional rotations on the trailing axis."""
    if values.shape[-1] % 2:
        raise ValueError("pairwise rotation requires an even width")
    if angles.shape != (*values.shape[:-1], values.shape[-1] // 2):
        raise ValueError(
            "angles must match values except for a halved trailing width"
        )
    pairs = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    first, second = pairs.unbind(dim=-1)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    return torch.stack(
        (
            cosine * first - sine * second,
            sine * first + cosine * second,
        ),
        dim=-1,
    ).flatten(-2)


def pairwise_affine_scan(
    angles: torch.Tensor,
    increments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose every causal prefix of pairwise orthogonal affine maps.

    The horizon axis is penultimate in both tensors. A local transformation is
    ``T_j(x) = R(angle_j) x + increment_j``. Chronological composition is

    ``T_2 o T_1 = (angle_2 + angle_1,
                   R(angle_2) increment_1 + increment_2)``.
    """
    if angles.ndim < 2 or increments.ndim != angles.ndim:
        raise ValueError("scan tensors need matching rank and a horizon axis")
    if increments.shape[:-1] != angles.shape[:-1]:
        raise ValueError("scan tensors must share non-feature dimensions")
    if increments.shape[-1] != 2 * angles.shape[-1]:
        raise ValueError("increment width must be twice the angle width")
    horizons = angles.shape[-2]
    if horizons < 1:
        raise ValueError("scan horizon must be positive")

    prefix_angles = angles
    prefix_increments = increments
    offset = 1
    while offset < horizons:
        old_angles = prefix_angles
        old_increments = prefix_increments
        later_angles = old_angles[..., offset:, :]
        composed_angles = (
            later_angles + old_angles[..., :-offset, :]
        )
        composed_increments = (
            rotate_pairwise(
                old_increments[..., :-offset, :],
                later_angles,
            )
            + old_increments[..., offset:, :]
        )
        prefix_angles = torch.cat(
            (old_angles[..., :offset, :], composed_angles),
            dim=-2,
        )
        prefix_increments = torch.cat(
            (
                old_increments[..., :offset, :],
                composed_increments,
            ),
            dim=-2,
        )
        offset *= 2
    return prefix_angles, prefix_increments


def apply_pairwise_affine_scan(
    root: torch.Tensor,
    angles: torch.Tensor,
    increments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all states and the cumulative affine maps from one root."""
    if root.shape != (*increments.shape[:-2], increments.shape[-1]):
        raise ValueError("root shape does not match affine tape")
    cumulative_angles, cumulative_increments = pairwise_affine_scan(
        angles,
        increments,
    )
    expanded_root = root.unsqueeze(-2).expand_as(increments)
    states = (
        rotate_pairwise(expanded_root, cumulative_angles)
        + cumulative_increments
    )
    return states, cumulative_angles, cumulative_increments


def sequential_pairwise_affine(
    root: torch.Tensor,
    angles: torch.Tensor,
    increments: torch.Tensor,
) -> torch.Tensor:
    """Literal reference for the scanned pairwise affine recurrence."""
    if root.shape != (*increments.shape[:-2], increments.shape[-1]):
        raise ValueError("root shape does not match affine tape")
    current = root
    states = []
    for horizon in range(angles.shape[-2]):
        current = (
            rotate_pairwise(
                current,
                angles[..., horizon, :],
            )
            + increments[..., horizon, :]
        )
        states.append(current)
    return torch.stack(states, dim=-2)


@dataclass
class PrefixOrthogonalScanOutput:
    states: torch.Tensor
    coordinate_states: torch.Tensor
    local_angles: torch.Tensor
    innovations: torch.Tensor
    log_scale: torch.Tensor
    cumulative_angles: torch.Tensor
    cumulative_innovations: torch.Tensor
    basis: torch.Tensor


class PrefixConditionedOrthogonalAffineScan(nn.Module):
    """Compile one prefix into a deterministic scan-compatible future tape."""

    def __init__(
        self,
        width: int,
        *,
        bottleneck: int = 128,
        position_width: int = 32,
        max_angle_delta: float = 0.25,
        init_scale: float = 0.05,
        min_scale: float = 1e-3,
        max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("width must be positive and even")
        if bottleneck < 1 or position_width < 2 or position_width % 2:
            raise ValueError(
                "bottleneck must be positive and position width positive/even"
            )
        if max_angle_delta <= 0:
            raise ValueError("maximum angle delta must be positive")
        if not 0 < min_scale < init_scale < max_scale:
            raise ValueError(
                "require 0 < min_scale < init_scale < max_scale"
            )

        self.width = int(width)
        self.pairs = width // 2
        self.bottleneck = int(bottleneck)
        self.position_width = int(position_width)
        self.max_angle_delta = float(max_angle_delta)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        self.basis_generator = nn.Parameter(torch.zeros(width, width))
        self.base_angle = nn.Parameter(torch.zeros(self.pairs))
        self.root_norm = nn.RMSNorm(width, eps=1e-6)
        self.context = nn.Linear(width, bottleneck, bias=False)
        self.position = nn.Linear(position_width, bottleneck, bias=False)
        self.angle = nn.Linear(bottleneck, self.pairs)
        self.innovation = nn.Linear(bottleneck, width, bias=False)
        self.scale = nn.Linear(bottleneck, 1)

        for layer in (
            self.context,
            self.position,
            self.angle,
            self.innovation,
        ):
            nn.init.normal_(layer.weight, std=0.02)
        nn.init.zeros_(self.angle.bias)
        nn.init.zeros_(self.scale.weight)
        scale_fraction = (
            (init_scale - min_scale) / (max_scale - min_scale)
        )
        nn.init.constant_(
            self.scale.bias,
            math.log(scale_fraction / (1.0 - scale_fraction)),
        )

        bands = position_width // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(bands, dtype=torch.float32)
            / max(1, bands - 1)
        )
        self.register_buffer(
            "position_frequencies",
            frequencies,
            persistent=False,
        )

    @property
    def basis(self) -> torch.Tensor:
        skew = self.basis_generator - self.basis_generator.T
        return torch.matrix_exp(skew)

    def position_features(
        self,
        horizons: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if horizons < 1:
            raise ValueError("horizons must be positive")
        positions = torch.arange(
            1,
            horizons + 1,
            device=device,
            dtype=torch.float32,
        )
        phases = positions[:, None] * self.position_frequencies[None]
        return torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)

    def compile_tape(
        self,
        roots: torch.Tensor,
        horizons: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return coordinate root, local angles/innovations, scales, and Q."""
        if roots.ndim != 3 or roots.shape[-1] != self.width:
            raise ValueError(
                "roots must have shape [batch,anchors,width], got "
                f"{tuple(roots.shape)}"
            )
        basis = self.basis
        coordinate_root = F.linear(roots, basis.T)
        position_features = self.position_features(
            horizons,
            device=roots.device,
        )
        features = F.silu(
            self.context(self.root_norm(roots)).unsqueeze(-2)
            + self.position(position_features)[None, None]
        )
        local_angles = self.base_angle.view(1, 1, 1, -1) + (
            self.max_angle_delta * torch.tanh(self.angle(features))
        )

        raw_innovations = self.innovation(features)
        directions = raw_innovations / raw_innovations.float().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt().clamp_min(1e-8).to(raw_innovations.dtype)
        scale = self.min_scale + (
            self.max_scale - self.min_scale
        ) * torch.sigmoid(self.scale(features).float())
        root_rms = roots.float().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt()
        innovations = (
            scale * root_rms.unsqueeze(-2) * directions.float()
        ).to(roots.dtype)
        return (
            coordinate_root,
            local_angles,
            innovations,
            scale.clamp_min(1e-8).log().squeeze(-1),
            basis,
        )

    def forward(
        self,
        roots: torch.Tensor,
        horizons: int,
    ) -> PrefixOrthogonalScanOutput:
        (
            coordinate_root,
            local_angles,
            innovations,
            log_scale,
            basis,
        ) = self.compile_tape(roots, horizons)
        (
            coordinate_states,
            cumulative_angles,
            cumulative_innovations,
        ) = apply_pairwise_affine_scan(
            coordinate_root,
            local_angles,
            innovations,
        )
        states = F.linear(coordinate_states, basis)
        return PrefixOrthogonalScanOutput(
            states=states,
            coordinate_states=coordinate_states,
            local_angles=local_angles,
            innovations=innovations,
            log_scale=log_scale,
            cumulative_angles=cumulative_angles,
            cumulative_innovations=cumulative_innovations,
            basis=basis,
        )


__all__ = [
    "PrefixConditionedOrthogonalAffineScan",
    "PrefixOrthogonalScanOutput",
    "apply_pairwise_affine_scan",
    "pairwise_affine_scan",
    "rotate_pairwise",
    "sequential_pairwise_affine",
]
