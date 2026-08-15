"""Input-conditioned correction in a shared spectral basis."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .prefix_orthogonal_scan import rotate_pairwise


@dataclass
class SpectralCorrectorOutput:
    """One vectorized application of ``T(stopgrad(q)) stopgrad(q)``."""

    states: torch.Tensor
    coordinate_states: torch.Tensor
    source_coordinates: torch.Tensor
    log_scale: torch.Tensor
    scale: torch.Tensor
    phase_delta: torch.Tensor


class SpectralStateCorrector(nn.Module):
    """Map proposal states through an input-conditioned spectral operator.

    For a fixed proposal ``q``, the represented map is complex diagonal in
    the supplied orthogonal basis. The controller makes the complete
    ``T(q)q`` map nonlinear. ``detach_inputs`` controls whether a corrector
    loss is isolated from or attached to the proposal geometry.
    """

    def __init__(
        self,
        width: int,
        *,
        bottleneck: int = 128,
        max_log_scale: float = 2.0,
        max_phase_delta: float = math.pi,
        detach_inputs: bool,
    ) -> None:
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("width must be positive and even")
        if bottleneck < 1:
            raise ValueError("bottleneck must be positive")
        if max_log_scale <= 0 or max_phase_delta <= 0:
            raise ValueError("correction bounds must be positive")

        self.width = int(width)
        self.pairs = self.width // 2
        self.bottleneck = int(bottleneck)
        self.max_log_scale = float(max_log_scale)
        self.max_phase_delta = float(max_phase_delta)
        self.detach_inputs = bool(detach_inputs)

        self.input_norm = nn.RMSNorm(width, eps=1e-6)
        self.context = nn.Linear(width, bottleneck, bias=False)
        self.log_scale_head = nn.Linear(bottleneck, self.pairs)
        self.phase_head = nn.Linear(bottleneck, self.pairs)

        nn.init.normal_(self.context.weight, std=0.02)
        # Exact identity initialization is an internal paired control.
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.zeros_(self.log_scale_head.bias)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)

    def forward(
        self,
        proposals: torch.Tensor,
        basis: torch.Tensor,
    ) -> SpectralCorrectorOutput:
        if proposals.shape[-1] != self.width:
            raise ValueError(
                f"proposal width must be {self.width}, got "
                f"{proposals.shape[-1]}"
            )
        if basis.shape != (self.width, self.width):
            raise ValueError(
                "basis must have shape "
                f"[{self.width},{self.width}], got {tuple(basis.shape)}"
            )

        output_dtype = proposals.dtype
        source = proposals.float()
        active_basis = basis.float()
        if self.detach_inputs:
            source = source.detach()
            active_basis = active_basis.detach()
        features = F.silu(self.context(self.input_norm(source)))
        log_scale = self.max_log_scale * torch.tanh(
            self.log_scale_head(features)
        )
        phase_delta = self.max_phase_delta * torch.tanh(
            self.phase_head(features)
        )
        scale = torch.exp(log_scale)

        source_coordinates = F.linear(source, active_basis)
        rotated = rotate_pairwise(source_coordinates, phase_delta)
        corrected_pairs = rotated.reshape(
            *rotated.shape[:-1], self.pairs, 2
        ) * scale.unsqueeze(-1)
        coordinate_states = corrected_pairs.flatten(-2)
        states = F.linear(coordinate_states, active_basis.T)
        return SpectralCorrectorOutput(
            states=states.to(output_dtype),
            coordinate_states=coordinate_states.to(output_dtype),
            source_coordinates=source_coordinates.to(output_dtype),
            log_scale=log_scale,
            scale=scale,
            phase_delta=phase_delta,
        )


class DetachedSpectralStateCorrector(SpectralStateCorrector):
    """Isolate T regression from both proposal and shared-basis gradients."""

    def __init__(
        self,
        width: int,
        *,
        bottleneck: int = 128,
        max_log_scale: float = 2.0,
        max_phase_delta: float = math.pi,
    ) -> None:
        super().__init__(
            width,
            bottleneck=bottleneck,
            max_log_scale=max_log_scale,
            max_phase_delta=max_phase_delta,
            detach_inputs=True,
        )


class AttachedSpectralStateCorrector(SpectralStateCorrector):
    """Allow T regression to update its proposal and shared spectral basis."""

    def __init__(
        self,
        width: int,
        *,
        bottleneck: int = 128,
        max_log_scale: float = 2.0,
        max_phase_delta: float = math.pi,
    ) -> None:
        super().__init__(
            width,
            bottleneck=bottleneck,
            max_log_scale=max_log_scale,
            max_phase_delta=max_phase_delta,
            detach_inputs=False,
        )


__all__ = [
    "AttachedSpectralStateCorrector",
    "DetachedSpectralStateCorrector",
    "SpectralStateCorrector",
    "SpectralCorrectorOutput",
]
