"""Explicit real-pair spectral rotations for global latent time shifts."""
from __future__ import annotations

import torch
from torch import nn


class SpectralRotationOperator(nn.Module):
    """One learned radians-per-token frequency per real coordinate pair.

    The native latent coordinates are the spectral basis.  Every adjacent
    pair represents one complex coefficient, and ``apply_power(x, n)``
    multiplies that coefficient by ``exp(i * n * omega)``.

    ``A`` intentionally aliases the frequency parameter so existing producer
    infrastructure can report the global-operator gradient norm without
    knowing which orthogonal parameterization is installed.
    """

    def __init__(
        self,
        width: int,
        *,
        initial_frequency_range: float = 0.05,
    ) -> None:
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("spectral rotation width must be positive and even")
        if initial_frequency_range < 0:
            raise ValueError("initial frequency range must be non-negative")
        self.width = int(width)
        self.pairs = self.width // 2
        self.initial_frequency_range = float(initial_frequency_range)

        if initial_frequency_range == 0:
            frequencies = torch.zeros(self.pairs)
        else:
            # Midpoints avoid duplicating the registered interval endpoints
            # and deterministically break the all-identity mode symmetry.
            index = torch.arange(self.pairs, dtype=torch.float32)
            frequencies = -initial_frequency_range + (
                (index + 0.5)
                * (2.0 * initial_frequency_range / self.pairs)
            )
        self.A = nn.Parameter(frequencies)

    @property
    def frequencies(self) -> torch.Tensor:
        return self.A

    def apply_power(
        self,
        values: torch.Tensor,
        steps: int = 1,
    ) -> torch.Tensor:
        """Apply ``K**steps`` without materializing a dense matrix."""
        if values.shape[-1] != self.width:
            raise ValueError(
                f"expected trailing width {self.width}, got {values.shape[-1]}"
            )
        if not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        paired = values.reshape(*values.shape[:-1], self.pairs, 2)
        angle = self.frequencies * steps
        cosine = torch.cos(angle).to(values.dtype)
        sine = torch.sin(angle).to(values.dtype)
        real = paired[..., 0]
        imaginary = paired[..., 1]
        rotated = torch.stack(
            (
                cosine * real - sine * imaginary,
                sine * real + cosine * imaginary,
            ),
            dim=-1,
        )
        return rotated.reshape_as(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.apply_power(values, 1)

    @property
    def weight(self) -> torch.Tensor:
        """Return the represented dense weight for diagnostics/checkpoints."""
        cosine = torch.cos(self.frequencies)
        sine = torch.sin(self.frequencies)
        blocks = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        return torch.block_diag(*blocks.unbind(dim=0))

    def mode_energy(self, values: torch.Tensor) -> torch.Tensor:
        """Return squared amplitude for every spectral pair."""
        if values.shape[-1] != self.width:
            raise ValueError(
                f"expected trailing width {self.width}, got {values.shape[-1]}"
            )
        return values.float().reshape(
            *values.shape[:-1], self.pairs, 2
        ).square().sum(dim=-1)


__all__ = ["SpectralRotationOperator"]
