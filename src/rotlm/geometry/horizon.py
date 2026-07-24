"""Fixed horizon operators used by the finite-K spectral controls."""
from __future__ import annotations

import math

import torch


def cyclic_shift_channels(states: torch.Tensor, steps: int, period: int) -> torch.Tensor:
    """Apply the real regular representation of the cyclic group C_period.

    The final feature dimension is partitioned into groups of ``period``.  A
    shift is a parameter-free orthogonal permutation, and applying ``period``
    steps is exactly the identity.
    """
    if period < 2:
        raise ValueError("period must be at least two")
    if states.shape[-1] % period:
        raise ValueError(
            f"feature width {states.shape[-1]} is not divisible by period {period}"
        )
    grouped = states.reshape(*states.shape[:-1], states.shape[-1] // period, period)
    return torch.roll(grouped, shifts=int(steps) % period, dims=-1).reshape_as(states)


def real_dft_basis(
    period: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an orthonormal real DFT synthesis matrix ``[time, mode]``.

    Columns are ordered as DC, cosine/sine pairs in ascending frequency, and
    (for even periods) the Nyquist column last.  For period 8 this gives
    ``DC, cos1, sin1, cos2, sin2, cos3, sin3, Nyquist``.
    """
    if period < 2:
        raise ValueError("period must be at least two")
    time = torch.arange(period, dtype=dtype, device=device)
    columns = [torch.ones(period, dtype=dtype, device=device) / math.sqrt(period)]
    upper = (period - 1) // 2
    scale = math.sqrt(2.0 / period)
    for mode in range(1, upper + 1):
        phase = 2.0 * math.pi * mode * time / period
        columns.extend((scale * torch.cos(phase), scale * torch.sin(phase)))
    if period % 2 == 0:
        columns.append(((-1.0) ** time) / math.sqrt(period))
    return torch.stack(columns, dim=1)


def geometric_horizon_features(
    frequencies: torch.Tensor, horizons: int
) -> torch.Tensor:
    """Unit-row-norm real phase features for a bank of rotation frequencies."""
    if frequencies.ndim != 1:
        raise ValueError("frequencies must be one-dimensional")
    steps = torch.arange(horizons, dtype=frequencies.dtype, device=frequencies.device)
    phase = steps[:, None] * frequencies[None, :]
    features = torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)
    return features / math.sqrt(frequencies.numel())
