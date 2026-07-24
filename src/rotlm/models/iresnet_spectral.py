"""i-ResNet-style soft spectral normalization for shared linear maps.

Behrmann et al. (2019), Eq. (2), use the effective weight

    W_tilde = W / max(1, sigma_tilde(W) / c)

where ``c < 1`` and ``sigma_tilde`` is updated with power iteration.  This is
a differentiable reparameterization of the raw weight, not a loss penalty and
not an in-place projection of the optimizer parameter.

The original i-ResNet hook updates its power vectors whenever a layer is
called.  The reversible language model calls the same layer several times in
one encode/inverse-decode computation, so changing those vectors per call
would make the two directions use different effective weights.  Here power
vectors are updated explicitly once per top-level optimizer step.  Calls made
after that update use the same Eq. (2) weight and therefore preserve the
analytic inverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.nn.utils import parametrize


class IResNetSoftSpectralNorm(nn.Module):
    """Soft spectral normalization from the official i-ResNet implementation."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        coeff: float = 0.9,
        n_power_iterations: int = 5,
        eps: float = 1e-12,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if weight.ndim < 2:
            raise ValueError("spectral normalization requires a matrix-like weight")
        if not 0.0 < coeff < 1.0:
            raise ValueError(f"i-ResNet coefficient must be in (0,1), got {coeff}")
        if n_power_iterations <= 0:
            raise ValueError("n_power_iterations must be positive")
        self.coeff = float(coeff)
        self.n_power_iterations = int(n_power_iterations)
        self.eps = float(eps)

        matrix = self.reshape_weight_to_matrix(weight)
        u = torch.randn(
            matrix.shape[0],
            dtype=weight.dtype,
            device=weight.device,
            generator=generator,
        )
        v = torch.randn(
            matrix.shape[1],
            dtype=weight.dtype,
            device=weight.device,
            generator=generator,
        )
        self.register_buffer("u", self._normalize(u))
        self.register_buffer("v", self._normalize(v))
        self.register_buffer("sigma_estimate", weight.new_ones(()))

    @staticmethod
    def reshape_weight_to_matrix(weight: torch.Tensor) -> torch.Tensor:
        return weight.reshape(weight.shape[0], -1)

    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        return vector / vector.norm().clamp_min(self.eps)

    @torch.no_grad()
    def update_power_vectors(self, weight: torch.Tensor) -> float:
        """Run the configured W/W^T power iterations and update logging state."""
        matrix = self.reshape_weight_to_matrix(weight)
        u, v = self.u, self.v
        for _ in range(self.n_power_iterations):
            v = self._normalize(torch.mv(matrix.T, u))
            u = self._normalize(torch.mv(matrix, v))
        self.u.copy_(u)
        self.v.copy_(v)
        sigma = torch.dot(self.u, torch.mv(matrix, self.v))
        self.sigma_estimate.copy_(sigma)
        return float(sigma)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        matrix = self.reshape_weight_to_matrix(weight)
        sigma = torch.dot(self.u, torch.mv(matrix, self.v))
        factor = torch.clamp(sigma / self.coeff, min=1.0)
        return weight / factor


def apply_iresnet_soft_spectral_norm(
    linear: nn.Linear,
    *,
    coeff: float = 0.9,
    n_power_iterations: int = 5,
    eps: float = 1e-12,
    generator: torch.Generator | None = None,
) -> IResNetSoftSpectralNorm:
    """Attach the i-ResNet Eq. (2) parameterization to one linear layer."""
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
    if parametrize.is_parametrized(linear, "weight"):
        raise ValueError("linear.weight is already parametrized")
    spectral = IResNetSoftSpectralNorm(
        linear.weight,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
        eps=eps,
        generator=generator,
    )
    parametrize.register_parametrization(linear, "weight", spectral)
    return spectral


def iter_iresnet_spectral_layers(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Linear, IResNetSoftSpectralNorm]]:
    """Yield every linear layer carrying this repository's i-ResNet parameterization."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not parametrize.is_parametrized(module, "weight"):
            continue
        parameterizations = module.parametrizations.weight
        for parameterization in parameterizations:
            if isinstance(parameterization, IResNetSoftSpectralNorm):
                yield name, module, parameterization


@dataclass(frozen=True)
class IResNetSpectralStats:
    layer_count: int
    active_layer_count: int
    raw_sigma_max_estimate: float
    effective_sigma_max_estimate: float


@torch.no_grad()
def update_iresnet_power_vectors(model: nn.Module) -> IResNetSpectralStats:
    """Update every power vector once, matching one normal i-ResNet forward."""
    raw_sigmas: list[float] = []
    effective_sigmas: list[float] = []
    active = 0
    for _, linear, spectral in iter_iresnet_spectral_layers(model):
        raw_weight = linear.parametrizations.weight.original
        sigma = spectral.update_power_vectors(raw_weight)
        raw_sigmas.append(sigma)
        if sigma > spectral.coeff:
            active += 1
        effective_sigmas.append(sigma / max(1.0, sigma / spectral.coeff))
    if not raw_sigmas:
        raise ValueError("model has no i-ResNet spectral layers")
    return IResNetSpectralStats(
        layer_count=len(raw_sigmas),
        active_layer_count=active,
        raw_sigma_max_estimate=max(raw_sigmas),
        effective_sigma_max_estimate=max(effective_sigmas),
    )


@torch.no_grad()
def exact_effective_spectral_norms(model: nn.Module) -> dict[str, float]:
    """Compute exact matrix spectral norms for audit-sized reporting calls."""
    values: dict[str, float] = {}
    with parametrize.cached():
        for name, linear, _ in iter_iresnet_spectral_layers(model):
            values[name] = float(torch.linalg.matrix_norm(linear.weight.float(), ord=2))
    return values
