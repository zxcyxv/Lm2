"""Prefix-conditioned block-frozen normal dynamics with radial retraction."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .prefix_orthogonal_scan import rotate_pairwise


@dataclass
class SpectralCollapseOutput:
    """Closed-form and unprojected future tapes for one prefix batch."""

    states: torch.Tensor
    coordinate_states: torch.Tensor
    unprojected_states: torch.Tensor
    unprojected_coordinate_states: torch.Tensor
    clean_first: torch.Tensor
    root_coordinates: torch.Tensor
    root_amplitudes: torch.Tensor
    radii: torch.Tensor
    rho: torch.Tensor
    phi: torch.Tensor
    phase_delta: torch.Tensor
    shell: torch.Tensor
    basis: torch.Tensor
    alpha: torch.Tensor


class InputConditionedSpectralCollapse(nn.Module):
    """Compile one prefix-conditioned normal operator and roll it out exactly.

    ``rho`` and ``phi`` depend on the prefix root but remain frozen across the
    requested future block. The shared basis is orthogonal. Consequently every
    fixed-prefix operator is real normal and all horizons admit a closed form.
    """

    def __init__(
        self,
        width: int,
        *,
        bottleneck: int = 128,
        initial_frequency_range: float = 0.05,
        max_phase_delta: float = 0.10,
        initial_radius: float = 0.90,
        max_decay_logit_delta: float = 2.0,
        alpha: float = 1.0,
        minimum_shell: float = 1e-4,
        amplitude_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("width must be positive and even")
        if bottleneck < 1:
            raise ValueError("bottleneck must be positive")
        if initial_frequency_range < 0 or max_phase_delta < 0:
            raise ValueError("frequency ranges must be non-negative")
        if not 0 < initial_radius < 1:
            raise ValueError("initial radius must be strictly between zero and one")
        if max_decay_logit_delta < 0:
            raise ValueError("decay-logit range must be non-negative")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0,1]")
        if minimum_shell <= 0 or amplitude_epsilon <= 0:
            raise ValueError("shell and amplitude floors must be positive")

        self.width = int(width)
        self.pairs = self.width // 2
        self.bottleneck = int(bottleneck)
        self.initial_frequency_range = float(initial_frequency_range)
        self.max_phase_delta = float(max_phase_delta)
        self.max_decay_logit_delta = float(max_decay_logit_delta)
        self.minimum_shell = float(minimum_shell)
        self.amplitude_epsilon = float(amplitude_epsilon)

        self.basis_generator = nn.Parameter(torch.zeros(width, width))
        index = torch.arange(self.pairs, dtype=torch.float32)
        base_phase = -initial_frequency_range + (
            (index + 0.5)
            * (2.0 * initial_frequency_range / self.pairs)
        )
        self.base_phase = nn.Parameter(base_phase)
        initial_nu = math.log(1.0 / initial_radius - 1.0)
        self.base_decay_logit = nn.Parameter(
            torch.full((self.pairs,), initial_nu)
        )

        self.input_norm = nn.RMSNorm(width, eps=1e-6)
        self.context = nn.Linear(width, bottleneck, bias=False)
        self.phase = nn.Linear(bottleneck, self.pairs)
        self.decay = nn.Linear(bottleneck, self.pairs)
        self.shell_raw = nn.Parameter(torch.zeros(self.pairs))
        self.register_buffer(
            "alpha_value",
            torch.tensor(float(alpha), dtype=torch.float32),
        )
        self.register_buffer(
            "shell_calibrated",
            torch.tensor(False),
        )

        nn.init.normal_(self.context.weight, std=0.02)
        # Small nonzero heads make K input-dependent at initialization while
        # remaining close to the deterministic global spectral baseline.
        nn.init.normal_(self.phase.weight, std=1e-3)
        nn.init.zeros_(self.phase.bias)
        nn.init.normal_(self.decay.weight, std=1e-3)
        nn.init.zeros_(self.decay.bias)

    @property
    def basis(self) -> torch.Tensor:
        skew = self.basis_generator - self.basis_generator.T
        return torch.matrix_exp(skew)

    @property
    def shell(self) -> torch.Tensor:
        return F.softplus(self.shell_raw) + self.minimum_shell

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_value

    @staticmethod
    def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
        return values + torch.log(-torch.expm1(-values))

    @torch.no_grad()
    def calibrate_shell(self, roots: torch.Tensor) -> None:
        """Initialize the positive shell from mean prefix amplitudes once."""
        if roots.ndim != 3 or roots.shape[-1] != self.width:
            raise ValueError("roots must have shape [batch,anchors,width]")
        basis = self.basis
        coordinates = F.linear(roots.float(), basis.float())
        amplitudes = coordinates.reshape(
            *coordinates.shape[:-1], self.pairs, 2
        ).square().sum(dim=-1).sqrt()
        target = amplitudes.mean(dim=(0, 1)).clamp_min(
            2.0 * self.minimum_shell
        )
        positive = (target - self.minimum_shell).clamp_min(
            self.minimum_shell
        )
        self.shell_raw.copy_(self._inverse_softplus(positive))
        self.shell_calibrated.fill_(True)

    def conditional_parameters(
        self,
        roots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return prefix-conditioned radius, phase, and phase delta."""
        if roots.shape[-1] != self.width:
            raise ValueError(
                f"root width must be {self.width}, got {roots.shape[-1]}"
            )
        condition = roots.detach().float()
        features = F.silu(self.context(self.input_norm(condition)))
        phase_delta = self.max_phase_delta * torch.tanh(
            self.phase(features)
        )
        decay_delta = self.max_decay_logit_delta * torch.tanh(
            self.decay(features)
        )
        phi = self.base_phase + phase_delta
        nu = self.base_decay_logit + decay_delta
        rho = torch.exp(-F.softplus(nu))
        return rho.to(roots.dtype), phi.to(roots.dtype), phase_delta

    def compile(
        self,
        roots: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compile coordinates and one block-frozen operator per root."""
        if roots.ndim != 3 or roots.shape[-1] != self.width:
            raise ValueError(
                "roots must have shape [batch,anchors,width], got "
                f"{tuple(roots.shape)}"
            )
        if not bool(self.shell_calibrated):
            raise RuntimeError("amplitude shell must be calibrated before use")
        basis = self.basis
        root_coordinates = F.linear(roots, basis)
        root_pairs = root_coordinates.reshape(
            *root_coordinates.shape[:-1], self.pairs, 2
        )
        root_amplitudes = root_pairs.float().square().sum(
            dim=-1
        ).sqrt().to(roots.dtype)
        rho, phi, phase_delta = self.conditional_parameters(roots)
        return root_coordinates, root_amplitudes, rho, phi, phase_delta, basis

    def project_to_shell(
        self,
        states: torch.Tensor,
        *,
        basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Retract arbitrary states onto the calibrated spectral shell."""
        if states.shape[-1] != self.width:
            raise ValueError(
                f"state width must be {self.width}, got {states.shape[-1]}"
            )
        if not bool(self.shell_calibrated):
            raise RuntimeError("amplitude shell must be calibrated before use")
        active_basis = self.basis if basis is None else basis
        if active_basis.shape != (self.width, self.width):
            raise ValueError(
                "basis must have shape "
                f"[{self.width},{self.width}], got {tuple(active_basis.shape)}"
            )
        coordinates = F.linear(states, active_basis)
        pairs = coordinates.reshape(
            *coordinates.shape[:-1],
            self.pairs,
            2,
        )
        amplitudes = pairs.float().square().sum(dim=-1).sqrt()
        directions = pairs / amplitudes.to(states.dtype).clamp_min(
            self.amplitude_epsilon
        ).unsqueeze(-1)
        shell = self.shell.to(states.dtype)
        projected_coordinates = (
            directions * shell.unsqueeze(-1)
        ).flatten(-2)
        return F.linear(projected_coordinates, active_basis.T)

    @staticmethod
    def _geometric_sum(beta: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
        denominator = 1.0 - beta
        near_one = denominator.abs() < 1e-6
        safe_denominator = torch.where(
            near_one,
            torch.ones_like(denominator),
            denominator,
        )
        ratio = (1.0 - powers) / safe_denominator.unsqueeze(-2)
        horizon = torch.arange(
            1,
            powers.shape[-2] + 1,
            device=powers.device,
            dtype=powers.dtype,
        ).view(*((1,) * (powers.ndim - 2)), -1, 1)
        return torch.where(near_one.unsqueeze(-2), horizon, ratio)

    def forward(
        self,
        roots: torch.Tensor,
        horizons: int,
    ) -> SpectralCollapseOutput:
        if horizons < 1:
            raise ValueError("horizons must be positive")
        (
            root_coordinates,
            root_amplitudes,
            rho,
            phi,
            phase_delta,
            basis,
        ) = self.compile(roots)
        leading = roots.ndim - 1
        steps = torch.arange(
            1,
            horizons + 1,
            device=roots.device,
            dtype=roots.dtype,
        ).view(*((1,) * leading), horizons, 1)
        alpha = self.alpha.to(device=roots.device, dtype=roots.dtype)
        beta = (1.0 - alpha) * rho
        beta_power = beta.unsqueeze(-2).pow(steps)
        geometric = self._geometric_sum(beta, beta_power)
        shell = self.shell.to(roots.dtype)
        radii = (
            beta_power * root_amplitudes.unsqueeze(-2)
            + alpha * shell * geometric
        )
        unprojected_radii = (
            rho.unsqueeze(-2).pow(steps)
            * root_amplitudes.unsqueeze(-2)
        )

        root_pairs = root_coordinates.reshape(
            *root_coordinates.shape[:-1], self.pairs, 2
        )
        denominator = root_amplitudes.clamp_min(
            self.amplitude_epsilon
        ).unsqueeze(-1)
        unit_pairs = root_pairs / denominator
        expanded_units = unit_pairs.unsqueeze(-3).expand(
            *unit_pairs.shape[:-2],
            horizons,
            self.pairs,
            2,
        ).flatten(-2)
        phase_tape = phi.unsqueeze(-2) * steps
        rotated_units = rotate_pairwise(expanded_units, phase_tape).reshape(
            *roots.shape[:-1], horizons, self.pairs, 2
        )
        coordinate_pairs = radii.unsqueeze(-1) * rotated_units
        unprojected_pairs = (
            unprojected_radii.unsqueeze(-1) * rotated_units
        )
        coordinate_states = coordinate_pairs.flatten(-2)
        unprojected_coordinate_states = unprojected_pairs.flatten(-2)
        states = F.linear(coordinate_states, basis.T)
        unprojected_states = F.linear(
            unprojected_coordinate_states,
            basis.T,
        )
        return SpectralCollapseOutput(
            states=states,
            coordinate_states=coordinate_states,
            unprojected_states=unprojected_states,
            unprojected_coordinate_states=unprojected_coordinate_states,
            clean_first=unprojected_states[..., 0, :],
            root_coordinates=root_coordinates,
            root_amplitudes=root_amplitudes,
            radii=radii,
            rho=rho,
            phi=phi,
            phase_delta=phase_delta,
            shell=shell,
            basis=basis,
            alpha=alpha,
        )

    def literal_rollout(
        self,
        roots: torch.Tensor,
        horizons: int,
    ) -> torch.Tensor:
        """Reference implementation of repeated ``Pi(K_A u)``."""
        (
            coordinates,
            _,
            rho,
            phi,
            _,
            basis,
        ) = self.compile(roots)
        alpha = self.alpha.to(device=roots.device, dtype=roots.dtype)
        shell = self.shell.to(roots.dtype)
        states = []
        current = coordinates
        for _ in range(horizons):
            current = rotate_pairwise(current, phi)
            pairs = current.reshape(
                *current.shape[:-1], self.pairs, 2
            )
            amplitude = pairs.float().square().sum(
                dim=-1
            ).sqrt().to(current.dtype)
            contracted = rho * amplitude
            projected = (
                (1.0 - alpha) * contracted + alpha * shell
            )
            unit = pairs / amplitude.clamp_min(
                self.amplitude_epsilon
            ).unsqueeze(-1)
            current = (projected.unsqueeze(-1) * unit).flatten(-2)
            states.append(F.linear(current, basis.T))
        return torch.stack(states, dim=-2)


__all__ = [
    "InputConditionedSpectralCollapse",
    "SpectralCollapseOutput",
]
