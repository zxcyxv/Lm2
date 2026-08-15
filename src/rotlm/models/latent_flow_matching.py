"""Conditional flow matching on decoder-visible latent rays."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .complex_self_prediction import ComplexSelfPredictedKVTransition
from .prefix_orthogonal_scan import rotate_pairwise


@dataclass
class SphericalFlowBridge:
    """A shortest-geodesic conditional bridge and its exact velocity."""

    state: torch.Tensor
    target_velocity: torch.Tensor
    angle: torch.Tensor


class ConditionalRayFlowTimeConditioner(nn.Module):
    """Inject explicit sinusoidal flow time without changing hidden width."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least two")
        self.width = int(width)
        self.directions = nn.Parameter(torch.zeros(2, width))

    def forward(
        self,
        states: torch.Tensor,
        flow_time: torch.Tensor | float,
    ) -> torch.Tensor:
        if states.shape[-1] != self.width:
            raise ValueError(
                f"state width must be {self.width}, got {states.shape[-1]}"
            )
        time = torch.as_tensor(
            flow_time,
            device=states.device,
            dtype=states.dtype,
        )
        while time.ndim < states.ndim:
            time = time.unsqueeze(-1)
        try:
            time = torch.broadcast_to(time, (*states.shape[:-1], 1))
        except RuntimeError as error:
            raise ValueError("flow_time does not broadcast over states") from error
        phase = math.pi * time
        shift = (
            phase.sin() * self.directions[0]
            + phase.cos() * self.directions[1]
        )
        return states + shift


def unit_rows(states: torch.Tensor) -> torch.Tensor:
    """Normalize the last axis in float32 for ray geometry."""
    return F.normalize(states.float(), dim=-1, eps=1e-8)


def tangent_projection(
    vectors: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    """Project vectors onto the unit-sphere tangent plane at ``base``."""
    if vectors.shape != base.shape:
        raise ValueError("vectors and base must have identical shapes")
    return vectors - (vectors * base).sum(dim=-1, keepdim=True) * base


def sphere_expmap(
    base: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    """Apply the unit-sphere exponential map row by row."""
    if base.shape != tangent.shape:
        raise ValueError("base and tangent must have identical shapes")
    base = unit_rows(base)
    tangent = tangent_projection(tangent.float(), base)
    angle = tangent.norm(dim=-1, keepdim=True)
    sinc = torch.where(
        angle > 1e-7,
        angle.sin() / angle.clamp_min(1e-12),
        1.0 - angle.square() / 6.0,
    )
    mapped = angle.cos() * base + sinc * tangent
    return unit_rows(mapped)


def conditional_tangent_source(
    root: torch.Tensor,
    horizons: int,
    noise_scale: float,
    *,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample equal-RMS independent tangent noise for every horizon row."""
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    root_unit = unit_rows(root)
    expanded_root = root_unit.unsqueeze(-2).expand(
        *root.shape[:-1],
        horizons,
        root.shape[-1],
    )
    expected_shape = expanded_root.shape
    if noise is None:
        noise = torch.randn(
            expected_shape,
            device=root.device,
            dtype=torch.float32,
            generator=generator,
        )
    elif noise.shape != expected_shape:
        raise ValueError(
            f"noise shape must be {tuple(expected_shape)}, got "
            f"{tuple(noise.shape)}"
        )
    tangent = tangent_projection(noise.float(), expanded_root)
    tangent = tangent * (float(noise_scale) / math.sqrt(root.shape[-1]))
    return sphere_expmap(expanded_root, tangent), noise


def shortest_spherical_bridge(
    source: torch.Tensor,
    target: torch.Tensor,
    flow_time: torch.Tensor | float,
) -> SphericalFlowBridge:
    """Return SLERP ``x_t`` and analytic ``d x_t / dt``."""
    if source.shape != target.shape:
        raise ValueError("source and target must have identical shapes")
    source = unit_rows(source)
    target = unit_rows(target)
    dot = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    orthogonal = target - dot * source
    orthogonal_norm = orthogonal.norm(dim=-1, keepdim=True)

    # Exact antipodes have no unique shortest geodesic.  Choose a stable
    # coordinate tangent; continuous random endpoints hit this only at
    # measure zero, but the branch makes the primitive total and testable.
    coordinate = source.abs().argmin(dim=-1, keepdim=True)
    basis = torch.zeros_like(source).scatter_(-1, coordinate, 1.0)
    fallback = tangent_projection(basis, source)
    fallback = F.normalize(fallback, dim=-1, eps=1e-8)
    direction = torch.where(
        orthogonal_norm > 1e-7,
        orthogonal / orthogonal_norm.clamp_min(1e-12),
        fallback,
    )
    angle = torch.atan2(orthogonal_norm, dot)
    angle = torch.where(
        (orthogonal_norm <= 1e-7) & (dot >= 0),
        torch.zeros_like(angle),
        angle,
    )

    time = torch.as_tensor(
        flow_time,
        device=source.device,
        dtype=source.dtype,
    )
    while time.ndim < source.ndim:
        time = time.unsqueeze(-1)
    try:
        time = torch.broadcast_to(time, (*source.shape[:-1], 1))
    except RuntimeError as error:
        raise ValueError("flow_time does not broadcast over endpoints") from error
    phase = time * angle
    state = phase.cos() * source + phase.sin() * direction
    velocity = angle * (
        -phase.sin() * source + phase.cos() * direction
    )
    return SphericalFlowBridge(
        state=unit_rows(state),
        target_velocity=velocity,
        angle=angle.squeeze(-1),
    )


def cumulative_hidden_angles(
    transition: ComplexSelfPredictedKVTransition,
    states: torch.Tensor,
) -> torch.Tensor:
    """Expand the learned horizon rotation for a complete tape."""
    if states.shape[-1] != transition.width:
        raise ValueError("state width does not match transition")
    horizons = states.shape[-2]
    return transition.hidden_phase.float().expand(
        *states.shape[:-2],
        horizons,
        -1,
    ).cumsum(dim=-2)


def to_rotating_frame_rays(
    transition: ComplexSelfPredictedKVTransition,
    states: torch.Tensor,
) -> torch.Tensor:
    """Normalize an original-frame H tape and co-rotate it by ``R^-h``."""
    state_unit = unit_rows(states)
    return rotate_pairwise(
        state_unit,
        -cumulative_hidden_angles(transition, state_unit),
    )


def from_rotating_frame_rays(
    transition: ComplexSelfPredictedKVTransition,
    rays: torch.Tensor,
) -> torch.Tensor:
    """Map unit rays to RMS-one original-frame decoder states."""
    rays = unit_rows(rays)
    original = rotate_pairwise(
        rays,
        cumulative_hidden_angles(transition, rays),
    )
    return original * math.sqrt(transition.width)


def conditional_ray_flow_velocity(
    transition: ComplexSelfPredictedKVTransition,
    time_conditioner: ConditionalRayFlowTimeConditioner,
    rotating_rays: torch.Tensor,
    flow_time: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate the shared central transition as a tangent flow field."""
    rays = unit_rows(rotating_rays)
    forcing = from_rotating_frame_rays(transition, rays)
    forcing = time_conditioner(forcing, flow_time)
    original_velocity = transition.forcing_velocity_tape(forcing).velocity
    rotating_velocity = rotate_pairwise(
        original_velocity.float() / math.sqrt(transition.width),
        -cumulative_hidden_angles(transition, rays),
    )
    return tangent_projection(rotating_velocity, rays)


def integrate_conditional_ray_flow(
    transition: ComplexSelfPredictedKVTransition,
    time_conditioner: ConditionalRayFlowTimeConditioner,
    source: torch.Tensor,
    *,
    steps: int = 8,
) -> torch.Tensor:
    """Integrate the tangent ODE with exponential-map Euler steps."""
    if steps < 1:
        raise ValueError("integration steps must be positive")
    state = unit_rows(source)
    step_size = 1.0 / steps
    for index in range(steps):
        flow_time = torch.full(
            (*state.shape[:-2], 1, 1),
            index * step_size,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = conditional_ray_flow_velocity(
            transition,
            time_conditioner,
            state,
            flow_time,
        )
        state = sphere_expmap(state, step_size * velocity)
    return state


__all__ = [
    "ConditionalRayFlowTimeConditioner",
    "SphericalFlowBridge",
    "conditional_ray_flow_velocity",
    "conditional_tangent_source",
    "cumulative_hidden_angles",
    "from_rotating_frame_rays",
    "integrate_conditional_ray_flow",
    "shortest_spherical_bridge",
    "sphere_expmap",
    "tangent_projection",
    "to_rotating_frame_rays",
    "unit_rows",
]
