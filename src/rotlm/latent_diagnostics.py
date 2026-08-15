"""Reusable geometric and decision-boundary diagnostics for latent models."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
import torch.nn.functional as F


@dataclass
class OrthogonalDriftDiagnostics:
    """Rowwise drift before and after one global orthogonal alignment."""

    raw_cosine: torch.Tensor
    raw_relative_mse: torch.Tensor
    aligned_cosine: torch.Tensor
    aligned_relative_mse: torch.Tensor
    norm_ratio: torch.Tensor
    centered_linear_cka: torch.Tensor
    alignment: torch.Tensor
    alignment_orthogonality_error: torch.Tensor


@dataclass
class DecisionMarginDiagnostics:
    """Exact top-1 boundary decomposition for two sets of logits."""

    reference_top1: torch.Tensor
    candidate_top1: torch.Tensor
    matches: torch.Tensor
    reference_top1_margin: torch.Tensor
    candidate_reference_margin: torch.Tensor
    reference_margin_to_candidate_competitor: torch.Tensor
    differential_toward_candidate_competitor: torch.Tensor
    crossing_excess: torch.Tensor
    max_absolute_logit_delta: torch.Tensor


@dataclass
class AdditiveResidualDiagnostics:
    """Geometry of additive components relative to a target residual."""

    prior_cosine: torch.Tensor
    full_cosine: torch.Tensor
    prior_relative_mse: torch.Tensor
    full_relative_mse: torch.Tensor
    error_reduction: torch.Tensor
    innovation_residual_cosine: torch.Tensor
    innovation_residual_norm_ratio: torch.Tensor
    innovation_target_energy_ratio: torch.Tensor
    innovation_residual_dot_ratio: torch.Tensor
    optimal_innovation_scale: torch.Tensor
    oracle_relative_mse: torch.Tensor
    fixed_scale_regret: torch.Tensor
    head_shapley_error_reduction: torch.Tensor
    head_residual_cosine: torch.Tensor
    head_residual_norm_ratio: torch.Tensor


@dataclass
class RecurrentStepScaleDiagnostics:
    """Forward scales at every consumer boundary of one complex step."""

    rotated_memory_rms: torch.Tensor
    innovation_memory_rms: torch.Tensor
    innovation_to_memory_ratio: torch.Tensor
    memory_innovation_cosine: torch.Tensor
    raw_updated_memory_rms: torch.Tensor
    stored_memory_rms: torch.Tensor
    raw_preliminary_hidden_rms: torch.Tensor
    preliminary_hidden_rms: torch.Tensor
    raw_full_hidden_rms: torch.Tensor
    stored_hidden_rms: torch.Tensor
    preliminary_norm_tangent_bound: torch.Tensor
    updated_memory_norm_tangent_bound: torch.Tensor
    full_hidden_norm_tangent_bound: torch.Tensor


@dataclass
class ComplexRecurrentScaleTape:
    """Rowwise scale tape for one target-free complex recurrence.

    Initial fields have the root leading shape. Step fields append one final
    horizon dimension. Raw successor scales and their exact pre-boundary
    denominators are retained separately from the normalized stored carriers.
    """

    raw_root_hidden_rms: torch.Tensor
    initialized_hidden_rms: torch.Tensor
    initialized_memory_rms: torch.Tensor
    initialized_hidden_cosine_to_raw: torch.Tensor
    initialized_hidden_to_raw_rms_ratio: torch.Tensor
    rotated_hidden_rms: torch.Tensor
    rotated_memory_rms: torch.Tensor
    preliminary_hidden_rms: torch.Tensor
    innovation_memory_rms: torch.Tensor
    raw_full_hidden_rms: torch.Tensor
    raw_updated_memory_rms: torch.Tensor
    hidden_boundary_denominator: torch.Tensor
    memory_boundary_denominator: torch.Tensor
    stored_hidden_rms: torch.Tensor
    stored_memory_rms: torch.Tensor


@dataclass
class JacobianSpectralEstimate:
    """Matrix-free largest-singular-value power-iteration result."""

    singular_value: float
    history: tuple[float, ...]
    right_vector: torch.Tensor


def row_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return squared row error relative to the target row energy."""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share a rank >= 2 shape")
    numerator = (prediction.float() - target.float()).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-12)
    return numerator / denominator


def recurrent_step_scale_diagnostics(
    *,
    rotated_memory: torch.Tensor,
    innovation_memory: torch.Tensor,
    raw_updated_memory: torch.Tensor,
    stored_memory: torch.Tensor,
    raw_preliminary_hidden: torch.Tensor,
    preliminary_hidden: torch.Tensor,
    raw_full_hidden: torch.Tensor,
    stored_hidden: torch.Tensor,
    eps: float = 1e-6,
) -> RecurrentStepScaleDiagnostics:
    """Measure scales without conflating raw values and stored carriers."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    memory_shape = rotated_memory.shape
    if any(
        value.shape != memory_shape
        for value in (
            innovation_memory,
            raw_updated_memory,
            stored_memory,
        )
    ):
        raise ValueError("all complex memory tensors must share one shape")
    if not all(
        value.is_complex()
        for value in (
            rotated_memory,
            innovation_memory,
            raw_updated_memory,
            stored_memory,
        )
    ):
        raise ValueError("memory diagnostics require complex tensors")
    hidden_shape = raw_preliminary_hidden.shape
    if any(
        value.shape != hidden_shape
        for value in (
            preliminary_hidden,
            raw_full_hidden,
            stored_hidden,
        )
    ):
        raise ValueError("all hidden tensors must share one shape")

    memory_dims = tuple(range(rotated_memory.ndim - 3, rotated_memory.ndim))
    rotated_energy = rotated_memory.abs().square().float().mean(memory_dims)
    innovation_energy = innovation_memory.abs().square().float().mean(
        memory_dims
    )
    raw_updated_energy = raw_updated_memory.abs().square().float().mean(
        memory_dims
    )
    stored_memory_energy = stored_memory.abs().square().float().mean(
        memory_dims
    )
    inner = (
        rotated_memory.conj() * innovation_memory
    ).sum(dim=memory_dims).real.float()
    norm_product = torch.sqrt(
        rotated_memory.abs().square().float().sum(memory_dims)
        * innovation_memory.abs().square().float().sum(memory_dims)
    ).clamp_min(1e-12)

    def hidden_energy(value: torch.Tensor) -> torch.Tensor:
        return value.float().square().mean(dim=-1)

    raw_preliminary_energy = hidden_energy(raw_preliminary_hidden)
    raw_full_energy = hidden_energy(raw_full_hidden)
    return RecurrentStepScaleDiagnostics(
        rotated_memory_rms=rotated_energy.sqrt(),
        innovation_memory_rms=innovation_energy.sqrt(),
        innovation_to_memory_ratio=(
            innovation_energy.sqrt() / rotated_energy.sqrt().clamp_min(1e-12)
        ),
        memory_innovation_cosine=inner / norm_product,
        raw_updated_memory_rms=raw_updated_energy.sqrt(),
        stored_memory_rms=stored_memory_energy.sqrt(),
        raw_preliminary_hidden_rms=raw_preliminary_energy.sqrt(),
        preliminary_hidden_rms=hidden_energy(preliminary_hidden).sqrt(),
        raw_full_hidden_rms=raw_full_energy.sqrt(),
        stored_hidden_rms=hidden_energy(stored_hidden).sqrt(),
        preliminary_norm_tangent_bound=torch.rsqrt(
            raw_preliminary_energy + eps
        ),
        updated_memory_norm_tangent_bound=torch.rsqrt(
            raw_updated_energy + eps
        ),
        full_hidden_norm_tangent_bound=torch.rsqrt(raw_full_energy + eps),
    )


def complex_recurrent_scale_tape(
    transition,
    roots: torch.Tensor,
    horizons: int,
) -> ComplexRecurrentScaleTape:
    """Trace recurrent carrier scales without decoding or token supervision.

    ``transition`` is intentionally duck-typed so the helper remains reusable
    across registered wrappers of ``ComplexSelfPredictedKVTransition``. It must
    expose ``initialize``, ``forward`` and ``post_norm_eps`` and return the
    standard complex step dataclass.
    """
    if roots.ndim < 2 or roots.is_complex() or not roots.is_floating_point():
        raise ValueError("roots must be a real floating tensor with rank >= 2")
    if horizons < 1:
        raise ValueError("horizons must be positive")
    eps = float(transition.post_norm_eps)
    if eps <= 0:
        raise ValueError("transition post_norm_eps must be positive")

    def hidden_rms(value: torch.Tensor) -> torch.Tensor:
        return value.float().square().mean(dim=-1).sqrt()

    def memory_rms(value: torch.Tensor) -> torch.Tensor:
        if not value.is_complex() or value.ndim < 3:
            raise ValueError("recurrent memory must be a complex tensor")
        dimensions = tuple(range(value.ndim - 3, value.ndim))
        return value.abs().float().square().mean(dim=dimensions).sqrt()

    raw_root_rms = hidden_rms(roots)
    state = transition.initialize(roots)
    initialized_hidden_rms = hidden_rms(state.hidden)
    initialized_memory_rms = memory_rms(state.memory)
    root_cosine = F.cosine_similarity(
        state.hidden.float(), roots.float(), dim=-1, eps=1e-12
    )

    values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "rotated_hidden_rms",
            "rotated_memory_rms",
            "preliminary_hidden_rms",
            "innovation_memory_rms",
            "raw_full_hidden_rms",
            "raw_updated_memory_rms",
            "hidden_boundary_denominator",
            "memory_boundary_denominator",
            "stored_hidden_rms",
            "stored_memory_rms",
        )
    }
    for _ in range(horizons):
        step = transition(state)
        raw_hidden_energy = step.raw_full_hidden.float().square().mean(dim=-1)
        memory_dimensions = tuple(
            range(
                step.raw_updated_memory.ndim - 3,
                step.raw_updated_memory.ndim,
            )
        )
        raw_memory_energy = (
            step.raw_updated_memory.abs().float().square().mean(
                dim=memory_dimensions
            )
        )
        values["rotated_hidden_rms"].append(hidden_rms(step.rotated_hidden))
        values["rotated_memory_rms"].append(memory_rms(step.rotated_memory))
        values["preliminary_hidden_rms"].append(
            hidden_rms(step.preliminary_hidden)
        )
        values["innovation_memory_rms"].append(
            memory_rms(step.innovation_memory)
        )
        values["raw_full_hidden_rms"].append(raw_hidden_energy.sqrt())
        values["raw_updated_memory_rms"].append(raw_memory_energy.sqrt())
        values["hidden_boundary_denominator"].append(
            (raw_hidden_energy + eps).sqrt()
        )
        values["memory_boundary_denominator"].append(
            (raw_memory_energy + eps).sqrt()
        )
        values["stored_hidden_rms"].append(hidden_rms(step.state.hidden))
        values["stored_memory_rms"].append(memory_rms(step.state.memory))
        state = step.state

    stacked = {
        name: torch.stack(tape, dim=-1) for name, tape in values.items()
    }
    return ComplexRecurrentScaleTape(
        raw_root_hidden_rms=raw_root_rms,
        initialized_hidden_rms=initialized_hidden_rms,
        initialized_memory_rms=initialized_memory_rms,
        initialized_hidden_cosine_to_raw=root_cosine,
        initialized_hidden_to_raw_rms_ratio=(
            initialized_hidden_rms / raw_root_rms.clamp_min(1e-12)
        ),
        **stacked,
    )


def jacobian_spectral_norm_power_iteration(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    *,
    iterations: int = 8,
    seed: int = 0,
    tolerance: float = 1e-12,
) -> JacobianSpectralEstimate:
    """Estimate ``sigma_max(J_function(point))`` with JVP/VJP products.

    Both ``point`` and the function output must be real floating tensors. The
    function may use complex-valued intermediates; representing its public
    input and output in real coordinates keeps the adjoint metric explicit.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not point.is_floating_point() or point.is_complex():
        raise ValueError("point must be a real floating tensor")

    point = point.detach()
    generator = torch.Generator(device=point.device).manual_seed(seed)
    right = torch.randn(
        point.shape,
        dtype=point.dtype,
        device=point.device,
        generator=generator,
    )
    right = right / right.norm().clamp_min(tolerance)
    history: list[float] = []

    for _ in range(iterations):
        output, tangent = torch.func.jvp(function, (point,), (right,))
        if not output.is_floating_point() or output.is_complex():
            raise ValueError("function output must be a real floating tensor")
        sigma = tangent.norm()
        history.append(float(sigma.detach()))
        if not bool(torch.isfinite(sigma)) or float(sigma) <= tolerance:
            return JacobianSpectralEstimate(
                singular_value=float(sigma.detach()),
                history=tuple(history),
                right_vector=right.detach(),
            )
        left = tangent / sigma
        _, pullback = torch.func.vjp(function, point)
        (adjoint,) = pullback(left)
        adjoint_norm = adjoint.norm()
        if (
            not bool(torch.isfinite(adjoint_norm))
            or float(adjoint_norm) <= tolerance
        ):
            return JacobianSpectralEstimate(
                singular_value=float(sigma.detach()),
                history=tuple(history),
                right_vector=right.detach(),
            )
        right = (adjoint / adjoint_norm).detach()

    _, tangent = torch.func.jvp(function, (point,), (right,))
    singular_value = float(tangent.norm().detach())
    history.append(singular_value)
    return JacobianSpectralEstimate(
        singular_value=singular_value,
        history=tuple(history),
        right_vector=right,
    )


def orthogonal_drift_diagnostics(
    source: torch.Tensor,
    target: torch.Tensor,
) -> OrthogonalDriftDiagnostics:
    """Compare fixed state rows across coordinate systems.

    The returned alignment minimizes ``||source @ R - target||_F`` over
    orthogonal ``R``. CKA is computed after feature centering and is therefore
    insensitive to one global isotropic scale and orthogonal rotation.
    """
    if (
        source.ndim != 2
        or target.shape != source.shape
        or source.shape[0] < 2
        or source.shape[1] < 2
    ):
        raise ValueError("source and target must share a nontrivial 2D shape")
    source_float = source.float()
    target_float = target.float()
    if not bool(torch.isfinite(source_float).all()) or not bool(
        torch.isfinite(target_float).all()
    ):
        raise ValueError("source and target must be finite")

    cross = source_float.T @ target_float
    left, _, right_h = torch.linalg.svd(cross, full_matrices=False)
    alignment = left @ right_h
    aligned = source_float @ alignment

    source_centered = source_float - source_float.mean(dim=0, keepdim=True)
    target_centered = target_float - target_float.mean(dim=0, keepdim=True)
    centered_cross = source_centered.T @ target_centered
    source_gram = source_centered.T @ source_centered
    target_gram = target_centered.T @ target_centered
    cka_numerator = centered_cross.square().sum()
    cka_denominator = torch.sqrt(
        source_gram.square().sum() * target_gram.square().sum()
    ).clamp_min(1e-12)

    identity = torch.eye(
        alignment.shape[0],
        device=alignment.device,
        dtype=alignment.dtype,
    )
    return OrthogonalDriftDiagnostics(
        raw_cosine=F.cosine_similarity(source_float, target_float, dim=-1),
        raw_relative_mse=row_relative_mse(source_float, target_float),
        aligned_cosine=F.cosine_similarity(aligned, target_float, dim=-1),
        aligned_relative_mse=row_relative_mse(aligned, target_float),
        norm_ratio=(
            source_float.norm(dim=-1)
            / target_float.norm(dim=-1).clamp_min(1e-12)
        ),
        centered_linear_cka=cka_numerator / cka_denominator,
        alignment=alignment,
        alignment_orthogonality_error=(
            alignment.T @ alignment - identity
        ).abs().amax(),
    )


def decision_margin_diagnostics(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> DecisionMarginDiagnostics:
    """Decompose whether a candidate perturbation crosses reference top-1.

    For reference winner ``w`` and the strongest non-``w`` candidate token
    ``c``, a flip occurs exactly when

    ``(candidate[c]-reference[c]) - (candidate[w]-reference[w])``
    exceeds ``reference[w]-reference[c]``, apart from exact ties.
    """
    if (
        reference_logits.shape != candidate_logits.shape
        or reference_logits.ndim < 2
        or reference_logits.shape[-1] < 2
    ):
        raise ValueError("logits must share a rank >= 2 shape and vocabulary")
    reference = reference_logits.float()
    candidate = candidate_logits.float()
    if not bool(torch.isfinite(reference).all()) or not bool(
        torch.isfinite(candidate).all()
    ):
        raise ValueError("logits must be finite")

    reference_top2 = reference.topk(2, dim=-1)
    reference_top1 = reference_top2.indices[..., 0]
    candidate_top1 = candidate.argmax(dim=-1)
    winner_mask = F.one_hot(
        reference_top1,
        num_classes=reference.shape[-1],
    ).bool()
    negative_infinity = torch.full_like(candidate, -torch.inf)
    candidate_without_winner = torch.where(
        winner_mask,
        negative_infinity,
        candidate,
    )
    competitor = candidate_without_winner.argmax(dim=-1)

    reference_winner = reference.gather(
        -1, reference_top1.unsqueeze(-1)
    ).squeeze(-1)
    candidate_winner = candidate.gather(
        -1, reference_top1.unsqueeze(-1)
    ).squeeze(-1)
    reference_competitor = reference.gather(
        -1, competitor.unsqueeze(-1)
    ).squeeze(-1)
    candidate_competitor = candidate.gather(
        -1, competitor.unsqueeze(-1)
    ).squeeze(-1)
    reference_margin_to_competitor = (
        reference_winner - reference_competitor
    )
    differential = (
        candidate_competitor
        - reference_competitor
        - candidate_winner
        + reference_winner
    )
    return DecisionMarginDiagnostics(
        reference_top1=reference_top1,
        candidate_top1=candidate_top1,
        matches=reference_top1.eq(candidate_top1),
        reference_top1_margin=(
            reference_top2.values[..., 0] - reference_top2.values[..., 1]
        ),
        candidate_reference_margin=(
            candidate_winner - candidate_without_winner.amax(dim=-1)
        ),
        reference_margin_to_candidate_competitor=(
            reference_margin_to_competitor
        ),
        differential_toward_candidate_competitor=differential,
        crossing_excess=differential - reference_margin_to_competitor,
        max_absolute_logit_delta=(candidate - reference).abs().amax(dim=-1),
    )


def additive_residual_diagnostics(
    prior: torch.Tensor,
    components: torch.Tensor,
    target: torch.Tensor,
) -> AdditiveResidualDiagnostics:
    """Explain an additive update against the residual left by ``prior``.

    ``components`` has shape ``[*rows, parts, width]`` while ``prior`` and
    ``target`` have shape ``[*rows, width]``.  The per-part allocation is the
    exact Shapley value of squared-error reduction for an additive update. Its
    sum therefore equals the full update's error reduction without choosing a
    component ordering.
    """
    if (
        prior.shape != target.shape
        or prior.ndim < 2
        or components.ndim != prior.ndim + 1
        or components.shape[:-2] != prior.shape[:-1]
        or components.shape[-1] != prior.shape[-1]
        or components.shape[-2] < 1
    ):
        raise ValueError("prior, components, and target shapes are incompatible")
    prior_float = prior.float()
    target_float = target.float()
    component_float = components.float()
    if not all(
        bool(torch.isfinite(value).all())
        for value in (prior_float, target_float, component_float)
    ):
        raise ValueError("additive diagnostic inputs must be finite")

    innovation = component_float.sum(dim=-2)
    full = prior_float + innovation
    residual = target_float - prior_float
    target_energy = target_float.square().sum(dim=-1).clamp_min(1e-12)
    residual_norm = residual.norm(dim=-1).clamp_min(1e-12)
    innovation_energy = innovation.square().sum(dim=-1)
    optimal_scale = (
        (innovation * residual).sum(dim=-1)
        / innovation_energy.clamp_min(1e-12)
    )
    optimal_scale = torch.where(
        innovation_energy > 1e-12,
        optimal_scale,
        torch.zeros_like(optimal_scale),
    )
    oracle = prior_float + optimal_scale.unsqueeze(-1) * innovation

    component_residual_dot = (
        component_float * residual.unsqueeze(-2)
    ).sum(dim=-1)
    component_total_dot = (
        component_float * innovation.unsqueeze(-2)
    ).sum(dim=-1)
    head_shapley = (
        2.0 * component_residual_dot - component_total_dot
    ) / target_energy.unsqueeze(-1)
    expanded_residual = residual.unsqueeze(-2).expand_as(component_float)

    prior_relative_mse = (
        residual.square().sum(dim=-1) / target_energy
    )
    full_relative_mse = row_relative_mse(full, target_float)
    oracle_relative_mse = row_relative_mse(oracle, target_float)
    return AdditiveResidualDiagnostics(
        prior_cosine=F.cosine_similarity(prior_float, target_float, dim=-1),
        full_cosine=F.cosine_similarity(full, target_float, dim=-1),
        prior_relative_mse=prior_relative_mse,
        full_relative_mse=full_relative_mse,
        error_reduction=prior_relative_mse - full_relative_mse,
        innovation_residual_cosine=F.cosine_similarity(
            innovation,
            residual,
            dim=-1,
        ),
        innovation_residual_norm_ratio=(
            innovation.norm(dim=-1) / residual_norm
        ),
        innovation_target_energy_ratio=(
            innovation_energy / target_energy
        ),
        innovation_residual_dot_ratio=(
            (innovation * residual).sum(dim=-1) / target_energy
        ),
        optimal_innovation_scale=optimal_scale,
        oracle_relative_mse=oracle_relative_mse,
        fixed_scale_regret=full_relative_mse - oracle_relative_mse,
        head_shapley_error_reduction=head_shapley,
        head_residual_cosine=F.cosine_similarity(
            component_float,
            expanded_residual,
            dim=-1,
        ),
        head_residual_norm_ratio=(
            component_float.norm(dim=-1) / residual_norm.unsqueeze(-1)
        ),
    )


__all__ = [
    "AdditiveResidualDiagnostics",
    "ComplexRecurrentScaleTape",
    "DecisionMarginDiagnostics",
    "JacobianSpectralEstimate",
    "OrthogonalDriftDiagnostics",
    "RecurrentStepScaleDiagnostics",
    "additive_residual_diagnostics",
    "complex_recurrent_scale_tape",
    "decision_margin_diagnostics",
    "jacobian_spectral_norm_power_iteration",
    "orthogonal_drift_diagnostics",
    "recurrent_step_scale_diagnostics",
    "row_relative_mse",
]
