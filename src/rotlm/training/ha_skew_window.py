"""Sparse-anchor clean multi-horizon training primitives for h_A orbits."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.prefix_orthogonal_scan import rotate_pairwise
from rotlm.models.spectral_collapse import SpectralCollapseOutput
from rotlm.models.spectral_corrector import SpectralCorrectorOutput


@dataclass
class SparseWindowOutput:
    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    predicted_states: torch.Tensor
    decode_states: torch.Tensor
    clean_states: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    noise: torch.Tensor | None = None
    log_sigma: torch.Tensor | None = None


@dataclass
class MultiTrajectoryWindowOutput:
    sparse: SharedMultiTrajectorySparseOutput
    token_ce: torch.Tensor
    trajectory_ce: torch.Tensor
    responsibilities: torch.Tensor
    trajectory_mse: torch.Tensor | None
    posterior_ce: torch.Tensor
    marginal_ce: torch.Tensor
    state_nce: torch.Tensor | None
    state_nce_positive_distance: torch.Tensor | None
    state_nce_negative_distance: torch.Tensor | None
    state_nce_retrieval_accuracy: torch.Tensor | None
    prior_loss: torch.Tensor | None = None
    prior_winner_accuracy: torch.Tensor | None = None
    branch_distance_error: torch.Tensor | None = None
    behavioral_closure_kl: torch.Tensor | None = None
    behavioral_closure_by_horizon: torch.Tensor | None = None
    reanchored_states: torch.Tensor | None = None


@dataclass
class TokenConditionedSequentialOutput:
    """One recurrent path whose successor states consume observed tokens."""

    logits: torch.Tensor
    teacher_logits: torch.Tensor
    token_ce: torch.Tensor
    proposal_states: torch.Tensor
    conditioned_states: torch.Tensor
    innovations: torch.Tensor
    log_scale: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    student_readout_tapes: torch.Tensor
    canonical_readout_tapes: torch.Tensor
    behavioral_closure_kl: torch.Tensor
    behavioral_closure_by_horizon: torch.Tensor


@dataclass
class PrefixOrthogonalScanWindowOutput:
    """One deterministic prefix-only future computed by associative scan."""

    logits: torch.Tensor
    teacher_logits: torch.Tensor
    token_ce: torch.Tensor
    predicted_states: torch.Tensor
    coordinate_states: torch.Tensor
    local_angles: torch.Tensor
    innovations: torch.Tensor
    log_scale: torch.Tensor
    cumulative_angles: torch.Tensor
    cumulative_innovations: torch.Tensor
    basis: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    reanchored_states: torch.Tensor
    behavioral_closure_kl: torch.Tensor
    behavioral_closure_by_horizon: torch.Tensor


@dataclass
class SpectralCollapseWindowOutput:
    """One prefix-conditioned block-frozen spectral-collapse tape."""

    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    token_ce: torch.Tensor
    predicted_states: torch.Tensor
    unprojected_states: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    prefix_encoded: torch.Tensor
    full_positions: torch.Tensor
    rollout: SpectralCollapseOutput


@dataclass
class SpectralCorrectorWindowOutput:
    """CE-only spectral proposals and detached canonical corrections."""

    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    token_ce: torch.Tensor
    proposal_states: torch.Tensor
    selected_tokens: torch.Tensor
    canonical_states: torch.Tensor
    corrected_states: torch.Tensor
    raw_canonical_mse: torch.Tensor
    corrected_canonical_mse: torch.Tensor
    corrected_logits: torch.Tensor | None
    corrected_decoded_hidden: torch.Tensor | None
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    prefix_encoded: torch.Tensor
    full_positions: torch.Tensor
    rollout: SpectralCollapseOutput
    correction: SpectralCorrectorOutput


@dataclass
class AttachedSpectralOneStepOutput:
    """Raw one-token CE proposal and attached self-canonical T regression."""

    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    token_ce: torch.Tensor
    proposal_states: torch.Tensor
    selected_tokens: torch.Tensor
    canonical_states: torch.Tensor
    corrected_states: torch.Tensor
    raw_canonical_mse: torch.Tensor
    corrected_canonical_mse: torch.Tensor
    corrected_logits: torch.Tensor | None
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    prefix_encoded: torch.Tensor
    full_positions: torch.Tensor
    rollout: SpectralCollapseOutput
    correction: SpectralCorrectorOutput


@dataclass
class SelfCanonicalRedecodeKLOutput:
    """Raw CE proposal plus stop-gradient self-redecode KL correction."""

    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    token_ce: torch.Tensor
    proposal_states: torch.Tensor
    selected_tokens: torch.Tensor
    canonical_states: torch.Tensor
    teacher_logits: torch.Tensor
    corrected_states: torch.Tensor
    corrected_decoded_hidden: torch.Tensor
    corrected_logits: torch.Tensor
    raw_teacher_kl: torch.Tensor
    corrected_teacher_kl: torch.Tensor
    raw_canonical_mse: torch.Tensor
    corrected_canonical_mse: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    prefix_encoded: torch.Tensor
    full_positions: torch.Tensor
    rollout: SpectralCollapseOutput
    correction: SpectralCorrectorOutput


@dataclass
class StateConditionedLatentTeacherForcingOutput:
    """Canonical one-step teachers and one self-fed latent rollout."""

    teacher_logits: torch.Tensor | None
    closure_teacher_logits: torch.Tensor | None
    open_logits: torch.Tensor | None
    teacher_token_ce: torch.Tensor
    open_token_ce: torch.Tensor
    open_supervised_token_nll: torch.Tensor
    canonical_inputs: torch.Tensor
    teacher_predicted_states: torch.Tensor
    open_states: torch.Tensor
    teacher_rotated_states: torch.Tensor
    open_rotated_states: torch.Tensor
    teacher_innovations: torch.Tensor
    open_innovations: torch.Tensor
    teacher_angles: torch.Tensor
    open_angles: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    behavioral_closure_kl: torch.Tensor
    behavioral_closure_by_horizon: torch.Tensor
    online_behavioral_closure_kl: torch.Tensor
    online_behavioral_closure_by_horizon: torch.Tensor


@dataclass
class ComplexSelfPredictedKVWindowOutput:
    """One target-free complex-memory tape with detached latent targets."""

    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    token_ce: torch.Tensor
    latent_relative_mse: torch.Tensor
    full_states: torch.Tensor
    read_states: torch.Tensor
    preliminary_states: torch.Tensor
    rotated_states: torch.Tensor
    innovation_deltas: torch.Tensor
    memory_energy: torch.Tensor
    innovation_energy: torch.Tensor
    beta: torch.Tensor | None
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    prefix_encoded: torch.Tensor
    full_positions: torch.Tensor


@dataclass
class ComplexStochasticRayFlowWindowOutput:
    """Noisy scan tape with detached sample-path ray velocities.

    The future encoder tape is used only as a collection of observed endpoint
    samples.  It is not interpreted as the prefix-conditioned predictive
    distribution.  ``ray_velocity_*`` compare discrete co-transport
    increments on the decoder-visible carrier ray.
    """

    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    token_ce: torch.Tensor
    full_states: torch.Tensor
    read_states: torch.Tensor
    innovation_deltas: torch.Tensor
    memory_states: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    prefix_encoded: torch.Tensor
    full_positions: torch.Tensor
    ray_velocity_loss_by_horizon: torch.Tensor
    ray_velocity_cosine_by_horizon: torch.Tensor
    value_noise: torch.Tensor


@dataclass
class ComplexSelfPredictionSelfRedecodeKLOutput:
    """Complex recurrence with a self-selected EMA redecode target."""

    base: ComplexSelfPredictedKVWindowOutput
    selected_tokens: torch.Tensor
    ema_prefix_encoded: torch.Tensor
    canonical_states: torch.Tensor
    canonical_logits: torch.Tensor
    canonical_log_probs: torch.Tensor
    raw_ema_logits: torch.Tensor
    raw_ema_log_probs: torch.Tensor
    student_decoded_hidden: torch.Tensor
    student_logits: torch.Tensor
    student_log_probs: torch.Tensor
    canonical_raw_kl: torch.Tensor
    canonical_student_kl: torch.Tensor
    canonical_relative_mse: torch.Tensor


@dataclass
class SharedMultiTrajectorySparseOutput:
    """One shared deterministic tape plus explicit noisy trajectory tapes.

    Shared fields have shapes ``[batch,anchors,horizons,...]``. Trajectory
    fields have an additional dimension after batch:
    ``[batch,trajectories,anchors,horizons,...]``.
    """

    logits: torch.Tensor
    decoded_hidden: torch.Tensor
    predicted_states: torch.Tensor
    decode_states: torch.Tensor
    clean_states: torch.Tensor
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    noise: torch.Tensor
    log_sigma: torch.Tensor
    prior_probs: torch.Tensor | None = None
    branch_residuals: torch.Tensor | None = None
    branch_operator_orthogonality_error: torch.Tensor | None = None
    prefix_encoded: torch.Tensor | None = None
    full_positions: torch.Tensor | None = None


def relative_mse_rows(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    numerator = (
        prediction.float() - target.float()
    ).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def forward_kl_rows(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    detach_teacher: bool = True,
) -> torch.Tensor:
    """Return ``KL(teacher || student)`` for every non-vocabulary row."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher and student logits must have identical shapes, got "
            f"{tuple(teacher_logits.shape)} and "
            f"{tuple(student_logits.shape)}"
        )
    active_teacher = (
        teacher_logits.detach() if detach_teacher else teacher_logits
    )
    calculation_dtype = (
        torch.float64
        if active_teacher.dtype == torch.float64
        or student_logits.dtype == torch.float64
        else torch.float32
    )
    teacher_log_probs = F.log_softmax(
        active_teacher.to(calculation_dtype), dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(
        student_logits.to(calculation_dtype), dim=-1
    )
    return (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)


def symmetric_state_info_nce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contrast attached state pairs against same-anchor batch negatives.

    Both tensors have shape ``[batch, anchors, width]``. The pairwise energy
    is scale-invariant symmetric relative squared distance, evaluated without
    materializing ``[anchors, batch, batch, width]`` differences.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must share [batch,anchors,width] shape"
        )
    if prediction.shape[0] < 2:
        raise ValueError("state InfoNCE requires at least two examples")
    if temperature <= 0:
        raise ValueError("state InfoNCE temperature must be positive")

    predicted = prediction.float().permute(1, 0, 2)
    observed = target.float().permute(1, 0, 2)
    predicted_norm_sq = predicted.square().sum(dim=-1)
    observed_norm_sq = observed.square().sum(dim=-1)
    dot = torch.bmm(predicted, observed.transpose(1, 2))
    squared_distance = (
        predicted_norm_sq[:, :, None]
        + observed_norm_sq[:, None, :]
        - 2.0 * dot
    ).clamp_min(0.0)
    geometric_energy_scale = torch.sqrt(
        predicted_norm_sq[:, :, None]
        * observed_norm_sq[:, None, :]
    ).clamp_min(1e-8)
    distance = squared_distance / geometric_energy_scale
    logits = -distance / temperature

    anchors, batch, _ = logits.shape
    labels = torch.arange(batch, device=logits.device).repeat(anchors)
    forward_loss = F.cross_entropy(logits.reshape(-1, batch), labels)
    reverse_loss = F.cross_entropy(
        logits.transpose(1, 2).reshape(-1, batch),
        labels,
    )
    loss = 0.5 * (forward_loss + reverse_loss)

    diagonal = distance.diagonal(dim1=1, dim2=2)
    negative_mask = ~torch.eye(
        batch, device=distance.device, dtype=torch.bool
    )[None]
    negative_distance = distance.masked_select(negative_mask).mean()
    retrieval_accuracy = 0.5 * (
        logits.argmax(dim=2)
        .eq(torch.arange(batch, device=logits.device)[None])
        .float()
        .mean()
        + logits.argmax(dim=1)
        .eq(torch.arange(batch, device=logits.device)[None])
        .float()
        .mean()
    )
    return (
        loss,
        diagonal.mean(),
        negative_distance,
        retrieval_accuracy,
    )


def sparse_anchor_indices(
    prefix_length: int,
    stride: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if prefix_length < 1 or stride < 1:
        raise ValueError("prefix_length and stride must be positive")
    return torch.arange(
        0, prefix_length, stride, device=device, dtype=torch.long
    )


def operator_orbit(
    model: K1DecoderAblationLM,
    roots: torch.Tensor,
    horizons: int,
    *,
    detach_between_horizons: bool = False,
) -> torch.Tensor:
    if horizons < 1:
        raise ValueError("horizons must be positive")
    states = []
    current = roots
    for _ in range(horizons):
        operator_input = (
            current.detach()
            if detach_between_horizons and states
            else current
        )
        current = apply_latent_operator(model, operator_input)
        states.append(current)
    return torch.stack(states, dim=2)


def apply_latent_operator(
    model: K1DecoderAblationLM,
    values: torch.Tensor,
    *,
    steps: int = 1,
) -> torch.Tensor:
    """Apply a model's global operator through its most exact interface."""
    if not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if hasattr(model.operator, "apply_power"):
        return model.operator.apply_power(values, steps)
    current = values
    weight = model.operator.weight
    bias = getattr(model.operator, "bias", None)
    if weight.ndim != 2:
        raise ValueError("operator weight must be a matrix")
    if steps < 0:
        raise ValueError("negative powers require an explicit operator method")
    for _ in range(steps):
        current = F.linear(current, weight, bias)
    return current


def _encoded_sparse_window_components(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    required_length = prefix_length + horizons
    if window.ndim != 2 or window.shape[1] != required_length:
        raise ValueError(
            f"window must have shape [batch,{required_length}]"
        )
    full_positions = torch.arange(
        required_length, device=window.device
    )
    full_encoded, _ = model.encode(window, full_positions)
    prefix_encoded = full_encoded[:, :prefix_length]
    anchor_indices = sparse_anchor_indices(
        prefix_length, anchor_stride, device=window.device
    )
    roots = prefix_encoded.index_select(1, anchor_indices)
    offsets = torch.arange(
        1, horizons + 1, device=window.device, dtype=torch.long
    )
    target_positions = anchor_indices[:, None] + offsets[None, :]
    flat_target_positions = target_positions.reshape(-1)
    gold_states = full_encoded.index_select(
        1, flat_target_positions
    ).reshape(
        window.shape[0],
        anchor_indices.numel(),
        horizons,
        model.width,
    )
    targets = window.index_select(
        1, flat_target_positions
    ).reshape(window.shape[0], anchor_indices.numel(), horizons)
    return (
        prefix_encoded,
        full_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    )


def _sparse_window_components(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        prefix_encoded,
        _,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _encoded_sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    return (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    )


def forward_sparse_complex_self_predicted_kv_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    write_scale: float = 1.0,
    read_beta: torch.Tensor | float | None = None,
    detach_latent_targets: bool = False,
    latent_target_model: K1DecoderAblationLM | None = None,
    detach_preliminary_for_update: bool = False,
    detach_cross_horizon_gradients: bool = False,
) -> ComplexSelfPredictedKVWindowOutput:
    """Roll out self-predicted KV writes and decode one causal latent tape.

    Held-out future tokens participate only in the online encoder targets and
    token labels.  The central path starts at each encoded prefix root and
    thereafter consumes only its own continuous hidden and matrix memory.
    """
    if not hasattr(model, "complex_self_prediction"):
        raise ValueError("model must define complex_self_prediction")
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    if latent_target_model is not None:
        if latent_target_model.width != model.width:
            raise ValueError("latent target and online widths must match")
        with torch.no_grad():
            (
                _,
                _,
                target_gold_states,
                target_tokens,
                target_anchor_indices,
                target_full_positions,
            ) = _sparse_window_components(
                latent_target_model,
                window,
                prefix_length=prefix_length,
                horizons=horizons,
                anchor_stride=anchor_stride,
            )
        if not torch.equal(target_tokens, targets):
            raise RuntimeError("latent target model changed token labels")
        if not torch.equal(target_anchor_indices, anchor_indices):
            raise RuntimeError("latent target model changed anchor indices")
        if not torch.equal(target_full_positions, full_positions):
            raise RuntimeError("latent target model changed positions")
        gold_states = target_gold_states
    rollout = model.complex_self_prediction.rollout(
        roots,
        horizons,
        write_scale=write_scale,
        read_beta=read_beta,
        detach_preliminary_for_update=detach_preliminary_for_update,
        detach_state_between_horizons=(
            detach_cross_horizon_gradients
        ),
    )
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        rollout.read_states,
        full_positions[:prefix_length],
        anchor_indices,
        detach_past_branch_states=detach_cross_horizon_gradients,
    )
    logits = model.token_logits(decoded_hidden)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    latent_relative_mse = relative_mse_rows(
        rollout.full_states,
        gold_states.detach() if detach_latent_targets else gold_states,
    )
    return ComplexSelfPredictedKVWindowOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        token_ce=token_ce,
        latent_relative_mse=latent_relative_mse,
        full_states=rollout.full_states,
        read_states=rollout.read_states,
        preliminary_states=rollout.preliminary_states,
        rotated_states=rollout.rotated_states,
        innovation_deltas=rollout.innovation_deltas,
        memory_energy=rollout.memory_energy,
        innovation_energy=rollout.innovation_energy,
        beta=rollout.beta,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        prefix_encoded=prefix_encoded,
        full_positions=full_positions,
    )


def sparse_complex_self_predicted_kv_ce_only_fast_logits(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    central_rollout: str = "feedback",
    feedback_rescans: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CE logits/targets without future encodes or diagnostic tapes.

    ``feedback_rescans`` applies only to the time-varying scan and performs
    finite shared-model Picard refinements toward the feedback recurrence.
    """
    required_length = prefix_length + horizons
    if window.ndim != 2 or window.shape[1] != required_length:
        raise ValueError(
            f"window must have shape [batch,{required_length}]"
        )
    prefix_positions = torch.arange(prefix_length, device=window.device)
    prefix_encoded, _ = model.encode(
        window[:, :prefix_length],
        prefix_positions,
    )
    anchor_indices = sparse_anchor_indices(
        prefix_length,
        anchor_stride,
        device=window.device,
    )
    roots = prefix_encoded.index_select(1, anchor_indices)
    offsets = torch.arange(
        1,
        horizons + 1,
        device=window.device,
        dtype=torch.long,
    )
    target_positions = anchor_indices[:, None] + offsets[None, :]
    targets = window.index_select(
        1,
        target_positions.reshape(-1),
    ).reshape(window.shape[0], anchor_indices.numel(), horizons)
    if not isinstance(feedback_rescans, int) or feedback_rescans < 0:
        raise ValueError("feedback_rescans must be a non-negative integer")
    if central_rollout == "feedback":
        if feedback_rescans:
            raise ValueError("feedback rollout does not accept rescans")
        read_states, _ = model.complex_self_prediction.rollout_read_states(
            roots,
            horizons,
        )
    elif central_rollout == "time-varying-scan":
        if feedback_rescans:
            read_states, _ = (
                model.complex_self_prediction
                .rollout_iterative_feedback_scan_read_states(
                    roots,
                    horizons,
                    feedback_rescans=feedback_rescans,
                )
            )
        else:
            read_states, _ = (
                model.complex_self_prediction
                .rollout_time_varying_scan_read_states(roots, horizons)
            )
    else:
        raise ValueError(f"unknown central rollout: {central_rollout}")
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        read_states,
        prefix_positions,
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    return logits, targets


def sparse_complex_self_predicted_kv_ce_only_fast_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    central_rollout: str = "feedback",
    feedback_rescans: int = 0,
) -> torch.Tensor:
    """Mean CE for either exact central recurrence implementation."""
    logits, targets = sparse_complex_self_predicted_kv_ce_only_fast_logits(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        central_rollout=central_rollout,
        feedback_rescans=feedback_rescans,
    )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
    )


def complex_standard_normal(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    real_dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample circular complex noise with unit expected squared magnitude."""
    if real_dtype not in (torch.float32, torch.float64):
        raise ValueError("complex noise requires float32 or float64 components")
    real = torch.randn(
        shape,
        device=device,
        dtype=real_dtype,
        generator=generator,
    )
    imaginary = torch.randn(
        shape,
        device=device,
        dtype=real_dtype,
        generator=generator,
    )
    return torch.complex(real, imaginary) * (2.0 ** -0.5)


def sparse_complex_stochastic_ray_flow_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    value_noise_scale: float,
    ray_flow_weight: float,
    ray_flow_horizons: int,
    value_noise: torch.Tensor | None = None,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    ComplexStochasticRayFlowWindowOutput,
]:
    """Train CE on a stochastic low-rank scan plus ray-velocity matching.

    Each complex noise row perturbs only the value side of one rank-one write.
    The complete noise tape is sampled before the scan, so the memory and
    latent recurrences remain exact causal prefix compositions.

    The auxiliary compares the decoder-visible carrier-ray increment

    ``normalize(z_j) - R normalize(z_(j-1))``

    with the same discrete increment on the detached observed future-encoder
    sample path.  This is a controlled stochastic discrete velocity objective,
    not a claim that one posterior encoder row is a predictive distribution or
    that the loss alone constitutes continuous-time flow matching.
    """
    if value_noise_scale < 0:
        raise ValueError("value_noise_scale must be non-negative")
    if ray_flow_weight < 0:
        raise ValueError("ray_flow_weight must be non-negative")
    if not 1 <= ray_flow_horizons <= horizons:
        raise ValueError("ray_flow_horizons must lie in [1, horizons]")
    if not hasattr(model, "complex_self_prediction"):
        raise ValueError("model must define complex_self_prediction")

    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    transition = model.complex_self_prediction
    expected_noise_shape = (
        *roots.shape[:-1],
        horizons,
        transition.heads,
        transition.value_dim,
    )
    if value_noise is None:
        value_noise = complex_standard_normal(
            expected_noise_shape,
            device=roots.device,
            real_dtype=roots.dtype,
            generator=noise_generator,
        )
    elif value_noise.shape != expected_noise_shape:
        raise ValueError(
            "value_noise shape mismatch: expected "
            f"{expected_noise_shape}, got {tuple(value_noise.shape)}"
        )

    rollout = transition.rollout_time_varying_scan(
        roots,
        horizons,
        value_noise=value_noise,
        value_noise_scale=value_noise_scale,
    )
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        rollout.read_states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    token_nll = token_ce.mean()

    predicted_unit = F.normalize(
        rollout.full_states.float(),
        dim=-1,
    )
    predicted_previous_unit = torch.cat(
        (
            F.normalize(roots.float(), dim=-1).unsqueeze(-2),
            predicted_unit[..., :-1, :],
        ),
        dim=-2,
    )
    local_angles = transition.hidden_phase.float().expand(
        *roots.shape[:-1],
        horizons,
        -1,
    )
    predicted_velocity = predicted_unit - rotate_pairwise(
        predicted_previous_unit,
        local_angles,
    )

    with torch.no_grad():
        gold_unit = F.normalize(gold_states.float(), dim=-1)
        gold_previous_unit = torch.cat(
            (
                F.normalize(roots.detach().float(), dim=-1).unsqueeze(-2),
                gold_unit[..., :-1, :],
            ),
            dim=-2,
        )
        target_velocity = gold_unit - rotate_pairwise(
            gold_previous_unit,
            local_angles.detach(),
        )

    target_energy = target_velocity.square().sum(dim=-1).clamp_min(1e-8)
    ray_velocity_rows = (
        (predicted_velocity - target_velocity).square().sum(dim=-1)
        / target_energy
    )
    ray_velocity_loss_by_horizon = ray_velocity_rows.mean(dim=(0, 1))
    ray_velocity_cosine_by_horizon = F.cosine_similarity(
        predicted_velocity,
        target_velocity,
        dim=-1,
        eps=1e-8,
    ).mean(dim=(0, 1))
    ray_flow_loss = ray_velocity_loss_by_horizon[
        :ray_flow_horizons
    ].mean()
    loss = token_nll + ray_flow_weight * ray_flow_loss

    output = ComplexStochasticRayFlowWindowOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        token_ce=token_ce,
        full_states=rollout.full_states,
        read_states=rollout.read_states,
        innovation_deltas=rollout.innovation_deltas,
        memory_states=rollout.memory_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        prefix_encoded=prefix_encoded,
        full_positions=full_positions,
        ray_velocity_loss_by_horizon=ray_velocity_loss_by_horizon,
        ray_velocity_cosine_by_horizon=ray_velocity_cosine_by_horizon,
        value_noise=value_noise,
    )
    return loss, token_nll, ray_flow_loss, output


@torch.inference_mode()
def evaluate_sparse_complex_self_predicted_kv_ce_only(
    model: K1DecoderAblationLM,
    windows: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    central_rollout: str = "feedback",
    feedback_rescans: int = 0,
    microbatch: int | None = None,
) -> dict[str, float]:
    """Evaluate block and per-horizon CE for a central rollout contract."""
    if windows.ndim != 2:
        raise ValueError("evaluation windows must have shape [batch,length]")
    if microbatch is None:
        microbatch = windows.shape[0]
    if microbatch < 1:
        raise ValueError("evaluation microbatch must be positive")

    was_training = model.training
    model.eval()
    nll_sum = torch.zeros(horizons, dtype=torch.float64)
    correct_sum = torch.zeros(horizons, dtype=torch.float64)
    labels_per_horizon = 0
    try:
        for begin in range(0, windows.shape[0], microbatch):
            logits, targets = (
                sparse_complex_self_predicted_kv_ce_only_fast_logits(
                    model,
                    windows[begin : begin + microbatch],
                    prefix_length=prefix_length,
                    horizons=horizons,
                    anchor_stride=anchor_stride,
                    central_rollout=central_rollout,
                    feedback_rescans=feedback_rescans,
                )
            )
            token_nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            nll_sum += token_nll.double().sum(dim=(0, 1)).cpu()
            correct_sum += (
                logits.argmax(dim=-1).eq(targets).double()
                .sum(dim=(0, 1)).cpu()
            )
            labels_per_horizon += targets.shape[0] * targets.shape[1]
    finally:
        model.train(was_training)

    horizon_nll = nll_sum / labels_per_horizon
    horizon_accuracy = correct_sum / labels_per_horizon
    metrics = {
        "block_nll": float(horizon_nll.mean()),
        "block_accuracy": float(horizon_accuracy.mean()),
    }
    for horizon in range(horizons):
        metrics[f"h{horizon + 1}_nll"] = float(horizon_nll[horizon])
        metrics[f"h{horizon + 1}_accuracy"] = float(
            horizon_accuracy[horizon]
        )
    return metrics


def sparse_complex_self_predicted_kv_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    latent_weight: float = 1.0,
    detach_latent_targets: bool = False,
    latent_target_model: K1DecoderAblationLM | None = None,
    detach_cross_horizon_gradients: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    ComplexSelfPredictedKVWindowOutput,
]:
    """Return raw-tied CE plus online latent closure.

    ``detach_latent_targets`` changes only the target-side backward path when
    the online encoder produces the target. ``latent_target_model`` instead
    obtains that target from a separate gradient-free model, such as an EMA
    copy; neither option changes the target-free central recurrence.
    """
    if latent_weight < 0:
        raise ValueError("latent_weight must be non-negative")
    output = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        detach_latent_targets=detach_latent_targets,
        latent_target_model=latent_target_model,
        detach_cross_horizon_gradients=(
            detach_cross_horizon_gradients
        ),
    )
    token_nll = output.token_ce.mean()
    latent_mse = output.latent_relative_mse.mean()
    return (
        token_nll + latent_weight * latent_mse,
        token_nll,
        latent_mse,
        output,
    )


def forward_sparse_complex_self_prediction_self_redecode_kl(
    model: K1DecoderAblationLM,
    ema_model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    anchor_stride: int,
    detach_preliminary_for_update: bool = False,
) -> ComplexSelfPredictionSelfRedecodeKLOutput:
    """Decode online Znext against the EMA redecode of P's own argmax.

    The full-model EMA is frozen. Its inverse decoder remains differentiable
    with respect to the online recurrent state, so KL updates the online
    transition without creating gradients for any EMA parameter.
    """
    if any(parameter.requires_grad for parameter in ema_model.parameters()):
        raise ValueError("EMA redecode model must be gradient-free")
    base = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=1,
        anchor_stride=anchor_stride,
        detach_preliminary_for_update=detach_preliminary_for_update,
    )
    selected_tokens = base.logits.detach().argmax(dim=-1)
    prefix_tokens = window[:, :prefix_length]
    positions = base.full_positions[:prefix_length]

    with torch.no_grad():
        ema_prefix_encoded, _ = ema_model.encode(prefix_tokens, positions)
        dense_actions = torch.zeros_like(prefix_tokens)
        dense_actions.index_copy_(
            1,
            base.anchor_indices,
            selected_tokens.squeeze(-1),
        )
        canonical_all = ema_model.dense_action_encode(
            prefix_tokens,
            dense_actions,
            positions,
        )
        canonical_states = canonical_all.index_select(
            1,
            base.anchor_indices,
        ).unsqueeze(2)
        canonical_decoded = (
            ema_model.exact_inverse_selected_decode_tape_states(
                ema_prefix_encoded,
                canonical_states,
                positions,
                base.anchor_indices,
            )
        )
        canonical_logits = ema_model.token_logits(canonical_decoded).float()
        canonical_log_probs = F.log_softmax(canonical_logits, dim=-1)

    raw_ema_decoded = ema_model.exact_inverse_selected_decode_tape_states(
        ema_prefix_encoded,
        base.read_states,
        positions,
        base.anchor_indices,
    )
    raw_ema_logits = ema_model.token_logits(raw_ema_decoded).float()
    raw_ema_log_probs = F.log_softmax(raw_ema_logits, dim=-1)
    student_decoded_hidden = (
        ema_model.exact_inverse_selected_decode_tape_states(
            ema_prefix_encoded,
            base.full_states,
            positions,
            base.anchor_indices,
        )
    )
    student_logits = ema_model.token_logits(student_decoded_hidden).float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    canonical_probs = canonical_log_probs.exp()
    canonical_raw_kl = (
        canonical_probs * (canonical_log_probs - raw_ema_log_probs)
    ).sum(dim=-1)
    canonical_student_kl = (
        canonical_probs * (canonical_log_probs - student_log_probs)
    ).sum(dim=-1)
    return ComplexSelfPredictionSelfRedecodeKLOutput(
        base=base,
        selected_tokens=selected_tokens,
        ema_prefix_encoded=ema_prefix_encoded,
        canonical_states=canonical_states,
        canonical_logits=canonical_logits,
        canonical_log_probs=canonical_log_probs,
        raw_ema_logits=raw_ema_logits,
        raw_ema_log_probs=raw_ema_log_probs,
        student_decoded_hidden=student_decoded_hidden,
        student_logits=student_logits,
        student_log_probs=student_log_probs,
        canonical_raw_kl=canonical_raw_kl,
        canonical_student_kl=canonical_student_kl,
        canonical_relative_mse=relative_mse_rows(
            base.full_states,
            canonical_states,
        ),
    )


def sparse_complex_self_prediction_self_redecode_kl_loss(
    model: K1DecoderAblationLM,
    ema_model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    anchor_stride: int,
    kl_weight: float = 1.0,
    hidden_mse_weight: float = 0.0,
    detach_preliminary_for_update: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    ComplexSelfPredictionSelfRedecodeKLOutput,
]:
    """Return ordinary CE plus self-selected EMA redecode forward KL."""
    if min(kl_weight, hidden_mse_weight) < 0:
        raise ValueError("closure weights must be non-negative")
    output = forward_sparse_complex_self_prediction_self_redecode_kl(
        model,
        ema_model,
        window,
        prefix_length=prefix_length,
        anchor_stride=anchor_stride,
        detach_preliminary_for_update=detach_preliminary_for_update,
    )
    token_nll = output.base.token_ce.mean()
    redecode_kl = output.canonical_student_kl.mean()
    hidden_mse = output.canonical_relative_mse.mean()
    closure = kl_weight * redecode_kl + hidden_mse_weight * hidden_mse
    return (
        token_nll + closure,
        token_nll,
        closure,
        output,
    )


def forward_sparse_clean_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    truncate_horizon_gradient: bool = False,
) -> SparseWindowOutput:
    """Encode real tokens and jointly decode clean K^1..K^H anchor tapes."""
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    predicted_states = operator_orbit(
        model,
        roots,
        horizons,
        detach_between_horizons=truncate_horizon_gradient,
    )
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        predicted_states,
        full_positions[:prefix_length],
        anchor_indices,
        detach_past_branch_states=truncate_horizon_gradient,
    )
    logits = model.token_logits(decoded_hidden)
    return SparseWindowOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        predicted_states=predicted_states,
        decode_states=predicted_states,
        clean_states=predicted_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
    )


def forward_sparse_compounding_noise_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    noise_generator: torch.Generator | None = None,
) -> SparseWindowOutput:
    """Decode locally noisy states while regressing propagated-noise states.

    The noise at horizon ``j`` is predicted from the clean ``K^j h_A`` state.
    Earlier noise is propagated by K into later MSE states, while the current
    horizon's noise is added only for decoding and for the next transition.
    """
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    (
        clean_states,
        predicted_states,
        decode_states,
        noise,
        log_sigma,
    ) = compounding_noise_orbit(
        model,
        roots,
        horizons,
        noise_generator=noise_generator,
    )
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        decode_states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    return SparseWindowOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        predicted_states=predicted_states,
        decode_states=decode_states,
        clean_states=clean_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        noise=noise,
        log_sigma=log_sigma,
    )


def compounding_noise_orbit(
    model: K1DecoderAblationLM,
    roots: torch.Tensor,
    horizons: int,
    *,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return clean, pre-local-noise, decoded, noise, and log-sigma tapes."""
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if not hasattr(model, "sigma_predictor"):
        raise ValueError("model must define sigma_predictor")
    bias = getattr(model.operator, "bias", None)
    if bias is not None:
        raise ValueError("compounding-noise orbit requires a bias-free K")

    clean_chain = []
    clean_current = roots
    for _ in range(horizons):
        clean_current = apply_latent_operator(model, clean_current)
        clean_chain.append(clean_current)
    clean_states = torch.stack(clean_chain, dim=2)

    log_sigma = model.sigma_predictor(clean_states)
    per_dimension_scale = (
        clean_states.detach().norm(dim=-1, keepdim=True)
        / (clean_states.shape[-1] ** 0.5)
    )
    standard_noise = torch.randn(
        clean_states.shape,
        device=clean_states.device,
        dtype=clean_states.dtype,
        generator=noise_generator,
    )
    noise = torch.exp(log_sigma) * per_dimension_scale * standard_noise

    predicted = []
    decode = []
    propagated_noise = torch.zeros_like(roots)
    for horizon in range(horizons):
        state = clean_states[:, :, horizon] + propagated_noise
        decode_state = state + noise[:, :, horizon]
        predicted.append(state)
        decode.append(decode_state)
        if horizon + 1 < horizons:
            propagated_noise = apply_latent_operator(
                model,
                propagated_noise + noise[:, :, horizon],
            )
    predicted_states = torch.stack(predicted, dim=2)
    decode_states = torch.stack(decode, dim=2)
    return (
        clean_states,
        predicted_states,
        decode_states,
        noise,
        log_sigma,
    )


def multi_trajectory_compounding_noise_orbit(
    model: K1DecoderAblationLM,
    roots: torch.Tensor,
    horizons: int,
    trajectories: int,
    *,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build many noisy tapes from one shared clean K orbit and sigma tape."""
    if horizons < 1 or trajectories < 2:
        raise ValueError("horizons and trajectories must be positive")
    if not hasattr(model, "sigma_predictor"):
        raise ValueError("model must define sigma_predictor")
    weight = model.operator.weight
    bias = getattr(model.operator, "bias", None)
    if bias is not None:
        raise ValueError("compounding-noise orbit requires a bias-free K")

    clean_chain = []
    clean_current = roots
    for _ in range(horizons):
        clean_current = F.linear(clean_current, weight)
        clean_chain.append(clean_current)
    clean_states = torch.stack(clean_chain, dim=2)
    log_sigma = model.sigma_predictor(clean_states)
    per_dimension_scale = (
        clean_states.detach().norm(dim=-1, keepdim=True)
        / (clean_states.shape[-1] ** 0.5)
    )
    standard_noise = torch.randn(
        (
            clean_states.shape[0],
            trajectories,
            *clean_states.shape[1:],
        ),
        device=clean_states.device,
        dtype=clean_states.dtype,
        generator=noise_generator,
    )
    noise = (
        torch.exp(log_sigma[:, None])
        * per_dimension_scale[:, None]
        * standard_noise
    )

    predicted = []
    decode = []
    propagated_noise = torch.zeros(
        (
            roots.shape[0],
            trajectories,
            roots.shape[1],
            roots.shape[2],
        ),
        device=roots.device,
        dtype=roots.dtype,
    )
    for horizon in range(horizons):
        state = (
            clean_states[:, None, :, horizon] + propagated_noise
        )
        decode_state = state + noise[:, :, :, horizon]
        predicted.append(state)
        decode.append(decode_state)
        if horizon + 1 < horizons:
            propagated_noise = F.linear(
                propagated_noise + noise[:, :, :, horizon],
                weight,
            )
    predicted_states = torch.stack(predicted, dim=3)
    decode_states = torch.stack(decode, dim=3)
    return (
        clean_states,
        predicted_states,
        decode_states,
        noise,
        log_sigma,
    )


def forward_sparse_multi_trajectory_compounding_noise_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    noise_generator: torch.Generator | None = None,
) -> SharedMultiTrajectorySparseOutput:
    """Encode one real prefix and share it across noisy trajectory decodes."""
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    (
        clean_states,
        predicted_states,
        decode_states,
        noise,
        log_sigma,
    ) = multi_trajectory_compounding_noise_orbit(
        model,
        roots,
        horizons,
        trajectories,
        noise_generator=noise_generator,
    )
    decoded_hidden = (
        model.exact_inverse_selected_decode_multi_trajectory_states(
            prefix_encoded,
            decode_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
    )
    logits = model.token_logits(decoded_hidden)
    return SharedMultiTrajectorySparseOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        predicted_states=predicted_states,
        decode_states=decode_states,
        clean_states=clean_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        noise=noise,
        log_sigma=log_sigma,
    )


def initial_condition_trajectory_orbit(
    model: K1DecoderAblationLM,
    roots: torch.Tensor,
    horizons: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create one branch initial condition and transport it only with K.

    Returns the clean orbit, branch orbit, transported residual tape,
    prefix-only branch priors, and the observed initial residual log scale.
    """
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if not hasattr(model, "trajectory_initializer"):
        raise ValueError("model must define trajectory_initializer")
    weight = model.operator.weight
    bias = getattr(model.operator, "bias", None)
    if bias is not None:
        raise ValueError("initial-condition orbit requires a bias-free K")

    clean_chain = []
    clean_current = roots
    for _ in range(horizons):
        clean_current = F.linear(clean_current, weight)
        clean_chain.append(clean_current)
    clean_states = torch.stack(clean_chain, dim=2)

    residuals, prior_probs, observed_log_scale = (
        model.trajectory_initializer(roots)
    )
    branch_current = clean_states[:, None, :, 0] + residuals
    branch_chain = [branch_current]
    for _ in range(1, horizons):
        branch_current = apply_latent_operator(model, branch_current)
        branch_chain.append(branch_current)
    branch_states = torch.stack(branch_chain, dim=3)
    residual_tape = branch_states - clean_states[:, None]
    return (
        clean_states,
        branch_states,
        residual_tape,
        prior_probs,
        observed_log_scale,
    )


def operator_commitment_trajectory_orbit(
    model: K1DecoderAblationLM,
    roots: torch.Tensor,
    horizons: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply one prefix/branch-conditioned ``T_i = Q_i K`` at every step.

    ``Q_i`` is compiled once from the real prefix root and reused unchanged
    across all horizons. The returned diagnostic is the maximum relative norm
    drift introduced by the represented plane rotation itself.
    """
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if not hasattr(model, "trajectory_operator"):
        raise ValueError("model must define trajectory_operator")
    weight = model.operator.weight
    bias = getattr(model.operator, "bias", None)
    if bias is not None:
        raise ValueError("operator commitment requires a bias-free K")

    clean_chain = []
    clean_current = roots
    for _ in range(horizons):
        clean_current = F.linear(clean_current, weight)
        clean_chain.append(clean_current)
    clean_states = torch.stack(clean_chain, dim=2)

    first, second, angles, prior_probs = model.trajectory_operator(roots)
    branch_current = roots[:, None].expand(
        roots.shape[0],
        first.shape[1],
        roots.shape[1],
        roots.shape[2],
    )
    branch_chain = []
    orthogonality_errors = []
    for _ in range(horizons):
        operator_state = F.linear(branch_current, weight)
        branch_current = model.trajectory_operator.rotate(
            operator_state,
            first,
            second,
            angles,
        )
        branch_chain.append(branch_current)
        with torch.no_grad():
            before = operator_state.float().square().sum(
                dim=-1
            ).sqrt().clamp_min(1e-8)
            after = branch_current.float().square().sum(dim=-1).sqrt()
            orthogonality_errors.append(
                ((after - before).abs() / before).amax()
            )
    branch_states = torch.stack(branch_chain, dim=3)
    residual_tape = branch_states - clean_states[:, None]
    initial_ratio = (
        residual_tape[:, :, :, 0].float().square().mean(
            dim=-1
        ).sqrt()
        / clean_states[:, None, :, 0].float().square().mean(
            dim=-1
        ).sqrt().clamp_min(1e-8)
    )
    observed_log_scale = initial_ratio.clamp_min(1e-8).log()
    orthogonality_error = torch.stack(orthogonality_errors).amax()
    return (
        clean_states,
        branch_states,
        residual_tape,
        prior_probs,
        observed_log_scale.to(roots.dtype),
        orthogonality_error.to(roots.dtype),
    )


def selective_innovation_trajectory_orbit(
    model: K1DecoderAblationLM,
    roots: torch.Tensor,
    horizons: int,
    *,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply a deterministic sampled innovation at every shared K step."""
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if not hasattr(model, "trajectory_innovation"):
        raise ValueError("model must define trajectory_innovation")
    weight = model.operator.weight
    bias = getattr(model.operator, "bias", None)
    if bias is not None:
        raise ValueError("innovation orbit requires a bias-free K")

    clean_chain = []
    clean_current = roots
    for _ in range(horizons):
        clean_current = F.linear(clean_current, weight)
        clean_chain.append(clean_current)
    clean_states = torch.stack(clean_chain, dim=2)

    (
        innovations,
        prior_probs,
        observed_log_scale,
        _,
        _,
        _,
    ) = model.trajectory_innovation(
        roots,
        horizons,
        noise_generator=noise_generator,
    )
    branch_current = roots[:, None].expand(
        roots.shape[0],
        innovations.shape[1],
        roots.shape[1],
        roots.shape[2],
    )
    branch_chain = []
    for horizon in range(horizons):
        branch_current = (
            F.linear(branch_current, weight)
            + innovations[:, :, :, horizon]
        )
        branch_chain.append(branch_current)
    branch_states = torch.stack(branch_chain, dim=3)
    return (
        clean_states,
        branch_states,
        innovations,
        prior_probs,
        observed_log_scale.to(roots.dtype),
    )


def _branch_distance_preservation_error(
    residual_tape: torch.Tensor,
) -> torch.Tensor:
    """Maximum relative pair-distance drift after the first horizon."""
    trajectories = residual_tape.shape[1]
    errors = []
    with torch.no_grad():
        for first in range(trajectories):
            for second in range(first + 1, trajectories):
                difference = (
                    residual_tape[:, first]
                    - residual_tape[:, second]
                )
                distance = difference.float().square().sum(
                    dim=-1
                ).sqrt()
                reference = distance[:, :, :1].clamp_min(1e-8)
                errors.append(
                    ((distance - reference).abs() / reference).amax()
                )
    if not errors:
        return residual_tape.new_zeros(())
    return torch.stack(errors).amax().to(residual_tape.dtype)


def forward_sparse_multi_trajectory_initial_condition_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> tuple[SharedMultiTrajectorySparseOutput, torch.Tensor]:
    """Decode persistent trajectories initialized once around clean K hA."""
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    (
        clean_states,
        branch_states,
        residual_tape,
        prior_probs,
        observed_log_scale,
    ) = initial_condition_trajectory_orbit(
        model,
        roots,
        horizons,
    )
    decoded_hidden = (
        model.exact_inverse_selected_decode_multi_trajectory_states(
            prefix_encoded,
            branch_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
    )
    output = SharedMultiTrajectorySparseOutput(
        logits=model.token_logits(decoded_hidden),
        decoded_hidden=decoded_hidden,
        predicted_states=branch_states,
        decode_states=branch_states,
        clean_states=clean_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        noise=residual_tape,
        log_sigma=observed_log_scale,
        prior_probs=prior_probs,
        branch_residuals=residual_tape,
    )
    return output, _branch_distance_preservation_error(residual_tape)


def forward_sparse_multi_trajectory_operator_commitment_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> tuple[SharedMultiTrajectorySparseOutput, None]:
    """Decode trajectories produced by one reusable branch operator."""
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    (
        clean_states,
        branch_states,
        residual_tape,
        prior_probs,
        observed_log_scale,
        branch_operator_orthogonality_error,
    ) = operator_commitment_trajectory_orbit(
        model,
        roots,
        horizons,
    )
    decoded_hidden = (
        model.exact_inverse_selected_decode_multi_trajectory_states(
            prefix_encoded,
            branch_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
    )
    output = SharedMultiTrajectorySparseOutput(
        logits=model.token_logits(decoded_hidden),
        decoded_hidden=decoded_hidden,
        predicted_states=branch_states,
        decode_states=branch_states,
        clean_states=clean_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        noise=residual_tape,
        log_sigma=observed_log_scale,
        prior_probs=prior_probs,
        branch_residuals=residual_tape,
        branch_operator_orthogonality_error=(
            branch_operator_orthogonality_error
        ),
    )
    return output, None


def forward_sparse_multi_trajectory_selective_innovation_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    noise_generator: torch.Generator | None = None,
) -> tuple[SharedMultiTrajectorySparseOutput, None]:
    """Decode deterministic trajectories compiled from sampled noise tapes."""
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    (
        clean_states,
        branch_states,
        innovations,
        prior_probs,
        observed_log_scale,
    ) = selective_innovation_trajectory_orbit(
        model,
        roots,
        horizons,
        noise_generator=noise_generator,
    )
    decoded_hidden = (
        model.exact_inverse_selected_decode_multi_trajectory_states(
            prefix_encoded,
            branch_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
    )
    output = SharedMultiTrajectorySparseOutput(
        logits=model.token_logits(decoded_hidden),
        decoded_hidden=decoded_hidden,
        predicted_states=branch_states,
        decode_states=branch_states,
        clean_states=clean_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        noise=innovations,
        log_sigma=observed_log_scale,
        prior_probs=prior_probs,
        branch_residuals=innovations,
        prefix_encoded=prefix_encoded,
        full_positions=full_positions,
    )
    return output, None


def sparse_clean_window_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    truncate_horizon_gradient: bool = False,
    detach_gold_targets: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SparseWindowOutput,
]:
    output = forward_sparse_clean_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        truncate_horizon_gradient=truncate_horizon_gradient,
    )
    ce = F.cross_entropy(
        output.logits.flatten(0, 2).float(),
        output.targets.flatten(),
    )
    gold_targets = (
        output.gold_states.detach()
        if detach_gold_targets
        else output.gold_states
    )
    mse_by_horizon = relative_mse_rows(
        output.predicted_states, gold_targets
    ).mean(dim=(0, 1))
    mse = mse_by_horizon.mean()
    return ce + mse, ce, mse, mse_by_horizon, output


def sparse_compounding_noise_window_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SparseWindowOutput,
]:
    output = forward_sparse_compounding_noise_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        noise_generator=noise_generator,
    )
    ce = F.cross_entropy(
        output.logits.flatten(0, 2).float(),
        output.targets.flatten(),
    )
    mse_by_horizon = relative_mse_rows(
        output.predicted_states, output.gold_states
    ).mean(dim=(0, 1))
    mse = mse_by_horizon.mean()
    return ce + mse, ce, mse, mse_by_horizon, output


def sparse_multi_trajectory_compounding_noise_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    temperature: float = 1.0,
    mse_weight: float = 0.0,
    mse_horizons: tuple[int, ...] | None = None,
    detach_mse_targets: bool = False,
    state_nce_weight: float = 0.0,
    state_nce_temperature: float = 0.2,
    state_nce_horizon: int = 1,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    """Train several noisy future tapes as a latent trajectory mixture.

    The real prefix, clean K orbit, and sigma tape are computed once per
    example. Only the noisy future tapes retain a trajectory dimension. The
    score of one trajectory at one anchor is the negative sum of its token CE
    across all horizons. Soft responsibilities are computed across
    trajectories and detached before weighting the per-trajectory losses.
    Consequently the selector itself is not a credit path, while every
    selected CE remains differentiable.

    The CE scalar has the value of the temperature-scaled Monte-Carlo
    marginal NLL and the gradient of the detached-responsibility surrogate.
    These gradients are identical:

        d[-T log mean_i exp(-C_i/T)] = sum_i softmax(-C/T)_i dC_i.

    ``mse_weight=0`` removes hidden-state regression from the optimization
    graph. For a nonzero weight, ``mse_horizons`` uses one-based horizon
    indices and defaults to every horizon. The gold states always come from
    the same current online encoder pass as the real prefix. By default their
    graph remains attached, so MSE can shape both the predicted orbit and the
    online encoder geometry.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if trajectories < 2:
        raise ValueError("trajectories must be at least two")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mse_weight < 0:
        raise ValueError("mse_weight must be non-negative")
    if state_nce_weight < 0:
        raise ValueError("state_nce_weight must be non-negative")
    if state_nce_temperature <= 0:
        raise ValueError("state_nce_temperature must be positive")
    if state_nce_horizon < 1 or state_nce_horizon > horizons:
        raise ValueError(
            f"state_nce_horizon must be in [1,{horizons}]"
        )
    selected_mse_horizons = (
        tuple(range(1, horizons + 1))
        if mse_horizons is None
        else tuple(mse_horizons)
    )
    if (
        not selected_mse_horizons
        or len(set(selected_mse_horizons)) != len(selected_mse_horizons)
        or any(
            horizon < 1 or horizon > horizons
            for horizon in selected_mse_horizons
        )
    ):
        raise ValueError(
            f"mse_horizons must be unique values in [1,{horizons}]"
        )

    batch = window.shape[0]
    sparse = forward_sparse_multi_trajectory_compounding_noise_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        trajectories=trajectories,
        noise_generator=noise_generator,
    )
    anchors = sparse.anchor_indices.numel()
    expanded_targets = sparse.targets[:, None].expand(
        batch,
        trajectories,
        anchors,
        horizons,
    )
    token_ce = F.cross_entropy(
        sparse.logits.reshape(-1, sparse.logits.shape[-1]).float(),
        expanded_targets.reshape(-1),
        reduction="none",
    ).reshape(batch, trajectories, anchors, horizons)
    trajectory_ce = token_ce.sum(dim=-1)
    responsibilities = torch.softmax(
        -trajectory_ce / temperature,
        dim=1,
    ).detach()

    posterior_ce = (
        responsibilities * trajectory_ce
    ).sum(dim=1).mean() / horizons
    marginal_ce_value = (
        -temperature
        * (
            torch.logsumexp(
                -trajectory_ce / temperature,
                dim=1,
            )
            - torch.log(
                torch.tensor(
                    float(trajectories),
                    device=trajectory_ce.device,
                    dtype=trajectory_ce.dtype,
                )
            )
        ).mean()
        / horizons
    )
    marginal_ce = (
        marginal_ce_value.detach()
        + posterior_ce
        - posterior_ce.detach()
    )

    trajectory_mse = None
    mse = marginal_ce.new_zeros(())
    loss = marginal_ce
    if mse_weight != 0.0:
        mse_targets = (
            sparse.gold_states.detach()
            if detach_mse_targets
            else sparse.gold_states
        )
        trajectory_mse_by_horizon = relative_mse_rows(
            sparse.predicted_states,
            mse_targets[:, None],
        )
        horizon_indices = torch.tensor(
            [
                horizon - 1
                for horizon in selected_mse_horizons
            ],
            device=trajectory_mse_by_horizon.device,
            dtype=torch.long,
        )
        trajectory_mse = trajectory_mse_by_horizon.index_select(
            -1,
            horizon_indices,
        ).mean(dim=-1)
        mse = (
            responsibilities * trajectory_mse
        ).sum(dim=1).mean()
        loss = marginal_ce + mse_weight * mse
    state_nce = None
    state_nce_positive_distance = None
    state_nce_negative_distance = None
    state_nce_retrieval_accuracy = None
    if state_nce_weight != 0.0:
        (
            state_nce,
            state_nce_positive_distance,
            state_nce_negative_distance,
            state_nce_retrieval_accuracy,
        ) = symmetric_state_info_nce(
            sparse.clean_states[:, :, state_nce_horizon - 1],
            sparse.gold_states[:, :, state_nce_horizon - 1],
            temperature=state_nce_temperature,
        )
        loss = loss + state_nce_weight * state_nce
    output = MultiTrajectoryWindowOutput(
        sparse=sparse,
        token_ce=token_ce,
        trajectory_ce=trajectory_ce,
        responsibilities=responsibilities,
        trajectory_mse=trajectory_mse,
        posterior_ce=posterior_ce,
        marginal_ce=marginal_ce,
        state_nce=state_nce,
        state_nce_positive_distance=state_nce_positive_distance,
        state_nce_negative_distance=state_nce_negative_distance,
        state_nce_retrieval_accuracy=state_nce_retrieval_accuracy,
    )
    return loss, marginal_ce, mse, output


def _sparse_multi_trajectory_persistent_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    forward_window,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    temperature: float = 0.05,
    mse_weight: float = 1.0,
    mse_horizons: tuple[int, ...] = (1,),
    detach_mse_targets: bool = False,
    prior_weight: float = 1.0,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    """Shared loss for persistent trajectory parameterizations."""
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if trajectories < 2:
        raise ValueError("trajectories must be at least two")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min(mse_weight, prior_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    selected_mse_horizons = tuple(mse_horizons)
    if (
        not selected_mse_horizons
        or len(set(selected_mse_horizons)) != len(selected_mse_horizons)
        or any(
            horizon < 1 or horizon > horizons
            for horizon in selected_mse_horizons
        )
    ):
        raise ValueError(
            f"mse_horizons must be unique values in [1,{horizons}]"
        )

    forward_arguments = {
        "prefix_length": prefix_length,
        "horizons": horizons,
        "anchor_stride": anchor_stride,
    }
    if noise_generator is not None:
        forward_arguments["noise_generator"] = noise_generator
    sparse, branch_distance_error = forward_window(
        model,
        window,
        **forward_arguments,
    )
    if sparse.prior_probs is None:
        raise RuntimeError("persistent trajectory output has no prior")
    if sparse.prior_probs.shape[1] != trajectories:
        raise ValueError(
            "trajectory initializer count does not match loss count"
        )

    batch = window.shape[0]
    anchors = sparse.anchor_indices.numel()
    expanded_targets = sparse.targets[:, None].expand(
        batch,
        trajectories,
        anchors,
        horizons,
    )
    token_ce = F.cross_entropy(
        sparse.logits.reshape(-1, sparse.logits.shape[-1]).float(),
        expanded_targets.reshape(-1),
        reduction="none",
    ).reshape(batch, trajectories, anchors, horizons)
    trajectory_ce = token_ce.sum(dim=-1)
    log_prior = sparse.prior_probs.float().clamp_min(1e-8).log()
    posterior_logits = log_prior - trajectory_ce / temperature
    responsibilities = torch.softmax(
        posterior_logits,
        dim=1,
    ).detach()

    posterior_ce = (
        responsibilities * trajectory_ce
    ).sum(dim=1).mean() / horizons
    marginal_ce_value = (
        -temperature
        * torch.logsumexp(posterior_logits, dim=1).mean()
        / horizons
    )
    marginal_ce = (
        marginal_ce_value.detach()
        + posterior_ce
        - posterior_ce.detach()
    )
    prior_loss = (
        responsibilities
        * (
            responsibilities.clamp_min(1e-8).log()
            - log_prior
        )
    ).sum(dim=1).mean()

    mse_targets = (
        sparse.gold_states.detach()
        if detach_mse_targets
        else sparse.gold_states
    )
    clean_mse_by_horizon = relative_mse_rows(
        sparse.clean_states,
        mse_targets,
    )
    horizon_indices = torch.tensor(
        [horizon - 1 for horizon in selected_mse_horizons],
        device=window.device,
        dtype=torch.long,
    )
    clean_mse_by_anchor = clean_mse_by_horizon.index_select(
        -1, horizon_indices
    ).mean(dim=-1)
    mse = clean_mse_by_anchor.mean()
    trajectory_mse = clean_mse_by_anchor[:, None].expand(
        batch, trajectories, anchors
    )
    loss = (
        marginal_ce
        + mse_weight * mse
        + prior_weight * prior_loss
    )
    prior_winner_accuracy = (
        sparse.prior_probs.argmax(dim=1)
        == trajectory_ce.argmin(dim=1)
    ).float().mean()
    output = MultiTrajectoryWindowOutput(
        sparse=sparse,
        token_ce=token_ce,
        trajectory_ce=trajectory_ce,
        responsibilities=responsibilities,
        trajectory_mse=trajectory_mse,
        posterior_ce=posterior_ce,
        marginal_ce=marginal_ce,
        state_nce=None,
        state_nce_positive_distance=None,
        state_nce_negative_distance=None,
        state_nce_retrieval_accuracy=None,
        prior_loss=prior_loss,
        prior_winner_accuracy=prior_winner_accuracy,
        branch_distance_error=branch_distance_error,
    )
    return loss, marginal_ce, mse, output


def sparse_multi_trajectory_initial_condition_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    temperature: float = 0.05,
    mse_weight: float = 1.0,
    mse_horizons: tuple[int, ...] = (1,),
    detach_mse_targets: bool = False,
    prior_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    """Train the one-shot residual initial-condition parameterization."""
    return _sparse_multi_trajectory_persistent_loss(
        model,
        window,
        forward_sparse_multi_trajectory_initial_condition_window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        trajectories=trajectories,
        temperature=temperature,
        mse_weight=mse_weight,
        mse_horizons=mse_horizons,
        detach_mse_targets=detach_mse_targets,
        prior_weight=prior_weight,
    )


def sparse_spectral_initial_condition_joint_nll_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    h1_closure_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    """Train deterministic initial hypotheses with a proper joint mixture.

    Every hypothesis assigns a complete path likelihood.  The prefix prior is
    mixed outside that path product at probability temperature one.  The
    detached exact posterior exists only as a diagnostic view; unlike the
    legacy persistent loss there is no straight-through path surrogate or
    separate posterior-to-prior KL.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if trajectories < 2:
        raise ValueError("trajectories must be at least two")
    if h1_closure_weight < 0:
        raise ValueError("h1_closure_weight must be non-negative")

    sparse, branch_distance_error = (
        forward_sparse_multi_trajectory_initial_condition_window(
            model,
            window,
            prefix_length=prefix_length,
            horizons=horizons,
            anchor_stride=anchor_stride,
        )
    )
    if sparse.prior_probs is None:
        raise RuntimeError("spectral hypotheses require prefix priors")
    if sparse.prior_probs.shape[1] != trajectories:
        raise ValueError(
            "trajectory initializer count does not match loss count"
        )

    batch = window.shape[0]
    anchors = sparse.anchor_indices.numel()
    expanded_targets = sparse.targets[:, None].expand(
        batch,
        trajectories,
        anchors,
        horizons,
    )
    token_ce = F.cross_entropy(
        sparse.logits.reshape(-1, sparse.logits.shape[-1]).float(),
        expanded_targets.reshape(-1),
        reduction="none",
    ).reshape(batch, trajectories, anchors, horizons)
    trajectory_ce = token_ce.sum(dim=-1)
    log_prior = sparse.prior_probs.float().clamp_min(1e-8).log()
    posterior_logits = log_prior - trajectory_ce
    joint_nll = (
        -torch.logsumexp(posterior_logits, dim=1).mean() / horizons
    )
    responsibilities = torch.softmax(
        posterior_logits,
        dim=1,
    ).detach()
    posterior_ce = (
        responsibilities * trajectory_ce
    ).sum(dim=1).mean() / horizons

    h1_closure_by_anchor = relative_mse_rows(
        sparse.clean_states[:, :, 0],
        sparse.gold_states[:, :, 0],
    )
    h1_closure = h1_closure_by_anchor.mean()
    loss = joint_nll + h1_closure_weight * h1_closure
    trajectory_mse = h1_closure_by_anchor[:, None].expand(
        batch,
        trajectories,
        anchors,
    )
    prior_winner_accuracy = (
        sparse.prior_probs.argmax(dim=1)
        == trajectory_ce.argmin(dim=1)
    ).float().mean()
    output = MultiTrajectoryWindowOutput(
        sparse=sparse,
        token_ce=token_ce,
        trajectory_ce=trajectory_ce,
        responsibilities=responsibilities,
        trajectory_mse=trajectory_mse,
        posterior_ce=posterior_ce,
        marginal_ce=joint_nll,
        state_nce=None,
        state_nce_positive_distance=None,
        state_nce_negative_distance=None,
        state_nce_retrieval_accuracy=None,
        prior_loss=None,
        prior_winner_accuracy=prior_winner_accuracy,
        branch_distance_error=branch_distance_error,
    )
    return loss, joint_nll, h1_closure, output


def forward_sparse_spectral_collapse_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> SpectralCollapseWindowOutput:
    """Decode one closed-form prefix-conditioned collapse tape per anchor."""
    if not hasattr(model, "spectral_collapse"):
        raise ValueError("model must define spectral_collapse")
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    rollout = model.spectral_collapse(roots, horizons)
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        rollout.states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return SpectralCollapseWindowOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        token_ce=token_ce,
        predicted_states=rollout.states,
        unprojected_states=rollout.unprojected_states,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        prefix_encoded=prefix_encoded,
        full_positions=full_positions,
        rollout=rollout,
    )


def sparse_spectral_collapse_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    mse_weight: float = 1.0,
    ce_horizons: tuple[int, ...] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SpectralCollapseWindowOutput,
]:
    """Train direct token CE plus detached-target clean one-step regression."""
    if mse_weight < 0:
        raise ValueError("mse_weight must be non-negative")
    selected_ce_horizons = (
        tuple(range(1, horizons + 1))
        if ce_horizons is None
        else tuple(ce_horizons)
    )
    if (
        not selected_ce_horizons
        or len(set(selected_ce_horizons)) != len(selected_ce_horizons)
        or any(
            horizon < 1 or horizon > horizons
            for horizon in selected_ce_horizons
        )
    ):
        raise ValueError(
            f"ce_horizons must be unique values in [1,{horizons}]"
        )
    output = forward_sparse_spectral_collapse_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    horizon_indices = torch.tensor(
        [horizon - 1 for horizon in selected_ce_horizons],
        device=window.device,
        dtype=torch.long,
    )
    token_nll = output.token_ce.index_select(
        -1,
        horizon_indices,
    ).mean()
    one_step_mse = relative_mse_rows(
        output.rollout.clean_first,
        output.gold_states[:, :, 0].detach(),
    ).mean()
    loss = token_nll + mse_weight * one_step_mse
    return loss, token_nll, one_step_mse, output


def forward_sparse_spectral_corrector_window(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    decode_corrected: bool = False,
) -> SpectralCorrectorWindowOutput:
    """Correct CE-only K proposals toward their own greedy re-encoding.

    The proposal and canonical-target paths are detached inside the
    corrector objective. The returned proposal logits remain fully attached
    so ordinary token CE can train the base model independently.
    """
    if not hasattr(model, "spectral_collapse"):
        raise ValueError("model must define spectral_collapse")
    if not hasattr(model, "spectral_corrector"):
        raise ValueError("model must define spectral_corrector")
    alpha = float(model.spectral_collapse.alpha.detach())
    if alpha != 0.0:
        raise ValueError("spectral corrector requires an alpha-zero proposal")
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    rollout = model.spectral_collapse(roots, horizons)
    proposal_states = rollout.states
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        proposal_states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    selected_tokens = logits.detach().argmax(dim=-1)

    with torch.no_grad():
        if (
            anchor_indices.numel() == prefix_length
            and torch.equal(
                anchor_indices,
                torch.arange(
                    prefix_length,
                    device=anchor_indices.device,
                    dtype=torch.long,
                ),
            )
        ):
            dense_actions = selected_tokens
        else:
            dense_actions = torch.zeros(
                window.shape[0],
                prefix_length,
                horizons,
                device=window.device,
                dtype=window.dtype,
            )
            dense_actions.index_copy_(
                1,
                anchor_indices,
                selected_tokens,
            )
        canonical_all = model.dense_action_tape_encode(
            window[:, :prefix_length],
            dense_actions,
            full_positions[:prefix_length],
        )
        canonical_states = canonical_all.index_select(
            1,
            anchor_indices,
        ).detach()

    correction = model.spectral_corrector(
        proposal_states,
        rollout.basis,
    )
    corrected_states = correction.states
    raw_canonical_mse = relative_mse_rows(
        proposal_states.detach(),
        canonical_states,
    )
    corrected_canonical_mse = relative_mse_rows(
        corrected_states,
        canonical_states,
    )

    corrected_decoded_hidden = None
    corrected_logits = None
    if decode_corrected:
        corrected_decoded_hidden = (
            model.exact_inverse_selected_decode_tape_states(
                prefix_encoded,
                corrected_states,
                full_positions[:prefix_length],
                anchor_indices,
            )
        )
        corrected_logits = model.token_logits(
            corrected_decoded_hidden
        )

    return SpectralCorrectorWindowOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        token_ce=token_ce,
        proposal_states=proposal_states,
        selected_tokens=selected_tokens,
        canonical_states=canonical_states,
        corrected_states=corrected_states,
        raw_canonical_mse=raw_canonical_mse,
        corrected_canonical_mse=corrected_canonical_mse,
        corrected_logits=corrected_logits,
        corrected_decoded_hidden=corrected_decoded_hidden,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        prefix_encoded=prefix_encoded,
        full_positions=full_positions,
        rollout=rollout,
        correction=correction,
    )


def sparse_spectral_corrector_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    corrector_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SpectralCorrectorWindowOutput,
]:
    """Return disjoint CE-only base and detached-MSE corrector losses."""
    if corrector_weight < 0:
        raise ValueError("corrector_weight must be non-negative")
    output = forward_sparse_spectral_corrector_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    token_nll = output.token_ce.mean()
    corrector_mse = output.corrected_canonical_mse.mean()
    loss = token_nll + corrector_weight * corrector_mse
    return loss, token_nll, corrector_mse, output


def forward_sparse_attached_spectral_one_step(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    anchor_stride: int,
    decode_corrected: bool = False,
) -> AttachedSpectralOneStepOutput:
    """Decode raw q while regressing attached ``T(q)q`` to self-canonical q."""
    if not hasattr(model, "spectral_collapse"):
        raise ValueError("model must define spectral_collapse")
    if not hasattr(model, "spectral_corrector"):
        raise ValueError("model must define spectral_corrector")
    if float(model.spectral_collapse.alpha.detach()) != 0.0:
        raise ValueError("one-step attached T training requires alpha zero")
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=1,
        anchor_stride=anchor_stride,
    )
    rollout = model.spectral_collapse(roots, 1)
    proposal_states = rollout.states
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        proposal_states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    selected_tokens = logits.detach().argmax(dim=-1)

    with torch.no_grad():
        dense_actions = torch.zeros(
            window.shape[0],
            prefix_length,
            device=window.device,
            dtype=window.dtype,
        )
        dense_actions.index_copy_(
            1,
            anchor_indices,
            selected_tokens.squeeze(-1),
        )
        canonical_all = model.dense_action_encode(
            window[:, :prefix_length],
            dense_actions,
            full_positions[:prefix_length],
        )
        canonical_states = canonical_all.index_select(
            1,
            anchor_indices,
        ).unsqueeze(2).detach()

    correction = model.spectral_corrector(
        proposal_states,
        rollout.basis,
    )
    corrected_states = correction.states
    raw_canonical_mse = relative_mse_rows(
        proposal_states.detach(),
        canonical_states,
    )
    corrected_canonical_mse = relative_mse_rows(
        corrected_states,
        canonical_states,
    )

    corrected_logits = None
    if decode_corrected:
        corrected_hidden = model.exact_inverse_selected_decode_tape_states(
            prefix_encoded,
            corrected_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
        corrected_logits = model.token_logits(corrected_hidden)

    return AttachedSpectralOneStepOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        token_ce=token_ce,
        proposal_states=proposal_states,
        selected_tokens=selected_tokens,
        canonical_states=canonical_states,
        corrected_states=corrected_states,
        raw_canonical_mse=raw_canonical_mse,
        corrected_canonical_mse=corrected_canonical_mse,
        corrected_logits=corrected_logits,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        prefix_encoded=prefix_encoded,
        full_positions=full_positions,
        rollout=rollout,
        correction=correction,
    )


def sparse_attached_spectral_one_step_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    anchor_stride: int,
    corrector_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AttachedSpectralOneStepOutput,
]:
    """Return raw one-token CE plus attached self-canonical T regression."""
    if corrector_weight < 0:
        raise ValueError("corrector_weight must be non-negative")
    output = forward_sparse_attached_spectral_one_step(
        model,
        window,
        prefix_length=prefix_length,
        anchor_stride=anchor_stride,
    )
    token_nll = output.token_ce.mean()
    corrector_mse = output.corrected_canonical_mse.mean()
    loss = token_nll + corrector_weight * corrector_mse
    return loss, token_nll, corrector_mse, output


def forward_sparse_self_canonical_redecode_kl_one_step(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    anchor_stride: int,
    detach_teacher: bool = True,
) -> SelfCanonicalRedecodeKLOutput:
    """Distill the model's greedy decode--reencode behavior through T.

    Ordinary token CE is always evaluated from the raw proposal ``q=K(h)h``.
    The target-independent greedy token from that path is re-encoded and
    immediately re-decoded under ``no_grad``. The corrected student state
    ``T(q)q`` is decoded with gradients, so its output distribution can train
    T and the attached proposal/decoder geometry without placing T in the CE
    path.
    """
    if not hasattr(model, "spectral_collapse"):
        raise ValueError("model must define spectral_collapse")
    if not hasattr(model, "spectral_corrector"):
        raise ValueError("model must define spectral_corrector")
    if float(model.spectral_collapse.alpha.detach()) != 0.0:
        raise ValueError("self-redecode KL requires an alpha-zero proposal")
    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=1,
        anchor_stride=anchor_stride,
    )
    rollout = model.spectral_collapse(roots, 1)
    proposal_states = rollout.states
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        proposal_states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    selected_tokens = logits.detach().argmax(dim=-1)

    def canonical_teacher() -> tuple[torch.Tensor, torch.Tensor]:
        dense_actions = torch.zeros(
            window.shape[0],
            prefix_length,
            device=window.device,
            dtype=window.dtype,
        )
        dense_actions.index_copy_(
            1,
            anchor_indices,
            selected_tokens.squeeze(-1),
        )
        canonical_all = model.dense_action_encode(
            window[:, :prefix_length],
            dense_actions,
            full_positions[:prefix_length],
        )
        canonical_states = canonical_all.index_select(
            1,
            anchor_indices,
        ).unsqueeze(2)
        teacher_hidden = model.exact_inverse_selected_decode_tape_states(
            prefix_encoded.detach() if detach_teacher else prefix_encoded,
            canonical_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
        teacher_logits = model.token_logits(teacher_hidden).float()
        return canonical_states, teacher_logits

    if detach_teacher:
        with torch.no_grad():
            canonical_states, teacher_logits = canonical_teacher()
    else:
        canonical_states, teacher_logits = canonical_teacher()

    correction = model.spectral_corrector(
        proposal_states,
        rollout.basis,
    )
    corrected_states = correction.states
    corrected_decoded_hidden = (
        model.exact_inverse_selected_decode_tape_states(
            prefix_encoded,
            corrected_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
    )
    corrected_logits = model.token_logits(corrected_decoded_hidden)
    raw_teacher_kl = forward_kl_rows(
        teacher_logits,
        logits,
        detach_teacher=detach_teacher,
    )
    corrected_teacher_kl = forward_kl_rows(
        teacher_logits,
        corrected_logits,
        detach_teacher=detach_teacher,
    )
    raw_canonical_mse = relative_mse_rows(
        proposal_states.detach(),
        canonical_states,
    )
    corrected_canonical_mse = relative_mse_rows(
        corrected_states,
        canonical_states,
    )
    return SelfCanonicalRedecodeKLOutput(
        logits=logits,
        decoded_hidden=decoded_hidden,
        token_ce=token_ce,
        proposal_states=proposal_states,
        selected_tokens=selected_tokens,
        canonical_states=canonical_states,
        teacher_logits=teacher_logits,
        corrected_states=corrected_states,
        corrected_decoded_hidden=corrected_decoded_hidden,
        corrected_logits=corrected_logits,
        raw_teacher_kl=raw_teacher_kl,
        corrected_teacher_kl=corrected_teacher_kl,
        raw_canonical_mse=raw_canonical_mse,
        corrected_canonical_mse=corrected_canonical_mse,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        prefix_encoded=prefix_encoded,
        full_positions=full_positions,
        rollout=rollout,
        correction=correction,
    )


def sparse_self_canonical_redecode_kl_one_step_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    anchor_stride: int,
    corrector_weight: float = 1.0,
    detach_teacher: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SelfCanonicalRedecodeKLOutput,
]:
    """Return raw one-token CE plus self-selected redecode distillation."""
    if corrector_weight < 0:
        raise ValueError("corrector_weight must be non-negative")
    output = forward_sparse_self_canonical_redecode_kl_one_step(
        model,
        window,
        prefix_length=prefix_length,
        anchor_stride=anchor_stride,
        detach_teacher=detach_teacher,
    )
    token_nll = output.token_ce.mean()
    corrector_kl = output.corrected_teacher_kl.mean()
    loss = token_nll + corrector_weight * corrector_kl
    return loss, token_nll, corrector_kl, output


def sparse_multi_trajectory_operator_commitment_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    temperature: float = 0.05,
    mse_weight: float = 1.0,
    mse_horizons: tuple[int, ...] = (1,),
    detach_mse_targets: bool = False,
    prior_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    """Train one fixed branch-conditioned operator per trajectory."""
    return _sparse_multi_trajectory_persistent_loss(
        model,
        window,
        forward_sparse_multi_trajectory_operator_commitment_window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        trajectories=trajectories,
        temperature=temperature,
        mse_weight=mse_weight,
        mse_horizons=mse_horizons,
        detach_mse_targets=detach_mse_targets,
        prior_weight=prior_weight,
    )


def sparse_multi_trajectory_selective_innovation_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    temperature: float = 0.05,
    mse_weight: float = 1.0,
    mse_horizons: tuple[int, ...] = (1,),
    detach_mse_targets: bool = False,
    prior_weight: float = 1.0,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    """Train deterministic trajectories compiled from sampled innovations."""
    return _sparse_multi_trajectory_persistent_loss(
        model,
        window,
        forward_sparse_multi_trajectory_selective_innovation_window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        trajectories=trajectories,
        temperature=temperature,
        mse_weight=mse_weight,
        mse_horizons=mse_horizons,
        detach_mse_targets=detach_mse_targets,
        prior_weight=prior_weight,
        noise_generator=noise_generator,
    )


def sparse_selective_innovation_joint_nll_behavioral_closure_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    trajectories: int,
    closure_weight: float = 1.0,
    noise_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    """Train a proper path mixture with same-tape predictive closure.

    Candidate ``i`` assigns the complete gold path probability

    ``exp(-C_i) = product_j p_i(gold_j)``.

    The primary objective is therefore the probability-temperature-one
    mixture NLL ``-log sum_i pi_i exp(-C_i)``.  Its exact posterior is used
    only as a detached weight for reporting and closure.

    For behavioral closure, the comparison path replaces the first open-loop
    state by the online gold ``hB`` and then reuses the identical already
    sampled innovations from horizons two onward.  The reanchored path is a
    stop-gradient teacher.  Closure begins at horizon two so the teacher never
    turns ``Head(D(hB))`` into a target for the already observed token B.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if horizons < 2:
        raise ValueError("behavioral closure requires at least two horizons")
    if trajectories < 2:
        raise ValueError("trajectories must be at least two")
    if closure_weight < 0:
        raise ValueError("closure_weight must be non-negative")

    sparse, branch_distance_error = (
        forward_sparse_multi_trajectory_selective_innovation_window(
            model,
            window,
            prefix_length=prefix_length,
            horizons=horizons,
            anchor_stride=anchor_stride,
            noise_generator=noise_generator,
        )
    )
    if sparse.prior_probs is None:
        raise RuntimeError("selective trajectories require candidate priors")
    if sparse.branch_residuals is None:
        raise RuntimeError("selective trajectories require innovation tapes")
    if sparse.prefix_encoded is None or sparse.full_positions is None:
        raise RuntimeError("closure requires the encoded literal prefix")
    if sparse.prior_probs.shape[1] != trajectories:
        raise ValueError(
            "trajectory innovation count does not match loss count"
        )

    batch = window.shape[0]
    anchors = sparse.anchor_indices.numel()
    expanded_targets = sparse.targets[:, None].expand(
        batch,
        trajectories,
        anchors,
        horizons,
    )
    token_ce = F.cross_entropy(
        sparse.logits.reshape(-1, sparse.logits.shape[-1]).float(),
        expanded_targets.reshape(-1),
        reduction="none",
    ).reshape(batch, trajectories, anchors, horizons)
    trajectory_ce = token_ce.sum(dim=-1)
    log_prior = sparse.prior_probs.float().clamp_min(1e-8).log()
    posterior_logits = log_prior - trajectory_ce

    # This is the actual finite-mixture negative log likelihood.  There is no
    # detached straight-through surrogate and no separate prior KL.
    joint_nll = (
        -torch.logsumexp(posterior_logits, dim=1).mean() / horizons
    )
    responsibilities = torch.softmax(
        posterior_logits,
        dim=1,
    ).detach()
    posterior_ce = (
        responsibilities * trajectory_ce
    ).sum(dim=1).mean() / horizons

    # The teacher consumes exactly hB and then the same r_i,2:H values as the
    # open path.  It is deliberately computed without an autograd graph:
    # closure updates the open trajectory, not the online target or teacher.
    with torch.no_grad():
        reanchored_current = sparse.gold_states[:, :, 0].detach()[
            :, None
        ].expand(batch, trajectories, anchors, model.width)
        reanchored_chain = [reanchored_current]
        operator_weight = model.operator.weight.detach()
        operator_bias = getattr(model.operator, "bias", None)
        if operator_bias is not None:
            operator_bias = operator_bias.detach()
        detached_innovations = sparse.branch_residuals.detach()
        for horizon in range(1, horizons):
            reanchored_current = (
                F.linear(
                    reanchored_current,
                    operator_weight,
                    operator_bias,
                )
                + detached_innovations[:, :, :, horizon]
            )
            reanchored_chain.append(reanchored_current)
        reanchored_states = torch.stack(reanchored_chain, dim=3)
        reanchored_hidden = (
            model.exact_inverse_selected_decode_multi_trajectory_states(
                sparse.prefix_encoded.detach(),
                reanchored_states,
                sparse.full_positions[:prefix_length],
                sparse.anchor_indices,
            )
        )
        teacher_logits = model.token_logits(
            reanchored_hidden[:, :, :, 1:]
        ).float()
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()

    student_log_probs = F.log_softmax(
        sparse.logits[:, :, :, 1:].float(),
        dim=-1,
    )
    closure_by_path_horizon = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    behavioral_closure_by_horizon = (
        responsibilities[..., None] * closure_by_path_horizon
    ).sum(dim=1).mean(dim=(0, 1))
    behavioral_closure_kl = behavioral_closure_by_horizon.mean()
    loss = joint_nll + closure_weight * behavioral_closure_kl

    prior_winner_accuracy = (
        sparse.prior_probs.argmax(dim=1)
        == trajectory_ce.argmin(dim=1)
    ).float().mean()
    output = MultiTrajectoryWindowOutput(
        sparse=sparse,
        token_ce=token_ce,
        trajectory_ce=trajectory_ce,
        responsibilities=responsibilities,
        trajectory_mse=None,
        posterior_ce=posterior_ce,
        marginal_ce=joint_nll,
        state_nce=None,
        state_nce_positive_distance=None,
        state_nce_negative_distance=None,
        state_nce_retrieval_accuracy=None,
        prior_loss=None,
        prior_winner_accuracy=prior_winner_accuracy,
        branch_distance_error=branch_distance_error,
        behavioral_closure_kl=behavioral_closure_kl,
        behavioral_closure_by_horizon=(
            behavioral_closure_by_horizon
        ),
        reanchored_states=reanchored_states,
    )
    return loss, joint_nll, behavioral_closure_kl, output


def _gather_anchor_horizons(
    values: torch.Tensor,
    anchor_indices: torch.Tensor,
    horizons: int,
) -> torch.Tensor:
    """Gather ``anchor + [0..H-1]`` rows without changing trailing axes."""
    if values.ndim < 2:
        raise ValueError("values must have batch and sequence axes")
    offsets = torch.arange(
        horizons,
        device=anchor_indices.device,
        dtype=torch.long,
    )
    positions = anchor_indices[:, None] + offsets[None, :]
    if positions.numel() and int(positions.max()) >= values.shape[1]:
        raise ValueError("anchor horizon exceeds the available sequence")
    selected = values.index_select(1, positions.reshape(-1))
    return selected.reshape(
        values.shape[0],
        anchor_indices.numel(),
        horizons,
        *values.shape[2:],
    )


def sparse_state_conditioned_latent_teacher_forcing_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    open_nll_weight: float = 0.0,
    closure_weight: float = 1.0,
    closure_teacher_model: K1DecoderAblationLM | None = None,
    materialize_diagnostics: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    StateConditionedLatentTeacherForcingOutput,
]:
    """Train ``T(h)=K(h)h+R(h)`` with latent TF and behavioral closure.

    Canonical one-step inputs ``h_A, h_B, ...`` receive ordinary token NLL.
    Independently, one self-fed rollout begins at each real ``h_A``.  From the
    second future decision onward, forward KL distills the stop-gradient
    canonical-input behavior into that open-loop rollout.

    The central recurrence never consumes a token ID, token embedding, noise,
    candidate index, prior, or global plan.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if horizons < 2:
        raise ValueError("behavioral closure requires at least two horizons")
    if open_nll_weight < 0:
        raise ValueError("open_nll_weight must be non-negative")
    if closure_weight < 0:
        raise ValueError("closure_weight must be non-negative")
    if not hasattr(model, "state_conditioned_transition"):
        raise ValueError("model must define state_conditioned_transition")

    (
        prefix_encoded,
        full_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _encoded_sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    transition = model.state_conditioned_transition

    # One canonical pre-central state exists for every literal next-token
    # decision.  A dense one-step inverse readout evaluates each decision
    # behind its actual causal prefix exactly once; overlapping H-step anchors
    # then gather the relevant decisions.
    canonical_all = full_encoded[:, :-1]
    teacher_all = transition(canonical_all)
    teacher_hidden_all = model.exact_inverse_dense_decode_states(
        canonical_all,
        teacher_all.state,
        full_positions[:-1],
    )
    teacher_logits_all = model.token_logits(teacher_hidden_all)

    canonical_inputs = _gather_anchor_horizons(
        canonical_all,
        anchor_indices,
        horizons,
    )
    teacher_predicted_states = _gather_anchor_horizons(
        teacher_all.state,
        anchor_indices,
        horizons,
    )
    teacher_rotated_states = _gather_anchor_horizons(
        teacher_all.rotated,
        anchor_indices,
        horizons,
    )
    teacher_innovations = _gather_anchor_horizons(
        teacher_all.innovation,
        anchor_indices,
        horizons,
    )
    teacher_angles = _gather_anchor_horizons(
        teacher_all.angles,
        anchor_indices,
        horizons,
    )
    teacher_logits = None
    if materialize_diagnostics:
        teacher_logits = _gather_anchor_horizons(
            teacher_logits_all,
            anchor_indices,
            horizons,
        )

    closure_teacher_logits_all = teacher_logits_all
    closure_teacher_logits = teacher_logits
    if closure_teacher_model is not None:
        if not hasattr(
            closure_teacher_model,
            "state_conditioned_transition",
        ):
            raise ValueError(
                "closure_teacher_model must define "
                "state_conditioned_transition"
            )
        if closure_teacher_model.width != model.width:
            raise ValueError(
                "closure teacher and student widths must match"
            )
        with torch.no_grad():
            (
                _,
                closure_full_encoded,
                _,
                _,
                _,
                closure_anchor_indices,
                closure_full_positions,
            ) = _encoded_sparse_window_components(
                closure_teacher_model,
                window,
                prefix_length=prefix_length,
                horizons=horizons,
                anchor_stride=anchor_stride,
            )
            if not torch.equal(
                closure_anchor_indices,
                anchor_indices,
            ):
                raise RuntimeError(
                    "closure teacher anchor indices do not match"
                )
            closure_canonical_all = closure_full_encoded[:, :-1]
            closure_teacher_all = (
                closure_teacher_model.state_conditioned_transition(
                    closure_canonical_all
                )
            )
            closure_teacher_hidden_all = (
                closure_teacher_model.exact_inverse_dense_decode_states(
                    closure_canonical_all,
                    closure_teacher_all.state,
                    closure_full_positions[:-1],
                )
            )
            closure_teacher_logits_all = (
                closure_teacher_model.token_logits(
                    closure_teacher_hidden_all
                )
            )
            if materialize_diagnostics:
                closure_teacher_logits = _gather_anchor_horizons(
                    closure_teacher_logits_all,
                    closure_anchor_indices,
                    horizons,
                )

    # Inference-equivalent central recurrence.  Only generated latent states
    # are fed back.  Readout happens after the complete tape exists.
    current = roots
    open_states_list = []
    open_rotated_list = []
    open_innovations_list = []
    open_angles_list = []
    for _ in range(horizons):
        step = transition(current)
        current = step.state
        open_states_list.append(step.state)
        open_rotated_list.append(step.rotated)
        open_innovations_list.append(step.innovation)
        open_angles_list.append(step.angles)
    open_states = torch.stack(open_states_list, dim=2)
    open_rotated_states = torch.stack(open_rotated_list, dim=2)
    open_innovations = torch.stack(open_innovations_list, dim=2)
    open_angles = torch.stack(open_angles_list, dim=2)
    open_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        open_states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    if materialize_diagnostics:
        if teacher_logits is None or closure_teacher_logits is None:
            raise RuntimeError("diagnostic logits were not materialized")
        open_logits = model.token_logits(open_hidden)
        teacher_token_ce = F.cross_entropy(
            teacher_logits.reshape(-1, teacher_logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        open_token_ce = F.cross_entropy(
            open_logits.reshape(-1, open_logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        open_log_probs = F.log_softmax(
            open_logits[:, :, 1:].float(),
            dim=-1,
        )
        closure_teacher_log_probs = F.log_softmax(
            closure_teacher_logits[:, :, 1:].detach().float(),
            dim=-1,
        )
        closure_teacher_probs = closure_teacher_log_probs.exp()
        closure_by_decision = (
            closure_teacher_probs
            * (closure_teacher_log_probs - open_log_probs)
        ).sum(dim=-1)
        online_teacher_log_probs = F.log_softmax(
            teacher_logits[:, :, 1:].detach().float(),
            dim=-1,
        )
        online_teacher_probs = online_teacher_log_probs.exp()
        online_closure_by_decision = (
            online_teacher_probs
            * (online_teacher_log_probs - open_log_probs)
        ).sum(dim=-1)
        online_behavioral_closure_by_horizon = (
            online_closure_by_decision.mean(dim=(0, 1))
        )
        online_behavioral_closure_kl = (
            online_behavioral_closure_by_horizon.mean()
        )
    else:
        # The canonical branch contains only 255 distinct next-token
        # decisions. The overlapping [anchor,horizon] representation repeats
        # those rows up to four times, so compute CE once and gather only the
        # scalar losses. This preserves the exact weighting of the registered
        # objective without materializing repeated vocabulary distributions.
        teacher_log_probs_all = F.log_softmax(
            teacher_logits_all.float(),
            dim=-1,
        )
        teacher_ce_all = -teacher_log_probs_all.gather(
            -1,
            window[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        teacher_token_ce = _gather_anchor_horizons(
            teacher_ce_all,
            anchor_indices,
            horizons,
        )

        # H1 Open is exactly the canonical H1 computation and is excluded
        # from both Open NLL and closure. Decode it for recurrent causality,
        # but do not pay for an otherwise unused vocabulary projection.
        open_logits = None
        open_supervised_logits = model.token_logits(open_hidden[:, :, 1:])
        open_log_probs = F.log_softmax(
            open_supervised_logits.float(),
            dim=-1,
        )
        open_supervised_ce = -open_log_probs.gather(
            -1,
            targets[:, :, 1:].unsqueeze(-1),
        ).squeeze(-1)
        open_token_ce = torch.cat(
            (teacher_token_ce[:, :, :1], open_supervised_ce),
            dim=2,
        )

        # Normalize each distinct canonical distribution once. Slicing it at
        # anchor+offset gives the same EMA target as gathering logits first
        # and then normalizing every repeated row.
        if closure_teacher_logits_all is teacher_logits_all:
            closure_teacher_log_probs_all = (
                teacher_log_probs_all.detach()
            )
        else:
            closure_teacher_log_probs_all = F.log_softmax(
                closure_teacher_logits_all.detach().float(),
                dim=-1,
            )
        closure_rows = []
        for open_horizon in range(horizons - 1):
            canonical_positions = (
                anchor_indices + open_horizon + 1
            )
            closure_teacher_log_probs = (
                closure_teacher_log_probs_all.index_select(
                    1,
                    canonical_positions,
                )
            )
            closure_teacher_probs = closure_teacher_log_probs.exp()
            closure_rows.append(
                (
                    closure_teacher_probs
                    * (
                        closure_teacher_log_probs
                        - open_log_probs[:, :, open_horizon]
                    )
                ).sum(dim=-1)
            )
        closure_by_decision = torch.stack(closure_rows, dim=2)
        nan_diagnostics = torch.full(
            (horizons - 1,),
            float("nan"),
            device=window.device,
        )
        online_behavioral_closure_by_horizon = nan_diagnostics
        online_behavioral_closure_kl = nan_diagnostics.mean()

    teacher_token_nll = teacher_token_ce.mean()
    # H1 is exactly the same transition and readout in the canonical and
    # self-fed graphs. Direct open-loop supervision therefore begins at H2.
    open_supervised_token_nll = open_token_ce[:, :, 1:].mean()
    behavioral_closure_by_horizon = closure_by_decision.mean(dim=(0, 1))
    behavioral_closure_kl = behavioral_closure_by_horizon.mean()
    loss = (
        teacher_token_nll
        + open_nll_weight * open_supervised_token_nll
        + closure_weight * behavioral_closure_kl
    )

    output = StateConditionedLatentTeacherForcingOutput(
        teacher_logits=teacher_logits,
        closure_teacher_logits=closure_teacher_logits,
        open_logits=open_logits,
        teacher_token_ce=teacher_token_ce,
        open_token_ce=open_token_ce,
        open_supervised_token_nll=open_supervised_token_nll,
        canonical_inputs=canonical_inputs,
        teacher_predicted_states=teacher_predicted_states,
        open_states=open_states,
        teacher_rotated_states=teacher_rotated_states,
        open_rotated_states=open_rotated_states,
        teacher_innovations=teacher_innovations,
        open_innovations=open_innovations,
        teacher_angles=teacher_angles,
        open_angles=open_angles,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        behavioral_closure_kl=behavioral_closure_kl,
        behavioral_closure_by_horizon=(
            behavioral_closure_by_horizon
        ),
        online_behavioral_closure_kl=(
            online_behavioral_closure_kl
        ),
        online_behavioral_closure_by_horizon=(
            online_behavioral_closure_by_horizon
        ),
    )
    return (
        loss,
        teacher_token_nll,
        behavioral_closure_kl,
        output,
    )


def backward_state_conditioned_latent_teacher_forcing_loss(
    teacher_token_nll: torch.Tensor,
    output: StateConditionedLatentTeacherForcingOutput,
    *,
    transition_parameters,
    open_nll_weight: float,
    closure_weight: float,
    scale: float = 1.0,
    closure_transition_only: bool = False,
) -> None:
    """Backpropagate the state-conditioned objective with explicit routing.

    Direct teacher/open token NLL updates the complete graph. When requested,
    behavioral closure is traversed only as far as the supplied central
    transition parameters. This preserves gradients from the frozen
    behavioral readout to recurrent states without accumulating KL gradients
    in the encoder, decoder, embedding, or vocabulary head.
    """
    if open_nll_weight < 0:
        raise ValueError("open_nll_weight must be non-negative")
    if closure_weight < 0:
        raise ValueError("closure_weight must be non-negative")
    if scale <= 0:
        raise ValueError("scale must be positive")

    supervised_loss = (
        teacher_token_nll
        + open_nll_weight * output.open_supervised_token_nll
    )
    scaled_supervised = scale * supervised_loss
    scaled_closure = (
        scale * closure_weight * output.behavioral_closure_kl
    )

    if not closure_transition_only:
        (scaled_supervised + scaled_closure).backward()
        return

    parameters = tuple(transition_parameters)
    if not parameters:
        raise ValueError("transition_parameters must not be empty")
    if closure_weight == 0:
        scaled_supervised.backward()
        return

    scaled_supervised.backward(retain_graph=True)
    torch.autograd.backward(
        scaled_closure,
        inputs=parameters,
    )


def _token_conditioned_readout_tapes(
    conditioned_states: torch.Tensor,
    proposal_states: torch.Tensor,
) -> torch.Tensor:
    """Build leakage-free tapes for every recurrent prediction horizon.

    Tape ``j`` contains conditioned states strictly before ``j``, followed by
    the still-unconditioned proposal at ``j``. Later slots are zero padding and
    cannot affect the diagonal slot under causal attention.
    """
    if (
        conditioned_states.shape != proposal_states.shape
        or conditioned_states.ndim != 4
    ):
        raise ValueError(
            "conditioned and proposal states must share "
            "[batch,anchors,horizons,width] shape"
        )
    horizons = proposal_states.shape[2]
    tapes = []
    for horizon in range(horizons):
        past = conditioned_states[:, :, :horizon]
        current = proposal_states[:, :, horizon : horizon + 1]
        future_padding = torch.zeros_like(
            proposal_states[:, :, horizon + 1 :]
        )
        tapes.append(
            torch.cat((past, current, future_padding), dim=2)
        )
    return torch.stack(tapes, dim=1)


def _decode_token_conditioned_diagonal(
    model: K1DecoderAblationLM,
    prefix_encoded: torch.Tensor,
    readout_tapes: torch.Tensor,
    positions: torch.Tensor,
    anchor_indices: torch.Tensor,
) -> torch.Tensor:
    """Decode the final active slot from each leakage-free readout tape."""
    decoded = model.exact_inverse_selected_decode_multi_trajectory_states(
        prefix_encoded,
        readout_tapes,
        positions,
        anchor_indices,
    )
    horizons = readout_tapes.shape[1]
    diagonal_hidden = torch.stack(
        [decoded[:, horizon, :, horizon] for horizon in range(horizons)],
        dim=2,
    )
    return model.token_logits(diagonal_hidden)


def sparse_token_conditioned_sequential_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    closure_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    TokenConditionedSequentialOutput,
]:
    """Train one observed-token-conditioned recurrent latent path.

    The proposal ``z_j = K s_(j-1)`` predicts token ``j`` before that token is
    supplied to ``token_conditioned_transition``. The resulting state ``s_j``
    is visible only to later predictions. Hence gold-token conditioning
    resolves the branch assignment without leaking the current label.

    The primary loss is ordinary token CE. Behavioral closure compares
    predictions from recurrent states with predictions reanchored to online
    encoder states, starting at the second token.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if horizons < 2:
        raise ValueError("behavioral closure requires at least two horizons")
    if closure_weight < 0:
        raise ValueError("closure_weight must be non-negative")
    if not hasattr(model, "token_conditioned_transition"):
        raise ValueError("model must define token_conditioned_transition")

    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )

    target_embeddings = F.embedding(targets, model.embedding_weight)
    current = roots
    proposals = []
    conditioned = []
    innovations = []
    log_scales = []
    for horizon in range(horizons):
        proposal = model.operator(current)
        current, innovation, log_scale = (
            model.token_conditioned_transition(
                proposal,
                target_embeddings[:, :, horizon],
            )
        )
        proposals.append(proposal)
        conditioned.append(current)
        innovations.append(innovation)
        log_scales.append(log_scale)
    proposal_states = torch.stack(proposals, dim=2)
    conditioned_states = torch.stack(conditioned, dim=2)
    innovation_tape = torch.stack(innovations, dim=2)
    observed_log_scale = torch.stack(log_scales, dim=2)

    student_readout_tapes = _token_conditioned_readout_tapes(
        conditioned_states,
        proposal_states,
    )
    logits = _decode_token_conditioned_diagonal(
        model,
        prefix_encoded,
        student_readout_tapes,
        full_positions[:prefix_length],
        anchor_indices,
    )
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    token_nll = token_ce.mean()

    # The teacher predicts the same next tokens from canonical online states.
    # Its first proposal is K hA; later proposals are K hB, K hC, ...
    # Neither teacher states nor teacher model calls retain an autograd graph.
    with torch.no_grad():
        canonical_previous = torch.cat(
            (roots.detach().unsqueeze(2), gold_states.detach()[:, :, :-1]),
            dim=2,
        )
        operator_weight = model.operator.weight.detach()
        operator_bias = getattr(model.operator, "bias", None)
        if operator_bias is not None:
            operator_bias = operator_bias.detach()
        canonical_proposals = F.linear(
            canonical_previous,
            operator_weight,
            operator_bias,
        )
        canonical_readout_tapes = _token_conditioned_readout_tapes(
            gold_states.detach(),
            canonical_proposals,
        )
        teacher_logits = _decode_token_conditioned_diagonal(
            model,
            prefix_encoded.detach(),
            canonical_readout_tapes,
            full_positions[:prefix_length],
            anchor_indices,
        ).float()
        teacher_log_probs = F.log_softmax(
            teacher_logits[:, :, 1:],
            dim=-1,
        )
        teacher_probs = teacher_log_probs.exp()

    student_log_probs = F.log_softmax(
        logits[:, :, 1:].float(),
        dim=-1,
    )
    closure_by_decision_horizon = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    behavioral_closure_by_horizon = (
        closure_by_decision_horizon.mean(dim=(0, 1))
    )
    behavioral_closure_kl = behavioral_closure_by_horizon.mean()
    loss = token_nll + closure_weight * behavioral_closure_kl

    output = TokenConditionedSequentialOutput(
        logits=logits,
        teacher_logits=teacher_logits,
        token_ce=token_ce,
        proposal_states=proposal_states,
        conditioned_states=conditioned_states,
        innovations=innovation_tape,
        log_scale=observed_log_scale,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        student_readout_tapes=student_readout_tapes,
        canonical_readout_tapes=canonical_readout_tapes,
        behavioral_closure_kl=behavioral_closure_kl,
        behavioral_closure_by_horizon=(
            behavioral_closure_by_horizon
        ),
    )
    return loss, token_nll, behavioral_closure_kl, output


def sparse_prefix_orthogonal_affine_scan_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    closure_weight: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PrefixOrthogonalScanWindowOutput,
]:
    """Train one prefix-only future tape evaluated by associative scan."""
    from rotlm.models.prefix_orthogonal_scan import (
        apply_pairwise_affine_scan,
    )

    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if horizons < 2:
        raise ValueError("behavioral closure requires at least two horizons")
    if closure_weight < 0:
        raise ValueError("closure_weight must be non-negative")
    if not hasattr(model, "prefix_orthogonal_scan"):
        raise ValueError("model must define prefix_orthogonal_scan")

    (
        prefix_encoded,
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    scan = model.prefix_orthogonal_scan(roots, horizons)
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        scan.states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    token_nll = token_ce.mean()

    # Reanchor once at observed hB, then reuse the already compiled suffix
    # maps exactly. The teacher is behavioral and predicts horizons 2..H.
    with torch.no_grad():
        detached_basis = scan.basis.detach()
        coordinate_hb = F.linear(
            gold_states[:, :, 0].detach(),
            detached_basis.T,
        )
        reanchored_suffix_coordinates = apply_pairwise_affine_scan(
            coordinate_hb,
            scan.local_angles[:, :, 1:].detach(),
            scan.innovations[:, :, 1:].detach(),
        )[0]
        reanchored_suffix = F.linear(
            reanchored_suffix_coordinates,
            detached_basis,
        )
        reanchored_states = torch.cat(
            (
                gold_states[:, :, :1].detach(),
                reanchored_suffix,
            ),
            dim=2,
        )
        teacher_hidden = model.exact_inverse_selected_decode_tape_states(
            prefix_encoded.detach(),
            reanchored_states,
            full_positions[:prefix_length],
            anchor_indices,
        )
        teacher_logits = model.token_logits(teacher_hidden).float()
        teacher_log_probs = F.log_softmax(
            teacher_logits[:, :, 1:],
            dim=-1,
        )
        teacher_probs = teacher_log_probs.exp()

    student_log_probs = F.log_softmax(
        logits[:, :, 1:].float(),
        dim=-1,
    )
    closure_by_decision_horizon = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    behavioral_closure_by_horizon = (
        closure_by_decision_horizon.mean(dim=(0, 1))
    )
    behavioral_closure_kl = behavioral_closure_by_horizon.mean()
    loss = token_nll + closure_weight * behavioral_closure_kl

    output = PrefixOrthogonalScanWindowOutput(
        logits=logits,
        teacher_logits=teacher_logits,
        token_ce=token_ce,
        predicted_states=scan.states,
        coordinate_states=scan.coordinate_states,
        local_angles=scan.local_angles,
        innovations=scan.innovations,
        log_scale=scan.log_scale,
        cumulative_angles=scan.cumulative_angles,
        cumulative_innovations=scan.cumulative_innovations,
        basis=scan.basis,
        gold_states=gold_states,
        targets=targets,
        anchor_indices=anchor_indices,
        reanchored_states=reanchored_states,
        behavioral_closure_kl=behavioral_closure_kl,
        behavioral_closure_by_horizon=(
            behavioral_closure_by_horizon
        ),
    )
    return loss, token_nll, behavioral_closure_kl, output


__all__ = [
    "AttachedSpectralOneStepOutput",
    "ComplexSelfPredictedKVWindowOutput",
    "ComplexSelfPredictionSelfRedecodeKLOutput",
    "ComplexStochasticRayFlowWindowOutput",
    "MultiTrajectoryWindowOutput",
    "PrefixOrthogonalScanWindowOutput",
    "SpectralCollapseWindowOutput",
    "SpectralCorrectorWindowOutput",
    "SelfCanonicalRedecodeKLOutput",
    "SharedMultiTrajectorySparseOutput",
    "SparseWindowOutput",
    "StateConditionedLatentTeacherForcingOutput",
    "TokenConditionedSequentialOutput",
    "compounding_noise_orbit",
    "complex_standard_normal",
    "multi_trajectory_compounding_noise_orbit",
    "forward_sparse_clean_window",
    "forward_sparse_complex_self_predicted_kv_window",
    "forward_sparse_complex_self_prediction_self_redecode_kl",
    "forward_sparse_compounding_noise_window",
    "forward_sparse_multi_trajectory_compounding_noise_window",
    "forward_sparse_multi_trajectory_initial_condition_window",
    "forward_sparse_multi_trajectory_operator_commitment_window",
    "forward_sparse_multi_trajectory_selective_innovation_window",
    "forward_sparse_spectral_collapse_window",
    "forward_sparse_spectral_corrector_window",
    "forward_sparse_self_canonical_redecode_kl_one_step",
    "forward_sparse_attached_spectral_one_step",
    "apply_latent_operator",
    "initial_condition_trajectory_orbit",
    "operator_commitment_trajectory_orbit",
    "selective_innovation_trajectory_orbit",
    "operator_orbit",
    "relative_mse_rows",
    "forward_kl_rows",
    "symmetric_state_info_nce",
    "sparse_anchor_indices",
    "sparse_clean_window_loss",
    "sparse_complex_self_predicted_kv_ce_only_fast_logits",
    "sparse_complex_self_predicted_kv_ce_only_fast_loss",
    "sparse_complex_stochastic_ray_flow_loss",
    "evaluate_sparse_complex_self_predicted_kv_ce_only",
    "sparse_complex_self_predicted_kv_loss",
    "sparse_complex_self_prediction_self_redecode_kl_loss",
    "sparse_compounding_noise_window_loss",
    "sparse_multi_trajectory_compounding_noise_loss",
    "sparse_multi_trajectory_initial_condition_loss",
    "sparse_multi_trajectory_operator_commitment_loss",
    "sparse_multi_trajectory_selective_innovation_loss",
    "sparse_spectral_initial_condition_joint_nll_loss",
    "sparse_spectral_collapse_loss",
    "sparse_spectral_corrector_loss",
    "sparse_self_canonical_redecode_kl_one_step_loss",
    "sparse_attached_spectral_one_step_loss",
    "sparse_selective_innovation_joint_nll_behavioral_closure_loss",
    "backward_state_conditioned_latent_teacher_forcing_loss",
    "sparse_state_conditioned_latent_teacher_forcing_loss",
    "sparse_prefix_orthogonal_affine_scan_loss",
    "sparse_token_conditioned_sequential_loss",
]
