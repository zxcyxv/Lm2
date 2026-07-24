"""Sparse-anchor clean multi-horizon training primitives for h_A orbits."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM


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


def relative_mse_rows(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    numerator = (
        prediction.float() - target.float()
    ).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


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
    weight = model.operator.weight
    bias = getattr(model.operator, "bias", None)
    if weight.ndim != 2:
        raise ValueError("operator weight must be a matrix")
    states = []
    current = roots
    for _ in range(horizons):
        operator_input = (
            current.detach()
            if detach_between_horizons and states
            else current
        )
        current = F.linear(operator_input, weight, bias)
        states.append(current)
    return torch.stack(states, dim=2)


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
        roots,
        gold_states,
        targets,
        anchor_indices,
        full_positions,
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
            propagated_noise = F.linear(
                propagated_noise + noise[:, :, horizon],
                weight,
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
    graph.  A nonzero value reuses the same posterior responsibilities, so a
    later MSE ablation does not require a different trajectory construction.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if trajectories < 2:
        raise ValueError("trajectories must be at least two")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mse_weight < 0:
        raise ValueError("mse_weight must be non-negative")

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
        trajectory_mse = relative_mse_rows(
            sparse.predicted_states,
            sparse.gold_states[:, None],
        ).mean(dim=-1)
        mse = (
            responsibilities * trajectory_mse
        ).sum(dim=1).mean()
        loss = marginal_ce + mse_weight * mse
    output = MultiTrajectoryWindowOutput(
        sparse=sparse,
        token_ce=token_ce,
        trajectory_ce=trajectory_ce,
        responsibilities=responsibilities,
        trajectory_mse=trajectory_mse,
        posterior_ce=posterior_ce,
        marginal_ce=marginal_ce,
    )
    return loss, marginal_ce, mse, output


__all__ = [
    "MultiTrajectoryWindowOutput",
    "SharedMultiTrajectorySparseOutput",
    "SparseWindowOutput",
    "compounding_noise_orbit",
    "multi_trajectory_compounding_noise_orbit",
    "forward_sparse_clean_window",
    "forward_sparse_compounding_noise_window",
    "forward_sparse_multi_trajectory_compounding_noise_window",
    "operator_orbit",
    "relative_mse_rows",
    "sparse_anchor_indices",
    "sparse_clean_window_loss",
    "sparse_compounding_noise_window_loss",
    "sparse_multi_trajectory_compounding_noise_loss",
]
