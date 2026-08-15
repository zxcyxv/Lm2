"""Sparse-window objectives and audits for conditional latent-ray flow."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.latent_flow_matching import (
    ConditionalRayFlowTimeConditioner,
    conditional_ray_flow_velocity,
    conditional_tangent_source,
    from_rotating_frame_rays,
    integrate_conditional_ray_flow,
    shortest_spherical_bridge,
    to_rotating_frame_rays,
    unit_rows,
)
from .ha_skew_window import _encoded_sparse_window_components


@dataclass
class ConditionalRayFlowLossOutput:
    """Simulation-free CFM rows and diagnostics for one endpoint batch."""

    loss: torch.Tensor
    loss_by_horizon: torch.Tensor
    relative_loss_by_horizon: torch.Tensor
    cosine_by_horizon: torch.Tensor
    target_energy_by_horizon: torch.Tensor
    source: torch.Tensor
    bridge_state: torch.Tensor
    target: torch.Tensor
    predicted_velocity: torch.Tensor
    target_velocity: torch.Tensor
    flow_time: torch.Tensor
    source_noise: torch.Tensor


@dataclass
class SparseScanConditionalRayFlowOutput:
    """Attached token scan plus detached-coordinate CFM supervision."""

    logits: torch.Tensor
    targets: torch.Tensor
    token_ce: torch.Tensor
    flow: ConditionalRayFlowLossOutput
    anchor_indices: torch.Tensor
    flow_anchor_indices: torch.Tensor
    prefix_encoded: torch.Tensor
    gold_states: torch.Tensor


@dataclass
class ConditionalRayFlowEndpointOutput:
    """Numerically integrated flow endpoint decoded through the inverse."""

    logits: torch.Tensor
    targets: torch.Tensor
    endpoint_rays: torch.Tensor
    target_rays: torch.Tensor
    source: torch.Tensor
    source_noise: torch.Tensor
    endpoint_cosine_by_horizon: torch.Tensor


def conditional_ray_flow_matching_loss(
    model: K1DecoderAblationLM,
    time_conditioner: ConditionalRayFlowTimeConditioner,
    roots: torch.Tensor,
    gold_states: torch.Tensor,
    *,
    noise_scale: float,
    flow_time: torch.Tensor | None = None,
    source_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ConditionalRayFlowLossOutput:
    """Regress the exact velocity of a prescribed spherical CFM bridge.

    Both endpoints and the sampled bridge are detached.  Consequently this
    loss trains the shared central vector field and time conditioner without
    letting a jointly trained encoder move the target coordinate system.
    """
    if not hasattr(model, "complex_self_prediction"):
        raise ValueError("model must define complex_self_prediction")
    if roots.ndim < 2 or gold_states.ndim != roots.ndim + 1:
        raise ValueError("roots and gold states have incompatible ranks")
    if gold_states.shape[:-2] != roots.shape[:-1]:
        raise ValueError("roots and gold states have incompatible leading axes")
    if gold_states.shape[-1] != roots.shape[-1]:
        raise ValueError("roots and gold states must have equal width")

    transition = model.complex_self_prediction
    horizons = gold_states.shape[-2]
    detached_roots = roots.detach().float()
    detached_gold = gold_states.detach().float()
    target = to_rotating_frame_rays(transition, detached_gold)
    source, source_noise = conditional_tangent_source(
        detached_roots,
        horizons,
        noise_scale,
        noise=source_noise,
        generator=generator,
    )
    if flow_time is None:
        flow_time = torch.rand(
            (*roots.shape[:-1], 1, 1),
            device=roots.device,
            dtype=torch.float32,
            generator=generator,
        )
    bridge = shortest_spherical_bridge(source, target, flow_time)
    bridge_state = bridge.state.detach()
    target_velocity = bridge.target_velocity.detach()
    predicted_velocity = conditional_ray_flow_velocity(
        transition,
        time_conditioner,
        bridge_state,
        flow_time.detach(),
    )
    squared_error = (
        predicted_velocity.float() - target_velocity.float()
    ).square().sum(dim=-1)
    target_energy = target_velocity.float().square().sum(dim=-1)
    loss_by_horizon = squared_error.mean(dim=tuple(range(squared_error.ndim - 1)))
    target_energy_by_horizon = target_energy.mean(
        dim=tuple(range(target_energy.ndim - 1))
    )
    relative_rows = squared_error / target_energy.clamp_min(1e-8)
    relative_loss_by_horizon = relative_rows.mean(
        dim=tuple(range(relative_rows.ndim - 1))
    )
    cosine_by_horizon = F.cosine_similarity(
        predicted_velocity.float(),
        target_velocity.float(),
        dim=-1,
        eps=1e-8,
    ).mean(dim=tuple(range(predicted_velocity.ndim - 2)))
    return ConditionalRayFlowLossOutput(
        loss=loss_by_horizon.mean(),
        loss_by_horizon=loss_by_horizon,
        relative_loss_by_horizon=relative_loss_by_horizon,
        cosine_by_horizon=cosine_by_horizon,
        target_energy_by_horizon=target_energy_by_horizon,
        source=source,
        bridge_state=bridge_state,
        target=target,
        predicted_velocity=predicted_velocity,
        target_velocity=target_velocity,
        flow_time=flow_time,
        source_noise=source_noise,
    )


def sparse_scan_conditional_ray_flow_loss(
    model: K1DecoderAblationLM,
    time_conditioner: ConditionalRayFlowTimeConditioner,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    ce_anchor_stride: int,
    flow_anchor_stride: int,
    noise_scale: float,
    flow_weight: float,
    flow_time: torch.Tensor | None = None,
    source_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SparseScanConditionalRayFlowOutput,
]:
    """Train the H16 scan CE and shared conditional vector field together."""
    if flow_weight < 0:
        raise ValueError("flow_weight must be non-negative")
    if flow_anchor_stride < ce_anchor_stride:
        raise ValueError("flow anchor stride cannot be finer than CE stride")
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
        anchor_stride=ce_anchor_stride,
    )
    read_states, _ = (
        model.complex_self_prediction.rollout_time_varying_scan_read_states(
            roots,
            horizons,
        )
    )
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        read_states,
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

    flow_selector = anchor_indices.remainder(flow_anchor_stride).eq(0)
    if not bool(flow_selector.any()):
        raise ValueError("flow anchor stride selected no prefix roots")
    flow_roots = roots[:, flow_selector, :]
    flow_gold = gold_states[:, flow_selector, :, :]
    flow = conditional_ray_flow_matching_loss(
        model,
        time_conditioner,
        flow_roots,
        flow_gold,
        noise_scale=noise_scale,
        flow_time=flow_time,
        source_noise=source_noise,
        generator=generator,
    )
    total = token_nll + float(flow_weight) * flow.loss
    output = SparseScanConditionalRayFlowOutput(
        logits=logits,
        targets=targets,
        token_ce=token_ce,
        flow=flow,
        anchor_indices=anchor_indices,
        flow_anchor_indices=anchor_indices[flow_selector],
        prefix_encoded=prefix_encoded,
        gold_states=gold_states,
    )
    return total, token_nll, flow.loss, output


def forward_sparse_conditional_ray_flow_endpoint(
    model: K1DecoderAblationLM,
    time_conditioner: ConditionalRayFlowTimeConditioner,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    noise_scale: float,
    integration_steps: int,
    source_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ConditionalRayFlowEndpointOutput:
    """Sample, integrate, and exactly decode one conditional ray tape."""
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
    source, source_noise = conditional_tangent_source(
        roots.detach(),
        horizons,
        noise_scale,
        noise=source_noise,
        generator=generator,
    )
    endpoint_rays = integrate_conditional_ray_flow(
        model.complex_self_prediction,
        time_conditioner,
        source,
        steps=integration_steps,
    )
    read_states = from_rotating_frame_rays(
        model.complex_self_prediction,
        endpoint_rays,
    )
    decoded_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        read_states,
        full_positions[:prefix_length],
        anchor_indices,
    )
    logits = model.token_logits(decoded_hidden)
    target_rays = to_rotating_frame_rays(
        model.complex_self_prediction,
        gold_states.detach(),
    )
    endpoint_cosine_by_horizon = F.cosine_similarity(
        endpoint_rays,
        target_rays,
        dim=-1,
        eps=1e-8,
    ).mean(dim=(0, 1))
    return ConditionalRayFlowEndpointOutput(
        logits=logits,
        targets=targets,
        endpoint_rays=endpoint_rays,
        target_rays=target_rays,
        source=source,
        source_noise=source_noise,
        endpoint_cosine_by_horizon=endpoint_cosine_by_horizon,
    )


@torch.inference_mode()
def evaluate_sparse_conditional_ray_flow(
    model: K1DecoderAblationLM,
    time_conditioner: ConditionalRayFlowTimeConditioner,
    windows: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    noise_scale: float,
    integration_steps: int,
    noise_seed: int,
    microbatch: int | None = None,
) -> dict[str, float]:
    """Evaluate fixed-noise endpoint decode and midpoint CFM alignment."""
    if windows.ndim != 2:
        raise ValueError("evaluation windows must have shape [batch,length]")
    if microbatch is None:
        microbatch = windows.shape[0]
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    was_training = model.training
    conditioner_was_training = time_conditioner.training
    model.eval()
    time_conditioner.eval()
    generator = torch.Generator(device=windows.device).manual_seed(noise_seed)
    nll_sum = torch.zeros(horizons, dtype=torch.float64)
    correct_sum = torch.zeros(horizons, dtype=torch.float64)
    endpoint_cosine_sum = torch.zeros(horizons, dtype=torch.float64)
    cfm_loss_sum = torch.zeros(horizons, dtype=torch.float64)
    cfm_relative_sum = torch.zeros(horizons, dtype=torch.float64)
    cfm_cosine_sum = torch.zeros(horizons, dtype=torch.float64)
    cfm_target_energy_sum = torch.zeros(horizons, dtype=torch.float64)
    labels_per_horizon = 0
    examples = 0
    try:
        for begin in range(0, windows.shape[0], microbatch):
            batch = windows[begin : begin + microbatch]
            endpoint = forward_sparse_conditional_ray_flow_endpoint(
                model,
                time_conditioner,
                batch,
                prefix_length=prefix_length,
                horizons=horizons,
                anchor_stride=anchor_stride,
                noise_scale=noise_scale,
                integration_steps=integration_steps,
                generator=generator,
            )
            token_nll = F.cross_entropy(
                endpoint.logits.reshape(-1, endpoint.logits.shape[-1]).float(),
                endpoint.targets.reshape(-1),
                reduction="none",
            ).reshape_as(endpoint.targets)
            nll_sum += token_nll.double().sum(dim=(0, 1)).cpu()
            correct_sum += endpoint.logits.argmax(dim=-1).eq(
                endpoint.targets
            ).double().sum(dim=(0, 1)).cpu()
            labels_per_horizon += (
                endpoint.targets.shape[0] * endpoint.targets.shape[1]
            )
            batch_examples = batch.shape[0]
            endpoint_cosine_sum += (
                endpoint.endpoint_cosine_by_horizon.double().cpu()
                * batch_examples
            )

            # Reuse the identical source draw at a fixed midpoint so epoch
            # comparisons do not confound vector-field progress with noise.
            # The sampled source and endpoint target pair fully determine this
            # fixed bridge, so no second encoder pass is required.
            bridge = shortest_spherical_bridge(
                endpoint.source,
                endpoint.target_rays,
                0.5,
            )
            predicted = conditional_ray_flow_velocity(
                model.complex_self_prediction,
                time_conditioner,
                bridge.state,
                0.5,
            )
            squared_error = (
                predicted - bridge.target_velocity
            ).square().sum(dim=-1)
            target_energy = bridge.target_velocity.square().sum(dim=-1)
            cfm_loss_sum += (
                squared_error.mean(dim=(0, 1)).double().cpu()
                * batch_examples
            )
            cfm_relative_sum += (
                (squared_error / target_energy.clamp_min(1e-8))
                .mean(dim=(0, 1)).double().cpu()
                * batch_examples
            )
            cfm_cosine_sum += (
                F.cosine_similarity(
                    predicted,
                    bridge.target_velocity,
                    dim=-1,
                    eps=1e-8,
                ).mean(dim=(0, 1)).double().cpu()
                * batch_examples
            )
            cfm_target_energy_sum += (
                target_energy.mean(dim=(0, 1)).double().cpu()
                * batch_examples
            )
            examples += batch_examples
    finally:
        model.train(was_training)
        time_conditioner.train(conditioner_was_training)

    horizon_nll = nll_sum / labels_per_horizon
    horizon_accuracy = correct_sum / labels_per_horizon
    endpoint_cosine = endpoint_cosine_sum / examples
    cfm_loss = cfm_loss_sum / examples
    cfm_relative = cfm_relative_sum / examples
    cfm_cosine = cfm_cosine_sum / examples
    cfm_target_energy = cfm_target_energy_sum / examples
    metrics = {
        "flow_endpoint_nll": float(horizon_nll.mean()),
        "flow_endpoint_accuracy": float(horizon_accuracy.mean()),
        "flow_endpoint_cosine": float(endpoint_cosine.mean()),
        "fixed_cfm_mse": float(cfm_loss.mean()),
        "fixed_cfm_relative_mse": float(cfm_relative.mean()),
        "fixed_cfm_cosine": float(cfm_cosine.mean()),
        "fixed_cfm_target_energy": float(cfm_target_energy.mean()),
    }
    for horizon in range(horizons):
        metrics[f"flow_h{horizon + 1}_nll"] = float(horizon_nll[horizon])
        metrics[f"flow_h{horizon + 1}_accuracy"] = float(
            horizon_accuracy[horizon]
        )
        metrics[f"flow_h{horizon + 1}_endpoint_cosine"] = float(
            endpoint_cosine[horizon]
        )
    return metrics


__all__ = [
    "ConditionalRayFlowEndpointOutput",
    "ConditionalRayFlowLossOutput",
    "SparseScanConditionalRayFlowOutput",
    "conditional_ray_flow_matching_loss",
    "evaluate_sparse_conditional_ray_flow",
    "forward_sparse_conditional_ray_flow_endpoint",
    "sparse_scan_conditional_ray_flow_loss",
]
