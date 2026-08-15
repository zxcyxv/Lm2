"""Small shared helpers used by the MSE+CE evaluation scripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from rotlm.dataset import DEFAULT_DATA_ROOT, token_memmap


SEED = 1337
DATA_ROOT = DEFAULT_DATA_ROOT


def memmap(split: str) -> np.memmap:
    return token_memmap(split, root=DATA_ROOT)


def windows(
    data: np.ndarray,
    starts: torch.Tensor,
    length: int,
    *,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    values = np.stack(
        [
            np.asarray(
                data[int(start) : int(start) + length],
                dtype=np.int64,
            )
            for start in starts
        ]
    )
    return torch.from_numpy(values).to(device, non_blocking=True)


def token_stats(tokens: torch.Tensor) -> dict[str, float]:
    values = tokens.tolist()
    bigrams = list(zip(values, values[1:]))
    return {
        "distinct_1": len(set(values)) / max(1, len(values)),
        "distinct_2": len(set(bigrams)) / max(1, len(bigrams)),
        "immediate_repeat": (
            sum(left == right for left, right in bigrams)
            / max(1, len(bigrams))
        ),
    }


GenerationPolicy = Literal["greedy", "sample"]
CentralRollout = Literal["feedback", "time-varying-scan"]


@dataclass(frozen=True)
class GoldBlockReinputMetrics:
    """Per-horizon likelihood metrics under canonical gold re-entry."""

    horizon_nll: tuple[float, ...]
    horizon_accuracy: tuple[float, ...]
    labels_per_horizon: int
    anchors_per_sample: int

    @property
    def total_labels(self) -> int:
        return self.labels_per_horizon * len(self.horizon_nll)


@dataclass(frozen=True)
class GreedyBlockReinputMetrics:
    """Gold-aligned scores under greedy generated-block re-entry."""

    horizon_target_cross_entropy: tuple[float, ...]
    horizon_target_accuracy: tuple[float, ...]
    labels_per_horizon: int
    anchors_per_sample: int

    @property
    def total_labels(self) -> int:
        return (
            self.labels_per_horizon
            * len(self.horizon_target_cross_entropy)
        )


@dataclass(frozen=True)
class GreedyChunkReinputRollouts:
    """Matched gold targets and model actions under chunkwise self re-entry.

    Tensors are returned on CPU with shapes ``[sample, anchor, horizon, ...]``.
    Every anchor begins from its gold prefix.  Inside the following horizon
    block, only greedy model actions are appended, in chunks of ``block_size``;
    no recurrent state or memory crosses a canonical re-encoding boundary.
    """

    logits: torch.Tensor
    actions: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    block_size: int

    @property
    def labels_per_horizon(self) -> int:
        return self.targets.shape[0] * self.targets.shape[1]

    @property
    def anchors_per_sample(self) -> int:
        return self.targets.shape[1]

    @property
    def total_labels(self) -> int:
        return self.targets.numel()


@dataclass
class TokenConditionedGeneration:
    """Generated tokens and same-history canonical consistency diagnostics."""

    tokens: torch.Tensor
    canonical_forward_kl: torch.Tensor
    canonical_top1_agreement: torch.Tensor
    canonical_max_logit_error: torch.Tensor


@dataclass
class SpectralBlockBoundaryRollouts:
    """Matched target-free rollouts from shared gold block boundaries."""

    boundary_ar_logits: torch.Tensor
    parallel_logits: torch.Tensor
    boundary_ar_decoded: torch.Tensor
    parallel_decoded: torch.Tensor
    boundary_ar_tokens: torch.Tensor
    parallel_tokens: torch.Tensor
    boundary_ar_proposal_states: torch.Tensor
    boundary_ar_states: torch.Tensor
    parallel_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor


@dataclass
class InputConditionedCompositionRollouts:
    """Greedy AR versus repeated state-conditioned operator composition."""

    boundary_ar_logits: torch.Tensor
    composition_logits: torch.Tensor
    boundary_ar_decoded: torch.Tensor
    composition_decoded: torch.Tensor
    boundary_ar_tokens: torch.Tensor
    composition_tokens: torch.Tensor
    boundary_ar_proposal_states: torch.Tensor
    boundary_ar_states: torch.Tensor
    composition_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor


@dataclass
class ComplexSelfPredictedKVARRollouts:
    """Self-fed complex-memory tape versus canonical greedy AR re-anchoring."""

    ar_logits: torch.Tensor
    parallel_logits: torch.Tensor
    no_write_logits: torch.Tensor
    full_gain_logits: torch.Tensor
    ar_tokens: torch.Tensor
    parallel_tokens: torch.Tensor
    no_write_tokens: torch.Tensor
    full_gain_tokens: torch.Tensor
    ar_read_states: torch.Tensor
    ar_proposal_states: torch.Tensor
    ar_states: torch.Tensor
    parallel_full_states: torch.Tensor
    parallel_read_states: torch.Tensor
    parallel_preliminary_states: torch.Tensor
    memory_energy: torch.Tensor
    innovation_energy: torch.Tensor
    beta: torch.Tensor | None
    gold_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor


@dataclass
class AttachedTBlockRollouts:
    """AR, iterated T-K, and shell-projected T-K paths from shared boundaries."""

    ar_logits: torch.Tensor
    t_logits: torch.Tensor
    projected_t_logits: torch.Tensor
    ar_tokens: torch.Tensor
    t_tokens: torch.Tensor
    projected_t_tokens: torch.Tensor
    t_states: torch.Tensor
    projected_t_states: torch.Tensor
    targets: torch.Tensor
    anchor_indices: torch.Tensor
    projected_shell_error: torch.Tensor


@torch.inference_mode()
def gold_block_reinput_metrics(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    context_length: int,
    horizons: int,
    block_size: int,
    anchor_stride: int,
    microbatch: int,
) -> GoldBlockReinputMetrics:
    """Evaluate a feedback model with gold re-entry at block boundaries.

    Each stride-selected position in the context defines an independent
    anchor.  Within an anchor, the model rolls out ``block_size`` continuous
    states without tokens.  The complete gold prefix is then canonically
    re-encoded before the next block.  Thus block size one is ordinary
    token-level teacher forcing, while larger blocks retain open-loop latent
    transitions inside each block.

    The model must provide ``encode``, ``complex_self_prediction`` with
    ``rollout_read_states``, ``exact_inverse_selected_decode_tape_states``,
    and ``token_logits``.  The returned metrics are accumulated in float64
    on CPU and contain one entry for every absolute horizon.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if model.training:
        raise ValueError("gold block re-input evaluation requires eval mode")
    if context_length < 1:
        raise ValueError("context_length must be positive")
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if block_size < 1 or horizons % block_size:
        raise ValueError("block_size must be a positive divisor of horizons")
    if anchor_stride < 1:
        raise ValueError("anchor_stride must be positive")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if window.shape[0] < 1:
        raise ValueError("window batch must be nonempty")
    if window.shape[1] < context_length + horizons:
        raise ValueError(
            "window is shorter than context_length + horizons"
        )

    anchors = tuple(range(0, context_length, anchor_stride))
    nll_sums = torch.zeros(horizons, dtype=torch.float64)
    correct_sums = torch.zeros(horizons, dtype=torch.float64)
    for anchor in anchors:
        for block_start in range(0, horizons, block_size):
            prefix_length = anchor + 1 + block_start
            for micro_window in window.split(microbatch):
                prefix = micro_window[:, :prefix_length]
                positions = torch.arange(
                    prefix_length,
                    device=prefix.device,
                )
                prefix_encoded, expanded_positions = model.encode(
                    prefix,
                    positions,
                )
                roots = prefix_encoded[:, -1:, :]
                read_states, _ = (
                    model.complex_self_prediction.rollout_read_states(
                        roots,
                        block_size,
                    )
                )
                anchor_indices = torch.tensor(
                    [prefix_length - 1],
                    device=prefix.device,
                    dtype=torch.long,
                )
                decoded = model.exact_inverse_selected_decode_tape_states(
                    prefix_encoded,
                    read_states,
                    expanded_positions,
                    anchor_indices,
                )[:, 0]
                logits = model.token_logits(decoded).float()
                targets = micro_window[
                    :,
                    prefix_length : prefix_length + block_size,
                ]
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                selected = slice(block_start, block_start + block_size)
                nll_sums[selected] += losses.double().sum(dim=0).cpu()
                correct_sums[selected] += (
                    logits.argmax(dim=-1)
                    .eq(targets)
                    .double()
                    .sum(dim=0)
                    .cpu()
                )

    labels_per_horizon = window.shape[0] * len(anchors)
    return GoldBlockReinputMetrics(
        horizon_nll=tuple((nll_sums / labels_per_horizon).tolist()),
        horizon_accuracy=tuple(
            (correct_sums / labels_per_horizon).tolist()
        ),
        labels_per_horizon=labels_per_horizon,
        anchors_per_sample=len(anchors),
    )


@torch.inference_mode()
def greedy_chunk_reinput_rollouts(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    context_length: int,
    horizons: int,
    block_size: int,
    anchor_stride: int,
    microbatch: int,
    central_rollout: CentralRollout = "feedback",
) -> GreedyChunkReinputRollouts:
    """Return matched logits, actions, and targets for greedy chunk re-entry.

    Every stride-selected gold prefix is an independent anchor. From that
    anchor the model predicts ``block_size`` open-loop latent steps, appends
    their greedy tokens, and canonically re-encodes the resulting generated
    history before the next block. Generated history is discarded when the
    next gold anchor begins; recurrent hidden and memory are never carried
    across a re-encoding boundary.

    The feedback and time-varying-scan paths differ only in the registered
    central rollout operation.  Returning the full matched tensors lets
    producer scripts derive calibration, ranking, and sequence metrics from
    exactly the same actions rather than independently reproducing inference.
    """
    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if model.training:
        raise ValueError("greedy block re-input evaluation requires eval mode")
    if context_length < 1:
        raise ValueError("context_length must be positive")
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if block_size < 1 or horizons % block_size:
        raise ValueError("block_size must be a positive divisor of horizons")
    if anchor_stride < 1:
        raise ValueError("anchor_stride must be positive")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if window.shape[0] < 1:
        raise ValueError("window batch must be nonempty")
    if window.shape[1] < context_length + horizons:
        raise ValueError(
            "window is shorter than context_length + horizons"
        )
    if central_rollout not in ("feedback", "time-varying-scan"):
        raise ValueError(f"unknown central rollout: {central_rollout}")

    anchors = tuple(range(0, context_length, anchor_stride))
    anchor_logits: list[torch.Tensor] = []
    anchor_actions: list[torch.Tensor] = []
    anchor_targets: list[torch.Tensor] = []
    for anchor in anchors:
        micro_logits: list[torch.Tensor] = []
        micro_actions: list[torch.Tensor] = []
        micro_targets: list[torch.Tensor] = []
        for micro_window in window.split(microbatch):
            rolling = micro_window[:, : anchor + 1]
            chunk_logits: list[torch.Tensor] = []
            chunk_actions: list[torch.Tensor] = []
            chunk_targets: list[torch.Tensor] = []
            for block_start in range(0, horizons, block_size):
                positions = torch.arange(
                    rolling.shape[1],
                    device=rolling.device,
                )
                prefix_encoded, expanded_positions = model.encode(
                    rolling,
                    positions,
                )
                roots = prefix_encoded[:, -1:, :]
                transition = model.complex_self_prediction
                if central_rollout == "feedback":
                    read_states, _ = transition.rollout_read_states(
                        roots,
                        block_size,
                    )
                else:
                    read_states, _ = (
                        transition.rollout_time_varying_scan_read_states(
                            roots,
                            block_size,
                        )
                    )
                anchor_indices = torch.tensor(
                    [rolling.shape[1] - 1],
                    device=rolling.device,
                    dtype=torch.long,
                )
                decoded = model.exact_inverse_selected_decode_tape_states(
                    prefix_encoded,
                    read_states,
                    expanded_positions,
                    anchor_indices,
                )[:, 0]
                logits = model.token_logits(decoded).float()
                target_start = anchor + 1 + block_start
                targets = micro_window[
                    :, target_start : target_start + block_size
                ]
                actions = logits.argmax(dim=-1)
                chunk_logits.append(logits.cpu())
                chunk_actions.append(actions.cpu())
                chunk_targets.append(targets.cpu())
                rolling = torch.cat((rolling, actions), dim=1)

            micro_logits.append(torch.cat(chunk_logits, dim=1))
            micro_actions.append(torch.cat(chunk_actions, dim=1))
            micro_targets.append(torch.cat(chunk_targets, dim=1))

        anchor_logits.append(torch.cat(micro_logits, dim=0))
        anchor_actions.append(torch.cat(micro_actions, dim=0))
        anchor_targets.append(torch.cat(micro_targets, dim=0))

    return GreedyChunkReinputRollouts(
        logits=torch.stack(anchor_logits, dim=1),
        actions=torch.stack(anchor_actions, dim=1),
        targets=torch.stack(anchor_targets, dim=1),
        anchor_indices=torch.tensor(anchors, dtype=torch.long),
        block_size=block_size,
    )


@torch.inference_mode()
def greedy_block_reinput_rollout_metrics(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    context_length: int,
    horizons: int,
    block_size: int,
    anchor_stride: int,
    microbatch: int,
    central_rollout: CentralRollout = "feedback",
) -> GreedyBlockReinputMetrics:
    """Score gold targets while re-entering greedy generated blocks.

    Because horizons after the first generated-token error are conditioned on
    a different history than the data, this is a gold-aligned rollout cross
    entropy, not a likelihood or NLL.  The values are reduced from
    :func:`greedy_chunk_reinput_rollouts` so richer diagnostics and this
    compatibility wrapper always use identical model actions.
    """
    rollout = greedy_chunk_reinput_rollouts(
        model,
        window,
        context_length=context_length,
        horizons=horizons,
        block_size=block_size,
        anchor_stride=anchor_stride,
        microbatch=microbatch,
        central_rollout=central_rollout,
    )
    losses = F.cross_entropy(
        rollout.logits.reshape(-1, rollout.logits.shape[-1]),
        rollout.targets.reshape(-1),
        reduction="none",
    ).reshape_as(rollout.targets)
    horizon_ce = losses.double().mean(dim=(0, 1))
    horizon_accuracy = (
        rollout.actions.eq(rollout.targets).double().mean(dim=(0, 1))
    )

    return GreedyBlockReinputMetrics(
        horizon_target_cross_entropy=tuple(horizon_ce.tolist()),
        horizon_target_accuracy=tuple(horizon_accuracy.tolist()),
        labels_per_horizon=rollout.labels_per_horizon,
        anchors_per_sample=rollout.anchors_per_sample,
    )


def _select_actions(
    logits: torch.Tensor,
    *,
    policy: GenerationPolicy,
    temperature: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if policy == "greedy":
        return logits.argmax(dim=-1)
    if policy != "sample":
        raise ValueError(f"unknown generation policy: {policy}")
    if temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(1)


@torch.inference_mode()
def spectral_block_boundary_rollouts(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> SpectralBlockBoundaryRollouts:
    """Compare sequential AR and one-shot spectral rollouts fairly.

    Every selected anchor is an independent block boundary backed by its
    literal gold prefix. The sequential path greedily feeds back and
    canonically re-encodes only its own selected tokens inside that block.
    The parallel path consumes no tokens after the shared boundary.
    """
    from rotlm.training.ha_skew_window import (
        forward_sparse_spectral_collapse_window,
    )

    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if horizons < 1:
        raise ValueError("horizons must be positive")
    if anchor_stride < 1:
        raise ValueError("anchor_stride must be positive")
    if not hasattr(model, "spectral_collapse"):
        raise ValueError("model must define spectral_collapse")
    if float(model.spectral_collapse.alpha.detach()) != 0.0:
        raise ValueError("matched rollout audit requires alpha zero")

    parallel = forward_sparse_spectral_collapse_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    prefix_tokens = window[:, :prefix_length]
    prefix_encoded = parallel.prefix_encoded
    positions = parallel.full_positions[:prefix_length]
    anchors = parallel.anchor_indices
    boundary_roots = prefix_encoded.index_select(1, anchors)
    selected_actions: list[torch.Tensor] = []
    ar_logits: list[torch.Tensor] = []
    ar_decoded: list[torch.Tensor] = []
    ar_proposal_states: list[torch.Tensor] = []
    ar_states: list[torch.Tensor] = []
    selected_action_states = None
    roots = boundary_roots

    for horizon in range(horizons):
        proposal = model.spectral_collapse(roots, 1).states
        ar_proposal_states.append(proposal[:, :, 0])
        decode_tape = proposal
        if selected_action_states is not None:
            decode_tape = torch.cat(
                (selected_action_states, proposal),
                dim=2,
            )
        decoded = model.exact_inverse_selected_decode_tape_states(
            prefix_encoded,
            decode_tape,
            positions,
            anchors,
        )[:, :, -1]
        logits = model.token_logits(decoded).float()
        ar_decoded.append(decoded)
        ar_logits.append(logits)
        selected_actions.append(logits.argmax(dim=-1))

        # The AR reference state is defined by the model's own greedy action,
        # not by the held-out continuation. Canonically re-encode that entire
        # self-generated tape and regenerate the input-conditioned K from its
        # final state on the next iteration.
        action_tape = torch.stack(selected_actions, dim=-1)
        dense_actions = torch.zeros(
            window.shape[0],
            prefix_length,
            horizon + 1,
            device=window.device,
            dtype=window.dtype,
        )
        dense_actions.index_copy_(1, anchors, action_tape)
        dense_action_states = model.dense_action_tape_encode(
            prefix_tokens,
            dense_actions,
            positions,
        )
        selected_action_states = dense_action_states.index_select(
            1,
            anchors,
        )
        roots = selected_action_states[:, :, -1]
        ar_states.append(roots)

    boundary_ar_logits = torch.stack(ar_logits, dim=2)
    boundary_ar_tokens = torch.stack(selected_actions, dim=2)
    parallel_logits = parallel.logits.float()
    return SpectralBlockBoundaryRollouts(
        boundary_ar_logits=boundary_ar_logits,
        parallel_logits=parallel_logits,
        boundary_ar_decoded=torch.stack(ar_decoded, dim=2),
        parallel_decoded=parallel.decoded_hidden,
        boundary_ar_tokens=boundary_ar_tokens,
        parallel_tokens=parallel_logits.argmax(dim=-1),
        boundary_ar_proposal_states=torch.stack(
            ar_proposal_states,
            dim=2,
        ),
        boundary_ar_states=torch.stack(ar_states, dim=2),
        parallel_states=parallel.predicted_states,
        targets=parallel.targets,
        anchor_indices=anchors,
    )


@torch.inference_mode()
def input_conditioned_spectral_composition_rollouts(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> InputConditionedCompositionRollouts:
    """Compare greedy AR with ``u[j+1] = K(u[j]) u[j]``.

    Unlike :func:`spectral_block_boundary_rollouts`, this path does not freeze
    the matrix generated at the initial boundary. Each continuous output is
    passed back to the K generator before the next step. Expanded from the
    initial state, step two is therefore ``K_1 K_0 h_0``.
    """
    from rotlm.training.ha_skew_window import (
        forward_sparse_spectral_collapse_window,
    )

    ar = spectral_block_boundary_rollouts(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    base = forward_sparse_spectral_collapse_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    if not torch.equal(ar.targets, base.targets):
        raise RuntimeError("AR and composition paths produced different targets")

    current = base.prefix_encoded.index_select(1, base.anchor_indices)
    states = []
    for _ in range(horizons):
        # K is regenerated from the previous continuous output. If current is
        # already K_t h_t, this applies only K_(t+1), not K_t a second time.
        current = model.spectral_collapse(current, 1).states[:, :, 0]
        states.append(current)
    composition_states = torch.stack(states, dim=2)
    decoded = model.exact_inverse_selected_decode_tape_states(
        base.prefix_encoded,
        composition_states,
        base.full_positions[:prefix_length],
        base.anchor_indices,
    )
    composition_logits = model.token_logits(decoded).float()
    return InputConditionedCompositionRollouts(
        boundary_ar_logits=ar.boundary_ar_logits,
        composition_logits=composition_logits,
        boundary_ar_decoded=ar.boundary_ar_decoded,
        composition_decoded=decoded,
        boundary_ar_tokens=ar.boundary_ar_tokens,
        composition_tokens=composition_logits.argmax(dim=-1),
        boundary_ar_proposal_states=ar.boundary_ar_proposal_states,
        boundary_ar_states=ar.boundary_ar_states,
        composition_states=composition_states,
        targets=base.targets,
        anchor_indices=base.anchor_indices,
    )


@torch.inference_mode()
def complex_self_predicted_kv_ar_rollouts(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
    latent_target_model: torch.nn.Module | None = None,
) -> ComplexSelfPredictedKVARRollouts:
    """Compare one self-fed latent tape with this model's greedy AR path.

    Both paths initialize the same matrix memory from the shared encoded
    boundary.  AR replaces only the recurrent hidden with the canonical
    re-encoding of its own selected token; it carries the same central memory
    recurrence and never supplies the selected token to a KV projection.
    """
    from rotlm.models.complex_self_prediction import ComplexMemoryState
    from rotlm.training.ha_skew_window import (
        forward_sparse_complex_self_predicted_kv_window,
    )

    if window.ndim != 2:
        raise ValueError("window must have shape [batch,length]")
    if horizons < 1 or anchor_stride < 1:
        raise ValueError("horizons and anchor_stride must be positive")
    if not hasattr(model, "complex_self_prediction"):
        raise ValueError("model must define complex_self_prediction")

    base = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
        latent_target_model=latent_target_model,
    )
    transition = model.complex_self_prediction
    roots = base.prefix_encoded.index_select(1, base.anchor_indices)
    parallel = transition.rollout(roots, horizons)
    no_write = transition.rollout(roots, horizons, write_scale=0.0)

    def decode(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = model.exact_inverse_selected_decode_tape_states(
            base.prefix_encoded,
            states,
            base.full_positions[:prefix_length],
            base.anchor_indices,
        )
        return hidden, model.token_logits(hidden).float()

    _, parallel_logits = decode(parallel.read_states)
    _, no_write_logits = decode(no_write.read_states)
    _, full_gain_logits = decode(parallel.full_states)

    prefix_tokens = window[:, :prefix_length]
    positions = base.full_positions[:prefix_length]
    selected_actions: list[torch.Tensor] = []
    ar_logits_list: list[torch.Tensor] = []
    ar_proposals: list[torch.Tensor] = []
    ar_reads: list[torch.Tensor] = []
    ar_states: list[torch.Tensor] = []
    selected_action_states = None
    state = transition.initialize(roots)

    for horizon in range(horizons):
        step = transition(state)
        ar_proposals.append(step.state.hidden)
        ar_reads.append(step.read_hidden)
        current_read = step.read_hidden.unsqueeze(2)
        decode_tape = current_read
        if selected_action_states is not None:
            decode_tape = torch.cat(
                (selected_action_states, current_read),
                dim=2,
            )
        decoded = model.exact_inverse_selected_decode_tape_states(
            base.prefix_encoded,
            decode_tape,
            positions,
            base.anchor_indices,
        )[:, :, -1]
        logits = model.token_logits(decoded).float()
        actions = logits.argmax(dim=-1)
        ar_logits_list.append(logits)
        selected_actions.append(actions)

        action_tape = torch.stack(selected_actions, dim=-1)
        dense_actions = torch.zeros(
            window.shape[0],
            prefix_length,
            horizon + 1,
            device=window.device,
            dtype=window.dtype,
        )
        dense_actions.index_copy_(1, base.anchor_indices, action_tape)
        dense_states = model.dense_action_tape_encode(
            prefix_tokens,
            dense_actions,
            positions,
        )
        selected_action_states = dense_states.index_select(
            1,
            base.anchor_indices,
        )
        canonical = selected_action_states[:, :, -1]
        ar_states.append(canonical)
        state = ComplexMemoryState(
            hidden=canonical,
            memory=step.state.memory,
            write_count=step.state.write_count,
        )

    ar_logits = torch.stack(ar_logits_list, dim=2)
    ar_tokens = torch.stack(selected_actions, dim=2)
    return ComplexSelfPredictedKVARRollouts(
        ar_logits=ar_logits,
        parallel_logits=parallel_logits,
        no_write_logits=no_write_logits,
        full_gain_logits=full_gain_logits,
        ar_tokens=ar_tokens,
        parallel_tokens=parallel_logits.argmax(dim=-1),
        no_write_tokens=no_write_logits.argmax(dim=-1),
        full_gain_tokens=full_gain_logits.argmax(dim=-1),
        ar_read_states=torch.stack(ar_reads, dim=2),
        ar_proposal_states=torch.stack(ar_proposals, dim=2),
        ar_states=torch.stack(ar_states, dim=2),
        parallel_full_states=parallel.full_states,
        parallel_read_states=parallel.read_states,
        parallel_preliminary_states=parallel.preliminary_states,
        memory_energy=parallel.memory_energy,
        innovation_energy=parallel.innovation_energy,
        beta=parallel.beta,
        gold_states=base.gold_states,
        targets=base.targets,
        anchor_indices=base.anchor_indices,
    )


@torch.inference_mode()
def generate_input_conditioned_composition_blocks(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    block: int,
    context_length: int | None = None,
) -> torch.Tensor:
    """Generate hard-token blocks from state-dependent continuous orbits.

    Inside each block, no hard token is fed back: ``u[j+1] = K(u[j])u[j]``.
    At block boundaries all selected tokens are appended and canonically
    re-encoded. Setting ``block=1`` is therefore ordinary greedy AR for this
    model, while ``block>1`` is the matched semi-parallel construction.
    """
    if prompt.ndim != 2 or prompt.shape[1] < 1:
        raise ValueError("prompt must have shape [batch,length>0]")
    if block < 1 or new_tokens < 1 or new_tokens % block:
        raise ValueError("new_tokens must be divisible by a positive block")
    if context_length is not None and context_length < 1:
        raise ValueError("context_length must be positive when provided")

    rolling = prompt
    generated = []
    for _ in range(new_tokens // block):
        context = (
            rolling
            if context_length is None
            else rolling[:, -context_length:]
        )
        positions = torch.arange(context.shape[1], device=context.device)
        encoded, expanded_positions = model.encode(context, positions)
        current = encoded[:, -1:]
        states = []
        for _ in range(block):
            current = model.spectral_collapse(current, 1).states[:, :, 0]
            states.append(current[:, 0])
        branch_states = torch.stack(states, dim=1).unsqueeze(1)
        decoded = model.exact_inverse_selected_decode_tape_states(
            encoded,
            branch_states,
            expanded_positions,
            torch.tensor(
                [context.shape[1] - 1],
                device=context.device,
                dtype=torch.long,
            ),
        )[:, 0]
        actions = model.token_logits(decoded).float().argmax(dim=-1)
        rolling = torch.cat((rolling, actions), dim=1)
        generated.append(actions)
    return torch.cat(generated, dim=1)


@torch.inference_mode()
def attached_t_block_rollouts(
    model: torch.nn.Module,
    window: torch.Tensor,
    *,
    prefix_length: int,
    horizons: int,
    anchor_stride: int,
) -> AttachedTBlockRollouts:
    """Generate matched AR, iterated T-K, and projected T-K future blocks."""
    from rotlm.training.ha_skew_window import (
        forward_sparse_spectral_collapse_window,
    )

    if not hasattr(model, "spectral_corrector"):
        raise ValueError("model must define spectral_corrector")
    ar = spectral_block_boundary_rollouts(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    base = forward_sparse_spectral_collapse_window(
        model,
        window,
        prefix_length=prefix_length,
        horizons=horizons,
        anchor_stride=anchor_stride,
    )
    if not torch.equal(ar.targets, base.targets):
        raise RuntimeError("AR and T paths produced different targets")
    roots = base.prefix_encoded.index_select(1, base.anchor_indices)

    def iterated_states(*, project: bool) -> tuple[torch.Tensor, torch.Tensor]:
        current = roots
        states = []
        shell_error = current.new_zeros(())
        for _ in range(horizons):
            proposal = model.spectral_collapse(current, 1)
            corrected = model.spectral_corrector(
                proposal.states,
                proposal.basis,
            ).states[:, :, 0]
            if project:
                corrected = model.spectral_collapse.project_to_shell(
                    corrected,
                    basis=proposal.basis,
                )
                coordinates = F.linear(corrected, proposal.basis)
                radii = coordinates.reshape(
                    *coordinates.shape[:-1],
                    model.spectral_collapse.pairs,
                    2,
                ).square().sum(dim=-1).sqrt()
                shell_error = torch.maximum(
                    shell_error,
                    (
                        radii
                        - model.spectral_collapse.shell.to(radii.dtype)
                    ).abs().max(),
                )
            states.append(corrected)
            current = corrected
        return torch.stack(states, dim=2), shell_error

    t_states, _ = iterated_states(project=False)
    projected_t_states, projected_shell_error = iterated_states(project=True)

    def decode(states: torch.Tensor) -> torch.Tensor:
        hidden = model.exact_inverse_selected_decode_tape_states(
            base.prefix_encoded,
            states,
            base.full_positions[:prefix_length],
            base.anchor_indices,
        )
        return model.token_logits(hidden).float()

    t_logits = decode(t_states)
    projected_t_logits = decode(projected_t_states)
    return AttachedTBlockRollouts(
        ar_logits=ar.boundary_ar_logits,
        t_logits=t_logits,
        projected_t_logits=projected_t_logits,
        ar_tokens=ar.boundary_ar_tokens,
        t_tokens=t_logits.argmax(dim=-1),
        projected_t_tokens=projected_t_logits.argmax(dim=-1),
        t_states=t_states,
        projected_t_states=projected_t_states,
        targets=base.targets,
        anchor_indices=base.anchor_indices,
        projected_shell_error=projected_shell_error,
    )


def _single_anchor_logits(
    model: torch.nn.Module,
    prefix_encoded: torch.Tensor,
    positions: torch.Tensor,
    future_states: torch.Tensor,
) -> torch.Tensor:
    """Decode the final state in one future tape behind the final prefix."""
    if future_states.ndim != 3:
        raise ValueError(
            "future_states must have shape [batch,horizon,width]"
        )
    anchor_indices = torch.tensor(
        [prefix_encoded.shape[1] - 1],
        device=prefix_encoded.device,
        dtype=torch.long,
    )
    decoded = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        future_states[:, None],
        positions,
        anchor_indices,
    )[:, 0, -1]
    return model.token_logits(decoded).float()


def reanchored_next_logits(
    model: torch.nn.Module,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Return canonical one-step logits after encoding the literal history."""
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    encoded, expanded_positions = model.encode(tokens, positions)
    proposal = model.operator(encoded[:, -1])
    return _single_anchor_logits(
        model,
        encoded,
        expanded_positions,
        proposal[:, None],
    )


@torch.inference_mode()
def generate_reanchored(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    policy: GenerationPolicy = "greedy",
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate by re-encoding the selected token after every decision."""
    if prompt.ndim != 2 or new_tokens < 1:
        raise ValueError("prompt must be [batch,length] and new_tokens positive")
    rolling = prompt
    generated = []
    for _ in range(new_tokens):
        logits = reanchored_next_logits(model, rolling)
        actions = _select_actions(
            logits,
            policy=policy,
            temperature=temperature,
            generator=generator,
        )
        generated.append(actions)
        rolling = torch.cat((rolling, actions[:, None]), dim=1)
    return torch.stack(generated, dim=1)


@torch.inference_mode()
def generate_token_conditioned(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    policy: GenerationPolicy = "greedy",
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> TokenConditionedGeneration:
    """Generate from one encoded prefix while feeding every selected token.

    The canonical comparison re-encodes the *same recurrently generated
    history*. It therefore measures recurrent/canonical equivalence without
    treating the held-out continuation as the only valid future.
    """
    if prompt.ndim != 2 or new_tokens < 1:
        raise ValueError("prompt must be [batch,length] and new_tokens positive")
    if not hasattr(model, "token_conditioned_transition"):
        raise ValueError("model must define token_conditioned_transition")

    prefix_positions = torch.arange(
        prompt.shape[1],
        device=prompt.device,
    )
    prefix_encoded, expanded_positions = model.encode(
        prompt,
        prefix_positions,
    )
    current = prefix_encoded[:, -1]
    rolling = prompt
    conditioned_states = []
    generated = []
    canonical_kls = []
    canonical_agreements = []
    canonical_max_errors = []

    for _ in range(new_tokens):
        proposal = model.operator(current)
        future_states = torch.stack(
            (*conditioned_states, proposal),
            dim=1,
        )
        recurrent_logits = _single_anchor_logits(
            model,
            prefix_encoded,
            expanded_positions,
            future_states,
        )
        canonical_logits = reanchored_next_logits(model, rolling)
        recurrent_log_probs = F.log_softmax(
            recurrent_logits.float(),
            dim=-1,
        )
        canonical_log_probs = F.log_softmax(
            canonical_logits.float(),
            dim=-1,
        )
        canonical_probs = canonical_log_probs.exp()
        canonical_kls.append(
            (
                canonical_probs
                * (canonical_log_probs - recurrent_log_probs)
            ).sum(dim=-1)
        )
        canonical_agreements.append(
            canonical_logits.argmax(dim=-1).eq(
                recurrent_logits.argmax(dim=-1)
            )
        )
        canonical_max_errors.append(
            (canonical_logits - recurrent_logits).abs().amax(dim=-1)
        )

        actions = _select_actions(
            recurrent_logits,
            policy=policy,
            temperature=temperature,
            generator=generator,
        )
        generated.append(actions)
        current = model.token_conditioned_transition(
            proposal,
            model.embedding_weight[actions],
        )[0]
        conditioned_states.append(current)
        rolling = torch.cat((rolling, actions[:, None]), dim=1)

    return TokenConditionedGeneration(
        tokens=torch.stack(generated, dim=1),
        canonical_forward_kl=torch.stack(canonical_kls, dim=1),
        canonical_top1_agreement=torch.stack(
            canonical_agreements,
            dim=1,
        ),
        canonical_max_logit_error=torch.stack(
            canonical_max_errors,
            dim=1,
        ),
    )


def distinct_n(tokens: list[int], n: int) -> float:
    count = len(tokens) - n + 1
    if count <= 0:
        return 0.0
    return len(
        {
            tuple(tokens[index : index + n])
            for index in range(count)
        }
    ) / count


def repeated_ngram_coverage(tokens: list[int], n: int) -> float:
    ngrams = [
        tuple(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    ]
    if not ngrams:
        return 0.0
    counts: dict[tuple[int, ...], int] = {}
    for ngram in ngrams:
        counts[ngram] = counts.get(ngram, 0) + 1
    return sum(counts[ngram] > 1 for ngram in ngrams) / len(ngrams)


def longest_identical_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    longest = 1
    current = 1
    for left, right in zip(tokens, tokens[1:]):
        if left == right:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def continuation_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    """Return target-free continuation accuracy and degeneration metrics."""
    if (
        generated.ndim != 1
        or generated.shape != reference.shape
        or generated.numel() < 1
    ):
        raise ValueError(
            "generated and reference must share a nonempty 1D shape"
        )
    values = generated.tolist()
    observed = reference.tolist()
    length = len(values)
    matching_prefix = 0
    for predicted, target in zip(values, observed):
        if predicted != target:
            break
        matching_prefix += 1
    immediate = [
        values[index] == values[index - 1]
        for index in range(1, length)
    ]
    run = longest_identical_run(values)

    def period_repeat(period: int) -> float:
        comparisons = length - period
        if comparisons <= 0:
            return 0.0
        return sum(
            values[index] == values[index - period]
            for index in range(period, length)
        ) / comparisons

    return {
        "first_token_accuracy": float(values[0] == observed[0]),
        "reference_accuracy": sum(
            predicted == target
            for predicted, target in zip(values, observed)
        )
        / length,
        "matching_prefix_tokens": float(matching_prefix),
        "distinct_1": distinct_n(values, 1),
        "distinct_2": distinct_n(values, 2),
        "distinct_4": distinct_n(values, 4),
        "immediate_repeat": (
            sum(immediate) / len(immediate) if immediate else 0.0
        ),
        "period_2_repeat": period_repeat(2),
        "period_4_repeat": period_repeat(4),
        "repeated_bigram_coverage": repeated_ngram_coverage(values, 2),
        "repeated_4gram_coverage": repeated_ngram_coverage(values, 4),
        "longest_identical_run": float(run),
        "collapsed": float(run >= 8 or distinct_n(values, 1) < 0.1),
    }


def aggregate_metrics(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric list")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }
