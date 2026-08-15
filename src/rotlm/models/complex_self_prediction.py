"""Complex unitary memory with writes generated from its own prediction."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .prefix_orthogonal_scan import (
    apply_pairwise_affine_scan,
    rotate_pairwise,
)


def _combine_complex_affine_real(
    left: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ``right o left`` using real views of complex affine pairs."""
    left_a_real, left_a_imag, left_b_real, left_b_imag = left
    right_a_real, right_a_imag, right_b_real, right_b_imag = right
    composed_a_real = (
        right_a_real * left_a_real - right_a_imag * left_a_imag
    )
    composed_a_imag = (
        right_a_real * left_a_imag + right_a_imag * left_a_real
    )
    rotated_b_real = (
        right_a_real * left_b_real - right_a_imag * left_b_imag
    )
    rotated_b_imag = (
        right_a_real * left_b_imag + right_a_imag * left_b_real
    )
    return (
        composed_a_real,
        composed_a_imag,
        right_b_real + rotated_b_real,
        right_b_imag + rotated_b_imag,
    )


def _fused_complex_memory_scan_no_autograd(
    multipliers: torch.Tensor,
    increments: torch.Tensor,
) -> torch.Tensor:
    """Run PyTorch's generated CUDA associative scan on real complex views."""
    from torch._higher_order_ops.associative_scan import associative_scan

    scan_dim = multipliers.ndim - 2
    outputs = associative_scan(
        _combine_complex_affine_real,
        (
            multipliers.real[..., None],
            multipliers.imag[..., None],
            increments.real,
            increments.imag,
        ),
        dim=scan_dim,
        combine_mode="pointwise",
    )
    return torch.complex(outputs[2], outputs[3])


class _FusedComplexMemoryAffineScan(torch.autograd.Function):
    """Generated CUDA prefix scan with an explicit reverse-scan backward."""

    @staticmethod
    def forward(
        ctx,
        multipliers: torch.Tensor,
        increments: torch.Tensor,
    ) -> torch.Tensor:
        states = _fused_complex_memory_scan_no_autograd(
            multipliers,
            increments,
        )
        ctx.save_for_backward(multipliers, states)
        return states

    @staticmethod
    def backward(
        ctx,
        state_gradients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multipliers, states = ctx.saved_tensors
        zero_multiplier = torch.zeros_like(multipliers[..., :1, :])
        backward_multipliers = torch.cat(
            (
                multipliers[..., 1:, :].conj(),
                zero_multiplier,
            ),
            dim=-2,
        )
        reverse_adjoint = _fused_complex_memory_scan_no_autograd(
            backward_multipliers.flip(-2),
            state_gradients.flip(-3),
        )
        adjoint = reverse_adjoint.flip(-3)
        zero_state = torch.zeros_like(states[..., :1, :, :])
        previous_states = torch.cat(
            (zero_state, states[..., :-1, :, :]),
            dim=-3,
        )
        multiplier_gradient = (
            adjoint * previous_states.conj()
        ).sum(dim=-1)
        return multiplier_gradient, adjoint


def _fused_pairwise_affine_scan(
    root: torch.Tensor,
    angles: torch.Tensor,
    increments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise affine scan through the fused complex scan primitive."""
    if root.shape != (*increments.shape[:-2], increments.shape[-1]):
        raise ValueError("root shape does not match fused pairwise tape")
    root_pairs = root.reshape(*root.shape[:-1], root.shape[-1] // 2, 2)
    increment_pairs = increments.reshape(
        *increments.shape[:-1], increments.shape[-1] // 2, 2
    )
    complex_root = torch.complex(root_pairs[..., 0], root_pairs[..., 1])
    complex_increments = torch.complex(
        increment_pairs[..., 0],
        increment_pairs[..., 1],
    )
    local_multipliers = torch.polar(torch.ones_like(angles), angles)
    cumulative_increments_complex = (
        _FusedComplexMemoryAffineScan.apply(
            local_multipliers,
            complex_increments[..., None],
        ).squeeze(-1)
    )
    cumulative_multipliers = torch.cumprod(local_multipliers, dim=-2)
    states_complex = (
        cumulative_multipliers * complex_root.unsqueeze(-2)
        + cumulative_increments_complex
    )
    states = torch.stack(
        (states_complex.real, states_complex.imag), dim=-1
    ).flatten(-2)
    cumulative_increments = torch.stack(
        (
            cumulative_increments_complex.real,
            cumulative_increments_complex.imag,
        ),
        dim=-1,
    ).flatten(-2)
    return states, angles.cumsum(dim=-2), cumulative_increments


def _triton_pairwise_affine_scan(
    root: torch.Tensor,
    angles: torch.Tensor,
    increments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise affine scan using one fused Triton prefix kernel."""
    from .triton_affine_scan import triton_complex_affine_scan

    if root.shape != (*increments.shape[:-2], increments.shape[-1]):
        raise ValueError("root shape does not match Triton pairwise tape")
    root_pairs = root.reshape(*root.shape[:-1], root.shape[-1] // 2, 2)
    increment_pairs = increments.reshape(
        *increments.shape[:-1], increments.shape[-1] // 2, 2
    )
    complex_root = torch.complex(root_pairs[..., 0], root_pairs[..., 1])
    complex_increments = torch.complex(
        increment_pairs[..., 0],
        increment_pairs[..., 1],
    )
    local_multipliers = torch.polar(torch.ones_like(angles), angles)
    cumulative_increments_complex = triton_complex_affine_scan(
        local_multipliers,
        complex_increments[..., None],
    ).squeeze(-1)
    cumulative_multipliers = torch.cumprod(local_multipliers, dim=-2)
    states_complex = (
        cumulative_multipliers * complex_root.unsqueeze(-2)
        + cumulative_increments_complex
    )
    states = torch.stack(
        (states_complex.real, states_complex.imag), dim=-1
    ).flatten(-2)
    cumulative_increments = torch.stack(
        (
            cumulative_increments_complex.real,
            cumulative_increments_complex.imag,
        ),
        dim=-1,
    ).flatten(-2)
    return states, angles.cumsum(dim=-2), cumulative_increments


def _rotating_frame_pairwise_affine_scan(
    root: torch.Tensor,
    angles: torch.Tensor,
    increments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve a unitary affine recurrence by an exact rotating-frame sum.

    For ``z_j = R_j z_(j-1) + delta_j`` and
    ``P_j = R_j ... R_1``, the transformed state satisfies

    ``P_j^-1 z_j = z_0 + sum_(n<=j) P_n^-1 delta_n``.

    Every prefix therefore needs one inverse rotation, one cumulative sum,
    and one forward rotation.  This is the unitary-specific linear-work form
    of the same recurrence, not a different horizon predictor.
    """
    if root.shape != (*increments.shape[:-2], increments.shape[-1]):
        raise ValueError("root shape does not match rotating-frame tape")
    if increments.shape[:-1] != angles.shape[:-1]:
        raise ValueError("rotating-frame angles and increments do not match")
    if increments.shape[-1] != 2 * angles.shape[-1]:
        raise ValueError("rotating-frame increment width is incompatible")

    root_pairs = root.reshape(*root.shape[:-1], root.shape[-1] // 2, 2)
    increment_pairs = increments.reshape(
        *increments.shape[:-1], increments.shape[-1] // 2, 2
    )
    complex_root = torch.complex(root_pairs[..., 0], root_pairs[..., 1])
    complex_increments = torch.complex(
        increment_pairs[..., 0],
        increment_pairs[..., 1],
    )
    cumulative_angles = angles.cumsum(dim=-2)
    cumulative_multipliers = torch.polar(
        torch.ones_like(cumulative_angles),
        cumulative_angles,
    )
    frame_increments = cumulative_multipliers.conj() * complex_increments
    frame_prefixes = frame_increments.cumsum(dim=-2)
    cumulative_increments_complex = cumulative_multipliers * frame_prefixes
    states_complex = cumulative_multipliers * (
        complex_root.unsqueeze(-2) + frame_prefixes
    )
    states = torch.stack(
        (states_complex.real, states_complex.imag), dim=-1
    ).flatten(-2)
    cumulative_increments = torch.stack(
        (
            cumulative_increments_complex.real,
            cumulative_increments_complex.imag,
        ),
        dim=-1,
    ).flatten(-2)
    return states, cumulative_angles, cumulative_increments


def complex_memory_affine_scan(
    multipliers: torch.Tensor,
    increments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a time-varying diagonal complex-memory recurrence.

    ``multipliers`` has shape ``[..., horizon, channel]`` and ``increments``
    has shape ``[..., horizon, channel, value]``.  The returned tensors are
    the multiplier and increment of every causal prefix of

    ``S_j = multiplier_j * S_(j-1) + increment_j``.

    Composition is associative, so Hillis--Steele doubling gives logarithmic
    dependency depth without changing the recurrence algebra.
    """
    if increments.ndim < 3 or multipliers.ndim != increments.ndim - 1:
        raise ValueError("complex scan tensors have incompatible ranks")
    if multipliers.shape != increments.shape[:-1]:
        raise ValueError("complex scan multiplier must match increment rows")
    horizons = multipliers.shape[-2]
    if horizons < 1:
        raise ValueError("complex scan horizon must be positive")

    prefix_multiplier = multipliers
    prefix_increment = increments
    offset = 1
    while offset < horizons:
        old_multiplier = prefix_multiplier
        old_increment = prefix_increment
        later_multiplier = old_multiplier[..., offset:, :]
        composed_multiplier = (
            later_multiplier * old_multiplier[..., :-offset, :]
        )
        composed_increment = (
            later_multiplier[..., None]
            * old_increment[..., :-offset, :, :]
            + old_increment[..., offset:, :, :]
        )
        prefix_multiplier = torch.cat(
            (
                old_multiplier[..., :offset, :],
                composed_multiplier,
            ),
            dim=-2,
        )
        prefix_increment = torch.cat(
            (
                old_increment[..., :offset, :, :],
                composed_increment,
            ),
            dim=-3,
        )
        offset *= 2
    return prefix_multiplier, prefix_increment


@torch.compile(fullgraph=True, mode="reduce-overhead")
def _compiled_branch_memory_measurement_real(
    memory: torch.Tensor,
    phase: torch.Tensor,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-pair implementation of one complex memory update and read."""
    memory_real, memory_imag = memory.unbind(dim=-1)
    query_real, query_imag = query.unbind(dim=-1)
    key_real, key_imag = keys.unbind(dim=-1)
    value_real, value_imag = values.unbind(dim=-1)
    cosine = torch.cos(phase)[..., None]
    sine = torch.sin(phase)[..., None]
    rotated_real = memory_real * cosine + memory_imag * sine
    rotated_imag = memory_imag * cosine - memory_real * sine
    write_real = torch.einsum(
        "...hk,...hv->...hkv", key_real, value_real
    ) + torch.einsum("...hk,...hv->...hkv", key_imag, value_imag)
    write_imag = torch.einsum(
        "...hk,...hv->...hkv", key_imag, value_real
    ) - torch.einsum("...hk,...hv->...hkv", key_real, value_imag)
    updated_real = rotated_real + write_real
    updated_imag = rotated_imag + write_imag
    frobenius = (
        updated_real.square() + updated_imag.square()
    ).sum(dim=(-3, -2, -1)).sqrt()
    read_real = (
        torch.einsum("...hk,...hkv->...hv", query_real, updated_real)
        + torch.einsum("...hk,...hkv->...hv", query_imag, updated_imag)
    ) / math.sqrt(keys.shape[-2])
    read_imag = (
        torch.einsum("...hk,...hkv->...hv", query_real, updated_imag)
        - torch.einsum("...hk,...hkv->...hv", query_imag, updated_real)
    ) / math.sqrt(keys.shape[-2])
    denominator = frobenius[..., None, None] + eps
    return (
        torch.stack((updated_real, updated_imag), dim=-1),
        torch.stack((read_real / denominator, read_imag / denominator), dim=-1),
    )


def heavy_tail(x: torch.Tensor) -> torch.Tensor:
    """``1 + x`` for ``x >= 0`` and ``1 / (1 - x)`` for ``x < 0``.

    This is the Mamba-3 parameterization of the continuous-time state decay
    rate: linear growth on one side, a heavy tail approaching zero on the
    other, so ``-heavy_tail`` stays strictly negative everywhere.
    """
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


@dataclass
class ComplexMemoryState:
    """Complete recurrent state of the lightweight central layer."""

    hidden: torch.Tensor
    memory: torch.Tensor
    write_count: int = 1


@dataclass
class ComplexSelfPredictionStep:
    """One self-predicted complex-memory transition."""

    state: ComplexMemoryState
    read_hidden: torch.Tensor
    preliminary_hidden: torch.Tensor
    rotated_hidden: torch.Tensor
    innovation_delta: torch.Tensor
    memory_energy: torch.Tensor | None
    innovation_energy: torch.Tensor | None
    beta: torch.Tensor | None
    raw_preliminary_hidden: torch.Tensor
    rotated_memory: torch.Tensor | None
    innovation_memory: torch.Tensor | None
    raw_updated_memory: torch.Tensor
    raw_full_hidden: torch.Tensor


@dataclass
class ComplexSelfPredictionRollout:
    """A target-free tape produced by repeated central transitions."""

    full_states: torch.Tensor
    read_states: torch.Tensor
    preliminary_states: torch.Tensor
    rotated_states: torch.Tensor
    innovation_deltas: torch.Tensor
    memory_energy: torch.Tensor
    innovation_energy: torch.Tensor
    beta: torch.Tensor | None
    final_state: ComplexMemoryState


@dataclass
class ComplexTimeVaryingScanRollout:
    """Exact parallel evaluation of the time-varying central recurrence.

    The Q/K/V forcing tape is compiled from the initial root and horizon,
    never from a measurement-updated successor.  Memory prefixes and latent
    prefixes are nevertheless the exact states of their respective literal
    recurrences, rather than independent horizon predictions.
    """

    full_states: torch.Tensor
    read_states: torch.Tensor
    schedule_states: torch.Tensor
    innovation_deltas: torch.Tensor
    memory_states: torch.Tensor
    cumulative_hidden_angles: torch.Tensor | None
    cumulative_hidden_increments: torch.Tensor | None
    cumulative_memory_multipliers: torch.Tensor | None
    final_state: ComplexMemoryState


@dataclass
class ComplexIterativeFeedbackScanRollout:
    """A finite Picard refinement of the compiled time-varying scan.

    Every refinement rebuilds Q/K/V from the rotated predecessors of the
    previous raw hidden tape, resets memory to zero, and solves the resulting
    affine hidden recurrence by the configured scan backend.  A fixed point
    therefore obeys the ordinary state-dependent feedback recurrence.
    """

    full_states: torch.Tensor
    read_states: torch.Tensor
    forcing_states: torch.Tensor
    innovation_deltas: torch.Tensor
    memory_states: torch.Tensor
    cumulative_hidden_angles: torch.Tensor | None
    cumulative_hidden_increments: torch.Tensor | None
    cumulative_memory_multipliers: torch.Tensor | None
    feedback_rescans: int
    final_state: ComplexMemoryState


@dataclass
class ComplexForcingVelocityTape:
    """Causal memory response to an externally supplied hidden tape.

    This is the shared vector-field primitive used by both the compiled
    root-orbit recurrence and conditional latent flow matching.  Supplying a
    tape changes only the Q/K/V forcing rows; the unitary memory prefix scan
    and decoder-width readout are identical.
    """

    velocity: torch.Tensor
    memory_states: torch.Tensor
    cumulative_memory_multipliers: torch.Tensor | None


class ComplexSelfPredictedKVTransition(nn.Module):
    """Unitary complex transport plus self-predicted rank-one writes.

    The reversible contract belongs to the surrounding encoder/decoder.  This
    central recurrence is deliberately allowed to be non-injective.  Its
    homogeneous hidden and memory transports are unit-modulus rotations, while
    a preliminary continuous future prediction produces the only KV write.

    ``full_hidden`` is the innovation-updated recurrent state and receives
    latent closure. ``read_hidden`` is the innovation-free prior readout and
    receives token loss.  A learned interpolation is retained only for the
    already-recorded tempered-readout predecessor experiment; disabling it
    removes the coefficient entirely and exposes the strict causal prior.
    """

    def __init__(
        self,
        width: int,
        *,
        heads: int = 8,
        key_dim: int = 16,
        value_dim: int = 31,
        initial_frequency_range: float = 0.05,
        initial_beta: float = 0.5,
        learned_read_beta: bool = True,
        residual_prior: bool = False,
        recurrent_hidden_mode: str = "split-query-residual",
        real_value: bool = False,
        normalize_read_by_memory_norm: bool = True,
        qkv_prenorm: bool = True,
        scan_memory_decay: bool = False,
        scan_memory_decay_init: float = 0.95,
        scan_memory_decay_max: float = 0.999,
        state_dependent_memory_decay: bool = False,
        memory_decay_floor: float = 1e-4,
        memory_decay_dt_min: float = 1e-3,
        memory_decay_dt_max: float = 1e-1,
        pre_normalize_qkv_inputs: bool = True,
        normalize_initial_recurrent_root: bool = False,
        normalize_accumulated_reads: bool = False,
        post_normalize_hidden_reads: bool = False,
        post_normalize_memory: bool = False,
        post_normalize_recurrent_state: bool = False,
        residual_step_scale: float = 0.1,
        post_norm_eps: float = 1e-6,
        shared_kv_heads: bool = False,
        fuse_branch_memory_kernel: bool = False,
        fuse_time_varying_memory_scan: bool = False,
        fuse_time_varying_hidden_scan: bool = False,
        use_triton_time_varying_scan: bool = False,
        use_rotating_frame_time_varying_scan: bool = False,
        use_fused_triton_rotating_frame_time_varying_scan: bool = False,
    ) -> None:
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("width must be positive and even")
        if heads < 1 or key_dim < 1 or value_dim < 1:
            raise ValueError("complex memory dimensions must be positive")
        if initial_frequency_range < 0:
            raise ValueError("initial frequency range must be non-negative")
        if post_norm_eps <= 0:
            raise ValueError("post_norm_eps must be positive")
        if learned_read_beta and not 0 < initial_beta < 1:
            raise ValueError("initial beta must lie strictly between zero and one")
        if recurrent_hidden_mode not in (
            "split-query-residual",
            "post-update-full-read",
            "rotated-hidden-innovation-residual",
            "unitary-block-residual",
            "unitary-branch-normalized-residual",
            "unitary-branch-normalized-no-residual",
            "unitary-raw-qkv-boundary-normalized-hidden-residual",
            "unitary-detached-input-raw-read-postnorm-residual",
            "unitary-full-bptt-raw-read-joint-postnorm-residual",
        ):
            raise ValueError(
                "unknown recurrent hidden mode: "
                f"{recurrent_hidden_mode}"
            )

        self.width = int(width)
        self.heads = int(heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        # Mamba's multi-value-attention head structure: B (key) and C (query)
        # are shared across heads and only the value X stays per head, so the
        # projection cost is ``width * key_dim`` for the whole layer instead of
        # ``width * heads * key_dim``.  Off by default -- the existing lineage
        # projects Q/K per head, which is the standard multi-head layout.
        self.shared_kv_heads = bool(shared_kv_heads)
        self.key_heads = 1 if self.shared_kv_heads else self.heads
        # The write value may be restricted to the real axis.  The memory and
        # its read stay complex either way, so only the value projection
        # narrows; ``read_width`` is unchanged.
        self.real_value = bool(real_value)
        # Dividing the read by ``||S||_F`` makes it scale invariant, which is
        # what keeps the measurement bounded while the memory itself is a
        # lossless accumulator.  With a contracting memory the norm is already
        # bounded, so the division can be dropped and the read left linear.
        self.normalize_read_by_memory_norm = bool(normalize_read_by_memory_norm)
        self.value_features = (
            heads * value_dim if self.real_value else 2 * heads * value_dim
        )
        self.read_width = 2 * heads * value_dim
        self.residual_prior = bool(residual_prior)
        self.recurrent_hidden_mode = recurrent_hidden_mode
        self.pre_normalize_qkv_inputs = bool(pre_normalize_qkv_inputs)
        self.qkv_prenorm = bool(qkv_prenorm)
        self.normalize_initial_recurrent_root = bool(
            normalize_initial_recurrent_root
        )
        self.normalize_accumulated_reads = bool(
            normalize_accumulated_reads
        )
        self.post_normalize_hidden_reads = bool(
            post_normalize_hidden_reads
        )
        self.post_normalize_memory = bool(post_normalize_memory)
        self.post_normalize_recurrent_state = bool(
            post_normalize_recurrent_state
        )
        if self.post_normalize_recurrent_state and (
            self.post_normalize_hidden_reads
            or self.post_normalize_memory
        ):
            raise ValueError(
                "final-carrier post-normalization cannot be combined with "
                "intermediate hidden or memory post-normalization"
            )
        self.post_norm_eps = float(post_norm_eps)
        self.fuse_branch_memory_kernel = bool(fuse_branch_memory_kernel)
        self.fuse_time_varying_memory_scan = bool(
            fuse_time_varying_memory_scan
        )
        self.fuse_time_varying_hidden_scan = bool(
            fuse_time_varying_hidden_scan
        )
        self.use_triton_time_varying_scan = bool(
            use_triton_time_varying_scan
        )
        self.use_rotating_frame_time_varying_scan = bool(
            use_rotating_frame_time_varying_scan
        )
        self.use_fused_triton_rotating_frame_time_varying_scan = bool(
            use_fused_triton_rotating_frame_time_varying_scan
        )
        if residual_step_scale <= 0:
            raise ValueError("residual_step_scale must be positive")
        self.residual_step_scale = float(residual_step_scale)

        self.input_norm = (
            nn.RMSNorm(width, eps=1e-6)
            if self.pre_normalize_qkv_inputs
            and self.recurrent_hidden_mode not in (
                "unitary-branch-normalized-residual",
                "unitary-branch-normalized-no-residual",
            )
            else nn.Identity()
        )
        self.query = nn.Linear(width, 2 * self.key_heads * key_dim, bias=False)
        self.key = nn.Linear(width, 2 * self.key_heads * key_dim, bias=False)
        self.value = nn.Linear(width, self.value_features, bias=False)
        self.output = nn.Linear(self.read_width, width, bias=False)

        # Mamba-3 style state-dependent decay and rotation increment.  Both are
        # predicted from the same normalized row that feeds Q/K/V and are scaled
        # by one shared step size DT, so a single timestep governs the decay
        # rate and the phase advance together (exponential-Euler).
        self.state_dependent_memory_decay = bool(state_dependent_memory_decay)
        self.memory_decay_floor = float(memory_decay_floor)
        if self.state_dependent_memory_decay:
            if not 0 < memory_decay_dt_min < memory_decay_dt_max:
                raise ValueError("memory decay dt range is invalid")
            if memory_decay_floor <= 0:
                raise ValueError("memory_decay_floor must be positive")
            self.memory_gate = nn.Linear(
                width, 2 * heads + heads * key_dim, bias=False
            )
            step = torch.exp(
                torch.rand(heads)
                * (math.log(memory_decay_dt_max) - math.log(memory_decay_dt_min))
                + math.log(memory_decay_dt_min)
            )
            self.dt_bias = nn.Parameter(step + torch.log(-torch.expm1(-step)))
        else:
            self.memory_gate = None
            self.register_parameter("dt_bias", None)

        # Constant per-(head,key) contraction on the forcing-tape memory scan.
        # ``S_j = lambda * U S_(j-1) + W_j`` stays diagonal and affine, so the
        # prefix composition is unchanged; only the multiplier leaves the unit
        # circle.  With ``||W_j|| <= B`` from the QKV prenorm the state is then
        # bounded by ``B / (1 - lambda)`` uniformly in the horizon count, which
        # is the condition under which the read needs no ``||S||_F`` division.
        self.scan_memory_decay = bool(scan_memory_decay)
        self.scan_memory_decay_max = float(scan_memory_decay_max)
        if self.scan_memory_decay:
            if not 0 < scan_memory_decay_max < 1:
                raise ValueError("scan_memory_decay_max must lie in (0, 1)")
            if not 0 < scan_memory_decay_init < scan_memory_decay_max:
                raise ValueError(
                    "scan_memory_decay_init must lie in "
                    f"(0, {scan_memory_decay_max})"
                )
            # lambda = max * sigmoid(logit) keeps the contraction strictly
            # below ``max`` with a gradient everywhere, so the bound cannot be
            # lost by the parameter drifting to one.
            ratio = scan_memory_decay_init / scan_memory_decay_max
            self.scan_decay_logit = nn.Parameter(
                torch.full(
                    (heads, key_dim),
                    math.log(ratio / (1.0 - ratio)),
                )
            )
        else:
            self.register_parameter("scan_decay_logit", None)

        self.hidden_phase = nn.Parameter(torch.zeros(width // 2))
        frequencies = torch.linspace(
            -initial_frequency_range,
            initial_frequency_range,
            heads * key_dim,
        ).reshape(heads, key_dim)
        self.memory_phase = nn.Parameter(frequencies)
        if learned_read_beta:
            self.beta_logit = nn.Parameter(
                torch.tensor(math.log(initial_beta / (1.0 - initial_beta)))
            )
        else:
            self.register_parameter("beta_logit", None)

        # Start the complex readout at a controlled scale without cutting the
        # first gradient to the complex projections.
        if self.recurrent_hidden_mode not in (
            "unitary-block-residual",
            "unitary-branch-normalized-residual",
            "unitary-branch-normalized-no-residual",
            "unitary-raw-qkv-boundary-normalized-hidden-residual",
            "unitary-detached-input-raw-read-postnorm-residual",
            "unitary-full-bptt-raw-read-joint-postnorm-residual",
        ):
            nn.init.normal_(
                self.output.weight,
                std=0.01 / math.sqrt(self.read_width),
            )

    @property
    def beta(self) -> torch.Tensor | None:
        if self.beta_logit is None:
            return None
        return torch.sigmoid(self.beta_logit)

    @property
    def scan_memory_decay_magnitude(self) -> torch.Tensor | None:
        """Per-(head,key) contraction ``lambda`` of the forcing-tape scan."""
        if self.scan_decay_logit is None:
            return None
        return self.scan_memory_decay_max * torch.sigmoid(self.scan_decay_logit)

    def _normalized(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.width:
            raise ValueError(
                f"hidden width must be {self.width}, got {hidden.shape[-1]}"
            )
        # The branch-normalized modes below hardwire the RMS normalization, so
        # ``pre_normalize_qkv_inputs`` cannot switch it off there.  This does.
        # The forcing tape is a unitary orbit ``c_h = R^h z_0``, so dropping
        # the normalization leaves ``||c_h||`` constant in the horizon anyway;
        # only its absolute scale stops being pinned, and now comes from the
        # encoder root instead.
        if not self.qkv_prenorm:
            return hidden
        if self.recurrent_hidden_mode in (
            "unitary-branch-normalized-residual",
            "unitary-branch-normalized-no-residual",
        ):
            return self._fixed_rms_normalize_hidden(hidden)
        return self.input_norm(hidden)

    def _post_normalize_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply a fixed, non-affine RMS post-normalization."""
        if not self.post_normalize_hidden_reads:
            return hidden
        return self._fixed_rms_normalize_hidden(hidden)

    def _fixed_rms_normalize_hidden(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """RMS-normalize hidden rows without learned affine parameters."""
        if hidden.shape[-1] != self.width:
            raise ValueError(
                f"hidden width must be {self.width}, got {hidden.shape[-1]}"
            )
        input_dtype = hidden.dtype
        hidden_float = hidden.float()
        variance = hidden_float.square().mean(dim=-1, keepdim=True)
        normalized = hidden_float * torch.rsqrt(variance + self.post_norm_eps)
        return normalized.to(input_dtype)

    def _normalize_initial_root(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the private recurrent root, leaving its encoder source raw."""
        if not self.normalize_initial_recurrent_root:
            return hidden
        return self._fixed_rms_normalize_hidden(hidden)

    def _post_normalize_complex_memory(
        self,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """RMS-normalize the complete complex recurrent carrier."""
        if not self.post_normalize_memory:
            return memory
        return self._fixed_rms_normalize_complex_memory(memory)

    def _fixed_rms_normalize_complex_memory(
        self,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Apply fixed non-affine RMS normalization to complex memory."""
        variance = memory.abs().square().float().mean(
            dim=(-3, -2, -1),
            keepdim=True,
        )
        return memory * torch.rsqrt(variance + self.post_norm_eps)

    def _post_normalize_final_carrier(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize both successor fields only at the recurrence boundary."""
        if not self.post_normalize_recurrent_state:
            return hidden, memory

        hidden_dtype = hidden.dtype
        hidden_float = hidden.float()
        hidden_variance = hidden_float.square().mean(dim=-1, keepdim=True)
        hidden = (
            hidden_float
            * torch.rsqrt(hidden_variance + self.post_norm_eps)
        ).to(hidden_dtype)

        memory_variance = memory.abs().square().float().mean(
            dim=(-3, -2, -1),
            keepdim=True,
        )
        memory = memory * torch.rsqrt(
            memory_variance + self.post_norm_eps
        )
        return hidden, memory

    def _broadcast_key_heads(self, values: torch.Tensor) -> torch.Tensor:
        """Share one key/query across every head under the MVA layout."""
        if not self.shared_kv_heads or self.heads == 1:
            return values
        return values.expand(*values.shape[:-2], self.heads, values.shape[-1])

    def _complex_projection(
        self,
        projection: nn.Linear,
        hidden: torch.Tensor,
        feature_dim: int,
        head_count: int | None = None,
    ) -> torch.Tensor:
        projected = projection(self._normalized(hidden))
        real, imaginary = projected.chunk(2, dim=-1)
        heads = self.heads if head_count is None else head_count
        shape = (*hidden.shape[:-1], heads, feature_dim)
        return torch.complex(real.reshape(shape), imaginary.reshape(shape))

    def queries(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._broadcast_key_heads(
            self._complex_projection(
                self.query,
                hidden,
                self.key_dim,
                self.key_heads,
            )
        )

    def keys(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._broadcast_key_heads(
            self._complex_projection(
                self.key,
                hidden,
                self.key_dim,
                self.key_heads,
            )
        )

    def values(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.real_value:
            projected = self.value(self._normalized(hidden))
            return projected.reshape(
                *hidden.shape[:-1], self.heads, self.value_dim
            )
        return self._complex_projection(
            self.value,
            hidden,
            self.value_dim,
        )

    def memory_multiplier(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the state-dependent complex memory multiplier.

        Follows the Mamba-3 SISO parameterization on the recurrence axis: the
        decay rate ``A`` and the signed rotation increment share one predicted
        step size ``DT``, so

        ``mult = exp(A * DT) * exp(-i * (theta + tanh(ang) * DT * pi))``

        with ``|mult| < 1``.  ``theta`` is the registered constant memory phase,
        so zeroing the predicted increment recovers the parent rotation exactly
        and the only strictly new behaviour is the decay.
        """
        if self.memory_gate is None:
            raise RuntimeError("state-dependent memory decay is disabled")
        projected = self.memory_gate(self._normalized(hidden))
        step_raw, decay_raw, angle_raw = projected.split(
            (self.heads, self.heads, self.heads * self.key_dim), dim=-1
        )
        # The parenthesis matters: ``heavy_tail`` is strictly positive, so
        # clamping before negating would collapse every input to the constant
        # ``+floor`` and invert the sign into growth.  Negate first, then floor.
        rate = (-heavy_tail(decay_raw.float())).clamp(
            max=-self.memory_decay_floor
        )
        step = F.softplus(step_raw.float() + self.dt_bias.float())
        decay = torch.exp(rate * step)
        angle_increment = (
            torch.tanh(angle_raw.float()).reshape(
                *hidden.shape[:-1], self.heads, self.key_dim
            )
            * step[..., None]
            * math.pi
        )
        phase = self.memory_phase + angle_increment
        multiplier = torch.polar(
            decay[..., None].expand_as(phase).contiguous(),
            -phase,
        ).to(torch.complex64)
        return multiplier, decay, angle_increment

    def _as_complex_value(self, values: torch.Tensor) -> torch.Tensor:
        """Return the write value as complex, whatever axis it lives on."""
        if values.is_complex():
            return values
        return values.to(
            torch.complex64 if values.dtype == torch.float32 else torch.complex128
        )

    def packed_qkv_weight(self) -> torch.Tensor:
        """Pack checkpoint-compatible Q/K/V parameters for one GEMM."""
        return torch.cat(
            (self.query.weight, self.key.weight, self.value.weight), dim=0
        )

    def project_qkv(
        self,
        hidden: torch.Tensor,
        packed_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Q/K/V from one normalized input and one linear projection."""
        if packed_weight is None:
            packed_weight = self.packed_qkv_weight()
        q_features = 2 * self.key_heads * self.key_dim
        v_features = self.value_features
        expected = 2 * q_features + v_features
        if packed_weight.shape != (expected, self.width):
            raise ValueError("packed QKV weight has unexpected shape")
        projected = F.linear(self._normalized(hidden), packed_weight)
        query_raw, key_raw, value_raw = projected.split(
            (q_features, q_features, v_features), dim=-1
        )

        def as_complex(
            values: torch.Tensor, feature_dim: int, head_count: int
        ) -> torch.Tensor:
            real, imaginary = values.chunk(2, dim=-1)
            shape = (*hidden.shape[:-1], head_count, feature_dim)
            return torch.complex(real.reshape(shape), imaginary.reshape(shape))

        if self.real_value:
            projected_values = value_raw.reshape(
                *hidden.shape[:-1], self.heads, self.value_dim
            )
        else:
            projected_values = as_complex(value_raw, self.value_dim, self.heads)
        return (
            self._broadcast_key_heads(
                as_complex(query_raw, self.key_dim, self.key_heads)
            ),
            self._broadcast_key_heads(
                as_complex(key_raw, self.key_dim, self.key_heads)
            ),
            projected_values,
        )

    def write(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return ``k(hidden) v(hidden)^dagger`` for every leading row."""
        keys = self.keys(hidden)
        values = self._as_complex_value(self.values(hidden))
        return torch.einsum("...hk,...hv->...hkv", keys, values.conj())

    def read(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        expected_query = (*memory.shape[:-2], self.key_dim)
        if query.shape != expected_query:
            raise ValueError(
                f"query must have shape {expected_query}, got {query.shape}"
            )
        return torch.einsum(
            "...hk,...hkv->...hv",
            query.conj(),
            memory,
        ) / math.sqrt(self.key_dim)

    def read_to_hidden(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-2:] != (self.heads, self.value_dim):
            raise ValueError("complex read has unexpected head/value dimensions")
        flattened = torch.cat(
            (values.real.flatten(-2), values.imag.flatten(-2)),
            dim=-1,
        )
        return self.output(flattened)

    def read_head_contributions(self, values: torch.Tensor) -> torch.Tensor:
        """Return the exactly additive hidden contribution of every head."""
        if values.shape[-2:] != (self.heads, self.value_dim):
            raise ValueError("complex read has unexpected head/value dimensions")
        real_weight = self.output.weight[:, : self.heads * self.value_dim]
        imaginary_weight = self.output.weight[:, self.heads * self.value_dim :]
        real_weight = real_weight.reshape(
            self.width,
            self.heads,
            self.value_dim,
        )
        imaginary_weight = imaginary_weight.reshape(
            self.width,
            self.heads,
            self.value_dim,
        )
        return (
            torch.einsum("...hv,dhv->...hd", values.real, real_weight)
            + torch.einsum("...hv,dhv->...hd", values.imag, imaginary_weight)
        )

    def initialize(self, hidden: torch.Tensor) -> ComplexMemoryState:
        """Initialize memory only from the known encoded prefix state."""
        recurrent_root = self._normalize_initial_root(hidden)
        if self.recurrent_hidden_mode in (
            "unitary-block-residual",
            "unitary-branch-normalized-residual",
            "unitary-branch-normalized-no-residual",
            "unitary-raw-qkv-boundary-normalized-hidden-residual",
            "unitary-detached-input-raw-read-postnorm-residual",
            "unitary-full-bptt-raw-read-joint-postnorm-residual",
        ):
            memory = torch.zeros(
                *recurrent_root.shape[:-1],
                self.heads,
                self.key_dim,
                self.value_dim,
                dtype=torch.complex64,
                device=recurrent_root.device,
            )
            return ComplexMemoryState(
                hidden=recurrent_root,
                memory=memory,
                write_count=0,
            )
        return ComplexMemoryState(
            hidden=recurrent_root,
            memory=self._post_normalize_complex_memory(
                self.write(recurrent_root)
            ),
            write_count=1,
        )

    def _normalize_accumulated_read(
        self,
        values: torch.Tensor,
        write_count: int,
    ) -> torch.Tensor:
        """Variance-normalize a read over ``write_count`` memory terms."""
        if not isinstance(write_count, int) or write_count < 1:
            raise ValueError("memory write_count must be a positive integer")
        if not self.normalize_accumulated_reads:
            return values
        return values / math.sqrt(write_count)

    def forward(
        self,
        state: ComplexMemoryState,
        *,
        write_scale: float = 1.0,
        read_beta: torch.Tensor | float | None = None,
        detach_preliminary_for_update: bool = False,
        collect_diagnostics: bool = True,
        packed_qkv_weight: torch.Tensor | None = None,
    ) -> ComplexSelfPredictionStep:
        if state.hidden.shape[:-1] != state.memory.shape[:-3]:
            raise ValueError("hidden and memory leading dimensions do not match")
        if state.memory.shape[-3:] != (
            self.heads,
            self.key_dim,
            self.value_dim,
        ):
            raise ValueError("memory has unexpected complex dimensions")
        if not state.memory.is_complex():
            raise ValueError("memory must be complex")
        minimum_write_count = (
            0
            if self.recurrent_hidden_mode in (
                "unitary-block-residual",
                "unitary-branch-normalized-residual",
                "unitary-branch-normalized-no-residual",
                "unitary-raw-qkv-boundary-normalized-hidden-residual",
                "unitary-detached-input-raw-read-postnorm-residual",
                "unitary-full-bptt-raw-read-joint-postnorm-residual",
            )
            else 1
        )
        if (
            not isinstance(state.write_count, int)
            or state.write_count < minimum_write_count
        ):
            raise ValueError("memory write_count is invalid for recurrent mode")
        if write_scale < 0:
            raise ValueError("write_scale must be non-negative")

        hidden_input = (
            state.hidden.detach()
            if self.recurrent_hidden_mode
            == "unitary-detached-input-raw-read-postnorm-residual"
            else state.hidden
        )
        hidden_angles = self.hidden_phase.expand(
            *hidden_input.shape[:-1],
            -1,
        )
        rotated_hidden = rotate_pairwise(hidden_input, hidden_angles)
        use_fused_branch = (
            self.recurrent_hidden_mode in (
                "unitary-branch-normalized-residual",
                "unitary-branch-normalized-no-residual",
            )
            and self.fuse_branch_memory_kernel
            and state.memory.is_cuda
            and write_scale == 1.0
            and not collect_diagnostics
        )
        if use_fused_branch:
            rotated_memory = None
        elif self.state_dependent_memory_decay:
            # The multiplier is predicted from the same rotated row that feeds
            # Q/K/V, so the memory transport becomes a contraction whose rate
            # and phase both depend on the current state.
            memory_multiplier, _, _ = self.memory_multiplier(rotated_hidden)
            rotated_memory = state.memory * memory_multiplier[..., None]
        else:
            memory_phase = torch.polar(
                torch.ones_like(self.memory_phase),
                -self.memory_phase,
            )
            rotated_memory = state.memory * memory_phase[..., None]

        if self.recurrent_hidden_mode in (
            "unitary-detached-input-raw-read-postnorm-residual",
            "unitary-full-bptt-raw-read-joint-postnorm-residual",
        ):
            joint_postnorm = (
                self.recurrent_hidden_mode
                == "unitary-full-bptt-raw-read-joint-postnorm-residual"
            )
            query, keys, values = self.project_qkv(
                rotated_hidden, packed_qkv_weight
            )
            innovation = torch.einsum(
                "...hk,...hv->...hkv", keys, values.conj()
            ) * float(write_scale)
            assert rotated_memory is not None
            raw_updated_memory = rotated_memory + innovation
            next_memory = (
                self._fixed_rms_normalize_complex_memory(
                    raw_updated_memory
                )
                if joint_postnorm
                else raw_updated_memory
            )
            next_write_count = state.write_count + 1

            # The measurement remains a raw linear read.  In the joint
            # postnorm mode the complete updated memory is normalized before
            # this read and before it becomes the successor carrier; there is
            # no additional realized-memory denominator in the read itself.
            measurement_read = self.read(query, next_memory)
            measurement_delta = self.read_to_hidden(measurement_read)
            raw_full_hidden = rotated_hidden + measurement_delta

            # This is the actual successor carrier, not a decoder-only view.
            full_hidden = self._fixed_rms_normalize_hidden(raw_full_hidden)
            memory_energy = (
                next_memory.abs().square().mean(dim=(-3, -2, -1))
                if collect_diagnostics
                else None
            )
            innovation_energy = (
                innovation.abs().square().mean(dim=(-3, -2, -1))
                if collect_diagnostics
                else None
            )
            return ComplexSelfPredictionStep(
                state=ComplexMemoryState(
                    hidden=full_hidden,
                    memory=next_memory,
                    write_count=next_write_count,
                ),
                read_hidden=full_hidden,
                preliminary_hidden=full_hidden,
                rotated_hidden=rotated_hidden,
                innovation_delta=measurement_delta,
                memory_energy=memory_energy,
                innovation_energy=innovation_energy,
                beta=None,
                raw_preliminary_hidden=raw_full_hidden,
                rotated_memory=rotated_memory,
                innovation_memory=innovation,
                raw_updated_memory=raw_updated_memory,
                raw_full_hidden=raw_full_hidden,
            )

        if self.recurrent_hidden_mode in (
            "unitary-branch-normalized-residual",
            "unitary-branch-normalized-no-residual",
            "unitary-raw-qkv-boundary-normalized-hidden-residual",
        ):
            boundary_normalized_hidden = (
                self.recurrent_hidden_mode
                == "unitary-raw-qkv-boundary-normalized-hidden-residual"
            )
            query, keys, values = self.project_qkv(
                rotated_hidden, packed_qkv_weight
            )
            if use_fused_branch:
                updated_pair, read_pair = (
                    _compiled_branch_memory_measurement_real(
                        torch.view_as_real(state.memory),
                        self.memory_phase,
                        torch.view_as_real(query),
                        torch.view_as_real(keys),
                        torch.view_as_real(values),
                        self.post_norm_eps,
                    )
                )
                raw_updated_memory = torch.view_as_complex(
                    updated_pair.contiguous()
                )
                measurement_read = torch.view_as_complex(
                    read_pair.contiguous()
                )
                innovation = None
            else:
                innovation = torch.einsum(
                    "...hk,...hv->...hkv",
                    keys,
                    self._as_complex_value(values).conj(),
                ) * float(write_scale)
                assert rotated_memory is not None
                raw_updated_memory = rotated_memory + innovation
                measurement_read = None
            next_memory = raw_updated_memory
            next_write_count = state.write_count + 1
            if measurement_read is None:
                measurement_read = self.read(query, next_memory)
                if self.normalize_read_by_memory_norm:
                    memory_frobenius = next_memory.abs().square().sum(
                        dim=(-3, -2, -1)
                    ).sqrt()
                    measurement_read = measurement_read / (
                        memory_frobenius[..., None, None] + self.post_norm_eps
                    )
            measurement_delta = self.read_to_hidden(measurement_read)
            full_hidden = (
                measurement_delta
                if self.recurrent_hidden_mode
                == "unitary-branch-normalized-no-residual"
                else rotated_hidden + measurement_delta
            )
            raw_full_hidden = full_hidden
            if boundary_normalized_hidden:
                # The successor carrier itself is normalized, so the state that
                # re-enters the next recurrent step and the decoder-facing view
                # are the same normalized ray.  Memory stays raw.
                full_hidden = self._fixed_rms_normalize_hidden(raw_full_hidden)
                read_hidden = full_hidden
            else:
                # The carrier stays raw. Only the decoder branch observes its
                # ray.
                read_hidden = self._fixed_rms_normalize_hidden(full_hidden)
            memory_energy = (
                next_memory.abs().square().mean(dim=(-3, -2, -1))
                if collect_diagnostics
                else None
            )
            innovation_energy = (
                innovation.abs().square().mean(dim=(-3, -2, -1))
                if collect_diagnostics
                else None
            )
            return ComplexSelfPredictionStep(
                state=ComplexMemoryState(
                    hidden=full_hidden,
                    memory=next_memory,
                    write_count=next_write_count,
                ),
                read_hidden=read_hidden,
                preliminary_hidden=read_hidden,
                rotated_hidden=rotated_hidden,
                innovation_delta=measurement_delta,
                memory_energy=memory_energy,
                innovation_energy=innovation_energy,
                beta=None,
                raw_preliminary_hidden=raw_full_hidden,
                rotated_memory=rotated_memory,
                innovation_memory=innovation,
                raw_updated_memory=raw_updated_memory,
                raw_full_hidden=raw_full_hidden,
            )

        if self.recurrent_hidden_mode == "unitary-block-residual":
            # Close the original single-layer SSM at the block boundary:
            # write the current latent once, read the updated superposition
            # once, and use that one measurement as a residual correction for
            # both the decoder-facing output and the next recurrent input.
            innovation = self.write(rotated_hidden) * float(write_scale)
            raw_updated_memory = rotated_memory + innovation
            next_memory = self._post_normalize_complex_memory(
                raw_updated_memory
            )
            next_write_count = state.write_count + 1
            measurement_read = self.read(
                self.queries(rotated_hidden),
                next_memory,
            )
            measurement_read = self._normalize_accumulated_read(
                measurement_read,
                next_write_count,
            )
            measurement_delta = self.read_to_hidden(measurement_read)
            full_hidden = (
                rotated_hidden
                + self.residual_step_scale * measurement_delta
            )
            raw_full_hidden = full_hidden
            full_hidden = self._post_normalize_hidden(full_hidden)
            full_hidden, next_memory = self._post_normalize_final_carrier(
                full_hidden,
                next_memory,
            )
            memory_energy = next_memory.abs().square().mean(
                dim=(-3, -2, -1)
            )
            innovation_energy = innovation.abs().square().mean(
                dim=(-3, -2, -1)
            )
            return ComplexSelfPredictionStep(
                state=ComplexMemoryState(
                    hidden=full_hidden,
                    memory=next_memory,
                    write_count=next_write_count,
                ),
                read_hidden=full_hidden,
                preliminary_hidden=full_hidden,
                rotated_hidden=rotated_hidden,
                innovation_delta=measurement_delta,
                memory_energy=memory_energy,
                innovation_energy=innovation_energy,
                beta=None,
                raw_preliminary_hidden=raw_full_hidden,
                rotated_memory=rotated_memory,
                innovation_memory=innovation,
                raw_updated_memory=raw_updated_memory,
                raw_full_hidden=raw_full_hidden,
            )

        # The decoder-facing prediction is made before this step's innovation
        # is written.  ``rotated_memory`` is S^- = U S from the state-space
        # notation, so this is exactly the causal q^dagger S^- readout.
        prior_read = self.read(self.queries(rotated_hidden), rotated_memory)
        prior_read = self._normalize_accumulated_read(
            prior_read,
            state.write_count,
        )
        preliminary = self.read_to_hidden(prior_read)
        if self.residual_prior:
            preliminary = rotated_hidden + preliminary
        raw_preliminary = preliminary
        preliminary = self._post_normalize_hidden(preliminary)

        # Some objectives treat the causal prior readout as a fixed input to
        # the corrector.  Detaching only this edge keeps ordinary token-loss
        # gradients to ``preliminary`` intact while preventing successor
        # closure from changing the prior merely to make correction easier.
        update_hidden = (
            preliminary.detach()
            if detach_preliminary_for_update
            else preliminary
        )
        innovation = self.write(update_hidden) * float(write_scale)
        updated_query = self.queries(update_hidden)
        innovation_read = self.read(updated_query, innovation)
        innovation_delta = self.read_to_hidden(innovation_read)
        raw_updated_memory = rotated_memory + innovation
        next_memory = self._post_normalize_complex_memory(
            raw_updated_memory
        )
        next_write_count = state.write_count + 1
        if self.recurrent_hidden_mode == "post-update-full-read":
            full_read = self.read(updated_query, next_memory)
            full_read = self._normalize_accumulated_read(
                full_read,
                next_write_count,
            )
            full_hidden = self.read_to_hidden(full_read)
        elif (
            self.recurrent_hidden_mode
            == "rotated-hidden-innovation-residual"
        ):
            # Preserve the same-level unitary carrier and add only the
            # innovation-memory read as an explicit correction.  Unlike a
            # post-update full read, this does not reinterpret an absolute
            # reread of the old memory as a residual delta.
            full_hidden = rotated_hidden + innovation_delta
        else:
            full_hidden = preliminary + innovation_delta
        raw_full_hidden = full_hidden
        full_hidden = self._post_normalize_hidden(full_hidden)

        if read_beta is None:
            beta = self.beta
            read_hidden = (
                preliminary
                if beta is None
                else preliminary + beta * innovation_delta
            )
        else:
            beta = torch.as_tensor(
                read_beta,
                device=full_hidden.device,
                dtype=full_hidden.dtype,
            )
            read_hidden = preliminary + beta * innovation_delta
        read_hidden = self._post_normalize_hidden(read_hidden)
        full_hidden, next_memory = self._post_normalize_final_carrier(
            full_hidden,
            next_memory,
        )
        memory_energy = next_memory.abs().square().mean(dim=(-3, -2, -1))
        innovation_energy = innovation.abs().square().mean(dim=(-3, -2, -1))
        return ComplexSelfPredictionStep(
            state=ComplexMemoryState(
                hidden=full_hidden,
                memory=next_memory,
                write_count=next_write_count,
            ),
            read_hidden=read_hidden,
            preliminary_hidden=preliminary,
            rotated_hidden=rotated_hidden,
            innovation_delta=innovation_delta,
            memory_energy=memory_energy,
            innovation_energy=innovation_energy,
            beta=beta,
            raw_preliminary_hidden=raw_preliminary,
            rotated_memory=rotated_memory,
            innovation_memory=innovation,
            raw_updated_memory=raw_updated_memory,
            raw_full_hidden=raw_full_hidden,
        )

    def rollout(
        self,
        root: torch.Tensor,
        horizons: int,
        *,
        write_scale: float = 1.0,
        read_beta: torch.Tensor | float | None = None,
        detach_preliminary_for_update: bool = False,
        detach_state_between_horizons: bool = False,
    ) -> ComplexSelfPredictionRollout:
        if horizons < 1:
            raise ValueError("horizons must be positive")
        state = self.initialize(root)
        steps = []
        packed_qkv_weight = (
            self.packed_qkv_weight()
            if self.recurrent_hidden_mode in (
                "unitary-branch-normalized-residual",
                "unitary-branch-normalized-no-residual",
                "unitary-raw-qkv-boundary-normalized-hidden-residual",
                "unitary-detached-input-raw-read-postnorm-residual",
                "unitary-full-bptt-raw-read-joint-postnorm-residual",
            )
            else None
        )
        for horizon in range(horizons):
            if detach_state_between_horizons and horizon:
                state = ComplexMemoryState(
                    hidden=state.hidden.detach(),
                    memory=state.memory.detach(),
                    write_count=state.write_count,
                )
            step = self(
                state,
                write_scale=write_scale,
                read_beta=read_beta,
                detach_preliminary_for_update=(
                    detach_preliminary_for_update
                ),
                packed_qkv_weight=packed_qkv_weight,
            )
            state = step.state
            steps.append(step)
        return ComplexSelfPredictionRollout(
            full_states=torch.stack([step.state.hidden for step in steps], dim=-2),
            read_states=torch.stack([step.read_hidden for step in steps], dim=-2),
            preliminary_states=torch.stack(
                [step.preliminary_hidden for step in steps], dim=-2
            ),
            rotated_states=torch.stack(
                [step.rotated_hidden for step in steps], dim=-2
            ),
            innovation_deltas=torch.stack(
                [step.innovation_delta for step in steps], dim=-2
            ),
            memory_energy=torch.stack(
                [step.memory_energy for step in steps], dim=-1
            ),
            innovation_energy=torch.stack(
                [step.innovation_energy for step in steps], dim=-1
            ),
            beta=steps[-1].beta,
            final_state=state,
        )

    def rollout_read_states(
        self,
        root: torch.Tensor,
        horizons: int,
        *,
        write_scale: float = 1.0,
    ) -> tuple[torch.Tensor, ComplexMemoryState]:
        """Return only decoder states for a CE-only training rollout."""
        if horizons < 1:
            raise ValueError("horizons must be positive")
        state = self.initialize(root)
        read_states = []
        packed_qkv_weight = (
            self.packed_qkv_weight()
            if self.recurrent_hidden_mode in (
                "unitary-branch-normalized-residual",
                "unitary-branch-normalized-no-residual",
                "unitary-raw-qkv-boundary-normalized-hidden-residual",
                "unitary-detached-input-raw-read-postnorm-residual",
                "unitary-full-bptt-raw-read-joint-postnorm-residual",
            )
            else None
        )
        for _ in range(horizons):
            step = self(
                state,
                write_scale=write_scale,
                collect_diagnostics=False,
                packed_qkv_weight=packed_qkv_weight,
            )
            state = step.state
            read_states.append(step.read_hidden)
        return torch.stack(read_states, dim=-2), state

    def forcing_velocity_tape(
        self,
        forcing_states: torch.Tensor,
        *,
        collect_scan_diagnostics: bool = False,
        value_noise: torch.Tensor | None = None,
        value_noise_scale: float = 0.0,
    ) -> ComplexForcingVelocityTape:
        """Return the shared causal vector field for a supplied state tape.

        The horizon axis is the penultimate axis.  Q/K/V are evaluated on the
        supplied rows, while the memory response remains the exact prefix scan
        used by :meth:`rollout_time_varying_scan`.  This public primitive lets
        a bridge state drive the same transition during conditional flow
        matching without copying the central model into an experiment script.
        """
        if forcing_states.ndim < 2:
            raise ValueError("forcing tape must include horizon and width axes")
        if forcing_states.shape[-2] < 1:
            raise ValueError("forcing tape horizons must be positive")
        if forcing_states.shape[-1] != self.width:
            raise ValueError(
                f"forcing width must be {self.width}, got "
                f"{forcing_states.shape[-1]}"
            )
        if self.recurrent_hidden_mode != "unitary-branch-normalized-residual":
            raise ValueError(
                "forcing velocity requires branch-normalized unitary mode"
            )
        if value_noise_scale < 0:
            raise ValueError("value_noise_scale must be non-negative")
        if value_noise is None and value_noise_scale != 0:
            raise ValueError(
                "non-zero value_noise_scale requires an explicit noise tape"
            )

        horizons = forcing_states.shape[-2]
        leading_shape = forcing_states.shape[:-2]
        query, keys, values = self.project_qkv(
            forcing_states,
            self.packed_qkv_weight(),
        )
        if value_noise is not None:
            if value_noise.shape != values.shape:
                raise ValueError(
                    "value_noise must match the projected value tape: "
                    f"expected {tuple(values.shape)}, got "
                    f"{tuple(value_noise.shape)}"
                )
            if not value_noise.is_complex():
                raise ValueError("value_noise must be complex")
            value_rms = values.abs().square().mean(
                dim=-1,
                keepdim=True,
            ).sqrt().detach()
            values = values + (
                float(value_noise_scale)
                * value_rms
                * value_noise.to(device=values.device, dtype=values.dtype)
            )
        writes = torch.einsum(
            "...hk,...hv->...hkv",
            keys,
            self._as_complex_value(values).conj(),
        )
        flattened_writes = writes.flatten(-3, -2)
        local_memory_angles = (-self.memory_phase).flatten().expand(
            *leading_shape,
            horizons,
            -1,
        )
        decay_magnitude = self.scan_memory_decay_magnitude
        if decay_magnitude is not None and (
            self.use_fused_triton_rotating_frame_time_varying_scan
            or self.use_rotating_frame_time_varying_scan
        ):
            # Both rotating-frame backends divide by the cumulative multiplier,
            # which is only invertible without amplification while it stays on
            # the unit circle.
            raise ValueError(
                "scan memory decay requires a general affine scan backend; "
                "the rotating-frame backends assume a unit-modulus multiplier"
            )
        if self.use_fused_triton_rotating_frame_time_varying_scan:
            if not forcing_states.is_cuda:
                raise ValueError("fused Triton rotating frame requires CUDA")
            from .triton_affine_scan import triton_constant_rotation_scan

            flattened_memory_states = triton_constant_rotation_scan(
                self.memory_phase.flatten(),
                flattened_writes,
                angle_sign=-1,
            )
            if collect_scan_diagnostics:
                cumulative_memory_angles = local_memory_angles.cumsum(
                    dim=-2
                )
                cumulative_memory_multipliers = torch.polar(
                    torch.ones_like(cumulative_memory_angles),
                    cumulative_memory_angles,
                )
            else:
                cumulative_memory_multipliers = None
        elif self.use_rotating_frame_time_varying_scan:
            cumulative_memory_angles = local_memory_angles.cumsum(dim=-2)
            cumulative_memory_multipliers = torch.polar(
                torch.ones_like(cumulative_memory_angles),
                cumulative_memory_angles,
            )
            frame_writes = (
                cumulative_memory_multipliers.conj()[..., None]
                * flattened_writes
            )
            flattened_memory_states = (
                cumulative_memory_multipliers[..., None]
                * frame_writes.cumsum(dim=-3)
            )
        else:
            local_memory_multipliers = torch.polar(
                torch.ones_like(local_memory_angles)
                if decay_magnitude is None
                else decay_magnitude.flatten().expand_as(local_memory_angles),
                local_memory_angles,
            )
        if (
            not self.use_fused_triton_rotating_frame_time_varying_scan
            and not self.use_rotating_frame_time_varying_scan
            and self.use_triton_time_varying_scan
            and forcing_states.is_cuda
        ):
            from .triton_affine_scan import triton_complex_affine_scan

            flattened_memory_states = triton_complex_affine_scan(
                local_memory_multipliers,
                flattened_writes,
            )
            cumulative_memory_multipliers = torch.cumprod(
                local_memory_multipliers,
                dim=-2,
            )
        elif (
            not self.use_fused_triton_rotating_frame_time_varying_scan
            and not self.use_rotating_frame_time_varying_scan
            and self.fuse_time_varying_memory_scan
            and forcing_states.is_cuda
        ):
            flattened_memory_states = _FusedComplexMemoryAffineScan.apply(
                local_memory_multipliers,
                flattened_writes,
            )
            cumulative_memory_multipliers = torch.cumprod(
                local_memory_multipliers,
                dim=-2,
            )
        elif not (
            self.use_fused_triton_rotating_frame_time_varying_scan
            or self.use_rotating_frame_time_varying_scan
        ):
            (
                cumulative_memory_multipliers,
                flattened_memory_states,
            ) = complex_memory_affine_scan(
                local_memory_multipliers,
                flattened_writes,
            )
        memory_states = flattened_memory_states.reshape(
            *leading_shape,
            horizons,
            self.heads,
            self.key_dim,
            self.value_dim,
        )
        measurement = self.read(query, memory_states)
        if self.normalize_read_by_memory_norm:
            memory_frobenius = memory_states.abs().square().sum(
                dim=(-3, -2, -1)
            ).sqrt()
            measurement = measurement / (
                memory_frobenius[..., None, None] + self.post_norm_eps
            )
        return ComplexForcingVelocityTape(
            velocity=self.read_to_hidden(measurement),
            memory_states=memory_states,
            cumulative_memory_multipliers=cumulative_memory_multipliers,
        )

    def _scan_hidden_innovation_tape(
        self,
        root: torch.Tensor,
        innovation_deltas: torch.Tensor,
        *,
        collect_scan_diagnostics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Solve ``z_h = R z_(h-1) + delta_h`` with one shared backend."""
        if innovation_deltas.ndim != root.ndim + 1:
            raise ValueError("innovation tape must add one horizon axis")
        if innovation_deltas.shape[:-2] != root.shape[:-1]:
            raise ValueError("innovation tape leading shape does not match root")
        if innovation_deltas.shape[-1] != self.width:
            raise ValueError("innovation tape width does not match transition")
        horizons = innovation_deltas.shape[-2]
        if horizons < 1:
            raise ValueError("innovation tape horizons must be positive")

        local_hidden_angles = self.hidden_phase.expand(
            *root.shape[:-1],
            horizons,
            -1,
        )
        cumulative_schedule_angles = local_hidden_angles.cumsum(dim=-2)
        if self.use_fused_triton_rotating_frame_time_varying_scan:
            from .triton_affine_scan import triton_constant_rotation_scan

            root_pairs = root.reshape(
                *root.shape[:-1],
                self.width // 2,
                2,
            )
            increment_pairs = innovation_deltas.reshape(
                *innovation_deltas.shape[:-1],
                self.width // 2,
                2,
            )
            complex_root = torch.view_as_complex(root_pairs)
            complex_increments = torch.view_as_complex(increment_pairs)
            complex_states = triton_constant_rotation_scan(
                self.hidden_phase,
                complex_increments[..., None],
                complex_root[..., None],
                angle_sign=1,
            ).squeeze(-1)
            full_states = torch.view_as_real(complex_states).flatten(-2)
            if collect_scan_diagnostics:
                cumulative_hidden_angles = cumulative_schedule_angles
                cumulative_hidden_increments = full_states - rotate_pairwise(
                    root.unsqueeze(-2).expand_as(full_states),
                    cumulative_schedule_angles,
                )
            else:
                cumulative_hidden_angles = None
                cumulative_hidden_increments = None
        elif self.use_rotating_frame_time_varying_scan:
            (
                full_states,
                cumulative_hidden_angles,
                cumulative_hidden_increments,
            ) = _rotating_frame_pairwise_affine_scan(
                root,
                local_hidden_angles,
                innovation_deltas,
            )
        elif self.use_triton_time_varying_scan and root.is_cuda:
            (
                full_states,
                cumulative_hidden_angles,
                cumulative_hidden_increments,
            ) = _triton_pairwise_affine_scan(
                root,
                local_hidden_angles,
                innovation_deltas,
            )
        elif self.fuse_time_varying_hidden_scan and root.is_cuda:
            (
                full_states,
                cumulative_hidden_angles,
                cumulative_hidden_increments,
            ) = _fused_pairwise_affine_scan(
                root,
                local_hidden_angles,
                innovation_deltas,
            )
        else:
            (
                full_states,
                cumulative_hidden_angles,
                cumulative_hidden_increments,
            ) = apply_pairwise_affine_scan(
                root,
                local_hidden_angles,
                innovation_deltas,
            )
        if not collect_scan_diagnostics:
            cumulative_hidden_angles = None
            cumulative_hidden_increments = None
        return (
            full_states,
            cumulative_hidden_angles,
            cumulative_hidden_increments,
        )

    def rollout_time_varying_scan(
        self,
        root: torch.Tensor,
        horizons: int,
        *,
        collect_scan_diagnostics: bool = True,
        value_noise: torch.Tensor | None = None,
        value_noise_scale: float = 0.0,
    ) -> ComplexTimeVaryingScanRollout:
        """Evaluate the scan-compatible time-varying recurrence exactly.

        Let ``c_0 = root`` be a fixed conditioning carrier.  Its unitary orbit
        ``c_j = R c_(j-1)`` compiles all Q/K/V writes before recurrent state
        evaluation.  The actual central states then obey

        ``S_j = U S_(j-1) + k(c_j) v(c_j)^dagger``

        ``z_j = R z_(j-1) + O(read(q(c_j), S_j))``.

        Both returned state tapes are causal prefix compositions of these
        recurrences.  Only the measurement-to-future-QKV feedback edge of the
        ordinary rollout is absent; no horizon output is computed as an
        independent direct prediction from ``root``.  When ``value_noise`` is
        supplied, the write value is replaced by

        ``v(c_j) + value_noise_scale * rms(v(c_j)) * value_noise_j``.

        The explicit tape is sampled by the caller.  It is therefore another
        time-varying low-rank write and preserves the exact scan algebra.
        """
        if horizons < 1:
            raise ValueError("horizons must be positive")
        if self.recurrent_hidden_mode != "unitary-branch-normalized-residual":
            raise ValueError(
                "time-varying scan requires branch-normalized unitary mode"
            )
        if root.shape[-1] != self.width:
            raise ValueError(
                f"root width must be {self.width}, got {root.shape[-1]}"
            )
        if value_noise_scale < 0:
            raise ValueError("value_noise_scale must be non-negative")
        if value_noise is None and value_noise_scale != 0:
            raise ValueError(
                "non-zero value_noise_scale requires an explicit noise tape"
            )

        local_hidden_angles = self.hidden_phase.expand(
            *root.shape[:-1],
            horizons,
            -1,
        )
        cumulative_schedule_angles = local_hidden_angles.cumsum(dim=-2)
        expanded_root = root.unsqueeze(-2).expand(
            *root.shape[:-1],
            horizons,
            self.width,
        )
        schedule_states = rotate_pairwise(
            expanded_root,
            cumulative_schedule_angles,
        )

        forcing = self.forcing_velocity_tape(
            schedule_states,
            collect_scan_diagnostics=collect_scan_diagnostics,
            value_noise=value_noise,
            value_noise_scale=value_noise_scale,
        )
        innovation_deltas = forcing.velocity
        memory_states = forcing.memory_states
        cumulative_memory_multipliers = (
            forcing.cumulative_memory_multipliers
        )
        (
            full_states,
            cumulative_hidden_angles,
            cumulative_hidden_increments,
        ) = self._scan_hidden_innovation_tape(
            root,
            innovation_deltas,
            collect_scan_diagnostics=collect_scan_diagnostics,
        )
        read_states = self._fixed_rms_normalize_hidden(full_states)
        final_state = ComplexMemoryState(
            hidden=full_states[..., -1, :],
            memory=memory_states[..., -1, :, :, :],
            write_count=horizons,
        )
        return ComplexTimeVaryingScanRollout(
            full_states=full_states,
            read_states=read_states,
            schedule_states=schedule_states,
            innovation_deltas=innovation_deltas,
            memory_states=memory_states,
            cumulative_hidden_angles=cumulative_hidden_angles,
            cumulative_hidden_increments=cumulative_hidden_increments,
            cumulative_memory_multipliers=(
                cumulative_memory_multipliers
            ),
            final_state=final_state,
        )

    def rollout_iterative_feedback_scan(
        self,
        root: torch.Tensor,
        horizons: int,
        *,
        feedback_rescans: int,
        collect_scan_diagnostics: bool = True,
        value_noise: torch.Tensor | None = None,
        value_noise_scale: float = 0.0,
    ) -> ComplexIterativeFeedbackScanRollout:
        """Refine the compiled scan toward the literal feedback recurrence.

        ``feedback_rescans=0`` is exactly :meth:`rollout_time_varying_scan`.
        Each additional rescan uses

        ``c_h = R [root, z_1, ..., z_(H-1)]_h``

        from the preceding raw state tape.  Memory is recomputed from zero on
        every rescan.  This is an R-preconditioned Picard iteration: rescan K
        is structurally exact through horizon K+1, while H-1 rescans recover
        the complete H-step feedback tape up to numerical scan tolerance.
        """
        if not isinstance(feedback_rescans, int) or feedback_rescans < 0:
            raise ValueError("feedback_rescans must be a non-negative integer")
        initial = self.rollout_time_varying_scan(
            root,
            horizons,
            collect_scan_diagnostics=collect_scan_diagnostics,
            value_noise=value_noise,
            value_noise_scale=value_noise_scale,
        )
        full_states = initial.full_states
        forcing_states = initial.schedule_states
        innovation_deltas = initial.innovation_deltas
        memory_states = initial.memory_states
        cumulative_hidden_angles = initial.cumulative_hidden_angles
        cumulative_hidden_increments = initial.cumulative_hidden_increments
        cumulative_memory_multipliers = (
            initial.cumulative_memory_multipliers
        )

        local_hidden_angles = self.hidden_phase.expand(
            *root.shape[:-1],
            horizons,
            -1,
        )
        for _ in range(feedback_rescans):
            predecessors = torch.cat(
                (root.unsqueeze(-2), full_states[..., :-1, :]),
                dim=-2,
            )
            forcing_states = rotate_pairwise(
                predecessors,
                local_hidden_angles,
            )
            forcing = self.forcing_velocity_tape(
                forcing_states,
                collect_scan_diagnostics=collect_scan_diagnostics,
                value_noise=value_noise,
                value_noise_scale=value_noise_scale,
            )
            innovation_deltas = forcing.velocity
            memory_states = forcing.memory_states
            cumulative_memory_multipliers = (
                forcing.cumulative_memory_multipliers
            )
            (
                full_states,
                cumulative_hidden_angles,
                cumulative_hidden_increments,
            ) = self._scan_hidden_innovation_tape(
                root,
                innovation_deltas,
                collect_scan_diagnostics=collect_scan_diagnostics,
            )

        read_states = self._fixed_rms_normalize_hidden(full_states)
        final_state = ComplexMemoryState(
            hidden=full_states[..., -1, :],
            memory=memory_states[..., -1, :, :, :],
            write_count=horizons,
        )
        return ComplexIterativeFeedbackScanRollout(
            full_states=full_states,
            read_states=read_states,
            forcing_states=forcing_states,
            innovation_deltas=innovation_deltas,
            memory_states=memory_states,
            cumulative_hidden_angles=cumulative_hidden_angles,
            cumulative_hidden_increments=cumulative_hidden_increments,
            cumulative_memory_multipliers=(
                cumulative_memory_multipliers
            ),
            feedback_rescans=feedback_rescans,
            final_state=final_state,
        )

    def rollout_time_varying_scan_read_states(
        self,
        root: torch.Tensor,
        horizons: int,
        *,
        value_noise: torch.Tensor | None = None,
        value_noise_scale: float = 0.0,
    ) -> tuple[torch.Tensor, ComplexMemoryState]:
        """Return decoder states from the exact time-varying scan."""
        rollout = self.rollout_time_varying_scan(
            root,
            horizons,
            collect_scan_diagnostics=False,
            value_noise=value_noise,
            value_noise_scale=value_noise_scale,
        )
        return rollout.read_states, rollout.final_state

    def rollout_iterative_feedback_scan_read_states(
        self,
        root: torch.Tensor,
        horizons: int,
        *,
        feedback_rescans: int,
    ) -> tuple[torch.Tensor, ComplexMemoryState]:
        """Return decoder states after finite feedback-scan refinements."""
        rollout = self.rollout_iterative_feedback_scan(
            root,
            horizons,
            feedback_rescans=feedback_rescans,
            collect_scan_diagnostics=False,
        )
        return rollout.read_states, rollout.final_state


__all__ = [
    "ComplexIterativeFeedbackScanRollout",
    "ComplexMemoryState",
    "ComplexSelfPredictedKVTransition",
    "ComplexSelfPredictionRollout",
    "ComplexSelfPredictionStep",
    "ComplexTimeVaryingScanRollout",
    "complex_memory_affine_scan",
]
