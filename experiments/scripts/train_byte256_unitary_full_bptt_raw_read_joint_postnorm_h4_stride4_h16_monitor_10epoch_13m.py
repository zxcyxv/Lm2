"""Ten-epoch H4 run with full BPTT and joint hidden/memory postnorm."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from rotlm.models.complex_self_prediction import ComplexMemoryState
import train_byte256_unitary_detached_input_raw_read_postnorm_h4_stride4_h16_monitor_10epoch_13m as control


base = control.base
parent = control.parent
branch = control.branch
EXPERIMENT_ID = (
    "EXP-20260807-byte256-unitary-full-bptt-raw-read-joint-postnorm-"
    "h4-stride4-h16-monitor-10epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
RECURRENT_HIDDEN_MODE = (
    "unitary-full-bptt-raw-read-joint-postnorm-residual"
)
GENERIC_H4_PREFLIGHT = control.GENERIC_H4_PREFLIGHT
H4_FINAL_CHECKS = control.H4_FINAL_CHECKS
H4_INTERPRETATION = control.H4_INTERPRETATION
TOTAL_STEPS = parent.TOTAL_STEPS
REPORT_STEPS = frozenset(
    {1, 50, *range(100, TOTAL_STEPS + 1, 100), *parent.REPORT_STEPS}
)


def _finite_nonzero(name: str, value: torch.Tensor | None) -> float:
    if (
        value is None
        or not bool(torch.isfinite(value).all())
        or not bool(value.norm() > 0)
    ):
        raise RuntimeError(f"no finite nonzero {name}")
    return float(value.norm())


def preflight(model, training, *, latent_target_model=None):
    values = GENERIC_H4_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    if latent_target_model is not None:
        raise RuntimeError("CE-only joint-postnorm run unexpectedly received EMA")

    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != RECURRENT_HIDDEN_MODE:
        raise RuntimeError("full-BPTT joint-postnorm recurrent mode is disabled")
    if transition.pre_normalize_qkv_inputs:
        raise RuntimeError("the post-normalized hidden received an extra QKV norm")
    if not isinstance(transition.input_norm, torch.nn.Identity):
        raise RuntimeError("the QKV input path is not identity")
    if not transition.normalize_initial_recurrent_root:
        raise RuntimeError("the initial recurrent root is not fixed-RMS normalized")
    if transition.normalize_accumulated_reads:
        raise RuntimeError("write-count read scaling is active")
    if any((
        transition.post_normalize_hidden_reads,
        transition.post_normalize_memory,
        transition.post_normalize_recurrent_state,
    )):
        raise RuntimeError("a second configurable carrier norm is active")
    if transition.fuse_branch_memory_kernel:
        raise RuntimeError("the Frobenius-normalizing fused branch is active")

    generator = torch.Generator().manual_seed(base.SEED + 31_807)
    window = base.sampled_windows(
        training,
        1,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        initial = transition.initialize(roots)
        first = transition(initial)
        second = transition(first.state)

        first_query, first_keys, first_values = transition.project_qkv(
            first.rotated_hidden
        )
        first_write = torch.einsum(
            "...hk,...hv->...hkv", first_keys, first_values.conj()
        )
        first_expected_raw_memory = first.rotated_memory + first_write
        first_expected_memory = (
            transition._fixed_rms_normalize_complex_memory(
                first_expected_raw_memory
            )
        )
        first_expected_read = transition.read(
            first_query,
            first_expected_memory,
        )
        first_expected_delta = transition.read_to_hidden(first_expected_read)
        first_expected_raw_hidden = first.rotated_hidden + first_expected_delta
        first_expected_hidden = transition._fixed_rms_normalize_hidden(
            first_expected_raw_hidden
        )

        second_query, second_keys, second_values = transition.project_qkv(
            second.rotated_hidden
        )
        second_write = torch.einsum(
            "...hk,...hv->...hkv", second_keys, second_values.conj()
        )
        second_expected_raw_memory = second.rotated_memory + second_write
        second_expected_memory = (
            transition._fixed_rms_normalize_complex_memory(
                second_expected_raw_memory
            )
        )
        second_expected_read = transition.read(
            second_query,
            second_expected_memory,
        )
        second_expected_delta = transition.read_to_hidden(
            second_expected_read
        )
        second_expected_raw_hidden = second.rotated_hidden + second_expected_delta
        second_expected_hidden = transition._fixed_rms_normalize_hidden(
            second_expected_raw_hidden
        )

    # Both fields of the predecessor carry must remain differentiable.
    hidden = roots[:, :1].detach().clone().requires_grad_(True)
    memory_shape = (
        *hidden.shape[:-1],
        transition.heads,
        transition.key_dim,
        transition.value_dim,
    )
    memory = torch.complex(
        torch.full(memory_shape, 0.125, device=hidden.device),
        torch.full(memory_shape, -0.25, device=hidden.device),
    ).requires_grad_(True)
    live = transition(
        ComplexMemoryState(hidden=hidden, memory=memory, write_count=1)
    )
    probe = torch.linspace(
        -1.0,
        1.0,
        transition.width,
        device=hidden.device,
        dtype=hidden.dtype,
    ).reshape(1, 1, -1)
    boundary_loss = (live.state.hidden * probe).sum()
    hidden_gradient, memory_gradient = torch.autograd.grad(
        boundary_loss,
        (hidden, memory),
    )
    hidden_gradient_norm = _finite_nonzero(
        "hidden-boundary gradient",
        hidden_gradient,
    )
    memory_gradient_norm = _finite_nonzero(
        "memory-boundary gradient",
        memory_gradient,
    )

    first_hidden_rms = first.state.hidden.float().square().mean(
        dim=-1
    ).sqrt()
    second_hidden_rms = second.state.hidden.float().square().mean(
        dim=-1
    ).sqrt()
    first_memory_rms = first.state.memory.abs().square().float().mean(
        dim=(-3, -2, -1)
    ).sqrt()
    second_memory_rms = second.state.memory.abs().square().float().mean(
        dim=(-3, -2, -1)
    ).sqrt()

    diagnostics = {
        "zero_initial_memory_max_abs": float(initial.memory.abs().max()),
        "initial_root_postnorm_max_abs_error": float(
            (
                initial.hidden
                - transition._fixed_rms_normalize_hidden(roots)
            ).abs().max()
        ),
        "first_raw_memory_max_abs_error": float(
            (first.raw_updated_memory - first_expected_raw_memory).abs().max()
        ),
        "first_postnorm_memory_max_abs_error": float(
            (first.state.memory - first_expected_memory).abs().max()
        ),
        "first_raw_read_max_abs_error": float(
            (first.innovation_delta - first_expected_delta).abs().max()
        ),
        "first_postnorm_hidden_max_abs_error": float(
            (first.state.hidden - first_expected_hidden).abs().max()
        ),
        "second_raw_memory_max_abs_error": float(
            (second.raw_updated_memory - second_expected_raw_memory).abs().max()
        ),
        "second_postnorm_memory_max_abs_error": float(
            (second.state.memory - second_expected_memory).abs().max()
        ),
        "second_raw_read_max_abs_error": float(
            (second.innovation_delta - second_expected_delta).abs().max()
        ),
        "second_postnorm_hidden_max_abs_error": float(
            (second.state.hidden - second_expected_hidden).abs().max()
        ),
        "first_hidden_rms_min": float(first_hidden_rms.min()),
        "first_hidden_rms_max": float(first_hidden_rms.max()),
        "second_hidden_rms_min": float(second_hidden_rms.min()),
        "second_hidden_rms_max": float(second_hidden_rms.max()),
        "first_memory_rms_min": float(first_memory_rms.min()),
        "first_memory_rms_max": float(first_memory_rms.max()),
        "second_memory_rms_min": float(second_memory_rms.min()),
        "second_memory_rms_max": float(second_memory_rms.max()),
        "attached_hidden_boundary_gradient_norm": hidden_gradient_norm,
        "attached_memory_boundary_gradient_norm": memory_gradient_norm,
        "initial_write_count": initial.write_count,
        "first_write_count": first.state.write_count,
        "second_write_count": second.state.write_count,
        "output_weight_initial_std": float(
            transition.output.weight.detach().std()
        ),
        "parameter_count_joint_postnorm": base.parameter_count(model),
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise RuntimeError("non-finite joint-postnorm preflight diagnostic")
    formula_tolerance = 2e-6 * max(1.0, transition.width / 1344)
    for name in (
        "zero_initial_memory_max_abs",
        "initial_root_postnorm_max_abs_error",
        "first_raw_memory_max_abs_error",
        "first_postnorm_memory_max_abs_error",
        "first_raw_read_max_abs_error",
        "first_postnorm_hidden_max_abs_error",
        "second_raw_memory_max_abs_error",
        "second_postnorm_memory_max_abs_error",
        "second_raw_read_max_abs_error",
        "second_postnorm_hidden_max_abs_error",
    ):
        if diagnostics[name] > formula_tolerance:
            raise RuntimeError(f"joint-postnorm formula mismatch: {name}")
    for name in (
        "first_hidden_rms_min",
        "second_hidden_rms_min",
        "first_memory_rms_min",
        "second_memory_rms_min",
    ):
        if diagnostics[name] <= 0.99:
            raise RuntimeError(f"joint-postnorm RMS too small: {name}")
    for name in (
        "first_hidden_rms_max",
        "second_hidden_rms_max",
        "first_memory_rms_max",
        "second_memory_rms_max",
    ):
        if diagnostics[name] > 1.0:
            raise RuntimeError(f"joint-postnorm RMS exceeded one: {name}")
    if (initial.write_count, first.state.write_count, second.state.write_count) != (
        0,
        1,
        2,
    ):
        raise RuntimeError("joint-postnorm write count mismatch")
    if diagnostics["output_weight_initial_std"] <= 0.01:
        raise RuntimeError("complex readout retained the legacy tiny init")
    if diagnostics["parameter_count_joint_postnorm"] != base.EXPECTED_PARAMETERS:
        raise RuntimeError("joint-postnorm mode changed the parameter count")
    model.zero_grad(set_to_none=True)
    return {**values, **diagnostics}


def additional_interpretation_lines(initial_row, final_row, rows):
    return [
        "## Full-BPTT joint post-normalized recurrent carry",
        "",
        "Both predecessor fields remained attached. Each central use applied "
        "unitary memory transport plus a rank-one write, fixed non-affine "
        "complex RMS post-normalization to the complete updated memory before "
        "the raw q^dagger S/sqrt(key_dim) read, and fixed non-affine RMS "
        "post-normalization to hidden-plus-readout before carrying and decoding "
        "the successor. There was no realized-memory read denominator, "
        "write-count scaling, damping, gate, EMA, or auxiliary loss.",
        "",
        f"Final raw pre-clip gradient norm: {final_row['gradient_norm']:.6f}.",
        "",
        *H4_INTERPRETATION(initial_row, final_row, rows),
    ]


for module in (
    control,
    parent,
    branch,
    branch.parent,
    branch.h16,
    branch.h4,
    parent.h4,
    base,
):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.FUSE_BRANCH_MEMORY_KERNEL = False
base.PRE_NORMALIZE_QKV_INPUTS = False
base.NORMALIZE_INITIAL_RECURRENT_ROOT = True
base.NORMALIZE_ACCUMULATED_READS = False
base.POST_NORMALIZE_HIDDEN_READS = False
base.POST_NORMALIZE_MEMORY = False
base.POST_NORMALIZE_RECURRENT_STATE = False
base.EXPECTED_PARAMETERS = 13_215_008
base.REPORT_STEPS = REPORT_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_full_bptt_raw_read_joint_postnorm_residual_ce_only_"
    "stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Equal H1--H4 token CE on a fully attached state-dependent recurrence. "
    "Q/K/V consumed the post-normalized hidden directly. Memory used unitary "
    "transport plus an unmodified rank-one write, then fixed non-affine RMS "
    "post-normalization before a raw q^dagger S/sqrt(key_dim) read. Fixed "
    "non-affine RMS of hidden-plus-readout formed the actual next hidden. "
    "There was no detach, realized-memory read denominator, count scaling, "
    "damping, gate, EMA, or auxiliary loss."
)
parent.ORIGINAL_PREFLIGHT = preflight
base.preflight = parent.preflight
base.additional_final_checks = H4_FINAL_CHECKS
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
