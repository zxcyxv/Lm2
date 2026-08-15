"""Audit final masked-H16 versus causally truncated-H4 equivalence.

This producer is intentionally read-only with respect to the checkpoint.  It
reconstructs the registered experiment wrapper, loads only the model state,
and compares both graphs on one deterministic training window.  It never
constructs or restores an optimizer and never reads the test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from rotlm.training.ha_skew_window import (
    forward_sparse_complex_self_predicted_kv_window,
)
import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as experiment


EXPERIMENT_ID = experiment.EXPERIMENT_ID
EXPECTED_STEP = experiment.TOTAL_STEPS
EXPECTED_PARAMETERS = 13_215_008
EXPECTED_TRAINABLE_PARAMETERS = 12_869_600
FULL_HORIZONS = experiment.MONITOR_HORIZONS
MASKED_HORIZONS = experiment.TRAIN_HORIZONS
ANCHOR_STRIDE = experiment.TRAIN_ANCHOR_STRIDE
AUDIT_SEED = experiment.base.SEED + 28_805
FORWARD_TOLERANCE = 5e-5
GRADIENT_MAX_RELATIVE_TOLERANCE = 2e-4
DEFAULT_RECORD = experiment.DEFAULT_RECORD
DEFAULT_CHECKPOINT = experiment.DEFAULT_OUTPUT / f"step{EXPECTED_STEP}.pt"


def scalar_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_metric(rows, metric: str, value, detail: str = "") -> None:
    rows.append(
        {
            "metric": metric,
            "value": scalar_text(value),
            "detail": detail,
        }
    )


def tensor_max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


def gradient_statistics(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    explicit_gradients: tuple[torch.Tensor | None, ...],
    truncated_gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, object]:
    explicit_l2_square = 0.0
    truncated_l2_square = 0.0
    error_l2_square = 0.0
    explicit_global_max = 0.0
    truncated_global_max = 0.0
    global_max_error = 0.0
    worst_parameter = ""
    explicit_unused = 0
    truncated_unused = 0
    finite = True
    parameter_rows = []

    for (name, parameter), explicit, truncated in zip(
        named_parameters,
        explicit_gradients,
        truncated_gradients,
        strict=True,
    ):
        explicit_unused += explicit is None
        truncated_unused += truncated is None
        if explicit is not None:
            finite = finite and bool(torch.isfinite(explicit).all())
            explicit_max = float(explicit.detach().abs().max())
            explicit_l2 = float(torch.linalg.vector_norm(explicit.detach()))
        else:
            explicit_max = 0.0
            explicit_l2 = 0.0
        if truncated is not None:
            finite = finite and bool(torch.isfinite(truncated).all())
            truncated_max = float(truncated.detach().abs().max())
            truncated_l2 = float(torch.linalg.vector_norm(truncated.detach()))
        else:
            truncated_max = 0.0
            truncated_l2 = 0.0

        if explicit is None and truncated is None:
            error_max = 0.0
            error_l2 = 0.0
        elif explicit is None:
            error_max = truncated_max
            error_l2 = truncated_l2
        elif truncated is None:
            error_max = explicit_max
            error_l2 = explicit_l2
        else:
            difference = explicit.detach() - truncated.detach()
            finite = finite and bool(torch.isfinite(difference).all())
            error_max = float(difference.abs().max())
            error_l2 = float(torch.linalg.vector_norm(difference))

        explicit_l2_square += explicit_l2 * explicit_l2
        truncated_l2_square += truncated_l2 * truncated_l2
        error_l2_square += error_l2 * error_l2
        explicit_global_max = max(explicit_global_max, explicit_max)
        truncated_global_max = max(truncated_global_max, truncated_max)
        if error_max > global_max_error:
            global_max_error = error_max
            worst_parameter = name
        parameter_rows.append(
            {
                "name": name,
                "elements": parameter.numel(),
                "explicit_unused": explicit is None,
                "truncated_unused": truncated is None,
                "explicit_max_abs": explicit_max,
                "truncated_max_abs": truncated_max,
                "max_abs_error": error_max,
                "l2_error": error_l2,
            }
        )

    explicit_l2 = math.sqrt(explicit_l2_square)
    truncated_l2 = math.sqrt(truncated_l2_square)
    error_l2 = math.sqrt(error_l2_square)
    return {
        "explicit_l2": explicit_l2,
        "truncated_l2": truncated_l2,
        "error_l2": error_l2,
        "global_l2_relative_error": error_l2 / max(explicit_l2, 1e-30),
        "explicit_global_max_abs": explicit_global_max,
        "truncated_global_max_abs": truncated_global_max,
        "global_max_abs_error": global_max_error,
        "global_max_relative_error": (
            global_max_error / max(explicit_global_max, 1e-30)
        ),
        "worst_parameter": worst_parameter,
        "explicit_unused": explicit_unused,
        "truncated_unused": truncated_unused,
        "finite": finite,
        "parameter_rows": parameter_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--expected-step", type=int, default=EXPECTED_STEP)
    parser.add_argument("--audit-seed", type=int, default=AUDIT_SEED)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(
            "the completed final checkpoint is required; audit was not run: "
            f"{args.checkpoint}"
        )
    if args.expected_step != EXPECTED_STEP:
        parser.error(
            f"this registered audit requires step {EXPECTED_STEP}, got "
            f"{args.expected_step}"
        )
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the registered system-Python model")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(experiment.base.SEED)
    torch.cuda.manual_seed_all(experiment.base.SEED)

    checkpoint_sha256 = sha256_file(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_step = int(checkpoint.get("step", -1))
    if checkpoint_step != args.expected_step:
        raise RuntimeError(
            f"checkpoint step {checkpoint_step} != expected {args.expected_step}"
        )
    if "model" not in checkpoint:
        raise RuntimeError("checkpoint has no model state")
    checkpoint_config = checkpoint.get("config", {})

    model = experiment.base.make_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda().eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = sum(
        parameter.numel() for _, parameter in named_parameters
    )

    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("registered model parameters are not strict float32")

    training = experiment.base.memmap("train")
    full_length = experiment.base.CONTEXT + FULL_HORIZONS
    generator = torch.Generator().manual_seed(args.audit_seed)
    window_start = int(
        torch.randint(
            0,
            len(training) - full_length,
            (1,),
            generator=generator,
        )[0]
    )
    full_window = experiment.base.windows_of_length(
        training,
        torch.tensor([window_start]),
        full_length,
    )
    truncated_window = full_window[:, : experiment.base.CONTEXT + MASKED_HORIZONS]
    window_bytes = (
        full_window.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
        .numpy()
        .tobytes()
    )
    window_sha256 = hashlib.sha256(window_bytes).hexdigest()

    full = forward_sparse_complex_self_predicted_kv_window(
        model,
        full_window,
        prefix_length=experiment.base.CONTEXT,
        horizons=FULL_HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
    )
    truncated = forward_sparse_complex_self_predicted_kv_window(
        model,
        truncated_window,
        prefix_length=experiment.base.CONTEXT,
        horizons=MASKED_HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
    )

    anchors_equal = torch.equal(full.anchor_indices, truncated.anchor_indices)
    targets_equal = torch.equal(
        full.targets[..., :MASKED_HORIZONS],
        truncated.targets,
    )
    forward_pairs = {
        "logits": (full.logits[..., :MASKED_HORIZONS, :], truncated.logits),
        "decoded_hidden": (
            full.decoded_hidden[..., :MASKED_HORIZONS, :],
            truncated.decoded_hidden,
        ),
        "full_states": (
            full.full_states[..., :MASKED_HORIZONS, :],
            truncated.full_states,
        ),
        "read_states": (
            full.read_states[..., :MASKED_HORIZONS, :],
            truncated.read_states,
        ),
        "preliminary_states": (
            full.preliminary_states[..., :MASKED_HORIZONS, :],
            truncated.preliminary_states,
        ),
        "rotated_states": (
            full.rotated_states[..., :MASKED_HORIZONS, :],
            truncated.rotated_states,
        ),
        "innovation_deltas": (
            full.innovation_deltas[..., :MASKED_HORIZONS, :],
            truncated.innovation_deltas,
        ),
    }
    forward_errors = {
        name: tensor_max_abs_error(left, right)
        for name, (left, right) in forward_pairs.items()
    }
    forward_worst_field = max(forward_errors, key=forward_errors.__getitem__)
    forward_max_error = forward_errors[forward_worst_field]

    masked_full_loss = F.cross_entropy(
        full.logits[..., :MASKED_HORIZONS, :]
        .reshape(-1, experiment.base.VOCABULARY)
        .float(),
        full.targets[..., :MASKED_HORIZONS].reshape(-1),
    )
    truncated_loss = truncated.token_ce.mean()
    loss_abs_error = float(
        (masked_full_loss.detach() - truncated_loss.detach()).abs()
    )
    loss_relative_error = loss_abs_error / max(
        abs(float(masked_full_loss.detach())),
        1e-30,
    )

    full_logit_gradient = torch.autograd.grad(
        masked_full_loss,
        full.logits,
        retain_graph=True,
    )[0]
    supervised_logit_gradient_l2 = float(
        torch.linalg.vector_norm(
            full_logit_gradient[..., :MASKED_HORIZONS, :]
        )
    )
    heldout_logit_gradient = full_logit_gradient[..., MASKED_HORIZONS:, :]
    heldout_logit_gradient_max_abs = float(heldout_logit_gradient.abs().max())
    heldout_logit_gradient_nonzero = int(
        torch.count_nonzero(heldout_logit_gradient)
    )

    parameter_tensors = [parameter for _, parameter in named_parameters]
    explicit_gradients = torch.autograd.grad(
        masked_full_loss,
        parameter_tensors,
        allow_unused=True,
    )
    truncated_gradients = torch.autograd.grad(
        truncated_loss,
        parameter_tensors,
        allow_unused=True,
    )
    gradient = gradient_statistics(
        named_parameters,
        explicit_gradients,
        truncated_gradients,
    )
    parameter_grad_buffers_untouched = all(
        parameter.grad is None for _, parameter in named_parameters
    )

    checks = {
        "checkpoint_step_matches": checkpoint_step == EXPECTED_STEP,
        "checkpoint_experiment_id_matches": (
            checkpoint_config.get("experiment_id") == EXPERIMENT_ID
        ),
        "checkpoint_objective_matches": (
            checkpoint_config.get("objective")
            == experiment.base.registered_objective_name()
        ),
        "parameter_count_matches": total_parameters == EXPECTED_PARAMETERS,
        "trainable_parameter_count_matches": (
            trainable_parameters == EXPECTED_TRAINABLE_PARAMETERS
        ),
        "anchor_indices_equal": anchors_equal,
        "h1_h4_targets_equal": targets_equal,
        "first4_forward_within_tolerance": (
            forward_max_error < FORWARD_TOLERANCE
        ),
        "selected_loss_matches": loss_abs_error <= 1e-7,
        "supervised_logit_gradient_nonzero": supervised_logit_gradient_l2 > 0.0,
        "heldout_h5_h16_logit_gradient_exact_zero": (
            heldout_logit_gradient_max_abs == 0.0
            and heldout_logit_gradient_nonzero == 0
        ),
        "all_trainable_parameter_gradients_present": (
            gradient["explicit_unused"] == 0
            and gradient["truncated_unused"] == 0
        ),
        "all_parameter_gradients_finite": bool(gradient["finite"]),
        "parameter_gradient_max_relative_within_tolerance": (
            gradient["global_max_relative_error"]
            < GRADIENT_MAX_RELATIVE_TOLERANCE
        ),
        "parameter_grad_buffers_untouched": parameter_grad_buffers_untouched,
    }
    audit_passed = all(checks.values())

    rows = []
    append_metric(rows, "experiment_id", EXPERIMENT_ID)
    append_metric(rows, "checkpoint_path", args.checkpoint)
    append_metric(rows, "checkpoint_sha256", checkpoint_sha256)
    append_metric(rows, "checkpoint_step", checkpoint_step)
    append_metric(
        rows,
        "checkpoint_config_experiment_id",
        checkpoint_config.get("experiment_id", ""),
    )
    append_metric(
        rows,
        "checkpoint_config_objective",
        checkpoint_config.get("objective", ""),
    )
    append_metric(rows, "split", "train")
    append_metric(rows, "test_split_read", False)
    append_metric(rows, "precision", "strict_float32_no_tf32")
    append_metric(rows, "python_executable", sys.executable)
    append_metric(rows, "torch_version", torch.__version__)
    append_metric(rows, "gpu", torch.cuda.get_device_name())
    append_metric(rows, "model_mode", "eval")
    append_metric(rows, "optimizer_constructed_or_loaded", False)
    append_metric(rows, "audit_seed", args.audit_seed)
    append_metric(rows, "window_start", window_start)
    append_metric(rows, "window_length", full_length)
    append_metric(rows, "window_sha256", window_sha256)
    append_metric(rows, "batch", 1)
    append_metric(rows, "context", experiment.base.CONTEXT)
    append_metric(rows, "explicit_horizons", FULL_HORIZONS)
    append_metric(rows, "selected_horizons", MASKED_HORIZONS)
    append_metric(rows, "anchor_stride", ANCHOR_STRIDE)
    append_metric(rows, "anchor_count", full.anchor_indices.numel())
    append_metric(rows, "selected_label_count", truncated.targets.numel())
    append_metric(rows, "parameters", total_parameters)
    append_metric(rows, "trainable_parameters", trainable_parameters)
    append_metric(rows, "trainable_parameter_tensors", len(named_parameters))
    append_metric(rows, "anchors_equal", anchors_equal)
    append_metric(rows, "h1_h4_targets_equal", targets_equal)
    for name, error in forward_errors.items():
        append_metric(rows, f"first4_{name}_max_abs_error", error)
    append_metric(rows, "first4_forward_max_abs_error", forward_max_error)
    append_metric(rows, "first4_forward_worst_field", forward_worst_field)
    append_metric(rows, "forward_tolerance", FORWARD_TOLERANCE)
    append_metric(rows, "masked_explicit_h16_loss", float(masked_full_loss.detach()))
    append_metric(rows, "causally_truncated_h4_loss", float(truncated_loss.detach()))
    append_metric(rows, "selected_loss_abs_error", loss_abs_error)
    append_metric(rows, "selected_loss_relative_error", loss_relative_error)
    append_metric(
        rows,
        "h1_h4_logit_gradient_l2",
        supervised_logit_gradient_l2,
    )
    append_metric(
        rows,
        "h5_h16_logit_gradient_max_abs",
        heldout_logit_gradient_max_abs,
    )
    append_metric(
        rows,
        "h5_h16_logit_gradient_nonzero_elements",
        heldout_logit_gradient_nonzero,
    )
    append_metric(
        rows,
        "explicit_trainable_gradient_unused_tensors",
        gradient["explicit_unused"],
    )
    append_metric(
        rows,
        "truncated_trainable_gradient_unused_tensors",
        gradient["truncated_unused"],
    )
    append_metric(rows, "all_parameter_gradients_finite", gradient["finite"])
    append_metric(rows, "explicit_gradient_global_l2", gradient["explicit_l2"])
    append_metric(rows, "truncated_gradient_global_l2", gradient["truncated_l2"])
    append_metric(rows, "gradient_global_l2_error", gradient["error_l2"])
    append_metric(
        rows,
        "gradient_global_l2_relative_error",
        gradient["global_l2_relative_error"],
    )
    append_metric(
        rows,
        "explicit_gradient_global_max_abs",
        gradient["explicit_global_max_abs"],
    )
    append_metric(
        rows,
        "truncated_gradient_global_max_abs",
        gradient["truncated_global_max_abs"],
    )
    append_metric(
        rows,
        "gradient_global_max_abs_error",
        gradient["global_max_abs_error"],
    )
    append_metric(
        rows,
        "gradient_max_relative_error_normalized_by_explicit_global_max",
        gradient["global_max_relative_error"],
    )
    append_metric(rows, "gradient_worst_parameter", gradient["worst_parameter"])
    append_metric(
        rows,
        "gradient_max_relative_tolerance",
        GRADIENT_MAX_RELATIVE_TOLERANCE,
    )
    append_metric(
        rows,
        "parameter_grad_buffers_untouched",
        parameter_grad_buffers_untouched,
    )
    for row in gradient["parameter_rows"]:
        detail = (
            f"elements={row['elements']};"
            f"explicit_unused={scalar_text(row['explicit_unused'])};"
            f"truncated_unused={scalar_text(row['truncated_unused'])};"
            f"explicit_max_abs={scalar_text(row['explicit_max_abs'])};"
            f"truncated_max_abs={scalar_text(row['truncated_max_abs'])};"
            f"l2_error={scalar_text(row['l2_error'])}"
        )
        append_metric(
            rows,
            "trainable_parameter_gradient_max_abs_error",
            row["max_abs_error"],
            detail=f"name={row['name']};{detail}",
        )
    for name, passed in checks.items():
        append_metric(rows, f"check_{name}", passed)
    append_metric(rows, "audit_passed", audit_passed)

    tsv_buffer = io.StringIO()
    writer = csv.DictWriter(
        tsv_buffer,
        fieldnames=("metric", "value", "detail"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

    status = "PASS" if audit_passed else "FAIL"
    check_lines = [
        f"| `{name}` | {'pass' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    ]
    markdown = "\n".join(
        [
            "# Final masked-H16 / truncated-H4 equivalence audit",
            "",
            f"Overall status: **{status}**.",
            "",
            "The final model checkpoint was reconstructed through the registered "
            "system-Python experiment wrapper. On one deterministic batch-1 "
            "training window, the explicit H16/stride-4 graph received CE only "
            "from H1--H4 and was compared with a causally truncated H4/stride-4 "
            "graph. Both arms used the same prefix root, anchors, targets, and "
            "checkpoint parameters.",
            "",
            "## Primary measurements",
            "",
            "| Measurement | Value |",
            "|---|---:|",
            f"| Checkpoint step | {checkpoint_step} |",
            f"| Total / trainable parameters | {total_parameters} / {trainable_parameters} |",
            f"| First-four forward max absolute error | {forward_max_error:.9g} |",
            f"| Selected-loss absolute error | {loss_abs_error:.9g} |",
            f"| H5--H16 logit-gradient max absolute value | {heldout_logit_gradient_max_abs:.9g} |",
            f"| H5--H16 nonzero logit-gradient elements | {heldout_logit_gradient_nonzero} |",
            f"| All-parameter gradient max absolute error | {gradient['global_max_abs_error']:.9g} |",
            f"| All-parameter gradient global L2 relative error | {gradient['global_l2_relative_error']:.9g} |",
            f"| Max error / explicit global max gradient | {gradient['global_max_relative_error']:.9g} |",
            f"| Worst parameter | `{gradient['worst_parameter']}` |",
            "",
            "## Registered checks",
            "",
            "| Check | Result |",
            "|---|:---:|",
            *check_lines,
            "",
            "The registered first-four forward tolerance is `5e-5`; the "
            "registered max-relative all-parameter gradient tolerance is "
            "`2e-4`. The global L2-relative gradient error is reported as an "
            "additional aggregate, without changing the preregistered gate.",
            "",
            "No optimizer was constructed or restored, parameter `.grad` buffers "
            "remained untouched, and only the training byte split was read. The "
            "test split remained unmaterialized and unread. Per-parameter maximum "
            "absolute gradient errors are retained in the TSV.",
            "",
        ]
    )

    args.record_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(args.record_dir / "final_mask_equivalence.tsv", tsv_buffer.getvalue())
    atomic_write(args.record_dir / "final_mask_equivalence.md", markdown)
    print(
        f"mask-equivalence {status.lower()} step={checkpoint_step} "
        f"forward={forward_max_error:.3e} loss={loss_abs_error:.3e} "
        f"tail_grad={heldout_logit_gradient_max_abs:.3e} "
        f"grad_max_rel={gradient['global_max_relative_error']:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
