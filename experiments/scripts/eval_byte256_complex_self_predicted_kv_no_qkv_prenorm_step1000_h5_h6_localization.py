"""Localize the native step-1000 exact-next-batch H5/H6 gradient event."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

import eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_one_update_forensics as audit
from rotlm.training.ha_skew_window import (
    _sparse_window_components,
    forward_sparse_complex_self_predicted_kv_window,
)


EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-"
    "step1000-h5-h6-localization"
)
RECORD_ROOT = Path("experiments/records") / EXPERIMENT_ID
OUTPUT_ROOT = Path("outputs/experiments") / EXPERIMENT_ID
HORIZONS = 16
SELECTED_HORIZONS = (5, 6)
ANCHOR_STRIDE = 16
GLOBAL_BATCH = 64
PHYSICAL_MICROBATCH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--skip-anchor-localization",
        action="store_true",
        help="retain row localization but skip the preregistered optional anchor split",
    )
    return parser.parse_args()


def named_parameters(model) -> tuple[list[str], list[torch.nn.Parameter]]:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return [item[0] for item in selected], [item[1] for item in selected]


def cpu_gradient_dict(
    names: list[str],
    parameters: list[torch.nn.Parameter],
    gradients: tuple[torch.Tensor | None, ...],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, parameter, gradient in zip(names, parameters, gradients, strict=True):
        result[name] = (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            if gradient is None
            else gradient.detach().float().cpu().clone()
        )
    return result


def selected_gradient_pair(
    model,
    window: torch.Tensor,
    *,
    row_ordinal: int | None = None,
    anchor_ordinal: int | None = None,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, float], dict[int, float]]:
    """Return exact global-objective H5/H6 contributions for one row scope.

    Every selected label retains its historical weight
    ``1 / (64 rows * 16 anchors * 16 horizons)``.  Consequently, summing
    physical microbatches, rows, or anchors reconstructs the full contribution.
    """
    device = next(model.parameters()).device
    names, parameters = named_parameters(model)
    model.zero_grad(set_to_none=True)
    output = forward_sparse_complex_self_predicted_kv_window(
        model,
        window.to(device),
        prefix_length=audit.base.CONTEXT,
        horizons=HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
        detach_cross_horizon_gradients=False,
    )
    anchors = output.token_ce.shape[1]
    if anchors != 16:
        raise RuntimeError(f"expected 16 anchors, observed {anchors}")
    if row_ordinal is not None and not 0 <= row_ordinal < output.token_ce.shape[0]:
        raise ValueError("row ordinal is outside the supplied physical microbatch")
    if anchor_ordinal is not None and row_ordinal is None:
        raise ValueError("anchor localization requires an explicitly selected row")

    losses: dict[int, torch.Tensor] = {}
    unweighted: dict[int, float] = {}
    weighted: dict[int, float] = {}
    label_weight = 1.0 / (GLOBAL_BATCH * anchors * HORIZONS)
    for horizon in SELECTED_HORIZONS:
        selected = output.token_ce[:, :, horizon - 1]
        if row_ordinal is not None:
            selected = selected[row_ordinal : row_ordinal + 1]
        if anchor_ordinal is not None:
            selected = selected[:, anchor_ordinal : anchor_ordinal + 1]
        losses[horizon] = selected.sum() * label_weight
        unweighted[horizon] = float(selected.detach().mean())
        weighted[horizon] = float(losses[horizon].detach())

    first = torch.autograd.grad(
        losses[5],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    first_cpu = cpu_gradient_dict(names, parameters, first)
    del first
    second = torch.autograd.grad(
        losses[6],
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    second_cpu = cpu_gradient_dict(names, parameters, second)
    del second, losses, output
    model.zero_grad(set_to_none=True)
    return {5: first_cpu, 6: second_cpu}, unweighted, weighted


VectorSpec = tuple[tuple[float, dict[str, torch.Tensor]], ...]


def vector_value(spec: VectorSpec, name: str) -> torch.Tensor:
    if len(spec) == 1 and spec[0][0] == 1.0:
        return spec[0][1][name]
    result = torch.zeros_like(spec[0][1][name])
    for coefficient, vector in spec:
        result.add_(vector[name], alpha=coefficient)
    return result


def vector_names(spec: VectorSpec) -> Iterable[str]:
    return spec[0][1].keys()


def vector_squared(spec: VectorSpec, group: str | None = None) -> float:
    total = 0.0
    for name in vector_names(spec):
        if group is not None and audit.parameter_group(name) != group:
            continue
        total += audit.squared_norm(vector_value(spec, name))
    return total


def vector_dot(left: VectorSpec, right: VectorSpec, group: str | None = None) -> float:
    total = 0.0
    for name in vector_names(left):
        if group is not None and audit.parameter_group(name) != group:
            continue
        total += float((vector_value(left, name) * vector_value(right, name)).sum())
    return total


def vector_max_abs(spec: VectorSpec, group: str | None = None) -> float:
    values = []
    for name in vector_names(spec):
        if group is not None and audit.parameter_group(name) != group:
            continue
        values.append(float(vector_value(spec, name).abs().amax()))
    return max(values, default=0.0)


def zero_vector_like(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in reference.items()}


def add_vector_(target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    for name in target:
        target[name].add_(source[name])


def relative_vector_error(
    reconstructed: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> float:
    error_squared = sum(
        audit.squared_norm(reconstructed[name] - reference[name])
        for name in reference
    )
    reference_squared = sum(audit.squared_norm(value) for value in reference.values())
    return math.sqrt(error_squared / max(reference_squared, 1e-30))


def component_specs(
    gradients: dict[int, dict[str, torch.Tensor]],
) -> dict[str, VectorSpec]:
    return {
        "h5": ((1.0, gradients[5]),),
        "h6": ((1.0, gradients[6]),),
        "h5+h6": ((1.0, gradients[5]), (1.0, gradients[6])),
    }


def component_values(values: dict[int, float]) -> dict[str, float]:
    return {
        "h5": values[5],
        "h6": values[6],
        "h5+h6": values[5] + values[6],
    }


def append_scope_rows(
    gradient_rows: list[dict[str, object]],
    alignment_rows: list[dict[str, object]],
    *,
    scope_level: str,
    scope_id: str,
    gradients: dict[int, dict[str, torch.Tensor]],
    unweighted_ce: dict[int, float],
    weighted_loss: dict[int, float],
    references: dict[str, VectorSpec],
    microbatch_index: int | None = None,
    global_row: int | None = None,
    local_row: int | None = None,
    anchor_ordinal: int | None = None,
) -> None:
    specs = component_specs(gradients)
    ce_values = {
        "h5": unweighted_ce[5],
        "h6": unweighted_ce[6],
        "h5+h6": 0.5 * (unweighted_ce[5] + unweighted_ce[6]),
    }
    weighted_values = component_values(weighted_loss)
    groups = sorted({audit.parameter_group(name) for name in gradients[5]})
    for label, spec in specs.items():
        for group in (None, *groups):
            squared = vector_squared(spec, group)
            gradient_rows.append(
                {
                    "scope_level": scope_level,
                    "scope_id": scope_id,
                    "microbatch_index": microbatch_index,
                    "global_row": global_row,
                    "local_row": local_row,
                    "anchor_ordinal": anchor_ordinal,
                    "loss_component": label,
                    "parameter_group": "all" if group is None else group,
                    "unweighted_ce_mean": ce_values[label],
                    "weighted_objective_contribution": weighted_values[label],
                    "gradient_l2": math.sqrt(max(squared, 0.0)),
                    "gradient_max_abs": vector_max_abs(spec, group),
                }
            )
        source_squared = vector_squared(spec)
        source_norm = math.sqrt(max(source_squared, 0.0))
        for reference_label, reference in references.items():
            reference_squared = vector_squared(reference)
            reference_norm = math.sqrt(max(reference_squared, 0.0))
            dot = vector_dot(spec, reference)
            projection_squared = dot * dot / max(reference_squared, 1e-30)
            orthogonal_residual_squared = max(source_squared - projection_squared, 0.0)
            alignment_rows.append(
                {
                    "scope_level": scope_level,
                    "scope_id": scope_id,
                    "microbatch_index": microbatch_index,
                    "global_row": global_row,
                    "local_row": local_row,
                    "anchor_ordinal": anchor_ordinal,
                    "source_component": label,
                    "reference_component": reference_label,
                    "source_gradient_l2": source_norm,
                    "reference_gradient_l2": reference_norm,
                    "dot": dot,
                    "cosine": dot / max(source_norm * reference_norm, 1e-30),
                    "projection_coefficient": dot / max(reference_squared, 1e-30),
                    "signed_projected_l2": dot / max(reference_norm, 1e-30),
                    "projection_squared_l2": projection_squared,
                    "orthogonal_residual_l2": math.sqrt(orthogonal_residual_squared),
                    "cosine_squared": dot * dot
                    / max(source_squared * reference_squared, 1e-30),
                }
            )


def tensor_rms(values: torch.Tensor) -> float:
    magnitude = values.detach().abs().float()
    return float(magnitude.square().mean().sqrt())


def tensor_max(values: torch.Tensor) -> float:
    return float(values.detach().abs().float().amax())


def per_anchor_rms(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.detach().abs().float()
    return magnitude.reshape(magnitude.shape[0], -1).square().mean(dim=1).sqrt()


@torch.no_grad()
def dominant_microbatch_forward_metrics(
    model,
    window: torch.Tensor,
    *,
    dominant_microbatch: int,
    dominant_global_row: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = next(model.parameters()).device
    micro_start = dominant_microbatch * PHYSICAL_MICROBATCH
    selected_cpu = window[micro_start : micro_start + PHYSICAL_MICROBATCH]
    selected = selected_cpu.to(device)
    (
        prefix_encoded,
        roots,
        _,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        selected,
        prefix_length=audit.base.CONTEXT,
        horizons=HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
    )
    transition = model.complex_self_prediction
    state = transition.initialize(roots)
    steps = []
    for _ in range(HORIZONS):
        step = transition(state)
        steps.append(step)
        state = step.state
    reads = torch.stack([step.read_hidden for step in steps], dim=2)
    decoded = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        reads,
        full_positions[: audit.base.CONTEXT],
        anchor_indices,
    )
    logits = model.token_logits(decoded)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    probabilities = logits.float().softmax(dim=-1)
    confidence, prediction = probabilities.max(dim=-1)

    row_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    eps = transition.post_norm_eps
    for local_row in range(PHYSICAL_MICROBATCH):
        global_row = micro_start + local_row
        for horizon in SELECTED_HORIZONS:
            step = steps[horizon - 1]
            z_denominator = (
                step.raw_full_hidden[local_row].float().square().mean(dim=-1) + eps
            ).sqrt()
            s_denominator = (
                step.raw_updated_memory[local_row]
                .abs()
                .float()
                .square()
                .mean(dim=(-3, -2, -1))
                + eps
            ).sqrt()
            tensors = {
                "p": step.preliminary_hidden[local_row],
                "innovation_w": step.innovation_memory[local_row],
                "s_raw": step.raw_updated_memory[local_row],
                "z_raw": step.raw_full_hidden[local_row],
                "stored_s": step.state.memory[local_row],
                "stored_z": step.state.hidden[local_row],
                "decoded_hidden": decoded[local_row, :, horizon - 1],
                "logits": logits[local_row, :, horizon - 1],
                "z_boundary_denominator": z_denominator,
                "s_boundary_denominator": s_denominator,
            }
            for tensor_name, tensor in tensors.items():
                rms_by_anchor = per_anchor_rms(tensor)
                row_rows.append(
                    {
                        "microbatch_index": dominant_microbatch,
                        "local_row": local_row,
                        "global_row": global_row,
                        "is_dominant_gradient_row": int(global_row == dominant_global_row),
                        "horizon": horizon,
                        "tensor": tensor_name,
                        "rms": tensor_rms(tensor),
                        "max_abs": tensor_max(tensor),
                        "anchor_rms_q50": float(torch.quantile(rms_by_anchor, 0.50)),
                        "anchor_rms_q90": float(torch.quantile(rms_by_anchor, 0.90)),
                        "anchor_rms_max": float(rms_by_anchor.max()),
                        "finite_fraction": float(
                            torch.isfinite(tensor.detach().abs()).float().mean()
                        ),
                        "row_nll": float(ce[local_row, :, horizon - 1].mean()),
                        "row_accuracy": float(
                            prediction[local_row, :, horizon - 1]
                            .eq(targets[local_row, :, horizon - 1])
                            .float()
                            .mean()
                        ),
                        "row_mean_max_probability": float(
                            confidence[local_row, :, horizon - 1].mean()
                        ),
                    }
                )
            per_anchor = {
                name: per_anchor_rms(value)
                for name, value in tensors.items()
            }
            for anchor in range(anchor_indices.numel()):
                anchor_position = int(anchor_indices[anchor])
                target_position = anchor_position + horizon
                token_id = int(targets[local_row, anchor, horizon - 1])
                input_id = int(selected_cpu[local_row, anchor_position])
                anchor_rows.append(
                    {
                        "microbatch_index": dominant_microbatch,
                        "local_row": local_row,
                        "global_row": global_row,
                        "is_dominant_gradient_row": int(global_row == dominant_global_row),
                        "anchor_ordinal": anchor,
                        "anchor_position": anchor_position,
                        "target_position": target_position,
                        "horizon": horizon,
                        "input_token_id": input_id,
                        "target_token_id": token_id,
                        "target_byte_hex": f"{token_id:02x}" if 0 <= token_id < 256 else "",
                        "predicted_token_id": int(prediction[local_row, anchor, horizon - 1]),
                        "correct": int(
                            prediction[local_row, anchor, horizon - 1]
                            == targets[local_row, anchor, horizon - 1]
                        ),
                        "nll": float(ce[local_row, anchor, horizon - 1]),
                        "max_probability": float(confidence[local_row, anchor, horizon - 1]),
                        **{
                            f"{name}_rms": float(values[anchor])
                            for name, values in per_anchor.items()
                        },
                    }
                )
    return row_rows, anchor_rows


def read_parent_reference() -> tuple[dict[int, float], float]:
    root = audit.RECORD_ROOT
    norms: dict[int, float] = {}
    with (root / "horizon_gradients.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            horizon = int(row["horizon"])
            if row["group"] == "all" and horizon in SELECTED_HORIZONS:
                norms[horizon] = float(row["contribution_gradient_l2"])
    cosine = math.nan
    with (root / "horizon_gradient_cosines.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["left_horizon"]) == 5 and int(row["right_horizon"]) == 6:
                cosine = float(row["cosine"])
                break
    if set(norms) != set(SELECTED_HORIZONS) or not math.isfinite(cosine):
        raise RuntimeError("parent H5/H6 reference metrics are incomplete")
    return norms, cosine


def main() -> None:
    args = parse_args()
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered audit")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    payload = audit.load_payload(audit.SOURCE_OUTPUT / "step1000.pt")
    window, _ = audit.exact_next_batch(payload)
    model = audit.checkpoint_model(payload, device)

    # Independently reproduce the parent full-batch H5/H6 vectors using its
    # registered physical-microbatch accumulation helper.
    full_runs = {
        horizon: audit.run_gradient(
            model,
            window,
            microbatch=PHYSICAL_MICROBATCH,
            detach=False,
            sqrt_reads=False,
            horizon=horizon,
            contribution_to_total=True,
        )
        for horizon in SELECTED_HORIZONS
    }
    full_gradients = {
        horizon: full_runs[horizon].gradients for horizon in SELECTED_HORIZONS
    }
    references = component_specs(full_gradients)
    parent_norms, parent_cosine = read_parent_reference()
    reproduced_norms = {
        horizon: math.sqrt(vector_squared(((1.0, full_gradients[horizon]),)))
        for horizon in SELECTED_HORIZONS
    }
    reproduced_cosine = vector_dot(references["h5"], references["h6"]) / max(
        reproduced_norms[5] * reproduced_norms[6], 1e-30
    )
    parent_relative_errors = {
        horizon: abs(reproduced_norms[horizon] - parent_norms[horizon])
        / max(parent_norms[horizon], 1e-30)
        for horizon in SELECTED_HORIZONS
    }

    gradient_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    append_scope_rows(
        gradient_rows,
        alignment_rows,
        scope_level="full_batch",
        scope_id="rows0:64",
        gradients=full_gradients,
        unweighted_ce={h: full_runs[h].loss for h in SELECTED_HORIZONS},
        weighted_loss={h: full_runs[h].loss / HORIZONS for h in SELECTED_HORIZONS},
        references=references,
    )

    micro_sum = {
        horizon: zero_vector_like(full_gradients[horizon])
        for horizon in SELECTED_HORIZONS
    }
    dominant_microbatch = -1
    dominant_microbatch_norm = -1.0
    for microbatch_index in range(GLOBAL_BATCH // PHYSICAL_MICROBATCH):
        start = microbatch_index * PHYSICAL_MICROBATCH
        micro_window = window[start : start + PHYSICAL_MICROBATCH]
        gradients, ce_values, weighted_values = selected_gradient_pair(
            model, micro_window
        )
        for horizon in SELECTED_HORIZONS:
            add_vector_(micro_sum[horizon], gradients[horizon])
        combined_norm = math.sqrt(vector_squared(component_specs(gradients)["h5+h6"]))
        if combined_norm > dominant_microbatch_norm:
            dominant_microbatch_norm = combined_norm
            dominant_microbatch = microbatch_index
        append_scope_rows(
            gradient_rows,
            alignment_rows,
            scope_level="physical_microbatch",
            scope_id=f"micro{microbatch_index}",
            gradients=gradients,
            unweighted_ce=ce_values,
            weighted_loss=weighted_values,
            references=references,
            microbatch_index=microbatch_index,
        )
        del gradients

    micro_errors = {
        horizon: relative_vector_error(micro_sum[horizon], full_gradients[horizon])
        for horizon in SELECTED_HORIZONS
    }
    if max(micro_errors.values()) >= 5e-4:
        raise RuntimeError(f"microbatch reconstruction failed: {micro_errors}")

    row_sum = {
        horizon: zero_vector_like(full_gradients[horizon])
        for horizon in SELECTED_HORIZONS
    }
    dominant_row = -1
    dominant_row_norm = -1.0
    dominant_start = dominant_microbatch * PHYSICAL_MICROBATCH
    dominant_micro_window = window[
        dominant_start : dominant_start + PHYSICAL_MICROBATCH
    ]
    for local_row in range(PHYSICAL_MICROBATCH):
        global_row = dominant_start + local_row
        gradients, ce_values, weighted_values = selected_gradient_pair(
            model,
            dominant_micro_window,
            row_ordinal=local_row,
        )
        for horizon in SELECTED_HORIZONS:
            add_vector_(row_sum[horizon], gradients[horizon])
        combined_norm = math.sqrt(vector_squared(component_specs(gradients)["h5+h6"]))
        if combined_norm > dominant_row_norm:
            dominant_row_norm = combined_norm
            dominant_row = global_row
        append_scope_rows(
            gradient_rows,
            alignment_rows,
            scope_level="row",
            scope_id=f"row{global_row}",
            gradients=gradients,
            unweighted_ce=ce_values,
            weighted_loss=weighted_values,
            references=references,
            microbatch_index=dominant_microbatch,
            global_row=global_row,
            local_row=local_row,
        )
        del gradients

    # Recompute the chosen microbatch reference independently from its rows.
    dominant_micro_gradients, _, _ = selected_gradient_pair(
        model,
        dominant_micro_window,
    )
    row_errors = {
        horizon: relative_vector_error(
            row_sum[horizon], dominant_micro_gradients[horizon]
        )
        for horizon in SELECTED_HORIZONS
    }
    if max(row_errors.values()) >= 5e-4:
        raise RuntimeError(f"row reconstruction failed: {row_errors}")
    del dominant_micro_gradients

    anchor_errors: dict[int, float] = {5: math.nan, 6: math.nan}
    dominant_anchor = -1
    dominant_anchor_norm = math.nan
    if not args.skip_anchor_localization:
        anchor_sum = {
            horizon: zero_vector_like(full_gradients[horizon])
            for horizon in SELECTED_HORIZONS
        }
        dominant_anchor_norm = -1.0
        for anchor_ordinal in range(16):
            gradients, ce_values, weighted_values = selected_gradient_pair(
                model,
                dominant_micro_window,
                row_ordinal=dominant_row - dominant_start,
                anchor_ordinal=anchor_ordinal,
            )
            for horizon in SELECTED_HORIZONS:
                add_vector_(anchor_sum[horizon], gradients[horizon])
            combined_norm = math.sqrt(
                vector_squared(component_specs(gradients)["h5+h6"])
            )
            if combined_norm > dominant_anchor_norm:
                dominant_anchor_norm = combined_norm
                dominant_anchor = anchor_ordinal
            append_scope_rows(
                gradient_rows,
                alignment_rows,
                scope_level="anchor",
                scope_id=f"row{dominant_row}.anchor{anchor_ordinal}",
                gradients=gradients,
                unweighted_ce=ce_values,
                weighted_loss=weighted_values,
                references=references,
                microbatch_index=dominant_microbatch,
                global_row=dominant_row,
                local_row=dominant_row - dominant_start,
                anchor_ordinal=anchor_ordinal,
            )
            del gradients
        dominant_row_gradients, _, _ = selected_gradient_pair(
            model,
            dominant_micro_window,
            row_ordinal=dominant_row - dominant_start,
        )
        anchor_errors = {
            horizon: relative_vector_error(
                anchor_sum[horizon], dominant_row_gradients[horizon]
            )
            for horizon in SELECTED_HORIZONS
        }
        if max(anchor_errors.values()) >= 5e-4:
            raise RuntimeError(f"anchor reconstruction failed: {anchor_errors}")
        del dominant_row_gradients, anchor_sum

    forward_rows, anchor_metadata = dominant_microbatch_forward_metrics(
        model,
        window,
        dominant_microbatch=dominant_microbatch,
        dominant_global_row=dominant_row,
    )

    for horizon in SELECTED_HORIZONS:
        gradient_rows.append(
            {
                "scope_level": "validation",
                "scope_id": "microbatch_sum_vs_full",
                "loss_component": f"h{horizon}",
                "parameter_group": "all",
                "gradient_l2": reproduced_norms[horizon],
                "reconstruction_relative_error": micro_errors[horizon],
                "parent_metric_relative_error": parent_relative_errors[horizon],
            }
        )
        gradient_rows.append(
            {
                "scope_level": "validation",
                "scope_id": "row_sum_vs_dominant_microbatch",
                "microbatch_index": dominant_microbatch,
                "loss_component": f"h{horizon}",
                "parameter_group": "all",
                "reconstruction_relative_error": row_errors[horizon],
            }
        )
        gradient_rows.append(
            {
                "scope_level": "validation",
                "scope_id": "anchor_sum_vs_dominant_row",
                "microbatch_index": dominant_microbatch,
                "global_row": dominant_row,
                "loss_component": f"h{horizon}",
                "parameter_group": "all",
                "reconstruction_relative_error": anchor_errors[horizon],
            }
        )

    audit.write_tsv(args.record_dir / "scope_gradients.tsv", gradient_rows)
    audit.write_tsv(args.record_dir / "gradient_alignments.tsv", alignment_rows)
    audit.write_tsv(args.record_dir / "row_forward_metrics.tsv", forward_rows)
    audit.write_tsv(args.record_dir / "anchor_metadata.tsv", anchor_metadata)

    # Compact evidence extraction for the Markdown interpretation.
    def all_gradient(scope_level: str, scope_id: str, component: str) -> dict[str, object]:
        return next(
            row
            for row in gradient_rows
            if row.get("scope_level") == scope_level
            and row.get("scope_id") == scope_id
            and row.get("loss_component") == component
            and row.get("parameter_group") == "all"
        )

    micro_summary = []
    for microbatch_index in range(4):
        row = all_gradient("physical_microbatch", f"micro{microbatch_index}", "h5+h6")
        micro_summary.append(
            (microbatch_index, float(row["gradient_l2"]), float(row["unweighted_ce_mean"]))
        )
    row_summary = []
    for global_row in range(dominant_start, dominant_start + PHYSICAL_MICROBATCH):
        row = all_gradient("row", f"row{global_row}", "h5+h6")
        alignment = next(
            item
            for item in alignment_rows
            if item["scope_level"] == "row"
            and item["scope_id"] == f"row{global_row}"
            and item["source_component"] == "h5+h6"
            and item["reference_component"] == "h5+h6"
        )
        row_summary.append(
            (
                global_row,
                float(row["gradient_l2"]),
                float(row["unweighted_ce_mean"]),
                float(alignment["cosine"]),
            )
        )
    top_rows = sorted(row_summary, key=lambda item: item[1], reverse=True)[:5]
    top_anchor_line = "Anchor localization was skipped."
    if dominant_anchor >= 0:
        anchor_gradient = all_gradient(
            "anchor", f"row{dominant_row}.anchor{dominant_anchor}", "h5+h6"
        )
        metadata = next(
            row
            for row in anchor_metadata
            if int(row["global_row"]) == dominant_row
            and int(row["anchor_ordinal"]) == dominant_anchor
            and int(row["horizon"]) == 5
        )
        top_anchor_line = (
            f"The largest selected-row anchor was ordinal `{dominant_anchor}` "
            f"(anchor position `{metadata['anchor_position']}`), with combined "
            f"gradient L2 `{float(anchor_gradient['gradient_l2']):.9g}`."
        )

    micro_lines = "\n".join(
        f"- micro {index}: combined L2 `{norm:.9g}`, mean H5/H6 CE `{ce:.9g}`"
        for index, norm, ce in micro_summary
    )
    row_lines = "\n".join(
        f"- row {index}: combined L2 `{norm:.9g}`, mean CE `{ce:.9g}`, "
        f"cosine to full H5+H6 `{cosine:.9g}`"
        for index, norm, ce, cosine in top_rows
    )
    interpretation = (
        "# Interpretation\n\n"
        "## Evidence boundary\n\n"
        "This is a frozen-weight localization of the native step-1000 exact-next "
        "batch. It does not reconstruct step 900, estimate event frequency, or "
        "assign a causal mechanism. Every gradient below already contains the "
        "historical `1/16` horizon weight and its exact global row/anchor weight.\n\n"
        "## Full-batch reproduction\n\n"
        f"H5 L2 was `{reproduced_norms[5]:.9g}` and H6 L2 was "
        f"`{reproduced_norms[6]:.9g}`; their cosine was "
        f"`{reproduced_cosine:.9g}`. Relative disagreement with the parent TSV "
        f"was `{parent_relative_errors[5]:.3g}` for H5, "
        f"`{parent_relative_errors[6]:.3g}` for H6, and "
        f"`{abs(reproduced_cosine-parent_cosine):.3g}` for their cosine.\n\n"
        "## Physical microbatches\n\n"
        f"{micro_lines}\n\n"
        f"The dominant physical microbatch was `{dominant_microbatch}`. The four "
        f"microbatch vectors reconstructed full H5/H6 with relative errors "
        f"`{micro_errors[5]:.3g}` / `{micro_errors[6]:.3g}`.\n\n"
        "## Rows within the dominant microbatch\n\n"
        f"{row_lines}\n\n"
        f"The dominant row was `{dominant_row}`. The 16 row vectors reconstructed "
        f"their microbatch with relative errors `{row_errors[5]:.3g}` / "
        f"`{row_errors[6]:.3g}`. A large row norm alone is not evidence that its "
        "loss or any one raw forward scale caused the full event; the TSV retains "
        "their separate measurements and directions.\n\n"
        "## Anchors and forward scales\n\n"
        f"{top_anchor_line} Anchor reconstruction relative errors were "
        f"`{anchor_errors[5]:.3g}` / `{anchor_errors[6]:.3g}`. "
        "`row_forward_metrics.tsv` contains P, W, Sraw, Zraw, stored carriers, "
        "decoder and logit scales for every row in the selected microbatch. "
        "`anchor_metadata.tsv` retains token IDs, losses, confidence, and the same "
        "scales per anchor. These are co-located observations, not a causal rank.\n"
    )
    (args.record_dir / "interpretation.md").write_text(interpretation)

    manifest_path = args.record_dir / "manifest.md"
    manifest = manifest_path.read_text()
    manifest = manifest.replace(
        "- State: preregistered; not yet executed",
        "- State: completed; native step-1000 H5/H6 localization executed",
    )
    manifest_path.write_text(manifest)
    print(
        f"H5={reproduced_norms[5]:.6f} H6={reproduced_norms[6]:.6f} "
        f"dominant_micro={dominant_microbatch} dominant_row={dominant_row} "
        f"dominant_anchor={dominant_anchor}",
        flush=True,
    )


if __name__ == "__main__":
    main()
