"""Audit one exact update from the native no-QKV-pre-norm step-1000 state."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from rotlm.latent_diagnostics import jacobian_spectral_norm_power_iteration
from rotlm.models.complex_self_prediction import ComplexMemoryState
from rotlm.training.ha_skew_window import (
    _sparse_window_components,
    forward_sparse_complex_self_predicted_kv_window,
)
from train_byte256_simplex_tied_self_composition_h1_ce_13m import (
    memmap,
    sampled_windows,
)
from train_k1_decoder_inverse_ablation_13m import CLIP_NORM, lr_at
import train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_h16_attached_stride16_ce_only_13m as source


EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-"
    "step1000-one-update-forensics"
)
RECORD_ROOT = Path("experiments/records") / EXPERIMENT_ID
OUTPUT_ROOT = Path("outputs/experiments") / EXPERIMENT_ID
SOURCE_OUTPUT = Path("outputs/experiments") / source.EXPERIMENT_ID
HORIZONS = 16
ANCHOR_STRIDE = 16
MICROBATCH = 16

base = source.base


@dataclass
class GradientRun:
    loss: float
    horizon_losses: tuple[float, ...]
    gradients: dict[str, torch.Tensor]
    forward_rows: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    parser.add_argument("--trace-examples", type=int, default=16)
    parser.add_argument("--jacobian-iterations", type=int, default=5)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_model(payload: dict[str, object], device: torch.device):
    model = base.make_model()
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).train()


def exact_next_batch(payload: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.set_state(payload["data_generator_state"])
    state_before = generator.get_state().clone()
    window = sampled_windows(
        memmap("train"),
        64,
        generator,
        base.CONTEXT + HORIZONS,
    )
    if tuple(window.shape) != (64, base.CONTEXT + HORIZONS):
        raise RuntimeError("native checkpoint generated an unexpected batch")
    return window.cpu(), state_before


def trainable_parameters(model) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_group(name: str) -> str:
    prefix = "complex_self_prediction."
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        if rest.startswith("query."):
            return "central.query"
        if rest.startswith("key."):
            return "central.key"
        if rest.startswith("value."):
            return "central.value"
        if rest.startswith("output."):
            return "central.output"
        if rest == "hidden_phase":
            return "central.hidden_phase"
        if rest == "memory_phase":
            return "central.memory_phase"
        return "central.other"
    if name.startswith("encoder.blocks."):
        parts = name.split(".")
        block = parts[2]
        family = parts[3]
        if family == "attn" and len(parts) > 4:
            family = f"attn.{parts[4]}"
        elif family == "ffn":
            family = "ffn"
        return f"revblock{block}.{family}"
    if name.startswith("encoder.embed"):
        return "embedding"
    if name.startswith("encoder.head_norm"):
        return "head_norm"
    return "other"


def squared_norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(tensor.detach().float().square().sum())


def vector_dot(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    group: str | None = None,
) -> float:
    result = 0.0
    for name in left:
        if group is not None and parameter_group(name) != group:
            continue
        result += float((left[name].float() * right[name].float()).sum())
    return result


def gradient_norm(gradients: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(squared_norm(value) for value in gradients.values()))


def _rms(values: torch.Tensor) -> float:
    if values.is_complex():
        return float(values.detach().abs().square().float().mean().sqrt())
    return float(values.detach().float().square().mean().sqrt())


def _max_abs(values: torch.Tensor) -> float:
    return float(values.detach().abs().float().amax())


def _horizon_forward_rows(output, condition: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in range(HORIZONS):
        values = {
            "p": output.preliminary_states[:, :, horizon],
            "stored_z": output.full_states[:, :, horizon],
            "innovation_delta": output.innovation_deltas[:, :, horizon],
            "decoded_hidden": output.decoded_hidden[:, :, horizon],
            "logits": output.logits[:, :, horizon],
        }
        for tensor_name, tensor in values.items():
            rows.append(
                {
                    "condition": condition,
                    "horizon": horizon + 1,
                    "tensor": tensor_name,
                    "rms": _rms(tensor),
                    "max_abs": _max_abs(tensor),
                }
            )
        rows.extend(
            (
                {
                    "condition": condition,
                    "horizon": horizon + 1,
                    "tensor": "stored_memory",
                    "rms": float(
                        output.memory_energy[:, :, horizon]
                        .detach()
                        .float()
                        .mean()
                        .sqrt()
                    ),
                    "max_abs": float(
                        output.memory_energy[:, :, horizon]
                        .detach()
                        .float()
                        .sqrt()
                        .amax()
                    ),
                },
                {
                    "condition": condition,
                    "horizon": horizon + 1,
                    "tensor": "innovation_memory",
                    "rms": float(
                        output.innovation_energy[:, :, horizon]
                        .detach()
                        .float()
                        .mean()
                        .sqrt()
                    ),
                    "max_abs": float(
                        output.innovation_energy[:, :, horizon]
                        .detach()
                        .float()
                        .sqrt()
                        .amax()
                    ),
                },
            )
        )
    return rows


def summarized_forward_rows(
    rows: list[dict[str, object]],
    *,
    phase: str,
    condition: str,
) -> list[dict[str, object]]:
    buckets: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(
            (int(row["horizon"]), str(row["tensor"])), []
        ).append(row)
    result = []
    for (horizon, tensor), selected in sorted(buckets.items()):
        result.append(
            {
                "phase": phase,
                "condition": condition,
                "horizon": horizon,
                "group": f"forward.{tensor}",
                "loss": math.nan,
                "gradient_l2": math.nan,
                "gradient_ratio_to_control": math.nan,
                "forward_rms": math.sqrt(
                    sum(float(row["rms"]) ** 2 for row in selected)
                    / len(selected)
                ),
                "forward_max_abs": max(
                    float(row["max_abs"]) for row in selected
                ),
            }
        )
    return result


def run_gradient(
    model,
    window: torch.Tensor,
    *,
    microbatch: int,
    detach: bool,
    sqrt_reads: bool,
    horizon: int | None = None,
    contribution_to_total: bool = False,
    collect_forward: bool = False,
    condition: str = "",
) -> GradientRun:
    transition = model.complex_self_prediction
    old_scaling = transition.normalize_accumulated_reads
    transition.normalize_accumulated_reads = bool(sqrt_reads)
    model.zero_grad(set_to_none=True)
    losses = torch.zeros(HORIZONS, dtype=torch.float64)
    forward_rows: list[dict[str, object]] = []
    micro_windows = window.split(microbatch)
    try:
        for micro_index, micro_cpu in enumerate(micro_windows):
            micro = micro_cpu.to(next(model.parameters()).device)
            output = forward_sparse_complex_self_predicted_kv_window(
                model,
                micro,
                prefix_length=base.CONTEXT,
                horizons=HORIZONS,
                anchor_stride=ANCHOR_STRIDE,
                detach_cross_horizon_gradients=detach,
            )
            horizon_means = output.token_ce.mean(dim=(0, 1))
            losses += horizon_means.detach().double().cpu() / len(micro_windows)
            selected = (
                output.token_ce.mean()
                if horizon is None
                else output.token_ce[:, :, horizon - 1].mean()
            )
            if contribution_to_total and horizon is not None:
                selected = selected / HORIZONS
            (selected / len(micro_windows)).backward()
            if collect_forward:
                forward_rows.extend(
                    _horizon_forward_rows(
                        output,
                        f"{condition}.micro{micro_index}",
                    )
                )
            del selected, horizon_means, output, micro
        gradients = {
            name: (
                torch.zeros_like(parameter, device="cpu")
                if parameter.grad is None
                else parameter.grad.detach().float().cpu().clone()
            )
            for name, parameter in trainable_parameters(model).items()
        }
        return GradientRun(
            loss=float(losses.mean() if horizon is None else losses[horizon - 1]),
            horizon_losses=tuple(float(value) for value in losses),
            gradients=gradients,
            forward_rows=forward_rows,
        )
    finally:
        transition.normalize_accumulated_reads = old_scaling
        model.zero_grad(set_to_none=True)


def spectral_norm_estimate(weight: torch.Tensor, iterations: int = 12) -> float:
    if weight.ndim != 2 or min(weight.shape) == 0:
        return math.nan
    matrix = weight.detach().float()
    vector = torch.ones(matrix.shape[1], device=matrix.device)
    vector /= vector.norm()
    for _ in range(iterations):
        left = matrix @ vector
        left /= left.norm().clamp_min(1e-12)
        vector = matrix.T @ left
        vector /= vector.norm().clamp_min(1e-12)
    return float((matrix @ vector).norm())


def parameter_gradient_rows(model, gradients: dict[str, torch.Tensor]):
    parameters = trainable_parameters(model)
    total_squared = sum(squared_norm(value) for value in gradients.values())
    rows: list[dict[str, object]] = []
    groups: dict[str, list[str]] = {}
    for name in parameters:
        groups.setdefault(parameter_group(name), []).append(name)
    for name, parameter in parameters.items():
        weight = parameter.detach().float()
        gradient = gradients[name].float()
        weight_l2 = float(weight.norm())
        grad_l2 = float(gradient.norm())
        rows.append(
            {
                "scope": "parameter",
                "name": name,
                "group": parameter_group(name),
                "numel": parameter.numel(),
                "weight_l2": weight_l2,
                "weight_rms": float(weight.square().mean().sqrt()),
                "weight_max_abs": float(weight.abs().amax()),
                "weight_spectral_norm_estimate": spectral_norm_estimate(weight),
                "gradient_l2": grad_l2,
                "gradient_rms": float(gradient.square().mean().sqrt()),
                "gradient_max_abs": float(gradient.abs().amax()),
                "gradient_to_weight_l2": grad_l2 / max(weight_l2, 1e-30),
                "squared_global_fraction": squared_norm(gradient)
                / max(total_squared, 1e-30),
            }
        )
    for group, names in groups.items():
        numel = sum(parameters[name].numel() for name in names)
        weight_squared = sum(
            squared_norm(parameters[name].detach()) for name in names
        )
        grad_squared = sum(squared_norm(gradients[name]) for name in names)
        grad_max = max(float(gradients[name].abs().amax()) for name in names)
        weight_max = max(
            float(parameters[name].detach().abs().amax()) for name in names
        )
        rows.append(
            {
                "scope": "group",
                "name": group,
                "group": group,
                "numel": numel,
                "weight_l2": math.sqrt(weight_squared),
                "weight_rms": math.sqrt(weight_squared / numel),
                "weight_max_abs": weight_max,
                "weight_spectral_norm_estimate": math.nan,
                "gradient_l2": math.sqrt(grad_squared),
                "gradient_rms": math.sqrt(grad_squared / numel),
                "gradient_max_abs": grad_max,
                "gradient_to_weight_l2": math.sqrt(grad_squared)
                / max(math.sqrt(weight_squared), 1e-30),
                "squared_global_fraction": grad_squared
                / max(total_squared, 1e-30),
            }
        )
    rows.append(
        {
            "scope": "global",
            "name": "all",
            "group": "all",
            "numel": sum(parameter.numel() for parameter in parameters.values()),
            "weight_l2": math.sqrt(
                sum(squared_norm(parameter.detach()) for parameter in parameters.values())
            ),
            "weight_rms": math.nan,
            "weight_max_abs": max(
                float(parameter.detach().abs().amax())
                for parameter in parameters.values()
            ),
            "weight_spectral_norm_estimate": math.nan,
            "gradient_l2": math.sqrt(total_squared),
            "gradient_rms": math.nan,
            "gradient_max_abs": max(
                float(value.abs().amax()) for value in gradients.values()
            ),
            "gradient_to_weight_l2": math.nan,
            "squared_global_fraction": 1.0,
        }
    )
    return rows


def parameter_update_rows(
    pre_parameters: dict[str, torch.Tensor],
    raw_gradients: dict[str, torch.Tensor],
    clipped_gradients: dict[str, torch.Tensor],
    post_parameters: dict[str, torch.Tensor],
):
    rows: list[dict[str, object]] = []
    groups: dict[str, list[str]] = {}
    for name in pre_parameters:
        groups.setdefault(parameter_group(name), []).append(name)

    def one_row(scope: str, label: str, names: list[str]):
        numel = sum(pre_parameters[name].numel() for name in names)
        weight_sq = sum(squared_norm(pre_parameters[name]) for name in names)
        raw_sq = sum(squared_norm(raw_gradients[name]) for name in names)
        clipped_sq = sum(
            squared_norm(clipped_gradients[name]) for name in names
        )
        delta_sq = sum(
            squared_norm(post_parameters[name] - pre_parameters[name])
            for name in names
        )
        delta_max = max(
            float((post_parameters[name] - pre_parameters[name]).abs().amax())
            for name in names
        )
        return {
            "scope": scope,
            "name": label,
            "group": label if scope != "parameter" else parameter_group(label),
            "numel": numel,
            "weight_l2": math.sqrt(weight_sq),
            "raw_gradient_l2": math.sqrt(raw_sq),
            "raw_gradient_rms": math.sqrt(raw_sq / numel),
            "clipped_gradient_l2": math.sqrt(clipped_sq),
            "clipped_gradient_rms": math.sqrt(clipped_sq / numel),
            "adam_parameter_delta_l2": math.sqrt(delta_sq),
            "adam_parameter_delta_rms": math.sqrt(delta_sq / numel),
            "adam_parameter_delta_max_abs": delta_max,
            "update_to_weight_l2": math.sqrt(delta_sq)
            / max(math.sqrt(weight_sq), 1e-30),
        }

    for name in pre_parameters:
        rows.append(one_row("parameter", name, [name]))
    for group, names in groups.items():
        rows.append(one_row("group", group, names))
    rows.append(one_row("global", "all", list(pre_parameters)))
    return rows


def horizon_gradient_rows(
    horizon_runs: list[GradientRun],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups = sorted(
        {parameter_group(name) for name in horizon_runs[0].gradients}
    )
    group_scopes: list[str | None] = [None, *groups]
    rows: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    for horizon, run in enumerate(horizon_runs, 1):
        for group in group_scopes:
            selected = {
                name: value
                for name, value in run.gradients.items()
                if group is None or parameter_group(name) == group
            }
            numel = sum(value.numel() for value in selected.values())
            norm = gradient_norm(selected)
            rows.append(
                {
                    "horizon": horizon,
                    "group": "all" if group is None else group,
                    "horizon_ce": run.loss,
                    "loss_weight_in_total": 1.0 / HORIZONS,
                    "contribution_gradient_l2": norm,
                    "contribution_gradient_rms": norm / math.sqrt(max(numel, 1)),
                    "contribution_gradient_max_abs": max(
                        (float(value.abs().amax()) for value in selected.values()),
                        default=0.0,
                    ),
                }
            )
    # The registered pairwise cosine matrix is global. Groupwise coherent/RSS
    # summaries are still exact, without redundantly repeating the 256 large
    # vector dot products for every disjoint group.
    group = None
    label = "all"
    norms = [
        math.sqrt(max(vector_dot(run.gradients, run.gradients), 0.0))
        for run in horizon_runs
    ]
    for left in range(HORIZONS):
        for right in range(HORIZONS):
            dot = vector_dot(
                horizon_runs[left].gradients,
                horizon_runs[right].gradients,
            )
            cosine_rows.append(
                {
                    "group": label,
                    "left_horizon": left + 1,
                    "right_horizon": right + 1,
                    "dot": dot,
                    "cosine": dot
                    / max(norms[left] * norms[right], 1e-30),
                }
            )
    for group in group_scopes:
        label = "all" if group is None else group
        norms = [
            math.sqrt(max(vector_dot(run.gradients, run.gradients, group=group), 0.0))
            for run in horizon_runs
        ]
        selected_names = [
            name
            for name in horizon_runs[0].gradients
            if group is None or parameter_group(name) == group
        ]
        sum_squared = 0.0
        for name in selected_names:
            total = sum(
                (run.gradients[name] for run in horizon_runs),
                torch.zeros_like(horizon_runs[0].gradients[name]),
            )
            sum_squared += squared_norm(total)
        rss = math.sqrt(sum(value * value for value in norms))
        coherent = math.sqrt(max(sum_squared, 0.0))
        rows.append(
            {
                "horizon": 0,
                "group": label,
                "horizon_ce": math.nan,
                "loss_weight_in_total": 1.0,
                "contribution_gradient_l2": coherent,
                "contribution_gradient_rms": math.nan,
                "contribution_gradient_max_abs": math.nan,
                "root_sum_square_horizon_norm": rss,
                "coherent_to_rss_ratio": coherent / max(rss, 1e-30),
            }
        )
    return rows, cosine_rows


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty TSV {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _retain(name: str, horizon: int, tensor: torch.Tensor, tape):
    if tensor.requires_grad:
        tensor.retain_grad()
    tape.append((name, horizon, tensor))
    return tensor


def traced_central_forward(model, window: torch.Tensor, loss_scope: str):
    (
        prefix_encoded,
        roots,
        _,
        targets,
        anchor_indices,
        full_positions,
    ) = _sparse_window_components(
        model,
        window,
        prefix_length=base.CONTEXT,
        horizons=HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
    )
    transition = model.complex_self_prediction
    module_calls: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        name: [] for name in ("query", "key", "value", "output")
    }
    handles = []
    for module_name in module_calls:
        module = getattr(transition, module_name)

        def hook(_module, inputs, output, *, tag=module_name):
            input_tensor = inputs[0]
            if input_tensor.requires_grad:
                input_tensor.retain_grad()
            if output.requires_grad:
                output.retain_grad()
            module_calls[tag].append((input_tensor, output))

        handles.append(module.register_forward_hook(hook))
    tape: list[tuple[str, int, torch.Tensor]] = []
    try:
        state = transition.initialize(roots)
        _retain("root_z", 0, roots, tape)
        _retain("initial_s", 0, state.memory, tape)
        steps = []
        for horizon in range(1, HORIZONS + 1):
            _retain("z_in", horizon, state.hidden, tape)
            _retain("s_in", horizon, state.memory, tape)
            step = transition(state)
            _retain("rotated_z", horizon, step.rotated_hidden, tape)
            _retain("rotated_s", horizon, step.rotated_memory, tape)
            _retain("p", horizon, step.preliminary_hidden, tape)
            _retain("innovation_w", horizon, step.innovation_memory, tape)
            _retain("s_raw", horizon, step.raw_updated_memory, tape)
            _retain("z_raw", horizon, step.raw_full_hidden, tape)
            _retain("stored_s", horizon, step.state.memory, tape)
            _retain("stored_z", horizon, step.state.hidden, tape)
            steps.append(step)
            state = step.state
        reads = torch.stack([step.read_hidden for step in steps], dim=2)
        decoded = model.exact_inverse_selected_decode_tape_states(
            prefix_encoded,
            reads,
            full_positions[: base.CONTEXT],
            anchor_indices,
        )
        logits = model.token_logits(decoded)
        _retain("decoded_hidden", 0, decoded, tape)
        _retain("logits", 0, logits, tape)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        if loss_scope == "h1":
            loss = ce[:, :, 0].mean()
        elif loss_scope == "h2_h16":
            loss = ce[:, :, 1:].mean()
        elif loss_scope == "total":
            loss = ce.mean()
        else:
            raise ValueError(loss_scope)
        model.zero_grad(set_to_none=True)
        loss.backward()

        # Map the exact repeated central Linear calls to semantic roles.
        _retain("init_k_linear", 0, module_calls["key"][0][1], tape)
        _retain("init_v_linear", 0, module_calls["value"][0][1], tape)
        for horizon in range(1, HORIZONS + 1):
            q0, q1 = module_calls["query"][2 * (horizon - 1) : 2 * horizon]
            key = module_calls["key"][horizon]
            value = module_calls["value"][horizon]
            out0, out1, out2 = module_calls["output"][
                3 * (horizon - 1) : 3 * horizon
            ]
            for name, pair in (
                ("q_rot_linear", q0),
                ("q_updated_linear", q1),
                ("k_linear", key),
                ("v_linear", value),
                ("prior_read_flat", out0),
                ("innovation_read_flat", out1),
                ("full_read_flat", out2),
            ):
                _retain(name + ".input", horizon, pair[0], tape)
                _retain(name + ".output", horizon, pair[1], tape)
        return float(loss.detach()), tensor_metric_rows(tape, loss_scope)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def semantic_dims(name: str, value: torch.Tensor) -> tuple[int, ...]:
    if name in {
        "initial_s",
        "s_in",
        "rotated_s",
        "innovation_w",
        "s_raw",
        "stored_s",
    }:
        return tuple(range(value.ndim - 3, value.ndim))
    if value.ndim:
        return (-1,)
    return ()


def tensor_metric_rows(tape, loss_scope: str) -> list[dict[str, object]]:
    rows = []
    seen: set[tuple[str, int, int]] = set()
    for name, horizon, value in tape:
        identity = (name, horizon, id(value))
        if identity in seen:
            continue
        seen.add(identity)
        detached = value.detach()
        magnitude = detached.abs().float()
        dims = semantic_dims(name.split(".")[0], value)
        row_rms = (
            magnitude.square().mean(dim=dims).sqrt()
            if dims
            else magnitude
        ).flatten()
        gradient = value.grad
        if gradient is None:
            grad_rms = grad_max = grad_q99 = 0.0
            grad_row_q50 = grad_row_q99 = 0.0
            grad_times_activation_rms = 0.0
        else:
            grad_mag = gradient.detach().abs().float()
            grad_rows = (
                grad_mag.square().mean(dim=dims).sqrt()
                if dims
                else grad_mag
            ).flatten()
            grad_rms = float(grad_mag.square().mean().sqrt())
            grad_max = float(grad_mag.amax())
            grad_q99 = float(torch.quantile(grad_mag.flatten(), 0.99))
            grad_row_q50 = float(torch.quantile(grad_rows, 0.50))
            grad_row_q99 = float(torch.quantile(grad_rows, 0.99))
            grad_times_activation_rms = float(
                (gradient.detach().conj() * detached)
                .real.float()
                .square()
                .mean()
                .sqrt()
            )
        rows.append(
            {
                "loss_scope": loss_scope,
                "horizon": horizon,
                "tensor": name,
                "shape": "x".join(str(part) for part in value.shape),
                "activation_rms": float(magnitude.square().mean().sqrt()),
                "activation_max_abs": float(magnitude.amax()),
                "activation_q99_abs": float(torch.quantile(magnitude.flatten(), 0.99)),
                "row_rms_q50": float(torch.quantile(row_rms, 0.50)),
                "row_rms_q99": float(torch.quantile(row_rms, 0.99)),
                "row_rms_max": float(row_rms.amax()),
                "adjoint_rms": grad_rms,
                "adjoint_max_abs": grad_max,
                "adjoint_q99_abs": grad_q99,
                "adjoint_row_rms_q50": grad_row_q50,
                "adjoint_row_rms_q99": grad_row_q99,
                "adjoint_times_activation_rms": grad_times_activation_rms,
                "finite_fraction": float(torch.isfinite(magnitude).float().mean()),
            }
        )
    return rows


def module_call_role(module_name: str, call: int) -> str:
    if call == 0:
        return "encoder_use"
    if ".attn.qkv" in module_name:
        return (
            "inverse_decoder_prefix_self"
            if call == 1
            else "inverse_decoder_prefix_branch_kv"
            if call == 2
            else "inverse_decoder_branch"
        )
    if ".attn.proj" in module_name:
        return "inverse_decoder_prefix_self" if call == 1 else "inverse_decoder_branch"
    return "inverse_decoder_prefix" if call == 1 else "inverse_decoder_branch"


def module_trace(model, window: torch.Tensor, loss_scope: str):
    records: list[tuple[str, int, torch.Tensor, torch.Tensor]] = []
    counts: dict[str, int] = {}
    handles = []
    for name, module in model.encoder.named_modules():
        selected = isinstance(
            module,
            (torch.nn.Linear, torch.nn.GELU, torch.nn.Embedding),
        ) or module.__class__.__name__ == "RMSNorm"
        if not name or not selected:
            continue

        def hook(_module, inputs, output, *, module_name=name):
            if not torch.is_tensor(output):
                return
            input_tensor = inputs[0]
            call = counts.get(module_name, 0)
            counts[module_name] = call + 1
            if input_tensor.requires_grad:
                input_tensor.retain_grad()
            if output.requires_grad:
                output.retain_grad()
            records.append((module_name, call, input_tensor, output))

        handles.append(module.register_forward_hook(hook))
    try:
        output = forward_sparse_complex_self_predicted_kv_window(
            model,
            window,
            prefix_length=base.CONTEXT,
            horizons=HORIZONS,
            anchor_stride=ANCHOR_STRIDE,
            detach_cross_horizon_gradients=False,
        )
        if loss_scope == "h1":
            loss = output.token_ce[:, :, 0].mean()
        elif loss_scope == "h2_h16":
            loss = output.token_ce[:, :, 1:].mean()
        else:
            loss = output.token_ce.mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        tape = [
            (f"encoder.{name}.call{call}", 0, tensor)
            for name, call, _, tensor in records
        ]
        metric_rows = tensor_metric_rows(tape, loss_scope)
        row_by_name = {row["tensor"]: row for row in metric_rows}
        role_vectors: dict[tuple[str, str], torch.Tensor] = {}
        for name, call, input_tensor, output_tensor in records:
            tensor_name = f"encoder.{name}.call{call}"
            row = row_by_name[tensor_name]
            role = module_call_role(name, call)
            row["module"] = f"encoder.{name}"
            row["call_index"] = call
            row["call_role"] = role
            module = model.encoder.get_submodule(name)
            if isinstance(module, torch.nn.Linear) and output_tensor.grad is not None:
                flat_gradient = output_tensor.grad.detach().float().reshape(
                    -1, output_tensor.shape[-1]
                )
                flat_input = input_tensor.detach().float().reshape(
                    -1, input_tensor.shape[-1]
                )
                call_gradient = flat_gradient.T @ flat_input
                row["call_weight_gradient_l2"] = float(call_gradient.norm())
                row["call_weight_gradient_rms"] = float(
                    call_gradient.square().mean().sqrt()
                )
                row["call_weight_gradient_max_abs"] = float(
                    call_gradient.abs().amax()
                )
                coarse_role = (
                    "encoder_use" if role == "encoder_use" else "inverse_decoder_use"
                )
                key = (name, coarse_role)
                role_vectors[key] = role_vectors.get(
                    key, torch.zeros_like(call_gradient)
                ) + call_gradient
            else:
                row["call_weight_gradient_l2"] = math.nan
                row["call_weight_gradient_rms"] = math.nan
                row["call_weight_gradient_max_abs"] = math.nan
        role_rows = []
        for (name, role), gradient in role_vectors.items():
            module = model.encoder.get_submodule(name)
            reconstructed = sum(
                (
                    value
                    for (candidate, _), value in role_vectors.items()
                    if candidate == name
                ),
                torch.zeros_like(gradient),
            )
            reference = module.weight.grad.detach().float()
            reconstruction_error = float(
                (reconstructed - reference).norm()
                / reference.norm().clamp_min(1e-30)
            )
            role_rows.append(
                {
                    "scope": "module_role",
                    "name": f"encoder.{name}.{role}",
                    "group": f"shared_revnet.{role}",
                    "numel": gradient.numel(),
                    "weight_l2": math.nan,
                    "weight_rms": math.nan,
                    "weight_max_abs": math.nan,
                    "weight_spectral_norm_estimate": math.nan,
                    "gradient_l2": float(gradient.norm()),
                    "gradient_rms": float(gradient.square().mean().sqrt()),
                    "gradient_max_abs": float(gradient.abs().amax()),
                    "gradient_to_weight_l2": math.nan,
                    "squared_global_fraction": math.nan,
                    "shared_call_sum_relative_error": reconstruction_error,
                }
            )
        return float(loss.detach()), metric_rows, role_rows
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def product_pack(state: ComplexMemoryState) -> torch.Tensor:
    hidden_size = state.hidden.numel()
    memory_size = state.memory.numel()
    return torch.cat(
        (
            state.hidden.reshape(-1) / math.sqrt(hidden_size),
            state.memory.real.reshape(-1) / math.sqrt(memory_size),
            state.memory.imag.reshape(-1) / math.sqrt(memory_size),
        )
    )


def product_unpack(coordinates: torch.Tensor, template: ComplexMemoryState):
    hidden_size = template.hidden.numel()
    memory_size = template.memory.numel()
    hidden = coordinates[:hidden_size].reshape_as(template.hidden)
    hidden = hidden * math.sqrt(hidden_size)
    real = coordinates[hidden_size : hidden_size + memory_size].reshape_as(
        template.memory.real
    )
    imaginary = coordinates[hidden_size + memory_size :].reshape_as(
        template.memory.imag
    )
    memory = torch.complex(real, imaginary) * math.sqrt(memory_size)
    return ComplexMemoryState(hidden, memory, template.write_count)


def state_map(transition, template: ComplexMemoryState, depth: int):
    def function(coordinates: torch.Tensor):
        state = product_unpack(coordinates, template)
        for _ in range(depth):
            state = transition(state).state
        return product_pack(state)

    return function


def jacobian_rows(model, window: torch.Tensor, iterations: int, phase: str):
    with torch.no_grad():
        prefix, roots, _, _, _, _ = _sparse_window_components(
            model,
            window[:1],
            prefix_length=base.CONTEXT,
            horizons=HORIZONS,
            anchor_stride=ANCHOR_STRIDE,
        )
        del prefix
        state = model.complex_self_prediction.initialize(roots[:1, :1])
    transition = model.complex_self_prediction
    rows = []
    root_state = ComplexMemoryState(
        state.hidden.detach(), state.memory.detach(), state.write_count
    )
    current = root_state
    for horizon in range(1, HORIZONS + 1):
        point = product_pack(current).detach()
        estimate = jacobian_spectral_norm_power_iteration(
            state_map(transition, current, 1),
            point,
            iterations=iterations,
            seed=19001 + horizon,
        )
        rows.append(
            {
                "phase": phase,
                "scope": "local_one_step",
                "horizon": horizon,
                "depth": 1,
                "spectral_norm": estimate.singular_value,
                "history": ",".join(f"{value:.8g}" for value in estimate.history),
            }
        )
        with torch.no_grad():
            current = transition(current).state
    root_point = product_pack(root_state).detach()
    for depth in range(1, HORIZONS + 1):
        estimate = jacobian_spectral_norm_power_iteration(
            state_map(transition, root_state, depth),
            root_point,
            iterations=iterations,
            seed=20011 + depth,
        )
        rows.append(
            {
                "phase": phase,
                "scope": "root_composition",
                "horizon": depth,
                "depth": depth,
                "spectral_norm": estimate.singular_value,
                "history": ",".join(f"{value:.8g}" for value in estimate.history),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.microbatch < 1 or args.trace_examples < 1:
        raise ValueError("microbatch and trace examples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("gradient forensics requires CUDA")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    pre_payload = load_payload(SOURCE_OUTPUT / "step1000.pt")
    if int(pre_payload["step"]) != 1000:
        raise RuntimeError("native source checkpoint is not step 1000")
    window, generator_state_before = exact_next_batch(pre_payload)
    torch.save(
        {
            "window": window,
            "data_generator_state_before": generator_state_before,
            "lr": lr_at(1000, 6000),
            "microbatch": args.microbatch,
        },
        args.output_dir / "update1001_batch.pt",
    )
    pre_model = checkpoint_model(pre_payload, device)

    control = run_gradient(
        pre_model,
        window,
        microbatch=args.microbatch,
        detach=False,
        sqrt_reads=False,
        collect_forward=True,
        condition="pre1001.attached.raw",
    )
    control_norm = gradient_norm(control.gradients)

    parameter_rows = parameter_gradient_rows(pre_model, control.gradients)

    horizon_runs = []
    for horizon in range(1, HORIZONS + 1):
        horizon_runs.append(
            run_gradient(
                pre_model,
                window,
                microbatch=args.microbatch,
                detach=False,
                sqrt_reads=False,
                horizon=horizon,
                contribution_to_total=True,
            )
        )
    horizon_rows, cosine_rows = horizon_gradient_rows(horizon_runs)
    horizon_sum_error_sq = 0.0
    control_sq = 0.0
    for name, reference in control.gradients.items():
        reconstructed = sum(
            (run.gradients[name] for run in horizon_runs),
            torch.zeros_like(reference),
        )
        horizon_sum_error_sq += squared_norm(reconstructed - reference)
        control_sq += squared_norm(reference)
    horizon_sum_relative_error = math.sqrt(horizon_sum_error_sq) / max(
        math.sqrt(control_sq), 1e-30
    )
    if horizon_sum_relative_error >= 5e-4:
        raise RuntimeError(
            "1/16 horizon gradients do not reconstruct the total gradient: "
            f"relative error {horizon_sum_relative_error}"
        )
    horizon_rows.append(
        {
            "horizon": -1,
            "group": "sum_verification",
            "horizon_ce": math.nan,
            "loss_weight_in_total": 1.0,
            "contribution_gradient_l2": math.sqrt(horizon_sum_error_sq),
            "contribution_gradient_rms": math.nan,
            "contribution_gradient_max_abs": math.nan,
            "gradient_sum_relative_error": horizon_sum_relative_error,
        }
    )
    write_tsv(args.record_dir / "horizon_gradients.tsv", horizon_rows)
    write_tsv(args.record_dir / "horizon_gradient_cosines.tsv", cosine_rows)

    counterfactual_rows = []
    for name, detach, sqrt_reads in (
        ("attached_raw", False, False),
        ("detached_raw", True, False),
        ("attached_sqrt", False, True),
        ("detached_sqrt", True, True),
    ):
        run = (
            control
            if name == "attached_raw"
            else run_gradient(
                pre_model,
                window,
                microbatch=args.microbatch,
                detach=detach,
                sqrt_reads=sqrt_reads,
                collect_forward=True,
                condition=f"pre1001.{name}",
            )
        )
        group_names = sorted({parameter_group(key) for key in run.gradients})
        for group in ("all", *group_names):
            selected = {
                key: value
                for key, value in run.gradients.items()
                if group == "all" or parameter_group(key) == group
            }
            control_selected = {
                key: value
                for key, value in control.gradients.items()
                if group == "all" or parameter_group(key) == group
            }
            counterfactual_rows.append(
                {
                    "phase": "pre_update1001_train_batch",
                    "condition": name,
                    "horizon": 0,
                    "group": group,
                    "loss": run.loss,
                    "gradient_l2": gradient_norm(selected),
                    "gradient_ratio_to_control": gradient_norm(selected)
                    / max(gradient_norm(control_selected), 1e-30),
                }
            )
        for horizon, loss in enumerate(run.horizon_losses, 1):
            counterfactual_rows.append(
                {
                    "phase": "pre_update1001_train_batch",
                    "condition": name,
                    "horizon": horizon,
                    "group": "loss_only",
                    "loss": loss,
                    "gradient_l2": math.nan,
                    "gradient_ratio_to_control": math.nan,
                }
            )
        counterfactual_rows.extend(
            summarized_forward_rows(
                run.forward_rows,
                phase="pre_update1001_train_batch",
                condition=name,
            )
        )

    trace_window = window[: min(args.trace_examples, len(window))].to(device)
    recurrent_rows = []
    module_rows = []
    for loss_scope in ("h1", "h2_h16", "total"):
        _, rows = traced_central_forward(pre_model, trace_window, loss_scope)
        for row in rows:
            row["phase"] = "pre_update1001"
        recurrent_rows.extend(rows)
        _, rows, role_rows = module_trace(pre_model, trace_window, loss_scope)
        for row in rows:
            row["phase"] = "pre_update1001"
        module_rows.extend(rows)
        if loss_scope == "total":
            parameter_rows.extend(role_rows)
    write_tsv(args.record_dir / "parameter_gradients.tsv", parameter_rows)

    jacobians = jacobian_rows(
        pre_model,
        trace_window,
        args.jacobian_iterations,
        "pre_update1001",
    )

    # Reinstall the exact raw gradients, apply the historical clip, and take
    # precisely one AdamW update from the checkpointed optimizer state.
    optimizer = torch.optim.AdamW(
        (parameter for parameter in pre_model.parameters() if parameter.requires_grad),
        lr=lr_at(1000, 6000),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    optimizer.load_state_dict(pre_payload["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = lr_at(1000, 6000)
    named = trainable_parameters(pre_model)
    pre_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in named.items()
    }
    for name, parameter in named.items():
        parameter.grad = control.gradients[name].to(device).clone()
    clipped_from = float(
        torch.nn.utils.clip_grad_norm_(pre_model.parameters(), CLIP_NORM)
    )
    if not math.isclose(clipped_from, control_norm, rel_tol=2e-5, abs_tol=1e-5):
        raise RuntimeError(
            f"clip saw {clipped_from}, expected raw norm {control_norm}"
        )
    clipped_gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in named.items()
    }
    optimizer.step()
    post_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in named.items()
    }
    update_rows = parameter_update_rows(
        pre_parameters,
        control.gradients,
        clipped_gradients,
        post_parameters,
    )
    write_tsv(args.record_dir / "parameter_updates.tsv", update_rows)
    torch.save(
        {
            "model": pre_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 1001,
            "source_step": 1000,
            "raw_gradient_norm": control_norm,
            "loss": control.loss,
        },
        args.output_dir / "post_update1001.pt",
    )
    post_model = pre_model

    for loss_scope in ("h1", "h2_h16", "total"):
        _, rows = traced_central_forward(post_model, trace_window, loss_scope)
        for row in rows:
            row["phase"] = "post_update1001"
        recurrent_rows.extend(rows)
        _, rows, _ = module_trace(post_model, trace_window, loss_scope)
        for row in rows:
            row["phase"] = "post_update1001"
        module_rows.extend(rows)
    write_tsv(args.record_dir / "recurrent_tensor_metrics.tsv", recurrent_rows)
    write_tsv(args.record_dir / "module_tensor_metrics.tsv", module_rows)

    jacobians.extend(
        jacobian_rows(
            post_model,
            trace_window,
            args.jacobian_iterations,
            "post_update1001",
        )
    )
    write_tsv(args.record_dir / "jacobian_metrics.tsv", jacobians)

    post_update = run_gradient(
        post_model,
        window,
        microbatch=args.microbatch,
        detach=False,
        sqrt_reads=False,
        collect_forward=True,
        condition="post1001.same_batch",
    )
    counterfactual_rows.extend(
        summarized_forward_rows(
            post_update.forward_rows,
            phase="post_update1001_same_batch",
            condition="attached_raw",
        )
    )
    counterfactual_rows.append(
        {
            "phase": "post_update1001_same_batch",
            "condition": "attached_raw",
            "horizon": 0,
            "group": "all",
            "loss": post_update.loss,
            "gradient_l2": gradient_norm(post_update.gradients),
            "gradient_ratio_to_control": gradient_norm(post_update.gradients)
            / max(control_norm, 1e-30),
        }
    )
    write_tsv(args.record_dir / "counterfactuals.tsv", counterfactual_rows)

    group_rows = [row for row in parameter_rows if row["scope"] == "group"]
    dominant_group = max(group_rows, key=lambda row: row["gradient_l2"])
    dominant_horizon = max(
        (row for row in horizon_rows if row["group"] == "all" and row["horizon"]),
        key=lambda row: row["contribution_gradient_l2"],
    )
    detached = next(
        row
        for row in counterfactual_rows
        if row["condition"] == "detached_raw" and row["group"] == "all"
    )
    sqrt_row = next(
        row
        for row in counterfactual_rows
        if row["condition"] == "attached_sqrt" and row["group"] == "all"
    )
    dominant_update = max(
        (row for row in update_rows if row["scope"] == "group"),
        key=lambda row: row["adam_parameter_delta_l2"],
    )
    dominant_adjoint = max(
        (
            row
            for row in recurrent_rows
            if row["phase"] == "pre_update1001"
            and row["loss_scope"] == "total"
        ),
        key=lambda row: row["adjoint_rms"],
    )
    interpretation = (
        "# Interpretation\n\n"
        f"The native step-1000 checkpoint's exact next batch had CE "
        f"`{control.loss:.9f}` and raw global gradient norm "
        f"`{control_norm:.9f}` before clipping to `{CLIP_NORM}`. The same "
        f"batch after the single update had CE `{post_update.loss:.9f}`.\n\n"
        f"The largest named parameter group was `{dominant_group['group']}` "
        f"with L2 gradient `{dominant_group['gradient_l2']:.6g}` and squared "
        f"global share `{dominant_group['squared_global_fraction']:.6f}`. "
        f"The largest 1/16-weighted horizon contribution was H"
        f"{dominant_horizon['horizon']} at "
        f"`{dominant_horizon['contribution_gradient_l2']:.6g}`.\n\n"
        f"After clipping and Adam, the largest group parameter movement was "
        f"`{dominant_update['group']}` with delta L2 "
        f"`{dominant_update['adam_parameter_delta_l2']:.6g}`. The largest "
        f"recorded recurrent total-loss adjoint RMS on the first "
        f"{len(trace_window)} exact-batch rows was "
        f"`{dominant_adjoint['tensor']}` at H"
        f"{dominant_adjoint['horizon']} "
        f"(`{dominant_adjoint['adjoint_rms']:.6g}`).\n\n"
        f"On frozen weights, horizon detachment changed the instantaneous "
        f"global norm to `{detached['gradient_l2']:.6g}`; accumulated-read "
        f"sqrt scaling changed it to `{sqrt_row['gradient_l2']:.6g}`. These "
        "are conditioning counterfactuals, not retraining evidence.\n\n"
        "`recurrent_tensor_metrics.tsv` separates H1, H2--H16, and total "
        "adjoints at every raw/stored carrier and Q/K/V/read call. "
        "Shared RevNet Linear call-site gradients are split into encoder-use "
        f"and inverse-decoder-use on the first {len(trace_window)} batch rows; "
        "they are sample-local rather than mislabeled as the full-batch "
        "update gradient. "
        "`jacobian_metrics.tsv` reports local and composed worst-direction "
        "gains before and after update 1001. Conflicting directions are "
        "retained in the TSV files.\n"
    )
    (args.record_dir / "interpretation.md").write_text(interpretation)
    print(
        f"step1000 next-batch loss={control.loss:.6f} gnorm={control_norm:.6f}; "
        f"dominant={dominant_group['group']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
