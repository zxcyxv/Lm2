"""Audit H16 complex recurrence scales and matrix-free tangent gains."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from rotlm.latent_diagnostics import (
    jacobian_spectral_norm_power_iteration,
    recurrent_step_scale_diagnostics,
)
from rotlm.models.complex_self_prediction import ComplexMemoryState
from rotlm.training.ha_skew_window import (
    forward_sparse_complex_self_predicted_kv_window,
)
import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base


EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "h16-scale-jacobian-audit"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
CONTEXT = 256
HORIZONS = 16
ANCHOR_STRIDE = 16
DEFAULT_EXAMPLES = 2
VALIDATION_STARTS = Path(
    "experiments/records/"
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "urm-boundary-postnorm-h16-ce-only-attached-stride16-13m/"
    "validation_starts.tsv"
)
OUTPUT_ROOT = Path("outputs/experiments")


CHECKPOINTS = (
    (
        "unnormalized_step0100",
        OUTPUT_ROOT
        / "EXP-20260802-byte256-complex-self-predicted-kv-"
        "post-update-full-read-h16-ce-only-attached-stride16-13m"
        / "step0100.pt",
        True,
    ),
    (
        "unnormalized_step0300",
        OUTPUT_ROOT
        / "EXP-20260802-byte256-complex-self-predicted-kv-"
        "post-update-full-read-h16-ce-only-attached-stride16-13m"
        / "last.pt",
        False,
    ),
    (
        "intermediate_step0100",
        OUTPUT_ROOT
        / "EXP-20260802-byte256-complex-self-predicted-kv-"
        "urm-postnorm-h16-ce-only-attached-stride16-13m"
        / "step0100.pt",
        True,
    ),
    (
        "intermediate_step0500",
        OUTPUT_ROOT
        / "EXP-20260802-byte256-complex-self-predicted-kv-"
        "urm-postnorm-h16-ce-only-attached-stride16-13m"
        / "step0500.pt",
        False,
    ),
    (
        "intermediate_step1000",
        OUTPUT_ROOT
        / "EXP-20260802-byte256-complex-self-predicted-kv-"
        "urm-postnorm-h16-ce-only-attached-stride16-13m"
        / "step1000.pt",
        False,
    ),
    (
        "boundary_step0100",
        OUTPUT_ROOT
        / "EXP-20260802-byte256-complex-self-predicted-kv-"
        "urm-boundary-postnorm-h16-ce-only-attached-stride16-13m"
        / "step0100.pt",
        True,
    ),
    (
        "boundary_step0300",
        OUTPUT_ROOT
        / "EXP-20260802-byte256-complex-self-predicted-kv-"
        "urm-boundary-postnorm-h16-ce-only-attached-stride16-13m"
        / "last.pt",
        False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--power-iterations", type=int, default=6)
    parser.add_argument(
        "--include-initializations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def rms(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().mean(dim=-1).sqrt()


def complex_rms(values: torch.Tensor) -> torch.Tensor:
    return values.abs().square().float().mean(dim=(-3, -2, -1)).sqrt()


def configure_model(config: dict[str, object]):
    """Use the registered common producer instead of copying model logic."""
    base.VOCABULARY = int(config["vocabulary"])
    base.WIDTH = int(config["width"])
    base.ENCODER_BLOCKS = int(config["encoder_blocks"])
    base.HEADS = int(config["heads"])
    base.KEY_DIM = int(config["key_dim"])
    base.VALUE_DIM = int(config["value_dim"])
    base.SIMPLEX_LOGIT_SCALE = float(config["simplex_logit_scale"])
    base.TEMPERED_READOUT = bool(config["tempered_readout"])
    base.RESIDUAL_PRIOR = bool(config["residual_prior"])
    base.RECURRENT_HIDDEN_MODE = str(config["recurrent_hidden_mode"])
    base.PRE_NORMALIZE_QKV_INPUTS = bool(
        config.get("pre_normalize_qkv_inputs", True)
    )
    base.NORMALIZE_INITIAL_RECURRENT_ROOT = bool(
        config.get("normalize_initial_recurrent_root", False)
    )
    base.NORMALIZE_ACCUMULATED_READS = bool(
        config.get("normalize_accumulated_reads", False)
    )
    base.POST_NORMALIZE_HIDDEN_READS = bool(
        config.get("post_normalize_hidden_reads", False)
    )
    base.POST_NORMALIZE_MEMORY = bool(
        config.get("post_normalize_memory", False)
    )
    base.POST_NORMALIZE_RECURRENT_STATE = bool(
        config.get("post_normalize_recurrent_state", False)
    )
    base.POST_NORM_EPS = float(config.get("post_norm_eps", 1e-6))
    if base.TEMPERED_READOUT:
        base.INITIAL_BETA = float(config["initial_beta"])
    return base.make_model()


def iter_conditions(include_initializations: bool):
    for label, path, include_init in CHECKPOINTS:
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if include_initializations and include_init:
            torch.manual_seed(int(checkpoint["config"]["seed"]))
            yield f"{label.rsplit('_step', 1)[0]}_init", configure_model(
                checkpoint["config"]
            ), checkpoint["config"], 0, None
        model = configure_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model"], strict=True)
        yield label, model, checkpoint["config"], int(checkpoint["step"]), path


def matrix_metrics(model, condition: str, step: int) -> list[dict[str, object]]:
    transition = model.complex_self_prediction
    rows = []
    for name in ("query", "key", "value", "output"):
        weight = getattr(transition, name).weight.detach().float()
        rows.append(
            {
                "condition": condition,
                "step": step,
                "matrix": name,
                "frobenius_norm": float(weight.norm()),
                "rms": float(weight.square().mean().sqrt()),
                "spectral_norm": float(torch.linalg.matrix_norm(weight, ord=2)),
            }
        )
    return rows


def product_pack(state: ComplexMemoryState) -> torch.Tensor:
    hidden = state.hidden
    memory = state.memory
    hidden_size = hidden.numel()
    memory_size = memory.numel()
    return torch.cat(
        (
            hidden.reshape(-1) / math.sqrt(hidden_size),
            memory.real.reshape(-1) / math.sqrt(memory_size),
            memory.imag.reshape(-1) / math.sqrt(memory_size),
        )
    )


def product_unpack(
    coordinates: torch.Tensor,
    template: ComplexMemoryState,
) -> ComplexMemoryState:
    hidden_size = template.hidden.numel()
    memory_size = template.memory.numel()
    hidden = coordinates[:hidden_size].reshape_as(template.hidden)
    hidden = hidden * math.sqrt(hidden_size)
    memory_real = coordinates[
        hidden_size : hidden_size + memory_size
    ].reshape_as(template.memory.real)
    memory_imag = coordinates[hidden_size + memory_size :].reshape_as(
        template.memory.imag
    )
    memory = torch.complex(memory_real, memory_imag) * math.sqrt(memory_size)
    return ComplexMemoryState(
        hidden=hidden,
        memory=memory,
        write_count=template.write_count,
    )


def state_map(transition, template: ComplexMemoryState, depth: int):
    def function(coordinates: torch.Tensor) -> torch.Tensor:
        state = product_unpack(coordinates, template)
        for _ in range(depth):
            state = transition(state).state
        return product_pack(state)

    return function


def restricted_tangent_gain(
    function,
    point: torch.Tensor,
    hidden_coordinates: int,
    *,
    field: str,
    seed: int,
) -> float:
    generator = torch.Generator(device=point.device).manual_seed(seed)
    tangent = torch.zeros_like(point)
    if field == "hidden":
        selected = tangent[:hidden_coordinates]
    elif field == "memory":
        selected = tangent[hidden_coordinates:]
    else:
        raise ValueError(field)
    selected.copy_(
        torch.randn(
            selected.shape,
            generator=generator,
            device=selected.device,
            dtype=selected.dtype,
        )
    )
    tangent = tangent / tangent.norm().clamp_min(1e-12)
    _, output_tangent = torch.func.jvp(function, (point,), (tangent,))
    return float(output_tangent.norm())


def jacobian_metrics(
    transition,
    initial_state: ComplexMemoryState,
    *,
    condition: str,
    step: int,
    power_iterations: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    state = ComplexMemoryState(
        hidden=initial_state.hidden[:1, :1].detach(),
        memory=initial_state.memory[:1, :1].detach(),
        write_count=initial_state.write_count,
    )
    hidden_coordinates = state.hidden.numel()
    for horizon in range(1, HORIZONS + 1):
        point = product_pack(state).detach()
        one_step = state_map(transition, state, 1)
        hidden_gain = restricted_tangent_gain(
            one_step,
            point,
            hidden_coordinates,
            field="hidden",
            seed=9173 + horizon,
        )
        memory_gain = restricted_tangent_gain(
            one_step,
            point,
            hidden_coordinates,
            field="memory",
            seed=10103 + horizon,
        )
        rows.append(
            {
                "condition": condition,
                "step": step,
                "scope": "local_random_probe",
                "horizon": horizon,
                "depth": 1,
                "hidden_input_gain": hidden_gain,
                "memory_input_gain": memory_gain,
                "spectral_norm": math.nan,
                "power_history": "",
            }
        )
        if condition in (
            "unnormalized_step0300",
            "intermediate_step1000",
            "boundary_step0300",
        ):
            local_estimate = jacobian_spectral_norm_power_iteration(
                one_step,
                point,
                iterations=max(3, power_iterations - 2),
                seed=10903 + horizon,
            )
            rows.append(
                {
                    "condition": condition,
                    "step": step,
                    "scope": "local_power_iteration",
                    "horizon": horizon,
                    "depth": 1,
                    "hidden_input_gain": math.nan,
                    "memory_input_gain": math.nan,
                    "spectral_norm": local_estimate.singular_value,
                    "power_history": ",".join(
                        f"{value:.8g}" for value in local_estimate.history
                    ),
                }
            )
        state = transition(state).state

    template = ComplexMemoryState(
        hidden=initial_state.hidden[:1, :1].detach(),
        memory=initial_state.memory[:1, :1].detach(),
        write_count=initial_state.write_count,
    )
    point = product_pack(template).detach()
    for depth in (1, HORIZONS):
        estimate = jacobian_spectral_norm_power_iteration(
            state_map(transition, template, depth),
            point,
            iterations=power_iterations,
            seed=11717 + depth,
        )
        rows.append(
            {
                "condition": condition,
                "step": step,
                "scope": "power_iteration",
                "horizon": depth,
                "depth": depth,
                "hidden_input_gain": math.nan,
                "memory_input_gain": math.nan,
                "spectral_norm": estimate.singular_value,
                "power_history": ",".join(
                    f"{value:.8g}" for value in estimate.history
                ),
            }
        )
    return rows


def rollout_internal_metrics(
    model,
    output,
    *,
    condition: str,
    step: int,
) -> tuple[list[dict[str, object]], ComplexMemoryState]:
    transition = model.complex_self_prediction
    roots = output.prefix_encoded.index_select(1, output.anchor_indices)
    state = transition.initialize(roots)
    initial_state = state
    root_rms = rms(roots)
    initial_memory_rms = complex_rms(state.memory)
    transported_write = torch.zeros_like(state.memory)
    write_energy_sum = torch.zeros_like(initial_memory_rms.square())
    rows: list[dict[str, object]] = []
    scale_names = tuple(
        field.name for field in fields(recurrent_step_scale_diagnostics(
            rotated_memory=state.memory,
            innovation_memory=state.memory,
            raw_updated_memory=state.memory,
            stored_memory=state.memory,
            raw_preliminary_hidden=state.hidden,
            preliminary_hidden=state.hidden,
            raw_full_hidden=state.hidden,
            stored_hidden=state.hidden,
        ))
    )

    for horizon in range(1, HORIZONS + 1):
        central_step = transition(state)
        diagnostics = recurrent_step_scale_diagnostics(
            rotated_memory=central_step.rotated_memory,
            innovation_memory=central_step.innovation_memory,
            raw_updated_memory=central_step.raw_updated_memory,
            stored_memory=central_step.state.memory,
            raw_preliminary_hidden=central_step.raw_preliminary_hidden,
            preliminary_hidden=central_step.preliminary_hidden,
            raw_full_hidden=central_step.raw_full_hidden,
            stored_hidden=central_step.state.hidden,
            eps=transition.post_norm_eps,
        )
        phase = torch.polar(
            torch.ones_like(transition.memory_phase),
            -transition.memory_phase,
        )
        transported_write = (
            transported_write * phase[..., None]
            + central_step.innovation_memory
        )
        write_energy_sum = (
            write_energy_sum
            + central_step.innovation_memory.abs().square().float().mean(
                dim=(-3, -2, -1)
            )
        )
        coherence = (
            transported_write.abs().square().float().mean(dim=(-3, -2, -1))
            / write_energy_sum.clamp_min(1e-12)
        )
        index = horizon - 1
        row: dict[str, object] = {
            "condition": condition,
            "step": step,
            "horizon": horizon,
            "root_hidden_rms": float(root_rms.mean()),
            "initial_memory_rms": float(initial_memory_rms.mean()),
            "canonical_future_hidden_rms": float(
                rms(output.gold_states[:, :, index]).mean()
            ),
            "decoded_hidden_rms": float(
                rms(output.decoded_hidden[:, :, index]).mean()
            ),
            "logit_rms": float(rms(output.logits[:, :, index]).mean()),
            "target_nll": float(output.token_ce[:, :, index].mean()),
            "transported_write_coherence": float(coherence.mean()),
        }
        for name in scale_names:
            row[name] = float(getattr(diagnostics, name).mean())
        rows.append(row)
        state = central_step.state
    return rows, initial_state


def counterfactual_readout_metrics(
    model,
    output,
    *,
    condition: str,
    step: int,
) -> list[dict[str, object]]:
    roots = output.prefix_encoded.index_select(1, output.anchor_indices)
    root_shell = rms(roots).unsqueeze(-1).unsqueeze(-1)
    p_shell = output.read_states.float()
    p_shell = p_shell * torch.rsqrt(
        p_shell.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    p_shell = p_shell * root_shell
    shell_decoded = model.exact_inverse_selected_decode_tape_states(
        output.prefix_encoded,
        p_shell.to(output.read_states.dtype),
        output.full_positions[:CONTEXT],
        output.anchor_indices,
    )
    shell_logits = model.token_logits(shell_decoded)
    normalized_hidden = F.normalize(
        output.decoded_hidden.float(), dim=-1, eps=1e-6
    )
    normalized_head_logits = model.simplex_logit_scale_value * (
        normalized_hidden @ model.embedding_weight.float().T
    )
    rows = []
    variants = (
        ("registered_raw", output.logits),
        ("token_hidden_l2_normalized", normalized_head_logits),
        ("root_shell_latent", shell_logits),
    )
    for horizon in range(HORIZONS):
        targets = output.targets[:, :, horizon]
        for variant, logits in variants:
            selected = logits[:, :, horizon].float()
            rows.append(
                {
                    "condition": condition,
                    "step": step,
                    "horizon": horizon + 1,
                    "variant": variant,
                    "nll": float(F.cross_entropy(
                        selected.flatten(0, 1), targets.flatten()
                    )),
                    "accuracy": float(
                        selected.argmax(dim=-1).eq(targets).float().mean()
                    ),
                    "logit_rms": float(rms(selected).mean()),
                    "max_probability": float(
                        selected.softmax(dim=-1).amax(dim=-1).mean()
                    ),
                }
            )
    return rows


def output_scale_counterfactual(
    model,
    window: torch.Tensor,
    *,
    condition: str,
    step: int,
) -> list[dict[str, object]]:
    transition = model.complex_self_prediction
    original = transition.output.weight.detach().clone()
    runs = {}
    try:
        for multiplier in (0.5, 1.0, 2.0):
            with torch.no_grad():
                transition.output.weight.copy_(original * multiplier)
                output = forward_sparse_complex_self_predicted_kv_window(
                    model,
                    window[:1],
                    prefix_length=CONTEXT,
                    horizons=HORIZONS,
                    anchor_stride=ANCHOR_STRIDE,
                )
                roots = output.prefix_encoded.index_select(
                    1, output.anchor_indices
                )
                state = transition.initialize(roots)
                memory_states = []
                for _ in range(HORIZONS):
                    state = transition(state).state
                    memory_states.append(state.memory)
                runs[multiplier] = (
                    output,
                    torch.stack(memory_states, dim=-4),
                )
        if 1.0 not in runs:
            raise RuntimeError("missing unit output-scale counterfactual")
        baseline_output, baseline_memory = runs[1.0]
        result = []
        for multiplier, (output, memory_states) in runs.items():
            for horizon in range(HORIZONS):
                hidden_relative = (
                    (
                        output.full_states[:, :, horizon]
                        - baseline_output.full_states[:, :, horizon]
                    ).float().square().sum(-1)
                    / baseline_output.full_states[:, :, horizon]
                    .float().square().sum(-1).clamp_min(1e-12)
                ).mean()
                memory_relative = (
                    (
                        memory_states[:, :, horizon]
                        - baseline_memory[:, :, horizon]
                    ).abs().square().float().sum(dim=(-3, -2, -1))
                    / baseline_memory[:, :, horizon]
                    .abs().square().float().sum(
                        dim=(-3, -2, -1)
                    ).clamp_min(1e-12)
                ).mean()
                result.append(
                    {
                        "condition": condition,
                        "step": step,
                        "horizon": horizon + 1,
                        "output_weight_multiplier": multiplier,
                        "p_rms": float(
                            rms(output.read_states[:, :, horizon]).mean()
                        ),
                        "logit_rms": float(
                            rms(output.logits[:, :, horizon]).mean()
                        ),
                        "stored_hidden_relative_mse_vs_unit": float(
                            hidden_relative
                        ),
                        "stored_memory_relative_mse_vs_unit": float(
                            memory_relative
                        ),
                    }
                )
        return result
    finally:
        with torch.no_grad():
            transition.output.weight.copy_(original)


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"refusing to write empty TSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(materialized[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(materialized)


def main() -> None:
    args = parse_args()
    if args.examples < 1:
        raise ValueError("examples must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = base.memmap("validation")
    with VALIDATION_STARTS.open(newline="") as handle:
        registered_start_count = sum(1 for _ in csv.DictReader(
            handle, delimiter="\t"
        ))
    starts = base.load_fixed_starts(
        VALIDATION_STARTS, registered_start_count
    )[: args.examples]
    window = base.windows_of_length(
        validation,
        starts,
        CONTEXT + HORIZONS,
    ).to(device)
    args.record_dir.mkdir(parents=True, exist_ok=True)

    horizon_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    jacobian_rows: list[dict[str, object]] = []
    readout_rows: list[dict[str, object]] = []
    output_scale_rows: list[dict[str, object]] = []
    condition_summaries: list[tuple[str, int, float, float]] = []

    for condition, model, config, step, path in iter_conditions(
        args.include_initializations
    ):
        print(f"audit {condition}", flush=True)
        model = model.to(device).eval()
        with torch.no_grad():
            output = forward_sparse_complex_self_predicted_kv_window(
                model,
                window,
                prefix_length=CONTEXT,
                horizons=HORIZONS,
                anchor_stride=ANCHOR_STRIDE,
            )
            scales, initial_state = rollout_internal_metrics(
                model,
                output,
                condition=condition,
                step=step,
            )
            horizon_rows.extend(scales)
            matrix_rows.extend(matrix_metrics(model, condition, step))
            readout_rows.extend(counterfactual_readout_metrics(
                model,
                output,
                condition=condition,
                step=step,
            ))
            condition_summaries.append(
                (
                    condition,
                    step,
                    float(output.token_ce.mean()),
                    float(rms(output.logits).mean()),
                )
            )
            if condition == "boundary_step0300":
                output_scale_rows.extend(output_scale_counterfactual(
                    model,
                    window,
                    condition=condition,
                    step=step,
                ))

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        jacobian_rows.extend(jacobian_metrics(
            model.complex_self_prediction,
            initial_state,
            condition=condition,
            step=step,
            power_iterations=args.power_iterations,
        ))
        del model, output, initial_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_tsv(args.record_dir / "horizon_metrics.tsv", horizon_rows)
    write_tsv(args.record_dir / "matrix_metrics.tsv", matrix_rows)
    write_tsv(args.record_dir / "jacobian_metrics.tsv", jacobian_rows)
    write_tsv(args.record_dir / "readout_counterfactual.tsv", readout_rows)
    write_tsv(
        args.record_dir / "output_scale_counterfactual.tsv",
        output_scale_rows,
    )
    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerows(
            (
                ("experiment_id", EXPERIMENT_ID),
                ("examples", args.examples),
                ("validation_start_indices", ",".join(map(str, starts.tolist()))),
                ("context", CONTEXT),
                ("horizons", HORIZONS),
                ("anchor_stride", ANCHOR_STRIDE),
                ("power_iterations", args.power_iterations),
                ("device", str(device)),
                ("precision", "strict_float32_no_tf32"),
            )
        )

    lines = [
        "# H16 scale and Jacobian audit",
        "",
        "The tables below are descriptive checkpoint evidence. Causal claims "
        "remain limited to the registered frozen-weight counterfactuals.",
        "",
        "| Condition | Step | Raw block NLL | Logit RMS |",
        "|---|---:|---:|---:|",
    ]
    for condition, step, nll, logit_scale in condition_summaries:
        lines.append(
            f"| {condition} | {step} | {nll:.6f} | {logit_scale:.6f} |"
        )
    lines.extend(
        (
            "",
            "Detailed activation scales, matrix norms, tangent gains, and "
            "readout counterfactuals are stored in the adjacent TSV files.",
            "",
        )
    )
    (args.record_dir / "interpretation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
