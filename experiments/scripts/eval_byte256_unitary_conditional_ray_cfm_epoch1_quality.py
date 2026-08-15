"""Matched epoch-1 generation and vector-field capacity audit."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import eval_byte256_unitary_fused_scan_step10071_flow_generation as generation
import eval_byte256_unitary_fused_scan_step10071_generation_quality as quality
from rotlm.models.latent_flow_matching import (
    ConditionalRayFlowTimeConditioner,
    conditional_tangent_source,
    from_rotating_frame_rays,
    integrate_conditional_ray_flow,
)
from rotlm.models.prefix_orthogonal_scan import rotate_pairwise
from rotlm.training.conditional_flow_matching import (
    conditional_ray_flow_matching_loss,
)
from rotlm.training.ha_skew_window import _encoded_sparse_window_components
import train_byte256_unitary_conditional_ray_cfm_h16_10epoch as train


base = train.base
STEP = 3_357
SAMPLES = 16
GENERATE = 128
REPORT_SAMPLES = 4
CFM_CHECKPOINT = train.DEFAULT_OUTPUT / f"step{STEP}.pt"
CE_CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-"
    "13m-rtx5090/step3357.pt"
)
RECORD = train.DEFAULT_RECORD


@torch.inference_mode()
def final_anchor_flow_predictions(
    model,
    time_conditioner,
    contexts: torch.Tensor,
    horizons: int,
    *,
    flow_generator: torch.Generator,
) -> torch.Tensor:
    contexts = contexts[:, -base.CONTEXT :]
    positions = torch.arange(base.CONTEXT, device=contexts.device)
    prefix_encoded, _ = model.encode(contexts, positions)
    roots = prefix_encoded[:, -1:, :]
    source, _ = conditional_tangent_source(
        roots,
        horizons,
        train.FLOW_NOISE_SCALE,
        generator=flow_generator,
    )
    endpoint = integrate_conditional_ray_flow(
        model.complex_self_prediction,
        time_conditioner,
        source,
        steps=train.FLOW_INTEGRATION_STEPS,
    )
    read_states = from_rotating_frame_rays(
        model.complex_self_prediction,
        endpoint,
    )
    anchor_indices = torch.tensor(
        [base.CONTEXT - 1],
        device=contexts.device,
        dtype=torch.long,
    )
    decoded = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        read_states,
        positions,
        anchor_indices,
    )
    return model.token_logits(decoded)[:, 0]


@torch.inference_mode()
def generate_flow(
    model,
    time_conditioner,
    prompts: torch.Tensor,
    *,
    flow_generator: torch.Generator,
    selector=None,
) -> torch.Tensor:
    sequences = prompts.clone()
    remaining = GENERATE
    while remaining:
        take = min(base.TRAIN_HORIZONS, remaining)
        logits = final_anchor_flow_predictions(
            model,
            time_conditioner,
            sequences,
            take,
            flow_generator=flow_generator,
        )
        predictions = logits.argmax(dim=-1) if selector is None else selector(logits)
        sequences = torch.cat((sequences, predictions), dim=1)
        remaining -= take
    return sequences[:, -GENERATE:]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def velocity_floor_rows(
    model,
    time_conditioner,
    windows: torch.Tensor,
) -> list[dict[str, object]]:
    (
        _,
        _,
        roots,
        gold_states,
        _,
        anchor_indices,
        _,
    ) = _encoded_sparse_window_components(
        model,
        windows,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
    )
    selected = anchor_indices.remainder(train.FLOW_ANCHOR_STRIDE).eq(0)
    roots = roots[:, selected]
    gold_states = gold_states[:, selected]
    flow_generator = torch.Generator(device="cuda").manual_seed(
        train.FLOW_NOISE_SEED + 1
    )
    flow = conditional_ray_flow_matching_loss(
        model,
        time_conditioner,
        roots,
        gold_states,
        noise_scale=train.FLOW_NOISE_SCALE,
        flow_time=torch.full(
            (windows.shape[0], int(selected.sum()), 1, 1),
            0.5,
            device=windows.device,
        ),
        generator=flow_generator,
    )
    error = (flow.predicted_velocity - flow.target_velocity).square().sum(-1)
    target_energy = flow.target_velocity.square().sum(-1).clamp_min(1e-8)

    transition = model.complex_self_prediction
    output_columns = transition.output.weight.float().T
    rows = []
    for horizon in range(base.TRAIN_HORIZONS):
        cumulative_angle = transition.hidden_phase.float() * (horizon + 1)
        rotated_columns = rotate_pairwise(
            output_columns,
            -cumulative_angle,
        ).T
        basis, _ = torch.linalg.qr(rotated_columns, mode="reduced")
        state = flow.bridge_state[..., horizon, :].reshape(-1, model.width)
        target = flow.target_velocity[..., horizon, :].reshape(-1, model.width)
        energy = target.square().sum(-1).clamp_min(1e-8)

        state_coordinates = state @ basis
        state_residual = state - state_coordinates @ basis.T
        residual_norm = state_residual.norm(dim=-1, keepdim=True)
        residual_direction = state_residual / residual_norm.clamp_min(1e-8)
        projection_energy = (target @ basis).square().sum(-1)
        projection_energy += torch.where(
            residual_norm.squeeze(-1) > 1e-7,
            (target * residual_direction).sum(-1).square(),
            torch.zeros_like(energy),
        )
        explained = (projection_energy / energy).clamp(0.0, 1.0)
        floor = 1.0 - explained
        actual = (
            error[..., horizon].reshape(-1)
            / target_energy[..., horizon].reshape(-1)
        )
        rows.append(
            {
                "horizon": horizon + 1,
                "readout_rank_upper_bound": basis.shape[1],
                "mean_target_energy": float(energy.mean()),
                "mean_actual_relative_mse": float(actual.mean()),
                "mean_output_subspace_floor_relative_mse": float(floor.mean()),
                "mean_optimizable_gap_above_floor": float(
                    (actual - floor).mean()
                ),
                "mean_representable_energy_fraction": float(explained.mean()),
            }
        )
    return rows


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != STEP:
        raise RuntimeError(f"expected step {STEP}: {checkpoint_path}")
    model = base.make_model().cuda().eval()
    train.scan.configure_scan_backend(model, "fused-triton-rotating-frame")
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def main() -> None:
    if not CFM_CHECKPOINT.exists() or not CE_CHECKPOINT.exists():
        raise FileNotFoundError("both matched epoch-1 checkpoints are required")
    cfm_checkpoint, cfm_model = load_model(CFM_CHECKPOINT)
    _, ce_model = load_model(CE_CHECKPOINT)
    time_conditioner = ConditionalRayFlowTimeConditioner(cfm_model.width).cuda().eval()
    time_conditioner.load_state_dict(cfm_checkpoint["flow_time_conditioner"])

    validation = base.memmap("validation")
    registered_starts = base.load_fixed_starts(train.scan.BASELINE_STARTS, 64)
    starts = registered_starts[:SAMPLES]
    prompts = base.windows_of_length(validation, starts, base.CONTEXT)
    gold = base.windows_of_length(
        validation,
        starts + base.CONTEXT,
        GENERATE,
    )

    def token_selector(seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return quality.top_p_selector(generator)

    outputs = {
        "gold": gold,
        "all_space": torch.full_like(gold, 32),
        "ce_h1_greedy": generation.generate(
            ce_model, prompts, 1, generate_count=GENERATE
        ),
        "ce_h16_greedy": generation.generate(
            ce_model, prompts, 16, generate_count=GENERATE
        ),
        "ce_h16_top_p": generation.generate(
            ce_model,
            prompts,
            16,
            selector=token_selector(base.SEED + 41_016),
            generate_count=GENERATE,
        ),
        "cfm_scan_h1_greedy": generation.generate(
            cfm_model, prompts, 1, generate_count=GENERATE
        ),
        "cfm_scan_h16_greedy": generation.generate(
            cfm_model, prompts, 16, generate_count=GENERATE
        ),
        "cfm_scan_h16_top_p": generation.generate(
            cfm_model,
            prompts,
            16,
            selector=token_selector(base.SEED + 41_016),
            generate_count=GENERATE,
        ),
        "cfm_flow_h16_greedy": generate_flow(
            cfm_model,
            time_conditioner,
            prompts,
            flow_generator=torch.Generator(device="cuda").manual_seed(
                base.SEED + 42_016
            ),
        ),
        "cfm_flow_h16_top_p": generate_flow(
            cfm_model,
            time_conditioner,
            prompts,
            flow_generator=torch.Generator(device="cuda").manual_seed(
                base.SEED + 42_016
            ),
            selector=token_selector(base.SEED + 43_016),
        ),
    }
    metric_rows = [
        {"step": STEP, **quality.aggregate_mode(name, values, gold)}
        for name, values in outputs.items()
    ]
    sample_rows = []
    for sample in range(REPORT_SAMPLES):
        prompt_values = prompts[sample, -96:].cpu().tolist()
        for name, values in outputs.items():
            sample_rows.append(
                {
                    "step": STEP,
                    "sample": sample,
                    "validation_start": int(starts[sample]),
                    "mode": name,
                    "prompt_suffix_escaped": quality.escaped(prompt_values),
                    "continuation_escaped": quality.escaped(
                        values[sample].cpu().tolist()
                    ),
                    "continuation_hex": bytes(
                        values[sample].cpu().tolist()
                    ).hex(),
                }
            )

    validation_windows = base.windows_of_length(
        validation,
        starts,
        base.EVAL_WINDOW_LENGTH,
    )
    floor_rows = velocity_floor_rows(
        cfm_model,
        time_conditioner,
        validation_windows,
    )
    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "generation_metrics.tsv", metric_rows)
    write_rows(RECORD / "generation_samples.tsv", sample_rows)
    write_rows(RECORD / "velocity_floor.tsv", floor_rows)
    for row in metric_rows:
        print(
            row["mode"],
            f"entropy={row['corpus_byte_entropy_bits']:.3f}",
            f"rep4={row['mean_repeated_4gram_fraction']:.3f}",
            f"utf8={row['strict_utf8_sample_fraction']:.3f}",
            f"acc16={row['aligned_accuracy_first_16']:.3f}",
            flush=True,
        )
    print(
        "velocity_floor",
        f"actual={sum(row['mean_actual_relative_mse'] for row in floor_rows) / len(floor_rows):.4f}",
        f"floor={sum(row['mean_output_subspace_floor_relative_mse'] for row in floor_rows) / len(floor_rows):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
