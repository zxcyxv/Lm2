"""Audit epoch-3 co-rotating velocity alignment and real generation."""
from __future__ import annotations

import csv
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from rotlm.models.prefix_orthogonal_scan import rotate_pairwise
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run
from train_byte256_unitary_time_varying_scan_h16_1000 import (
    configure_scan_backend,
)


base = run.base
CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-"
    "13m-rtx5090/step10071.pt"
)
RECORD = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-step10071-"
    "flow-generation-audit"
)
BASELINE_STARTS = Path(
    "experiments/records/"
    "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-"
    "normalized-residual-h16-ce-only-attached-stride16-1000step-13m/"
    "validation_starts.tsv"
)
VELOCITY_SAMPLES = 64
GENERATION_SAMPLES = 8
GENERATE = 128


def printable(values: list[int]) -> str:
    return bytes(values).decode("utf-8", errors="replace").replace(
        "\n", "\\n"
    )


@torch.inference_mode()
def final_anchor_predictions(
    model,
    contexts: torch.Tensor,
    horizons: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = contexts[:, -base.CONTEXT :]
    positions = torch.arange(base.CONTEXT, device=contexts.device)
    prefix_encoded, _ = model.encode(contexts, positions)
    roots = prefix_encoded[:, -1:, :]
    rollout = model.complex_self_prediction.rollout_time_varying_scan(
        roots,
        horizons,
    )
    anchor_indices = torch.tensor(
        [base.CONTEXT - 1],
        device=contexts.device,
        dtype=torch.long,
    )
    decoded = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        rollout.read_states,
        positions,
        anchor_indices,
    )
    logits = model.token_logits(decoded)[:, 0]
    return logits, rollout.full_states[:, 0], rollout.innovation_deltas[:, 0]


@torch.inference_mode()
def generate(
    model,
    prompts: torch.Tensor,
    block: int,
    *,
    selector=None,
    generate_count: int = GENERATE,
) -> torch.Tensor:
    sequences = prompts.clone()
    remaining = generate_count
    while remaining:
        take = min(block, remaining)
        logits, _, _ = final_anchor_predictions(model, sequences, take)
        predictions = (
            logits.argmax(dim=-1)
            if selector is None
            else selector(logits)
        )
        sequences = torch.cat((sequences, predictions), dim=1)
        remaining -= take
    return sequences[:, -generate_count:]


@torch.inference_mode()
def velocity_rows(model, windows: torch.Tensor) -> list[dict[str, object]]:
    positions = torch.arange(windows.shape[1], device=windows.device)
    full_encoded, _ = model.encode(windows, positions)
    prefix_encoded = full_encoded[:, : base.CONTEXT]
    roots = prefix_encoded[:, -1:, :]
    gold_states = full_encoded[
        :, base.CONTEXT : base.CONTEXT + base.TRAIN_HORIZONS
    ]
    rollout = model.complex_self_prediction.rollout_time_varying_scan(
        roots,
        base.TRAIN_HORIZONS,
    )
    anchor_indices = torch.tensor(
        [base.CONTEXT - 1],
        device=windows.device,
        dtype=torch.long,
    )
    prefix_positions = torch.arange(base.CONTEXT, device=windows.device)
    decoded = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        rollout.read_states,
        prefix_positions,
        anchor_indices,
    )
    logits = model.token_logits(decoded)[:, 0]
    predictions = logits.argmax(dim=-1)
    targets = windows[:, base.CONTEXT : base.CONTEXT + base.TRAIN_HORIZONS]

    previous_gold = torch.cat((roots, gold_states[:, :-1]), dim=1)
    local_angles = model.complex_self_prediction.hidden_phase.expand(
        windows.shape[0],
        base.TRAIN_HORIZONS,
        -1,
    )
    rotated_previous_gold = rotate_pairwise(previous_gold, local_angles)
    target_innovations = gold_states - rotated_previous_gold
    predicted_states = rollout.full_states[:, 0].float()
    predicted_innovations = rollout.innovation_deltas[:, 0]
    cumulative_angles = local_angles.cumsum(dim=1)
    frame_predicted = rotate_pairwise(
        predicted_innovations,
        -cumulative_angles,
    ).float()
    frame_target = rotate_pairwise(
        target_innovations,
        -cumulative_angles,
    ).float()
    cosine = F.cosine_similarity(frame_predicted, frame_target, dim=-1)
    predicted_norm = frame_predicted.norm(dim=-1)
    target_norm = frame_target.norm(dim=-1)
    relative_error = (
        (frame_predicted - frame_target).norm(dim=-1)
        / target_norm.clamp_min(1e-8)
    )
    signed_target_projection = (
        (frame_predicted * frame_target).sum(dim=-1)
        / target_norm.clamp_min(1e-8)
    )

    # The decoder consumes a fixed-RMS representative of each carrier ray.
    # Measure the induced ray displacement separately so that an arbitrary
    # radial gauge cannot masquerade as a directional-flow failure.
    predicted_previous = torch.cat(
        (roots[:, 0].float().unsqueeze(1), predicted_states[:, :-1]),
        dim=1,
    )
    gold_states_float = gold_states.float()
    gold_previous_float = previous_gold.float()
    predicted_unit = F.normalize(predicted_states, dim=-1)
    gold_unit = F.normalize(gold_states_float, dim=-1)
    predicted_previous_unit = F.normalize(predicted_previous, dim=-1)
    gold_previous_unit = F.normalize(gold_previous_float, dim=-1)
    rotated_predicted_previous_unit = rotate_pairwise(
        predicted_previous_unit,
        local_angles,
    )
    rotated_gold_previous_unit = rotate_pairwise(
        gold_previous_unit,
        local_angles,
    )
    predicted_ray_velocity = (
        predicted_unit - rotated_predicted_previous_unit
    )
    target_ray_velocity = gold_unit - rotated_gold_previous_unit
    frame_predicted_ray = rotate_pairwise(
        predicted_ray_velocity,
        -cumulative_angles,
    ).float()
    frame_target_ray = rotate_pairwise(
        target_ray_velocity,
        -cumulative_angles,
    ).float()
    ray_velocity_cosine = F.cosine_similarity(
        frame_predicted_ray,
        frame_target_ray,
        dim=-1,
    )
    predicted_ray_norm = frame_predicted_ray.norm(dim=-1)
    target_ray_norm = frame_target_ray.norm(dim=-1)
    ray_velocity_relative_error = (
        (frame_predicted_ray - frame_target_ray).norm(dim=-1)
        / target_ray_norm.clamp_min(1e-8)
    )
    state_direction_cosine = F.cosine_similarity(
        predicted_unit,
        gold_unit,
        dim=-1,
    )

    probabilities = logits.float().softmax(dim=-1)
    rows = []
    for sample in range(windows.shape[0]):
        for horizon in range(base.TRAIN_HORIZONS):
            target = int(targets[sample, horizon])
            prediction = int(predictions[sample, horizon])
            rows.append(
                {
                    "sample": sample,
                    "horizon": horizon + 1,
                    "target_byte": target,
                    "predicted_byte": prediction,
                    "correct": target == prediction,
                    "target_probability": float(
                        probabilities[sample, horizon, target]
                    ),
                    "frame_velocity_cosine": float(cosine[sample, horizon]),
                    "frame_velocity_relative_error": float(
                        relative_error[sample, horizon]
                    ),
                    "frame_predicted_norm": float(
                        predicted_norm[sample, horizon]
                    ),
                    "frame_target_norm": float(target_norm[sample, horizon]),
                    "signed_target_projection": float(
                        signed_target_projection[sample, horizon]
                    ),
                    "frame_ray_velocity_cosine": float(
                        ray_velocity_cosine[sample, horizon]
                    ),
                    "frame_ray_velocity_relative_error": float(
                        ray_velocity_relative_error[sample, horizon]
                    ),
                    "frame_predicted_ray_velocity_norm": float(
                        predicted_ray_norm[sample, horizon]
                    ),
                    "frame_target_ray_velocity_norm": float(
                        target_ray_norm[sample, horizon]
                    ),
                    "state_direction_cosine": float(
                        state_direction_cosine[sample, horizon]
                    ),
                }
            )
    return rows


def aggregate_velocity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "target_probability",
        "frame_velocity_cosine",
        "frame_velocity_relative_error",
        "frame_predicted_norm",
        "frame_target_norm",
        "signed_target_projection",
        "frame_ray_velocity_cosine",
        "frame_ray_velocity_relative_error",
        "frame_predicted_ray_velocity_norm",
        "frame_target_ray_velocity_norm",
        "state_direction_cosine",
    )
    groups: list[tuple[str, list[dict[str, object]]]] = [
        ("all", rows),
        ("correct", [row for row in rows if bool(row["correct"])]),
        ("incorrect", [row for row in rows if not bool(row["correct"])]),
        (
            "shallow_h1_h4_correct",
            [
                row
                for row in rows
                if int(row["horizon"]) <= 4 and bool(row["correct"])
            ],
        ),
        (
            "shallow_h1_h4_incorrect",
            [
                row
                for row in rows
                if int(row["horizon"]) <= 4 and not bool(row["correct"])
            ],
        ),
    ]
    for horizon in range(1, base.TRAIN_HORIZONS + 1):
        groups.append(
            (
                f"h{horizon}",
                [row for row in rows if int(row["horizon"]) == horizon],
            )
        )
    return [
        {
            "group": name,
            "count": len(group),
            "accuracy": statistics.mean(
                float(bool(row["correct"])) for row in group
            )
            if group
            else float("nan"),
            **{
                f"mean_{metric}": statistics.mean(
                    float(row[metric]) for row in group
                )
                if group
                else float("nan")
                for metric in metrics
            },
        }
        for name, group in groups
    ]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 10071:
        raise RuntimeError("expected the epoch-3 checkpoint")
    model = base.make_model().cuda().eval()
    configure_scan_backend(model, "fused-triton-rotating-frame")
    model.load_state_dict(checkpoint["model"])

    validation = base.memmap("validation")
    starts = base.load_fixed_starts(BASELINE_STARTS, VELOCITY_SAMPLES)
    windows = base.windows_of_length(
        validation,
        starts,
        base.EVAL_WINDOW_LENGTH,
    )
    detailed_velocity = velocity_rows(model, windows)
    velocity_summary = aggregate_velocity(detailed_velocity)

    generation_starts = starts[:GENERATION_SAMPLES]
    prompts = base.windows_of_length(
        validation,
        generation_starts,
        base.CONTEXT,
    )
    gold = base.windows_of_length(
        validation,
        generation_starts + base.CONTEXT,
        GENERATE,
    )
    h1 = generate(model, prompts, 1)
    h16 = generate(model, prompts, 16)
    generation_rows = []
    for sample in range(GENERATION_SAMPLES):
        prompt_values = prompts[sample, -96:].cpu().tolist()
        gold_values = gold[sample].cpu().tolist()
        h1_values = h1[sample].cpu().tolist()
        h16_values = h16[sample].cpu().tolist()
        generation_rows.append(
            {
                "sample": sample,
                "validation_start": int(generation_starts[sample]),
                "prompt_suffix_text": printable(prompt_values),
                "gold_text": printable(gold_values),
                "h1_reanchored_text": printable(h1_values),
                "h16_block_text": printable(h16_values),
                "h1_exact_byte_accuracy": statistics.mean(
                    int(a == b) for a, b in zip(h1_values, gold_values)
                ),
                "h16_exact_byte_accuracy": statistics.mean(
                    int(a == b) for a, b in zip(h16_values, gold_values)
                ),
                "h1_unique_bytes": len(set(h1_values)),
                "h16_unique_bytes": len(set(h16_values)),
            }
        )

    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "velocity.tsv", detailed_velocity)
    write_rows(RECORD / "velocity_summary.tsv", velocity_summary)
    write_rows(RECORD / "generation.tsv", generation_rows)
    print(
        next(row for row in velocity_summary if row["group"] == "all"),
        flush=True,
    )
    print(
        next(row for row in velocity_summary if row["group"] == "correct"),
        flush=True,
    )
    print(
        next(row for row in velocity_summary if row["group"] == "incorrect"),
        flush=True,
    )


if __name__ == "__main__":
    main()
