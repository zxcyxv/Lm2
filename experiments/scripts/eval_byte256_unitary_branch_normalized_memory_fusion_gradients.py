"""Compare compiled real-pair and eager complex memory gradients."""
from __future__ import annotations

import csv
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
CHECKPOINT = Path(
    "outputs/experiments/EXP-20260802-byte256-unitary-branch-normalized-"
    "h16-6000step-micro64/step6000.pt"
)
RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-qkv-memory-fusion-benchmark"
)


def make(fused: bool, state):
    base.FUSE_BRANCH_MEMORY_KERNEL = fused
    model = base.make_model().cuda().train()
    model.load_state_dict(state)
    return model


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    training = base.memmap("train")
    window = base.sampled_windows(
        training,
        2,
        torch.Generator().manual_seed(base.SEED + 6161),
        base.TRAIN_WINDOW_LENGTH,
    ).cuda()
    reference = make(False, checkpoint["model"])
    candidate = make(True, checkpoint["model"])
    objective = base.TRAIN_OBJECTIVE_OVERRIDE
    reference_loss = objective(reference, window)[0]
    candidate_loss = objective(candidate, window)[0]
    reference_parameters = tuple(
        parameter for parameter in reference.parameters()
        if parameter.requires_grad
    )
    candidate_parameters = tuple(
        parameter for parameter in candidate.parameters()
        if parameter.requires_grad
    )
    reference_gradients = torch.autograd.grad(
        reference_loss,
        reference_parameters,
        allow_unused=True,
    )
    candidate_gradients = torch.autograd.grad(
        candidate_loss,
        candidate_parameters,
        allow_unused=True,
    )
    dot = reference_loss.new_zeros((), dtype=torch.float64)
    reference_square = dot.clone()
    candidate_square = dot.clone()
    difference_square = dot.clone()
    maximum_absolute = 0.0
    compared = 0
    for reference_gradient, candidate_gradient in zip(
        reference_gradients, candidate_gradients
    ):
        if reference_gradient is None or candidate_gradient is None:
            if reference_gradient is not None or candidate_gradient is not None:
                raise RuntimeError("gradient presence mismatch")
            continue
        reference_double = reference_gradient.double()
        candidate_double = candidate_gradient.double()
        dot += (reference_double * candidate_double).sum()
        reference_square += reference_double.square().sum()
        candidate_square += candidate_double.square().sum()
        difference_square += (
            reference_double - candidate_double
        ).square().sum()
        maximum_absolute = max(
            maximum_absolute,
            float((reference_gradient - candidate_gradient).abs().max()),
        )
        compared += 1
    row = {
        "reference_loss": float(reference_loss),
        "candidate_loss": float(candidate_loss),
        "loss_abs_difference": float((reference_loss - candidate_loss).abs()),
        "gradient_tensors_compared": compared,
        "reference_gradient_norm": float(reference_square.sqrt()),
        "candidate_gradient_norm": float(candidate_square.sqrt()),
        "gradient_cosine": float(
            dot / (reference_square * candidate_square).sqrt().clamp_min(1e-30)
        ),
        "gradient_relative_l2_error": float(
            difference_square.sqrt() / reference_square.sqrt().clamp_min(1e-30)
        ),
        "gradient_max_abs_error": maximum_absolute,
    }
    with (RECORD / "gradient_equivalence.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
