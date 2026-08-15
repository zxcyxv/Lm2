"""Density-spectrum and exact Born cross-term audit at step 6000."""
from __future__ import annotations

import csv
import math
import statistics
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
    "step6000-density-born-audit"
)
SAMPLES = 8
HORIZONS = 16


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def density_metrics(memory: torch.Tensor) -> tuple[torch.Tensor, ...]:
    singular = torch.linalg.svdvals(memory)
    eigenvalues = singular.square().flatten(1)
    eigenvalues = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    purity = eigenvalues.square().sum(dim=-1)
    entropy = -(
        eigenvalues * eigenvalues.clamp_min(1e-30).log()
    ).sum(dim=-1)
    effective_rank = entropy.exp()
    largest = eigenvalues.max(dim=-1).values
    return purity, entropy, effective_rank, largest, eigenvalues.sum(dim=-1)


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    base.FUSE_BRANCH_MEMORY_KERNEL = False
    model = base.make_model().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    transition = model.complex_self_prediction
    validation = base.memmap("validation")
    starts = torch.randint(
        0,
        len(validation) - base.EVAL_WINDOW_LENGTH,
        (64,),
        generator=torch.Generator().manual_seed(base.SEED + 999),
    )[:SAMPLES]
    window = base.windows_of_length(
        validation, starts, base.EVAL_WINDOW_LENGTH
    ).cuda()

    token_rows = []
    spectrum_rows = []
    with torch.inference_mode():
        _, _, _, output = base.objective(model, window)
        anchor_slot = output.anchor_indices.numel() - 1
        roots = output.prefix_encoded.index_select(
            1, output.anchor_indices[anchor_slot : anchor_slot + 1]
        )[:, 0]
        targets = output.targets[:, anchor_slot]
        logits = output.logits[:, anchor_slot]
        predictions = logits.argmax(dim=-1)
        probabilities = logits.softmax(dim=-1).amax(dim=-1)
        state = transition.initialize(roots)
        writes = []
        for read_index in range(HORIZONS):
            result = transition(state)
            writes.append(result.innovation_memory)
            memory = result.state.memory
            purity, entropy, effective_rank, largest, eigen_sum = density_metrics(memory)
            query = transition.queries(result.rotated_hidden)
            query_norm = query.abs().square().sum(dim=(-2, -1)).sqrt()
            memory_norm = memory.abs().square().sum(dim=(-3, -2, -1)).sqrt()
            amplitudes = []
            for write_index, write in enumerate(writes):
                age = read_index - write_index
                phase = torch.exp(-1j * transition.memory_phase * age)
                transported = write * phase[..., None]
                amplitude = transition.read(query, transported) / (
                    query_norm * memory_norm
                ).clamp_min(1e-30)[..., None, None]
                amplitudes.append(amplitude)
            amplitude_stack = torch.stack(amplitudes, dim=1)
            total_amplitude = amplitude_stack.sum(dim=1)
            total_energy = total_amplitude.abs().square().sum(dim=(-2, -1))
            diagonal_energy = amplitude_stack.abs().square().sum(dim=(-3, -2, -1))
            positive = torch.zeros_like(total_energy)
            negative = torch.zeros_like(total_energy)
            for left in range(read_index + 1):
                for right in range(left + 1, read_index + 1):
                    cross = 2 * (
                        amplitude_stack[:, left].conj()
                        * amplitude_stack[:, right]
                    ).real.sum(dim=(-2, -1))
                    positive += cross.clamp_min(0)
                    negative += (-cross.clamp_max(0))
            reconstructed_energy = diagonal_energy + positive - negative
            reconstruction_error = (reconstructed_energy - total_energy).abs()
            born_weight = total_energy
            for sample in range(SAMPLES):
                target = int(targets[sample, read_index])
                predicted = int(predictions[sample, read_index])
                token_rows.append({
                    "sample": sample,
                    "horizon": read_index + 1,
                    "target_byte": target,
                    "predicted_byte": predicted,
                    "correct": target == predicted,
                    "predicted_space": predicted == 32,
                    "correct_nonspace": target == predicted and target != 32,
                    "top_probability": float(probabilities[sample, read_index]),
                    "density_purity": float(purity[sample]),
                    "density_entropy_nats": float(entropy[sample]),
                    "density_effective_rank": float(effective_rank[sample]),
                    "density_largest_eigenvalue": float(largest[sample]),
                    "density_eigenvalue_sum": float(eigen_sum[sample]),
                    "born_weight": float(born_weight[sample]),
                    "total_measurement_energy": float(total_energy[sample]),
                    "diagonal_write_energy": float(diagonal_energy[sample]),
                    "positive_cross_term": float(positive[sample]),
                    "negative_cross_term": float(negative[sample]),
                    "net_cross_term": float(positive[sample] - negative[sample]),
                    "cross_to_diagonal_ratio": float(
                        (positive[sample] - negative[sample])
                        / diagonal_energy[sample].clamp_min(1e-30)
                    ),
                    "negative_to_positive_cross_ratio": float(
                        negative[sample] / positive[sample].clamp_min(1e-30)
                    ),
                    "born_reconstruction_error": float(reconstruction_error[sample]),
                })
                for eigen_index, eigenvalue in enumerate(
                    torch.linalg.svdvals(memory[sample]).square().flatten()
                    / memory[sample].abs().square().sum().clamp_min(1e-30)
                ):
                    spectrum_rows.append({
                        "sample": sample,
                        "horizon": read_index + 1,
                        "eigen_index": eigen_index,
                        "normalized_eigenvalue": float(eigenvalue),
                    })
            state = result.state

    maximum_sum_error = max(abs(row["density_eigenvalue_sum"] - 1) for row in token_rows)
    maximum_born_error = max(row["born_reconstruction_error"] for row in token_rows)
    if maximum_sum_error > 1e-5 or maximum_born_error > 1e-5:
        raise RuntimeError(
            f"audit reconstruction failed: density={maximum_sum_error}, born={maximum_born_error}"
        )

    groups = {
        "all": token_rows,
        "shallow_h1_h4_correct": [r for r in token_rows if r["horizon"] <= 4 and r["correct"]],
        "shallow_h1_h4_incorrect": [r for r in token_rows if r["horizon"] <= 4 and not r["correct"]],
        "shallow_h1_h4_correct_nonspace": [r for r in token_rows if r["horizon"] <= 4 and r["correct_nonspace"]],
        "late_h8_h16_space": [r for r in token_rows if r["horizon"] >= 8 and r["predicted_space"]],
        "late_h8_h16_nonspace": [r for r in token_rows if r["horizon"] >= 8 and not r["predicted_space"]],
    }
    metrics = (
        "density_purity", "density_entropy_nats", "density_effective_rank",
        "density_largest_eigenvalue", "born_weight", "diagonal_write_energy",
        "positive_cross_term", "negative_cross_term", "net_cross_term",
        "cross_to_diagonal_ratio", "negative_to_positive_cross_ratio",
    )
    group_rows = []
    for name, rows in groups.items():
        group_rows.append({
            "group": name,
            "count": len(rows),
            **{
                f"mean_{metric}": statistics.mean(row[metric] for row in rows)
                if rows else math.nan
                for metric in metrics
            },
        })
    for horizon in range(1, HORIZONS + 1):
        rows = [r for r in token_rows if r["horizon"] == horizon]
        group_rows.append({
            "group": f"h{horizon}",
            "count": len(rows),
            **{f"mean_{metric}": statistics.mean(row[metric] for row in rows) for metric in metrics},
        })

    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "tokens.tsv", token_rows)
    write_rows(RECORD / "density_spectrum.tsv", spectrum_rows)
    write_rows(RECORD / "summary.tsv", group_rows)


if __name__ == "__main__":
    main()
