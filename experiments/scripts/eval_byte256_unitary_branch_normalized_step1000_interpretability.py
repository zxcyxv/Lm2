"""Real-sample interference and transported-residual audit at step 1000."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from rotlm.models.prefix_orthogonal_scan import rotate_pairwise
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
CHECKPOINT = Path(
    "outputs/experiments/EXP-20260802-byte256-complex-self-predicted-kv-"
    "unitary-branch-normalized-residual-h16-ce-only-attached-stride16-"
    "1000step-13m/step1000.pt"
)
RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step1000-interpretability-audit"
)
SAMPLES = 8
H = 16


def printable(values: list[int]) -> str:
    return bytes(values).decode("utf-8", errors="replace").replace("\n", "\\n")


def rotate_steps(values: torch.Tensor, phase: torch.Tensor, steps: int) -> torch.Tensor:
    if steps == 0:
        return values
    return rotate_pairwise(
        values,
        (phase * steps).expand(*values.shape[:-1], -1),
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
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
    window = base.windows_of_length(validation, starts, base.EVAL_WINDOW_LENGTH).cuda()

    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        anchor_slot = output.anchor_indices.numel() - 1
        anchor_position = int(output.anchor_indices[anchor_slot])
        roots = output.prefix_encoded[:, anchor_slot]
        state = transition.initialize(roots)
        initial_root = state.hidden
        writes = []
        deltas = []
        states = []
        memory_norms = []
        signed_maps = torch.zeros(SAMPLES, H, H, device="cuda")
        reconstruction_errors = []
        for r in range(H):
            result = transition(state)
            writes.append(result.innovation_memory)
            deltas.append(result.innovation_delta)
            states.append(result.state.hidden)
            frobenius = result.state.memory.abs().square().sum(
                dim=(-3, -2, -1)
            ).sqrt()
            memory_norms.append(frobenius)
            query = transition.queries(result.rotated_hidden)
            contribution_deltas = []
            for n in range(r + 1):
                age = r - n
                phase = torch.polar(
                    torch.ones_like(transition.memory_phase),
                    -transition.memory_phase * age,
                )
                transported_write = writes[n] * phase[..., None]
                read = transition.read(query, transported_write)
                read = read / (frobenius[..., None, None] + transition.post_norm_eps)
                contribution_deltas.append(transition.read_to_hidden(read))
            contributions = torch.stack(contribution_deltas, dim=1)
            reconstructed = contributions.sum(dim=1)
            reconstruction_errors.append(
                (reconstructed - result.innovation_delta).abs().amax(dim=-1)
            )
            denominator = result.innovation_delta.square().sum(dim=-1).clamp_min(1e-30)
            signed_maps[:, r, : r + 1] = (
                contributions * result.innovation_delta[:, None]
            ).sum(dim=-1) / denominator[:, None]
            state = result.state

        logits = output.logits[:, anchor_slot]
        probabilities = logits.softmax(dim=-1)
        top_probability, predictions = probabilities.max(dim=-1)
        targets = output.targets[:, anchor_slot]
        delta_tensor = torch.stack(deltas, dim=1)
        state_tensor = torch.stack(states, dim=1)
        memory_tensor = torch.stack(memory_norms, dim=1)

        transported = []
        for r in range(H):
            transported.append(
                rotate_steps(deltas[r], transition.hidden_phase, H - 1 - r)
            )
        homogeneous_final = rotate_steps(initial_root, transition.hidden_phase, H)
        reconstructed_final = homogeneous_final + torch.stack(transported, dim=1).sum(dim=1)
        final_error = (reconstructed_final - states[-1]).abs().amax(dim=-1)

    sample_rows = []
    horizon_rows = []
    map_rows = []
    for sample in range(SAMPLES):
        matrix = signed_maps[sample].cpu()
        singular = torch.linalg.svdvals(matrix)
        normalized_singular = singular / singular.sum().clamp_min(1e-30)
        effective_rank = float(
            torch.exp(-(normalized_singular * normalized_singular.clamp_min(1e-30).log()).sum())
        )
        pred = predictions[sample].cpu().tolist()
        gold = targets[sample].cpu().tolist()
        context_values = window[sample, max(0, anchor_position - 63): anchor_position + 1].cpu().tolist()
        sample_rows.append({
            "sample": sample,
            "validation_start": int(starts[sample]),
            "anchor_position": anchor_position,
            "context_suffix_bytes": " ".join(map(str, context_values)),
            "context_suffix_text": printable(context_values),
            "gold_bytes": " ".join(map(str, gold)),
            "gold_text": printable(gold),
            "predicted_bytes": " ".join(map(str, pred)),
            "predicted_text": printable(pred),
            "exact_token_accuracy": sum(a == b for a, b in zip(pred, gold)) / H,
            "unique_predicted_bytes": len(set(pred)),
            "interference_effective_rank": effective_rank,
            "max_measurement_reconstruction_error": float(torch.stack(reconstruction_errors, dim=1)[sample].max()),
            "final_frame_reconstruction_error": float(final_error[sample]),
        })
        for r in range(H):
            horizon_rows.append({
                "sample": sample,
                "horizon": r + 1,
                "target_byte": gold[r],
                "predicted_byte": pred[r],
                "correct": pred[r] == gold[r],
                "top_probability": float(top_probability[sample, r]),
                "latent_norm": float(state_tensor[sample, r].norm()),
                "correction_norm": float(delta_tensor[sample, r].norm()),
                "memory_frobenius": float(memory_tensor[sample, r]),
                "transported_final_contribution_norm": float(transported[r][sample].norm()),
            })
            for n in range(r + 1):
                map_rows.append({
                    "sample": sample,
                    "read_horizon": r + 1,
                    "write_horizon": n + 1,
                    "signed_projection_share": float(signed_maps[sample, r, n]),
                })

    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "samples.tsv", sample_rows)
    write_rows(RECORD / "horizons.tsv", horizon_rows)
    write_rows(RECORD / "interference_map.tsv", map_rows)


if __name__ == "__main__":
    main()
