"""Channel-resolved frequency, phase, and exact interference audit."""
from __future__ import annotations

import csv
import math
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
    "step6000-phase-interference-audit"
)
SAMPLES = 8
HORIZONS = 16


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def channel_hidden_contributions(transition, values: torch.Tensor) -> torch.Tensor:
    """Map [...,head,key,value] reads to exact [...,head,key,width] terms."""
    real_weight = transition.output.weight[:, : transition.heads * transition.value_dim]
    imag_weight = transition.output.weight[:, transition.heads * transition.value_dim :]
    real_weight = real_weight.reshape(transition.width, transition.heads, transition.value_dim)
    imag_weight = imag_weight.reshape(transition.width, transition.heads, transition.value_dim)
    return (
        torch.einsum("...hkv,dhv->...hkd", values.real, real_weight)
        + torch.einsum("...hkv,dhv->...hkd", values.imag, imag_weight)
    )


def effective_count(weights: torch.Tensor) -> torch.Tensor:
    probabilities = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
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
    window = base.windows_of_length(
        validation, starts, base.EVAL_WINDOW_LENGTH
    ).cuda()

    theta = transition.memory_phase.detach()
    ages = torch.arange(HORIZONS, device="cuda", dtype=theta.dtype)
    phase_codes = torch.exp(-1j * ages[:, None] * theta.flatten()[None, :])
    phase_gram = (phase_codes @ phase_codes.conj().T).abs() / theta.numel()
    frequency_rows = []
    for head in range(transition.heads):
        for key_channel in range(transition.key_dim):
            frequency_rows.append({
                "head": head,
                "key_channel": key_channel,
                "theta_rad_per_step": float(theta[head, key_channel]),
                "period_steps": (
                    2 * math.pi / abs(float(theta[head, key_channel]))
                    if float(theta[head, key_channel]) != 0 else math.inf
                ),
            })
    age_rows = []
    for a in range(HORIZONS):
        for b in range(a + 1, HORIZONS):
            age_rows.append({
                "age_a": a,
                "age_b": b,
                "age_distance": b - a,
                "phase_code_abs_similarity": float(phase_gram[a, b]),
            })

    channel_rows = []
    summary_rows = []
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(
            1, output.anchor_indices[-1:]
        )[:, 0]
        state = transition.initialize(roots)
        writes = []
        keys = []
        values = []
        for read_index in range(HORIZONS):
            result = transition(state)
            write_hidden = result.rotated_hidden
            writes.append(result.innovation_memory)
            keys.append(transition.keys(write_hidden))
            values.append(transition.values(write_hidden))
            query = transition.queries(write_hidden)
            frobenius = result.state.memory.abs().square().sum(
                dim=(-3, -2, -1)
            ).sqrt()
            denominator = frobenius + transition.post_norm_eps
            total_delta = result.innovation_delta
            total_square = total_delta.square().sum(dim=-1).clamp_min(1e-30)
            all_hidden = []
            all_projection = []
            all_norm = []
            all_coefficient_magnitude = []
            all_coefficient_phase = []
            for write_index in range(read_index + 1):
                age = read_index - write_index
                transport = torch.exp(-1j * theta * age)
                coefficient = (
                    query.conj() * transport * keys[write_index]
                ) / math.sqrt(transition.key_dim)
                channel_values = (
                    coefficient[..., None]
                    * values[write_index].conj()[..., None, :]
                ) / denominator[..., None, None, None]
                hidden = channel_hidden_contributions(transition, channel_values)
                projection = (
                    hidden * total_delta[..., None, None, :]
                ).sum(dim=-1) / total_square[..., None, None]
                norm = hidden.norm(dim=-1)
                all_hidden.append(hidden)
                all_projection.append(projection)
                all_norm.append(norm)
                all_coefficient_magnitude.append(coefficient.abs())
                all_coefficient_phase.append(torch.angle(coefficient))
                for sample in range(SAMPLES):
                    for head in range(transition.heads):
                        for key_channel in range(transition.key_dim):
                            channel_rows.append({
                                "sample": sample,
                                "read_horizon": read_index + 1,
                                "write_horizon": write_index + 1,
                                "age": age,
                                "head": head,
                                "key_channel": key_channel,
                                "theta_rad_per_step": float(theta[head, key_channel]),
                                "coefficient_magnitude": float(coefficient[sample, head, key_channel].abs()),
                                "coefficient_phase_rad": float(torch.angle(coefficient[sample, head, key_channel])),
                                "coefficient_phase_cosine": float(torch.cos(torch.angle(coefficient[sample, head, key_channel]))),
                                "hidden_signed_projection_share": float(projection[sample, head, key_channel]),
                                "hidden_contribution_norm": float(norm[sample, head, key_channel]),
                            })
            hidden_stack = torch.stack(all_hidden, dim=1)
            projection_stack = torch.stack(all_projection, dim=1)
            norm_stack = torch.stack(all_norm, dim=1)
            coefficient_magnitude = torch.stack(all_coefficient_magnitude, dim=1)
            coefficient_phase = torch.stack(all_coefficient_phase, dim=1)
            reconstructed = hidden_stack.sum(dim=(1, 2, 3))
            error = (reconstructed - total_delta).abs().amax(dim=-1)
            flat_projection = projection_stack.flatten(1)
            flat_abs_projection = flat_projection.abs()
            positive = flat_projection.clamp_min(0).sum(dim=-1)
            negative = (-flat_projection.clamp_max(0)).sum(dim=-1)
            flat_norm = norm_stack.flatten(1)
            weighted_phase_vector = (
                coefficient_magnitude * torch.exp(1j * coefficient_phase)
            ).flatten(1).sum(dim=-1)
            phase_coherence = weighted_phase_vector.abs() / coefficient_magnitude.flatten(1).sum(dim=-1).clamp_min(1e-30)
            current_abs = flat_abs_projection[:, -transition.heads * transition.key_dim :].sum(dim=-1)
            total_abs = flat_abs_projection.sum(dim=-1).clamp_min(1e-30)
            for sample in range(SAMPLES):
                summary_rows.append({
                    "sample": sample,
                    "read_horizon": read_index + 1,
                    "channel_write_terms": (read_index + 1) * transition.heads * transition.key_dim,
                    "positive_projection_mass": float(positive[sample]),
                    "negative_projection_mass": float(negative[sample]),
                    "gross_projection_mass": float(positive[sample] + negative[sample]),
                    "negative_to_positive_ratio": float(negative[sample] / positive[sample].clamp_min(1e-30)),
                    "effective_abs_projection_terms": float(effective_count(flat_abs_projection)[sample]),
                    "dominant_abs_projection_share": float(flat_abs_projection[sample].max() / total_abs[sample]),
                    "current_write_abs_projection_share": float(current_abs[sample] / total_abs[sample]),
                    "effective_norm_terms": float(effective_count(flat_norm)[sample]),
                    "coefficient_phase_coherence": float(phase_coherence[sample]),
                    "max_reconstruction_error": float(error[sample]),
                })
            state = result.state

    RECORD.mkdir(parents=True, exist_ok=True)
    maximum_error = max(row["max_reconstruction_error"] for row in summary_rows)
    if maximum_error > 1e-5:
        raise RuntimeError(
            f"channel decomposition reconstruction error {maximum_error}"
        )
    write_rows(RECORD / "frequencies.tsv", frequency_rows)
    write_rows(RECORD / "age_code_gram.tsv", age_rows)
    write_rows(RECORD / "channel_contributions.tsv", channel_rows)
    write_rows(RECORD / "summary.tsv", summary_rows)


if __name__ == "__main__":
    main()
