"""Audit raw memory/write norm growth in a trained 121M unitary recurrence.

The producer is intentionally read-only with respect to the source checkpoint.
It measures the exact per-step Frobenius-energy decomposition

    ||US + W||_F^2 - ||S||_F^2
      = ||W||_F^2 + 2 Re <US, W>_F,

where W = k v^dagger, on fixed validation roots.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import torch

from rotlm.training.ha_skew_window import sparse_anchor_indices
import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch_121m as run


base = run.base
SOURCE_RUN_ID = (
    "EXP-20260807-byte256-unitary-feedback-h4-width4352-121m-lr2e-4"
)
DEFAULT_CHECKPOINT = Path("outputs/experiments") / SOURCE_RUN_ID / "last.pt"
DEFAULT_STARTS = (
    Path("experiments/records") / SOURCE_RUN_ID / "validation_starts.tsv"
)
DEFAULT_RECORD = Path("experiments/records") / (
    "EXP-20260807-byte256-unitary-feedback-121m-step13428-memory-norm-audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation-starts", type=Path, default=DEFAULT_STARTS)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    for name in ("examples", "microbatch", "horizons", "anchor_stride"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def tensor_norm(values: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Use float64 reductions so the energy audit is reduction-stable."""
    if values.is_complex():
        squared = values.abs().double().square()
    else:
        squared = values.double().square()
    return squared.sum(dim=dims).sqrt()


def complex_energy(values: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return values.abs().double().square().sum(dim=dims)


def real_inner(
    left: torch.Tensor,
    right: torch.Tensor,
    dims: tuple[int, ...],
) -> torch.Tensor:
    return (left.conj() * right).real.double().sum(dim=dims)


def finite_values(rows: Iterable[dict[str, object]], key: str) -> torch.Tensor:
    values = [float(row[key]) for row in rows]
    result = torch.tensor(values, dtype=torch.float64)
    return result[torch.isfinite(result)]


def describe(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = finite_values(rows, key)
    if values.numel() == 0:
        return {
            f"{key}_mean": math.nan,
            f"{key}_std": math.nan,
            f"{key}_min": math.nan,
            f"{key}_p05": math.nan,
            f"{key}_median": math.nan,
            f"{key}_p95": math.nan,
            f"{key}_max": math.nan,
        }
    return {
        f"{key}_mean": float(values.mean()),
        f"{key}_std": float(values.std(unbiased=False)),
        f"{key}_min": float(values.min()),
        f"{key}_p05": float(torch.quantile(values, 0.05)),
        f"{key}_median": float(torch.quantile(values, 0.50)),
        f"{key}_p95": float(torch.quantile(values, 0.95)),
        f"{key}_max": float(values.max()),
    }


def fraction(rows: list[dict[str, object]], key: str) -> float:
    values = finite_values(rows, key)
    return float(values.mean()) if values.numel() else math.nan


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_registered_starts(path: Path, examples: int) -> torch.Tensor:
    """Load a deterministic prefix without weakening the source-file audit."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) < examples:
        raise RuntimeError(
            f"{path} has {len(rows)} starts, fewer than requested {examples}"
        )
    indices = [int(row["index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(f"{path} has non-contiguous validation indices")
    return torch.tensor(
        [int(row["token_start"]) for row in rows[:examples]],
        dtype=torch.long,
    )


def summarize_horizons(
    sample_rows: list[dict[str, object]],
    horizons: int,
) -> list[dict[str, object]]:
    described = (
        "previous_memory_norm",
        "rotated_memory_norm",
        "write_norm",
        "updated_memory_norm",
        "updated_over_previous_norm",
        "updated_over_incoherent_write_rss",
        "updated_over_sum_write_norm",
        "energy_delta",
        "write_energy",
        "cross_energy",
        "write_alignment_cosine",
        "q_norm",
        "k_norm",
        "v_norm",
        "raw_read_norm",
        "frob_normalized_read_norm",
        "previous_hidden_norm",
        "measurement_delta_norm",
        "updated_hidden_norm",
        "decoder_read_hidden_norm",
        "write_real_negative_fraction",
        "entrywise_cancellation_fraction",
    )
    output = []
    for horizon in range(1, horizons + 1):
        rows = [row for row in sample_rows if row["horizon"] == horizon]
        summary: dict[str, object] = {
            "checkpoint_step": rows[0]["checkpoint_step"],
            "horizon": horizon,
            "trajectories": len(rows),
        }
        for key in described:
            summary.update(describe(rows, key))
        summary.update({
            "cross_energy_negative_fraction": fraction(
                rows, "cross_energy_negative"
            ),
            "memory_energy_decreased_fraction": fraction(
                rows, "memory_energy_decreased"
            ),
            "memory_norm_decreased_fraction": fraction(
                rows, "memory_norm_decreased"
            ),
            "unitary_norm_error_max": max(
                float(row["unitary_norm_error_abs"]) for row in rows
            ),
            "energy_identity_error_max_abs": max(
                abs(float(row["energy_identity_error"])) for row in rows
            ),
            "energy_identity_error_max_relative": max(
                float(row["energy_identity_error_relative_abs"])
                for row in rows
            ),
            "outer_product_norm_error_max_relative": max(
                float(row["outer_product_norm_error_relative_abs"])
                for row in rows
            ),
            "forward_write_recompute_error_max": max(
                float(row["forward_write_recompute_error_max"])
                for row in rows
            ),
        })
        output.append(summary)
    return output


def summarize_heads(
    head_rows: list[dict[str, object]],
    horizons: int,
    heads: int,
) -> list[dict[str, object]]:
    output = []
    for horizon in range(1, horizons + 1):
        for head in range(heads):
            rows = [
                row
                for row in head_rows
                if row["horizon"] == horizon and row["head"] == head
            ]
            summary: dict[str, object] = {
                "checkpoint_step": rows[0]["checkpoint_step"],
                "horizon": horizon,
                "head": head,
                "trajectories": len(rows),
            }
            for key in (
                "previous_memory_norm",
                "write_norm",
                "updated_memory_norm",
                "energy_delta",
                "write_energy",
                "cross_energy",
                "write_alignment_cosine",
                "k_norm",
                "v_norm",
            ):
                summary.update(describe(rows, key))
            summary["cross_energy_negative_fraction"] = fraction(
                rows, "cross_energy_negative"
            )
            summary["memory_energy_decreased_fraction"] = fraction(
                rows, "memory_energy_decreased"
            )
            output.append(summary)
    return output


def render_interpretation(
    *,
    checkpoint: Path,
    checkpoint_step: int,
    examples: int,
    anchors: int,
    horizons: int,
    sample_rows: list[dict[str, object]],
    horizon_rows: list[dict[str, object]],
) -> str:
    h1 = horizon_rows[0]
    last = horizon_rows[-1]
    final_samples = [row for row in sample_rows if row["horizon"] == horizons]
    total_delta = sum(float(row["energy_delta"]) for row in sample_rows)
    total_write = sum(float(row["write_energy"]) for row in sample_rows)
    total_cross = sum(float(row["cross_energy"]) for row in sample_rows)
    identity_error = abs(total_delta - total_write - total_cross)
    identity_scale = max(abs(total_delta), total_write, abs(total_cross), 1e-30)
    all_noninitial = [row for row in sample_rows if row["horizon"] > 1]
    cross_negative = fraction(all_noninitial, "cross_energy_negative")
    energy_decreased = fraction(all_noninitial, "memory_energy_decreased")
    real_negative = finite_values(sample_rows, "write_real_negative_fraction")
    final_ratio = finite_values(final_samples, "updated_over_incoherent_write_rss")
    min_final = float(finite_values(final_samples, "updated_memory_norm").min())
    max_final = float(finite_values(final_samples, "updated_memory_norm").max())
    h1_mean = float(h1["updated_memory_norm_mean"])
    final_mean = float(last["updated_memory_norm_mean"])
    lines = [
        "# 121M current-checkpoint memory norm audit",
        "",
        "## Scope",
        "",
        f"- Source checkpoint: `{checkpoint}` (step {checkpoint_step})",
        f"- Fixed validation only: {examples} windows x {anchors} roots = "
        f"{examples * anchors} trajectories, rolled through H{horizons}",
        "- This is a forward-state audit. It does not infer causality for loss "
        "or gradient behavior.",
        "",
        "## Direct result",
        "",
        f"The raw memory Frobenius norm increased from mean `{h1_mean:.6f}` "
        f"at H1 to `{final_mean:.6f}` at H{horizons} "
        f"(`{final_mean / h1_mean:.3f}x`). At H{horizons}, individual "
        f"trajectory norms ranged from `{min_final:.6f}` to `{max_final:.6f}`.",
        "",
        f"Across H2--H{horizons}, the cross term was negative in "
        f"`{100 * cross_negative:.2f}%` of trajectory-steps, while total "
        f"memory energy actually decreased in `{100 * energy_decreased:.2f}%`. "
        "Thus writes do sometimes cancel the rotated state; whether that "
        "cancellation is strong enough to overcome the write's own positive "
        "energy is a separate question.",
        "",
        "## Exact energy decomposition",
        "",
        "Summed over all measured trajectories and horizons:",
        "",
        f"- total `Delta ||S||^2`: `{total_delta:.6e}`",
        f"- total `||kv^dagger||^2`: `{total_write:.6e}`",
        f"- total `2 Re<US,kv^dagger>`: `{total_cross:.6e}`",
        f"- decomposition closure error: `{identity_error:.6e}` "
        f"(relative `{identity_error / identity_scale:.3e}`)",
        "",
        "The first term is always non-negative. Negative/complex entries in "
        "`kv^dagger` affect the signed cross term, not the positivity of "
        "`||kv^dagger||^2`.",
        "",
        "## Coherence versus an incoherent-write baseline",
        "",
        f"At H{horizons}, `||S|| / sqrt(sum_r ||W_r||^2)` had mean "
        f"`{float(final_ratio.mean()):.6f}`, median "
        f"`{float(torch.quantile(final_ratio, 0.5)):.6f}`, and range "
        f"`[{float(final_ratio.min()):.6f}, {float(final_ratio.max()):.6f}]`. "
        "A value near one is the energy scale of mutually incoherent writes; "
        "values above one indicate net constructive cross-term accumulation.",
        "",
        "## About negative KV entries",
        "",
        f"The real part of actual write entries was negative on average in "
        f"`{100 * float(real_negative.mean()):.2f}%` of entries. This does not "
        "by itself imply Frobenius-norm cancellation: a complex matrix has no "
        "global positive/negative ordering, and norm growth is decided by the "
        "inner product with the already-rotated memory.",
        "",
        "See `metrics.tsv` for per-horizon distributions, `samples.tsv` for "
        "every trajectory-step, and `head_metrics.tsv` for channel-level "
        "decomposition.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base.SEED)

    checkpoint_stat_before = args.checkpoint.stat()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_stat_after = args.checkpoint.stat()
    before_identity = (
        checkpoint_stat_before.st_size,
        checkpoint_stat_before.st_mtime_ns,
    )
    after_identity = (
        checkpoint_stat_after.st_size,
        checkpoint_stat_after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError("checkpoint changed while it was being read")
    checkpoint_step = int(checkpoint["step"])
    model = base.make_model().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    if base.parameter_count(model) != run.EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"unexpected model size: {base.parameter_count(model)}"
        )
    device = torch.device(args.device)
    model = model.to(device)
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != "unitary-branch-normalized-residual":
        raise RuntimeError("checkpoint restored into the wrong recurrence")
    if transition.fuse_branch_memory_kernel:
        raise RuntimeError("audit requires explicit innovation diagnostics")

    validation = base.memmap("validation")
    starts = load_registered_starts(args.validation_starts, args.examples)
    anchors_tensor = sparse_anchor_indices(
        base.CONTEXT,
        args.anchor_stride,
        device=device,
    )
    positions = torch.arange(base.CONTEXT, device=device)
    sample_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for offset in range(0, args.examples, args.microbatch):
            batch_starts = starts[offset : offset + args.microbatch]
            prefix_tokens = base.windows_of_length(
                validation,
                batch_starts,
                base.CONTEXT,
            ).to(device)
            encoded, _ = model.encode(prefix_tokens, positions)
            roots = encoded.index_select(1, anchors_tensor)
            state = transition.initialize(roots)
            cumulative_write_energy = torch.zeros(
                roots.shape[:-1], dtype=torch.float64, device=device
            )
            cumulative_write_norm = torch.zeros_like(cumulative_write_energy)

            for horizon in range(1, args.horizons + 1):
                previous_memory = state.memory
                previous_hidden = state.hidden
                result = transition(state, collect_diagnostics=True)
                rotated_memory = result.rotated_memory
                write = result.innovation_memory
                if rotated_memory is None or write is None:
                    raise RuntimeError("transition omitted required diagnostics")
                updated_memory = result.state.memory
                query, keys, values = transition.project_qkv(
                    result.rotated_hidden
                )
                recomputed_write = torch.einsum(
                    "...hk,...hv->...hkv", keys, values.conj()
                )

                memory_dims = (-3, -2, -1)
                feature_dims = (-2, -1)
                previous_energy = complex_energy(previous_memory, memory_dims)
                rotated_energy = complex_energy(rotated_memory, memory_dims)
                write_energy = complex_energy(write, memory_dims)
                updated_energy = complex_energy(updated_memory, memory_dims)
                previous_norm = previous_energy.sqrt()
                rotated_norm = rotated_energy.sqrt()
                write_norm = write_energy.sqrt()
                updated_norm = updated_energy.sqrt()
                inner = real_inner(rotated_memory, write, memory_dims)
                cross_energy = 2.0 * inner
                energy_delta = updated_energy - previous_energy
                identity_error = energy_delta - write_energy - cross_energy
                identity_scale = torch.maximum(
                    torch.maximum(energy_delta.abs(), write_energy),
                    cross_energy.abs(),
                ).clamp_min(1e-30)
                cumulative_write_energy += write_energy
                cumulative_write_norm += write_norm

                q_head_norm = tensor_norm(query, (-1,))
                k_head_norm = tensor_norm(keys, (-1,))
                v_head_norm = tensor_norm(values, (-1,))
                q_norm = q_head_norm.square().sum(dim=-1).sqrt()
                k_norm = k_head_norm.square().sum(dim=-1).sqrt()
                v_norm = v_head_norm.square().sum(dim=-1).sqrt()
                outer_predicted_norm = (
                    k_head_norm.square() * v_head_norm.square()
                ).sum(dim=-1).sqrt()
                outer_error_relative = (
                    (write_norm - outer_predicted_norm).abs()
                    / write_norm.clamp_min(1e-30)
                )
                alignment = inner / (rotated_norm * write_norm).clamp_min(1e-30)
                alignment = torch.where(
                    rotated_norm > 0,
                    alignment,
                    torch.full_like(alignment, math.nan),
                )

                raw_read = transition.read(query, updated_memory)
                normalized_read = raw_read / (
                    updated_norm[..., None, None] + transition.post_norm_eps
                )
                raw_read_norm = tensor_norm(raw_read, feature_dims)
                normalized_read_norm = tensor_norm(normalized_read, feature_dims)
                previous_hidden_norm = tensor_norm(previous_hidden, (-1,))
                measurement_delta_norm = tensor_norm(
                    result.innovation_delta, (-1,)
                )
                updated_hidden_norm = tensor_norm(result.state.hidden, (-1,))
                decoder_hidden_norm = tensor_norm(result.read_hidden, (-1,))
                write_real_negative = (write.real < 0).double().mean(
                    dim=memory_dims
                )
                entrywise_cancellation = (
                    updated_memory.abs() < rotated_memory.abs()
                ).double().mean(dim=memory_dims)
                forward_write_error = (write - recomputed_write).abs().amax(
                    dim=memory_dims
                )

                updated_over_previous = torch.where(
                    previous_norm > 0,
                    updated_norm / previous_norm,
                    torch.full_like(updated_norm, math.nan),
                )
                rss_ratio = updated_norm / cumulative_write_energy.sqrt().clamp_min(
                    1e-30
                )
                sum_ratio = updated_norm / cumulative_write_norm.clamp_min(1e-30)

                # Per-head terms use the same exact energy identity.
                previous_head_energy = complex_energy(previous_memory, (-2, -1))
                write_head_energy = complex_energy(write, (-2, -1))
                updated_head_energy = complex_energy(updated_memory, (-2, -1))
                head_inner = real_inner(rotated_memory, write, (-2, -1))
                head_cross = 2.0 * head_inner
                head_delta = updated_head_energy - previous_head_energy
                head_alignment = head_inner / (
                    previous_head_energy.sqrt()
                    * write_head_energy.sqrt()
                ).clamp_min(1e-30)
                head_alignment = torch.where(
                    previous_head_energy > 0,
                    head_alignment,
                    torch.full_like(head_alignment, math.nan),
                )

                batch_size = roots.shape[0]
                anchor_count = roots.shape[1]
                for local_example in range(batch_size):
                    example = offset + local_example
                    token_start = int(batch_starts[local_example])
                    for anchor_slot in range(anchor_count):
                        index = (local_example, anchor_slot)
                        sample_rows.append({
                            "checkpoint_step": checkpoint_step,
                            "example": example,
                            "token_start": token_start,
                            "anchor_position": int(anchors_tensor[anchor_slot]),
                            "horizon": horizon,
                            "previous_memory_norm": float(previous_norm[index]),
                            "rotated_memory_norm": float(rotated_norm[index]),
                            "write_norm": float(write_norm[index]),
                            "updated_memory_norm": float(updated_norm[index]),
                            "updated_over_previous_norm": float(updated_over_previous[index]),
                            "updated_over_incoherent_write_rss": float(rss_ratio[index]),
                            "updated_over_sum_write_norm": float(sum_ratio[index]),
                            "energy_delta": float(energy_delta[index]),
                            "write_energy": float(write_energy[index]),
                            "cross_energy": float(cross_energy[index]),
                            "write_alignment_cosine": float(alignment[index]),
                            "cross_energy_negative": float(cross_energy[index] < 0),
                            "memory_energy_decreased": float(energy_delta[index] < 0),
                            "memory_norm_decreased": float(updated_norm[index] < previous_norm[index]),
                            "unitary_norm_error_abs": float((rotated_norm[index] - previous_norm[index]).abs()),
                            "energy_identity_error": float(identity_error[index]),
                            "energy_identity_error_relative_abs": float((identity_error[index].abs() / identity_scale[index])),
                            "q_norm": float(q_norm[index]),
                            "k_norm": float(k_norm[index]),
                            "v_norm": float(v_norm[index]),
                            "outer_product_norm_error_relative_abs": float(outer_error_relative[index]),
                            "forward_write_recompute_error_max": float(forward_write_error[index]),
                            "raw_read_norm": float(raw_read_norm[index]),
                            "frob_normalized_read_norm": float(normalized_read_norm[index]),
                            "previous_hidden_norm": float(previous_hidden_norm[index]),
                            "measurement_delta_norm": float(measurement_delta_norm[index]),
                            "updated_hidden_norm": float(updated_hidden_norm[index]),
                            "decoder_read_hidden_norm": float(decoder_hidden_norm[index]),
                            "write_real_negative_fraction": float(write_real_negative[index]),
                            "entrywise_cancellation_fraction": float(entrywise_cancellation[index]),
                        })
                        for head in range(transition.heads):
                            head_index = (*index, head)
                            head_rows.append({
                                "checkpoint_step": checkpoint_step,
                                "example": example,
                                "token_start": token_start,
                                "anchor_position": int(anchors_tensor[anchor_slot]),
                                "horizon": horizon,
                                "head": head,
                                "previous_memory_norm": float(previous_head_energy[head_index].sqrt()),
                                "write_norm": float(write_head_energy[head_index].sqrt()),
                                "updated_memory_norm": float(updated_head_energy[head_index].sqrt()),
                                "energy_delta": float(head_delta[head_index]),
                                "write_energy": float(write_head_energy[head_index]),
                                "cross_energy": float(head_cross[head_index]),
                                "write_alignment_cosine": float(head_alignment[head_index]),
                                "cross_energy_negative": float(head_cross[head_index] < 0),
                                "memory_energy_decreased": float(head_delta[head_index] < 0),
                                "k_norm": float(k_head_norm[head_index]),
                                "v_norm": float(v_head_norm[head_index]),
                            })
                state = result.state

    horizon_rows = summarize_horizons(sample_rows, args.horizons)
    head_summary_rows = summarize_heads(
        head_rows, args.horizons, transition.heads
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.record_dir / "samples.tsv", sample_rows)
    write_tsv(args.record_dir / "metrics.tsv", horizon_rows)
    write_tsv(args.record_dir / "head_metrics.tsv", head_summary_rows)
    write_tsv(args.record_dir / "run.tsv", [{
        "source_run_id": SOURCE_RUN_ID,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_size_bytes": checkpoint_stat_after.st_size,
        "checkpoint_mtime_ns": checkpoint_stat_after.st_mtime_ns,
        "model_parameters": base.parameter_count(model),
        "validation_starts": str(args.validation_starts),
        "validation_examples": args.examples,
        "anchor_stride": args.anchor_stride,
        "anchors_per_example": anchors_tensor.numel(),
        "horizons": args.horizons,
        "device": str(device),
        "dtype": "float32/complex64 with float64 diagnostic reductions",
        "test_split_read": False,
    }])
    interpretation = render_interpretation(
        checkpoint=args.checkpoint,
        checkpoint_step=checkpoint_step,
        examples=args.examples,
        anchors=anchors_tensor.numel(),
        horizons=args.horizons,
        sample_rows=sample_rows,
        horizon_rows=horizon_rows,
    )
    (args.record_dir / "interpretation.md").write_text(interpretation)
    print(interpretation)


if __name__ == "__main__":
    main()
