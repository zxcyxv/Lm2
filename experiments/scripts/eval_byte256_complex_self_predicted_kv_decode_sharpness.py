"""Measure whether P and Znext decode to the same token basin."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.evaluation import complex_self_predicted_kv_ar_rollouts
from train_byte256_simplex_tied_self_composition_h1_ce_13m import (
    load_fixed_starts,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length
import train_byte256_complex_self_predicted_kv_post_update_full_read_h1_ema_sg_mse_ce_13m as producer


PARENT_ID = producer.base.EXPERIMENT_ID
AUDIT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "post-update-full-read-decode-sharpness-audit"
)
DEFAULT_RECORD = Path("experiments/records") / AUDIT_ID
PARENT_RECORD = Path("experiments/records") / PARENT_ID
PARENT_OUTPUT = Path("outputs/experiments") / PARENT_ID
CHECKPOINT_STEP = 1000
EXAMPLES = 64
MICROBATCH = 2
CONTEXT = producer.base.CONTEXT
HORIZONS = producer.base.HORIZONS
ANCHOR_STRIDE = producer.base.EVAL_ANCHOR_STRIDE
WINDOW_LENGTH = CONTEXT + HORIZONS

FLOAT_METRICS = (
    "p_max_probability",
    "a_max_probability",
    "z_max_probability",
    "canonical_max_probability",
    "p_entropy",
    "a_entropy",
    "z_entropy",
    "canonical_entropy",
    "a_probability_of_p_top1",
    "z_probability_of_p_top1",
    "p_target_probability",
    "a_target_probability",
    "z_target_probability",
    "canonical_target_probability",
)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return {
            "mean": math.nan,
            "p10": math.nan,
            "median": math.nan,
            "p90": math.nan,
        }
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("sharpness diagnostic is non-finite")
    p10, median, p90 = torch.quantile(
        values,
        torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype),
    )
    return {
        "mean": float(values.mean()),
        "p10": float(p10),
        "median": float(median),
        "p90": float(p90),
    }


def _prefixed(name: str, values: torch.Tensor) -> dict[str, float]:
    return {
        f"{name}_{statistic}": value
        for statistic, value in _quantiles(values).items()
    }


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty TSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_ema_checkpoint(model: torch.nn.Module) -> dict:
    path = PARENT_OUTPUT / f"step{CHECKPOINT_STEP:04d}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if (
        config.get("recurrent_hidden_mode") != "post-update-full-read"
        or config.get("decoder_readout") != "innovation_free_prior"
        or config.get("inference_model") != "full_model_ema"
        or "ema_model" not in checkpoint
    ):
        raise RuntimeError("checkpoint does not match the registered EMA model")
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    return checkpoint


def _distribution_values(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    probabilities = F.softmax(logits.float(), dim=-1)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    top_probability, top_token = probabilities.max(dim=-1)
    return {
        "probabilities": probabilities,
        "top_probability": top_probability,
        "top_token": top_token,
        "entropy": -(probabilities * log_probabilities).sum(dim=-1),
        "target_probability": probabilities.gather(
            -1,
            targets.unsqueeze(-1),
        ).squeeze(-1),
    }


@torch.inference_mode()
def _audit(
    model: torch.nn.Module,
    validation,
    starts: torch.Tensor,
    microbatch: int,
) -> tuple[list[dict], list[dict], dict[str, float]]:
    names = (
        "example",
        "anchor",
        "horizon",
        "target",
        "p_top1",
        "a_top1",
        "z_top1",
        "canonical_top1",
        "p_a_match",
        "p_z_match",
        "a_z_match",
        "p_correct",
        "a_correct",
        "z_correct",
        "canonical_correct",
        "innovation_enters_p_basin",
        "innovation_leaves_p_basin",
        *FLOAT_METRICS,
    )
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    max_full_replay_error = 0.0
    max_p_replay_error = 0.0
    max_probability_sum_error = 0.0
    positions = torch.arange(WINDOW_LENGTH, device="cuda")
    transition = model.complex_self_prediction

    for begin in range(0, len(starts), microbatch):
        window = windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            WINDOW_LENGTH,
        )
        output = complex_self_predicted_kv_ar_rollouts(
            model,
            window,
            prefix_length=CONTEXT,
            horizons=HORIZONS,
            anchor_stride=ANCHOR_STRIDE,
        )
        full_encoded, _ = model.encode(window, positions)
        prefix_encoded = full_encoded[:, :CONTEXT]
        roots = prefix_encoded.index_select(1, output.anchor_indices)

        state = transition.initialize(roots)
        local_preinnovation_states = []
        replay_full_states = []
        replay_p_states = []
        for horizon in range(HORIZONS):
            step = transition(state)
            innovation = transition.write(step.preliminary_hidden)
            rotated_memory = step.state.memory - innovation
            updated_query = transition.queries(step.preliminary_hidden)
            preinnovation = transition.read_to_hidden(
                transition.read(updated_query, rotated_memory)
            )
            local_preinnovation_states.append(preinnovation)
            replay_full_states.append(step.state.hidden)
            replay_p_states.append(step.read_hidden)
            state = step.state
        replay_full = torch.stack(replay_full_states, dim=2)
        replay_p = torch.stack(replay_p_states, dim=2)
        preinnovation = torch.stack(local_preinnovation_states, dim=2)
        max_full_replay_error = max(
            max_full_replay_error,
            float((replay_full - output.parallel_full_states).abs().max()),
        )
        max_p_replay_error = max(
            max_p_replay_error,
            float((replay_p - output.parallel_read_states).abs().max()),
        )

        local_a_logits = []
        for horizon in range(HORIZONS):
            local_tape = output.parallel_full_states[:, :, : horizon + 1].clone()
            local_tape[:, :, -1] = preinnovation[:, :, horizon]
            decoded = model.exact_inverse_selected_decode_tape_states(
                prefix_encoded,
                local_tape,
                positions[:CONTEXT],
                output.anchor_indices,
            )[:, :, -1]
            local_a_logits.append(model.token_logits(decoded).float())
        a_logits = torch.stack(local_a_logits, dim=2)

        canonical_decoded = model.exact_inverse_selected_decode_tape_states(
            prefix_encoded,
            output.gold_states,
            positions[:CONTEXT],
            output.anchor_indices,
        )
        canonical_logits = model.token_logits(canonical_decoded).float()
        distributions = {
            "p": _distribution_values(output.parallel_logits, output.targets),
            "a": _distribution_values(a_logits, output.targets),
            "z": _distribution_values(output.full_gain_logits, output.targets),
            "canonical": _distribution_values(canonical_logits, output.targets),
        }
        for values in distributions.values():
            max_probability_sum_error = max(
                max_probability_sum_error,
                float(
                    (
                        values["probabilities"].sum(dim=-1) - 1.0
                    ).abs().max()
                ),
            )

        p_top = distributions["p"]["top_token"]
        a_top = distributions["a"]["top_token"]
        z_top = distributions["z"]["top_token"]
        canonical_top = distributions["canonical"]["top_token"]
        p_a_match = p_top.eq(a_top)
        p_z_match = p_top.eq(z_top)
        batch, anchors, _ = output.targets.shape
        horizon_ids = torch.arange(
            1,
            HORIZONS + 1,
            device="cuda",
            dtype=torch.long,
        )[None, None].expand(batch, anchors, HORIZONS)
        example_ids = torch.arange(
            begin,
            begin + batch,
            device="cuda",
            dtype=torch.long,
        )[:, None, None].expand(batch, anchors, HORIZONS)
        anchor_ids = output.anchor_indices[None, :, None].expand(
            batch,
            anchors,
            HORIZONS,
        )
        values = {
            "example": example_ids,
            "anchor": anchor_ids,
            "horizon": horizon_ids,
            "target": output.targets,
            "p_top1": p_top,
            "a_top1": a_top,
            "z_top1": z_top,
            "canonical_top1": canonical_top,
            "p_a_match": p_a_match,
            "p_z_match": p_z_match,
            "a_z_match": a_top.eq(z_top),
            "p_correct": p_top.eq(output.targets),
            "a_correct": a_top.eq(output.targets),
            "z_correct": z_top.eq(output.targets),
            "canonical_correct": canonical_top.eq(output.targets),
            "innovation_enters_p_basin": (~p_a_match) & p_z_match,
            "innovation_leaves_p_basin": p_a_match & (~p_z_match),
        }
        for prefix, distribution in distributions.items():
            values[f"{prefix}_max_probability"] = distribution[
                "top_probability"
            ]
            values[f"{prefix}_entropy"] = distribution["entropy"]
            values[f"{prefix}_target_probability"] = distribution[
                "target_probability"
            ]
        values["a_probability_of_p_top1"] = distributions["a"][
            "probabilities"
        ].gather(-1, p_top.unsqueeze(-1)).squeeze(-1)
        values["z_probability_of_p_top1"] = distributions["z"][
            "probabilities"
        ].gather(-1, p_top.unsqueeze(-1)).squeeze(-1)
        for name, value in values.items():
            collected[name].append(value.reshape(-1).cpu())

    flattened = {name: torch.cat(parts) for name, parts in collected.items()}
    row_count = len(flattened["horizon"])
    expected_rows = EXAMPLES * len(range(0, CONTEXT, ANCHOR_STRIDE)) * HORIZONS
    if row_count != expected_rows:
        raise RuntimeError(f"collected {row_count} rows, expected {expected_rows}")

    integer_fields = set(names) - set(FLOAT_METRICS)
    row_records = [
        {
            name: (
                int(value[index])
                if name in integer_fields
                else float(value[index])
            )
            for name, value in flattened.items()
        }
        for index in range(row_count)
    ]

    summary_records = []
    for horizon in range(1, HORIZONS + 1):
        horizon_mask = flattened["horizon"].eq(horizon)
        groups = {
            "all": horizon_mask,
            "p_correct": horizon_mask & flattened["p_correct"].bool(),
            "p_incorrect": horizon_mask & ~flattened["p_correct"].bool(),
            "p_z_match": horizon_mask & flattened["p_z_match"].bool(),
            "p_z_mismatch": horizon_mask & ~flattened["p_z_match"].bool(),
        }
        for group, mask in groups.items():
            record: dict[str, float | int | str] = {
                "horizon": horizon,
                "group": group,
                "rows": int(mask.sum()),
                "p_a_top1_agreement": float(
                    flattened["p_a_match"][mask].float().mean()
                ),
                "p_z_top1_agreement": float(
                    flattened["p_z_match"][mask].float().mean()
                ),
                "a_z_top1_agreement": float(
                    flattened["a_z_match"][mask].float().mean()
                ),
                "innovation_enters_p_basin_fraction": float(
                    flattened["innovation_enters_p_basin"][mask]
                    .float()
                    .mean()
                ),
                "innovation_leaves_p_basin_fraction": float(
                    flattened["innovation_leaves_p_basin"][mask]
                    .float()
                    .mean()
                ),
                "innovation_net_p_basin_gain": float(
                    flattened["p_z_match"][mask].float().mean()
                    - flattened["p_a_match"][mask].float().mean()
                ),
                "p_target_accuracy": float(
                    flattened["p_correct"][mask].float().mean()
                ),
                "a_target_accuracy": float(
                    flattened["a_correct"][mask].float().mean()
                ),
                "z_target_accuracy": float(
                    flattened["z_correct"][mask].float().mean()
                ),
                "canonical_target_accuracy": float(
                    flattened["canonical_correct"][mask].float().mean()
                ),
                "z_max_probability_ge_090": float(
                    flattened["z_max_probability"][mask]
                    .ge(0.90)
                    .float()
                    .mean()
                ),
                "z_max_probability_ge_099": float(
                    flattened["z_max_probability"][mask]
                    .ge(0.99)
                    .float()
                    .mean()
                ),
                "z_max_probability_ge_0999": float(
                    flattened["z_max_probability"][mask]
                    .ge(0.999)
                    .float()
                    .mean()
                ),
            }
            for name in FLOAT_METRICS:
                record.update(_prefixed(name, flattened[name][mask]))
            summary_records.append(record)

    structural = {
        "rows": float(row_count),
        "max_full_replay_error": max_full_replay_error,
        "max_p_replay_error": max_p_replay_error,
        "max_probability_sum_error": max_probability_sum_error,
        "canonical_top1_errors": float(
            (~flattened["canonical_correct"].bool()).sum()
        ),
        "canonical_max_probability_mean": float(
            flattened["canonical_max_probability"].mean()
        ),
    }
    return row_records, summary_records, structural


def _lookup(rows: list[dict], horizon: int, group: str) -> dict:
    return next(
        row
        for row in rows
        if row["horizon"] == horizon and row["group"] == group
    )


def _write_interpretation(
    path: Path,
    summary_rows: list[dict],
    structural: dict[str, float],
) -> None:
    lines = [
        "# Interpretation",
        "",
        "H1 is the primary basin test. H2--H4 are supplementary rollout "
        "diagnostics.",
        "",
        "## H1 basin test",
        "",
        "| Group | Rows | P target acc. | P/A top-1 | P/Znext top-1 | Innovation enters P basin | Innovation leaves P basin | P max prob. | Znext max prob. | Znext >=99% | Znext target acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", "p_correct", "p_incorrect"):
        row = _lookup(summary_rows, 1, group)
        lines.append(
            f"| {group} | {row['rows']} | "
            f"{row['p_target_accuracy']:.6f} | "
            f"{row['p_a_top1_agreement']:.6f} | "
            f"{row['p_z_top1_agreement']:.6f} | "
            f"{row['innovation_enters_p_basin_fraction']:.6f} | "
            f"{row['innovation_leaves_p_basin_fraction']:.6f} | "
            f"{row['p_max_probability_mean']:.6f} | "
            f"{row['z_max_probability_mean']:.6f} | "
            f"{row['z_max_probability_ge_099']:.6f} | "
            f"{row['z_target_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Supplementary horizons",
            "",
            "| H | P/Znext top-1 | P max prob. | Znext max prob. | Znext >=99% | P entropy | Znext entropy |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in range(1, HORIZONS + 1):
        row = _lookup(summary_rows, horizon, "all")
        lines.append(
            f"| {horizon} | {row['p_z_top1_agreement']:.6f} | "
            f"{row['p_max_probability_mean']:.6f} | "
            f"{row['z_max_probability_mean']:.6f} | "
            f"{row['z_max_probability_ge_099']:.6f} | "
            f"{row['p_entropy_mean']:.6f} | "
            f"{row['z_entropy_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Calibration and scope",
            "",
            f"- canonical target top-1 errors: {int(structural['canonical_top1_errors'])}",
            f"- canonical mean max-softmax probability: {structural['canonical_max_probability_mean']:.6f}",
            "- no matched no-MSE model was trained; this audit measures the "
            "basin reached by the MSE-trained model, not causal improvement "
            "over no-MSE training.",
            "",
            "The final conclusion is added after reviewing the complete TSVs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--examples", type=int, default=EXAMPLES)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    args = parser.parse_args()
    if args.examples != EXAMPLES:
        parser.error(f"registered audit requires exactly {EXAMPLES} examples")
    if args.microbatch < 1 or args.examples % args.microbatch:
        parser.error("microbatch must divide the example count")
    if not torch.cuda.is_available():
        parser.error("this audit requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(producer.base.SEED)
    torch.cuda.manual_seed_all(producer.base.SEED)
    validation = producer.base.memmap("validation")
    starts = load_fixed_starts(
        PARENT_RECORD / "validation_starts.tsv",
        EXAMPLES,
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)

    model = producer.base.make_model().cuda().eval()
    checkpoint = _load_ema_checkpoint(model)
    row_records, summary_records, structural = _audit(
        model,
        validation,
        starts,
        args.microbatch,
    )
    h1_all = _lookup(summary_records, 1, "all")
    structural["parent_h1_p_z_top1_agreement"] = checkpoint["metrics"][
        "h1_posterior_readout_token_agreement"
    ]
    structural["audit_h1_p_z_top1_agreement"] = h1_all[
        "p_z_top1_agreement"
    ]
    structural["parent_agreement_abs_error"] = abs(
        structural["parent_h1_p_z_top1_agreement"]
        - structural["audit_h1_p_z_top1_agreement"]
    )

    _write_dict_rows(args.record_dir / "sharpness_rows.tsv", row_records)
    _write_dict_rows(args.record_dir / "sharpness_summary.tsv", summary_records)
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for name, value in structural.items():
            writer.writerow((name, value))
    _write_interpretation(
        args.record_dir / "interpretation.md",
        summary_records,
        structural,
    )

    if max(
        structural["max_full_replay_error"],
        structural["max_p_replay_error"],
        structural["parent_agreement_abs_error"],
    ) >= 1e-5:
        raise RuntimeError("sharpness audit did not reproduce the parent path")
    if structural["max_probability_sum_error"] >= 1e-5:
        raise RuntimeError("softmax probabilities do not sum to one")
    if structural["canonical_top1_errors"] != 0:
        raise RuntimeError("canonical exact-inverse decode missed its target")
    print(
        "sharpness audit complete: "
        f"h1_p_z={h1_all['p_z_top1_agreement']:.6f} "
        f"h1_z_ge99={h1_all['z_max_probability_ge_099']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
