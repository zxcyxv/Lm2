"""Audit encoder drift and AR decision margins for complex KV checkpoints."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.evaluation import complex_self_predicted_kv_ar_rollouts
from rotlm.latent_diagnostics import (
    decision_margin_diagnostics,
    orthogonal_drift_diagnostics,
    row_relative_mse,
)
from train_byte256_simplex_tied_self_composition_h1_ce_13m import (
    load_fixed_starts,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length
import train_byte256_complex_self_predicted_kv_full_innovation_h1_online_mse_ce_13m as producer


PARENT_ID = producer.base.EXPERIMENT_ID
AUDIT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "drift-margin-audit"
)
DEFAULT_RECORD = Path("experiments/records") / AUDIT_ID
PARENT_RECORD = Path("experiments/records") / PARENT_ID
PARENT_OUTPUT = Path("outputs/experiments") / PARENT_ID
CHECKPOINT_STEPS = (100, 250, 500, 750, 1000)
DRIFT_PAIRS = (
    (100, 250),
    (250, 500),
    (500, 750),
    (750, 1000),
    (100, 1000),
    (250, 1000),
    (500, 1000),
)
EXAMPLES = 64
MICROBATCH = 2
CONTEXT = producer.base.CONTEXT
HORIZONS = producer.base.HORIZONS
ANCHOR_STRIDE = producer.base.EVAL_ANCHOR_STRIDE
WINDOW_LENGTH = CONTEXT + HORIZONS


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
        raise RuntimeError("diagnostic values are non-finite")
    requested = torch.tensor(
        [0.1, 0.5, 0.9],
        device=values.device,
        dtype=values.dtype,
    )
    p10, median, p90 = torch.quantile(values, requested)
    return {
        "mean": float(values.mean()),
        "p10": float(p10),
        "median": float(median),
        "p90": float(p90),
    }


def _prefixed_summary(
    prefix: str,
    values: torch.Tensor,
) -> dict[str, float]:
    return {
        f"{prefix}_{name}": value
        for name, value in _quantiles(values).items()
    }


def _load_checkpoint(model: torch.nn.Module, step: int) -> dict:
    path = PARENT_OUTPUT / f"step{step:04d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    if (
        config.get("tempered_readout") is not False
        or config.get("decoder_readout") != "innovation_free_prior"
        or config.get("residual_prior") is not False
    ):
        raise RuntimeError(f"step {step} is not the registered causal model")
    beta_keys = [key for key in checkpoint["model"] if "beta" in key]
    if beta_keys:
        raise RuntimeError(f"step {step} unexpectedly contains beta: {beta_keys}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


@torch.inference_mode()
def _collect_h1_targets(
    model: torch.nn.Module,
    validation,
    starts: torch.Tensor,
    microbatch: int,
) -> torch.Tensor:
    positions = torch.arange(WINDOW_LENGTH, device="cuda")
    target_positions = torch.arange(
        1,
        CONTEXT + 1,
        ANCHOR_STRIDE,
        device="cuda",
        dtype=torch.long,
    )
    rows = []
    for begin in range(0, len(starts), microbatch):
        window = windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            WINDOW_LENGTH,
        )
        encoded, _ = model.encode(window, positions)
        selected = encoded.index_select(1, target_positions)
        rows.append(selected.reshape(-1, selected.shape[-1]).cpu())
    result = torch.cat(rows, dim=0).contiguous()
    expected_rows = len(starts) * len(target_positions)
    if result.shape != (expected_rows, producer.base.WIDTH):
        raise RuntimeError(f"unexpected drift-state shape {tuple(result.shape)}")
    return result


def _drift_rows(
    states: dict[int, torch.Tensor],
) -> list[dict[str, float | int]]:
    rows = []
    for source_step, target_step in DRIFT_PAIRS:
        source = states[source_step].cuda()
        target = states[target_step].cuda()
        diagnostics = orthogonal_drift_diagnostics(source, target)
        row: dict[str, float | int] = {
            "source_step": source_step,
            "target_step": target_step,
            "delta_steps": target_step - source_step,
            "rows": source.shape[0],
            "centered_linear_cka": float(diagnostics.centered_linear_cka),
            "alignment_orthogonality_error": float(
                diagnostics.alignment_orthogonality_error
            ),
        }
        row.update(_prefixed_summary("raw_cosine", diagnostics.raw_cosine))
        row.update(
            _prefixed_summary(
                "raw_relative_mse",
                diagnostics.raw_relative_mse,
            )
        )
        row.update(
            _prefixed_summary(
                "aligned_cosine",
                diagnostics.aligned_cosine,
            )
        )
        row.update(
            _prefixed_summary(
                "aligned_relative_mse",
                diagnostics.aligned_relative_mse,
            )
        )
        row.update(_prefixed_summary("norm_ratio", diagnostics.norm_ratio))
        rows.append(row)
        del source, target, diagnostics
        torch.cuda.empty_cache()
    return rows


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _margin_audit(
    model: torch.nn.Module,
    validation,
    starts: torch.Tensor,
    microbatch: int,
) -> tuple[list[dict], list[dict], list[dict], dict[str, float]]:
    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "example",
            "anchor",
            "horizon",
            "target",
            "reference_top1",
            "candidate_top1",
            "match",
            "reference_top1_margin",
            "candidate_reference_margin",
            "reference_margin_to_candidate_competitor",
            "differential_toward_candidate_competitor",
            "crossing_excess",
            "max_absolute_logit_delta",
            "read_state_cosine",
            "read_state_relative_mse",
            "posterior_canonical_cosine",
            "posterior_canonical_relative_mse",
        )
    }
    correct = {
        name: torch.zeros(HORIZONS, dtype=torch.long)
        for name in ("prior", "posterior", "canonical", "greedy_ar")
    }
    total = torch.zeros(HORIZONS, dtype=torch.long)
    max_flip_identity_error = 0.0
    flip_classification_errors = 0
    non_tied_rows = 0
    gold_state_reencode_error = 0.0
    positions = torch.arange(WINDOW_LENGTH, device="cuda")

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
        target_positions = (
            output.anchor_indices[:, None]
            + torch.arange(1, HORIZONS + 1, device="cuda")[None]
        )
        selected_gold = full_encoded.index_select(
            1, target_positions.reshape(-1)
        ).reshape_as(output.gold_states)
        gold_state_reencode_error = max(
            gold_state_reencode_error,
            float((selected_gold - output.gold_states).abs().max()),
        )
        canonical_decoded = model.exact_inverse_selected_decode_tape_states(
            prefix_encoded,
            output.gold_states,
            positions[:CONTEXT],
            output.anchor_indices,
        )
        canonical_logits = model.token_logits(canonical_decoded).float()

        margins = decision_margin_diagnostics(
            output.ar_logits,
            output.parallel_logits,
        )
        identity_error = (
            margins.crossing_excess + margins.candidate_reference_margin
        ).abs()
        max_flip_identity_error = max(
            max_flip_identity_error,
            float(identity_error.max()),
        )
        non_tied = margins.candidate_reference_margin.abs() > 1e-7
        predicted_flip = margins.crossing_excess > 0
        observed_flip = ~margins.matches
        flip_classification_errors += int(
            predicted_flip.ne(observed_flip).logical_and(non_tied).sum()
        )
        non_tied_rows += int(non_tied.sum())

        read_cosine = F.cosine_similarity(
            output.parallel_read_states.float(),
            output.ar_read_states.float(),
            dim=-1,
        )
        read_mse = row_relative_mse(
            output.parallel_read_states,
            output.ar_read_states,
        )
        posterior_cosine = F.cosine_similarity(
            output.parallel_full_states.float(),
            output.ar_states.float(),
            dim=-1,
        )
        posterior_mse = row_relative_mse(
            output.parallel_full_states,
            output.ar_states,
        )

        batch, anchors, horizons = margins.matches.shape
        example = torch.arange(
            begin,
            begin + batch,
            device="cuda",
            dtype=torch.long,
        )[:, None, None].expand(batch, anchors, horizons)
        anchor = output.anchor_indices[None, :, None].expand(
            batch, anchors, horizons
        )
        horizon = torch.arange(
            1,
            HORIZONS + 1,
            device="cuda",
            dtype=torch.long,
        )[None, None].expand(batch, anchors, horizons)
        values = {
            "example": example,
            "anchor": anchor,
            "horizon": horizon,
            "target": output.targets,
            "reference_top1": margins.reference_top1,
            "candidate_top1": margins.candidate_top1,
            "match": margins.matches,
            "reference_top1_margin": margins.reference_top1_margin,
            "candidate_reference_margin": margins.candidate_reference_margin,
            "reference_margin_to_candidate_competitor": (
                margins.reference_margin_to_candidate_competitor
            ),
            "differential_toward_candidate_competitor": (
                margins.differential_toward_candidate_competitor
            ),
            "crossing_excess": margins.crossing_excess,
            "max_absolute_logit_delta": margins.max_absolute_logit_delta,
            "read_state_cosine": read_cosine,
            "read_state_relative_mse": read_mse,
            "posterior_canonical_cosine": posterior_cosine,
            "posterior_canonical_relative_mse": posterior_mse,
        }
        for name, value in values.items():
            collected[name].append(value.reshape(-1).cpu())

        predictions = {
            "prior": output.parallel_tokens,
            "posterior": output.full_gain_tokens,
            "canonical": canonical_logits.argmax(dim=-1),
            "greedy_ar": output.ar_tokens,
        }
        for name, prediction in predictions.items():
            correct[name] += prediction.eq(output.targets).sum(
                dim=(0, 1)
            ).cpu()
        total += torch.full(
            (HORIZONS,),
            batch * anchors,
            dtype=torch.long,
        )

    flattened = {name: torch.cat(parts) for name, parts in collected.items()}
    row_count = len(flattened["match"])
    margin_rows = [
        {
            name: (
                int(value[index])
                if name
                in {
                    "example",
                    "anchor",
                    "horizon",
                    "target",
                    "reference_top1",
                    "candidate_top1",
                    "match",
                }
                else float(value[index])
            )
            for name, value in flattened.items()
        }
        for index in range(row_count)
    ]

    metric_names = tuple(
        name
        for name in flattened
        if name
        not in {
            "example",
            "anchor",
            "horizon",
            "target",
            "reference_top1",
            "candidate_top1",
            "match",
        }
    )
    summary_rows = []
    for horizon in range(1, HORIZONS + 1):
        horizon_mask = flattened["horizon"].eq(horizon)
        for group in ("all", "match", "mismatch"):
            mask = horizon_mask
            if group == "match":
                mask = mask & flattened["match"].bool()
            elif group == "mismatch":
                mask = mask & ~flattened["match"].bool()
            row: dict[str, float | int | str] = {
                "horizon": horizon,
                "group": group,
                "rows": int(mask.sum()),
                "agreement": float(
                    flattened["match"][horizon_mask].float().mean()
                ),
            }
            for name in metric_names:
                row.update(_prefixed_summary(name, flattened[name][mask]))
            summary_rows.append(row)

    accuracy_rows = [
        {
            "horizon": horizon + 1,
            "rows": int(total[horizon]),
            **{
                f"{name}_target_accuracy": float(
                    correct[name][horizon].float() / total[horizon]
                )
                for name in correct
            },
        }
        for horizon in range(HORIZONS)
    ]
    structural = {
        "margin_rows": float(row_count),
        "non_tied_margin_rows": float(non_tied_rows),
        "flip_classification_errors": float(flip_classification_errors),
        "max_flip_identity_error": max_flip_identity_error,
        "gold_state_reencode_max_error": gold_state_reencode_error,
        "canonical_gold_min_accuracy": min(
            row["canonical_target_accuracy"] for row in accuracy_rows
        ),
    }
    return margin_rows, summary_rows, accuracy_rows, structural


def _lookup_pair(
    rows: list[dict], source: int, target: int
) -> dict:
    return next(
        row
        for row in rows
        if row["source_step"] == source and row["target_step"] == target
    )


def _lookup_margin(rows: list[dict], horizon: int, group: str) -> dict:
    return next(
        row
        for row in rows
        if row["horizon"] == horizon and row["group"] == group
    )


def _write_interpretation(
    path: Path,
    drift_rows: list[dict],
    margin_summary: list[dict],
    accuracy_rows: list[dict],
    structural: dict[str, float],
) -> None:
    late_a = _lookup_pair(drift_rows, 500, 750)
    late_b = _lookup_pair(drift_rows, 750, 1000)
    lines = [
        "# Interpretation",
        "",
        "This file is generated from the preregistered read-only audit. See "
        "the TSV files for the complete measurements.",
        "",
        "## Late checkpoint drift",
        "",
        "| Pair | Raw cosine | Aligned cosine | Aligned rel. MSE | CKA |",
        "|---|---:|---:|---:|---:|",
        (
            f"| 500->750 | {late_a['raw_cosine_mean']:.6f} | "
            f"{late_a['aligned_cosine_mean']:.6f} | "
            f"{late_a['aligned_relative_mse_mean']:.6f} | "
            f"{late_a['centered_linear_cka']:.6f} |"
        ),
        (
            f"| 750->1000 | {late_b['raw_cosine_mean']:.6f} | "
            f"{late_b['aligned_cosine_mean']:.6f} | "
            f"{late_b['aligned_relative_mse_mean']:.6f} | "
            f"{late_b['centered_linear_cka']:.6f} |"
        ),
        "",
        "## Step-1000 decision margins",
        "",
        "| H | Agreement | Match AR margin median | Mismatch AR margin median | Mismatch read cosine median | Mismatch read rel. MSE median |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in range(2, HORIZONS + 1):
        matching = _lookup_margin(margin_summary, horizon, "match")
        mismatch = _lookup_margin(margin_summary, horizon, "mismatch")
        lines.append(
            f"| {horizon} | {mismatch['agreement']:.6f} | "
            f"{matching['reference_top1_margin_median']:.6f} | "
            f"{mismatch['reference_top1_margin_median']:.6f} | "
            f"{mismatch['read_state_cosine_median']:.6f} | "
            f"{mismatch['read_state_relative_mse_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Target decoding",
            "",
            "| H | Prior accuracy | Posterior accuracy | Canonical accuracy |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in accuracy_rows:
        lines.append(
            f"| {row['horizon']} | {row['prior_target_accuracy']:.6f} | "
            f"{row['posterior_target_accuracy']:.6f} | "
            f"{row['canonical_target_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Structural checks",
            "",
            f"- flip classification errors: {int(structural['flip_classification_errors'])}",
            f"- max algebraic flip identity error: {structural['max_flip_identity_error']:.3e}",
            f"- minimum canonical target accuracy: {structural['canonical_gold_min_accuracy']:.6f}",
            "",
            "The causal interpretation and EMA decision are finalized after "
            "reviewing these generated measurements.",
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
        parser.error("microbatch must divide the registered example count")
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
    states = {}
    checkpoint_metrics = []
    for step in CHECKPOINT_STEPS:
        checkpoint = _load_checkpoint(model, step)
        states[step] = _collect_h1_targets(
            model,
            validation,
            starts,
            args.microbatch,
        )
        checkpoint_metrics.append(
            {
                "step": step,
                "parent_h1_validation_nll": checkpoint["metrics"][
                    "h1_validation_nll"
                ],
                "parent_h1_attached_relative_mse": checkpoint["metrics"][
                    "h1_attached_relative_mse"
                ],
                "parent_h2_h4_token_agreement": checkpoint["metrics"][
                    "h2_h4_token_agreement"
                ],
            }
        )
        del checkpoint
        print(f"collected fixed H1 states at step {step}", flush=True)

    drift_rows = _drift_rows(states)
    _write_dict_rows(args.record_dir / "drift.tsv", drift_rows)
    _write_dict_rows(
        args.record_dir / "checkpoint_metrics.tsv",
        checkpoint_metrics,
    )
    del states

    _load_checkpoint(model, 1000)
    margin_rows, margin_summary, accuracy_rows, structural = _margin_audit(
        model,
        validation,
        starts,
        args.microbatch,
    )
    _write_dict_rows(args.record_dir / "margin_rows.tsv", margin_rows)
    _write_dict_rows(
        args.record_dir / "margin_summary.tsv",
        margin_summary,
    )
    _write_dict_rows(args.record_dir / "accuracy.tsv", accuracy_rows)
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in structural.items():
            writer.writerow((key, value))
    _write_interpretation(
        args.record_dir / "interpretation.md",
        drift_rows,
        margin_summary,
        accuracy_rows,
        structural,
    )
    if structural["flip_classification_errors"] != 0:
        raise RuntimeError("algebraic margin decomposition missed a flip")
    if structural["canonical_gold_min_accuracy"] != 1.0:
        raise RuntimeError("canonical exact-inverse target retrieval failed")
    print(
        "audit complete: "
        f"rows={int(structural['margin_rows'])} "
        f"flip_errors={int(structural['flip_classification_errors'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
