"""Audit post-update hidden agreement by exact additive decomposition."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.evaluation import complex_self_predicted_kv_ar_rollouts
from rotlm.latent_diagnostics import (
    additive_residual_diagnostics,
    decision_margin_diagnostics,
    row_relative_mse,
)
from train_byte256_simplex_tied_self_composition_h1_ce_13m import (
    load_fixed_starts,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length
import train_byte256_complex_self_predicted_kv_post_update_full_read_h1_ema_sg_mse_ce_13m as producer


PARENT_ID = producer.base.EXPERIMENT_ID
AUDIT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "post-update-full-read-residual-shapley-audit"
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
HEADS = producer.base.HEADS
WINDOW_LENGTH = CONTEXT + HORIZONS

ROW_METRICS = (
    "prior_cosine",
    "full_cosine",
    "prior_relative_mse",
    "full_relative_mse",
    "error_reduction",
    "innovation_residual_cosine",
    "innovation_residual_norm_ratio",
    "innovation_target_energy_ratio",
    "innovation_residual_dot_ratio",
    "optimal_innovation_scale",
    "oracle_relative_mse",
    "fixed_scale_regret",
    "read_state_cosine",
    "read_state_relative_mse",
    "ar_top1_margin",
    "candidate_reference_margin",
    "crossing_excess",
    "max_absolute_logit_delta",
)
HEAD_METRICS = (
    "head_shapley_error_reduction",
    "head_residual_cosine",
    "head_residual_norm_ratio",
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
        raise RuntimeError("diagnostic values are non-finite")
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


def _prefixed(prefix: str, values: torch.Tensor) -> dict[str, float]:
    return {
        f"{prefix}_{name}": value
        for name, value in _quantiles(values).items()
    }


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty TSV")
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
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    expected = {
        "tempered_readout": False,
        "residual_prior": False,
        "recurrent_hidden_mode": "post-update-full-read",
        "decoder_readout": "innovation_free_prior",
        "latent_target": "full_model_ema",
        "inference_model": "full_model_ema",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise RuntimeError(
                f"checkpoint {name}={config.get(name)!r}, expected {value!r}"
            )
    if "ema_model" not in checkpoint:
        raise RuntimeError("checkpoint has no full-model EMA state")
    if any("beta" in name for name in checkpoint["ema_model"]):
        raise RuntimeError("EMA state unexpectedly contains beta")
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    return checkpoint


@torch.inference_mode()
def _collect_decomposition(
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
            "ar_token",
            "central_token",
            "match",
            *ROW_METRICS,
        )
    }
    head_collected: dict[str, list[torch.Tensor]] = {
        name: [] for name in HEAD_METRICS
    }
    max_additive_error = 0.0
    max_head_sum_error = 0.0
    max_shapley_sum_error = 0.0
    max_parallel_replay_error = 0.0
    max_read_replay_error = 0.0
    max_flip_identity_error = 0.0
    flip_classification_errors = 0
    non_tied_rows = 0
    positions = torch.arange(CONTEXT, device="cuda")
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
        flip_classification_errors += int(
            margins.crossing_excess.gt(0)
            .ne(~margins.matches)
            .logical_and(non_tied)
            .sum()
        )
        non_tied_rows += int(non_tied.sum())

        prefix_encoded, _ = model.encode(window[:, :CONTEXT], positions)
        roots = prefix_encoded.index_select(1, output.anchor_indices)
        state = transition.initialize(roots)
        diagnostics_by_horizon = []
        head_diagnostics_by_horizon = {
            name: [] for name in HEAD_METRICS
        }
        for horizon in range(HORIZONS):
            step = transition(state)
            innovation = transition.write(step.preliminary_hidden)
            rotated_memory = step.state.memory - innovation
            updated_query = transition.queries(step.preliminary_hidden)
            prior_values = transition.read(updated_query, rotated_memory)
            innovation_values = transition.read(updated_query, innovation)
            prior_hidden = transition.read_to_hidden(prior_values)
            head_contributions = transition.read_head_contributions(
                innovation_values
            )
            reconstructed = prior_hidden + head_contributions.sum(dim=-2)
            target = output.ar_states[:, :, horizon]
            diagnostics = additive_residual_diagnostics(
                prior_hidden,
                head_contributions,
                target,
            )

            max_additive_error = max(
                max_additive_error,
                float((reconstructed - step.state.hidden).abs().max()),
            )
            max_head_sum_error = max(
                max_head_sum_error,
                float(
                    (
                        head_contributions.sum(dim=-2)
                        - step.innovation_delta
                    ).abs().max()
                ),
            )
            max_shapley_sum_error = max(
                max_shapley_sum_error,
                float(
                    (
                        diagnostics.head_shapley_error_reduction.sum(dim=-1)
                        - diagnostics.error_reduction
                    ).abs().max()
                ),
            )
            max_parallel_replay_error = max(
                max_parallel_replay_error,
                float(
                    (
                        step.state.hidden
                        - output.parallel_full_states[:, :, horizon]
                    ).abs().max()
                ),
            )
            max_read_replay_error = max(
                max_read_replay_error,
                float(
                    (
                        step.read_hidden
                        - output.parallel_read_states[:, :, horizon]
                    ).abs().max()
                ),
            )
            diagnostics_by_horizon.append(diagnostics)
            head_diagnostics_by_horizon[
                "head_shapley_error_reduction"
            ].append(diagnostics.head_shapley_error_reduction)
            head_diagnostics_by_horizon["head_residual_cosine"].append(
                diagnostics.head_residual_cosine
            )
            head_diagnostics_by_horizon[
                "head_residual_norm_ratio"
            ].append(diagnostics.head_residual_norm_ratio)
            state = step.state

        batch, anchors, _ = margins.matches.shape
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
        row_values = {
            "example": example_ids,
            "anchor": anchor_ids,
            "horizon": horizon_ids,
            "ar_token": margins.reference_top1,
            "central_token": margins.candidate_top1,
            "match": margins.matches,
            "prior_cosine": torch.stack(
                [value.prior_cosine for value in diagnostics_by_horizon],
                dim=2,
            ),
            "full_cosine": torch.stack(
                [value.full_cosine for value in diagnostics_by_horizon],
                dim=2,
            ),
            "prior_relative_mse": torch.stack(
                [
                    value.prior_relative_mse
                    for value in diagnostics_by_horizon
                ],
                dim=2,
            ),
            "full_relative_mse": torch.stack(
                [value.full_relative_mse for value in diagnostics_by_horizon],
                dim=2,
            ),
            "error_reduction": torch.stack(
                [value.error_reduction for value in diagnostics_by_horizon],
                dim=2,
            ),
            "innovation_residual_cosine": torch.stack(
                [
                    value.innovation_residual_cosine
                    for value in diagnostics_by_horizon
                ],
                dim=2,
            ),
            "innovation_residual_norm_ratio": torch.stack(
                [
                    value.innovation_residual_norm_ratio
                    for value in diagnostics_by_horizon
                ],
                dim=2,
            ),
            "innovation_target_energy_ratio": torch.stack(
                [
                    value.innovation_target_energy_ratio
                    for value in diagnostics_by_horizon
                ],
                dim=2,
            ),
            "innovation_residual_dot_ratio": torch.stack(
                [
                    value.innovation_residual_dot_ratio
                    for value in diagnostics_by_horizon
                ],
                dim=2,
            ),
            "optimal_innovation_scale": torch.stack(
                [
                    value.optimal_innovation_scale
                    for value in diagnostics_by_horizon
                ],
                dim=2,
            ),
            "oracle_relative_mse": torch.stack(
                [
                    value.oracle_relative_mse
                    for value in diagnostics_by_horizon
                ],
                dim=2,
            ),
            "fixed_scale_regret": torch.stack(
                [value.fixed_scale_regret for value in diagnostics_by_horizon],
                dim=2,
            ),
            "read_state_cosine": F.cosine_similarity(
                output.parallel_read_states.float(),
                output.ar_read_states.float(),
                dim=-1,
            ),
            "read_state_relative_mse": row_relative_mse(
                output.parallel_read_states,
                output.ar_read_states,
            ),
            "ar_top1_margin": margins.reference_top1_margin,
            "candidate_reference_margin": margins.candidate_reference_margin,
            "crossing_excess": margins.crossing_excess,
            "max_absolute_logit_delta": margins.max_absolute_logit_delta,
        }
        for name, value in row_values.items():
            collected[name].append(value.reshape(-1).cpu())
        for name, parts in head_diagnostics_by_horizon.items():
            value = torch.stack(parts, dim=2)
            head_collected[name].append(value.reshape(-1, HEADS).cpu())

    flattened = {name: torch.cat(parts) for name, parts in collected.items()}
    flattened_heads = {
        name: torch.cat(parts) for name, parts in head_collected.items()
    }
    row_count = len(flattened["match"])
    expected_rows = EXAMPLES * len(range(0, CONTEXT, ANCHOR_STRIDE)) * HORIZONS
    if row_count != expected_rows:
        raise RuntimeError(f"collected {row_count} rows, expected {expected_rows}")
    if any(value.shape[0] != row_count for value in flattened_heads.values()):
        raise RuntimeError("head diagnostics and row diagnostics are misaligned")

    integer_fields = {
        "example",
        "anchor",
        "horizon",
        "ar_token",
        "central_token",
        "match",
    }
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

    masks = [
        (str(horizon), flattened["horizon"].eq(horizon))
        for horizon in range(1, HORIZONS + 1)
    ]
    masks.append(("2-4", flattened["horizon"].ge(2)))
    summary_records = []
    for horizon, horizon_mask in masks:
        for group in ("all", "match", "mismatch"):
            mask = horizon_mask
            if group == "match":
                mask = mask & flattened["match"].bool()
            elif group == "mismatch":
                mask = mask & ~flattened["match"].bool()
            record: dict[str, float | int | str] = {
                "horizon": horizon,
                "group": group,
                "rows": int(mask.sum()),
                "agreement": float(
                    flattened["match"][horizon_mask].float().mean()
                ),
                "innovation_helpful_fraction": float(
                    flattened["error_reduction"][mask].gt(0).float().mean()
                ),
                "negative_optimal_scale_fraction": float(
                    flattened["optimal_innovation_scale"][mask]
                    .lt(0)
                    .float()
                    .mean()
                ),
            }
            innovation_energy = flattened[
                "innovation_target_energy_ratio"
            ][mask]
            residual_dot = flattened[
                "innovation_residual_dot_ratio"
            ][mask]
            global_scale = float(
                residual_dot.sum()
                / innovation_energy.sum().clamp_min(1e-12)
            )
            record["global_optimal_innovation_scale"] = global_scale
            record["global_oracle_relative_mse"] = float(
                (
                    flattened["prior_relative_mse"][mask]
                    - 2.0 * global_scale * residual_dot
                    + global_scale * global_scale * innovation_energy
                ).mean()
            )
            for name in ROW_METRICS:
                record.update(_prefixed(name, flattened[name][mask]))
            summary_records.append(record)

    head_records = []
    for horizon, horizon_mask in masks:
        for head in range(HEADS):
            for group in ("all", "match", "mismatch"):
                mask = horizon_mask
                if group == "match":
                    mask = mask & flattened["match"].bool()
                elif group == "mismatch":
                    mask = mask & ~flattened["match"].bool()
                shapley = flattened_heads[
                    "head_shapley_error_reduction"
                ][mask, head]
                record = {
                    "horizon": horizon,
                    "head": head,
                    "group": group,
                    "rows": int(mask.sum()),
                    "positive_shapley_fraction": float(
                        shapley.gt(0).float().mean()
                    ),
                }
                for name in HEAD_METRICS:
                    record.update(
                        _prefixed(name, flattened_heads[name][mask, head])
                    )
                head_records.append(record)

    structural = {
        "rows": float(row_count),
        "non_tied_margin_rows": float(non_tied_rows),
        "flip_classification_errors": float(flip_classification_errors),
        "max_flip_identity_error": max_flip_identity_error,
        "max_additive_reconstruction_error": max_additive_error,
        "max_head_sum_error": max_head_sum_error,
        "max_shapley_sum_error": max_shapley_sum_error,
        "max_parallel_replay_error": max_parallel_replay_error,
        "max_read_replay_error": max_read_replay_error,
    }
    return row_records, summary_records, head_records, structural


@torch.inference_mode()
def _same_weight_counterfactual(
    model: torch.nn.Module,
    validation,
    starts: torch.Tensor,
    microbatch: int,
) -> tuple[list[dict], dict[str, float]]:
    transition = model.complex_self_prediction
    registered_mode = transition.recurrent_hidden_mode
    modes = ("post-update-full-read", "split-query-residual")
    agreement = {
        mode: torch.zeros(HORIZONS, dtype=torch.float64) for mode in modes
    }
    exact = {mode: 0 for mode in modes}
    total = 0
    max_ar_reference_error = 0.0
    max_h1_central_error = 0.0

    try:
        for begin in range(0, len(starts), microbatch):
            window = windows_of_length(
                validation,
                starts[begin : begin + microbatch],
                WINDOW_LENGTH,
            )
            outputs = {}
            for mode in modes:
                transition.recurrent_hidden_mode = mode
                outputs[mode] = complex_self_predicted_kv_ar_rollouts(
                    model,
                    window,
                    prefix_length=CONTEXT,
                    horizons=HORIZONS,
                    anchor_stride=ANCHOR_STRIDE,
                )
                matches = outputs[mode].parallel_tokens.eq(
                    outputs[mode].ar_tokens
                )
                agreement[mode] += matches.double().sum(dim=(0, 1)).cpu()
                exact[mode] += int(matches.all(dim=-1).sum())
            rows = window.shape[0] * outputs[modes[0]].anchor_indices.numel()
            total += rows
            max_ar_reference_error = max(
                max_ar_reference_error,
                float(
                    (
                        outputs[modes[0]].ar_logits
                        - outputs[modes[1]].ar_logits
                    ).abs().max()
                ),
            )
            max_h1_central_error = max(
                max_h1_central_error,
                float(
                    (
                        outputs[modes[0]].parallel_logits[:, :, 0]
                        - outputs[modes[1]].parallel_logits[:, :, 0]
                    ).abs().max()
                ),
            )
    finally:
        transition.recurrent_hidden_mode = registered_mode

    records = []
    for mode in modes:
        means = agreement[mode] / total
        records.append(
            {
                "mode": mode,
                "rows": total,
                "h1_agreement": float(means[0]),
                "h2_agreement": float(means[1]),
                "h3_agreement": float(means[2]),
                "h4_agreement": float(means[3]),
                "h2_h4_agreement": float(means[1:].mean()),
                "exact_block_agreement": exact[mode] / total,
            }
        )
    structural = {
        "counterfactual_ar_reference_max_logit_error": max_ar_reference_error,
        "counterfactual_h1_central_max_logit_error": max_h1_central_error,
    }
    return records, structural


def _lookup(rows: list[dict], horizon: str, group: str) -> dict:
    return next(
        row
        for row in rows
        if row["horizon"] == horizon and row["group"] == group
    )


def _write_interpretation(
    path: Path,
    summary_rows: list[dict],
    head_rows: list[dict],
    counterfactual_rows: list[dict],
    structural: dict[str, float],
) -> None:
    lines = [
        "# Interpretation",
        "",
        "This audit uses the model's own greedy-AR canonical hidden as the "
        "reference. Held-out continuation states are not the agreement target.",
        "",
        "## Innovation versus canonical AR residual",
        "",
        "| H | Agreement | Prior rel. MSE | Full rel. MSE | Error reduction | Helpful rows | Residual cosine | Global scale | Global oracle MSE | Rowwise oracle MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in ("1", "2", "3", "4", "2-4"):
        row = _lookup(summary_rows, horizon, "all")
        lines.append(
            f"| {horizon} | {row['agreement']:.6f} | "
            f"{row['prior_relative_mse_mean']:.6f} | "
            f"{row['full_relative_mse_mean']:.6f} | "
            f"{row['error_reduction_mean']:.6f} | "
            f"{row['innovation_helpful_fraction']:.6f} | "
            f"{row['innovation_residual_cosine_mean']:.6f} | "
            f"{row['global_optimal_innovation_scale']:.6f} | "
            f"{row['global_oracle_relative_mse']:.6f} | "
            f"{row['oracle_relative_mse_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Agreement stratification over H2--H4",
            "",
            "| Group | Rows | Full rel. MSE | Innovation reduction | Residual cosine | AR margin | Max logit delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in ("match", "mismatch"):
        row = _lookup(summary_rows, "2-4", group)
        lines.append(
            f"| {group} | {row['rows']} | "
            f"{row['full_relative_mse_mean']:.6f} | "
            f"{row['error_reduction_mean']:.6f} | "
            f"{row['innovation_residual_cosine_mean']:.6f} | "
            f"{row['ar_top1_margin_median']:.6f} | "
            f"{row['max_absolute_logit_delta_median']:.6f} |"
        )

    h2_h4_all_heads = [
        row
        for row in head_rows
        if row["horizon"] == "2-4" and row["group"] == "all"
    ]
    ranked_heads = sorted(
        h2_h4_all_heads,
        key=lambda row: row["head_shapley_error_reduction_mean"],
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## Head-level exact Shapley allocation over H2--H4",
            "",
            "| Head | Mean error reduction | Positive rows | Residual cosine | Norm ratio |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranked_heads:
        lines.append(
            f"| {row['head']} | "
            f"{row['head_shapley_error_reduction_mean']:.6f} | "
            f"{row['positive_shapley_fraction']:.6f} | "
            f"{row['head_residual_cosine_mean']:.6f} | "
            f"{row['head_residual_norm_ratio_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Same-weight recurrent-form counterfactual",
            "",
            "| Mode | H2 | H3 | H4 | H2--H4 | Exact block |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in counterfactual_rows:
        lines.append(
            f"| {row['mode']} | {row['h2_agreement']:.6f} | "
            f"{row['h3_agreement']:.6f} | {row['h4_agreement']:.6f} | "
            f"{row['h2_h4_agreement']:.6f} | "
            f"{row['exact_block_agreement']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Structural checks",
            "",
            f"- additive reconstruction max error: {structural['max_additive_reconstruction_error']:.3e}",
            f"- head Shapley sum max error: {structural['max_shapley_sum_error']:.3e}",
            f"- non-tied flip classification errors: {int(structural['flip_classification_errors'])}",
            f"- counterfactual AR-reference max logit error: {structural['counterfactual_ar_reference_max_logit_error']:.3e}",
            "",
            "Causal conclusions are finalized after reviewing the complete TSVs.",
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
    checkpoint = _load_ema_checkpoint(model)
    row_records, summary_records, head_records, structural = (
        _collect_decomposition(model, validation, starts, args.microbatch)
    )
    counterfactual_records, counterfactual_structural = (
        _same_weight_counterfactual(
            model,
            validation,
            starts,
            args.microbatch,
        )
    )
    structural.update(counterfactual_structural)
    structural["parent_h2_h4_agreement"] = checkpoint["metrics"][
        "h2_h4_token_agreement"
    ]

    _write_dict_rows(args.record_dir / "residual_rows.tsv", row_records)
    _write_dict_rows(args.record_dir / "residual_summary.tsv", summary_records)
    _write_dict_rows(args.record_dir / "head_shapley.tsv", head_records)
    _write_dict_rows(args.record_dir / "counterfactual.tsv", counterfactual_records)
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for name, value in structural.items():
            writer.writerow((name, value))
    _write_interpretation(
        args.record_dir / "interpretation.md",
        summary_records,
        head_records,
        counterfactual_records,
        structural,
    )

    tolerance = 1e-5
    exact_checks = (
        structural["max_additive_reconstruction_error"],
        structural["max_head_sum_error"],
        structural["max_shapley_sum_error"],
        structural["max_parallel_replay_error"],
        structural["max_read_replay_error"],
    )
    if max(exact_checks) >= tolerance:
        raise RuntimeError(f"additive structural check exceeded {tolerance}")
    if structural["flip_classification_errors"] != 0:
        raise RuntimeError("decision-margin decomposition missed a flip")
    if structural["counterfactual_ar_reference_max_logit_error"] >= tolerance:
        raise RuntimeError("counterfactual changed the greedy-AR reference")
    if structural["counterfactual_h1_central_max_logit_error"] >= tolerance:
        raise RuntimeError("counterfactual changed H1 central CE logits")
    print(
        "audit complete: "
        f"rows={int(structural['rows'])} "
        f"max_additive_error={structural['max_additive_reconstruction_error']:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
