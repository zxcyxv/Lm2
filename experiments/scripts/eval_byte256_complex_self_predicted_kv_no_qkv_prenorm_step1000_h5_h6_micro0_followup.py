"""Localize the second-dominant native step-1000 H5/H6 microbatch."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

import eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_h5_h6_localization as primary


EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-"
    "step1000-h5-h6-micro0-followup"
)
RECORD_ROOT = Path("experiments/records") / EXPERIMENT_ID
OUTPUT_ROOT = Path("outputs/experiments") / EXPERIMENT_ID
PRIMARY_RECORD = primary.RECORD_ROOT
SELECTED_MICROBATCH = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def all_gradient_row(
    rows: list[dict[str, object]],
    *,
    scope_level: str,
    scope_id: str,
    component: str,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row.get("scope_level") == scope_level
        and row.get("scope_id") == scope_id
        and row.get("loss_component") == component
        and row.get("parameter_group") == "all"
    )


def matching_alignment(
    rows: list[dict[str, object]],
    *,
    scope_level: str,
    scope_id: str,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["scope_level"] == scope_level
        and row["scope_id"] == scope_id
        and row["source_component"] == "h5+h6"
        and row["reference_component"] == "h5+h6"
    )


def primary_micro0_norms() -> dict[str, float]:
    with (PRIMARY_RECORD / "scope_gradients.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        component: float(
            next(
                row
                for row in rows
                if row["scope_level"] == "physical_microbatch"
                and row["scope_id"] == "micro0"
                and row["loss_component"] == component
                and row["parameter_group"] == "all"
            )["gradient_l2"]
        )
        for component in ("h5", "h6", "h5+h6")
    }


def selected_anchor_pattern_rows(
    anchor_metadata: list[dict[str, object]],
    *,
    source: str,
    selected_row: int,
    selected_anchor: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected_rows = [
        row
        for row in anchor_metadata
        if int(row["global_row"]) == selected_row
    ]
    for horizon in primary.SELECTED_HORIZONS:
        horizon_rows = [
            row for row in selected_rows if int(row["horizon"]) == horizon
        ]
        selected = next(
            row
            for row in horizon_rows
            if int(row["anchor_ordinal"]) == selected_anchor
        )
        others = [
            row
            for row in horizon_rows
            if int(row["anchor_ordinal"]) != selected_anchor
        ]
        for tensor in (
            "p",
            "innovation_w",
            "s_raw",
            "z_raw",
            "stored_s",
            "stored_z",
            "decoded_hidden",
            "logits",
            "z_boundary_denominator",
            "s_boundary_denominator",
        ):
            field = f"{tensor}_rms"
            values = torch.tensor([float(row[field]) for row in others])
            median = float(values.median())
            selected_value = float(selected[field])
            rows.append(
                {
                    "source": source,
                    "global_row": selected_row,
                    "anchor_ordinal": selected_anchor,
                    "horizon": horizon,
                    "tensor": tensor,
                    "selected_anchor_rms": selected_value,
                    "other_anchor_median_rms": median,
                    "selected_to_other_median_ratio": selected_value
                    / max(median, 1e-30),
                    "input_token_id": selected["input_token_id"],
                    "target_token_id": selected["target_token_id"],
                    "nll": selected["nll"],
                    "max_probability": selected["max_probability"],
                }
            )
    return rows


def load_primary_pattern_rows() -> list[dict[str, object]]:
    with (PRIMARY_RECORD / "anchor_metadata.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return selected_anchor_pattern_rows(
        rows,
        source="primary_micro3",
        selected_row=55,
        selected_anchor=0,
    )


def main() -> None:
    args = parse_args()
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered follow-up")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    payload = primary.audit.load_payload(
        primary.audit.SOURCE_OUTPUT / "step1000.pt"
    )
    window, _ = primary.audit.exact_next_batch(payload)
    model = primary.audit.checkpoint_model(payload, device)

    full_runs = {
        horizon: primary.audit.run_gradient(
            model,
            window,
            microbatch=primary.PHYSICAL_MICROBATCH,
            detach=False,
            sqrt_reads=False,
            horizon=horizon,
            contribution_to_total=True,
        )
        for horizon in primary.SELECTED_HORIZONS
    }
    full_gradients = {
        horizon: full_runs[horizon].gradients
        for horizon in primary.SELECTED_HORIZONS
    }
    references = primary.component_specs(full_gradients)

    start = SELECTED_MICROBATCH * primary.PHYSICAL_MICROBATCH
    micro_window = window[start : start + primary.PHYSICAL_MICROBATCH]
    micro_gradients, micro_ce, micro_weighted = primary.selected_gradient_pair(
        model, micro_window
    )
    registered_norms = primary_micro0_norms()
    reproduced_norms = {
        component: math.sqrt(primary.vector_squared(spec))
        for component, spec in primary.component_specs(micro_gradients).items()
    }
    primary_metric_errors = {
        component: abs(reproduced_norms[component] - registered_norms[component])
        / max(registered_norms[component], 1e-30)
        for component in registered_norms
    }
    if max(primary_metric_errors.values()) >= 5e-4:
        raise RuntimeError(
            f"micro0 does not reproduce the primary record: {primary_metric_errors}"
        )

    gradient_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    primary.append_scope_rows(
        gradient_rows,
        alignment_rows,
        scope_level="physical_microbatch",
        scope_id="micro0",
        gradients=micro_gradients,
        unweighted_ce=micro_ce,
        weighted_loss=micro_weighted,
        references=references,
        microbatch_index=SELECTED_MICROBATCH,
    )

    row_sum = {
        horizon: primary.zero_vector_like(full_gradients[horizon])
        for horizon in primary.SELECTED_HORIZONS
    }
    dominant_row = -1
    dominant_row_norm = -1.0
    for local_row in range(primary.PHYSICAL_MICROBATCH):
        global_row = start + local_row
        gradients, ce_values, weighted_values = primary.selected_gradient_pair(
            model,
            micro_window,
            row_ordinal=local_row,
        )
        for horizon in primary.SELECTED_HORIZONS:
            primary.add_vector_(row_sum[horizon], gradients[horizon])
        combined_norm = math.sqrt(
            primary.vector_squared(primary.component_specs(gradients)["h5+h6"])
        )
        if combined_norm > dominant_row_norm:
            dominant_row_norm = combined_norm
            dominant_row = global_row
        primary.append_scope_rows(
            gradient_rows,
            alignment_rows,
            scope_level="row",
            scope_id=f"row{global_row}",
            gradients=gradients,
            unweighted_ce=ce_values,
            weighted_loss=weighted_values,
            references=references,
            microbatch_index=SELECTED_MICROBATCH,
            global_row=global_row,
            local_row=local_row,
        )
        del gradients
    row_errors = {
        horizon: primary.relative_vector_error(
            row_sum[horizon], micro_gradients[horizon]
        )
        for horizon in primary.SELECTED_HORIZONS
    }
    if max(row_errors.values()) >= 5e-4:
        raise RuntimeError(f"row reconstruction failed: {row_errors}")

    anchor_sum = {
        horizon: primary.zero_vector_like(full_gradients[horizon])
        for horizon in primary.SELECTED_HORIZONS
    }
    dominant_anchor = -1
    dominant_anchor_norm = -1.0
    local_dominant_row = dominant_row - start
    for anchor_ordinal in range(16):
        gradients, ce_values, weighted_values = primary.selected_gradient_pair(
            model,
            micro_window,
            row_ordinal=local_dominant_row,
            anchor_ordinal=anchor_ordinal,
        )
        for horizon in primary.SELECTED_HORIZONS:
            primary.add_vector_(anchor_sum[horizon], gradients[horizon])
        combined_norm = math.sqrt(
            primary.vector_squared(primary.component_specs(gradients)["h5+h6"])
        )
        if combined_norm > dominant_anchor_norm:
            dominant_anchor_norm = combined_norm
            dominant_anchor = anchor_ordinal
        primary.append_scope_rows(
            gradient_rows,
            alignment_rows,
            scope_level="anchor",
            scope_id=f"row{dominant_row}.anchor{anchor_ordinal}",
            gradients=gradients,
            unweighted_ce=ce_values,
            weighted_loss=weighted_values,
            references=references,
            microbatch_index=SELECTED_MICROBATCH,
            global_row=dominant_row,
            local_row=local_dominant_row,
            anchor_ordinal=anchor_ordinal,
        )
        del gradients
    dominant_row_gradients, _, _ = primary.selected_gradient_pair(
        model,
        micro_window,
        row_ordinal=local_dominant_row,
    )
    anchor_errors = {
        horizon: primary.relative_vector_error(
            anchor_sum[horizon], dominant_row_gradients[horizon]
        )
        for horizon in primary.SELECTED_HORIZONS
    }
    if max(anchor_errors.values()) >= 5e-4:
        raise RuntimeError(f"anchor reconstruction failed: {anchor_errors}")

    forward_rows, anchor_metadata = primary.dominant_microbatch_forward_metrics(
        model,
        window,
        dominant_microbatch=SELECTED_MICROBATCH,
        dominant_global_row=dominant_row,
    )
    pattern_rows = selected_anchor_pattern_rows(
        anchor_metadata,
        source="followup_micro0",
        selected_row=dominant_row,
        selected_anchor=dominant_anchor,
    )
    pattern_rows.extend(load_primary_pattern_rows())

    for horizon in primary.SELECTED_HORIZONS:
        gradient_rows.append(
            {
                "scope_level": "validation",
                "scope_id": "row_sum_vs_micro0",
                "microbatch_index": SELECTED_MICROBATCH,
                "loss_component": f"h{horizon}",
                "parameter_group": "all",
                "reconstruction_relative_error": row_errors[horizon],
            }
        )
        gradient_rows.append(
            {
                "scope_level": "validation",
                "scope_id": "anchor_sum_vs_dominant_row",
                "microbatch_index": SELECTED_MICROBATCH,
                "global_row": dominant_row,
                "loss_component": f"h{horizon}",
                "parameter_group": "all",
                "reconstruction_relative_error": anchor_errors[horizon],
            }
        )

    primary.audit.write_tsv(args.record_dir / "scope_gradients.tsv", gradient_rows)
    primary.audit.write_tsv(
        args.record_dir / "gradient_alignments.tsv", alignment_rows
    )
    primary.audit.write_tsv(
        args.record_dir / "row_forward_metrics.tsv", forward_rows
    )
    primary.audit.write_tsv(
        args.record_dir / "anchor_metadata.tsv", anchor_metadata
    )
    primary.audit.write_tsv(
        args.record_dir / "pattern_comparison.tsv", pattern_rows
    )

    row_summaries = []
    for global_row in range(start, start + primary.PHYSICAL_MICROBATCH):
        gradient = all_gradient_row(
            gradient_rows,
            scope_level="row",
            scope_id=f"row{global_row}",
            component="h5+h6",
        )
        alignment = matching_alignment(
            alignment_rows,
            scope_level="row",
            scope_id=f"row{global_row}",
        )
        row_summaries.append(
            (
                global_row,
                float(gradient["gradient_l2"]),
                float(gradient["unweighted_ce_mean"]),
                float(alignment["cosine"]),
                float(alignment["projection_coefficient"]),
            )
        )
    row_summaries.sort(key=lambda item: item[1], reverse=True)
    anchor_summaries = []
    for anchor in range(16):
        gradient = all_gradient_row(
            gradient_rows,
            scope_level="anchor",
            scope_id=f"row{dominant_row}.anchor{anchor}",
            component="h5+h6",
        )
        alignment = matching_alignment(
            alignment_rows,
            scope_level="anchor",
            scope_id=f"row{dominant_row}.anchor{anchor}",
        )
        anchor_summaries.append(
            (
                anchor,
                float(gradient["gradient_l2"]),
                float(gradient["unweighted_ce_mean"]),
                float(alignment["cosine"]),
                float(alignment["projection_coefficient"]),
            )
        )
    anchor_summaries.sort(key=lambda item: item[1], reverse=True)

    def pattern(source: str, horizon: int, tensor: str) -> dict[str, object]:
        return next(
            row
            for row in pattern_rows
            if row["source"] == source
            and int(row["horizon"]) == horizon
            and row["tensor"] == tensor
        )

    selected_metadata = next(
        row
        for row in anchor_metadata
        if int(row["global_row"]) == dominant_row
        and int(row["anchor_ordinal"]) == dominant_anchor
        and int(row["horizon"]) == 5
    )
    top_rows = "\n".join(
        f"- row {row}: L2 `{norm:.9g}`, CE `{ce:.9g}`, cosine "
        f"`{cosine:.9g}`, full projection coefficient `{coefficient:.9g}`"
        for row, norm, ce, cosine, coefficient in row_summaries[:5]
    )
    top_anchors = "\n".join(
        f"- anchor {anchor}: L2 `{norm:.9g}`, CE `{ce:.9g}`, cosine "
        f"`{cosine:.9g}`, full projection coefficient `{coefficient:.9g}`"
        for anchor, norm, ce, cosine, coefficient in anchor_summaries[:5]
    )
    comparison_lines = []
    for source in ("followup_micro0", "primary_micro3"):
        h5_z = pattern(source, 5, "z_boundary_denominator")
        h5_s = pattern(source, 5, "s_boundary_denominator")
        h6_p = pattern(source, 6, "p")
        h6_w = pattern(source, 6, "innovation_w")
        h6_z = pattern(source, 6, "z_raw")
        h6_s = pattern(source, 6, "s_raw")
        comparison_lines.append(
            f"- {source}: H5 Z/S denominator ratios "
            f"`{float(h5_z['selected_to_other_median_ratio']):.6g}` / "
            f"`{float(h5_s['selected_to_other_median_ratio']):.6g}`; H6 "
            f"P/W/Zraw/Sraw ratios "
            f"`{float(h6_p['selected_to_other_median_ratio']):.6g}` / "
            f"`{float(h6_w['selected_to_other_median_ratio']):.6g}` / "
            f"`{float(h6_z['selected_to_other_median_ratio']):.6g}` / "
            f"`{float(h6_s['selected_to_other_median_ratio']):.6g}`"
        )

    interpretation = (
        "# Interpretation\n\n"
        "## Evidence boundary\n\n"
        "This follow-up was preregistered after microbatch 0 was identified as "
        "the second large physical component. It reuses the same native batch "
        "and therefore tests within-batch repetition, not population frequency "
        "or causality. Every differentiation preserves the original batch-16 "
        "forward shape.\n\n"
        "## Reconstruction\n\n"
        f"Microbatch-0 H5/H6/combined norms were "
        f"`{reproduced_norms['h5']:.9g}` / `{reproduced_norms['h6']:.9g}` / "
        f"`{reproduced_norms['h5+h6']:.9g}` and reproduced the primary TSV "
        f"within maximum relative error `{max(primary_metric_errors.values()):.3g}`. "
        f"Row reconstruction errors were `{row_errors[5]:.3g}` / "
        f"`{row_errors[6]:.3g}`; anchor reconstruction errors were "
        f"`{anchor_errors[5]:.3g}` / `{anchor_errors[6]:.3g}`.\n\n"
        "## Row localization\n\n"
        f"{top_rows}\n\n"
        f"The dominant row was `{dominant_row}`.\n\n"
        "## Anchor localization\n\n"
        f"{top_anchors}\n\n"
        f"The dominant anchor was `{dominant_anchor}` at position "
        f"`{selected_metadata['anchor_position']}`, with input token ID "
        f"`{selected_metadata['input_token_id']}`.\n\n"
        "## Direct pattern comparison\n\n"
        + "\n".join(comparison_lines)
        + "\n\nThe ratios are descriptive co-location. They do not establish that the "
        "small H5 denominator is the causal derivative path; the complete raw "
        "metrics and conflicting anchors remain in the TSV files.\n"
    )
    (args.record_dir / "interpretation.md").write_text(interpretation)
    manifest_path = args.record_dir / "manifest.md"
    manifest_path.write_text(
        manifest_path.read_text().replace(
            "- State: preregistered; not yet executed",
            "- State: completed; native microbatch-0 follow-up executed",
        )
    )
    print(
        f"micro0={reproduced_norms['h5+h6']:.6f} "
        f"dominant_row={dominant_row} dominant_anchor={dominant_anchor}",
        flush=True,
    )


if __name__ == "__main__":
    main()
