"""Analyze epoch-level shallow-to-deep transfer in the H4-loss feedback run."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


EPOCH_STEPS = 3_357
EPOCHS = 10
HORIZONS = 16
TRAIN_HORIZONS = 4
DEFAULT_RECORD = Path(
    "experiments/records/"
    "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "10epoch-13m"
)
DEFAULT_SCAN_METRICS = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-h16-"
    "10epoch-13m-rtx5090/epoch-checkpoint-run/metrics.tsv"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def epoch_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_step = {int(row["step"]): row for row in rows}
    expected = [EPOCH_STEPS * epoch for epoch in range(1, EPOCHS + 1)]
    missing = [step for step in expected if step not in by_step]
    if missing:
        raise RuntimeError(f"missing epoch rows: {missing}")
    return [by_step[step] for step in expected]


def mean(values) -> float:
    active = list(values)
    return sum(active) / len(active)


def pearson(left, right) -> float:
    x = list(left)
    y = list(right)
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    x_mean = mean(x)
    y_mean = mean(y)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in x)
        * sum((value - y_mean) ** 2 for value in y)
    )
    return numerator / denominator if denominator else math.nan


def ranks(values) -> list[float]:
    active = list(values)
    order = sorted(range(len(active)), key=active.__getitem__)
    result = [0.0] * len(active)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and active[order[end]] == active[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        for index in order[start:end]:
            result[index] = average_rank
        start = end
    return result


def spearman(left, right) -> float:
    return pearson(ranks(left), ranks(right))


def ols_slope(x_values, y_values) -> float:
    x = list(x_values)
    y = list(y_values)
    x_mean = mean(x)
    y_mean = mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y)
        )
        / denominator
        if denominator
        else math.nan
    )


def theil_sen_slope(x_values, y_values) -> float:
    x = list(x_values)
    y = list(y_values)
    slopes = [
        (y[right] - y[left]) / (x[right] - x[left])
        for left in range(len(x))
        for right in range(left + 1, len(x))
        if x[right] != x[left]
    ]
    return statistics.median(slopes)


def write_tsv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def nll(row: dict[str, str], horizon: int, *, scan: bool = False) -> float:
    name = f"h{horizon}_nll" if scan else f"val_h{horizon}_target_nll"
    return float(row[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_RECORD / "metrics.tsv",
    )
    parser.add_argument("--scan-metrics", type=Path, default=DEFAULT_SCAN_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    feedback = epoch_rows(read_rows(args.metrics))
    scan = epoch_rows(read_rows(args.scan_metrics))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feedback_nll = {
        horizon: [nll(row, horizon) for row in feedback]
        for horizon in range(1, HORIZONS + 1)
    }
    scan_nll = {
        horizon: [nll(row, horizon, scan=True) for row in scan]
        for horizon in range(1, HORIZONS + 1)
    }
    improvements = {
        horizon: [
            values[index - 1] - values[index]
            for index in range(1, len(values))
        ]
        for horizon, values in feedback_nll.items()
    }
    velocities = {
        horizon: [-value for value in horizon_improvements]
        for horizon, horizon_improvements in improvements.items()
    }
    accelerations = {
        horizon: [
            values[index] - values[index - 1]
            for index in range(1, len(values))
        ]
        for horizon, values in velocities.items()
    }

    epoch_output = []
    for epoch, row in enumerate(feedback, 1):
        for horizon in range(1, HORIZONS + 1):
            epoch_output.append(
                {
                    "epoch": epoch,
                    "step": int(row["step"]),
                    "horizon": horizon,
                    "supervised_horizon": horizon <= TRAIN_HORIZONS,
                    "nll": feedback_nll[horizon][epoch - 1],
                }
            )
    write_tsv(
        args.output_dir / "epoch_horizon_nll.tsv",
        ("epoch", "step", "horizon", "supervised_horizon", "nll"),
        epoch_output,
    )

    interval_output = []
    for horizon in range(1, HORIZONS + 1):
        for index, improvement in enumerate(improvements[horizon]):
            interval_output.append(
                {
                    "horizon": horizon,
                    "supervised_horizon": horizon <= TRAIN_HORIZONS,
                    "epoch_from": index + 1,
                    "epoch_to": index + 2,
                    "nll_from": feedback_nll[horizon][index],
                    "nll_to": feedback_nll[horizon][index + 1],
                    "improvement": improvement,
                    "velocity": -improvement,
                    "acceleration": (
                        accelerations[horizon][index - 1]
                        if index > 0
                        else math.nan
                    ),
                }
            )
    write_tsv(
        args.output_dir / "interval_metrics.tsv",
        (
            "horizon",
            "supervised_horizon",
            "epoch_from",
            "epoch_to",
            "nll_from",
            "nll_to",
            "improvement",
            "velocity",
            "acceleration",
        ),
        interval_output,
    )

    horizon_output = []
    for horizon in range(1, HORIZONS + 1):
        horizon_improvements = improvements[horizon]
        horizon_velocities = velocities[horizon]
        horizon_accelerations = accelerations[horizon]
        positive_weights = [max(0.0, value) for value in horizon_improvements]
        weight_sum = sum(positive_weights)
        centroid = (
            sum(
                (index + 1.5) * weight
                for index, weight in enumerate(positive_weights)
            )
            / weight_sum
            if weight_sum
            else math.nan
        )
        peak_index = max(
            range(len(horizon_improvements)),
            key=horizon_improvements.__getitem__,
        )
        horizon_output.append(
            {
                "horizon": horizon,
                "supervised_horizon": horizon <= TRAIN_HORIZONS,
                "epoch1_nll": feedback_nll[horizon][0],
                "epoch10_nll": feedback_nll[horizon][-1],
                "epoch1_to_epoch10_improvement": (
                    feedback_nll[horizon][0] - feedback_nll[horizon][-1]
                ),
                "improving_intervals": sum(
                    value > 0.0 for value in horizon_improvements
                ),
                "interval_count": len(horizon_improvements),
                "first_improvement": horizon_improvements[0],
                "last_improvement": horizon_improvements[-1],
                "positive_accelerations": sum(
                    value > 0.0 for value in horizon_accelerations
                ),
                "acceleration_count": len(horizon_accelerations),
                "mean_acceleration": mean(horizon_accelerations),
                "median_acceleration": statistics.median(
                    horizon_accelerations
                ),
                "velocity_ols_slope": ols_slope(
                    range(1, len(horizon_velocities) + 1),
                    horizon_velocities,
                ),
                "velocity_theil_sen_slope": theil_sen_slope(
                    range(1, len(horizon_velocities) + 1),
                    horizon_velocities,
                ),
                "peak_improvement_epoch_from": peak_index + 1,
                "peak_improvement_epoch_to": peak_index + 2,
                "positive_improvement_centroid": centroid,
            }
        )
    write_tsv(
        args.output_dir / "horizon_summary.tsv",
        tuple(horizon_output[0]),
        horizon_output,
    )

    comparison_output = [
        {
            "horizon": horizon,
            "feedback_supervised_horizon": horizon <= TRAIN_HORIZONS,
            "feedback_epoch1_nll": feedback_nll[horizon][0],
            "feedback_epoch10_nll": feedback_nll[horizon][-1],
            "scan_epoch1_nll": scan_nll[horizon][0],
            "scan_epoch10_nll": scan_nll[horizon][-1],
            "epoch10_feedback_minus_scan_nll": (
                feedback_nll[horizon][-1] - scan_nll[horizon][-1]
            ),
        }
        for horizon in range(1, HORIZONS + 1)
    ]
    write_tsv(
        args.output_dir / "scan_comparison.tsv",
        tuple(comparison_output[0]),
        comparison_output,
    )

    shallow = [
        mean(improvements[horizon][index] for horizon in range(1, 5))
        for index in range(EPOCHS - 1)
    ]
    deep = [
        mean(improvements[horizon][index] for horizon in range(5, 17))
        for index in range(EPOCHS - 1)
    ]
    far = [
        mean(improvements[horizon][index] for horizon in range(13, 17))
        for index in range(EPOCHS - 1)
    ]
    h16_relative_to_shallow = [
        improvements[16][index] / shallow[index]
        if shallow[index] != 0.0
        else math.nan
        for index in range(EPOCHS - 1)
    ]
    deep_relative_to_shallow = [
        deep[index] / shallow[index]
        if shallow[index] != 0.0
        else math.nan
        for index in range(EPOCHS - 1)
    ]
    deep_bounded_share = [
        deep[index] / (deep[index] + shallow[index])
        if deep[index] + shallow[index] != 0.0
        else math.nan
        for index in range(EPOCHS - 1)
    ]
    group_interval_output = [
        {
            "epoch_from": index + 1,
            "epoch_to": index + 2,
            "shallow_h1_h4_mean_improvement": shallow[index],
            "deep_h5_h16_mean_improvement": deep[index],
            "far_h13_h16_mean_improvement": far[index],
            "deep_to_shallow_relative_speed": deep_relative_to_shallow[index],
            "deep_bounded_relative_share": deep_bounded_share[index],
            "h16_to_shallow_relative_speed": h16_relative_to_shallow[index],
        }
        for index in range(EPOCHS - 1)
    ]
    write_tsv(
        args.output_dir / "group_interval_metrics.tsv",
        tuple(group_interval_output[0]),
        group_interval_output,
    )
    peak_epochs = [
        row["peak_improvement_epoch_to"] for row in horizon_output
    ]
    centroids = [row["positive_improvement_centroid"] for row in horizon_output]
    deep_acceleration = [
        deep[index] - deep[index - 1] for index in range(1, len(deep))
    ]
    cascade_rows = [
        ("shallow_deep_concurrent_pearson", pearson(shallow, deep), len(deep)),
        ("shallow_leads_deep_pearson", pearson(shallow[:-1], deep[1:]), len(deep) - 1),
        ("deep_leads_shallow_pearson", pearson(deep[:-1], shallow[1:]), len(deep) - 1),
        ("shallow_leads_far_h13_h16_pearson", pearson(shallow[:-1], far[1:]), len(far) - 1),
        ("h1_leads_h2_pearson", pearson(improvements[1][:-1], improvements[2][1:]), 8),
        ("h1_leads_h16_pearson", pearson(improvements[1][:-1], improvements[16][1:]), 8),
        ("h2_h16_concurrent_pearson", pearson(improvements[2], improvements[16]), 9),
        ("h2_leads_h16_pearson", pearson(improvements[2][:-1], improvements[16][1:]), 8),
        ("h16_leads_h2_pearson", pearson(improvements[16][:-1], improvements[2][1:]), 8),
        ("horizon_peak_timing_spearman", spearman(range(1, 17), peak_epochs), 16),
        ("horizon_centroid_spearman", spearman(range(1, 17), centroids), 16),
        ("shallow_improvement_to_next_deep_acceleration_pearson", pearson(shallow[:-1], deep_acceleration), 8),
        ("deep_to_shallow_speed_time_pearson", pearson(range(1, 10), deep_relative_to_shallow), 9),
        ("deep_to_shallow_speed_time_spearman", spearman(range(1, 10), deep_relative_to_shallow), 9),
        ("deep_bounded_share_time_pearson", pearson(range(1, 10), deep_bounded_share), 9),
        ("h16_to_shallow_speed_time_pearson", pearson(range(1, 10), h16_relative_to_shallow), 9),
        ("shallow_late2_to_early2_retention", mean(shallow[-2:]) / mean(shallow[:2]), 4),
        ("deep_late2_to_early2_retention", mean(deep[-2:]) / mean(deep[:2]), 4),
        ("far_late2_to_early2_retention", mean(far[-2:]) / mean(far[:2]), 4),
        ("h16_late2_to_early2_retention", mean(improvements[16][-2:]) / mean(improvements[16][:2]), 4),
    ]
    cascade_output = [
        {"metric": name, "value": value, "n": count}
        for name, value, count in cascade_rows
    ]
    write_tsv(
        args.output_dir / "cascade_summary.tsv",
        ("metric", "value", "n"),
        cascade_output,
    )

    heldout_improvement = [
        feedback_nll[horizon][0] - feedback_nll[horizon][-1]
        for horizon in range(5, 17)
    ]
    heldout_improved = sum(value > 0.0 for value in heldout_improvement)
    cascade_map = {row["metric"]: row["value"] for row in cascade_output}
    lines = [
        "# H4-loss feedback H16-transfer analysis",
        "",
        "This analysis uses only the ten equal 3,357-step epoch rows. H1--H4 "
        "were supervised; H5--H16 were evaluated at unsupervised conditional "
        "depths but are not held-out token identities.",
        "",
        "## Transfer result",
        "",
        f"From epoch 1 to epoch 10, {heldout_improved}/12 H5--H16 horizons "
        f"improved. Their mean NLL change expressed as positive improvement "
        f"was {mean(heldout_improvement):.6f}.",
        "",
        "## Cascade diagnostics",
        "",
        f"Mean H1--H4 versus mean H5--H16 improvement had concurrent Pearson "
        f"r={cascade_map['shallow_deep_concurrent_pearson']:.3f}; shallow "
        f"one-epoch lead r={cascade_map['shallow_leads_deep_pearson']:.3f}; "
        f"the reverse lead was r={cascade_map['deep_leads_shallow_pearson']:.3f}.",
        "",
        f"H1 one-epoch lead correlation was "
        f"r={cascade_map['h1_leads_h2_pearson']:.3f} for H2 and "
        f"r={cascade_map['h1_leads_h16_pearson']:.3f} for H16. Horizon versus "
        f"peak-improvement timing Spearman rho was "
        f"{cascade_map['horizon_peak_timing_spearman']:.3f}.",
        "",
        "The user's relative-acceleration criterion removes the common "
        "diminishing-return envelope rather than asking raw improvement to "
        "increase. Mean H5--H16 late/early retention was "
        f"{cascade_map['deep_late2_to_early2_retention']:.3f}, versus "
        f"{cascade_map['shallow_late2_to_early2_retention']:.3f} for H1--H4. "
        "The deep/shallow relative-speed time correlation was "
        f"r={cascade_map['deep_to_shallow_speed_time_pearson']:.3f}; its "
        "bounded-share correlation was "
        f"r={cascade_map['deep_bounded_share_time_pearson']:.3f}.",
        "",
        "A positive signed NLL acceleration means a negative descent velocity "
        "is flattening; it is not evidence that deep learning is accelerating. "
        "The interval TSV therefore reports positive improvement, signed "
        "velocity, and signed acceleration separately.",
        "",
        "## Comparison boundary",
        "",
        "Ratios become sensitive when the shallow envelope approaches zero, "
        "so relative-speed, bounded-share, retention, and lag diagnostics "
        "must be considered together. The scan comparison changes the "
        "future-QKV feedback edge and also "
        "supervises all H16 horizons. It is descriptive rather than a matched "
        "single-factor control. One seed, nine epoch intervals, shared "
        "validation examples, and the common cosine LR schedule prevent a "
        "causal interpretation of lag correlations.",
        "",
    ]
    (args.output_dir / "transfer_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
