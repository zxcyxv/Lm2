"""Retrospective curvature and horizon-cascade analysis for the H16 scan run.

This producer reads only the completed epoch-checkpoint metric tape from the
registered ten-epoch fused-scan experiment.  It deliberately excludes the
parent record's interrupted root-level partial tape and the irregularly
spaced step-0/100/300/1000 reports.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import statistics
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ID = (
    "EXP-20260805-byte256-unitary-fused-scan-horizon-cascade-analysis"
)
PARENT_EXPERIMENT_ID = (
    "EXP-20260803-byte256-unitary-fused-triton-scan-"
    "h16-10epoch-13m-rtx5090"
)
SOURCE_RELATIVE_PATH = (
    Path("experiments/records")
    / PARENT_EXPERIMENT_ID
    / "epoch-checkpoint-run"
    / "metrics.tsv"
)
SOURCE_PATH = REPOSITORY_ROOT / SOURCE_RELATIVE_PATH
DEFAULT_RECORD_DIR = (
    REPOSITORY_ROOT / "experiments/records" / EXPERIMENT_ID
)
EXPECTED_SOURCE_SHA256 = (
    "e08ec803c181c13e389f974e8beb27206c2d02d2847d34bd2064cb42c5e71025"
)

EPOCH_STEP = 3357
EPOCHS = tuple(range(1, 11))
EXPECTED_STEPS = tuple(EPOCH_STEP * epoch for epoch in EPOCHS)
HORIZONS = tuple(range(1, 17))
SEED = 1337
PEAK_LR = 3e-4
WARMUP_STEPS = 100
SCHEDULE_STEPS = 33570
BLOCK_MEAN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class EpochPoint:
    epoch: int
    step: int
    block_nll: float
    horizon_nll: tuple[float, ...]


@dataclass(frozen=True)
class SeriesDefinition:
    label: str
    minimum_horizon: int
    maximum_horizon: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class EnvelopeDefinition:
    label: str
    kind: str
    source_horizons: str
    values: tuple[float, ...]


def require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite {label}: {value!r}")
    return value


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_epoch_points(path: Path) -> tuple[list[EpochPoint], str, int, float]:
    digest = source_sha256(path)
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "authoritative metric checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, observed {digest}"
        )

    required = {"step", "block_nll"}
    required.update(f"h{horizon}_nll" for horizon in HORIZONS)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"source metric columns are missing: {missing}")
        source_rows = list(reader)

    selected: dict[int, EpochPoint] = {}
    maximum_block_mean_error = 0.0
    for row_number, row in enumerate(source_rows, start=2):
        try:
            step = int(row["step"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid step at source row {row_number}") from error
        if step not in EXPECTED_STEPS:
            continue
        if step in selected:
            raise ValueError(f"duplicate selected epoch step: {step}")

        epoch = step // EPOCH_STEP
        block_nll = require_finite(
            float(row["block_nll"]), f"block NLL at step {step}"
        )
        horizon_nll = tuple(
            require_finite(
                float(row[f"h{horizon}_nll"]),
                f"H{horizon} NLL at step {step}",
            )
            for horizon in HORIZONS
        )
        block_mean_error = abs(block_nll - statistics.fmean(horizon_nll))
        maximum_block_mean_error = max(
            maximum_block_mean_error, block_mean_error
        )
        if block_mean_error > BLOCK_MEAN_TOLERANCE:
            raise ValueError(
                f"block NLL is not the H1--H16 mean at step {step}: "
                f"error={block_mean_error}"
            )
        selected[step] = EpochPoint(
            epoch=epoch,
            step=step,
            block_nll=block_nll,
            horizon_nll=horizon_nll,
        )

    missing_steps = sorted(set(EXPECTED_STEPS) - set(selected))
    if missing_steps:
        raise ValueError(f"authoritative epoch steps are missing: {missing_steps}")
    points = [selected[step] for step in EXPECTED_STEPS]
    if [point.epoch for point in points] != list(EPOCHS):
        raise AssertionError("selected epoch ordering is inconsistent")
    return points, digest, len(source_rows), maximum_block_mean_error


def lr_at(update_index: int) -> float:
    if update_index < WARMUP_STEPS:
        return PEAK_LR * (update_index + 1) / WARMUP_STEPS
    progress = (update_index - WARMUP_STEPS) / max(
        1, SCHEDULE_STEPS - WARMUP_STEPS
    )
    return PEAK_LR * (
        0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
    )


def interval_average_lr(start_step: int, end_step: int) -> float:
    if end_step <= start_step:
        raise ValueError("an LR interval must contain at least one update")
    return statistics.fmean(lr_at(index) for index in range(start_step, end_step))


def differences(values: Sequence[float]) -> list[float]:
    return [values[index] - values[index - 1] for index in range(1, len(values))]


def ols_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("OLS slope requires at least two values")
    x_values = tuple(float(index) for index in range(len(values)))
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        raise ValueError("OLS x variance is zero")
    return require_finite(
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, values)
        )
        / denominator,
        "OLS slope",
    )


def theil_sen_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("Theil--Sen slope requires at least two values")
    slopes = [
        (values[right] - values[left]) / (right - left)
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return require_finite(statistics.median(slopes), "Theil--Sen slope")


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = 0.5 * (begin + end - 1) + 1.0
        for position in range(begin, end):
            ranks[order[position]] = rank
        begin = end
    return ranks


def pearson_correlation(
    first: Sequence[float], second: Sequence[float]
) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("correlation series must have equal length >= 2")
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    first_energy = sum((value - first_mean) ** 2 for value in first)
    second_energy = sum((value - second_mean) ** 2 for value in second)
    denominator = math.sqrt(first_energy * second_energy)
    if denominator == 0.0:
        raise ValueError("correlation variance is zero")
    return require_finite(
        sum(
            (first_value - first_mean) * (second_value - second_mean)
            for first_value, second_value in zip(first, second)
        )
        / denominator,
        "Pearson correlation",
    )


def spearman_correlation(
    first: Sequence[float], second: Sequence[float]
) -> float:
    return pearson_correlation(average_ranks(first), average_ranks(second))


def linear_residuals(values: Sequence[float]) -> list[float]:
    slope = ols_slope(values)
    x_mean = 0.5 * (len(values) - 1)
    intercept = statistics.fmean(values) - slope * x_mean
    return [
        value - (intercept + slope * index)
        for index, value in enumerate(values)
    ]


def lagged_series(
    source: Sequence[float], target: Sequence[float], lag: int
) -> tuple[list[float], list[float]]:
    if len(source) != len(target):
        raise ValueError("lagged source and target lengths differ")
    if lag > 0:
        return list(source[:-lag]), list(target[lag:])
    if lag < 0:
        offset = -lag
        return list(source[offset:]), list(target[:-offset])
    return list(source), list(target)


def format_tsv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        require_finite(value, "output metric")
        return format(value, ".17g")
    return value


def write_tsv(
    path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]
) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            if set(row) != set(fieldnames):
                raise ValueError(f"row schema differs for {path}: {row.keys()}")
            writer.writerow(
                {key: format_tsv_value(row[key]) for key in fieldnames}
            )


def horizon_values(
    points: Sequence[EpochPoint], horizon: int
) -> list[float]:
    return [point.horizon_nll[horizon - 1] for point in points]


def interval_lrs(points: Sequence[EpochPoint]) -> list[float]:
    return [
        interval_average_lr(points[index - 1].step, points[index].step)
        for index in range(1, len(points))
    ]


def summarize_series(
    values: Sequence[float], average_lrs: Sequence[float]
) -> dict[str, float | int]:
    velocities = differences(values)
    improvements = [-velocity for velocity in velocities]
    second_differences = differences(velocities)
    normalized_improvements = [
        improvement * PEAK_LR / average_lr
        for improvement, average_lr in zip(improvements, average_lrs)
    ]
    positive_improvements = [max(0.0, value) for value in improvements]
    positive_total = sum(positive_improvements)
    if positive_total <= 0.0:
        raise ValueError("positive-improvement timing is undefined")
    interval_midpoints = [index + 1.5 for index in range(len(improvements))]
    centroid = sum(
        midpoint * weight
        for midpoint, weight in zip(interval_midpoints, positive_improvements)
    ) / positive_total
    peak_index = max(range(len(improvements)), key=improvements.__getitem__)
    early_mean = statistics.fmean(improvements[:2])
    late_mean = statistics.fmean(improvements[-2:])
    if early_mean == 0.0:
        raise ValueError("early improvement mean is zero")

    summary: dict[str, float | int] = {
        "epoch1_nll": values[0],
        "epoch10_nll": values[-1],
        "net_nll_change": values[-1] - values[0],
        "total_nll_reduction": values[0] - values[-1],
        "interval_count": len(velocities),
        "decrease_interval_count": sum(value < 0.0 for value in velocities),
        "regression_interval_count": sum(value > 0.0 for value in velocities),
        "first_velocity": velocities[0],
        "last_velocity": velocities[-1],
        "second_difference_count": len(second_differences),
        "positive_second_difference_count": sum(
            value > 0.0 for value in second_differences
        ),
        "positive_second_difference_fraction": (
            sum(value > 0.0 for value in second_differences)
            / len(second_differences)
        ),
        "mean_second_difference": statistics.fmean(second_differences),
        "median_second_difference": statistics.median(second_differences),
        "velocity_ols_slope": ols_slope(velocities),
        "velocity_theil_sen_slope": theil_sen_slope(velocities),
        "early_mean_improvement": early_mean,
        "late_mean_improvement": late_mean,
        "late_to_early_improvement_ratio": late_mean / early_mean,
        "positive_improvement_total": positive_total,
        "peak_improvement": improvements[peak_index],
        "peak_interval_start_epoch": peak_index + 1,
        "peak_interval_end_epoch": peak_index + 2,
        "positive_improvement_centroid_epoch": centroid,
        "positive_improvement_share_e5_to_e10": (
            sum(positive_improvements[4:]) / positive_total
        ),
        "lr_normalized_early_mean_improvement": statistics.fmean(
            normalized_improvements[:2]
        ),
        "lr_normalized_late_mean_improvement": statistics.fmean(
            normalized_improvements[-2:]
        ),
        "lr_normalized_improvement_time_pearson": pearson_correlation(
            list(range(len(normalized_improvements))), normalized_improvements
        ),
    }
    for name, value in summary.items():
        if isinstance(value, float):
            require_finite(value, f"summary {name}")
    return summary


def make_epoch_rows(points: Sequence[EpochPoint]) -> list[dict[str, object]]:
    return [
        {
            "epoch": point.epoch,
            "step": point.step,
            "horizon": horizon,
            "nll": point.horizon_nll[horizon - 1],
            "block_nll": point.block_nll,
        }
        for point in points
        for horizon in HORIZONS
    ]


def make_interval_rows(
    points: Sequence[EpochPoint], average_lrs: Sequence[float]
) -> tuple[list[dict[str, object]], dict[int, list[float]]]:
    rows: list[dict[str, object]] = []
    improvements_by_horizon: dict[int, list[float]] = {}
    for horizon in HORIZONS:
        values = horizon_values(points, horizon)
        improvements: list[float] = []
        for index in range(1, len(points)):
            velocity = values[index] - values[index - 1]
            improvement = -velocity
            improvements.append(improvement)
            average_lr = average_lrs[index - 1]
            rows.append(
                {
                    "horizon": horizon,
                    "interval_index": index,
                    "start_epoch": points[index - 1].epoch,
                    "end_epoch": points[index].epoch,
                    "start_step": points[index - 1].step,
                    "end_step": points[index].step,
                    "start_nll": values[index - 1],
                    "end_nll": values[index],
                    "nll_velocity_per_epoch": velocity,
                    "nll_improvement_per_epoch": improvement,
                    "average_learning_rate": average_lr,
                    "average_learning_rate_fraction_of_peak": (
                        average_lr / PEAK_LR
                    ),
                    "peak_lr_equivalent_improvement": (
                        improvement * PEAK_LR / average_lr
                    ),
                }
            )
        improvements_by_horizon[horizon] = improvements
    return rows, improvements_by_horizon


def make_curvature_rows(
    points: Sequence[EpochPoint],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        values = horizon_values(points, horizon)
        velocities = differences(values)
        for center_index in range(1, len(points) - 1):
            previous_velocity = velocities[center_index - 1]
            next_velocity = velocities[center_index]
            acceleration = next_velocity - previous_velocity
            rows.append(
                {
                    "horizon": horizon,
                    "center_epoch": points[center_index].epoch,
                    "previous_step": points[center_index - 1].step,
                    "center_step": points[center_index].step,
                    "next_step": points[center_index + 1].step,
                    "previous_nll": values[center_index - 1],
                    "center_nll": values[center_index],
                    "next_nll": values[center_index + 1],
                    "previous_nll_velocity": previous_velocity,
                    "next_nll_velocity": next_velocity,
                    "nll_second_difference": acceleration,
                    "improvement_rate_change": -acceleration,
                    "positive_nll_acceleration": acceleration > 0.0,
                    "improvement_rate_accelerated": acceleration < 0.0,
                }
            )
    return rows


def make_horizon_summary_rows(
    points: Sequence[EpochPoint], average_lrs: Sequence[float]
) -> tuple[list[dict[str, object]], dict[int, dict[str, float | int]]]:
    rows: list[dict[str, object]] = []
    summaries: dict[int, dict[str, float | int]] = {}
    for horizon in HORIZONS:
        summary = summarize_series(horizon_values(points, horizon), average_lrs)
        summaries[horizon] = summary
        rows.append({"horizon": horizon, **summary})
    return rows, summaries


def mean_series(
    improvements_by_horizon: dict[int, list[float]],
    minimum_horizon: int,
    maximum_horizon: int,
) -> tuple[float, ...]:
    return tuple(
        statistics.fmean(
            improvements_by_horizon[horizon][interval]
            for horizon in range(minimum_horizon, maximum_horizon + 1)
        )
        for interval in range(len(next(iter(improvements_by_horizon.values()))))
    )


def cascade_pairs(
    improvements_by_horizon: dict[int, list[float]],
) -> list[tuple[SeriesDefinition, SeriesDefinition]]:
    single = {
        horizon: SeriesDefinition(
            label=f"H{horizon}",
            minimum_horizon=horizon,
            maximum_horizon=horizon,
            values=tuple(improvements_by_horizon[horizon]),
        )
        for horizon in HORIZONS
    }
    pairs = [(single[1], single[horizon]) for horizon in range(2, 17)]
    pairs.append((single[2], single[16]))
    for minimum_horizon in (2, 5, 9, 13):
        pairs.append(
            (
                single[1],
                SeriesDefinition(
                    label=f"mean_H{minimum_horizon}_H16",
                    minimum_horizon=minimum_horizon,
                    maximum_horizon=16,
                    values=mean_series(
                        improvements_by_horizon, minimum_horizon, 16
                    ),
                ),
            )
        )
    return pairs


def transformed_series(
    values: Sequence[float],
    transform: str,
    average_lrs: Sequence[float],
) -> list[float]:
    if transform == "raw_improvement":
        return list(values)
    if transform == "linear_detrended_improvement":
        return linear_residuals(values)
    if transform == "peak_lr_equivalent_improvement":
        return [
            value * PEAK_LR / average_lr
            for value, average_lr in zip(values, average_lrs)
        ]
    raise ValueError(f"unknown series transform: {transform}")


def make_cascade_correlation_rows(
    pairs: Sequence[tuple[SeriesDefinition, SeriesDefinition]],
    average_lrs: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index, (source, target) in enumerate(pairs, start=1):
        for transform in (
            "raw_improvement",
            "linear_detrended_improvement",
            "peak_lr_equivalent_improvement",
        ):
            source_values = transformed_series(
                source.values, transform, average_lrs
            )
            target_values = transformed_series(
                target.values, transform, average_lrs
            )
            for drop_first_interval in (False, True):
                trimmed_source = source_values[1:] if drop_first_interval else source_values
                trimmed_target = target_values[1:] if drop_first_interval else target_values
                for lag in (-1, 0, 1):
                    aligned_source, aligned_target = lagged_series(
                        trimmed_source, trimmed_target, lag
                    )
                    if lag > 0:
                        direction = "source_leads_target"
                    elif lag < 0:
                        direction = "target_leads_source"
                    else:
                        direction = "contemporaneous"
                    rows.append(
                        {
                            "pair_index": pair_index,
                            "source_series": source.label,
                            "source_min_horizon": source.minimum_horizon,
                            "source_max_horizon": source.maximum_horizon,
                            "target_series": target.label,
                            "target_min_horizon": target.minimum_horizon,
                            "target_max_horizon": target.maximum_horizon,
                            "transform": transform,
                            "drop_first_interval": drop_first_interval,
                            "lag_intervals": lag,
                            "lag_direction": direction,
                            "sample_count": len(aligned_source),
                            "pearson_r": pearson_correlation(
                                aligned_source, aligned_target
                            ),
                            "spearman_rho": spearman_correlation(
                                aligned_source, aligned_target
                            ),
                        }
                    )
    return rows


def make_cascade_acceleration_rows(
    pairs: Sequence[tuple[SeriesDefinition, SeriesDefinition]],
    average_lrs: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index, (source, target) in enumerate(pairs, start=1):
        for transform in (
            "raw_improvement",
            "peak_lr_equivalent_improvement",
        ):
            source_values = transformed_series(
                source.values, transform, average_lrs
            )
            target_values = transformed_series(
                target.values, transform, average_lrs
            )
            target_improvement_change = differences(target_values)
            preceding_source = source_values[:-1]
            for drop_first_transition in (False, True):
                if drop_first_transition:
                    aligned_source = preceding_source[1:]
                    aligned_target_change = target_improvement_change[1:]
                else:
                    aligned_source = preceding_source
                    aligned_target_change = target_improvement_change
                rows.append(
                    {
                        "pair_index": pair_index,
                        "source_series": source.label,
                        "source_min_horizon": source.minimum_horizon,
                        "source_max_horizon": source.maximum_horizon,
                        "target_series": target.label,
                        "target_min_horizon": target.minimum_horizon,
                        "target_max_horizon": target.maximum_horizon,
                        "transform": transform,
                        "drop_first_transition": drop_first_transition,
                        "sample_count": len(aligned_source),
                        "pearson_r": pearson_correlation(
                            aligned_source, aligned_target_change
                        ),
                        "spearman_rho": spearman_correlation(
                            aligned_source, aligned_target_change
                        ),
                    }
                )
    return rows


def fit_against_predictor(
    values: Sequence[float], predictor: Sequence[float]
) -> tuple[list[float], float, float, float]:
    """Fit ``values = intercept + slope * predictor`` and return residuals."""
    if len(values) != len(predictor) or len(values) < 3:
        raise ValueError("an envelope regression needs equal length >= 3")
    predictor_mean = statistics.fmean(predictor)
    value_mean = statistics.fmean(values)
    predictor_energy = sum(
        (value - predictor_mean) ** 2 for value in predictor
    )
    if predictor_energy == 0.0:
        raise ValueError("envelope predictor variance is zero")
    slope = sum(
        (x_value - predictor_mean) * (y_value - value_mean)
        for x_value, y_value in zip(predictor, values)
    ) / predictor_energy
    intercept = value_mean - slope * predictor_mean
    residuals = [
        y_value - (intercept + slope * x_value)
        for x_value, y_value in zip(predictor, values)
    ]
    total_energy = sum((value - value_mean) ** 2 for value in values)
    residual_energy = sum(value**2 for value in residuals)
    if total_energy == 0.0:
        raise ValueError("envelope response variance is zero")
    r_squared = 1.0 - residual_energy / total_energy
    for label, value in (
        ("envelope intercept", intercept),
        ("envelope slope", slope),
        ("envelope R squared", r_squared),
    ):
        require_finite(value, label)
    return residuals, intercept, slope, r_squared


def make_common_envelopes(
    points: Sequence[EpochPoint],
    improvements_by_horizon: dict[int, list[float]],
    average_lrs: Sequence[float],
) -> list[EnvelopeDefinition]:
    interval_count = len(points) - 1
    block_improvement = tuple(
        points[index - 1].block_nll - points[index].block_nll
        for index in range(1, len(points))
    )
    shallow_mean = tuple(
        statistics.fmean(
            improvements_by_horizon[horizon][interval]
            for horizon in range(1, 5)
        )
        for interval in range(interval_count)
    )
    shallow_median = tuple(
        statistics.median(
            improvements_by_horizon[horizon][interval]
            for horizon in range(1, 5)
        )
        for interval in range(interval_count)
    )
    all_horizon_median = tuple(
        statistics.median(
            improvements_by_horizon[horizon][interval]
            for horizon in HORIZONS
        )
        for interval in range(interval_count)
    )
    lr_fraction = tuple(value / PEAK_LR for value in average_lrs)

    interval_indices = list(range(interval_count))
    block_log = [math.log(value) for value in block_improvement]
    _, log_intercept, log_slope, _ = fit_against_predictor(
        block_log, interval_indices
    )
    exponential_fit = tuple(
        math.exp(log_intercept + log_slope * index)
        for index in interval_indices
    )

    envelopes = [
        EnvelopeDefinition(
            "block_signed_mean",
            "observed_signed_improvement",
            "H1-H16 mean; target included",
            block_improvement,
        ),
        EnvelopeDefinition(
            "shallow_h1_h4_mean",
            "observed_signed_improvement",
            "H1-H4 mean",
            shallow_mean,
        ),
        EnvelopeDefinition(
            "shallow_h1_h4_median",
            "observed_signed_improvement",
            "H1-H4 median",
            shallow_median,
        ),
        EnvelopeDefinition(
            "all_horizon_median",
            "observed_signed_improvement",
            "H1-H16 median",
            all_horizon_median,
        ),
        EnvelopeDefinition(
            "cosine_lr_fraction_of_peak",
            "registered_schedule_exposure",
            "no horizon NLL; average LR / peak LR",
            lr_fraction,
        ),
        EnvelopeDefinition(
            "block_exponential_fit",
            "smooth_log_linear_fit",
            "log-linear fit to H1-H16 block improvement",
            exponential_fit,
        ),
    ]
    for envelope in envelopes:
        if len(envelope.values) != interval_count:
            raise AssertionError(f"wrong envelope length: {envelope.label}")
        for interval, value in enumerate(envelope.values, start=1):
            require_finite(value, f"{envelope.label} interval {interval}")
            if value <= 0.0:
                raise ValueError(
                    f"common envelope must remain positive: "
                    f"{envelope.label} interval {interval}={value}"
                )
    return envelopes


def make_envelope_rows(
    envelopes: Sequence[EnvelopeDefinition],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    interval_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for envelope in envelopes:
        for interval, value in enumerate(envelope.values, start=1):
            interval_rows.append(
                {
                    "envelope": envelope.label,
                    "envelope_kind": envelope.kind,
                    "source_horizons": envelope.source_horizons,
                    "interval_index": interval,
                    "start_epoch": interval,
                    "end_epoch": interval + 1,
                    "envelope_value": value,
                }
            )
        log_values = [math.log(value) for value in envelope.values]
        _, _, log_slope, log_fit_r_squared = fit_against_predictor(
            log_values, list(range(len(log_values)))
        )
        minimum = min(envelope.values)
        maximum = max(envelope.values)
        median = statistics.median(envelope.values)
        summary_rows.append(
            {
                "envelope": envelope.label,
                "envelope_kind": envelope.kind,
                "source_horizons": envelope.source_horizons,
                "interval_count": len(envelope.values),
                "minimum_value": minimum,
                "maximum_value": maximum,
                "median_value": median,
                "minimum_to_median_ratio": minimum / median,
                "maximum_to_minimum_ratio": maximum / minimum,
                "denominator_amplification_warning_ge_20x": (
                    maximum / minimum >= 20.0
                ),
                "raw_ols_slope": ols_slope(envelope.values),
                "raw_theil_sen_slope": theil_sen_slope(envelope.values),
                "log_ols_slope": log_slope,
                "log_fit_r_squared": log_fit_r_squared,
            }
        )
    return interval_rows, summary_rows


def make_relative_improvement_rows(
    improvements_by_horizon: dict[int, list[float]],
    envelopes: Sequence[EnvelopeDefinition],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    interval_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    consensus_rows: list[dict[str, object]] = []
    summary_by_horizon: dict[int, list[dict[str, object]]] = {
        horizon: [] for horizon in HORIZONS
    }

    for horizon in HORIZONS:
        improvement = improvements_by_horizon[horizon]
        for envelope in envelopes:
            signed_relative = [
                value / denominator
                for value, denominator in zip(improvement, envelope.values)
            ]
            positive_relative = [
                max(0.0, value) / denominator
                for value, denominator in zip(improvement, envelope.values)
            ]
            for interval, (
                signed_value,
                positive_value,
                raw_value,
                denominator,
            ) in enumerate(
                zip(
                    signed_relative,
                    positive_relative,
                    improvement,
                    envelope.values,
                ),
                start=1,
            ):
                interval_rows.append(
                    {
                        "horizon": horizon,
                        "envelope": envelope.label,
                        "interval_index": interval,
                        "start_epoch": interval,
                        "end_epoch": interval + 1,
                        "signed_improvement": raw_value,
                        "positive_improvement": max(0.0, raw_value),
                        "envelope_value": denominator,
                        "signed_relative_improvement": signed_value,
                        "positive_relative_improvement": positive_value,
                    }
                )
            for index in range(1, len(signed_relative)):
                change_rows.append(
                    {
                        "horizon": horizon,
                        "envelope": envelope.label,
                        "previous_interval_index": index,
                        "current_interval_index": index + 1,
                        "center_epoch": index + 1,
                        "previous_signed_relative_improvement": signed_relative[index - 1],
                        "current_signed_relative_improvement": signed_relative[index],
                        "signed_relative_improvement_change": (
                            signed_relative[index] - signed_relative[index - 1]
                        ),
                        "relative_improvement_accelerated": (
                            signed_relative[index] > signed_relative[index - 1]
                        ),
                    }
                )

            signed_early = statistics.fmean(signed_relative[:2])
            signed_late = statistics.fmean(signed_relative[-2:])
            positive_early = statistics.fmean(positive_relative[:2])
            positive_late = statistics.fmean(positive_relative[-2:])
            if signed_early == 0.0 or positive_early == 0.0:
                raise ValueError(
                    f"relative early mean is zero for H{horizon}/{envelope.label}"
                )
            summary = {
                "horizon": horizon,
                "envelope": envelope.label,
                "interval_count": len(signed_relative),
                "signed_early_mean": signed_early,
                "signed_late_mean": signed_late,
                "signed_late_minus_early": signed_late - signed_early,
                "signed_late_to_early_ratio": signed_late / signed_early,
                "signed_ols_slope": ols_slope(signed_relative),
                "signed_theil_sen_slope": theil_sen_slope(signed_relative),
                "signed_time_pearson": pearson_correlation(
                    list(range(len(signed_relative))), signed_relative
                ),
                "positive_signed_change_count": sum(
                    value > 0.0 for value in differences(signed_relative)
                ),
                "positive_early_mean": positive_early,
                "positive_late_mean": positive_late,
                "positive_late_minus_early": positive_late - positive_early,
                "positive_late_to_early_ratio": positive_late / positive_early,
                "positive_ols_slope": ols_slope(positive_relative),
                "positive_theil_sen_slope": theil_sen_slope(
                    positive_relative
                ),
            }
            summary_rows.append(summary)
            summary_by_horizon[horizon].append(summary)

    for horizon in HORIZONS:
        summaries = summary_by_horizon[horizon]
        consensus_rows.append(
            {
                "horizon": horizon,
                "envelope_count": len(summaries),
                "positive_signed_late_minus_early_envelope_count": sum(
                    float(row["signed_late_minus_early"]) > 0.0
                    for row in summaries
                ),
                "positive_signed_ols_slope_envelope_count": sum(
                    float(row["signed_ols_slope"]) > 0.0 for row in summaries
                ),
                "positive_signed_theil_sen_slope_envelope_count": sum(
                    float(row["signed_theil_sen_slope"]) > 0.0
                    for row in summaries
                ),
                "positive_only_late_minus_early_envelope_count": sum(
                    float(row["positive_late_minus_early"]) > 0.0
                    for row in summaries
                ),
                "positive_only_ols_slope_envelope_count": sum(
                    float(row["positive_ols_slope"]) > 0.0 for row in summaries
                ),
                "positive_only_theil_sen_slope_envelope_count": sum(
                    float(row["positive_theil_sen_slope"]) > 0.0
                    for row in summaries
                ),
            }
        )
    return interval_rows, change_rows, summary_rows, consensus_rows


def make_horizon_share_rows(
    improvements_by_horizon: dict[int, list[float]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    interval_count = len(next(iter(improvements_by_horizon.values())))
    positive_totals = [
        sum(
            max(0.0, improvements_by_horizon[horizon][interval])
            for horizon in HORIZONS
        )
        for interval in range(interval_count)
    ]
    if any(value <= 0.0 for value in positive_totals):
        raise ValueError("positive horizon-share denominator is zero")

    share_by_horizon: dict[int, list[float]] = {}
    interval_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        positive = [
            max(0.0, value) for value in improvements_by_horizon[horizon]
        ]
        shares = [
            value / total for value, total in zip(positive, positive_totals)
        ]
        share_by_horizon[horizon] = shares
        for interval, (raw, clipped, total, share) in enumerate(
            zip(
                improvements_by_horizon[horizon],
                positive,
                positive_totals,
                shares,
            ),
            start=1,
        ):
            interval_rows.append(
                {
                    "horizon": horizon,
                    "interval_index": interval,
                    "start_epoch": interval,
                    "end_epoch": interval + 1,
                    "signed_improvement": raw,
                    "positive_improvement": clipped,
                    "all_horizon_positive_improvement_total": total,
                    "positive_horizon_share": share,
                }
            )
        early = statistics.fmean(shares[:2])
        late = statistics.fmean(shares[-2:])
        summary_rows.append(
            {
                "horizon": horizon,
                "early_mean_positive_share": early,
                "late_mean_positive_share": late,
                "late_minus_early_positive_share": late - early,
                "late_to_early_positive_share_ratio": (
                    late / early if early > 0.0 else 0.0
                ),
                "share_ols_slope": ols_slope(shares),
                "share_theil_sen_slope": theil_sen_slope(shares),
                "share_time_pearson": pearson_correlation(
                    list(range(len(shares))), shares
                ),
                "cumulative_positive_improvement_share": (
                    sum(
                        max(0.0, value)
                        for value in improvements_by_horizon[horizon]
                    )
                    / sum(positive_totals)
                ),
            }
        )

    group_rows: list[dict[str, object]] = []
    for label, minimum_horizon, maximum_horizon in (
        ("shallow_H1_H4", 1, 4),
        ("deep_H5_H16", 5, 16),
        ("deep_H9_H16", 9, 16),
        ("deep_H13_H16", 13, 16),
    ):
        shares = [
            sum(
                share_by_horizon[horizon][interval]
                for horizon in range(minimum_horizon, maximum_horizon + 1)
            )
            for interval in range(interval_count)
        ]
        early = statistics.fmean(shares[:2])
        late = statistics.fmean(shares[-2:])
        group_rows.append(
            {
                "group": label,
                "minimum_horizon": minimum_horizon,
                "maximum_horizon": maximum_horizon,
                "early_mean_positive_share": early,
                "late_mean_positive_share": late,
                "late_minus_early_positive_share": late - early,
                "late_to_early_positive_share_ratio": late / early,
                "share_ols_slope": ols_slope(shares),
                "share_theil_sen_slope": theil_sen_slope(shares),
                "share_time_pearson": pearson_correlation(
                    list(range(len(shares))), shares
                ),
            }
        )
    return interval_rows, summary_rows, group_rows


def make_residual_lead_lag_rows(
    pairs: Sequence[tuple[SeriesDefinition, SeriesDefinition]],
    envelopes: Sequence[EnvelopeDefinition],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index, (source, target) in enumerate(pairs, start=1):
        for envelope in envelopes:
            (
                source_residual,
                source_intercept,
                source_slope,
                source_fit_r_squared,
            ) = fit_against_predictor(source.values, envelope.values)
            (
                target_residual,
                target_intercept,
                target_slope,
                target_fit_r_squared,
            ) = fit_against_predictor(target.values, envelope.values)
            source_lead = source_residual[:-1]
            target_next = target_residual[1:]
            next_envelope = list(envelope.values[1:])
            for drop_first_transition in (False, True):
                if drop_first_transition:
                    aligned_source = source_lead[1:]
                    aligned_target = target_next[1:]
                    aligned_envelope = next_envelope[1:]
                else:
                    aligned_source = source_lead
                    aligned_target = target_next
                    aligned_envelope = next_envelope
                residual_lead_pearson = pearson_correlation(
                    aligned_source, aligned_target
                )
                source_partial, _, _, _ = fit_against_predictor(
                    aligned_source, aligned_envelope
                )
                target_partial, _, _, _ = fit_against_predictor(
                    aligned_target, aligned_envelope
                )
                partial_pearson = pearson_correlation(
                    source_partial, target_partial
                )
                source_partial_energy = sum(
                    value**2 for value in source_partial
                )
                if source_partial_energy == 0.0:
                    raise ValueError("partial source residual energy is zero")
                partial_slope = sum(
                    source_value * target_value
                    for source_value, target_value in zip(
                        source_partial, target_partial
                    )
                ) / source_partial_energy
                rows.append(
                    {
                        "pair_index": pair_index,
                        "source_series": source.label,
                        "target_series": target.label,
                        "envelope": envelope.label,
                        "drop_first_transition": drop_first_transition,
                        "sample_count": len(aligned_source),
                        "source_envelope_intercept": source_intercept,
                        "source_envelope_slope": source_slope,
                        "source_envelope_fit_r_squared": source_fit_r_squared,
                        "target_envelope_intercept": target_intercept,
                        "target_envelope_slope": target_slope,
                        "target_envelope_fit_r_squared": target_fit_r_squared,
                        "residual_source_lead_pearson": residual_lead_pearson,
                        "residual_source_lead_spearman": spearman_correlation(
                            aligned_source, aligned_target
                        ),
                        "next_envelope_partial_pearson": partial_pearson,
                        "next_envelope_partial_regression_slope": partial_slope,
                        "next_envelope_partial_delta_r_squared": (
                            partial_pearson**2
                        ),
                    }
                )
    return rows


def find_correlation(
    rows: Sequence[dict[str, object]],
    *,
    source: str,
    target: str,
    transform: str = "raw_improvement",
    drop_first: bool = False,
    lag: int = 0,
    statistic: str = "pearson_r",
) -> float:
    matches = [
        row
        for row in rows
        if row["source_series"] == source
        and row["target_series"] == target
        and row["transform"] == transform
        and row["drop_first_interval"] is drop_first
        and row["lag_intervals"] == lag
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one cascade correlation, found {len(matches)}")
    return float(matches[0][statistic])


def find_acceleration_correlation(
    rows: Sequence[dict[str, object]],
    *,
    source: str,
    target: str,
    transform: str = "raw_improvement",
    drop_first: bool = False,
    statistic: str = "pearson_r",
) -> float:
    matches = [
        row
        for row in rows
        if row["source_series"] == source
        and row["target_series"] == target
        and row["transform"] == transform
        and row["drop_first_transition"] is drop_first
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one cascade-acceleration correlation, found {len(matches)}"
        )
    return float(matches[0][statistic])


def find_relative_summary(
    rows: Sequence[dict[str, object]],
    *,
    horizon: int,
    envelope: str,
    statistic: str,
) -> float:
    matches = [
        row
        for row in rows
        if row["horizon"] == horizon and row["envelope"] == envelope
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one relative summary, found {len(matches)}")
    return float(matches[0][statistic])


def find_group_share(
    rows: Sequence[dict[str, object]], *, group: str, statistic: str
) -> float:
    matches = [row for row in rows if row["group"] == group]
    if len(matches) != 1:
        raise ValueError(f"expected one horizon-share group, found {len(matches)}")
    return float(matches[0][statistic])


def find_residual_lead(
    rows: Sequence[dict[str, object]],
    *,
    source: str,
    target: str,
    envelope: str,
    drop_first: bool = False,
    statistic: str = "next_envelope_partial_pearson",
) -> float:
    matches = [
        row
        for row in rows
        if row["source_series"] == source
        and row["target_series"] == target
        and row["envelope"] == envelope
        and row["drop_first_transition"] is drop_first
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one residual lead row, found {len(matches)}")
    return float(matches[0][statistic])


def aggregate_metric_rows(
    points: Sequence[EpochPoint],
    horizon_summaries: dict[int, dict[str, float | int]],
    block_summary: dict[str, float | int],
    cascade_rows: Sequence[dict[str, object]],
    acceleration_rows: Sequence[dict[str, object]],
    envelope_summary_rows: Sequence[dict[str, object]],
    relative_summary_rows: Sequence[dict[str, object]],
    relative_consensus_rows: Sequence[dict[str, object]],
    group_share_rows: Sequence[dict[str, object]],
    residual_lead_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    second_difference_total = sum(
        int(summary["second_difference_count"])
        for summary in horizon_summaries.values()
    )
    positive_second_difference_total = sum(
        int(summary["positive_second_difference_count"])
        for summary in horizon_summaries.values()
    )
    horizons = list(HORIZONS)
    centroids = [
        float(horizon_summaries[horizon]["positive_improvement_centroid_epoch"])
        for horizon in HORIZONS
    ]
    peak_midpoints = [
        0.5
        * (
            float(horizon_summaries[horizon]["peak_interval_start_epoch"])
            + float(horizon_summaries[horizon]["peak_interval_end_epoch"])
        )
        for horizon in HORIZONS
    ]
    retention = [
        float(horizon_summaries[horizon]["late_to_early_improvement_ratio"])
        for horizon in HORIZONS
    ]
    consensus_by_horizon = {
        int(row["horizon"]): row for row in relative_consensus_rows
    }
    h2_h16_residual = [
        float(row["next_envelope_partial_pearson"])
        for row in residual_lead_rows
        if row["source_series"] == "H2"
        and row["target_series"] == "H16"
        and row["drop_first_transition"] is False
    ]
    h2_h16_residual_drop = [
        float(row["next_envelope_partial_pearson"])
        for row in residual_lead_rows
        if row["source_series"] == "H2"
        and row["target_series"] == "H16"
        and row["drop_first_transition"] is True
    ]
    if len(h2_h16_residual) != 6 or len(h2_h16_residual_drop) != 6:
        raise ValueError("H2--H16 residual envelope coverage is incomplete")
    envelope_condition = {
        str(row["envelope"]): float(row["maximum_to_minimum_ratio"])
        for row in envelope_summary_rows
    }

    metrics: list[tuple[str, str, float, str, str]] = [
        ("completeness", "selected_epoch_count", len(points), "count", "E1--E10"),
        ("completeness", "horizon_count", len(HORIZONS), "count", "H1--H16"),
        ("curvature", "local_second_difference_count", second_difference_total, "count", "all horizons"),
        ("curvature", "positive_local_second_difference_count", positive_second_difference_total, "count", "all horizons"),
        ("curvature", "positive_local_second_difference_fraction", positive_second_difference_total / second_difference_total, "fraction", "all horizons"),
        ("curvature", "positive_mean_second_difference_horizon_count", sum(float(summary["mean_second_difference"]) > 0.0 for summary in horizon_summaries.values()), "count", "H1--H16"),
        ("curvature", "positive_median_second_difference_horizon_count", sum(float(summary["median_second_difference"]) > 0.0 for summary in horizon_summaries.values()), "count", "H1--H16"),
        ("curvature", "positive_velocity_ols_slope_horizon_count", sum(float(summary["velocity_ols_slope"]) > 0.0 for summary in horizon_summaries.values()), "count", "H1--H16"),
        ("curvature", "positive_velocity_theil_sen_slope_horizon_count", sum(float(summary["velocity_theil_sen_slope"]) > 0.0 for summary in horizon_summaries.values()), "count", "H1--H16"),
        ("block", "epoch1_nll", float(block_summary["epoch1_nll"]), "nll_per_token", "H1--H16 mean"),
        ("block", "epoch10_nll", float(block_summary["epoch10_nll"]), "nll_per_token", "H1--H16 mean"),
        ("block", "net_nll_change", float(block_summary["net_nll_change"]), "nll_per_token", "E1 to E10"),
        ("block", "positive_second_difference_count", float(block_summary["positive_second_difference_count"]), "count", "8 centers"),
        ("block", "mean_second_difference", float(block_summary["mean_second_difference"]), "nll_per_token_per_epoch_squared", "8 centers"),
        ("block", "median_second_difference", float(block_summary["median_second_difference"]), "nll_per_token_per_epoch_squared", "8 centers"),
        ("block", "velocity_ols_slope", float(block_summary["velocity_ols_slope"]), "nll_per_token_per_epoch_squared", "9 intervals"),
        ("timing", "horizon_vs_positive_improvement_centroid_pearson", pearson_correlation(horizons, centroids), "correlation", "16 horizons"),
        ("timing", "horizon_vs_positive_improvement_centroid_spearman", spearman_correlation(horizons, centroids), "correlation", "16 horizons"),
        ("timing", "horizon_vs_peak_interval_midpoint_pearson", pearson_correlation(horizons, peak_midpoints), "correlation", "16 horizons"),
        ("timing", "horizon_vs_peak_interval_midpoint_spearman", spearman_correlation(horizons, peak_midpoints), "correlation", "16 horizons"),
        ("timing", "horizon_vs_late_early_retention_pearson", pearson_correlation(horizons, retention), "correlation", "16 horizons"),
        ("timing", "horizon_vs_late_early_retention_spearman", spearman_correlation(horizons, retention), "correlation", "16 horizons"),
        ("cascade", "h2_to_h16_same_interval_pearson", find_correlation(cascade_rows, source="H2", target="H16"), "correlation", "n=9"),
        ("cascade", "h2_to_h16_source_lead1_pearson", find_correlation(cascade_rows, source="H2", target="H16", lag=1), "correlation", "n=8"),
        ("cascade", "h2_to_h16_target_lead1_pearson", find_correlation(cascade_rows, source="H2", target="H16", lag=-1), "correlation", "n=8"),
        ("cascade", "h2_to_h16_same_interval_drop_first_pearson", find_correlation(cascade_rows, source="H2", target="H16", drop_first=True), "correlation", "n=8"),
        ("cascade", "h1_to_h16_source_lead1_pearson", find_correlation(cascade_rows, source="H1", target="H16", lag=1), "correlation", "n=8"),
        ("cascade", "h1_to_mean_h13_h16_source_lead1_pearson", find_correlation(cascade_rows, source="H1", target="mean_H13_H16", lag=1), "correlation", "n=8"),
        ("cascade", "h2_to_h16_source_lead1_lr_normalized_pearson", find_correlation(cascade_rows, source="H2", target="H16", transform="peak_lr_equivalent_improvement", lag=1), "correlation", "n=8"),
        ("cascade_acceleration", "h1_improvement_to_h16_improvement_change_pearson", find_acceleration_correlation(acceleration_rows, source="H1", target="H16"), "correlation", "n=8"),
        ("cascade_acceleration", "h1_improvement_to_h16_improvement_change_drop_first_pearson", find_acceleration_correlation(acceleration_rows, source="H1", target="H16", drop_first=True), "correlation", "n=7"),
        ("cascade_acceleration", "h1_improvement_to_mean_h13_h16_improvement_change_pearson", find_acceleration_correlation(acceleration_rows, source="H1", target="mean_H13_H16"), "correlation", "n=8"),
        ("cascade_acceleration", "h1_improvement_to_mean_h13_h16_improvement_change_drop_first_pearson", find_acceleration_correlation(acceleration_rows, source="H1", target="mean_H13_H16", drop_first=True), "correlation", "n=7"),
        ("relative", "horizon_vs_raw_late_early_retention_spearman", spearman_correlation(horizons, retention), "correlation", "16 horizons"),
        ("relative", "h2_raw_late_to_early_improvement_ratio", float(horizon_summaries[2]["late_to_early_improvement_ratio"]), "ratio", "raw improvement"),
        ("relative", "h16_raw_late_to_early_improvement_ratio", float(horizon_summaries[16]["late_to_early_improvement_ratio"]), "ratio", "raw improvement"),
        ("relative", "h2_lr_normalized_early_mean", float(horizon_summaries[2]["lr_normalized_early_mean_improvement"]), "peak_lr_equivalent_nll_improvement", "first two intervals"),
        ("relative", "h2_lr_normalized_late_mean", float(horizon_summaries[2]["lr_normalized_late_mean_improvement"]), "peak_lr_equivalent_nll_improvement", "last two intervals"),
        ("relative", "h16_lr_normalized_early_mean", float(horizon_summaries[16]["lr_normalized_early_mean_improvement"]), "peak_lr_equivalent_nll_improvement", "first two intervals"),
        ("relative", "h16_lr_normalized_late_mean", float(horizon_summaries[16]["lr_normalized_late_mean_improvement"]), "peak_lr_equivalent_nll_improvement", "last two intervals"),
        ("relative", "h16_positive_signed_late_minus_early_envelope_count", float(consensus_by_horizon[16]["positive_signed_late_minus_early_envelope_count"]), "count", "6 envelopes"),
        ("relative", "h16_positive_signed_ols_slope_envelope_count", float(consensus_by_horizon[16]["positive_signed_ols_slope_envelope_count"]), "count", "6 envelopes"),
        ("relative", "h16_positive_signed_theil_sen_slope_envelope_count", float(consensus_by_horizon[16]["positive_signed_theil_sen_slope_envelope_count"]), "count", "6 envelopes"),
        ("relative", "h16_block_relative_early_mean", find_relative_summary(relative_summary_rows, horizon=16, envelope="block_signed_mean", statistic="signed_early_mean"), "ratio", "first two intervals"),
        ("relative", "h16_block_relative_late_mean", find_relative_summary(relative_summary_rows, horizon=16, envelope="block_signed_mean", statistic="signed_late_mean"), "ratio", "last two intervals"),
        ("relative", "h16_shallow_mean_relative_early_mean", find_relative_summary(relative_summary_rows, horizon=16, envelope="shallow_h1_h4_mean", statistic="signed_early_mean"), "ratio", "first two intervals"),
        ("relative", "h16_shallow_mean_relative_late_mean", find_relative_summary(relative_summary_rows, horizon=16, envelope="shallow_h1_h4_mean", statistic="signed_late_mean"), "ratio", "last two intervals"),
        ("share", "shallow_h1_h4_early_positive_share", find_group_share(group_share_rows, group="shallow_H1_H4", statistic="early_mean_positive_share"), "fraction", "first two intervals"),
        ("share", "shallow_h1_h4_late_positive_share", find_group_share(group_share_rows, group="shallow_H1_H4", statistic="late_mean_positive_share"), "fraction", "last two intervals"),
        ("share", "deep_h5_h16_early_positive_share", find_group_share(group_share_rows, group="deep_H5_H16", statistic="early_mean_positive_share"), "fraction", "first two intervals"),
        ("share", "deep_h5_h16_late_positive_share", find_group_share(group_share_rows, group="deep_H5_H16", statistic="late_mean_positive_share"), "fraction", "last two intervals"),
        ("share", "deep_h9_h16_early_positive_share", find_group_share(group_share_rows, group="deep_H9_H16", statistic="early_mean_positive_share"), "fraction", "first two intervals"),
        ("share", "deep_h9_h16_late_positive_share", find_group_share(group_share_rows, group="deep_H9_H16", statistic="late_mean_positive_share"), "fraction", "last two intervals"),
        ("share", "deep_h13_h16_early_positive_share", find_group_share(group_share_rows, group="deep_H13_H16", statistic="early_mean_positive_share"), "fraction", "first two intervals"),
        ("share", "deep_h13_h16_late_positive_share", find_group_share(group_share_rows, group="deep_H13_H16", statistic="late_mean_positive_share"), "fraction", "last two intervals"),
        ("residual_lead", "h2_to_h16_positive_partial_envelope_count", sum(value > 0.0 for value in h2_h16_residual), "count", "6 envelopes; n=8"),
        ("residual_lead", "h2_to_h16_partial_pearson_minimum", min(h2_h16_residual), "correlation", "6 envelopes; n=8"),
        ("residual_lead", "h2_to_h16_partial_pearson_maximum", max(h2_h16_residual), "correlation", "6 envelopes; n=8"),
        ("residual_lead", "h2_to_h16_drop_first_positive_partial_envelope_count", sum(value > 0.0 for value in h2_h16_residual_drop), "count", "6 envelopes; n=7"),
        ("residual_lead", "h2_to_h16_drop_first_partial_pearson_minimum", min(h2_h16_residual_drop), "correlation", "6 envelopes; n=7"),
        ("residual_lead", "h2_to_h16_drop_first_partial_pearson_maximum", max(h2_h16_residual_drop), "correlation", "6 envelopes; n=7"),
        ("denominator", "block_maximum_to_minimum", envelope_condition["block_signed_mean"], "ratio", "9 intervals"),
        ("denominator", "shallow_mean_maximum_to_minimum", envelope_condition["shallow_h1_h4_mean"], "ratio", "9 intervals"),
        ("denominator", "shallow_median_maximum_to_minimum", envelope_condition["shallow_h1_h4_median"], "ratio", "9 intervals"),
        ("denominator", "all_median_maximum_to_minimum", envelope_condition["all_horizon_median"], "ratio", "9 intervals"),
    ]
    return [
        {
            "section": section,
            "metric": metric,
            "value": require_finite(float(value), metric),
            "unit": unit,
            "scope": scope,
        }
        for section, metric, value, unit, scope in metrics
    ]


def interpretation_text(
    horizon_summaries: dict[int, dict[str, float | int]],
    block_summary: dict[str, float | int],
    cascade_rows: Sequence[dict[str, object]],
    acceleration_rows: Sequence[dict[str, object]],
    envelope_summary_rows: Sequence[dict[str, object]],
    relative_summary_rows: Sequence[dict[str, object]],
    relative_consensus_rows: Sequence[dict[str, object]],
    group_share_rows: Sequence[dict[str, object]],
    residual_lead_rows: Sequence[dict[str, object]],
) -> str:
    local_positive = sum(
        int(summary["positive_second_difference_count"])
        for summary in horizon_summaries.values()
    )
    local_total = sum(
        int(summary["second_difference_count"])
        for summary in horizon_summaries.values()
    )
    positive_median = sum(
        float(summary["median_second_difference"]) > 0.0
        for summary in horizon_summaries.values()
    )
    peak_first = sum(
        int(summary["peak_interval_start_epoch"]) == 1
        for summary in horizon_summaries.values()
    )
    centroids = [
        float(horizon_summaries[horizon]["positive_improvement_centroid_epoch"])
        for horizon in HORIZONS
    ]
    peak_midpoints = [
        0.5
        * (
            float(horizon_summaries[horizon]["peak_interval_start_epoch"])
            + float(horizon_summaries[horizon]["peak_interval_end_epoch"])
        )
        for horizon in HORIZONS
    ]
    horizon_numbers = list(HORIZONS)

    h2_h16_same = find_correlation(
        cascade_rows, source="H2", target="H16"
    )
    h2_h16_lead = find_correlation(
        cascade_rows, source="H2", target="H16", lag=1
    )
    h2_h16_reverse = find_correlation(
        cascade_rows, source="H2", target="H16", lag=-1
    )
    h2_h16_drop = find_correlation(
        cascade_rows, source="H2", target="H16", drop_first=True
    )
    h1_h16_same = find_correlation(
        cascade_rows, source="H1", target="H16"
    )
    h1_h16_same_drop = find_correlation(
        cascade_rows,
        source="H1",
        target="H16",
        drop_first=True,
    )
    h1_h16_lead = find_correlation(
        cascade_rows, source="H1", target="H16", lag=1
    )
    h1_h2_lead = find_correlation(
        cascade_rows, source="H1", target="H2", lag=1
    )
    h1_h4_lead = find_correlation(
        cascade_rows, source="H1", target="H4", lag=1
    )
    h1_deep_lead = find_correlation(
        cascade_rows, source="H1", target="mean_H13_H16", lag=1
    )
    h2_h16_lr_lead = find_correlation(
        cascade_rows,
        source="H2",
        target="H16",
        transform="peak_lr_equivalent_improvement",
        lag=1,
    )
    acceleration_all = find_acceleration_correlation(
        acceleration_rows, source="H1", target="H16"
    )
    acceleration_drop = find_acceleration_correlation(
        acceleration_rows,
        source="H1",
        target="H16",
        drop_first=True,
    )
    retention = [
        float(horizon_summaries[horizon]["late_to_early_improvement_ratio"])
        for horizon in HORIZONS
    ]
    h16_consensus_matches = [
        row for row in relative_consensus_rows if row["horizon"] == 16
    ]
    if len(h16_consensus_matches) != 1:
        raise ValueError("H16 relative consensus row is missing")
    h16_consensus = h16_consensus_matches[0]
    envelope_condition = {
        str(row["envelope"]): float(row["maximum_to_minimum_ratio"])
        for row in envelope_summary_rows
    }
    h2_h16_residual = [
        float(row["next_envelope_partial_pearson"])
        for row in residual_lead_rows
        if row["source_series"] == "H2"
        and row["target_series"] == "H16"
        and row["drop_first_transition"] is False
    ]
    h2_h16_residual_drop = [
        float(row["next_envelope_partial_pearson"])
        for row in residual_lead_rows
        if row["source_series"] == "H2"
        and row["target_series"] == "H16"
        and row["drop_first_transition"] is True
    ]
    h1_h16_residual = [
        float(row["next_envelope_partial_pearson"])
        for row in residual_lead_rows
        if row["source_series"] == "H1"
        and row["target_series"] == "H16"
        and row["drop_first_transition"] is False
    ]
    if not all(
        len(values) == 6
        for values in (
            h2_h16_residual,
            h2_h16_residual_drop,
            h1_h16_residual,
        )
    ):
        raise ValueError("residual envelope sensitivity is incomplete")

    return f"""# Parallel-scan H1--H16 curvature and cascade interpretation

## NLL 곡률의 부호

이 분석에서 `v_e = L_e - L_(e-1)`이므로 NLL이 감소할 때 속도는
음수다. 감소가 점점 느려지는 convex-up 곡선은
`a_e = v_e - v_(e-1) > 0`이다. 반대로 양수로 정의한 감소량
`I_e = -v_e` 자체의 변화는 `Delta I_e = -a_e`이므로 음수다.

epoch 1--10의 전체 추세에서는 flattening이 우세하다. H1--H16 모두
평균 2차 차분, interval velocity의 OLS slope, Theil--Sen slope가
양수다. block NLL은 `{float(block_summary['epoch1_nll']):.6f}`에서
`{float(block_summary['epoch10_nll']):.6f}`로 변했고, block의 8개
국소 2차 차분 중 `{int(block_summary['positive_second_difference_count'])}`개가
양수였다. 전체 horizon-center 128개 중 양수는 `{local_positive}`개
(`{local_positive / local_total:.1%}`), horizon별 median도 양수인 것은
16개 중 `{positive_median}`개다. 따라서 장기적인 양의 NLL 가속도는
관측되지만 모든 checkpoint 사이에서 pointwise하게 양수라는 주장은
성립하지 않는다. H16은 9개 interval 중
`{int(horizon_summaries[16]['decrease_interval_count'])}`개에서만 감소해
깊은 horizon의 작은 후반 차분은 특히 진동성이 크다.

## 얕은 horizon에서 깊은 horizon으로의 cascade

순서 있는 학습 파동은 뚜렷하지 않다. 최대 개선 interval이 E1->E2인
horizon은 16개 중 `{peak_first}`개이며 H2와 H16도 모두 여기에
포함된다. horizon 번호와 positive-improvement 시간 중심의 Spearman
상관은 `{spearman_correlation(horizon_numbers, centroids):+.3f}`, peak
interval midpoint와의 상관은
`{spearman_correlation(horizon_numbers, peak_midpoints):+.3f}`다. H2와
H16의 시간 중심은 각각
`{float(horizon_summaries[2]['positive_improvement_centroid_epoch']):.3f}`와
`{float(horizon_summaries[16]['positive_improvement_centroid_epoch']):.3f}`
epoch이다.

interval 개선량의 Pearson 상관에서 H2--H16 동시 상관은
`{h2_h16_same:+.3f}` (`n=9`), H2가 한 interval 선행할 때
`{h2_h16_lead:+.3f}` (`n=8`), 반대 방향은
`{h2_h16_reverse:+.3f}` (`n=8`)다. 첫 개선 interval을 빼면 동시
상관은 `{h2_h16_drop:+.3f}`이다. 따라서 이 tape는 H2가 H16을
선행한다고 raw interval 상관만으로 지지하지 않는다. H1의 동시 H16 상관
`{h1_h16_same:+.3f}`도 첫 interval 제거 후
`{h1_h16_same_drop:+.3f}`로 약해지고, H1의 한-interval 선행 H16
상관은 `{h1_h16_lead:+.3f}`이다.

가까운 horizon에는 제한적인 동조가 있다. H1의 한-interval 선행
상관은 H2에서 `{h1_h2_lead:+.3f}`, H4에서
`{h1_h4_lead:+.3f}`이지만 mean H13--H16에서는
`{h1_deep_lead:+.3f}`이다. cosine learning-rate 면적으로 나눈
heuristic에서 H2->H16 선행 상관도 `{h2_h16_lr_lead:+.3f}` (`n=8`)에
그친다.

“얕은 개선이 다음 깊은 개선률의 가속을 만든다”는 직접 지표도
민감하다. `I_H1[t]`와 `I_H16[t+1]-I_H16[t]`의 상관은 전체 8개
transition에서 `{acceleration_all:+.3f}`이지만 첫 transition을 빼면
`{acceleration_drop:+.3f}` (`n=7`)로 부호가 바뀐다. 이는 cascade의
양성 증거가 아니라 표본 수와 초기 interval에 대한 민감도다.

## 공통 diminishing-return envelope 제거

raw improvement의 early/late ratio 자체도 깊은 horizon에서 조금 더
크다는 약한 신호가 있다. 이 ratio와 horizon 번호의 Spearman 상관은
`{spearman_correlation(horizon_numbers, retention):+.3f}`이고, H2는
`{float(horizon_summaries[2]['late_to_early_improvement_ratio']):.3f}`, H16은
`{float(horizon_summaries[16]['late_to_early_improvement_ratio']):.3f}`이다.
cosine-LR exposure로 나눈 peak-LR-equivalent 개선량은 H2가
`{float(horizon_summaries[2]['lr_normalized_early_mean_improvement']):.5f}`에서
`{float(horizon_summaries[2]['lr_normalized_late_mean_improvement']):.5f}`로
줄지만, H16은
`{float(horizon_summaries[16]['lr_normalized_early_mean_improvement']):.5f}`에서
`{float(horizon_summaries[16]['lr_normalized_late_mean_improvement']):.5f}`로
소폭 커진다.

여섯 common envelope에 대한 signed ratio `s_h/g`에서도 H16의 endpoint
late-minus-early와 OLS slope는 각각
`{int(h16_consensus['positive_signed_late_minus_early_envelope_count'])}/6`,
`{int(h16_consensus['positive_signed_ols_slope_envelope_count'])}/6`에서
양수다. block envelope 대비 H16 ratio는
`{find_relative_summary(relative_summary_rows, horizon=16, envelope='block_signed_mean', statistic='signed_early_mean'):.3f}`에서
`{find_relative_summary(relative_summary_rows, horizon=16, envelope='block_signed_mean', statistic='signed_late_mean'):.3f}`로, shallow-H1--H4 mean 대비로는
`{find_relative_summary(relative_summary_rows, horizon=16, envelope='shallow_h1_h4_mean', statistic='signed_early_mean'):.3f}`에서
`{find_relative_summary(relative_summary_rows, horizon=16, envelope='shallow_h1_h4_mean', statistic='signed_late_mean'):.3f}`로 변한다. 이는 raw 감소율이 작아져도 H16의 상대적 비중이
커진다는 방향의 evidence다.

하지만 robust slope인 Theil--Sen이 양수인 envelope는
`{int(h16_consensus['positive_signed_theil_sen_slope_envelope_count'])}/6`뿐이다.
observed block, shallow mean, shallow median denominator의 최대/최소 비는
각각 `{envelope_condition['block_signed_mean']:.1f}x`,
`{envelope_condition['shallow_h1_h4_mean']:.1f}x`,
`{envelope_condition['shallow_h1_h4_median']:.1f}x`여서 후반 ratio가 작은
분모와 checkpoint 진동을 증폭한다. all-horizon median도
`{envelope_condition['all_horizon_median']:.1f}x` 변한다. 9개 interval에
비해 자유도가 큰 power-law fit은 추가하지 않았고, smooth 대안은
log-linear block exponential 하나만 보존했다.

positive-only compositional share는 같은 방향을 더 직접적으로 보인다.
H1--H4의 early/late share는
`{find_group_share(group_share_rows, group='shallow_H1_H4', statistic='early_mean_positive_share'):.3f}`에서
`{find_group_share(group_share_rows, group='shallow_H1_H4', statistic='late_mean_positive_share'):.3f}`로 줄고, H5--H16은
`{find_group_share(group_share_rows, group='deep_H5_H16', statistic='early_mean_positive_share'):.3f}`에서
`{find_group_share(group_share_rows, group='deep_H5_H16', statistic='late_mean_positive_share'):.3f}`로 커진다. H9--H16은
`{find_group_share(group_share_rows, group='deep_H9_H16', statistic='early_mean_positive_share'):.3f}`에서
`{find_group_share(group_share_rows, group='deep_H9_H16', statistic='late_mean_positive_share'):.3f}`다. 다만 이 통계는 regression interval을 0으로 clip하는
구성비이므로 signed 개선을 대체하지 않는다. H13--H16도 endpoint
share는
`{find_group_share(group_share_rows, group='deep_H13_H16', statistic='early_mean_positive_share'):.3f}`에서
`{find_group_share(group_share_rows, group='deep_H13_H16', statistic='late_mean_positive_share'):.3f}`로 늘지만 Theil--Sen slope는
`{find_group_share(group_share_rows, group='deep_H13_H16', statistic='share_theil_sen_slope'):+.4f}`여서 가장 깊은 group의 증가는
마지막 두 interval에 민감하다.

공통 envelope에 대해 각 series를 회귀한 residual의 one-interval
partial lead에서는 H2->H16이 여섯 envelope 모두 양수이고 범위는
`{min(h2_h16_residual):+.3f}`--`{max(h2_h16_residual):+.3f}` (`n=8`)다.
첫 transition을 빼도 범위는
`{min(h2_h16_residual_drop):+.3f}`--`{max(h2_h16_residual_drop):+.3f}`
(`n=7`)다. 이것은 raw envelope를 제거한 뒤의 일관된 부호라는 약한
힌트다. 그러나 H1->H16의 같은 범위는
`{min(h1_h16_residual):+.3f}`--`{max(h1_h16_residual):+.3f}`로 음수이며,
표본은 매우 작고 일부 envelope가 source 또는 target horizon을
포함한다. 따라서 residual 결과는 H2가 H16을 구동한다는 causal proof가
아니라 다음 matched ablation을 정당화할 수 있는 탐색적 신호다.

## 증거 범위

이 결과는 seed 1337의 단일 training trajectory와 고정 64-example
validation aggregate의 10개 epoch 지점에 한정된다. 인접 차분은 같은
checkpoint를 공유하고 horizon도 같은 모델과 validation token을
공유한다. cosine learning-rate 감소와 공통 초기 학습이 상관의
confound이며, 원 TSV에는 per-example NLL이 없어 표준오차를 복원할 수
없다. 따라서 상관은 기술통계이고 causal direction 또는 H2가 H16을
구동한다는 증거가 아니다. 정확한 interval, 곡률, horizon 요약과 모든
민감도 행은 각각의 TSV에 보존한다.
"""


def write_run_tsv(
    record_dir: Path,
    digest: str,
    source_row_count: int,
    maximum_block_mean_error: float,
) -> None:
    rows = [
        {"key": "experiment_id", "value": EXPERIMENT_ID},
        {"key": "analysis_type", "value": "retrospective_post_hoc"},
        {"key": "producer", "value": Path(__file__).name},
        {"key": "parent_experiment_id", "value": PARENT_EXPERIMENT_ID},
        {"key": "source", "value": SOURCE_RELATIVE_PATH.as_posix()},
        {"key": "source_sha256", "value": digest},
        {"key": "source_data_row_count", "value": source_row_count},
        {"key": "seed", "value": SEED},
        {"key": "split", "value": "wikitext103_bytes_train_fixed_validation_test_unread"},
        {"key": "selected_steps", "value": ",".join(map(str, EXPECTED_STEPS))},
        {"key": "selected_epoch_count", "value": len(EPOCHS)},
        {"key": "horizon_count", "value": len(HORIZONS)},
        {"key": "intervals_per_horizon", "value": len(EPOCHS) - 1},
        {"key": "curvature_centers_per_horizon", "value": len(EPOCHS) - 2},
        {"key": "common_envelope_count", "value": 6},
        {"key": "relative_normalization", "value": "signed_improvement_over_positive_common_envelope"},
        {"key": "share_normalization", "value": "positive_improvement_over_all_horizon_positive_total"},
        {"key": "residual_model", "value": "intercept_plus_common_envelope_then_one_interval_partial_lead"},
        {"key": "peak_lr", "value": PEAK_LR},
        {"key": "warmup_steps", "value": WARMUP_STEPS},
        {"key": "schedule_steps", "value": SCHEDULE_STEPS},
        {"key": "block_mean_tolerance", "value": BLOCK_MEAN_TOLERANCE},
        {"key": "maximum_observed_block_mean_error", "value": maximum_block_mean_error},
    ]
    write_tsv(record_dir / "run.tsv", rows, ("key", "value"))


def validate_row_counts(
    epoch_rows: Sequence[dict[str, object]],
    interval_rows: Sequence[dict[str, object]],
    curvature_rows: Sequence[dict[str, object]],
    horizon_rows: Sequence[dict[str, object]],
    cascade_rows: Sequence[dict[str, object]],
    acceleration_rows: Sequence[dict[str, object]],
    envelope_interval_rows: Sequence[dict[str, object]],
    envelope_summary_rows: Sequence[dict[str, object]],
    relative_interval_rows: Sequence[dict[str, object]],
    relative_change_rows: Sequence[dict[str, object]],
    relative_summary_rows: Sequence[dict[str, object]],
    relative_consensus_rows: Sequence[dict[str, object]],
    share_interval_rows: Sequence[dict[str, object]],
    share_summary_rows: Sequence[dict[str, object]],
    group_share_rows: Sequence[dict[str, object]],
    residual_lead_rows: Sequence[dict[str, object]],
) -> None:
    expected = {
        "epoch": len(EPOCHS) * len(HORIZONS),
        "interval": (len(EPOCHS) - 1) * len(HORIZONS),
        "curvature": (len(EPOCHS) - 2) * len(HORIZONS),
        "horizon": len(HORIZONS),
        "cascade": 20 * 3 * 2 * 3,
        "cascade_acceleration": 20 * 2 * 2,
        "envelope_interval": 6 * (len(EPOCHS) - 1),
        "envelope_summary": 6,
        "relative_interval": 6 * len(HORIZONS) * (len(EPOCHS) - 1),
        "relative_change": 6 * len(HORIZONS) * (len(EPOCHS) - 2),
        "relative_summary": 6 * len(HORIZONS),
        "relative_consensus": len(HORIZONS),
        "share_interval": len(HORIZONS) * (len(EPOCHS) - 1),
        "share_summary": len(HORIZONS),
        "group_share": 4,
        "residual_lead": 20 * 6 * 2,
    }
    observed = {
        "epoch": len(epoch_rows),
        "interval": len(interval_rows),
        "curvature": len(curvature_rows),
        "horizon": len(horizon_rows),
        "cascade": len(cascade_rows),
        "cascade_acceleration": len(acceleration_rows),
        "envelope_interval": len(envelope_interval_rows),
        "envelope_summary": len(envelope_summary_rows),
        "relative_interval": len(relative_interval_rows),
        "relative_change": len(relative_change_rows),
        "relative_summary": len(relative_summary_rows),
        "relative_consensus": len(relative_consensus_rows),
        "share_interval": len(share_interval_rows),
        "share_summary": len(share_summary_rows),
        "group_share": len(group_share_rows),
        "residual_lead": len(residual_lead_rows),
    }
    if observed != expected:
        raise AssertionError(f"analysis row counts differ: {observed} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the registered fused-scan epoch metric tape."
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=DEFAULT_RECORD_DIR,
        help="output record directory; the source tape is intentionally fixed",
    )
    args = parser.parse_args()
    record_dir = args.record_dir.resolve()
    record_dir.mkdir(parents=True, exist_ok=True)

    points, digest, source_row_count, maximum_block_mean_error = (
        load_epoch_points(SOURCE_PATH)
    )
    average_lrs = interval_lrs(points)
    epoch_rows = make_epoch_rows(points)
    interval_rows, improvements_by_horizon = make_interval_rows(
        points, average_lrs
    )
    curvature_rows = make_curvature_rows(points)
    horizon_rows, horizon_summaries = make_horizon_summary_rows(
        points, average_lrs
    )
    pairs = cascade_pairs(improvements_by_horizon)
    cascade_rows = make_cascade_correlation_rows(pairs, average_lrs)
    acceleration_rows = make_cascade_acceleration_rows(pairs, average_lrs)
    envelopes = make_common_envelopes(
        points, improvements_by_horizon, average_lrs
    )
    envelope_interval_rows, envelope_summary_rows = make_envelope_rows(
        envelopes
    )
    (
        relative_interval_rows,
        relative_change_rows,
        relative_summary_rows,
        relative_consensus_rows,
    ) = make_relative_improvement_rows(improvements_by_horizon, envelopes)
    (
        share_interval_rows,
        share_summary_rows,
        group_share_rows,
    ) = make_horizon_share_rows(improvements_by_horizon)
    residual_lead_rows = make_residual_lead_lag_rows(pairs, envelopes)
    block_summary = summarize_series(
        [point.block_nll for point in points], average_lrs
    )
    aggregate_rows = aggregate_metric_rows(
        points,
        horizon_summaries,
        block_summary,
        cascade_rows,
        acceleration_rows,
        envelope_summary_rows,
        relative_summary_rows,
        relative_consensus_rows,
        group_share_rows,
        residual_lead_rows,
    )
    validate_row_counts(
        epoch_rows,
        interval_rows,
        curvature_rows,
        horizon_rows,
        cascade_rows,
        acceleration_rows,
        envelope_interval_rows,
        envelope_summary_rows,
        relative_interval_rows,
        relative_change_rows,
        relative_summary_rows,
        relative_consensus_rows,
        share_interval_rows,
        share_summary_rows,
        group_share_rows,
        residual_lead_rows,
    )

    write_run_tsv(
        record_dir,
        digest,
        source_row_count,
        maximum_block_mean_error,
    )
    write_tsv(
        record_dir / "epoch_nll.tsv",
        epoch_rows,
        ("epoch", "step", "horizon", "nll", "block_nll"),
    )
    write_tsv(
        record_dir / "interval_metrics.tsv",
        interval_rows,
        (
            "horizon",
            "interval_index",
            "start_epoch",
            "end_epoch",
            "start_step",
            "end_step",
            "start_nll",
            "end_nll",
            "nll_velocity_per_epoch",
            "nll_improvement_per_epoch",
            "average_learning_rate",
            "average_learning_rate_fraction_of_peak",
            "peak_lr_equivalent_improvement",
        ),
    )
    write_tsv(
        record_dir / "curvature_metrics.tsv",
        curvature_rows,
        (
            "horizon",
            "center_epoch",
            "previous_step",
            "center_step",
            "next_step",
            "previous_nll",
            "center_nll",
            "next_nll",
            "previous_nll_velocity",
            "next_nll_velocity",
            "nll_second_difference",
            "improvement_rate_change",
            "positive_nll_acceleration",
            "improvement_rate_accelerated",
        ),
    )
    horizon_fields = ("horizon", *tuple(horizon_rows[0].keys())[1:])
    write_tsv(
        record_dir / "horizon_summary.tsv",
        horizon_rows,
        horizon_fields,
    )
    write_tsv(
        record_dir / "cascade_correlations.tsv",
        cascade_rows,
        (
            "pair_index",
            "source_series",
            "source_min_horizon",
            "source_max_horizon",
            "target_series",
            "target_min_horizon",
            "target_max_horizon",
            "transform",
            "drop_first_interval",
            "lag_intervals",
            "lag_direction",
            "sample_count",
            "pearson_r",
            "spearman_rho",
        ),
    )
    write_tsv(
        record_dir / "cascade_acceleration.tsv",
        acceleration_rows,
        (
            "pair_index",
            "source_series",
            "source_min_horizon",
            "source_max_horizon",
            "target_series",
            "target_min_horizon",
            "target_max_horizon",
            "transform",
            "drop_first_transition",
            "sample_count",
            "pearson_r",
            "spearman_rho",
        ),
    )
    write_tsv(
        record_dir / "common_envelopes.tsv",
        envelope_interval_rows,
        (
            "envelope",
            "envelope_kind",
            "source_horizons",
            "interval_index",
            "start_epoch",
            "end_epoch",
            "envelope_value",
        ),
    )
    write_tsv(
        record_dir / "envelope_summary.tsv",
        envelope_summary_rows,
        (
            "envelope",
            "envelope_kind",
            "source_horizons",
            "interval_count",
            "minimum_value",
            "maximum_value",
            "median_value",
            "minimum_to_median_ratio",
            "maximum_to_minimum_ratio",
            "denominator_amplification_warning_ge_20x",
            "raw_ols_slope",
            "raw_theil_sen_slope",
            "log_ols_slope",
            "log_fit_r_squared",
        ),
    )
    write_tsv(
        record_dir / "relative_improvement_metrics.tsv",
        relative_interval_rows,
        (
            "horizon",
            "envelope",
            "interval_index",
            "start_epoch",
            "end_epoch",
            "signed_improvement",
            "positive_improvement",
            "envelope_value",
            "signed_relative_improvement",
            "positive_relative_improvement",
        ),
    )
    write_tsv(
        record_dir / "relative_acceleration_metrics.tsv",
        relative_change_rows,
        (
            "horizon",
            "envelope",
            "previous_interval_index",
            "current_interval_index",
            "center_epoch",
            "previous_signed_relative_improvement",
            "current_signed_relative_improvement",
            "signed_relative_improvement_change",
            "relative_improvement_accelerated",
        ),
    )
    write_tsv(
        record_dir / "relative_horizon_summary.tsv",
        relative_summary_rows,
        (
            "horizon",
            "envelope",
            "interval_count",
            "signed_early_mean",
            "signed_late_mean",
            "signed_late_minus_early",
            "signed_late_to_early_ratio",
            "signed_ols_slope",
            "signed_theil_sen_slope",
            "signed_time_pearson",
            "positive_signed_change_count",
            "positive_early_mean",
            "positive_late_mean",
            "positive_late_minus_early",
            "positive_late_to_early_ratio",
            "positive_ols_slope",
            "positive_theil_sen_slope",
        ),
    )
    write_tsv(
        record_dir / "relative_horizon_consensus.tsv",
        relative_consensus_rows,
        (
            "horizon",
            "envelope_count",
            "positive_signed_late_minus_early_envelope_count",
            "positive_signed_ols_slope_envelope_count",
            "positive_signed_theil_sen_slope_envelope_count",
            "positive_only_late_minus_early_envelope_count",
            "positive_only_ols_slope_envelope_count",
            "positive_only_theil_sen_slope_envelope_count",
        ),
    )
    write_tsv(
        record_dir / "horizon_share_metrics.tsv",
        share_interval_rows,
        (
            "horizon",
            "interval_index",
            "start_epoch",
            "end_epoch",
            "signed_improvement",
            "positive_improvement",
            "all_horizon_positive_improvement_total",
            "positive_horizon_share",
        ),
    )
    write_tsv(
        record_dir / "horizon_share_summary.tsv",
        share_summary_rows,
        (
            "horizon",
            "early_mean_positive_share",
            "late_mean_positive_share",
            "late_minus_early_positive_share",
            "late_to_early_positive_share_ratio",
            "share_ols_slope",
            "share_theil_sen_slope",
            "share_time_pearson",
            "cumulative_positive_improvement_share",
        ),
    )
    write_tsv(
        record_dir / "horizon_group_share_summary.tsv",
        group_share_rows,
        (
            "group",
            "minimum_horizon",
            "maximum_horizon",
            "early_mean_positive_share",
            "late_mean_positive_share",
            "late_minus_early_positive_share",
            "late_to_early_positive_share_ratio",
            "share_ols_slope",
            "share_theil_sen_slope",
            "share_time_pearson",
        ),
    )
    write_tsv(
        record_dir / "residual_lead_lag.tsv",
        residual_lead_rows,
        (
            "pair_index",
            "source_series",
            "target_series",
            "envelope",
            "drop_first_transition",
            "sample_count",
            "source_envelope_intercept",
            "source_envelope_slope",
            "source_envelope_fit_r_squared",
            "target_envelope_intercept",
            "target_envelope_slope",
            "target_envelope_fit_r_squared",
            "residual_source_lead_pearson",
            "residual_source_lead_spearman",
            "next_envelope_partial_pearson",
            "next_envelope_partial_regression_slope",
            "next_envelope_partial_delta_r_squared",
        ),
    )
    write_tsv(
        record_dir / "aggregate_metrics.tsv",
        aggregate_rows,
        ("section", "metric", "value", "unit", "scope"),
    )
    (record_dir / "interpretation.md").write_text(
        interpretation_text(
            horizon_summaries,
            block_summary,
            cascade_rows,
            acceleration_rows,
            envelope_summary_rows,
            relative_summary_rows,
            relative_consensus_rows,
            group_share_rows,
            residual_lead_rows,
        ),
        encoding="utf-8",
    )

    print(
        f"wrote retrospective analysis to {record_dir} "
        f"from {len(points)} epochs, {len(interval_rows)} intervals, "
        f"{len(cascade_rows)} raw lead-lag rows, and "
        f"{len(relative_interval_rows)} envelope-relative rows"
    )


if __name__ == "__main__":
    main()
