"""Relate exact H1--H4 successes to channel-resolved interference."""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


PHASE_RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-phase-interference-audit"
)
DECODE_RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-interpretability-audit"
)
RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-correct-shallow-interference-audit"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def visible(byte: int) -> str:
    if byte == 32:
        return "<space>"
    if byte == 10:
        return "<newline>"
    return bytes([byte]).decode("utf-8", errors="replace")


def main() -> None:
    horizons = {
        (int(row["sample"]), int(row["horizon"])): row
        for row in read_rows(DECODE_RECORD / "horizons.tsv")
        if int(row["horizon"]) <= 4
    }
    summaries = {
        (int(row["sample"]), int(row["read_horizon"])): row
        for row in read_rows(PHASE_RECORD / "summary.tsv")
        if int(row["read_horizon"]) <= 4
    }
    age_mass: dict[tuple[int, int], dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in read_rows(PHASE_RECORD / "channel_contributions.tsv"):
        horizon = int(row["read_horizon"])
        if horizon <= 4:
            key = (int(row["sample"]), horizon)
            age_mass[key][int(row["age"])] += abs(
                float(row["hidden_signed_projection_share"])
            )

    token_rows = []
    metric_names = (
        "positive_projection_mass",
        "negative_projection_mass",
        "gross_projection_mass",
        "negative_to_positive_ratio",
        "effective_abs_projection_terms",
        "dominant_abs_projection_share",
        "current_write_abs_projection_share",
        "effective_norm_terms",
        "coefficient_phase_coherence",
    )
    for key in sorted(horizons):
        decoded = horizons[key]
        summary = summaries[key]
        masses = age_mass[key]
        mass_total = sum(masses.values())
        age_centroid = sum(age * mass for age, mass in masses.items()) / mass_total
        target = int(decoded["target_byte"])
        predicted = int(decoded["predicted_byte"])
        correct = decoded["correct"] == "True"
        token_rows.append({
            "sample": key[0],
            "horizon": key[1],
            "target_byte": target,
            "target_visible": visible(target),
            "predicted_byte": predicted,
            "predicted_visible": visible(predicted),
            "correct": correct,
            "correct_space": correct and target == 32,
            "correct_nonspace": correct and target != 32,
            "top_probability": float(decoded["top_probability"]),
            **{name: float(summary[name]) for name in metric_names},
            "abs_projection_age_centroid": age_centroid,
        })

    groups = {
        "all_correct": [row for row in token_rows if row["correct"]],
        "all_incorrect": [row for row in token_rows if not row["correct"]],
        "correct_space": [row for row in token_rows if row["correct_space"]],
        "correct_nonspace": [row for row in token_rows if row["correct_nonspace"]],
    }
    aggregate_rows = []
    aggregate_metrics = (
        "top_probability",
        *metric_names,
        "abs_projection_age_centroid",
    )
    for group, rows in groups.items():
        aggregate_rows.append({
            "group": group,
            "count": len(rows),
            **{
                f"mean_{name}": statistics.mean(row[name] for row in rows)
                if rows else float("nan")
                for name in aggregate_metrics
            },
        })

    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "tokens.tsv", token_rows)
    write_rows(RECORD / "summary.tsv", aggregate_rows)


if __name__ == "__main__":
    main()
