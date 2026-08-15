"""Compare epoch-2 H1/H4 feedback models under gold token re-entry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import torch

from rotlm.evaluation import (
    GoldBlockReinputMetrics,
    gold_block_reinput_metrics,
)
import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
EXPERIMENT_ID = (
    "EXP-20260805-byte256-unitary-feedback-h1-h4loss-epoch2-"
    "token1-reinput-h16-nll"
)
H1_EXPERIMENT_ID = (
    "EXP-20260805-byte256-unitary-feedback-h1-stride1-h16-monitor-"
    "2epoch-13m"
)
H4_EXPERIMENT_ID = (
    "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "10epoch-13m"
)
SCAN_EXPERIMENT_ID = (
    "EXP-20260803-byte256-unitary-fused-triton-scan-h16-"
    "10epoch-13m-rtx5090"
)
DEFAULT_H1_CHECKPOINT = (
    Path("outputs/experiments") / H1_EXPERIMENT_ID / "step6714.pt"
)
DEFAULT_H4_CHECKPOINT = (
    Path("outputs/experiments") / H4_EXPERIMENT_ID / "step6714.pt"
)
H1_RECORD = Path("experiments/records") / H1_EXPERIMENT_ID
H4_RECORD = Path("experiments/records") / H4_EXPERIMENT_ID
H4_BLOCK4_RECORD = Path(
    "experiments/records/"
    "EXP-20260805-byte256-unitary-feedback-h4loss-epoch2-"
    "block4-reinput-h16-nll"
)
SCAN_RECORD = (
    Path("experiments/records")
    / SCAN_EXPERIMENT_ID
    / "epoch-checkpoint-run"
)
SCAN_VALIDATION_STARTS = Path(
    "experiments/records/"
    "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-"
    "normalized-residual-h16-ce-only-attached-stride16-1000step-13m/"
    "validation_starts.tsv"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
CHECKPOINT_STEP = 6_714
HORIZONS = 16
REINPUT_BLOCK_SIZE = 1
ANCHOR_STRIDE = 16
SAMPLES = 64
EXPECTED_OBJECTIVES = {
    H1_EXPERIMENT_ID: (
        "one_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_"
        "attached_stride1_h16_monitor"
    ),
    H4_EXPERIMENT_ID: (
        "four_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_"
        "attached_stride4_h16_monitor"
    ),
}


def read_step(path: Path, step: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if int(row["step"]) == step]
    if len(selected) != 1:
        raise RuntimeError(f"expected one step-{step} row in {path}")
    return selected[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def open_feedback_metrics(row: dict[str, str]) -> tuple[list[float], list[float]]:
    return (
        [
            float(row[f"val_h{horizon}_target_nll"])
            for horizon in range(1, HORIZONS + 1)
        ],
        [
            float(row[f"val_h{horizon}_target_accuracy"])
            for horizon in range(1, HORIZONS + 1)
        ],
    )


def open_scan_metrics(row: dict[str, str]) -> tuple[list[float], list[float]]:
    return (
        [
            float(row[f"h{horizon}_nll"])
            for horizon in range(1, HORIZONS + 1)
        ],
        [
            float(row[f"h{horizon}_accuracy"])
            for horizon in range(1, HORIZONS + 1)
        ],
    )


def recorded_horizon_metrics(
    path: Path,
    mode: str,
) -> tuple[list[float], list[float]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["mode"] == mode
        ]
    rows.sort(key=lambda row: int(row["absolute_horizon"]))
    if [int(row["absolute_horizon"]) for row in rows] != list(
        range(1, HORIZONS + 1)
    ):
        raise RuntimeError(f"expected H1--H16 for {mode} in {path}")
    if any(int(row["labels"]) != SAMPLES * 16 for row in rows):
        raise RuntimeError(f"unexpected label count for {mode} in {path}")
    return (
        [float(row["nll"]) for row in rows],
        [float(row["accuracy"]) for row in rows],
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    windows: torch.Tensor,
    *,
    microbatch: int,
    expected_experiment_id: str,
) -> tuple[GoldBlockReinputMetrics, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["step"]) != CHECKPOINT_STEP:
        raise RuntimeError(
            f"{checkpoint_path} is not the preserved epoch-2 checkpoint"
        )
    config = checkpoint.get("config", {})
    expected_config = {
        "experiment_id": expected_experiment_id,
        "objective": EXPECTED_OBJECTIVES[expected_experiment_id],
        "schedule_steps": 33_570,
        "batch": 64,
        "microbatch": 64,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise RuntimeError(
                f"{checkpoint_path} config {name} mismatch: "
                f"{config.get(name)!r} != {expected!r}"
            )
    model = base.make_model().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    parameters = base.parameter_count(model)
    if parameters != base.EXPECTED_PARAMETERS:
        raise RuntimeError("model parameter count changed")

    torch.cuda.reset_peak_memory_stats()
    metrics = gold_block_reinput_metrics(
        model,
        windows,
        context_length=base.CONTEXT,
        horizons=HORIZONS,
        block_size=REINPUT_BLOCK_SIZE,
        anchor_stride=ANCHOR_STRIDE,
        microbatch=microbatch,
    )
    run_values = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_experiment_id": config["experiment_id"],
        "checkpoint_objective": config["objective"],
        "parameters": parameters,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    del model, checkpoint
    torch.cuda.empty_cache()
    return metrics, run_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h1-checkpoint",
        type=Path,
        default=DEFAULT_H1_CHECKPOINT,
    )
    parser.add_argument(
        "--h4-checkpoint",
        type=Path,
        default=DEFAULT_H4_CHECKPOINT,
    )
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--microbatch", type=int, default=8)
    args = parser.parse_args()
    if args.microbatch < 1 or SAMPLES % args.microbatch:
        parser.error("microbatch must be a positive divisor of 64")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    h1_open_row = read_step(H1_RECORD / "metrics.tsv", CHECKPOINT_STEP)
    h4_open_row = read_step(H4_RECORD / "metrics.tsv", CHECKPOINT_STEP)
    scan_open_row = read_step(SCAN_RECORD / "metrics.tsv", CHECKPOINT_STEP)
    starts_paths = (
        H1_RECORD / "validation_starts.tsv",
        H4_RECORD / "validation_starts.tsv",
        SCAN_VALIDATION_STARTS,
    )
    starts_bytes = [path.read_bytes() for path in starts_paths]
    if not all(value == starts_bytes[0] for value in starts_bytes[1:]):
        raise RuntimeError("H1, H4, and scan validation starts differ")
    starts = base.load_fixed_starts(starts_paths[0], SAMPLES)
    validation = base.memmap("validation")
    windows = base.windows_of_length(
        validation,
        starts,
        base.CONTEXT + HORIZONS,
    )

    h1_reinput, h1_run = evaluate_checkpoint(
        args.h1_checkpoint,
        windows,
        microbatch=args.microbatch,
        expected_experiment_id=H1_EXPERIMENT_ID,
    )
    h4_reinput, h4_run = evaluate_checkpoint(
        args.h4_checkpoint,
        windows,
        microbatch=args.microbatch,
        expected_experiment_id=H4_EXPERIMENT_ID,
    )
    if h1_reinput.labels_per_horizon != h4_reinput.labels_per_horizon:
        raise RuntimeError("H1 and H4 re-input label counts differ")
    if h1_reinput.total_labels != SAMPLES * 16 * HORIZONS:
        raise RuntimeError("unexpected token-one re-input label count")

    h1_open = open_feedback_metrics(h1_open_row)
    h4_open = open_feedback_metrics(h4_open_row)
    scan_open = open_scan_metrics(scan_open_row)
    h4_block4_reinput = recorded_horizon_metrics(
        H4_BLOCK4_RECORD / "horizon_metrics.tsv",
        "feedback_gold_block4_reinput",
    )
    modes = {
        "feedback_h1loss_open_h16": h1_open,
        "feedback_h1loss_gold_token1_reinput": (
            h1_reinput.horizon_nll,
            h1_reinput.horizon_accuracy,
        ),
        "feedback_h4loss_open_h16": h4_open,
        "feedback_h4loss_gold_token1_reinput": (
            h4_reinput.horizon_nll,
            h4_reinput.horizon_accuracy,
        ),
        "feedback_h4loss_gold_block4_reinput": h4_block4_reinput,
        "parallel_scan_open_h16": scan_open,
    }
    if not all(
        math.isfinite(value)
        for vectors in modes.values()
        for vector in vectors
        for value in vector
    ):
        raise RuntimeError("non-finite evaluation metric")

    first_horizon_errors = {
        "h1loss": abs(h1_open[0][0] - h1_reinput.horizon_nll[0]),
        "h4loss": abs(h4_open[0][0] - h4_reinput.horizon_nll[0]),
    }
    if max(first_horizon_errors.values()) >= 2e-3:
        raise RuntimeError(
            "gold token-one H1 changed the matching open H1 NLL: "
            f"{first_horizon_errors}"
        )
    for row, values in ((h1_open_row, h1_open), (h4_open_row, h4_open)):
        recorded_mean = float(row["val_block_validation_nll"])
        if abs(recorded_mean - sum(values[0]) / HORIZONS) >= 1e-9:
            raise RuntimeError("feedback aggregate does not match horizons")

    per_horizon_count = h1_reinput.labels_per_horizon
    horizon_rows = [
        {
            "mode": mode,
            "absolute_horizon": horizon,
            "nll": values[0][horizon - 1],
            "accuracy": values[1][horizon - 1],
            "labels": per_horizon_count,
        }
        for mode, values in modes.items()
        for horizon in range(1, HORIZONS + 1)
    ]
    metric_rows = [
        {
            "mode": mode,
            "h1_h16_mean_nll": sum(values[0]) / HORIZONS,
            "h1_h16_mean_accuracy": sum(values[1]) / HORIZONS,
            "labels": per_horizon_count * HORIZONS,
        }
        for mode, values in modes.items()
    ]
    aggregate = {
        row["mode"]: float(row["h1_h16_mean_nll"])
        for row in metric_rows
    }
    token1_difference = (
        aggregate["feedback_h1loss_gold_token1_reinput"]
        - aggregate["feedback_h4loss_gold_token1_reinput"]
    )
    open_difference = (
        aggregate["feedback_h1loss_open_h16"]
        - aggregate["feedback_h4loss_open_h16"]
    )
    natural_chunk_difference = (
        aggregate["feedback_h1loss_gold_token1_reinput"]
        - aggregate["feedback_h4loss_gold_block4_reinput"]
    )
    comparisons = (
        (
            "h1_vs_h4_open",
            "feedback_h1loss_open_h16",
            "feedback_h4loss_open_h16",
        ),
        (
            "h1_vs_h4_gold_token1",
            "feedback_h1loss_gold_token1_reinput",
            "feedback_h4loss_gold_token1_reinput",
        ),
        (
            "natural_reinput_h1_token1_vs_h4_block4",
            "feedback_h1loss_gold_token1_reinput",
            "feedback_h4loss_gold_block4_reinput",
        ),
        (
            "h1_open_vs_parallel_scan_open",
            "feedback_h1loss_open_h16",
            "parallel_scan_open_h16",
        ),
        (
            "h4_open_vs_parallel_scan_open",
            "feedback_h4loss_open_h16",
            "parallel_scan_open_h16",
        ),
    )
    comparison_rows = [
        {
            "comparison": name,
            "left_mode": left,
            "right_mode": right,
            "left_minus_right_nll": aggregate[left] - aggregate[right],
        }
        for name, left, right in comparisons
    ]

    args.record_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.record_dir / "horizon_metrics.tsv", horizon_rows)
    write_rows(args.record_dir / "metrics.tsv", metric_rows)
    write_rows(args.record_dir / "comparisons.tsv", comparison_rows)
    run_rows = [
        ("key", "value"),
        ("experiment_id", EXPERIMENT_ID),
        ("seed", base.SEED),
        ("split", "wikitext103_bytes_validation"),
        ("samples", SAMPLES),
        ("anchors_per_sample", h1_reinput.anchors_per_sample),
        ("per_horizon_labels", per_horizon_count),
        ("total_labels_per_mode", h1_reinput.total_labels),
        ("horizons", HORIZONS),
        ("reinput_block_size", REINPUT_BLOCK_SIZE),
        ("microbatch", args.microbatch),
        ("precision", "strict_float32_no_tf32"),
        ("gpu", torch.cuda.get_device_name()),
        ("torch", torch.__version__),
        ("validation_starts_sha256", sha256(starts_paths[0])),
        (
            "h4_block4_horizon_metrics_sha256",
            sha256(H4_BLOCK4_RECORD / "horizon_metrics.tsv"),
        ),
        ("h1_first_horizon_open_nll_abs_error", first_horizon_errors["h1loss"]),
        ("h4_first_horizon_open_nll_abs_error", first_horizon_errors["h4loss"]),
        *[(f"h1_{key}", value) for key, value in h1_run.items()],
        *[(f"h4_{key}", value) for key, value in h4_run.items()],
    ]
    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(run_rows)

    lines = [
        "# Epoch-2 H1/H4 feedback: gold token-one H16 NLL",
        "",
        "At every absolute horizon, the complete gold prefix was "
        "canonically re-encoded before one state-dependent transition. "
        "This is proper token-level teacher forcing, not free-running "
        "generation.",
        "",
        "| Mode | H1--H16 mean NLL |",
        "|---|---:|",
        *[
            f"| {row['mode']} | {float(row['h1_h16_mean_nll']):.6f} |"
            for row in metric_rows
        ],
        "",
        f"Under each objective's natural chunking, H1 token-one minus H4 "
        f"block-four was {natural_chunk_difference:+.6f} NLL.",
        f"Under identical gold-token conditioning, H1-loss minus H4-loss "
        f"was {token1_difference:+.6f} NLL.",
        f"Open H16 H1-loss minus H4-loss was "
        f"{open_difference:+.6f} NLL.",
        "",
        f"The epoch-matched parallel-scan open-H16 mean was "
        f"{aggregate['parallel_scan_open_h16']:.6f}; its single H16 NLL "
        f"was {float(scan_open_row['h16_nll']):.6f}. These are different "
        "quantities.",
        "",
        "Open-H16 and gold-token-one rows answer different questions. The "
        "H1 open-H16 row is an out-of-training-contract diagnostic, not a "
        "performance ranking. "
        "Only the two gold-token-one rows are conditioning-matched for the "
        "H1-loss versus H4-loss comparison.",
        "The H1 token-one versus H4 block-four row is the requested natural "
        "chunk-size comparison, but it changes both training horizon and "
        "conditioning frequency.",
        "",
    ]
    (args.record_dir / "interpretation.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    for row in metric_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
