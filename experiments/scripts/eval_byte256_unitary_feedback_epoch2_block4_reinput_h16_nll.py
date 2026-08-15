"""Evaluate proper H16 blockwise NLL with gold re-input every four bytes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import torch

from rotlm.evaluation import gold_block_reinput_metrics

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
EXPERIMENT_ID = (
    "EXP-20260805-byte256-unitary-feedback-h4loss-epoch2-"
    "block4-reinput-h16-nll"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "10epoch-13m/step6714.pt"
)
PARENT_RECORD = Path(
    "experiments/records/"
    "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "10epoch-13m"
)
SCAN_RECORD = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-h16-"
    "10epoch-13m-rtx5090/epoch-checkpoint-run"
)
SCAN_VALIDATION_STARTS = Path(
    "experiments/records/"
    "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-"
    "normalized-residual-h16-ce-only-attached-stride16-1000step-13m/"
    "validation_starts.tsv"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
CHECKPOINT_STEP = 6_714
BLOCK_SIZE = 4
BLOCKS = 4
HORIZONS = BLOCK_SIZE * BLOCKS
ANCHOR_STRIDE = 16
SAMPLES = 64


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


def write_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["step"]) != CHECKPOINT_STEP:
        raise RuntimeError("expected the preserved epoch-2 checkpoint")
    model = base.make_model().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    if base.parameter_count(model) != base.EXPECTED_PARAMETERS:
        raise RuntimeError("model parameter count changed")

    current = read_step(PARENT_RECORD / "metrics.tsv", CHECKPOINT_STEP)
    scan = read_step(SCAN_RECORD / "metrics.tsv", CHECKPOINT_STEP)
    parent_starts_path = PARENT_RECORD / "validation_starts.tsv"
    if parent_starts_path.read_bytes() != SCAN_VALIDATION_STARTS.read_bytes():
        raise RuntimeError(
            "feedback and parallel-scan validation starts differ"
        )
    starts = base.load_fixed_starts(
        parent_starts_path,
        SAMPLES,
    )
    validation = base.memmap("validation")
    windows = base.windows_of_length(
        validation,
        starts,
        base.CONTEXT + HORIZONS,
    )
    torch.cuda.reset_peak_memory_stats()
    reinput = gold_block_reinput_metrics(
        model,
        windows,
        context_length=base.CONTEXT,
        horizons=HORIZONS,
        block_size=BLOCK_SIZE,
        anchor_stride=ANCHOR_STRIDE,
        microbatch=args.microbatch,
    )
    per_horizon_count = reinput.labels_per_horizon

    modes = {
        "feedback_open_h16": (
            [
                float(current[f"val_h{horizon}_target_nll"])
                for horizon in range(1, HORIZONS + 1)
            ],
            [
                float(current[f"val_h{horizon}_target_accuracy"])
                for horizon in range(1, HORIZONS + 1)
            ],
        ),
        "feedback_gold_block4_reinput": (
            reinput.horizon_nll,
            reinput.horizon_accuracy,
        ),
        "parallel_scan_open_h16": (
            [
                float(scan[f"h{horizon}_nll"])
                for horizon in range(1, HORIZONS + 1)
            ],
            [
                float(scan[f"h{horizon}_accuracy"])
                for horizon in range(1, HORIZONS + 1)
            ],
        ),
    }
    if not all(
        math.isfinite(value)
        for values in modes.values()
        for vector in values
        for value in vector
    ):
        raise RuntimeError("non-finite evaluation metric")
    first_block_error = max(
        abs(
            modes["feedback_open_h16"][0][index]
            - modes["feedback_gold_block4_reinput"][0][index]
        )
        for index in range(BLOCK_SIZE)
    )
    if first_block_error >= 2e-3:
        raise RuntimeError(
            f"first re-input block changed open H1-H4 NLL: {first_block_error}"
        )

    args.record_dir.mkdir(parents=True, exist_ok=True)
    horizon_rows = [
        {
            "mode": mode,
            "absolute_horizon": horizon,
            "block_1based": (horizon - 1) // BLOCK_SIZE + 1,
            "within_block_horizon": (horizon - 1) % BLOCK_SIZE + 1,
            "nll": values[0][horizon - 1],
            "accuracy": values[1][horizon - 1],
            "labels": per_horizon_count,
        }
        for mode, values in modes.items()
        for horizon in range(1, HORIZONS + 1)
    ]
    write_rows(
        args.record_dir / "horizon_metrics.tsv",
        tuple(horizon_rows[0]),
        horizon_rows,
    )
    block_rows = []
    for mode, (nll_values, accuracy_values) in modes.items():
        for block in range(BLOCKS):
            selected = slice(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE)
            block_rows.append({
                "mode": mode,
                "block_1based": block + 1,
                "absolute_horizon_start": block * BLOCK_SIZE + 1,
                "absolute_horizon_end": (block + 1) * BLOCK_SIZE,
                "mean_nll": sum(nll_values[selected]) / BLOCK_SIZE,
                "mean_accuracy": sum(accuracy_values[selected]) / BLOCK_SIZE,
                "labels": per_horizon_count * BLOCK_SIZE,
            })
    write_rows(
        args.record_dir / "block_metrics.tsv",
        tuple(block_rows[0]),
        block_rows,
    )
    metric_rows = [
        {
            "mode": mode,
            "h16_block_nll": sum(values[0]) / HORIZONS,
            "h16_block_accuracy": sum(values[1]) / HORIZONS,
            "labels": per_horizon_count * HORIZONS,
        }
        for mode, values in modes.items()
    ]
    write_rows(
        args.record_dir / "metrics.tsv",
        tuple(metric_rows[0]),
        metric_rows,
    )
    run_rows = [
        ("key", "value"),
        ("experiment_id", EXPERIMENT_ID),
        ("checkpoint", args.checkpoint),
        ("checkpoint_sha256", sha256(args.checkpoint)),
        ("checkpoint_step", CHECKPOINT_STEP),
        ("seed", base.SEED),
        ("split", "wikitext103_bytes_validation"),
        ("samples", SAMPLES),
        ("anchors_per_sample", len(range(0, base.CONTEXT, ANCHOR_STRIDE))),
        ("per_horizon_labels", per_horizon_count),
        ("total_labels", per_horizon_count * HORIZONS),
        ("block_size", BLOCK_SIZE),
        ("blocks", BLOCKS),
        ("microbatch", args.microbatch),
        ("parameters", base.parameter_count(model)),
        ("precision", "strict_float32_no_tf32"),
        ("gpu", torch.cuda.get_device_name()),
        ("torch", torch.__version__),
        ("first_block_open_nll_max_error", first_block_error),
        ("peak_vram_bytes", torch.cuda.max_memory_allocated()),
    ]
    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(run_rows)

    aggregate = {row["mode"]: row for row in metric_rows}
    feedback_open = aggregate["feedback_open_h16"]["h16_block_nll"]
    feedback_reinput = aggregate[
        "feedback_gold_block4_reinput"
    ]["h16_block_nll"]
    scan_open = aggregate["parallel_scan_open_h16"]["h16_block_nll"]
    lines = [
        "# Epoch-2 block-4 gold-reinput H16 NLL",
        "",
        "Every four gold validation bytes were appended and canonically "
        "re-encoded before predicting the next four-byte block. This is a "
        "proper blockwise teacher-forced likelihood; it is not free-running "
        "greedy re-input.",
        "",
        "Each boundary re-encoding initializes the central complex memory "
        "from the canonical gold prefix root. It does not preserve the "
        "previous four-step rollout memory while replacing tokens.",
        "",
        "| Mode | H1--H16 block NLL |",
        "|---|---:|",
        f"| Feedback open H16 | {feedback_open:.6f} |",
        f"| Feedback gold block-4 reinput | {feedback_reinput:.6f} |",
        f"| Parallel scan open H16 | {scan_open:.6f} |",
        "",
        f"Gold block-4 reinput changed feedback NLL by "
        f"{feedback_reinput - feedback_open:+.6f} versus open H16 and was "
        f"{feedback_reinput - scan_open:+.6f} relative to the epoch-matched "
        "parallel scan.",
        "",
        f"The parallel-scan H16 single-horizon NLL was "
        f"{float(scan['h16_nll']):.6f}; its H1--H16 mean block NLL was "
        f"{scan_open:.6f}. These are different quantities.",
        "The feedback reinput and parallel-scan open-H16 rows also use "
        "different conditioning protocols, so their difference is the "
        "requested metric comparison rather than a pure architecture "
        "ablation.",
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
