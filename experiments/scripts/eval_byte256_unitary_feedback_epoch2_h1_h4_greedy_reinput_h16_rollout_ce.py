"""Compare epoch-2 natural greedy re-input chunks over gold H16 blocks."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import torch

from rotlm.evaluation import greedy_block_reinput_rollout_metrics
import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
ROOT = Path("experiments/records")
H1_ID = "EXP-20260805-byte256-unitary-feedback-h1-stride1-h16-monitor-2epoch-13m"
H4_ID = "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m"
SCAN_ID = "EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090"
RECORD = ROOT / "EXP-20260805-byte256-unitary-feedback-h1-h4loss-epoch2-greedy-reinput-h16-rollout-ce"
OUTPUT = Path("outputs/experiments")
STEP = 6_714
HORIZONS = 16
SAMPLES = 64
OBJECTIVES = {
    H1_ID: "one_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_attached_stride1_h16_monitor",
    H4_ID: "four_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_attached_stride4_h16_monitor",
}


def read_step(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if int(row["step"]) == STEP
        ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one step-{STEP} row in {path}")
    return rows[0]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def evaluate(checkpoint_path: Path, experiment_id: str, block: int, windows):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected = {
        "experiment_id": experiment_id,
        "objective": OBJECTIVES[experiment_id],
        "schedule_steps": 33_570,
        "batch": 64,
        "microbatch": 64,
    }
    if int(checkpoint["step"]) != STEP:
        raise RuntimeError(f"wrong checkpoint step: {checkpoint_path}")
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"{checkpoint_path} {key} mismatch")
    model = base.make_model().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    if base.parameter_count(model) != 13_215_008:
        raise RuntimeError("parameter count changed")
    metrics = greedy_block_reinput_rollout_metrics(
        model,
        windows,
        context_length=base.CONTEXT,
        horizons=HORIZONS,
        block_size=block,
        anchor_stride=16,
        microbatch=8,
    )
    info = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": digest(checkpoint_path),
        "experiment_id": experiment_id,
    }
    del model, checkpoint
    torch.cuda.empty_cache()
    return metrics, info


def write_rows(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    h1_record, h4_record = ROOT / H1_ID, ROOT / H4_ID
    scan_record = ROOT / SCAN_ID / "epoch-checkpoint-run"
    scan_starts = ROOT / (
        "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-"
        "normalized-residual-h16-ce-only-attached-stride16-1000step-13m"
    ) / "validation_starts.tsv"
    starts_paths = (
        h1_record / "validation_starts.tsv",
        h4_record / "validation_starts.tsv",
        scan_starts,
    )
    if len({path.read_bytes() for path in starts_paths}) != 1:
        raise RuntimeError("validation starts differ")
    starts = base.load_fixed_starts(starts_paths[0], SAMPLES)
    windows = base.windows_of_length(
        base.memmap("validation"), starts, base.CONTEXT + HORIZONS
    )
    h1, h1_info = evaluate(OUTPUT / H1_ID / "step6714.pt", H1_ID, 1, windows)
    h4, h4_info = evaluate(OUTPUT / H4_ID / "step6714.pt", H4_ID, 4, windows)
    scan = read_step(scan_record / "metrics.tsv")
    if h1.total_labels != 16_384 or h4.total_labels != 16_384:
        raise RuntimeError("unexpected label count")

    h1_open, h4_open = read_step(h1_record / "metrics.tsv"), read_step(h4_record / "metrics.tsv")
    h1_first_error = abs(h1.horizon_target_cross_entropy[0] - float(h1_open["val_h1_target_nll"]))
    h4_first_error = max(
        abs(h4.horizon_target_cross_entropy[index] - float(h4_open[f"val_h{index + 1}_target_nll"]))
        for index in range(4)
    )
    if max(h1_first_error, h4_first_error) >= 2e-3:
        raise RuntimeError("first natural chunk does not match open evaluation")

    modes = {
        "feedback_h1loss_greedy_block1": (1, h1.horizon_target_cross_entropy, h1.horizon_target_accuracy),
        "feedback_h4loss_greedy_block4": (4, h4.horizon_target_cross_entropy, h4.horizon_target_accuracy),
        "parallel_scan_open_block16": (
            16,
            tuple(float(scan[f"h{h}_nll"]) for h in range(1, 17)),
            tuple(float(scan[f"h{h}_accuracy"]) for h in range(1, 17)),
        ),
    }
    if not all(math.isfinite(value) for _, nll, acc in modes.values() for vector in (nll, acc) for value in vector):
        raise RuntimeError("non-finite metric")
    horizon_rows = [
        {
            "mode": mode,
            "chunk_size": chunk,
            "horizon": horizon,
            "gold_aligned_rollout_ce": nll[horizon - 1],
            "gold_target_accuracy": accuracy[horizon - 1],
            "labels": 1024,
        }
        for mode, (chunk, nll, accuracy) in modes.items()
        for horizon in range(1, 17)
    ]
    metric_rows = [
        {
            "mode": mode,
            "chunk_size": chunk,
            "h1_h16_mean_rollout_ce": sum(nll) / 16,
            "h1_h16_mean_accuracy": sum(accuracy) / 16,
            "labels": 16384,
        }
        for mode, (chunk, nll, accuracy) in modes.items()
    ]
    values = {row["mode"]: float(row["h1_h16_mean_rollout_ce"]) for row in metric_rows}
    comparison_rows = [
        {"comparison": "h1_block1_minus_h4_block4", "left_minus_right_ce": values["feedback_h1loss_greedy_block1"] - values["feedback_h4loss_greedy_block4"]},
        {"comparison": "h1_block1_minus_scan_block16", "left_minus_right_ce": values["feedback_h1loss_greedy_block1"] - values["parallel_scan_open_block16"]},
        {"comparison": "h4_block4_minus_scan_block16", "left_minus_right_ce": values["feedback_h4loss_greedy_block4"] - values["parallel_scan_open_block16"]},
    ]
    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "horizon_metrics.tsv", horizon_rows)
    write_rows(RECORD / "metrics.tsv", metric_rows)
    write_rows(RECORD / "comparisons.tsv", comparison_rows)
    with (RECORD / "run.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows([
            ("key", "value"), ("seed", base.SEED), ("split", "wikitext103_bytes_validation"),
            ("samples", SAMPLES), ("anchors_per_sample", 16), ("labels_per_mode", 16384),
            ("policy", "greedy_argmax"), ("gold_reset_interval", 16),
            ("h1_first_chunk_max_error", h1_first_error), ("h4_first_chunk_max_error", h4_first_error),
            ("h1_checkpoint_sha256", h1_info["checkpoint_sha256"]),
            ("h4_checkpoint_sha256", h4_info["checkpoint_sha256"]),
            ("validation_starts_sha256", digest(starts_paths[0])),
        ])
    (RECORD / "interpretation.md").write_text(
        "# Epoch-2 greedy self-reinput H16 block comparison\n\n"
        "Gold context resets only at each sixteen-token boundary. Inside it, H1 re-enters one greedy token, H4 re-enters four greedy tokens, and scan predicts sixteen open-loop. Values are gold-aligned rollout CE, not teacher-forced likelihood.\n\n"
        + "\n".join(f"- {row['mode']}: {float(row['h1_h16_mean_rollout_ce']):.9f}" for row in metric_rows)
        + "\n",
        encoding="utf-8",
    )
    for row in metric_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
