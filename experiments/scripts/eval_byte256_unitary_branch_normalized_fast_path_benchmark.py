"""Compare the full and CE-only fast training compute paths at step 6000."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
CHECKPOINT = Path(
    "outputs/experiments/EXP-20260802-byte256-unitary-branch-normalized-"
    "h16-6000step-micro64/step6000.pt"
)
RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-fast-path-benchmark"
)
WARMUP = 3
REPEATS = 10


def measure(name, objective, checkpoint, window):
    model = base.make_model().cuda().train()
    model.load_state_dict(checkpoint["model"])
    initial = float(objective(model, window)[0].detach())
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(WARMUP + REPEATS):
        model.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss, _, _, output = objective(model, window)
        loss.backward()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        del loss, output
        if iteration >= WARMUP:
            timings.append(elapsed)
    peak = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    return {
        "path": name,
        "initial_loss": initial,
        "iterations": REPEATS,
        "mean_iteration_ms": statistics.mean(timings),
        "median_iteration_ms": statistics.median(timings),
        "min_iteration_ms": min(timings),
        "max_iteration_ms": max(timings),
        "peak_allocated_bytes": peak,
        "byte_tokens_per_s": 64 * base.TRAIN_WINDOW_LENGTH * 1000.0
        / statistics.median(timings),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    training = base.memmap("train")
    generator = torch.Generator().manual_seed(base.SEED + 6060)
    window = base.sampled_windows(
        training, 64, generator, base.TRAIN_WINDOW_LENGTH
    ).cuda()

    full = run.h4.objective
    fast = base.TRAIN_OBJECTIVE_OVERRIDE
    if fast is None:
        raise RuntimeError("fast objective is not installed")
    rows = [
        measure("full_diagnostic", full, checkpoint, window),
        measure("ce_only_fast", fast, checkpoint, window),
    ]
    full_row, fast_row = rows
    loss_abs_difference = abs(
        full_row["initial_loss"] - fast_row["initial_loss"]
    )
    torch.testing.assert_close(
        torch.tensor(full_row["initial_loss"]),
        torch.tensor(fast_row["initial_loss"]),
        atol=2e-4,
        rtol=1e-4,
    )
    for row in rows:
        row["loss_abs_difference"] = loss_abs_difference
        row["speedup_vs_full"] = (
            full_row["median_iteration_ms"] / row["median_iteration_ms"]
        )
        row["peak_vram_ratio_vs_full"] = (
            row["peak_allocated_bytes"] / full_row["peak_allocated_bytes"]
        )

    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
