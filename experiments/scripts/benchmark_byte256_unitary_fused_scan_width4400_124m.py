"""Measure width-only 124M fused-scan optimizer steps on one RTX 5090."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics

import torch

from rotlm.training.ha_skew_window import (
    sparse_complex_self_predicted_kv_ce_only_fast_loss,
)
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
WIDTH = 4400
BATCH = 64
WARMUP = 5
REPEATS = 15
WINDOW_SEED = base.SEED + 124_000_000
TRAIN_TOKENS = 55_000_000
SCHEDULE_STEPS = 33_570
RECORD = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-fused-scan-width4400-124m-rtx5090-eta"
)


def objective(model, window):
    return sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        central_rollout="time-varying-scan",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    original_width = base.WIDTH
    base.WIDTH = WIDTH
    try:
        model = base.make_model().cuda().train()
    finally:
        base.WIDTH = original_width
    transition = model.complex_self_prediction
    transition.use_fused_triton_rotating_frame_time_varying_scan = True
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if total_parameters != 123_941_528:
        raise RuntimeError(f"unexpected parameter count: {total_parameters}")

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=base.lr_at(0, SCHEDULE_STEPS),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    training = base.memmap("train")
    window = base.sampled_windows(
        training,
        BATCH,
        torch.Generator().manual_seed(WINDOW_SEED),
        base.TRAIN_WINDOW_LENGTH,
    )

    forward_ms = []
    backward_clip_ms = []
    optimizer_ms = []
    total_ms = []
    first_loss = math.nan
    final_loss = math.nan
    final_gradient_norm = math.nan
    first_timed_lr = math.nan
    final_timed_lr = math.nan
    for iteration in range(WARMUP + REPEATS):
        learning_rate = base.lr_at(iteration, SCHEDULE_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        begin = torch.cuda.Event(enable_timing=True)
        after_forward = torch.cuda.Event(enable_timing=True)
        after_backward = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        loss = objective(model, window)
        after_forward.record()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            base.CLIP_NORM,
        )
        after_backward.record()
        optimizer.step()
        end.record()
        end.synchronize()
        loss_value = float(loss.detach())
        gradient_value = float(gradient_norm)
        if not math.isfinite(loss_value) or not math.isfinite(gradient_value):
            raise RuntimeError("non-finite width-4400 timing step")
        if iteration == WARMUP:
            first_loss = loss_value
            first_timed_lr = learning_rate
            torch.cuda.reset_peak_memory_stats()
        if iteration >= WARMUP:
            forward_ms.append(begin.elapsed_time(after_forward))
            backward_clip_ms.append(
                after_forward.elapsed_time(after_backward)
            )
            optimizer_ms.append(after_backward.elapsed_time(end))
            total_ms.append(begin.elapsed_time(end))
            final_loss = loss_value
            final_gradient_norm = gradient_value
            final_timed_lr = learning_rate

    median_total_ms = statistics.median(total_ms)
    supervised_targets_per_step = (
        BATCH
        * len(range(0, base.CONTEXT, base.TRAIN_ANCHOR_STRIDE))
        * base.TRAIN_HORIZONS
    )
    sampled_bytes_per_step = BATCH * base.TRAIN_WINDOW_LENGTH
    supervised_steps_10epoch = (
        10.0 * TRAIN_TOKENS / supervised_targets_per_step
    )
    sampled_steps_10epoch = 10.0 * TRAIN_TOKENS / sampled_bytes_per_step
    row = {
        "width": WIDTH,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "batch": BATCH,
        "context": base.CONTEXT,
        "horizons": base.TRAIN_HORIZONS,
        "anchor_stride": base.TRAIN_ANCHOR_STRIDE,
        "warmup_steps": WARMUP,
        "timed_steps": REPEATS,
        "first_timed_loss": first_loss,
        "final_timed_loss": final_loss,
        "final_gradient_norm": final_gradient_norm,
        "schedule_steps": SCHEDULE_STEPS,
        "first_timed_lr": first_timed_lr,
        "final_timed_lr": final_timed_lr,
        "mean_forward_ms": statistics.mean(forward_ms),
        "median_forward_ms": statistics.median(forward_ms),
        "mean_backward_clip_ms": statistics.mean(backward_clip_ms),
        "median_backward_clip_ms": statistics.median(backward_clip_ms),
        "mean_optimizer_ms": statistics.mean(optimizer_ms),
        "median_optimizer_ms": statistics.median(optimizer_ms),
        "mean_total_ms": statistics.mean(total_ms),
        "median_total_ms": median_total_ms,
        "min_total_ms": min(total_ms),
        "max_total_ms": max(total_ms),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "sampled_bytes_per_step": sampled_bytes_per_step,
        "supervised_targets_per_step": supervised_targets_per_step,
        "sampled_bytes_per_s": sampled_bytes_per_step * 1000.0
        / median_total_ms,
        "supervised_targets_per_s": supervised_targets_per_step * 1000.0
        / median_total_ms,
        "sampled_definition_10epoch_steps": sampled_steps_10epoch,
        "sampled_definition_10epoch_hours": (
            sampled_steps_10epoch * median_total_ms / 3_600_000.0
        ),
        "supervised_definition_10epoch_steps": supervised_steps_10epoch,
        "supervised_definition_10epoch_hours": (
            supervised_steps_10epoch * median_total_ms / 3_600_000.0
        ),
    }
    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "metrics_warmup_schedule.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)


if __name__ == "__main__":
    main()
