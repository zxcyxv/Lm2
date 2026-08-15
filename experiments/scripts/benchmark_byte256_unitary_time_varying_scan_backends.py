"""Benchmark exact implementations of the H16 time-varying recurrence."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics

import torch

from rotlm.training.ha_skew_window import (
    sparse_complex_self_predicted_kv_ce_only_fast_logits,
    sparse_complex_self_predicted_kv_ce_only_fast_loss,
)
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-"
    "rtx5090/step1000.pt"
)
RECORD = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-time-varying-scan-fusion-benchmark"
)
BATCH = 64
WINDOW_SEED = base.SEED + 8316
WARMUP = 5
REPEATS = 15


def configure(model, backend: str) -> str:
    transition = model.complex_self_prediction
    transition.fuse_time_varying_memory_scan = False
    transition.fuse_time_varying_hidden_scan = False
    transition.use_triton_time_varying_scan = False
    transition.use_rotating_frame_time_varying_scan = False
    transition.use_fused_triton_rotating_frame_time_varying_scan = False
    if backend == "feedback":
        return "feedback"
    if backend == "eager-hillis-steele":
        return "time-varying-scan"
    if backend == "generated-associative":
        transition.fuse_time_varying_memory_scan = True
        transition.fuse_time_varying_hidden_scan = True
        return "time-varying-scan"
    if backend == "triton-associative":
        transition.use_triton_time_varying_scan = True
        return "time-varying-scan"
    if backend == "rotating-frame-cumsum":
        transition.use_rotating_frame_time_varying_scan = True
        return "time-varying-scan"
    if backend == "fused-triton-rotating-frame":
        transition.use_fused_triton_rotating_frame_time_varying_scan = True
        return "time-varying-scan"
    raise ValueError(f"unknown backend: {backend}")


def logits(model, window, central_rollout: str):
    return sparse_complex_self_predicted_kv_ce_only_fast_logits(
        model,
        window,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        central_rollout=central_rollout,
    )[0]


def loss(model, window, central_rollout: str):
    return sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        central_rollout=central_rollout,
    )


def loaded_model(checkpoint):
    model = base.make_model().cuda().train()
    model.load_state_dict(checkpoint["model"])
    return model


def compare_gradients(checkpoint, window, candidate_backend: str):
    model = loaded_model(checkpoint)
    central_rollout = configure(model, "eager-hillis-steele")
    model.zero_grad(set_to_none=True)
    eager_loss = loss(model, window, central_rollout)
    eager_loss.backward()
    eager_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    central_rollout = configure(model, candidate_backend)
    model.zero_grad(set_to_none=True)
    rotating_loss = loss(model, window, central_rollout)
    rotating_loss.backward()

    dot = torch.zeros((), device="cuda", dtype=torch.float64)
    eager_square = torch.zeros_like(dot)
    rotating_square = torch.zeros_like(dot)
    difference_square = torch.zeros_like(dot)
    max_abs_difference = torch.zeros((), device="cuda")
    for name, parameter in model.named_parameters():
        if name not in eager_gradients:
            continue
        rotating_gradient = parameter.grad
        if rotating_gradient is None:
            raise RuntimeError(f"rotating backend lost gradient: {name}")
        eager_gradient = eager_gradients[name]
        difference = rotating_gradient - eager_gradient
        dot += (rotating_gradient * eager_gradient).sum(dtype=torch.float64)
        eager_square += eager_gradient.square().sum(dtype=torch.float64)
        rotating_square += rotating_gradient.square().sum(dtype=torch.float64)
        difference_square += difference.square().sum(dtype=torch.float64)
        max_abs_difference = torch.maximum(
            max_abs_difference,
            difference.abs().max(),
        )
    cosine = dot / (eager_square.sqrt() * rotating_square.sqrt())
    relative_l2 = (difference_square / eager_square).sqrt()
    row = {
        "comparison": f"{candidate_backend}_vs_eager-hillis-steele",
        "eager_loss": float(eager_loss.detach()),
        "candidate_loss": float(rotating_loss.detach()),
        "loss_abs_difference": abs(
            float(eager_loss.detach()) - float(rotating_loss.detach())
        ),
        "gradient_cosine": float(cosine),
        "gradient_relative_l2": float(relative_l2),
        "gradient_max_abs_difference": float(max_abs_difference),
    }
    del model, eager_gradients, eager_loss, rotating_loss
    torch.cuda.empty_cache()
    return row


def measure(backend: str, checkpoint, window):
    model = loaded_model(checkpoint)
    central_rollout = configure(model, backend)
    forward_ms = []
    backward_ms = []
    total_ms = []
    objective_value = math.nan
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(WARMUP + REPEATS):
        model.zero_grad(set_to_none=True)
        begin = torch.cuda.Event(enable_timing=True)
        middle = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        objective = loss(model, window, central_rollout)
        middle.record()
        objective.backward()
        end.record()
        end.synchronize()
        objective_value = float(objective.detach())
        if iteration >= WARMUP:
            forward_ms.append(begin.elapsed_time(middle))
            backward_ms.append(middle.elapsed_time(end))
            total_ms.append(begin.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated()
    row = {
        "backend": backend,
        "central_rollout": central_rollout,
        "batch": BATCH,
        "horizons": base.TRAIN_HORIZONS,
        "anchor_stride": base.TRAIN_ANCHOR_STRIDE,
        "anchors_per_sample": len(
            range(0, base.CONTEXT, base.TRAIN_ANCHOR_STRIDE)
        ),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "loss": objective_value,
        "mean_forward_ms": statistics.mean(forward_ms),
        "median_forward_ms": statistics.median(forward_ms),
        "mean_backward_ms": statistics.mean(backward_ms),
        "median_backward_ms": statistics.median(backward_ms),
        "mean_total_ms": statistics.mean(total_ms),
        "median_total_ms": statistics.median(total_ms),
        "min_total_ms": min(total_ms),
        "max_total_ms": max(total_ms),
        "peak_allocated_bytes": peak,
        "nominal_sampled_bytes_per_s": (
            BATCH * base.TRAIN_WINDOW_LENGTH * 1000.0
            / statistics.median(total_ms)
        ),
    }
    del model, objective
    torch.cuda.empty_cache()
    return row


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    training = base.memmap("train")
    window = base.sampled_windows(
        training,
        BATCH,
        torch.Generator().manual_seed(WINDOW_SEED),
        base.TRAIN_WINDOW_LENGTH,
    )

    correctness_rows = []
    for candidate_backend in (
        "rotating-frame-cumsum",
        "fused-triton-rotating-frame",
    ):
        reference_model = loaded_model(checkpoint)
        reference_rollout = configure(
            reference_model,
            "eager-hillis-steele",
        )
        with torch.inference_mode():
            reference_logits = logits(
                reference_model,
                window,
                reference_rollout,
            )
            candidate_rollout = configure(
                reference_model,
                candidate_backend,
            )
            candidate_logits = logits(
                reference_model,
                window,
                candidate_rollout,
            )
            logit_max_abs_difference = float(
                (candidate_logits - reference_logits).abs().max()
            )
            logit_relative_l2 = float(
                (candidate_logits - reference_logits).float().norm()
                / reference_logits.float().norm()
            )
        del reference_model, reference_logits, candidate_logits
        torch.cuda.empty_cache()
        correctness = compare_gradients(
            checkpoint,
            window,
            candidate_backend,
        )
        correctness["logit_max_abs_difference"] = (
            logit_max_abs_difference
        )
        correctness["logit_relative_l2"] = logit_relative_l2
        correctness_rows.append(correctness)

    backends = (
        "feedback",
        "eager-hillis-steele",
        "generated-associative",
        "triton-associative",
        "rotating-frame-cumsum",
        "fused-triton-rotating-frame",
    )
    rows = [measure(name, checkpoint, window) for name in backends]
    eager_time = next(
        row["median_total_ms"]
        for row in rows
        if row["backend"] == "eager-hillis-steele"
    )
    feedback_time = next(
        row["median_total_ms"]
        for row in rows
        if row["backend"] == "feedback"
    )
    for row in rows:
        row["speedup_vs_eager_scan"] = eager_time / row["median_total_ms"]
        row["speedup_vs_feedback"] = feedback_time / row["median_total_ms"]

    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (RECORD / "correctness.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(correctness_rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(correctness_rows)
    for row in rows:
        print(
            f"{row['backend']}: total={row['median_total_ms']:.3f}ms "
            f"fwd={row['median_forward_ms']:.3f}ms "
            f"bwd={row['median_backward_ms']:.3f}ms "
            f"vs_eager={row['speedup_vs_eager_scan']:.3f}x",
            flush=True,
        )
    for correctness in correctness_rows:
        print(correctness, flush=True)


if __name__ == "__main__":
    main()
