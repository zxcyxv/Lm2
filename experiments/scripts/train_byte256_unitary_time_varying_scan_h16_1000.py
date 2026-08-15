"""Train only the exact time-varying scan central recurrence on one RTX 5090."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import torch

from rotlm.training.ha_skew_window import (
    evaluate_sparse_complex_self_predicted_kv_ce_only,
    sparse_complex_self_predicted_kv_ce_only_fast_logits,
    sparse_complex_self_predicted_kv_ce_only_fast_loss,
)
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
EXPERIMENT_ID = (
    "EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
BASELINE_STARTS = Path(
    "experiments/records/"
    "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-"
    "normalized-residual-h16-ce-only-attached-stride16-1000step-13m/"
    "validation_starts.tsv"
)
REPORT_STEPS = frozenset((100, 300, 1000))


def configure_scan_backend(model, backend: str) -> None:
    transition = model.complex_self_prediction
    transition.fuse_time_varying_memory_scan = False
    transition.fuse_time_varying_hidden_scan = False
    transition.use_triton_time_varying_scan = False
    transition.use_rotating_frame_time_varying_scan = False
    transition.use_fused_triton_rotating_frame_time_varying_scan = False
    if backend == "eager":
        return
    if backend == "generated-associative":
        transition.fuse_time_varying_memory_scan = True
        transition.fuse_time_varying_hidden_scan = True
        return
    if backend == "triton-associative":
        transition.use_triton_time_varying_scan = True
        return
    if backend == "rotating-frame-cumsum":
        transition.use_rotating_frame_time_varying_scan = True
        return
    if backend == "fused-triton-rotating-frame":
        transition.use_fused_triton_rotating_frame_time_varying_scan = True
        return
    raise ValueError(f"unknown scan backend: {backend}")


def scan_loss(model, window):
    return sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        central_rollout="time-varying-scan",
    )


def scan_metrics(model, validation_windows, microbatch):
    return evaluate_sparse_complex_self_predicted_kv_ce_only(
        model,
        validation_windows,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        anchor_stride=base.EVAL_ANCHOR_STRIDE,
        central_rollout="time-varying-scan",
        microbatch=microbatch,
    )


def h1_feedback_equivalence(model, training) -> float:
    window = base.sampled_windows(
        training,
        2,
        torch.Generator().manual_seed(base.SEED + 8303),
        base.CONTEXT + 1,
    )
    with torch.inference_mode():
        feedback, feedback_targets = (
            sparse_complex_self_predicted_kv_ce_only_fast_logits(
                model,
                window,
                prefix_length=base.CONTEXT,
                horizons=1,
                anchor_stride=base.TRAIN_ANCHOR_STRIDE,
                central_rollout="feedback",
            )
        )
        scanned, scanned_targets = (
            sparse_complex_self_predicted_kv_ce_only_fast_logits(
                model,
                window,
                prefix_length=base.CONTEXT,
                horizons=1,
                anchor_stride=base.TRAIN_ANCHOR_STRIDE,
                central_rollout="time-varying-scan",
            )
        )
    if not torch.equal(feedback_targets, scanned_targets):
        raise RuntimeError("H1 scan and feedback targets differ")
    return float((feedback.float() - scanned.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--microbatch",
        type=int,
        default=None,
        help=(
            "physical batch per forward/backward; gradients accumulate "
            "over batch/microbatch steps before one optimizer.step(). "
            "Defaults to --batch (no accumulation)."
        ),
    )
    parser.add_argument("--schedule-steps", type=int, default=6000)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument(
        "--scan-backend",
        choices=(
            "eager",
            "generated-associative",
            "triton-associative",
            "rotating-frame-cumsum",
            "fused-triton-rotating-frame",
        ),
        default="eager",
    )
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch,
        args.schedule_steps,
        args.eval_examples,
        args.eval_microbatch,
    ) < 1:
        parser.error("all counts must be positive")
    if args.report_every < 0 or args.checkpoint_every < 0:
        parser.error("report/checkpoint intervals must be non-negative")
    if args.microbatch is None:
        args.microbatch = args.batch
    if args.microbatch < 1 or args.batch % args.microbatch:
        parser.error("--microbatch must be positive and divide --batch")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    training = base.memmap("train")
    validation = base.memmap("validation")
    validation_starts = base.load_fixed_starts(
        BASELINE_STARTS,
        args.eval_examples,
    )
    validation_windows = base.windows_of_length(
        validation,
        validation_starts,
        base.EVAL_WINDOW_LENGTH,
    )
    model = base.make_model().cuda().train()
    configure_scan_backend(model, args.scan_backend)
    h1_error = h1_feedback_equivalence(model, training)
    if h1_error > 5e-5:
        raise RuntimeError(f"H1 scan equivalence failed: {h1_error}")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=base.PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    data_generator = torch.Generator().manual_seed(base.SEED)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for item in (
            ("experiment_id", args.experiment_id),
            ("seed", base.SEED),
            ("split", "wikitext103_bytes_train_validation"),
            ("steps", args.steps),
            ("schedule_steps", args.schedule_steps),
            ("effective_batch", args.batch),
            ("microbatch", args.microbatch),
            ("horizons", base.TRAIN_HORIZONS),
            ("anchor_stride", base.TRAIN_ANCHOR_STRIDE),
            ("central_rollout", "time-varying-scan"),
            ("scan_backend", args.scan_backend),
            ("report_every", args.report_every),
            ("checkpoint_every", args.checkpoint_every),
            ("precision", "strict_float32_no_tf32"),
            ("gpu", torch.cuda.get_device_name()),
            ("torch", torch.__version__),
            ("h1_feedback_max_logit_error", h1_error),
        ):
            writer.writerow(item)

    metric_names = [
        "step",
        "train_ce",
        "gradient_norm",
        "lr",
        "block_nll",
        "block_accuracy",
    ]
    for horizon in range(1, base.TRAIN_HORIZONS + 1):
        metric_names.extend((f"h{horizon}_nll", f"h{horizon}_accuracy"))
    metric_names.extend(
        (
            "training_wall_s",
            "optimizer_steps_per_s",
            "nominal_sampled_bytes_per_s",
            "peak_vram_bytes",
        )
    )

    initial_metrics = scan_metrics(
        model,
        validation_windows,
        args.eval_microbatch,
    )
    initial_row = {
        "step": 0,
        "train_ce": math.nan,
        "gradient_norm": math.nan,
        "lr": 0.0,
        **initial_metrics,
        "training_wall_s": 0.0,
        "optimizer_steps_per_s": 0.0,
        "nominal_sampled_bytes_per_s": 0.0,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    print(
        f"scan step=0 block_nll={initial_metrics['block_nll']:.4f} "
        f"h1={initial_metrics['h1_nll']:.4f} "
        f"h16={initial_metrics['h16_nll']:.4f} "
        f"h1_error={h1_error:.3e}",
        flush=True,
    )

    report_steps = set(step for step in REPORT_STEPS if step <= args.steps)
    if args.report_every:
        report_steps.update(
            range(args.report_every, args.steps + 1, args.report_every)
        )
    if args.checkpoint_every:
        report_steps.update(
            range(
                args.checkpoint_every,
                args.steps + 1,
                args.checkpoint_every,
            )
        )
    report_steps.add(args.steps)
    metrics_path = args.record_dir / "metrics.tsv"
    training_wall = 0.0
    last_train_ce = math.nan
    last_gradient_norm = math.nan
    torch.cuda.reset_peak_memory_stats()
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metric_names,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(initial_row)
        handle.flush()

        torch.cuda.synchronize()
        segment_started = time.perf_counter()
        for index in range(args.steps):
            lr = base.lr_at(index, args.schedule_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = base.sampled_windows(
                training,
                args.batch,
                data_generator,
                base.TRAIN_WINDOW_LENGTH,
            )
            optimizer.zero_grad(set_to_none=True)
            micro_windows = window.split(args.microbatch)
            accumulated_ce = 0.0
            for micro_window in micro_windows:
                micro_loss = scan_loss(model, micro_window)
                if not bool(torch.isfinite(micro_loss)):
                    raise RuntimeError(
                        f"non-finite scan loss at step {index + 1}"
                    )
                (micro_loss / len(micro_windows)).backward()
                accumulated_ce += float(micro_loss.detach()) / len(micro_windows)
            last_train_ce = accumulated_ce
            last_gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP_NORM)
            )
            if not math.isfinite(last_gradient_norm):
                raise RuntimeError(
                    f"non-finite scan gradient at step {index + 1}"
                )
            optimizer.step()
            step = index + 1
            if step in report_steps:
                torch.cuda.synchronize()
                training_wall += time.perf_counter() - segment_started
                metrics = scan_metrics(
                    model,
                    validation_windows,
                    args.eval_microbatch,
                )
                row = {
                    "step": step,
                    "train_ce": last_train_ce,
                    "gradient_norm": last_gradient_norm,
                    "lr": lr,
                    **metrics,
                    "training_wall_s": training_wall,
                    "optimizer_steps_per_s": step / training_wall,
                    "nominal_sampled_bytes_per_s": (
                        step * args.batch * base.TRAIN_WINDOW_LENGTH
                        / training_wall
                    ),
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                eta_s = (training_wall / step) * (args.steps - step)
                print(
                    f"scan step={step}/{args.steps} ce={last_train_ce:.4f} "
                    f"gnorm={last_gradient_norm:.4f} "
                    f"block_nll={metrics['block_nll']:.4f} "
                    f"h1={metrics['h1_nll']:.4f} "
                    f"h16={metrics['h16_nll']:.4f} "
                    f"step_s={training_wall / step:.4f} "
                    f"elapsed={training_wall/60:.1f}m eta={eta_s/60:.1f}m",
                    flush=True,
                )
                if (
                    args.checkpoint_every
                    and step % args.checkpoint_every == 0
                ):
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "data_generator_state": (
                                data_generator.get_state()
                            ),
                            "step": step,
                            "config": {
                                "experiment_id": args.experiment_id,
                                "central_rollout": "time-varying-scan",
                                "scan_backend": args.scan_backend,
                                "batch": args.batch,
                                "schedule_steps": args.schedule_steps,
                                "seed": base.SEED,
                            },
                        },
                        args.output_dir / f"step{step}.pt",
                    )
                torch.cuda.synchronize()
                segment_started = time.perf_counter()

    final_checkpoint = args.output_dir / f"step{args.steps}.pt"
    if not final_checkpoint.exists():
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_generator_state": data_generator.get_state(),
                "step": args.steps,
                "config": {
                    "experiment_id": args.experiment_id,
                    "central_rollout": "time-varying-scan",
                    "scan_backend": args.scan_backend,
                    "batch": args.batch,
                    "schedule_steps": args.schedule_steps,
                    "seed": base.SEED,
                },
            },
            final_checkpoint,
        )


if __name__ == "__main__":
    main()
