"""Train the 13M fused scan with conditional latent-ray flow matching."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import time

import torch

from rotlm.models.latent_flow_matching import (
    ConditionalRayFlowTimeConditioner,
)
from rotlm.training.conditional_flow_matching import (
    evaluate_sparse_conditional_ray_flow,
    sparse_scan_conditional_ray_flow_loss,
)
import train_byte256_unitary_time_varying_scan_h16_1000 as scan


base = scan.base
EXPERIMENT_ID = (
    "EXP-20260804-byte256-unitary-conditional-ray-cfm-h16-10epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
DEFAULT_STEPS = 33_570
DEFAULT_EPOCH_STEPS = 3_357
FLOW_NOISE_SEED = 25_617
FLOW_NOISE_SCALE = 0.8
FLOW_WEIGHT = 0.1
FLOW_ANCHOR_STRIDE = 64
FLOW_INTEGRATION_STEPS = 8


def train_loss(
    model,
    time_conditioner,
    window,
    *,
    flow_generator,
    flow_anchor_stride: int,
    flow_noise_scale: float,
    flow_weight: float,
):
    return sparse_scan_conditional_ray_flow_loss(
        model,
        time_conditioner,
        window,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        ce_anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        flow_anchor_stride=flow_anchor_stride,
        noise_scale=flow_noise_scale,
        flow_weight=flow_weight,
        generator=flow_generator,
    )


def checkpoint_payload(
    *,
    model,
    time_conditioner,
    optimizer,
    data_generator,
    flow_generator,
    step: int,
    args,
) -> dict:
    return {
        "model": model.state_dict(),
        "flow_time_conditioner": time_conditioner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_generator_state": data_generator.get_state(),
        "flow_generator_state": flow_generator.get_state(),
        "step": step,
        "config": {
            "experiment_id": args.experiment_id,
            "central_rollout": "time-varying-scan",
            "scan_backend": args.scan_backend,
            "batch": args.batch,
            "schedule_steps": args.schedule_steps,
            "seed": base.SEED,
            "flow_noise_seed": FLOW_NOISE_SEED,
            "flow_noise_scale": args.flow_noise_scale,
            "flow_weight": args.flow_weight,
            "flow_anchor_stride": args.flow_anchor_stride,
            "flow_integration_steps": args.flow_integration_steps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--schedule-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=16)
    parser.add_argument("--flow-eval-examples", type=int, default=16)
    parser.add_argument("--flow-eval-microbatch", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=DEFAULT_EPOCH_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_EPOCH_STEPS)
    parser.add_argument("--flow-noise-scale", type=float, default=FLOW_NOISE_SCALE)
    parser.add_argument("--flow-weight", type=float, default=FLOW_WEIGHT)
    parser.add_argument(
        "--flow-anchor-stride", type=int, default=FLOW_ANCHOR_STRIDE
    )
    parser.add_argument(
        "--flow-integration-steps",
        type=int,
        default=FLOW_INTEGRATION_STEPS,
    )
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
        default="fused-triton-rotating-frame",
    )
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    counts = (
        args.steps,
        args.batch,
        args.schedule_steps,
        args.eval_examples,
        args.eval_microbatch,
        args.flow_eval_examples,
        args.flow_eval_microbatch,
        args.flow_anchor_stride,
        args.flow_integration_steps,
    )
    if min(counts) < 1:
        parser.error("all count arguments must be positive")
    if args.report_every < 0 or args.checkpoint_every < 0:
        parser.error("report/checkpoint intervals must be non-negative")
    if args.flow_eval_examples > args.eval_examples:
        parser.error("flow eval examples cannot exceed scan eval examples")
    if args.flow_noise_scale < 0 or args.flow_weight < 0:
        parser.error("flow scale and weight must be non-negative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    training = base.memmap("train")
    validation = base.memmap("validation")
    registered_validation_starts = base.load_fixed_starts(
        scan.BASELINE_STARTS,
        64,
    )
    if args.eval_examples > len(registered_validation_starts):
        parser.error("at most 64 registered validation examples are available")
    validation_starts = registered_validation_starts[: args.eval_examples]
    validation_windows = base.windows_of_length(
        validation,
        validation_starts,
        base.EVAL_WINDOW_LENGTH,
    )
    flow_validation_windows = validation_windows[: args.flow_eval_examples]

    model = base.make_model().cuda().train()
    scan.configure_scan_backend(model, args.scan_backend)
    time_conditioner = ConditionalRayFlowTimeConditioner(
        model.width
    ).cuda().train()
    trainable_parameters = tuple(
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ) + tuple(time_conditioner.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_count += sum(
        parameter.numel() for parameter in time_conditioner.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    h1_error = scan.h1_feedback_equivalence(model, training)
    if h1_error > 5e-5:
        raise RuntimeError(f"H1 scan equivalence failed: {h1_error}")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=base.PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    data_generator = torch.Generator().manual_seed(base.SEED)
    flow_generator = torch.Generator(device="cuda").manual_seed(
        FLOW_NOISE_SEED
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for item in (
            ("experiment_id", args.experiment_id),
            ("seed", base.SEED),
            ("flow_noise_seed", FLOW_NOISE_SEED),
            ("split", "wikitext103_bytes_train_validation"),
            ("steps", args.steps),
            ("schedule_steps", args.schedule_steps),
            ("effective_batch", args.batch),
            ("microbatch", args.batch),
            ("parameter_count", parameter_count),
            ("trainable_parameter_count", trainable_parameter_count),
            ("horizons", base.TRAIN_HORIZONS),
            ("ce_anchor_stride", base.TRAIN_ANCHOR_STRIDE),
            ("flow_anchor_stride", args.flow_anchor_stride),
            ("flow_noise_scale", args.flow_noise_scale),
            ("flow_weight", args.flow_weight),
            ("flow_integration_steps", args.flow_integration_steps),
            ("flow_geometry", "shortest_sphere_slerp"),
            ("flow_source", "independent_equal_rms_tangent_gaussian"),
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
        "train_total",
        "train_ce",
        "train_cfm_mse",
        "train_cfm_relative_mse",
        "train_cfm_cosine",
        "train_cfm_target_energy",
        "gradient_norm",
        "lr",
        "block_nll",
        "block_accuracy",
    ]
    for horizon in range(1, base.TRAIN_HORIZONS + 1):
        metric_names.extend((f"h{horizon}_nll", f"h{horizon}_accuracy"))
    metric_names.extend(
        (
            "flow_endpoint_nll",
            "flow_endpoint_accuracy",
            "flow_endpoint_cosine",
            "fixed_cfm_mse",
            "fixed_cfm_relative_mse",
            "fixed_cfm_cosine",
            "fixed_cfm_target_energy",
        )
    )
    for horizon in range(1, base.TRAIN_HORIZONS + 1):
        metric_names.extend(
            (
                f"flow_h{horizon}_nll",
                f"flow_h{horizon}_accuracy",
                f"flow_h{horizon}_endpoint_cosine",
            )
        )
    metric_names.extend(
        (
            "training_wall_s",
            "optimizer_steps_per_s",
            "nominal_sampled_bytes_per_s",
            "peak_vram_bytes",
        )
    )

    initial_scan = scan.scan_metrics(
        model,
        validation_windows,
        args.eval_microbatch,
    )
    initial_flow = evaluate_sparse_conditional_ray_flow(
        model,
        time_conditioner,
        flow_validation_windows,
        prefix_length=base.CONTEXT,
        horizons=base.TRAIN_HORIZONS,
        anchor_stride=args.flow_anchor_stride,
        noise_scale=args.flow_noise_scale,
        integration_steps=args.flow_integration_steps,
        noise_seed=FLOW_NOISE_SEED + 1,
        microbatch=args.flow_eval_microbatch,
    )
    initial_row = {
        "step": 0,
        "train_total": math.nan,
        "train_ce": math.nan,
        "train_cfm_mse": math.nan,
        "train_cfm_relative_mse": math.nan,
        "train_cfm_cosine": math.nan,
        "train_cfm_target_energy": math.nan,
        "gradient_norm": math.nan,
        "lr": 0.0,
        **initial_scan,
        **initial_flow,
        "training_wall_s": 0.0,
        "optimizer_steps_per_s": 0.0,
        "nominal_sampled_bytes_per_s": 0.0,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    print(
        f"cfm step=0 scan_nll={initial_scan['block_nll']:.4f} "
        f"flow_nll={initial_flow['flow_endpoint_nll']:.4f} "
        f"flow_cos={initial_flow['fixed_cfm_cosine']:.4f} "
        f"params={parameter_count}",
        flush=True,
    )

    report_steps = {step for step in (100, 300, 1000) if step <= args.steps}
    if args.report_every:
        report_steps.update(
            range(args.report_every, args.steps + 1, args.report_every)
        )
    if args.checkpoint_every:
        report_steps.update(
            range(args.checkpoint_every, args.steps + 1, args.checkpoint_every)
        )
    report_steps.add(args.steps)

    metrics_path = args.record_dir / "metrics.tsv"
    training_wall = 0.0
    last_values = {
        "train_total": math.nan,
        "train_ce": math.nan,
        "train_cfm_mse": math.nan,
        "train_cfm_relative_mse": math.nan,
        "train_cfm_cosine": math.nan,
        "train_cfm_target_energy": math.nan,
        "gradient_norm": math.nan,
    }
    torch.cuda.reset_peak_memory_stats()
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_names, delimiter="\t")
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
            total, token_nll, cfm_loss, output = train_loss(
                model,
                time_conditioner,
                window,
                flow_generator=flow_generator,
                flow_anchor_stride=args.flow_anchor_stride,
                flow_noise_scale=args.flow_noise_scale,
                flow_weight=args.flow_weight,
            )
            if not bool(torch.isfinite(total)):
                raise RuntimeError(f"non-finite total loss at step {index + 1}")
            total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_parameters, base.CLIP_NORM)
            )
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient at step {index + 1}")
            optimizer.step()
            last_values = {
                "train_total": float(total.detach()),
                "train_ce": float(token_nll.detach()),
                "train_cfm_mse": float(cfm_loss.detach()),
                "train_cfm_relative_mse": float(
                    output.flow.relative_loss_by_horizon.mean().detach()
                ),
                "train_cfm_cosine": float(
                    output.flow.cosine_by_horizon.mean().detach()
                ),
                "train_cfm_target_energy": float(
                    output.flow.target_energy_by_horizon.mean().detach()
                ),
                "gradient_norm": gradient_norm,
            }
            step = index + 1
            if step in report_steps:
                torch.cuda.synchronize()
                training_wall += time.perf_counter() - segment_started
                scan_metrics = scan.scan_metrics(
                    model,
                    validation_windows,
                    args.eval_microbatch,
                )
                flow_metrics = evaluate_sparse_conditional_ray_flow(
                    model,
                    time_conditioner,
                    flow_validation_windows,
                    prefix_length=base.CONTEXT,
                    horizons=base.TRAIN_HORIZONS,
                    anchor_stride=args.flow_anchor_stride,
                    noise_scale=args.flow_noise_scale,
                    integration_steps=args.flow_integration_steps,
                    noise_seed=FLOW_NOISE_SEED + 1,
                    microbatch=args.flow_eval_microbatch,
                )
                row = {
                    "step": step,
                    **last_values,
                    "lr": lr,
                    **scan_metrics,
                    **flow_metrics,
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
                print(
                    f"cfm step={step} total={last_values['train_total']:.4f} "
                    f"ce={last_values['train_ce']:.4f} "
                    f"fm={last_values['train_cfm_mse']:.4f} "
                    f"scan_nll={scan_metrics['block_nll']:.4f} "
                    f"flow_nll={flow_metrics['flow_endpoint_nll']:.4f} "
                    f"flow_cos={flow_metrics['fixed_cfm_cosine']:.4f} "
                    f"step_s={training_wall / step:.4f}",
                    flush=True,
                )
                if args.checkpoint_every and step % args.checkpoint_every == 0:
                    torch.save(
                        checkpoint_payload(
                            model=model,
                            time_conditioner=time_conditioner,
                            optimizer=optimizer,
                            data_generator=data_generator,
                            flow_generator=flow_generator,
                            step=step,
                            args=args,
                        ),
                        args.output_dir / f"step{step}.pt",
                    )
                torch.cuda.synchronize()
                segment_started = time.perf_counter()

    final_checkpoint = args.output_dir / f"step{args.steps}.pt"
    if not final_checkpoint.exists():
        torch.save(
            checkpoint_payload(
                model=model,
                time_conditioner=time_conditioner,
                optimizer=optimizer,
                data_generator=data_generator,
                flow_generator=flow_generator,
                step=args.steps,
                args=args,
            ),
            final_checkpoint,
        )


if __name__ == "__main__":
    main()
