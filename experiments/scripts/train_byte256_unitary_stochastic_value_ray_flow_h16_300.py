"""Matched 300-step process-noise and shallow ray-velocity continuation."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time

import torch

from rotlm.training.ha_skew_window import (
    complex_standard_normal,
    sparse_complex_stochastic_ray_flow_loss,
)
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run
from train_byte256_unitary_time_varying_scan_h16_1000 import (
    configure_scan_backend,
)


base = run.base
EXPERIMENT_ID = (
    "EXP-20260803-byte256-unitary-stochastic-value-ray-flow-h16-300step"
)
RECORD = Path("experiments/records") / EXPERIMENT_ID
OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-"
    "13m-rtx5090/step33570.pt"
)
VALIDATION_STARTS = Path(
    "experiments/records/"
    "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-"
    "normalized-residual-h16-ce-only-attached-stride16-1000step-13m/"
    "validation_starts.tsv"
)
STEPS = 300
REPORT_STEPS = frozenset((0, 50, 100, 300))
LEARNING_RATE = 3e-5
DIFFUSION_RATE = 0.2
DELTA_TAU = 1.0 / base.TRAIN_HORIZONS
VALUE_NOISE_SCALE = DIFFUSION_RATE * math.sqrt(DELTA_TAU)
RAY_FLOW_WEIGHT = 0.01
RAY_FLOW_HORIZONS = 4
DATA_SEED = base.SEED
NOISE_SEED = base.SEED + 24280
EVAL_NOISE_SEED = base.SEED + 24281


@dataclass(frozen=True)
class Arm:
    name: str
    value_noise_scale: float
    ray_flow_weight: float


ARMS = (
    Arm("deterministic_ce", 0.0, 0.0),
    Arm("stochastic_ce", VALUE_NOISE_SCALE, 0.0),
    Arm("stochastic_ray_flow", VALUE_NOISE_SCALE, RAY_FLOW_WEIGHT),
)


def load_model() -> torch.nn.Module:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = base.make_model()
    model.load_state_dict(payload["model"])
    configure_scan_backend(model, "fused-triton-rotating-frame")
    return model.cuda()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    windows: torch.Tensor,
    *,
    value_noise_scale: float,
    microbatch: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    horizons = base.TRAIN_HORIZONS
    deterministic_nll = torch.zeros(horizons, dtype=torch.float64)
    stochastic_nll = torch.zeros(horizons, dtype=torch.float64)
    deterministic_correct = torch.zeros(horizons, dtype=torch.float64)
    stochastic_correct = torch.zeros(horizons, dtype=torch.float64)
    ray_loss = torch.zeros(horizons, dtype=torch.float64)
    ray_cosine = torch.zeros(horizons, dtype=torch.float64)
    labels_per_horizon = 0
    noise_logit_squared_delta = 0.0
    noise_logit_count = 0
    noise_top1_disagreement = 0
    noise_token_count = 0
    noise_generator = torch.Generator(device="cuda").manual_seed(
        EVAL_NOISE_SEED
    )
    try:
        for begin in range(0, len(windows), microbatch):
            micro = windows[begin : begin + microbatch]
            anchors = len(
                range(0, base.CONTEXT, base.EVAL_ANCHOR_STRIDE)
            )
            value_noise = complex_standard_normal(
                (
                    micro.shape[0],
                    anchors,
                    horizons,
                    model.complex_self_prediction.heads,
                    model.complex_self_prediction.value_dim,
                ),
                device=micro.device,
                real_dtype=torch.float32,
                generator=noise_generator,
            )
            _, _, _, stochastic = sparse_complex_stochastic_ray_flow_loss(
                model,
                micro,
                prefix_length=base.CONTEXT,
                horizons=horizons,
                anchor_stride=base.EVAL_ANCHOR_STRIDE,
                value_noise_scale=value_noise_scale,
                ray_flow_weight=0.0,
                ray_flow_horizons=RAY_FLOW_HORIZONS,
                value_noise=value_noise,
            )
            _, _, _, deterministic = (
                sparse_complex_stochastic_ray_flow_loss(
                    model,
                    micro,
                    prefix_length=base.CONTEXT,
                    horizons=horizons,
                    anchor_stride=base.EVAL_ANCHOR_STRIDE,
                    value_noise_scale=0.0,
                    ray_flow_weight=0.0,
                    ray_flow_horizons=RAY_FLOW_HORIZONS,
                    value_noise=value_noise,
                )
            )
            if not torch.equal(deterministic.targets, stochastic.targets):
                raise RuntimeError("deterministic and stochastic targets differ")
            deterministic_logits = deterministic.logits
            deterministic_nll += deterministic.token_ce.double().sum(
                dim=(0, 1)
            ).cpu()
            stochastic_nll += stochastic.token_ce.double().sum(
                dim=(0, 1)
            ).cpu()
            deterministic_predictions = deterministic_logits.argmax(dim=-1)
            stochastic_predictions = stochastic.logits.argmax(dim=-1)
            deterministic_correct += deterministic_predictions.eq(
                deterministic.targets
            ).double().sum(dim=(0, 1)).cpu()
            stochastic_correct += stochastic_predictions.eq(
                stochastic.targets
            ).double().sum(dim=(0, 1)).cpu()
            rows = stochastic.targets.shape[0] * stochastic.targets.shape[1]
            ray_loss += (
                stochastic.ray_velocity_loss_by_horizon.double().cpu()
                * rows
            )
            ray_cosine += (
                stochastic.ray_velocity_cosine_by_horizon.double().cpu()
                * rows
            )
            labels_per_horizon += rows
            logit_delta = (
                stochastic.logits.float() - deterministic_logits.float()
            )
            noise_logit_squared_delta += float(logit_delta.square().sum())
            noise_logit_count += logit_delta.numel()
            noise_top1_disagreement += int(
                stochastic_predictions.ne(deterministic_predictions).sum()
            )
            noise_token_count += stochastic_predictions.numel()
    finally:
        model.train(was_training)

    deterministic_nll /= labels_per_horizon
    stochastic_nll /= labels_per_horizon
    deterministic_accuracy = deterministic_correct / labels_per_horizon
    stochastic_accuracy = stochastic_correct / labels_per_horizon
    ray_loss /= labels_per_horizon
    ray_cosine /= labels_per_horizon
    metrics = {
        "deterministic_block_nll": float(deterministic_nll.mean()),
        "stochastic_block_nll": float(stochastic_nll.mean()),
        "deterministic_block_accuracy": float(
            deterministic_accuracy.mean()
        ),
        "stochastic_block_accuracy": float(stochastic_accuracy.mean()),
        "noise_logit_rms": math.sqrt(
            noise_logit_squared_delta / max(noise_logit_count, 1)
        ),
        "noise_top1_disagreement": (
            noise_top1_disagreement / max(noise_token_count, 1)
        ),
        "ray_loss_h1_h4": float(ray_loss[:RAY_FLOW_HORIZONS].mean()),
        "ray_cosine_h1_h4": float(
            ray_cosine[:RAY_FLOW_HORIZONS].mean()
        ),
    }
    for horizon in range(horizons):
        suffix = horizon + 1
        metrics[f"deterministic_h{suffix}_nll"] = float(
            deterministic_nll[horizon]
        )
        metrics[f"stochastic_h{suffix}_nll"] = float(
            stochastic_nll[horizon]
        )
        metrics[f"ray_h{suffix}_loss"] = float(ray_loss[horizon])
        metrics[f"ray_h{suffix}_cosine"] = float(ray_cosine[horizon])
    return metrics


def metric_names() -> list[str]:
    names = [
        "arm",
        "step",
        "value_noise_scale",
        "ray_flow_weight",
        "train_total",
        "train_ce",
        "train_ray_flow",
        "gradient_norm",
        "lr",
        "deterministic_block_nll",
        "stochastic_block_nll",
        "deterministic_block_accuracy",
        "stochastic_block_accuracy",
        "noise_logit_rms",
        "noise_top1_disagreement",
        "ray_loss_h1_h4",
        "ray_cosine_h1_h4",
    ]
    for horizon in range(1, base.TRAIN_HORIZONS + 1):
        names.extend(
            (
                f"deterministic_h{horizon}_nll",
                f"stochastic_h{horizon}_nll",
                f"ray_h{horizon}_loss",
                f"ray_h{horizon}_cosine",
            )
        )
    names.extend(("training_wall_s", "peak_vram_bytes"))
    return names


def run_arm(
    arm: Arm,
    *,
    steps: int,
    batch: int,
    validation_windows: torch.Tensor,
    eval_microbatch: int,
    writer: csv.DictWriter,
    metrics_handle,
) -> None:
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    model = load_model().train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    training = base.memmap("train")
    data_generator = torch.Generator().manual_seed(DATA_SEED)
    noise_generator = torch.Generator(device="cuda").manual_seed(NOISE_SEED)
    torch.cuda.reset_peak_memory_stats()
    training_wall = 0.0
    last_total = math.nan
    last_ce = math.nan
    last_ray_flow = math.nan
    last_gradient_norm = math.nan

    def report(step: int) -> None:
        metrics = evaluate(
            model,
            validation_windows,
            value_noise_scale=arm.value_noise_scale,
            microbatch=eval_microbatch,
        )
        row = {
            "arm": arm.name,
            "step": step,
            "value_noise_scale": arm.value_noise_scale,
            "ray_flow_weight": arm.ray_flow_weight,
            "train_total": last_total,
            "train_ce": last_ce,
            "train_ray_flow": last_ray_flow,
            "gradient_norm": last_gradient_norm,
            "lr": 0.0 if step == 0 else LEARNING_RATE,
            **metrics,
            "training_wall_s": training_wall,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        writer.writerow(row)
        metrics_handle.flush()
        print(
            f"{arm.name} step={step} "
            f"ce={last_ce:.4f} flow={last_ray_flow:.4f} "
            f"gnorm={last_gradient_norm:.4f} "
            f"det_nll={metrics['deterministic_block_nll']:.4f} "
            f"stoch_nll={metrics['stochastic_block_nll']:.4f} "
            f"ray={metrics['ray_loss_h1_h4']:.4f} "
            f"noise_logit={metrics['noise_logit_rms']:.5f}",
            flush=True,
        )

    report(0)
    torch.cuda.synchronize()
    segment_started = time.perf_counter()
    for index in range(steps):
        window = base.sampled_windows(
            training,
            batch,
            data_generator,
            base.TRAIN_WINDOW_LENGTH,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, token_nll, ray_flow, _ = (
            sparse_complex_stochastic_ray_flow_loss(
                model,
                window,
                prefix_length=base.CONTEXT,
                horizons=base.TRAIN_HORIZONS,
                anchor_stride=base.TRAIN_ANCHOR_STRIDE,
                value_noise_scale=arm.value_noise_scale,
                ray_flow_weight=arm.ray_flow_weight,
                ray_flow_horizons=RAY_FLOW_HORIZONS,
                noise_generator=noise_generator,
            )
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite {arm.name} loss at {index + 1}")
        loss.backward()
        last_total = float(loss.detach())
        last_ce = float(token_nll.detach())
        last_ray_flow = float(ray_flow.detach())
        last_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP_NORM)
        )
        if not math.isfinite(last_gradient_norm):
            raise RuntimeError(
                f"non-finite {arm.name} gradient at {index + 1}"
            )
        optimizer.step()
        step = index + 1
        if step in REPORT_STEPS or step == steps:
            torch.cuda.synchronize()
            training_wall += time.perf_counter() - segment_started
            report(step)
            torch.cuda.synchronize()
            segment_started = time.perf_counter()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": steps,
            "parent_step": 33570,
            "config": {
                "experiment_id": EXPERIMENT_ID,
                "arm": arm.name,
                "value_noise_scale": arm.value_noise_scale,
                "ray_flow_weight": arm.ray_flow_weight,
                "ray_flow_horizons": RAY_FLOW_HORIZONS,
                "learning_rate": LEARNING_RATE,
                "seed": base.SEED,
            },
        },
        OUTPUT / f"{arm.name}_step{steps}.pt",
    )
    del optimizer, model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_examples, args.eval_microbatch) < 1:
        parser.error("all counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not CHECKPOINT.exists():
        parser.error(f"missing checkpoint: {CHECKPOINT}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    RECORD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation = base.memmap("validation")
    validation_starts = base.load_fixed_starts(
        VALIDATION_STARTS,
        args.eval_examples,
    )
    validation_windows = base.windows_of_length(
        validation,
        validation_starts,
        base.EVAL_WINDOW_LENGTH,
    )

    with (RECORD / "run.tsv").open("w", newline="") as handle:
        run_writer = csv.writer(handle, delimiter="\t")
        run_writer.writerow(("key", "value"))
        for key, value in (
            ("experiment_id", EXPERIMENT_ID),
            ("parent_checkpoint", CHECKPOINT),
            ("seed", base.SEED),
            ("data_seed", DATA_SEED),
            ("noise_seed", NOISE_SEED),
            ("steps", args.steps),
            ("batch", args.batch),
            ("horizons", base.TRAIN_HORIZONS),
            ("anchor_stride", base.TRAIN_ANCHOR_STRIDE),
            ("diffusion_rate", DIFFUSION_RATE),
            ("delta_tau", DELTA_TAU),
            ("value_noise_scale", VALUE_NOISE_SCALE),
            ("ray_flow_weight", RAY_FLOW_WEIGHT),
            ("ray_flow_horizons", RAY_FLOW_HORIZONS),
            ("learning_rate", LEARNING_RATE),
            ("gpu", torch.cuda.get_device_name()),
            ("torch", torch.__version__),
        ):
            run_writer.writerow((key, value))

    fields = metric_names()
    with (RECORD / "metrics.tsv").open("w", newline="") as metrics_handle:
        writer = csv.DictWriter(
            metrics_handle,
            fieldnames=fields,
            delimiter="\t",
        )
        writer.writeheader()
        for arm in ARMS:
            run_arm(
                arm,
                steps=args.steps,
                batch=args.batch,
                validation_windows=validation_windows,
                eval_microbatch=args.eval_microbatch,
                writer=writer,
                metrics_handle=metrics_handle,
            )


if __name__ == "__main__":
    main()
