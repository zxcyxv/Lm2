"""Train a prefix-only orthogonal affine tape by associative scan."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.prefix_orthogonal_scan import (
    PrefixConditionedOrthogonalAffineScan,
    apply_pairwise_affine_scan,
    rotate_pairwise,
    sequential_pairwise_affine,
)
from rotlm.training.ha_skew_window import (
    PrefixOrthogonalScanWindowOutput,
    relative_mse_rows,
    sparse_prefix_orthogonal_affine_scan_loss,
)
from train_k1_decoder_inverse_ablation_13m import (
    CLIP_NORM,
    CONTEXT,
    PEAK_LR,
    SEED,
    VOCABULARY,
    active_head_self_retrieval,
    lr_at,
    memmap,
    parameter_count,
    trainable_parameter_count,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length


EXPERIMENT_ID = (
    "EXP-20260726-k4-ha-skew-stride1-batch64-micro16-"
    "prefix-conditioned-orthogonal-affine-scan-ce-closure-13m"
)
WIDTH = 896
HORIZONS = 4
ANCHOR_STRIDE = 1
WINDOW_LENGTH = CONTEXT + HORIZONS
ANCHOR_COUNT = len(range(0, CONTEXT, ANCHOR_STRIDE))
PLANNER_BOTTLENECK = 128
POSITION_WIDTH = 32
MAX_ANGLE_DELTA = 0.25
INITIAL_SCALE = 0.05
MIN_SCALE = 1e-3
MAX_SCALE = 0.25
CLOSURE_WEIGHT = 1.0
EFFECTIVE_BATCH = 64
MICROBATCH = 16
STEPS = 1000
SCHEDULE_STEPS = 6000
BENCHMARK_HORIZON = 128
BENCHMARK_REPEATS = 10
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID


def make_model() -> K1DecoderAblationLM:
    model = K1DecoderAblationLM(
        VOCABULARY,
        width=WIDTH,
        encoder_blocks=2,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
        trainable_cosine_scale=True,
    )
    model.operator = nn.Identity()
    model.prefix_orthogonal_scan = (
        PrefixConditionedOrthogonalAffineScan(
            WIDTH,
            bottleneck=PLANNER_BOTTLENECK,
            position_width=POSITION_WIDTH,
            max_angle_delta=MAX_ANGLE_DELTA,
            init_scale=INITIAL_SCALE,
            min_scale=MIN_SCALE,
            max_scale=MAX_SCALE,
        )
    )
    return model


def sampled_windows(
    data,
    batch: int,
    generator: torch.Generator,
) -> torch.Tensor:
    starts = torch.randint(
        0,
        len(data) - WINDOW_LENGTH,
        (batch,),
        generator=generator,
    )
    return windows_of_length(data, starts, WINDOW_LENGTH)


def basis_orthogonality_error(model: K1DecoderAblationLM) -> float:
    basis = model.prefix_orthogonal_scan.basis.detach()
    identity = torch.eye(WIDTH, device=basis.device)
    return float((basis.T @ basis - identity).abs().max())


def objective(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PrefixOrthogonalScanWindowOutput,
]:
    return sparse_prefix_orthogonal_affine_scan_loss(
        model,
        window,
        prefix_length=CONTEXT,
        horizons=HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
        closure_weight=CLOSURE_WEIGHT,
    )


def _require_finite_nonzero(
    name: str,
    gradient: torch.Tensor | None,
) -> None:
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.norm() > 0)
    ):
        raise RuntimeError(f"no finite nonzero gradient to {name}")


def preflight(
    model: K1DecoderAblationLM,
    training,
) -> dict[str, float]:
    window = sampled_windows(
        training,
        2,
        torch.Generator().manual_seed(SEED + 8242),
    )
    loss, token_nll, closure, output = objective(model, window)
    expected = (2, ANCHOR_COUNT, HORIZONS)
    if output.token_ce.shape != expected:
        raise RuntimeError(
            f"unexpected token CE shape {tuple(output.token_ce.shape)}"
        )
    if output.logits.shape != (*expected, VOCABULARY):
        raise RuntimeError("unexpected scan logit shape")
    if output.local_angles.shape != (
        2,
        ANCHOR_COUNT,
        HORIZONS,
        WIDTH // 2,
    ):
        raise RuntimeError("unexpected local-angle shape")
    if output.behavioral_closure_by_horizon.shape != (HORIZONS - 1,):
        raise RuntimeError("closure must cover horizons 2 through H")
    if output.teacher_logits.requires_grad:
        raise RuntimeError("closure teacher unexpectedly has a graph")
    if output.reanchored_states.requires_grad:
        raise RuntimeError("reanchored states unexpectedly have a graph")
    torch.testing.assert_close(
        loss,
        token_nll + CLOSURE_WEIGHT * closure,
    )
    if (
        not bool(torch.isfinite(loss))
        or float(closure.detach()) < -1e-6
    ):
        raise RuntimeError("preflight loss is invalid")

    positions = torch.arange(WINDOW_LENGTH, device=window.device)
    full_encoded, _ = model.encode(window, positions)
    roots = full_encoded[:, :CONTEXT].index_select(
        1,
        output.anchor_indices,
    )
    coordinate_root = torch.nn.functional.linear(
        roots,
        output.basis.T,
    )
    sequential_states = sequential_pairwise_affine(
        coordinate_root,
        output.local_angles,
        output.innovations,
    )
    scan_error = float(
        (
            output.coordinate_states - sequential_states
        ).detach().abs().max()
    )
    if scan_error >= 1e-5:
        raise RuntimeError("associative scan does not match recurrence")
    reconstructed = (
        rotate_pairwise(
            coordinate_root.unsqueeze(2).expand_as(
                output.coordinate_states
            ),
            output.cumulative_angles,
        )
        + output.cumulative_innovations
    )
    composition_error = float(
        (
            reconstructed - output.coordinate_states
        ).detach().abs().max()
    )
    if composition_error >= 1e-6:
        raise RuntimeError("cumulative affine map reconstruction failed")

    rotated_root = rotate_pairwise(
        coordinate_root,
        output.local_angles[:, :, 0],
    )
    norm_error = float(
        (
            (
                rotated_root.float().square().sum(dim=-1).sqrt()
                - coordinate_root.float().square().sum(dim=-1).sqrt()
            ).abs()
            / coordinate_root.float().square().sum(
                dim=-1
            ).sqrt().clamp_min(1e-8)
        ).detach().max()
    )
    if norm_error >= 1e-6:
        raise RuntimeError("local pairwise operator is not orthogonal")

    changed_future = window.clone()
    changed_future[:, CONTEXT:] = (
        changed_future[:, CONTEXT:] + 1
    ) % VOCABULARY
    with torch.no_grad():
        changed_output = objective(model, changed_future)[3]
    future_state_error = float(
        (
            output.predicted_states.detach()
            - changed_output.predicted_states
        ).abs().max()
    )
    future_logit_error = float(
        (
            output.logits.detach() - changed_output.logits
        ).abs().max()
    )
    if max(future_state_error, future_logit_error) >= 1e-6:
        raise RuntimeError("open-loop scan depends on future gold tokens")

    scan_module = model.prefix_orthogonal_scan
    ce_parameters = (
        scan_module.basis_generator,
        scan_module.context.weight,
        scan_module.position.weight,
        scan_module.angle.weight,
        scan_module.innovation.weight,
        scan_module.scale.weight,
        model.encoder.blocks[0].attn.qkv.weight,
        model.embedding_weight,
    )
    ce_gradients = torch.autograd.grad(
        token_nll,
        ce_parameters,
        retain_graph=True,
    )
    for name, gradient in zip(
        (
            "CE orthogonal basis",
            "CE planner context",
            "CE planner position",
            "CE angle head",
            "CE innovation head",
            "CE innovation scale",
            "CE encoder attention",
            "CE tied embedding",
        ),
        ce_gradients,
    ):
        _require_finite_nonzero(name, gradient)

    closure_gradients = torch.autograd.grad(
        closure,
        (
            scan_module.basis_generator,
            scan_module.angle.weight,
            scan_module.innovation.weight,
            scan_module.scale.weight,
        ),
        retain_graph=True,
    )
    for name, gradient in zip(
        (
            "closure orthogonal basis",
            "closure angle head",
            "closure innovation head",
            "closure innovation scale",
        ),
        closure_gradients,
    ):
        _require_finite_nonzero(name, gradient)
    gold_gradient = torch.autograd.grad(
        closure,
        output.gold_states,
        allow_unused=True,
    )[0]
    if gold_gradient is not None and bool(gold_gradient.norm() > 0):
        raise RuntimeError("closure teacher leaked into online gold states")

    scale = output.log_scale.detach().float().exp()
    if (
        float(scale.min()) < MIN_SCALE - 1e-6
        or float(scale.max()) > MAX_SCALE + 1e-6
    ):
        raise RuntimeError("innovation scale escaped registered bounds")
    basis = output.basis.detach()
    identity = torch.eye(WIDTH, device=window.device)
    identity_difference = float((basis - identity).abs().max())
    initial_orthogonality = basis_orthogonality_error(model)
    if identity_difference >= 1e-4 or initial_orthogonality >= 1e-4:
        raise RuntimeError("Q is not identity/orthogonal at initialization")
    if any(
        hasattr(model, name)
        for name in (
            "trajectory_innovation",
            "trajectory_initializer",
            "trajectory_operator",
            "token_conditioned_transition",
        )
    ):
        raise RuntimeError("a forbidden branch module is still present")

    values = {
        "initial_loss": float(loss.detach()),
        "initial_token_nll": float(token_nll.detach()),
        "initial_behavioral_closure_kl": float(closure.detach()),
        "initial_scan_recurrence_error": scan_error,
        "initial_scan_composition_error": composition_error,
        "initial_local_rotation_norm_error": norm_error,
        "initial_future_gold_state_error": future_state_error,
        "initial_future_gold_logit_error": future_logit_error,
        "initial_ce_basis_gradient_norm": float(
            ce_gradients[0].norm()
        ),
        "initial_ce_angle_gradient_norm": float(
            ce_gradients[3].norm()
        ),
        "initial_ce_innovation_gradient_norm": float(
            ce_gradients[4].norm()
        ),
        "initial_closure_innovation_gradient_norm": float(
            closure_gradients[2].norm()
        ),
        "closure_gold_target_gradient_norm": (
            0.0
            if gold_gradient is None
            else float(gold_gradient.norm())
        ),
        "initial_angle_abs_mean": float(
            output.local_angles.detach().float().abs().mean()
        ),
        "initial_angle_abs_max": float(
            output.local_angles.detach().float().abs().max()
        ),
        "initial_innovation_scale_mean": float(scale.mean()),
        "initial_innovation_scale_min": float(scale.min()),
        "initial_innovation_scale_max": float(scale.max()),
        "basis_identity_diff": identity_difference,
        "basis_orthogonality_error": initial_orthogonality,
        "head_self_retrieval": active_head_self_retrieval(model),
    }
    model.zero_grad(set_to_none=True)
    return values


@torch.inference_mode()
def benchmark_central_rollout(
    model: K1DecoderAblationLM,
    *,
    horizon: int = BENCHMARK_HORIZON,
    repeats: int = BENCHMARK_REPEATS,
) -> dict[str, float]:
    generator = torch.Generator(device="cuda").manual_seed(SEED + 8383)
    roots = torch.randn(
        1,
        ANCHOR_COUNT,
        WIDTH,
        device="cuda",
        generator=generator,
    )
    (
        coordinate_root,
        angles,
        innovations,
        _,
        _,
    ) = model.prefix_orthogonal_scan.compile_tape(roots, horizon)

    scanned = apply_pairwise_affine_scan(
        coordinate_root,
        angles,
        innovations,
    )[0]
    sequential = sequential_pairwise_affine(
        coordinate_root,
        angles,
        innovations,
    )
    torch.cuda.synchronize()
    error = float((scanned - sequential).abs().max())

    def measure(function) -> list[float]:
        timings = []
        for _ in range(2):
            function()
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
        return timings

    scan_times = measure(
        lambda: apply_pairwise_affine_scan(
            coordinate_root,
            angles,
            innovations,
        )[0]
    )
    sequential_times = measure(
        lambda: sequential_pairwise_affine(
            coordinate_root,
            angles,
            innovations,
        )
    )
    scan_median = statistics.median(scan_times)
    sequential_median = statistics.median(sequential_times)
    return {
        "central_benchmark_horizon": float(horizon),
        "central_benchmark_batch": 1.0,
        "central_benchmark_anchors": float(ANCHOR_COUNT),
        "central_scan_sequential_max_error": error,
        "central_scan_median_ms": scan_median,
        "central_sequential_median_ms": sequential_median,
        "central_scan_speedup": sequential_median / scan_median,
    }


@torch.inference_mode()
def evaluate(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    microbatch: int,
    basis_gradient_norm: float,
) -> dict[str, float]:
    model.eval()
    nll_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    correct_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    state_mse_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    closure_sum = torch.zeros(HORIZONS - 1, dtype=torch.float64)
    scale_sum = 0.0
    scale_count = 0
    scale_min = math.inf
    scale_max = -math.inf
    angle_abs_sum = 0.0
    angle_count = 0
    angle_abs_max = 0.0
    decisions = 0

    for begin in range(0, len(starts), microbatch):
        window = windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            WINDOW_LENGTH,
        )
        _, _, _, output = objective(model, window)
        batch = window.shape[0]
        anchors = output.anchor_indices.numel()
        count = batch * anchors
        predictions = output.logits.argmax(dim=-1)
        nll_sum += output.token_ce.float().sum(dim=(0, 1)).cpu()
        correct_sum += predictions.eq(output.targets).float().sum(
            dim=(0, 1)
        ).cpu()
        state_mse_sum += relative_mse_rows(
            output.predicted_states,
            output.gold_states,
        ).sum(dim=(0, 1)).cpu()
        closure_sum += (
            output.behavioral_closure_by_horizon.float().cpu() * count
        )

        scale = output.log_scale.float().exp()
        scale_sum += float(scale.sum())
        scale_count += scale.numel()
        scale_min = min(scale_min, float(scale.min()))
        scale_max = max(scale_max, float(scale.max()))
        angles = output.local_angles.float().abs()
        angle_abs_sum += float(angles.sum())
        angle_count += angles.numel()
        angle_abs_max = max(angle_abs_max, float(angles.max()))
        decisions += count

    metrics = {
        "token_nll": float((nll_sum / decisions).mean()),
        "token_accuracy": float((correct_sum / decisions).mean()),
        "behavioral_closure_kl": float(
            closure_sum.sum() / decisions / (HORIZONS - 1)
        ),
        "innovation_scale_mean": scale_sum / scale_count,
        "innovation_scale_min": scale_min,
        "innovation_scale_max": scale_max,
        "angle_abs_mean": angle_abs_sum / angle_count,
        "angle_abs_max": angle_abs_max,
        "basis_gradient_norm": basis_gradient_norm,
        "basis_orthogonality_error": basis_orthogonality_error(model),
    }
    for horizon in range(HORIZONS):
        suffix = horizon + 1
        metrics[f"h{suffix}_nll"] = float(nll_sum[horizon] / decisions)
        metrics[f"h{suffix}_accuracy"] = float(
            correct_sum[horizon] / decisions
        )
        metrics[f"h{suffix}_state_relative_mse"] = float(
            state_mse_sum[horizon] / decisions
        )
        metrics[f"h{suffix}_closure_kl"] = (
            math.nan
            if horizon == 0
            else float(closure_sum[horizon - 1] / decisions)
        )
    model.train()
    return metrics


def save_checkpoint(
    path: Path,
    model: K1DecoderAblationLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "config": {
                "experiment_id": EXPERIMENT_ID,
                "vocabulary": VOCABULARY,
                "width": WIDTH,
                "context": CONTEXT,
                "horizons": HORIZONS,
                "candidate_trajectories": 0,
                "anchor_stride": ANCHOR_STRIDE,
                "planner_bottleneck": PLANNER_BOTTLENECK,
                "position_width": POSITION_WIDTH,
                "max_angle_delta": MAX_ANGLE_DELTA,
                "innovation_scale": (
                    INITIAL_SCALE,
                    MIN_SCALE,
                    MAX_SCALE,
                ),
                "closure_weight": CLOSURE_WEIGHT,
                "seed": SEED,
                "operator": "prefix_conditioned_pairwise_orthogonal",
                "objective": "token_ce_plus_future_behavioral_closure",
                "scan": "hillis_steele_pairwise_affine",
                "token_conditioning": "none_open_loop",
            },
        },
        path,
    )


def should_report(step: int, final_step: int) -> bool:
    return (
        step in (1, 50)
        or step % 100 == 0
        or (step > 1000 and step % 500 == 0)
        or step == final_step
    )


def should_save_milestone(step: int, final_step: int) -> bool:
    return (
        step in (100, 250, 500, 750, 1000)
        or (step > 1000 and step % 1000 == 0)
        or step == final_step
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=EFFECTIVE_BATCH)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    parser.add_argument("--schedule-steps", type=int, default=SCHEDULE_STEPS)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-microbatch", type=int, default=2)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch,
        args.microbatch,
        args.schedule_steps,
        args.eval_examples,
        args.eval_microbatch,
    ) < 1:
        parser.error("step, batch, and evaluation counts must be positive")
    if args.batch % args.microbatch:
        parser.error("batch must be divisible by microbatch")
    if not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    training, validation = memmap("train"), memmap("validation")
    validation_starts = torch.randint(
        0,
        len(validation) - WINDOW_LENGTH,
        (args.eval_examples,),
        generator=torch.Generator().manual_seed(SEED + 999),
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.record_dir / "validation_starts.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "token_start"))
        writer.writerows(enumerate(validation_starts.tolist()))

    model = make_model().cuda().train()
    preflight_values = preflight(model, training)
    benchmark_values = benchmark_central_rollout(model)
    if (
        benchmark_values["central_scan_sequential_max_error"] >= 1e-4
    ):
        raise RuntimeError("horizon-128 scan does not match recurrence")
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    accumulation_steps = args.batch // args.microbatch
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"paths=1 horizons={HORIZONS} batch={args.batch} "
        f"micro={args.microbatch}x{accumulation_steps} "
        f"initial_nll={preflight_values['initial_token_nll']:.4f} "
        f"scan128={benchmark_values['central_scan_median_ms']:.3f}ms "
        f"seq128={benchmark_values['central_sequential_median_ms']:.3f}ms",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in (
            ("experiment_id", EXPERIMENT_ID),
            ("seed", SEED),
            ("steps", args.steps),
            ("schedule_steps", args.schedule_steps),
            ("effective_batch", args.batch),
            ("microbatch", args.microbatch),
            ("accumulation_steps", accumulation_steps),
            ("context", CONTEXT),
            ("window_length", WINDOW_LENGTH),
            ("horizons", HORIZONS),
            ("candidate_trajectories", 0),
            ("anchor_stride", ANCHOR_STRIDE),
            ("anchor_count", ANCHOR_COUNT),
            ("planner_bottleneck", PLANNER_BOTTLENECK),
            ("position_width", POSITION_WIDTH),
            ("max_angle_delta", MAX_ANGLE_DELTA),
            ("initial_innovation_scale", INITIAL_SCALE),
            ("minimum_innovation_scale", MIN_SCALE),
            ("maximum_innovation_scale", MAX_SCALE),
            ("behavioral_closure_weight", CLOSURE_WEIGHT),
            ("behavioral_closure_start_horizon", 2),
            ("precision", "strict_float32_no_tf32"),
            ("operator", "prefix_conditioned_pairwise_orthogonal"),
            ("scan", "hillis_steele_pairwise_affine"),
            ("token_conditioning", "none_open_loop"),
            ("candidate_prior", "none"),
            ("sampled_noise", "none"),
            ("trajectory_comparison", "none"),
            ("latent_mse", "diagnostic_only"),
            ("objective", "token_ce_plus_future_behavioral_closure"),
            ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            *preflight_values.items(),
            *benchmark_values.items(),
        ):
            writer.writerow((key, value))

    metric_names = (
        "token_nll",
        "token_accuracy",
        "behavioral_closure_kl",
        "innovation_scale_mean",
        "innovation_scale_min",
        "innovation_scale_max",
        "angle_abs_mean",
        "angle_abs_max",
        "basis_gradient_norm",
        "basis_orthogonality_error",
        *tuple(
            f"h{horizon}_{metric}"
            for horizon in range(1, HORIZONS + 1)
            for metric in (
                "nll",
                "accuracy",
                "state_relative_mse",
                "closure_kl",
            )
        ),
    )
    fields = (
        "step",
        "train_token_nll",
        "train_behavioral_closure_kl",
        "train_state_relative_mse",
        "train_innovation_scale_mean",
        "train_angle_abs_mean",
        *tuple(f"val_{name}" for name in metric_names),
        "lr",
        "wall_s",
        "unique_tokens_per_s",
        "open_loop_labels_per_s",
        "peak_vram_bytes",
    )

    data_generator = torch.Generator().manual_seed(SEED)
    best_nll = math.inf
    best_step = 0
    started = time.time()
    processed_unique_tokens = 0
    processed_labels = 0
    torch.cuda.reset_peak_memory_stats()
    with (args.record_dir / "metrics.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
        )
        writer.writeheader()
        for index in range(args.steps):
            lr = lr_at(index, args.schedule_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(training, args.batch, data_generator)
            optimizer.zero_grad(set_to_none=True)
            train_sums = {
                "token_nll": 0.0,
                "behavioral_closure_kl": 0.0,
                "state_relative_mse": 0.0,
                "innovation_scale_mean": 0.0,
                "angle_abs_mean": 0.0,
            }
            micro_windows = window.split(args.microbatch)
            for micro_window in micro_windows:
                loss, token_nll, closure, output = objective(
                    model,
                    micro_window,
                )
                (loss / len(micro_windows)).backward()
                train_sums["token_nll"] += (
                    float(token_nll.detach()) / len(micro_windows)
                )
                train_sums["behavioral_closure_kl"] += (
                    float(closure.detach()) / len(micro_windows)
                )
                train_sums["state_relative_mse"] += (
                    float(
                        relative_mse_rows(
                            output.predicted_states.detach(),
                            output.gold_states.detach(),
                        ).mean()
                    )
                    / len(micro_windows)
                )
                train_sums["innovation_scale_mean"] += (
                    float(output.log_scale.detach().float().exp().mean())
                    / len(micro_windows)
                )
                train_sums["angle_abs_mean"] += (
                    float(output.local_angles.detach().float().abs().mean())
                    / len(micro_windows)
                )
                del loss, token_nll, closure, output

            basis_gradient_norm = float(
                model.prefix_orthogonal_scan.basis_generator.grad.float().norm()
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            processed_unique_tokens += args.batch * WINDOW_LENGTH
            processed_labels += args.batch * ANCHOR_COUNT * HORIZONS
            step = index + 1

            if should_report(step, args.steps):
                metrics = evaluate(
                    model,
                    validation,
                    validation_starts,
                    args.eval_microbatch,
                    basis_gradient_norm,
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    **{
                        f"train_{key}": value
                        for key, value in train_sums.items()
                    },
                    **{
                        f"val_{name}": metrics[name]
                        for name in metric_names
                    },
                    "lr": lr,
                    "wall_s": wall,
                    "unique_tokens_per_s": processed_unique_tokens / wall,
                    "open_loop_labels_per_s": processed_labels / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                remaining = args.steps - step
                eta_s = (wall / step) * remaining
                print(
                    f"step={step:4d}/{args.steps} nll={metrics['token_nll']:.4f} "
                    f"closure={metrics['behavioral_closure_kl']:.4f} "
                    f"scale={metrics['innovation_scale_mean']:.4f} "
                    f"angle={metrics['angle_abs_mean']:.4f} "
                    f"elapsed={wall/60:.1f}m eta={eta_s/60:.1f}m",
                    flush=True,
                )
                print(
                    "  accuracy[h1-h4]: "
                    + " ".join(
                        f"{metrics[f'h{h}_accuracy']:.3f}"
                        for h in range(1, HORIZONS + 1)
                    ),
                    flush=True,
                )
                save_checkpoint(
                    args.output_dir / "last.pt",
                    model,
                    optimizer,
                    step,
                    metrics,
                )
                if should_save_milestone(step, args.steps):
                    save_checkpoint(
                        args.output_dir / f"step{step:04d}.pt",
                        model,
                        optimizer,
                        step,
                        metrics,
                    )
                if metrics["token_nll"] < best_nll:
                    best_nll = metrics["token_nll"]
                    best_step = step
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        step,
                        metrics,
                    )

    final_benchmark = benchmark_central_rollout(model)
    with (args.record_dir / "summary.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("best_step", best_step))
        writer.writerow(("best_val_token_nll", best_nll))
        writer.writerow(("wall_s", time.time() - started))
        writer.writerow(
            ("peak_vram_bytes", torch.cuda.max_memory_allocated())
        )
        for key, value in final_benchmark.items():
            writer.writerow((f"final_{key}", value))


if __name__ == "__main__":
    main()
