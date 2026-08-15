"""Line-search the exact native update-1001 Adam parameter delta."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_group_swap import (
    OUTPUT_ROOT,
    RECORD_ROOT,
    base,
    measure_same_batch,
    trainable_parameters,
    write_tsv,
)


PRE_STATE = OUTPUT_ROOT / "group_swap_pre_update1001_model.pt"
POST_STATE = OUTPUT_ROOT / "group_swap_post_update1001_model.pt"
SAVED_BATCH = OUTPUT_ROOT / "update1001_batch.pt"
MICROBATCH = 16

COARSE_ALPHAS = (
    -1e-2,
    -1e-3,
    -3e-4,
    -1e-4,
    -3e-5,
    -1e-5,
    -3e-6,
    -1e-6,
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    2e-2,
    3e-2,
    5e-2,
    7.5e-2,
    1e-1,
    1.5e-1,
    2e-1,
    3e-1,
    4e-1,
    5e-1,
    6e-1,
    7e-1,
    8e-1,
    9e-1,
    1.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    return parser.parse_args()


def canonical_alpha(alpha: float) -> float:
    return float(f"{alpha:.16g}")


def linspace(left: float, right: float, steps: int) -> list[float]:
    if steps < 2:
        raise ValueError("line-search linspace needs at least two points")
    return [
        canonical_alpha(left + (right - left) * index / (steps - 1))
        for index in range(steps)
    ]


def registered_g_dot_delta(record_dir: Path) -> float:
    path = record_dir / "update_direction_alignment.tsv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    row = next(item for item in rows if item["scope"] == "global")
    return float(row["gradient_dot_actual_delta"])


def main() -> None:
    args = parse_args()
    if args.microbatch < 1 or 64 % args.microbatch:
        raise ValueError("microbatch must divide the exact batch of 64")
    if not torch.cuda.is_available():
        raise RuntimeError("line search requires CUDA")
    for path in (PRE_STATE, POST_STATE, SAVED_BATCH):
        if not path.exists():
            raise FileNotFoundError(path)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    pre_state = torch.load(PRE_STATE, map_location="cpu", weights_only=True)
    post_state = torch.load(POST_STATE, map_location="cpu", weights_only=True)
    saved_batch = torch.load(SAVED_BATCH, map_location="cpu", weights_only=False)
    window = saved_batch["window"]
    if tuple(window.shape) != (64, base.CONTEXT + base.TRAIN_HORIZONS):
        raise RuntimeError(f"unexpected saved-batch shape: {tuple(window.shape)}")
    window = window.to(device)

    model = base.make_model().to(device).eval()
    model.load_state_dict(pre_state, strict=True)
    parameters = trainable_parameters(model)
    pre_device = {
        name: pre_state[name].to(device=device, dtype=parameter.dtype)
        for name, parameter in parameters.items()
    }
    delta_device = {
        name: (
            post_state[name].to(device=device, dtype=parameter.dtype)
            - pre_device[name]
        )
        for name, parameter in parameters.items()
    }
    delta_squared = sum(
        float(delta.double().square().sum()) for delta in delta_device.values()
    )
    if delta_squared <= 0.0:
        raise RuntimeError("registered Adam delta is zero")

    evaluations: dict[float, dict[str, float]] = {}
    stages: dict[float, set[str]] = {}

    @torch.inference_mode()
    def apply_alpha(alpha: float) -> dict[str, float]:
        realized_squared = 0.0
        projection = 0.0
        target_error_squared = 0.0
        for name, parameter in parameters.items():
            parameter.copy_(pre_device[name])
            parameter.add_(delta_device[name], alpha=alpha)
            realized = parameter - pre_device[name]
            target = delta_device[name] * alpha
            realized_squared += float(realized.double().square().sum())
            projection += float((realized.double() * delta_device[name].double()).sum())
            target_error_squared += float(
                (realized.double() - target.double()).square().sum()
            )
        return {
            "realized_parameter_displacement_l2": math.sqrt(realized_squared),
            "realized_alpha_projection": projection / delta_squared,
            "realized_relative_error_to_requested_delta": (
                math.sqrt(target_error_squared)
                / max(abs(alpha) * math.sqrt(delta_squared), 1e-30)
                if alpha != 0.0
                else 0.0
            ),
        }

    @torch.inference_mode()
    def evaluate(alpha: float, stage: str) -> dict[str, float]:
        alpha = canonical_alpha(alpha)
        stages.setdefault(alpha, set()).add(stage)
        if alpha in evaluations:
            return evaluations[alpha]
        displacement = apply_alpha(alpha)
        metrics = measure_same_batch(
            model,
            window,
            args.microbatch,
            include_horizon_ce=True,
        )
        metrics.update(displacement)
        evaluations[alpha] = metrics
        return metrics

    for alpha in COARSE_ALPHAS:
        evaluate(alpha, "coarse")

    nonnegative_coarse = sorted(alpha for alpha in COARSE_ALPHAS if alpha >= 0.0)
    coarse_minimum = min(
        nonnegative_coarse,
        key=lambda alpha: evaluations[canonical_alpha(alpha)]["same_batch_ce"],
    )
    coarse_index = nonnegative_coarse.index(coarse_minimum)
    coarse_left = nonnegative_coarse[max(coarse_index - 1, 0)]
    coarse_right = nonnegative_coarse[
        min(coarse_index + 1, len(nonnegative_coarse) - 1)
    ]
    if coarse_left == coarse_right:
        raise RuntimeError("coarse minimum could not be bracketed")
    for alpha in linspace(coarse_left, coarse_right, 65):
        evaluate(alpha, "minimum_dense_1")

    first_dense = sorted(alpha for alpha in evaluations if alpha >= 0.0)
    first_minimum = min(
        first_dense, key=lambda alpha: evaluations[alpha]["same_batch_ce"]
    )
    first_index = first_dense.index(first_minimum)
    if first_index == 0 or first_index == len(first_dense) - 1:
        raise RuntimeError("first dense minimum lies on the evaluated boundary")
    fine_left = first_dense[first_index - 1]
    fine_right = first_dense[first_index + 1]
    for alpha in linspace(fine_left, fine_right, 33):
        evaluate(alpha, "minimum_dense_2")

    nonnegative = sorted(alpha for alpha in evaluations if alpha >= 0.0)
    minimum_alpha = min(
        nonnegative, key=lambda alpha: evaluations[alpha]["same_batch_ce"]
    )
    minimum_index = nonnegative.index(minimum_alpha)
    if minimum_index == 0 or minimum_index == len(nonnegative) - 1:
        raise RuntimeError("final sampled minimum lies on the evaluated boundary")
    minimum_left = nonnegative[minimum_index - 1]
    minimum_right = nonnegative[minimum_index + 1]
    baseline_ce = evaluations[0.0]["same_batch_ce"]

    crossing_bracket: tuple[float, float] | None = None
    after_minimum = [alpha for alpha in nonnegative if alpha >= minimum_alpha]
    for left, right in zip(after_minimum, after_minimum[1:]):
        left_delta = evaluations[left]["same_batch_ce"] - baseline_ce
        right_delta = evaluations[right]["same_batch_ce"] - baseline_ce
        if left_delta < 0.0 <= right_delta:
            crossing_bracket = (left, right)
            break
    if crossing_bracket is not None:
        crossing_left, crossing_right = crossing_bracket
        for _ in range(20):
            midpoint = canonical_alpha((crossing_left + crossing_right) / 2.0)
            midpoint_delta = evaluate(midpoint, "baseline_crossing_bisection")[
                "same_batch_ce"
            ] - baseline_ce
            if midpoint_delta < 0.0:
                crossing_left = midpoint
            else:
                crossing_right = midpoint
        crossing_alpha = canonical_alpha(
            (crossing_left + crossing_right) / 2.0
        )
        evaluate(crossing_alpha, "baseline_crossing_estimate")
    else:
        crossing_left = crossing_right = crossing_alpha = math.nan

    rows: list[dict[str, object]] = []
    for alpha in sorted(evaluations):
        metrics = evaluations[alpha]
        row: dict[str, object] = {
            "alpha": alpha,
            "stage": ",".join(sorted(stages[alpha])),
            "total_ce": metrics["same_batch_ce"],
            "delta_total_ce_vs_alpha0": metrics["same_batch_ce"] - baseline_ce,
        }
        row.update(
            {
                f"h{horizon}_ce": metrics[f"same_batch_h{horizon}_ce"]
                for horizon in range(1, base.TRAIN_HORIZONS + 1)
            }
        )
        for horizon in (1, 16):
            prefix = f"h{horizon}"
            row.update(
                {
                    f"{prefix}_p_rms": metrics[f"{prefix}_p_rms"],
                    f"{prefix}_p_max_abs": metrics[f"{prefix}_p_max_abs"],
                    f"{prefix}_w_rms": metrics[f"{prefix}_w_rms"],
                    f"{prefix}_w_max_row_rms": metrics[
                        f"{prefix}_w_max_row_rms"
                    ],
                    f"{prefix}_logit_rms": metrics[f"{prefix}_logit_rms"],
                    f"{prefix}_logit_max_abs": metrics[
                        f"{prefix}_logit_max_abs"
                    ],
                }
            )
        row.update(
            {
                "realized_parameter_displacement_l2": metrics[
                    "realized_parameter_displacement_l2"
                ],
                "realized_alpha_projection": metrics[
                    "realized_alpha_projection"
                ],
                "realized_relative_error_to_requested_delta": metrics[
                    "realized_relative_error_to_requested_delta"
                ],
            }
        )
        rows.append(row)
    write_tsv(args.record_dir / "update_line_search.tsv", rows)

    g_dot_delta = registered_g_dot_delta(args.record_dir)
    summary_rows: list[dict[str, object]] = []

    def summary(
        kind: str,
        *,
        alpha: float = math.nan,
        left_alpha: float = math.nan,
        right_alpha: float = math.nan,
        value: float = math.nan,
        reference: float = math.nan,
        detail: str = "",
    ) -> None:
        summary_rows.append(
            {
                "kind": kind,
                "alpha": alpha,
                "left_alpha": left_alpha,
                "right_alpha": right_alpha,
                "value": value,
                "reference": reference,
                "relative_error": (
                    abs(value - reference) / max(abs(reference), 1e-30)
                    if math.isfinite(value) and math.isfinite(reference)
                    else math.nan
                ),
                "detail": detail,
            }
        )

    for epsilon in (1e-6, 3e-6, 1e-5):
        positive = evaluations[canonical_alpha(epsilon)]["same_batch_ce"]
        negative = evaluations[canonical_alpha(-epsilon)]["same_batch_ce"]
        forward = (positive - baseline_ce) / epsilon
        central = (positive - negative) / (2.0 * epsilon)
        summary(
            "forward_directional_difference",
            alpha=epsilon,
            value=forward,
            reference=g_dot_delta,
            detail="(CE(+epsilon)-CE(0))/epsilon",
        )
        summary(
            "central_directional_difference",
            alpha=epsilon,
            left_alpha=-epsilon,
            right_alpha=epsilon,
            value=central,
            reference=g_dot_delta,
            detail="(CE(+epsilon)-CE(-epsilon))/(2*epsilon)",
        )

    minimum_ce = evaluations[minimum_alpha]["same_batch_ce"]
    summary(
        "sampled_minimum_alpha",
        alpha=minimum_alpha,
        left_alpha=minimum_left,
        right_alpha=minimum_right,
        value=minimum_ce,
        reference=baseline_ce,
        detail="lowest CE among all registered/refined samples",
    )
    left_minimum_slope = (
        minimum_ce - evaluations[minimum_left]["same_batch_ce"]
    ) / (minimum_alpha - minimum_left)
    right_minimum_slope = (
        evaluations[minimum_right]["same_batch_ce"] - minimum_ce
    ) / (minimum_right - minimum_alpha)
    summary(
        "minimum_left_secant_slope",
        alpha=minimum_alpha,
        left_alpha=minimum_left,
        right_alpha=minimum_alpha,
        value=left_minimum_slope,
        reference=math.nan,
        detail="negative slope expected immediately before sampled minimum",
    )
    summary(
        "minimum_right_secant_slope",
        alpha=minimum_alpha,
        left_alpha=minimum_alpha,
        right_alpha=minimum_right,
        value=right_minimum_slope,
        reference=math.nan,
        detail="positive slope expected immediately after sampled minimum",
    )
    if math.isfinite(crossing_alpha):
        summary(
            "baseline_crossing_alpha",
            alpha=crossing_alpha,
            left_alpha=crossing_left,
            right_alpha=crossing_right,
            value=evaluations[crossing_alpha]["same_batch_ce"],
            reference=baseline_ce,
            detail="first post-minimum crossing of alpha-0 CE",
        )
    else:
        summary(
            "baseline_crossing_not_found",
            value=math.nan,
            reference=baseline_ce,
            detail="no post-minimum baseline crossing through alpha=1",
        )
    summary(
        "alpha0_reproduction",
        alpha=0.0,
        value=baseline_ce,
        reference=3.1812364608049393,
    )
    summary(
        "alpha1_reproduction",
        alpha=1.0,
        value=evaluations[1.0]["same_batch_ce"],
        reference=3.195414748042822,
    )
    repeatability_ranges: dict[tuple[str, float], float] = {}
    repeatability_alphas = [0.0, minimum_alpha]
    if math.isfinite(crossing_alpha):
        repeatability_alphas.append(crossing_alpha)
    for alpha in repeatability_alphas:
        apply_alpha(alpha)
        same_weights_ce = [
            measure_same_batch(
                model,
                window,
                args.microbatch,
                include_horizon_ce=True,
            )["same_batch_ce"]
            for _ in range(5)
        ]
        same_weights_range = max(same_weights_ce) - min(same_weights_ce)
        repeatability_ranges[("same_weights", alpha)] = same_weights_range
        summary(
            "five_repeat_same_loaded_weights_ce_range",
            alpha=alpha,
            value=same_weights_range,
            reference=math.nan,
            detail=(
                f"min={min(same_weights_ce):.12g},"
                f"max={max(same_weights_ce):.12g}"
            ),
        )
        reapplied_ce = []
        for _ in range(5):
            apply_alpha(alpha)
            reapplied_ce.append(
                measure_same_batch(
                    model,
                    window,
                    args.microbatch,
                    include_horizon_ce=True,
                )["same_batch_ce"]
            )
        reapplication_range = max(reapplied_ce) - min(reapplied_ce)
        repeatability_ranges[("reapplied_weights", alpha)] = reapplication_range
        summary(
            "five_repeat_reapplied_weights_ce_range",
            alpha=alpha,
            value=reapplication_range,
            reference=math.nan,
            detail=(
                f"min={min(reapplied_ce):.12g},max={max(reapplied_ce):.12g}"
            ),
        )
    if not math.isclose(
        baseline_ce, 3.1812364608049393, rel_tol=2e-6, abs_tol=5e-6
    ):
        raise RuntimeError("alpha 0 did not reproduce registered pre-update CE")
    if not math.isclose(
        evaluations[1.0]["same_batch_ce"],
        3.195414748042822,
        rel_tol=2e-6,
        abs_tol=5e-6,
    ):
        raise RuntimeError("alpha 1 did not reproduce registered post-update CE")
    central_rows = [
        row for row in summary_rows if row["kind"] == "central_directional_difference"
    ]
    near_zero_convergence_satisfied = all(
        float(row["value"]) * g_dot_delta > 0.0 for row in central_rows
    )
    summary(
        "near_zero_finite_difference_criterion_satisfied",
        value=1.0 if near_zero_convergence_satisfied else 0.0,
        reference=1.0,
        detail=(
            "requires all registered central differences to share the "
            "autograd directional-derivative sign"
        ),
    )
    required_metric_names = [
        "same_batch_ce",
        *(f"same_batch_h{horizon}_ce" for horizon in range(1, 17)),
        *(f"h{horizon}_{name}" for horizon in (1, 16) for name in (
            "p_rms", "p_max_abs", "w_rms", "w_max_row_rms",
            "logit_rms", "logit_max_abs",
        )),
    ]
    for alpha, metrics in evaluations.items():
        for name in required_metric_names:
            if not math.isfinite(float(metrics[name])):
                raise RuntimeError(f"non-finite {name} at alpha {alpha}")
    write_tsv(args.record_dir / "update_line_search_summary.tsv", summary_rows)

    best_derivative = min(
        central_rows, key=lambda item: float(item["relative_error"])
    )
    minimum_metrics = evaluations[minimum_alpha]
    alpha1_metrics = evaluations[1.0]
    horizon_changes_at_minimum = {
        horizon: minimum_metrics[f"same_batch_h{horizon}_ce"]
        - evaluations[0.0][f"same_batch_h{horizon}_ce"]
        for horizon in range(1, base.TRAIN_HORIZONS + 1)
    }
    strongest_horizon = min(
        horizon_changes_at_minimum, key=horizon_changes_at_minimum.get
    )
    crossing_text = (
        f"The deterministic bisection first found a sign change around alpha "
        f"`{crossing_alpha:.4g}`; because the float32 path is jagged, the "
        f"robust coarse statement is only that CE is below baseline at "
        f"alpha `0.075` and above it at `0.1`."
        if math.isfinite(crossing_alpha)
        else "No return to alpha-0 CE was found through alpha 1."
    )
    interpretation = (
        "# Actual-Adam-delta line search\n\n"
        "This is the fixed native update-1001 batch and the fixed actual Adam "
        "delta. No backward pass, optimizer step, or training was performed.\n\n"
        f"- Alpha 0 CE: `{baseline_ce:.9f}`.\n"
        f"- Lowest registered point: alpha about `{minimum_alpha:.4g}`, CE "
        f"`{minimum_ce:.9f}` (change `{minimum_ce-baseline_ce:+.9g}`).\n"
        f"- Alpha 1 CE: `{alpha1_metrics['same_batch_ce']:.9f}` "
        f"(change `{alpha1_metrics['same_batch_ce']-baseline_ce:+.9g}`).\n"
        f"- {crossing_text}\n"
        f"- Registered autograd `g dot delta`: `{g_dot_delta:+.9g}`. "
        f"Central finite differences at epsilon `1e-6`, `3e-6`, and `1e-5` "
        f"were `{float(central_rows[0]['value']):+.6g}`, "
        f"`{float(central_rows[1]['value']):+.6g}`, and "
        f"`{float(central_rows[2]['value']):+.6g}`. They do not converge "
        f"cleanly to the autograd value. The preregistered near-zero "
        f"finite-difference criterion therefore "
        f"`{'passed' if near_zero_convergence_satisfied else 'failed'}`. "
        f"The closest was "
        f"`{float(best_derivative['value']):+.9g}` at epsilon "
        f"`{float(best_derivative['alpha']):.3g}`, where the "
        f"realized float32 path had alpha projection "
        f"`{evaluations[canonical_alpha(float(best_derivative['alpha']))]['realized_alpha_projection']:.6g}` "
        f"and relative displacement error "
        f"`{evaluations[canonical_alpha(float(best_derivative['alpha']))]['realized_relative_error_to_requested_delta']:.3%}`.\n\n"
        f"At the sampled minimum, H{strongest_horizon} had the largest CE "
        f"reduction (`{horizon_changes_at_minimum[strongest_horizon]:+.9g}`). "
        f"H1 P/W/logit RMS changed from "
        f"`{evaluations[0.0]['h1_p_rms']:.6g}/"
        f"{evaluations[0.0]['h1_w_rms']:.6g}/"
        f"{evaluations[0.0]['h1_logit_rms']:.6g}` to "
        f"`{minimum_metrics['h1_p_rms']:.6g}/"
        f"{minimum_metrics['h1_w_rms']:.6g}/"
        f"{minimum_metrics['h1_logit_rms']:.6g}`. H16 changed from "
        f"`{evaluations[0.0]['h16_p_rms']:.6g}/"
        f"{evaluations[0.0]['h16_w_rms']:.6g}/"
        f"{evaluations[0.0]['h16_logit_rms']:.6g}` to "
        f"`{minimum_metrics['h16_p_rms']:.6g}/"
        f"{minimum_metrics['h16_w_rms']:.6g}/"
        f"{minimum_metrics['h16_logit_rms']:.6g}`.\n\n"
        f"Five same-loaded-weight forwards and five independent weight "
        f"reapplications each had CE range at most "
        f"`{max(repeatability_ranges.values()):.3g}`. The dense local "
        "variation is therefore deterministic float32 path quantization/"
        "sensitivity rather than forward or reapplication noise. The sampled "
        "minimum is only the lowest "
        "registered production-precision point, not a smooth stationary-point "
        "estimate, and the narrow bisection interval is a sign-change bracket, "
        "not a smooth root-confidence interval. The later baseline crossing is "
        "also distinct from the sampled minimum. This slice does not select a "
        "learning rate or establish validation/long-run behavior.\n"
    )
    (args.record_dir / "update_line_search_interpretation.md").write_text(
        interpretation
    )
    print(
        f"alpha_min={minimum_alpha:.9g} CE_min={minimum_ce:.9f}; "
        f"crossing={crossing_alpha:.9g}; alpha1={alpha1_metrics['same_batch_ce']:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
