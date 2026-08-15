"""Compare finite feedback rescans with the exact H16 feedback recurrence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from rotlm.training.ha_skew_window import (
    _encoded_sparse_window_components,
    forward_kl_rows,
    sparse_complex_self_predicted_kv_ce_only_fast_logits,
)
import train_byte256_unitary_conditional_ray_cfm_h16_10epoch as cfm


base = cfm.base
scan = cfm.scan
EXPERIMENT_ID = (
    "EXP-20260804-byte256-unitary-cfm-step3357-"
    "iterative-feedback-scan-audit"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
CHECKPOINTS = (
    (
        "cfm_step3357",
        cfm.DEFAULT_OUTPUT / "step3357.pt",
        3_357,
        "bdf3b93edc4cdbe60bd6bc73f6c16398c0643576de10d394bcac11d8e60b647b",
    ),
    (
        "ce_scan_step3357",
        Path(
            "outputs/experiments/"
            "EXP-20260803-byte256-unitary-fused-triton-scan-h16-"
            "10epoch-13m-rtx5090/step3357.pt"
        ),
        3_357,
        "0129c8c3289e5a3aa2fba5d92bc93df4e1a47aff45254fdf3033e79e8a781a12",
    ),
    (
        "ce_scan_step33570",
        Path(
            "outputs/experiments/"
            "EXP-20260803-byte256-unitary-fused-triton-scan-h16-"
            "10epoch-13m-rtx5090/step33570.pt"
        ),
        33_570,
        "ed97d7081a0760c50ddda5e24b0d767e804b07207dc9f54e3d7a986ba1cbec14",
    ),
)
MODES = (
    ("compiled_scan", "time-varying-scan", 0),
    ("rescan_1", "time-varying-scan", 1),
    ("rescan_2", "time-varying-scan", 2),
    ("sequential_feedback", "feedback", 0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty TSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def relative_rms(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.double() - target.double()).square().sum()
    denominator = target.double().square().sum().clamp_min(1e-24)
    return float((numerator / denominator).sqrt())


def centered_logit_rms(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    prediction = prediction.double() - prediction.double().mean(
        dim=-1, keepdim=True
    )
    target = target.double() - target.double().mean(dim=-1, keepdim=True)
    error_rms = (prediction - target).square().mean().sqrt()
    target_rms = target.square().mean().sqrt().clamp_min(1e-24)
    return float(error_rms), float(error_rms / target_rms)


def paired_interval(values: torch.Tensor) -> tuple[float, float, float, float]:
    values = values.double()
    mean = float(values.mean())
    if values.numel() < 2:
        return mean, math.nan, math.nan, math.nan
    standard_error = float(values.std(unbiased=True) / math.sqrt(values.numel()))
    return (
        mean,
        standard_error,
        mean - 1.96 * standard_error,
        mean + 1.96 * standard_error,
    )


@torch.inference_mode()
def collect_checkpoint_outputs(
    model,
    windows: torch.Tensor,
    *,
    microbatch: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    logits_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name, _, _ in MODES
    }
    state_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name, _, _ in MODES
    }
    state_parts["rescan_3"] = []
    target_parts: list[torch.Tensor] = []

    for begin in range(0, windows.shape[0], microbatch):
        batch = windows[begin : begin + microbatch].cuda()
        expected_targets = None
        for name, central_rollout, feedback_rescans in MODES:
            logits, targets = sparse_complex_self_predicted_kv_ce_only_fast_logits(
                model,
                batch,
                prefix_length=base.CONTEXT,
                horizons=base.TRAIN_HORIZONS,
                anchor_stride=base.EVAL_ANCHOR_STRIDE,
                central_rollout=central_rollout,
                feedback_rescans=feedback_rescans,
            )
            if expected_targets is None:
                expected_targets = targets
            elif not torch.equal(expected_targets, targets):
                raise RuntimeError("central rollout changed validation targets")
            logits_parts[name].append(logits.float().cpu())
        assert expected_targets is not None
        target_parts.append(expected_targets.cpu())

        (
            _,
            _,
            roots,
            _,
            component_targets,
            _,
            _,
        ) = _encoded_sparse_window_components(
            model,
            batch,
            prefix_length=base.CONTEXT,
            horizons=base.TRAIN_HORIZONS,
            anchor_stride=base.EVAL_ANCHOR_STRIDE,
        )
        if not torch.equal(expected_targets, component_targets):
            raise RuntimeError("state audit changed validation targets")
        transition = model.complex_self_prediction
        for name, feedback_rescans in (
            ("compiled_scan", 0),
            ("rescan_1", 1),
            ("rescan_2", 2),
            ("rescan_3", 3),
        ):
            rollout = transition.rollout_iterative_feedback_scan(
                roots,
                base.TRAIN_HORIZONS,
                feedback_rescans=feedback_rescans,
                collect_scan_diagnostics=False,
            )
            state_parts[name].append(rollout.full_states.float().cpu())
        feedback = transition.rollout(roots, base.TRAIN_HORIZONS)
        state_parts["sequential_feedback"].append(
            feedback.full_states.float().cpu()
        )

    logits_by_mode = {
        name: torch.cat(parts, dim=0) for name, parts in logits_parts.items()
    }
    states_by_mode = {
        name: torch.cat(parts, dim=0) for name, parts in state_parts.items()
    }
    return logits_by_mode, states_by_mode, torch.cat(target_parts, dim=0)


@torch.inference_mode()
def benchmark_modes(
    model,
    windows: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float]]:
    batch = windows.cuda()
    timings: dict[str, dict[str, float]] = {}
    for name, central_rollout, feedback_rescans in MODES:
        def call():
            return sparse_complex_self_predicted_kv_ce_only_fast_logits(
                model,
                batch,
                prefix_length=base.CONTEXT,
                horizons=base.TRAIN_HORIZONS,
                anchor_stride=base.EVAL_ANCHOR_STRIDE,
                central_rollout=central_rollout,
                feedback_rescans=feedback_rescans,
            )

        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            elapsed.append(float(start.elapsed_time(end)))
        peak_allocated = torch.cuda.max_memory_allocated()
        timings[name] = {
            "latency_ms_mean": statistics.fmean(elapsed),
            "latency_ms_median": statistics.median(elapsed),
            "latency_ms_min": min(elapsed),
            "latency_ms_max": max(elapsed),
            "baseline_allocated_bytes": float(baseline_allocated),
            "peak_allocated_bytes": float(peak_allocated),
            "incremental_peak_bytes": float(
                max(0, peak_allocated - baseline_allocated)
            ),
        }
    scan_median = timings["compiled_scan"]["latency_ms_median"]
    for values in timings.values():
        values["latency_ratio_vs_scan"] = (
            values["latency_ms_median"] / scan_median
        )
    return timings


def summarize_checkpoint(
    checkpoint_name: str,
    checkpoint_step: int,
    logits_by_mode: dict[str, torch.Tensor],
    states_by_mode: dict[str, torch.Tensor],
    targets: torch.Tensor,
    timings: dict[str, dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feedback_logits = logits_by_mode["sequential_feedback"]
    feedback_states = states_by_mode["sequential_feedback"]
    scan_nll = F.cross_entropy(
        logits_by_mode["compiled_scan"].reshape(-1, 256),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    next_states = {
        "compiled_scan": states_by_mode["rescan_1"],
        "rescan_1": states_by_mode["rescan_2"],
        "rescan_2": states_by_mode["rescan_3"],
        "sequential_feedback": feedback_states,
    }
    frontier = {
        "compiled_scan": 1,
        "rescan_1": 2,
        "rescan_2": 3,
        "sequential_feedback": base.TRAIN_HORIZONS,
    }

    metric_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    for name, _, feedback_rescans in MODES:
        logits = logits_by_mode[name]
        states = states_by_mode[name]
        token_nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        correct = logits.argmax(dim=-1).eq(targets)
        per_example_delta = token_nll.mean(dim=(1, 2)) - scan_nll.mean(
            dim=(1, 2)
        )
        delta_mean, delta_se, delta_low, delta_high = paired_interval(
            per_example_delta
        )
        raw_kl = forward_kl_rows(
            feedback_logits.double(),
            logits.double(),
        ).double()
        # KL is non-negative analytically; retain the raw minimum separately
        # and clamp only sub-ulp float32 cancellation in the aggregate metric.
        kl = raw_kl.clamp_min(0.0)
        cosine = F.cosine_similarity(
            states.double(),
            feedback_states.double(),
            dim=-1,
        )
        logit_rms, relative_logit_rms = centered_logit_rms(
            logits,
            feedback_logits,
        )
        tail = slice(3, None)
        tail_logit_rms, tail_relative_logit_rms = centered_logit_rms(
            logits[..., tail, :],
            feedback_logits[..., tail, :],
        )
        defect_states = next_states[name]
        exact_prefix = frontier[name]
        exact_frontier_error = float(
            (
                logits[..., :exact_prefix, :]
                - feedback_logits[..., :exact_prefix, :]
            ).abs().max()
        )
        row: dict[str, object] = {
            "checkpoint": checkpoint_name,
            "step": checkpoint_step,
            "mode": name,
            "feedback_rescans": feedback_rescans,
            "labels": targets.numel(),
            "block_nll": float(token_nll.double().mean()),
            "block_accuracy": float(correct.double().mean()),
            "paired_nll_delta_vs_scan": delta_mean,
            "paired_nll_delta_se": delta_se,
            "paired_nll_delta_ci95_low": delta_low,
            "paired_nll_delta_ci95_high": delta_high,
            "feedback_kl": float(kl.mean()),
            "min_raw_feedback_kl": float(raw_kl.min()),
            "tail_h4_h16_feedback_kl": float(kl[..., tail].mean()),
            "state_relative_rms": relative_rms(states, feedback_states),
            "tail_h4_h16_state_relative_rms": relative_rms(
                states[..., tail, :],
                feedback_states[..., tail, :],
            ),
            "state_cosine": float(cosine.mean()),
            "tail_h4_h16_state_cosine": float(cosine[..., tail].mean()),
            "centered_logit_rms": logit_rms,
            "centered_logit_relative_rms": relative_logit_rms,
            "tail_h4_h16_centered_logit_rms": tail_logit_rms,
            "tail_h4_h16_centered_logit_relative_rms": (
                tail_relative_logit_rms
            ),
            "top1_agreement_with_feedback": float(
                logits.argmax(-1).eq(feedback_logits.argmax(-1)).double().mean()
            ),
            "tail_h4_h16_top1_agreement_with_feedback": float(
                logits[..., tail, :]
                .argmax(-1)
                .eq(feedback_logits[..., tail, :].argmax(-1))
                .double()
                .mean()
            ),
            "fixed_point_defect_relative_rms": relative_rms(
                defect_states,
                states,
            ),
            "tail_h4_h16_fixed_point_defect_relative_rms": relative_rms(
                defect_states[..., tail, :],
                states[..., tail, :],
            ),
            "registered_exact_prefix_horizons": exact_prefix,
            "exact_prefix_max_abs_logit_error": exact_frontier_error,
            **timings[name],
        }
        metric_rows.append(row)

        for horizon in range(base.TRAIN_HORIZONS):
            horizon_states = states[..., horizon, :]
            horizon_feedback_states = feedback_states[..., horizon, :]
            horizon_logit_rms, horizon_relative_logit_rms = centered_logit_rms(
                logits[..., horizon, :],
                feedback_logits[..., horizon, :],
            )
            horizon_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "step": checkpoint_step,
                    "mode": name,
                    "feedback_rescans": feedback_rescans,
                    "horizon": horizon + 1,
                    "nll": float(token_nll[..., horizon].double().mean()),
                    "accuracy": float(correct[..., horizon].double().mean()),
                    "feedback_kl": float(kl[..., horizon].mean()),
                    "min_raw_feedback_kl": float(
                        raw_kl[..., horizon].min()
                    ),
                    "state_relative_rms": relative_rms(
                        horizon_states,
                        horizon_feedback_states,
                    ),
                    "state_cosine": float(cosine[..., horizon].mean()),
                    "centered_logit_rms": horizon_logit_rms,
                    "centered_logit_relative_rms": horizon_relative_logit_rms,
                    "top1_agreement_with_feedback": float(
                        logits[..., horizon, :]
                        .argmax(-1)
                        .eq(feedback_logits[..., horizon, :].argmax(-1))
                        .double()
                        .mean()
                    ),
                    "fixed_point_defect_relative_rms": relative_rms(
                        defect_states[..., horizon, :],
                        horizon_states,
                    ),
                    "max_abs_logit_error": float(
                        (
                            logits[..., horizon, :]
                            - feedback_logits[..., horizon, :]
                        ).abs().max()
                    ),
                }
            )
    return metric_rows, horizon_rows


def load_model(path: Path, expected_step: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != expected_step:
        raise RuntimeError(
            f"expected checkpoint step {expected_step}, got "
            f"{checkpoint['step']}: {path}"
        )
    model = base.make_model().cuda().eval()
    scan.configure_scan_backend(model, "fused-triton-rotating-frame")
    model.load_state_dict(checkpoint["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--timing-examples", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()
    if min(
        args.eval_examples,
        args.microbatch,
        args.timing_examples,
        args.timing_warmup,
        args.timing_repeats,
    ) < 1:
        parser.error("all count arguments must be positive")
    if args.eval_examples > 64:
        parser.error("at most 64 registered validation examples are available")
    if args.timing_examples > args.eval_examples:
        parser.error("timing examples cannot exceed evaluation examples")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    for _, path, _, expected_hash in CHECKPOINTS:
        if not path.exists():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"checkpoint checksum mismatch: {path}")

    validation = base.memmap("validation")
    registered_starts = base.load_fixed_starts(scan.BASELINE_STARTS, 64)
    starts = registered_starts[: args.eval_examples]
    windows = base.windows_of_length(
        validation,
        starts,
        base.EVAL_WINDOW_LENGTH,
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows: list[dict[str, object]] = []
    all_horizon_rows: list[dict[str, object]] = []
    for checkpoint_name, path, expected_step, _ in CHECKPOINTS:
        model = load_model(path, expected_step)
        logits_by_mode, states_by_mode, targets = collect_checkpoint_outputs(
            model,
            windows,
            microbatch=args.microbatch,
        )
        timings = benchmark_modes(
            model,
            windows[: args.timing_examples],
            warmup=args.timing_warmup,
            repeats=args.timing_repeats,
        )
        metric_rows, horizon_rows = summarize_checkpoint(
            checkpoint_name,
            expected_step,
            logits_by_mode,
            states_by_mode,
            targets,
            timings,
        )
        all_metric_rows.extend(metric_rows)
        all_horizon_rows.extend(horizon_rows)
        print(
            checkpoint_name,
            " ".join(
                f"{row['mode']}={row['block_nll']:.6f}"
                for row in metric_rows
            ),
            flush=True,
        )
        del model, logits_by_mode, states_by_mode, targets
        torch.cuda.empty_cache()

    write_rows(args.record_dir / "metrics.tsv", all_metric_rows)
    write_rows(args.record_dir / "horizon_metrics.tsv", all_horizon_rows)
    write_rows(
        args.record_dir / "run.tsv",
        [
            {"key": "experiment_id", "value": EXPERIMENT_ID},
            {"key": "seed", "value": base.SEED},
            {"key": "split", "value": "wikitext103_bytes_validation"},
            {"key": "eval_examples", "value": args.eval_examples},
            {"key": "labels_per_checkpoint", "value": int(args.eval_examples * 16 * 16)},
            {"key": "microbatch", "value": args.microbatch},
            {"key": "timing_examples", "value": args.timing_examples},
            {"key": "timing_warmup", "value": args.timing_warmup},
            {"key": "timing_repeats", "value": args.timing_repeats},
            {"key": "scan_backend", "value": "fused-triton-rotating-frame"},
            {"key": "precision", "value": "strict_float32_no_tf32"},
            {"key": "gpu", "value": torch.cuda.get_device_name()},
            {"key": "torch", "value": torch.__version__},
        ],
    )


if __name__ == "__main__":
    main()
