"""Train tied h1 CE and audit K powers against the model's own greedy AR states."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from rotlm.evaluation import spectral_block_boundary_rollouts
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.spectral_collapse import InputConditionedSpectralCollapse
from rotlm.training.ha_skew_window import (
    SpectralCollapseWindowOutput,
    forward_sparse_spectral_collapse_window,
    relative_mse_rows,
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
    "EXP-20260801-k4-input-conditioned-kpower-h1-ce-only-"
    "tied-ar-closure-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
WIDTH = 896
HORIZONS = 4
TRAIN_HORIZONS = 1
TRAIN_ANCHOR_STRIDE = 1
EVAL_ANCHOR_STRIDE = 4
TRAIN_WINDOW_LENGTH = CONTEXT + TRAIN_HORIZONS
EVAL_WINDOW_LENGTH = CONTEXT + HORIZONS
TRAIN_ANCHORS = len(range(0, CONTEXT, TRAIN_ANCHOR_STRIDE))
CONTROLLER_BOTTLENECK = 128
INITIAL_FREQUENCY_RANGE = 0.05
MAX_PHASE_DELTA = 0.10
INITIAL_RADIUS = 0.99
MAX_DECAY_LOGIT_DELTA = 2.0
ALPHA = 0.0
CALIBRATION_BATCH = 16
CALIBRATION_SEED = SEED + 9100
EFFECTIVE_BATCH = 64
MICROBATCH = 16
STEPS = 1000
SCHEDULE_STEPS = 6000
REPORT_STEPS = frozenset((1, 50, 100, 250, 500, 750, 1000))


def make_model() -> K1DecoderAblationLM:
    model = K1DecoderAblationLM(
        VOCABULARY,
        width=WIDTH,
        encoder_blocks=2,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
    )
    # The experiment has one input-conditioned K implementation only.
    model.operator = nn.Identity()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        WIDTH,
        bottleneck=CONTROLLER_BOTTLENECK,
        initial_frequency_range=INITIAL_FREQUENCY_RANGE,
        max_phase_delta=MAX_PHASE_DELTA,
        initial_radius=INITIAL_RADIUS,
        max_decay_logit_delta=MAX_DECAY_LOGIT_DELTA,
        alpha=ALPHA,
    )
    # Required by the shared interface but mathematically inactive at alpha 0.
    model.spectral_collapse.shell_raw.requires_grad_(False)
    return model


def sampled_windows(
    data,
    batch: int,
    generator: torch.Generator,
    length: int,
) -> torch.Tensor:
    starts = torch.randint(
        0,
        len(data) - length,
        (batch,),
        generator=generator,
    )
    return windows_of_length(data, starts, length)


@torch.no_grad()
def calibrate_interface(
    model: K1DecoderAblationLM,
    training,
) -> torch.Tensor:
    starts = torch.randint(
        0,
        len(training) - TRAIN_WINDOW_LENGTH,
        (CALIBRATION_BATCH,),
        generator=torch.Generator().manual_seed(CALIBRATION_SEED),
    )
    tokens = windows_of_length(training, starts, CONTEXT)
    encoded, _ = model.encode(tokens)
    model.spectral_collapse.calibrate_shell(encoded)
    return starts


def objective(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
) -> tuple[torch.Tensor, SpectralCollapseWindowOutput]:
    output = forward_sparse_spectral_collapse_window(
        model,
        window,
        prefix_length=CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=TRAIN_ANCHOR_STRIDE,
    )
    # This is deliberately the only optimization signal.
    return output.token_ce[..., 0].mean(), output


def _require_finite_nonzero(
    name: str,
    gradient: torch.Tensor | None,
) -> None:
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.norm() > 0)
    ):
        raise RuntimeError(f"no finite nonzero CE gradient to {name}")


def basis_orthogonality_error(model: K1DecoderAblationLM) -> float:
    basis = model.spectral_collapse.basis.detach().float()
    identity = torch.eye(WIDTH, device=basis.device)
    return float((basis.T @ basis - identity).abs().max())


def preflight(
    model: K1DecoderAblationLM,
    training,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 9242)
    train_window = sampled_windows(
        training,
        2,
        generator,
        TRAIN_WINDOW_LENGTH,
    )
    loss, train_output = objective(model, train_window)
    expected = (2, TRAIN_ANCHORS, 1)
    if train_output.token_ce.shape != expected:
        raise RuntimeError(
            f"unexpected training CE shape {tuple(train_output.token_ce.shape)}"
        )
    torch.testing.assert_close(loss, train_output.token_ce[..., 0].mean())
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("initial CE is non-finite")

    if model.decoder_mode != "exact-inverse" or model.codebook_head is not None:
        raise RuntimeError("model is not exact-inverse with RMS-tied codebook")
    vocabulary_linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == VOCABULARY
    ]
    if vocabulary_linears:
        raise RuntimeError(
            "independent vocabulary projections found: "
            + ", ".join(vocabulary_linears)
        )
    embedding_tables = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding)
    ]
    if embedding_tables != ["encoder.embed"]:
        raise RuntimeError(f"unexpected embedding tables: {embedding_tables}")

    eval_window = sampled_windows(
        training,
        2,
        generator,
        EVAL_WINDOW_LENGTH,
    )
    parallel = forward_sparse_spectral_collapse_window(
        model,
        eval_window,
        prefix_length=CONTEXT,
        horizons=HORIZONS,
        anchor_stride=EVAL_ANCHOR_STRIDE,
    )
    roots = parallel.prefix_encoded.index_select(
        1,
        parallel.anchor_indices,
    )
    literal = model.spectral_collapse.literal_rollout(roots, HORIZONS)
    recurrence_error = float(
        (parallel.predicted_states - literal).detach().abs().max()
    )
    raw_state_error = float(
        (
            parallel.predicted_states - parallel.unprojected_states
        ).detach().abs().max()
    )
    if max(recurrence_error, raw_state_error) >= 2e-5:
        raise RuntimeError("alpha-zero states are not literal K powers")

    rho_variance = float(
        parallel.rollout.rho.detach().float().var(dim=(0, 1)).mean()
    )
    phase_variance = float(
        parallel.rollout.phase_delta.detach().float().var(
            dim=(0, 1)
        ).mean()
    )
    if min(rho_variance, phase_variance) <= 0:
        raise RuntimeError("generated K does not vary across input roots")

    changed_future = eval_window.clone()
    changed_future[:, CONTEXT:] = (
        changed_future[:, CONTEXT:] + 1
    ) % VOCABULARY
    with torch.no_grad():
        changed = forward_sparse_spectral_collapse_window(
            model,
            changed_future,
            prefix_length=CONTEXT,
            horizons=HORIZONS,
            anchor_stride=EVAL_ANCHOR_STRIDE,
        )
    future_state_error = float(
        (
            parallel.predicted_states.detach()
            - changed.predicted_states
        ).abs().max()
    )
    future_logit_error = float(
        (parallel.logits.detach() - changed.logits).abs().max()
    )
    if max(future_state_error, future_logit_error) >= 1e-6:
        raise RuntimeError("held-out future tokens leaked into predictions")

    target_gradient = torch.autograd.grad(
        loss,
        train_output.gold_states,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if target_gradient is not None and bool(target_gradient.norm() > 0):
        raise RuntimeError("CE unexpectedly supervises future hidden states")

    spectral = model.spectral_collapse
    ce_parameters = (
        spectral.basis_generator,
        spectral.context.weight,
        spectral.phase.weight,
        spectral.decay.weight,
        model.encoder.blocks[0].attn.qkv.weight,
        model.embedding_weight,
    )
    ce_gradients = torch.autograd.grad(
        loss,
        ce_parameters,
        retain_graph=True,
    )
    for name, gradient in zip(
        (
            "spectral basis",
            "K controller",
            "conditional phase",
            "conditional radius",
            "reversible encoder/decoder",
            "tied embedding",
        ),
        ce_gradients,
    ):
        _require_finite_nonzero(name, gradient)

    matched = spectral_block_boundary_rollouts(
        model,
        eval_window[:1],
        prefix_length=CONTEXT,
        horizons=HORIZONS,
        anchor_stride=EVAL_ANCHOR_STRIDE,
    )
    h1_logit_error = float(
        (
            matched.boundary_ar_logits[:, :, 0]
            - matched.parallel_logits[:, :, 0]
        ).abs().max()
    )
    if h1_logit_error >= 2e-4:
        raise RuntimeError("parallel and AR paths do not share the first step")

    self_retrieval = active_head_self_retrieval(model)
    if self_retrieval < 0.999:
        raise RuntimeError("tied codebook does not retrieve its embedding rows")

    values = {
        "initial_ce": float(loss.detach()),
        "closed_form_recurrence_error": recurrence_error,
        "alpha_zero_raw_state_error": raw_state_error,
        "future_state_leak_error": future_state_error,
        "future_logit_leak_error": future_logit_error,
        "future_hidden_target_gradient_norm": (
            0.0
            if target_gradient is None
            else float(target_gradient.norm())
        ),
        "initial_rho_prefix_variance": rho_variance,
        "initial_phase_prefix_variance": phase_variance,
        "h1_ar_parallel_max_logit_error": h1_logit_error,
        "basis_orthogonality_error": basis_orthogonality_error(model),
        "tied_codebook_self_retrieval": self_retrieval,
        "ce_basis_gradient_norm": float(ce_gradients[0].norm()),
        "ce_controller_gradient_norm": float(ce_gradients[1].norm()),
        "ce_phase_gradient_norm": float(ce_gradients[2].norm()),
        "ce_radius_gradient_norm": float(ce_gradients[3].norm()),
        "ce_encoder_gradient_norm": float(ce_gradients[4].norm()),
        "ce_embedding_gradient_norm": float(ce_gradients[5].norm()),
    }
    model.zero_grad(set_to_none=True)
    return values


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


@torch.inference_mode()
def evaluate(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    microbatch: int,
) -> dict[str, float]:
    """Compare frozen initial K powers with self-generated greedy AR states."""
    model.eval()
    cosine_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    relative_mse_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    cosine_099_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    token_agreement_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    forward_kl_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    h1_nll_sum = 0.0
    h1_correct = 0
    exact_blocks = 0
    decisions = 0
    h1_max_logit_error = 0.0

    for begin in range(0, len(starts), microbatch):
        window = windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            EVAL_WINDOW_LENGTH,
        )
        output = spectral_block_boundary_rollouts(
            model,
            window,
            prefix_length=CONTEXT,
            horizons=HORIZONS,
            anchor_stride=EVAL_ANCHOR_STRIDE,
        )
        parallel_states = output.parallel_states.float()
        ar_states = output.boundary_ar_states.float()
        cosine = F.cosine_similarity(parallel_states, ar_states, dim=-1)
        state_relative_mse = relative_mse_rows(
            parallel_states,
            ar_states,
        )
        matches = output.parallel_tokens.eq(output.boundary_ar_tokens)
        count = window.shape[0] * output.anchor_indices.numel()
        decisions += count
        cosine_sum += cosine.double().sum(dim=(0, 1)).cpu()
        relative_mse_sum += state_relative_mse.double().sum(
            dim=(0, 1)
        ).cpu()
        cosine_099_sum += cosine.ge(0.99).double().sum(dim=(0, 1)).cpu()
        token_agreement_sum += matches.double().sum(dim=(0, 1)).cpu()
        exact_blocks += int(matches.all(dim=-1).sum())

        for horizon in range(HORIZONS):
            ar_log_probs = F.log_softmax(
                output.boundary_ar_logits[:, :, horizon],
                dim=-1,
            )
            parallel_log_probs = F.log_softmax(
                output.parallel_logits[:, :, horizon],
                dim=-1,
            )
            forward_kl = (
                ar_log_probs.exp()
                * (ar_log_probs - parallel_log_probs)
            ).sum(dim=-1)
            forward_kl_sum[horizon] += float(forward_kl.sum())

        h1_targets = output.targets[:, :, 0]
        h1_nll_sum += float(
            F.cross_entropy(
                output.parallel_logits[:, :, 0].flatten(0, 1),
                h1_targets.flatten(),
                reduction="sum",
            )
        )
        h1_correct += int(
            output.parallel_tokens[:, :, 0].eq(h1_targets).sum()
        )
        h1_max_logit_error = max(
            h1_max_logit_error,
            float(
                (
                    output.boundary_ar_logits[:, :, 0]
                    - output.parallel_logits[:, :, 0]
                ).abs().max()
            ),
        )

    if decisions < 1:
        raise RuntimeError("evaluation produced no block boundaries")
    horizon_cosine = cosine_sum / decisions
    horizon_relative_mse = relative_mse_sum / decisions
    horizon_cosine_099 = cosine_099_sum / decisions
    horizon_token_agreement = token_agreement_sum / decisions
    horizon_forward_kl = forward_kl_sum / decisions
    cosine_values = [float(value) for value in horizon_cosine]
    relative_mse_values = [float(value) for value in horizon_relative_mse]
    metrics = {
        "h1_validation_nll": h1_nll_sum / decisions,
        "h1_validation_accuracy": h1_correct / decisions,
        "mean_state_cosine": sum(cosine_values) / HORIZONS,
        "median_state_cosine": _median(cosine_values),
        "mean_state_relative_mse": sum(relative_mse_values) / HORIZONS,
        "median_state_relative_mse": _median(relative_mse_values),
        "mean_token_agreement": float(horizon_token_agreement.mean()),
        "h2_h4_token_agreement": float(horizon_token_agreement[1:].mean()),
        "exact_block_token_agreement": exact_blocks / decisions,
        "h1_max_logit_error": h1_max_logit_error,
        "basis_orthogonality_error": basis_orthogonality_error(model),
        "evaluated_boundaries": float(decisions),
    }
    for horizon in range(HORIZONS):
        slot = horizon + 1
        metrics[f"h{slot}_state_cosine"] = float(horizon_cosine[horizon])
        metrics[f"h{slot}_state_relative_mse"] = float(
            horizon_relative_mse[horizon]
        )
        metrics[f"h{slot}_state_cosine_ge_099"] = float(
            horizon_cosine_099[horizon]
        )
        metrics[f"h{slot}_token_agreement"] = float(
            horizon_token_agreement[horizon]
        )
        metrics[f"h{slot}_ar_to_parallel_kl"] = float(
            horizon_forward_kl[horizon]
        )
    model.train()
    return metrics


METRIC_NAMES = (
    "h1_validation_nll",
    "h1_validation_accuracy",
    "mean_state_cosine",
    "median_state_cosine",
    "mean_state_relative_mse",
    "median_state_relative_mse",
    "mean_token_agreement",
    "h2_h4_token_agreement",
    "exact_block_token_agreement",
    "h1_max_logit_error",
    "basis_orthogonality_error",
    "evaluated_boundaries",
    *tuple(
        f"h{horizon}_{metric}"
        for horizon in range(1, HORIZONS + 1)
        for metric in (
            "state_cosine",
            "state_relative_mse",
            "state_cosine_ge_099",
            "token_agreement",
            "ar_to_parallel_kl",
        )
    ),
)


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
                "train_horizons": TRAIN_HORIZONS,
                "monitor_horizons": HORIZONS,
                "train_anchor_stride": TRAIN_ANCHOR_STRIDE,
                "eval_anchor_stride": EVAL_ANCHOR_STRIDE,
                "head": "rms_tied_embedding_no_independent_lm_head",
                "decoder": "analytic_exact_inverse",
                "operator": "input_conditioned_block_frozen_real_normal",
                "alpha": ALPHA,
                "objective": "h1_token_ce_only",
                "seed": SEED,
            },
        },
        path,
    )


def closure_result(
    rows: list[dict[str, float]],
    final_step: int,
) -> tuple[bool | None, dict[str, float]]:
    by_step = {int(row["step"]): row for row in rows}
    required = (0, 500, 750, 1000)
    if final_step != 1000 or any(step not in by_step for step in required):
        return None, {}
    initial = by_step[0]
    final = by_step[1000]
    cosine_improved = sum(
        final[f"val_h{h}_state_cosine"]
        > initial[f"val_h{h}_state_cosine"]
        for h in range(1, HORIZONS + 1)
    )
    mse_improved = sum(
        final[f"val_h{h}_state_relative_mse"]
        < initial[f"val_h{h}_state_relative_mse"]
        for h in range(1, HORIZONS + 1)
    )
    baseline_median = initial["val_median_state_cosine"]
    late_above_baseline = all(
        by_step[step]["val_median_state_cosine"] > baseline_median
        for step in (500, 750, 1000)
    )
    final_gain = final["val_median_state_cosine"] - baseline_median
    details = {
        "cosine_horizons_improved": float(cosine_improved),
        "relative_mse_horizons_improved": float(mse_improved),
        "late_reports_above_initial_median_cosine": float(
            late_above_baseline
        ),
        "final_median_cosine_gain": final_gain,
    }
    success = (
        cosine_improved >= 3
        and mse_improved >= 3
        and late_above_baseline
        and final_gain >= 0.01
    )
    return success, details


def write_interpretation(
    path: Path,
    rows: list[dict[str, float]],
    success: bool | None,
    details: dict[str, float],
) -> None:
    initial = rows[0]
    final = rows[-1]
    if success is None:
        verdict = "Smoke/partial run: the preregistered criterion was not evaluated."
    elif success:
        verdict = (
            "The preregistered greedy-AR closure-improvement criterion passed."
        )
    else:
        verdict = (
            "The preregistered greedy-AR closure-improvement criterion failed; "
            "the byte-level follow-up is required."
        )
    lines = [
        "# Interpretation",
        "",
        verdict,
        "",
        "The state reference is the model's own greedy AR generation followed "
        "by canonical re-encoding. No teacher-forced future hidden state is "
        "used in these closure metrics.",
        "",
        "| Metric | Step 0 | Final |",
        "|---|---:|---:|",
        (
            "| Median state cosine | "
            f"{initial['val_median_state_cosine']:.6f} | "
            f"{final['val_median_state_cosine']:.6f} |"
        ),
        (
            "| Median state relative MSE | "
            f"{initial['val_median_state_relative_mse']:.6f} | "
            f"{final['val_median_state_relative_mse']:.6f} |"
        ),
        (
            "| H2--H4 token agreement | "
            f"{initial['val_h2_h4_token_agreement']:.6f} | "
            f"{final['val_h2_h4_token_agreement']:.6f} |"
        ),
        (
            "| H1 validation NLL | "
            f"{initial['val_h1_validation_nll']:.6f} | "
            f"{final['val_h1_validation_nll']:.6f} |"
        ),
        "",
    ]
    if details:
        lines.extend(
            [
                "Preregistered criterion components:",
                "",
                *[
                    f"- {key}: {value}"
                    for key, value in details.items()
                ],
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=EFFECTIVE_BATCH)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    parser.add_argument("--schedule-steps", type=int, default=SCHEDULE_STEPS)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=4)
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
        parser.error("all counts must be positive")
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
        len(validation) - EVAL_WINDOW_LENGTH,
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
    calibration_starts = calibrate_interface(model, training)
    with (args.record_dir / "calibration_starts.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "token_start"))
        writer.writerows(enumerate(calibration_starts.tolist()))

    preflight_values = preflight(model, training)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    accumulation_steps = args.batch // args.microbatch
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"batch={args.batch} micro={args.microbatch}x{accumulation_steps} "
        f"initial_ce={preflight_values['initial_ce']:.4f}",
        flush=True,
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
            ("training_horizons", TRAIN_HORIZONS),
            ("monitor_horizons", HORIZONS),
            ("training_anchor_stride", TRAIN_ANCHOR_STRIDE),
            ("monitor_anchor_stride", EVAL_ANCHOR_STRIDE),
            ("validation_examples", args.eval_examples),
            ("validation_seed", SEED + 999),
            ("vocabulary", VOCABULARY),
            ("width", WIDTH),
            ("encoder_blocks", 2),
            ("decoder", "analytic_exact_inverse"),
            ("head", "rms_tied_embedding_no_independent_lm_head"),
            ("objective", "h1_token_ce_only"),
            ("latent_regression_weight", 0.0),
            ("operator", "input_conditioned_block_frozen_real_normal"),
            ("controller_bottleneck", CONTROLLER_BOTTLENECK),
            ("initial_radius", INITIAL_RADIUS),
            ("alpha", ALPHA),
            ("precision", "strict_float32_no_tf32"),
            ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    fields = (
        "step",
        "train_loss",
        "train_rho_mean",
        "train_phase_delta_abs_mean",
        "gradient_norm",
        *tuple(f"val_{name}" for name in METRIC_NAMES),
        "lr",
        "wall_s",
        "unique_tokens_per_s",
        "ce_labels_per_s",
        "peak_vram_bytes",
    )
    data_generator = torch.Generator().manual_seed(SEED)
    started = time.time()
    processed_unique_tokens = 0
    processed_labels = 0
    rows: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats()

    with (args.record_dir / "metrics.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()

        initial_metrics = evaluate(
            model,
            validation,
            validation_starts,
            args.eval_microbatch,
        )
        initial_row = {
            "step": 0,
            "train_loss": math.nan,
            "train_rho_mean": math.nan,
            "train_phase_delta_abs_mean": math.nan,
            "gradient_norm": math.nan,
            **{f"val_{name}": initial_metrics[name] for name in METRIC_NAMES},
            "lr": 0.0,
            "wall_s": time.time() - started,
            "unique_tokens_per_s": 0.0,
            "ce_labels_per_s": 0.0,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        writer.writerow(initial_row)
        handle.flush()
        rows.append(initial_row)
        print(
            "step=   0 "
            f"val_h1_nll={initial_metrics['h1_validation_nll']:.4f} "
            f"state_cos={initial_metrics['median_state_cosine']:.4f} "
            f"state_rel_mse={initial_metrics['median_state_relative_mse']:.4f} "
            f"tok_h2-h4={initial_metrics['h2_h4_token_agreement']:.3f}",
            flush=True,
        )

        for index in range(args.steps):
            lr = lr_at(index, args.schedule_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(
                training,
                args.batch,
                data_generator,
                TRAIN_WINDOW_LENGTH,
            )
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            train_rho_mean = 0.0
            train_phase_delta_abs_mean = 0.0
            micro_windows = window.split(args.microbatch)
            for micro_window in micro_windows:
                loss, output = objective(model, micro_window)
                (loss / len(micro_windows)).backward()
                scale = 1.0 / len(micro_windows)
                train_loss += float(loss.detach()) * scale
                train_rho_mean += float(output.rollout.rho.detach().mean()) * scale
                train_phase_delta_abs_mean += (
                    float(output.rollout.phase_delta.detach().abs().mean())
                    * scale
                )
                del loss, output
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            )
            optimizer.step()
            processed_unique_tokens += args.batch * TRAIN_WINDOW_LENGTH
            processed_labels += args.batch * TRAIN_ANCHORS
            step = index + 1

            if step in REPORT_STEPS or step == args.steps:
                metrics = evaluate(
                    model,
                    validation,
                    validation_starts,
                    args.eval_microbatch,
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "train_rho_mean": train_rho_mean,
                    "train_phase_delta_abs_mean": train_phase_delta_abs_mean,
                    "gradient_norm": gradient_norm,
                    **{f"val_{name}": metrics[name] for name in METRIC_NAMES},
                    "lr": lr,
                    "wall_s": wall,
                    "unique_tokens_per_s": processed_unique_tokens / wall,
                    "ce_labels_per_s": processed_labels / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(
                    f"step={step:4d} train_ce={train_loss:.4f} "
                    f"val_h1_nll={metrics['h1_validation_nll']:.4f} "
                    f"state_cos={metrics['median_state_cosine']:.4f} "
                    f"state_rel_mse={metrics['median_state_relative_mse']:.4f} "
                    f"tok_h2-h4={metrics['h2_h4_token_agreement']:.3f}",
                    flush=True,
                )
                save_checkpoint(
                    args.output_dir / "last.pt",
                    model,
                    optimizer,
                    step,
                    metrics,
                )
                if step in (500, 1000) or step == args.steps:
                    save_checkpoint(
                        args.output_dir / f"step{step:04d}.pt",
                        model,
                        optimizer,
                        step,
                        metrics,
                    )

    success, criterion_details = closure_result(rows, args.steps)
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("final_step", args.steps))
        writer.writerow(
            (
                "closure_criterion",
                "not_evaluated" if success is None else str(success).lower(),
            )
        )
        writer.writerow(("byte_followup_required", success is False))
        writer.writerow(("wall_s", time.time() - started))
        writer.writerow(("peak_vram_bytes", torch.cuda.max_memory_allocated()))
        for key, value in criterion_details.items():
            writer.writerow((key, value))
    write_interpretation(
        args.record_dir / "interpretation.md",
        rows,
        success,
        criterion_details,
    )
    if success is not None:
        print(
            "closure_criterion=" + ("passed" if success else "failed"),
            flush=True,
        )


if __name__ == "__main__":
    main()
