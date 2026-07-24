"""Train clean K^1..K^3 h_A targets on non-overlapping 3-token windows."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.training.ha_skew_window import (
    compounding_noise_orbit,
    forward_sparse_compounding_noise_window,
    forward_sparse_clean_window,
    relative_mse_rows,
    sparse_compounding_noise_window_loss,
    sparse_clean_window_loss,
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
from train_k1_ha_skew_learned_noise_mse_ce_13m import (
    HORIZONS as ROLLOUT_HORIZONS,
    INIT_SIGMA,
    SigmaPredictor,
    SkewOrthogonalOperator,
    check_extrapolation,
    windows_of_length,
)


EXPERIMENT_ID = "EXP-20260724-k3-ha-skew-clean-window3-mse-ce-13m"
WIDTH = 896
TRAIN_HORIZONS = 3
ANCHOR_STRIDE = 3
TRUNCATE_HORIZON_GRADIENT = False
DETACH_GOLD_TARGETS = False
USE_LEARNED_COMPOUNDING_NOISE = False
WINDOW_LENGTH = CONTEXT + TRAIN_HORIZONS
ANCHOR_COUNT = len(range(0, CONTEXT, ANCHOR_STRIDE))
DECISIONS_PER_EXAMPLE = ANCHOR_COUNT * TRAIN_HORIZONS
STEPS = 6000
EXTRAPOLATION_SAMPLE = 256
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
    model.operator = SkewOrthogonalOperator(WIDTH)
    if USE_LEARNED_COMPOUNDING_NOISE:
        model.sigma_predictor = SigmaPredictor(WIDTH)
    return model


def sampled_windows(
    data, batch: int, generator: torch.Generator
) -> torch.Tensor:
    starts = torch.randint(
        0, len(data) - WINDOW_LENGTH, (batch,), generator=generator
    )
    return windows_of_length(data, starts, WINDOW_LENGTH)


def orthogonality_error(model: K1DecoderAblationLM) -> float:
    weight = model.operator.weight.detach()
    identity = torch.eye(weight.shape[0], device=weight.device)
    return float((weight.T @ weight - identity).abs().max())


def experiment_window_loss(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    noise_generator: torch.Generator | None = None,
):
    if USE_LEARNED_COMPOUNDING_NOISE:
        return sparse_compounding_noise_window_loss(
            model,
            window,
            prefix_length=CONTEXT,
            horizons=TRAIN_HORIZONS,
            anchor_stride=ANCHOR_STRIDE,
            noise_generator=noise_generator,
        )
    return sparse_clean_window_loss(
        model,
        window,
        prefix_length=CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
        truncate_horizon_gradient=TRUNCATE_HORIZON_GRADIENT,
        detach_gold_targets=DETACH_GOLD_TARGETS,
    )


def experiment_window_forward(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    noise_generator: torch.Generator | None = None,
):
    if USE_LEARNED_COMPOUNDING_NOISE:
        return forward_sparse_compounding_noise_window(
            model,
            window,
            prefix_length=CONTEXT,
            horizons=TRAIN_HORIZONS,
            anchor_stride=ANCHOR_STRIDE,
            noise_generator=noise_generator,
        )
    return forward_sparse_clean_window(
        model,
        window,
        prefix_length=CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
    )


def evaluate(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    micro: int,
    operator_gradient_norm: float,
) -> dict[str, float]:
    model.eval()
    nll_sum = torch.zeros(TRAIN_HORIZONS, dtype=torch.float64)
    correct = torch.zeros(TRAIN_HORIZONS, dtype=torch.float64)
    mse_sum = torch.zeros(TRAIN_HORIZONS, dtype=torch.float64)
    cosine_sum = torch.zeros(TRAIN_HORIZONS, dtype=torch.float64)
    log_sigma_sum = 0.0
    log_sigma_count = 0
    log_sigma_min = math.inf
    log_sigma_max = -math.inf
    decisions = 0
    noise_generator = None
    if USE_LEARNED_COMPOUNDING_NOISE:
        noise_generator = torch.Generator(device="cuda").manual_seed(
            SEED + 5151
        )
    with torch.inference_mode():
        for begin in range(0, len(starts), micro):
            window = windows_of_length(
                validation,
                starts[begin : begin + micro],
                WINDOW_LENGTH,
            )
            output = experiment_window_forward(
                model,
                window,
                noise_generator=noise_generator,
            )
            if output.log_sigma is not None:
                log_sigma_sum += float(output.log_sigma.sum())
                log_sigma_count += output.log_sigma.numel()
                log_sigma_min = min(
                    log_sigma_min, float(output.log_sigma.min())
                )
                log_sigma_max = max(
                    log_sigma_max, float(output.log_sigma.max())
                )
            for horizon in range(TRAIN_HORIZONS):
                logits = output.logits[:, :, horizon].float()
                targets = output.targets[:, :, horizon]
                nll_sum[horizon] += float(
                    F.cross_entropy(
                        logits.flatten(0, 1),
                        targets.flatten(),
                        reduction="sum",
                    )
                )
                correct[horizon] += int(
                    logits.argmax(dim=-1).eq(targets).sum()
                )
                mse_sum[horizon] += float(
                    relative_mse_rows(
                        output.predicted_states[:, :, horizon],
                        output.gold_states[:, :, horizon],
                    ).sum()
                )
                cosine_sum[horizon] += float(
                    F.cosine_similarity(
                        output.predicted_states[:, :, horizon].float(),
                        output.gold_states[:, :, horizon].float(),
                        dim=-1,
                    ).sum()
                )
            decisions += window.shape[0] * ANCHOR_COUNT
    model.train()
    horizon_nll = nll_sum / decisions
    horizon_accuracy = correct / decisions
    horizon_mse = mse_sum / decisions
    horizon_cosine = cosine_sum / decisions
    metrics = {
        "joint_nll": float(horizon_nll.mean()),
        "joint_accuracy": float(horizon_accuracy.mean()),
        "combined_loss": float(
            horizon_nll.mean() + horizon_mse.mean()
        ),
        "operator_gradient_norm": operator_gradient_norm,
        "operator_orthogonality_error": orthogonality_error(model),
    }
    if log_sigma_count:
        metrics.update(
            {
                "log_sigma_mean": log_sigma_sum / log_sigma_count,
                "log_sigma_min": log_sigma_min,
                "log_sigma_max": log_sigma_max,
            }
        )
    for horizon in range(TRAIN_HORIZONS):
        suffix = horizon + 1
        metrics[f"h{suffix}_nll"] = float(horizon_nll[horizon])
        metrics[f"h{suffix}_accuracy"] = float(
            horizon_accuracy[horizon]
        )
        metrics[f"h{suffix}_state_relative_mse"] = float(
            horizon_mse[horizon]
        )
        metrics[f"h{suffix}_state_cosine"] = float(
            horizon_cosine[horizon]
        )
    return metrics


@torch.inference_mode()
def check_compounding_noise_extrapolation(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    micro: int = 32,
) -> dict[str, list[float]]:
    model.eval()
    direct_correct = [0] * ROLLOUT_HORIZONS
    hard_correct = [0] * ROLLOUT_HORIZONS
    total = 0
    noise_generator = torch.Generator(device="cuda").manual_seed(
        SEED + 6161
    )
    length = CONTEXT + ROLLOUT_HORIZONS
    for begin in range(0, len(starts), micro):
        window = windows_of_length(
            validation,
            starts[begin : begin + micro],
            length,
        )
        prefix = window[:, :CONTEXT]
        targets = window[:, CONTEXT:]
        batch = prefix.shape[0]
        positions = torch.arange(CONTEXT, device=prefix.device)
        encoded, _ = model.encode(prefix, positions)
        roots = encoded[:, -1:].contiguous()
        _, _, decode_states, _, _ = compounding_noise_orbit(
            model,
            roots,
            ROLLOUT_HORIZONS,
            noise_generator=noise_generator,
        )
        future = decode_states[:, 0]
        full_positions = torch.arange(
            CONTEXT + ROLLOUT_HORIZONS,
            device=prefix.device,
        ).unsqueeze(0).expand(batch, -1)
        latent = torch.cat((encoded, future), dim=1)
        decoded = model.encoder.decode_hidden(latent, full_positions)
        direct_prediction = model.token_logits(
            decoded[:, -ROLLOUT_HORIZONS:]
        ).float().argmax(dim=-1)
        for horizon in range(ROLLOUT_HORIZONS):
            direct_correct[horizon] += int(
                direct_prediction[:, horizon]
                .eq(targets[:, horizon])
                .sum()
            )

        rolling = prefix
        for horizon in range(ROLLOUT_HORIZONS):
            step_positions = torch.arange(
                rolling.shape[1], device=prefix.device
            )
            step_encoded, _ = model.encode(rolling, step_positions)
            clean_state = model.operator(step_encoded[:, -1])
            log_sigma = model.sigma_predictor(clean_state)
            scale = (
                clean_state.detach().norm(dim=-1, keepdim=True)
                / (clean_state.shape[-1] ** 0.5)
            )
            standard_noise = torch.randn(
                clean_state.shape,
                device=clean_state.device,
                dtype=clean_state.dtype,
                generator=noise_generator,
            )
            decode_state = (
                clean_state
                + torch.exp(log_sigma) * scale * standard_noise
            )
            step_latent = torch.cat(
                (step_encoded, decode_state[:, None]), dim=1
            )
            step_full_positions = torch.arange(
                rolling.shape[1] + 1, device=prefix.device
            ).unsqueeze(0).expand(batch, -1)
            step_decoded = model.encoder.decode_hidden(
                step_latent, step_full_positions
            )
            action = model.token_logits(
                step_decoded[:, -1]
            ).float().argmax(dim=-1)
            hard_correct[horizon] += int(
                action.eq(targets[:, horizon]).sum()
            )
            rolling = torch.cat((rolling, action[:, None]), dim=1)
        total += batch
    model.train()
    return {
        "direct": [correct / total for correct in direct_correct],
        "hard": [correct / total for correct in hard_correct],
    }


def experiment_extrapolation(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
) -> dict[str, list[float]]:
    if USE_LEARNED_COMPOUNDING_NOISE:
        return check_compounding_noise_extrapolation(
            model, validation, starts
        )
    clean = check_extrapolation(model, validation, starts)
    return {
        "direct": clean["direct_clean"],
        "hard": clean["hard_token"],
    }


def preflight(
    model: K1DecoderAblationLM, training, batch: int
) -> dict[str, float]:
    window = sampled_windows(
        training,
        min(batch, 2),
        torch.Generator().manual_seed(SEED + 4242),
    )
    noise_generator = None
    if USE_LEARNED_COMPOUNDING_NOISE:
        noise_generator = torch.Generator(device="cuda").manual_seed(
            SEED + 4243
        )
    loss, ce, mse, mse_by_horizon, output = experiment_window_loss(
        model,
        window,
        noise_generator=noise_generator,
    )
    if not bool(torch.isfinite(output.logits).all()):
        raise RuntimeError("preflight produced non-finite logits")
    ce_gradient = torch.autograd.grad(
        ce, model.operator.A, retain_graph=True
    )[0]
    mse_gradient_norms = []
    for horizon in range(TRAIN_HORIZONS):
        gradient = torch.autograd.grad(
            mse_by_horizon[horizon],
            model.operator.A,
            retain_graph=True,
        )[0]
        if not bool(torch.isfinite(gradient).all()) or not bool(
            gradient.norm() > 0
        ):
            raise RuntimeError(
                f"no finite MSE gradient reaches K at horizon {horizon + 1}"
            )
        mse_gradient_norms.append(float(gradient.norm()))
    target_gradient = torch.autograd.grad(
        mse,
        output.gold_states,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if DETACH_GOLD_TARGETS:
        if target_gradient is not None and bool(
            torch.count_nonzero(target_gradient)
        ):
            raise RuntimeError(
                "MSE gradient unexpectedly reaches detached gold states"
            )
        target_gradient_norm = 0.0
    else:
        if target_gradient is None:
            raise RuntimeError(
                "MSE gradient does not reach attached gold states"
            )
        target_gradient_norm = float(target_gradient.norm())
    for name, gradient in (("CE to K", ce_gradient),):
        if not bool(torch.isfinite(gradient).all()) or not bool(
            gradient.norm() > 0
        ):
            raise RuntimeError(f"preflight produced no finite {name} gradient")
    if not DETACH_GOLD_TARGETS and (
        not bool(torch.isfinite(target_gradient).all())
        or not bool(target_gradient.norm() > 0)
    ):
        raise RuntimeError(
            "preflight produced no finite MSE gradient to attached gold states"
        )
    sigma_ce_gradient_norm = 0.0
    sigma_mse_gradient_norm = 0.0
    if USE_LEARNED_COMPOUNDING_NOISE:
        sigma_parameter = model.sigma_predictor.linear.weight
        sigma_ce_gradient = torch.autograd.grad(
            ce,
            sigma_parameter,
            retain_graph=True,
        )[0]
        sigma_mse_gradient = torch.autograd.grad(
            mse,
            sigma_parameter,
            retain_graph=True,
        )[0]
        for name, gradient in (
            ("CE to sigma predictor", sigma_ce_gradient),
            ("MSE to sigma predictor", sigma_mse_gradient),
        ):
            if not bool(torch.isfinite(gradient).all()) or not bool(
                gradient.norm() > 0
            ):
                raise RuntimeError(
                    f"preflight produced no finite {name} gradient"
                )
        sigma_ce_gradient_norm = float(sigma_ce_gradient.norm())
        sigma_mse_gradient_norm = float(sigma_mse_gradient.norm())

    identity = torch.eye(WIDTH, device=window.device)
    identity_difference = float(
        (model.operator.weight.detach() - identity).abs().max()
    )
    initial_orthogonality = orthogonality_error(model)
    if identity_difference >= 1e-4 or initial_orthogonality >= 1e-4:
        raise RuntimeError("K is not identity/orthogonal at initialization")
    if (
        not USE_LEARNED_COMPOUNDING_NOISE
        and hasattr(model, "sigma_predictor")
    ):
        raise RuntimeError("clean model unexpectedly has a sigma predictor")
    if (
        USE_LEARNED_COMPOUNDING_NOISE
        and not hasattr(model, "sigma_predictor")
    ):
        raise RuntimeError("noise model has no sigma predictor")
    parametrized_modules = sum(
        int(hasattr(module, "parametrizations"))
        for module in model.modules()
    )
    if parametrized_modules:
        raise RuntimeError("clean model unexpectedly has parametrizations")

    full_positions = torch.arange(WINDOW_LENGTH, device=window.device)
    full_encoded, _ = model.encode(window, full_positions)
    prefix_encoded = full_encoded[:, :CONTEXT]
    probe_slot = min(1, output.anchor_indices.numel() - 1)
    probe_anchor = int(output.anchor_indices[probe_slot])
    probe_states = output.decode_states[:, probe_slot]
    literal_tape = torch.cat(
        (prefix_encoded[:, : probe_anchor + 1], probe_states), dim=1
    )
    literal_positions = torch.arange(
        probe_anchor + 1 + TRAIN_HORIZONS, device=window.device
    ).unsqueeze(0).expand(window.shape[0], -1)
    literal_hidden = model.encoder.decode_hidden(
        literal_tape, literal_positions
    )[:, -TRAIN_HORIZONS:]
    sparse_literal_difference = float(
        (
            output.decoded_hidden[:, probe_slot] - literal_hidden
        ).detach().abs().max()
    )
    if sparse_literal_difference >= 1e-4:
        raise RuntimeError(
            "sparse-anchor decode does not match a literal tape"
        )
    restored, _ = model.encoder.encode_hidden(
        model.encoder.decode_hidden(literal_tape, literal_positions),
        literal_positions,
    )
    inverse_roundtrip_relative_l2 = float(
        relative_mse_rows(
            restored, literal_tape
        ).detach().mean().sqrt()
    )
    if inverse_roundtrip_relative_l2 >= 1e-4:
        raise RuntimeError("exact inverse roundtrip failed")

    changed_future = output.decode_states.clone()
    changed_future[:, :, -1] += torch.randn_like(
        changed_future[:, :, -1]
    )
    changed_hidden = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded,
        changed_future,
        full_positions[:CONTEXT],
        output.anchor_indices,
    )
    future_causal_difference = float(
        (
            changed_hidden[:, :, :-1]
            - output.decoded_hidden[:, :, :-1]
        ).detach().abs().max()
    )
    if future_causal_difference >= 1e-5:
        raise RuntimeError(
            "later branch state changed an earlier output: "
            f"max_abs={future_causal_difference:.3e}"
        )

    values = {
        "initial_combined": float(loss.detach()),
        "initial_ce": float(ce.detach()),
        "initial_mse": float(mse.detach()),
        "initial_ce_k_gradient_norm": float(ce_gradient.norm()),
        "gold_target_gradient_norm": target_gradient_norm,
        "initial_sigma_ce_gradient_norm": sigma_ce_gradient_norm,
        "initial_sigma_mse_gradient_norm": sigma_mse_gradient_norm,
        "operator_identity_diff": identity_difference,
        "operator_orthogonality_error": initial_orthogonality,
        "head_self_retrieval": active_head_self_retrieval(model),
        "sparse_literal_decode_max_abs": sparse_literal_difference,
        "inverse_roundtrip_relative_l2": inverse_roundtrip_relative_l2,
        "future_causal_difference": future_causal_difference,
    }
    if output.log_sigma is not None:
        values.update(
            {
                "initial_log_sigma_mean": float(
                    output.log_sigma.detach().mean()
                ),
                "initial_log_sigma_min": float(
                    output.log_sigma.detach().min()
                ),
                "initial_log_sigma_max": float(
                    output.log_sigma.detach().max()
                ),
            }
        )
    for horizon, norm in enumerate(mse_gradient_norms, 1):
        values[f"h{horizon}_mse_k_gradient_norm"] = norm
    model.zero_grad(set_to_none=True)
    return values


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
                "anchor_stride": ANCHOR_STRIDE,
                "seed": SEED,
                "operator": "skew_matrix_exp_no_query",
                "noise": (
                    "learned_compounding_reparameterized"
                    if USE_LEARNED_COMPOUNDING_NOISE
                    else "none"
                ),
                "noise_predictor_input": (
                    "clean_K_power"
                    if USE_LEARNED_COMPOUNDING_NOISE
                    else "none"
                ),
                "spectral_constraint": "none",
                "target_gradient": (
                    "stop_gradient"
                    if DETACH_GOLD_TARGETS
                    else "attached"
                ),
                "detach_gold_targets": DETACH_GOLD_TARGETS,
                "truncate_horizon_gradient": (
                    TRUNCATE_HORIZON_GRADIENT
                ),
            },
        },
        path,
    )


def should_report(step: int, final_step: int) -> bool:
    return (
        step in (1, 50, 100)
        or step % 250 == 0
        or step == final_step
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-micro", type=int, default=8)
    parser.add_argument(
        "--extrapolation-examples",
        type=int,
        default=EXTRAPOLATION_SAMPLE,
    )
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch,
        args.eval_examples,
        args.eval_micro,
        args.extrapolation_examples,
    ) < 1:
        parser.error("steps and sample counts must be positive")
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
    extrapolation_starts = torch.randint(
        0,
        len(validation) - CONTEXT - ROLLOUT_HORIZONS - 1,
        (args.extrapolation_examples,),
        generator=torch.Generator().manual_seed(SEED + 7777),
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.record_dir / "validation_starts.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "token_start"))
        writer.writerows(enumerate(validation_starts.tolist()))

    model = make_model().cuda().train()
    preflight_values = preflight(model, training, args.batch)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"anchors={ANCHOR_COUNT} decisions/example={DECISIONS_PER_EXAMPLE} "
        f"initial_ce={preflight_values['initial_ce']:.4f} "
        f"initial_mse={preflight_values['initial_mse']:.4f}",
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
            ("batch", args.batch),
            ("context", CONTEXT),
            ("window_length", WINDOW_LENGTH),
            ("train_horizons", TRAIN_HORIZONS),
            ("anchor_stride", ANCHOR_STRIDE),
            ("anchor_count", ANCHOR_COUNT),
            ("decisions_per_example", DECISIONS_PER_EXAMPLE),
            ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"),
            ("operator", "skew_matrix_exp_no_query"),
            (
                "noise",
                "learned_compounding_reparameterized"
                if USE_LEARNED_COMPOUNDING_NOISE
                else "none",
            ),
            (
                "noise_predictor_input",
                "clean_K_power"
                if USE_LEARNED_COMPOUNDING_NOISE
                else "none",
            ),
            (
                "noise_draws",
                "independent_per_horizon"
                if USE_LEARNED_COMPOUNDING_NOISE
                else "none",
            ),
            (
                "init_sigma",
                INIT_SIGMA
                if USE_LEARNED_COMPOUNDING_NOISE
                else "none",
            ),
            ("spectral_constraint", "none"),
            (
                "target_gradient",
                "stop_gradient"
                if DETACH_GOLD_TARGETS
                else "attached",
            ),
            ("detach_gold_targets", DETACH_GOLD_TARGETS),
            (
                "truncate_horizon_gradient",
                TRUNCATE_HORIZON_GRADIENT,
            ),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    metric_names = (
        "joint_nll",
        "joint_accuracy",
        "combined_loss",
        "operator_gradient_norm",
        "operator_orthogonality_error",
        *tuple(
            f"h{h}_{metric}"
            for h in range(1, TRAIN_HORIZONS + 1)
            for metric in (
                "nll",
                "accuracy",
                "state_relative_mse",
                "state_cosine",
            )
        ),
    )
    if USE_LEARNED_COMPOUNDING_NOISE:
        metric_names = (
            *metric_names,
            "log_sigma_mean",
            "log_sigma_min",
            "log_sigma_max",
        )
    direct_rollout_label = (
        "direct_noisy"
        if USE_LEARNED_COMPOUNDING_NOISE
        else "direct_clean"
    )
    hard_rollout_label = (
        "hard_noisy_token"
        if USE_LEARNED_COMPOUNDING_NOISE
        else "hard_token"
    )
    fields = (
        "step",
        "train_combined",
        "train_ce",
        "train_mse",
        "train_h1_mse",
        "train_h2_mse",
        "train_h3_mse",
        *tuple(f"val_{name}" for name in metric_names),
        *tuple(
            f"{direct_rollout_label}_h{h}"
            for h in range(1, ROLLOUT_HORIZONS + 1)
        ),
        *tuple(
            f"{hard_rollout_label}_h{h}"
            for h in range(1, ROLLOUT_HORIZONS + 1)
        ),
        "lr",
        "wall_s",
        "tokens_per_s",
        "labels_per_s",
        "peak_vram_bytes",
    )
    data_generator = torch.Generator().manual_seed(SEED)
    training_noise_generator = None
    if USE_LEARNED_COMPOUNDING_NOISE:
        training_noise_generator = torch.Generator(
            device="cuda"
        ).manual_seed(SEED + 7171)
    best_nll = math.inf
    best_step = 0
    started = time.time()
    processed_tokens = 0
    processed_labels = 0
    torch.cuda.reset_peak_memory_stats()
    with (args.record_dir / "metrics.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t"
        )
        writer.writeheader()
        for index in range(args.steps):
            lr = lr_at(index, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(
                training, args.batch, data_generator
            )
            optimizer.zero_grad(set_to_none=True)
            loss, ce, mse, mse_by_horizon, _ = (
                experiment_window_loss(
                    model,
                    window,
                    noise_generator=training_noise_generator,
                )
            )
            loss.backward()
            operator_gradient_norm = float(
                model.operator.A.grad.float().norm()
            )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CLIP_NORM
            )
            optimizer.step()
            processed_tokens += args.batch * WINDOW_LENGTH
            processed_labels += args.batch * DECISIONS_PER_EXAMPLE
            step = index + 1
            if should_report(step, args.steps):
                metrics = evaluate(
                    model,
                    validation,
                    validation_starts,
                    args.eval_micro,
                    operator_gradient_norm,
                )
                extrapolation = experiment_extrapolation(
                    model, validation, extrapolation_starts
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    "train_combined": float(loss.detach()),
                    "train_ce": float(ce.detach()),
                    "train_mse": float(mse.detach()),
                    **{
                        f"train_h{h + 1}_mse": float(
                            mse_by_horizon[h].detach()
                        )
                        for h in range(TRAIN_HORIZONS)
                    },
                    **{
                        f"val_{name}": metrics[name]
                        for name in metric_names
                    },
                    **{
                        f"{direct_rollout_label}_h{h + 1}": value
                        for h, value in enumerate(
                            extrapolation["direct"]
                        )
                    },
                    **{
                        f"{hard_rollout_label}_h{h + 1}": value
                        for h, value in enumerate(
                            extrapolation["hard"]
                        )
                    },
                    "lr": lr,
                    "wall_s": wall,
                    "tokens_per_s": processed_tokens / wall,
                    "labels_per_s": processed_labels / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d} val_joint_nll="
                    f"{metrics['joint_nll']:.4f} "
                    f"acc[h1/h2/h3]="
                    f"{metrics['h1_accuracy']:.3f}/"
                    f"{metrics['h2_accuracy']:.3f}/"
                    f"{metrics['h3_accuracy']:.3f} "
                    f"mse[h1/h2/h3]="
                    f"{metrics['h1_state_relative_mse']:.4f}/"
                    f"{metrics['h2_state_relative_mse']:.4f}/"
                    f"{metrics['h3_state_relative_mse']:.4f} "
                    f"wall={wall:.1f}s",
                    flush=True,
                )
                print(
                    f"  h1-5 {direct_rollout_label}: "
                    + " ".join(
                        f"{value:.3f}"
                        for value in extrapolation["direct"]
                    ),
                    flush=True,
                )
                print(
                    f"  h1-5 {hard_rollout_label}:   "
                    + " ".join(
                        f"{value:.3f}"
                        for value in extrapolation["hard"]
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
                if step == 1000:
                    save_checkpoint(
                        args.output_dir / "step1000.pt",
                        model,
                        optimizer,
                        step,
                        metrics,
                    )
                if metrics["joint_nll"] < best_nll:
                    best_nll = metrics["joint_nll"]
                    best_step = step
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        step,
                        metrics,
                    )

    with (args.record_dir / "summary.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("best_step", best_step))
        writer.writerow(("best_val_joint_nll", best_nll))
        writer.writerow(("wall_s", time.time() - started))
        writer.writerow(
            ("peak_vram_bytes", torch.cuda.max_memory_allocated())
        )


if __name__ == "__main__":
    main()
