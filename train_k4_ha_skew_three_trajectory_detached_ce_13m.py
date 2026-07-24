"""Train a three-sample, four-token latent trajectory mixture."""
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
    MultiTrajectoryWindowOutput,
    relative_mse_rows,
    sparse_multi_trajectory_compounding_noise_loss,
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
    INIT_SIGMA,
    SigmaPredictor,
    SkewOrthogonalOperator,
    windows_of_length,
)


EXPERIMENT_ID = (
    "EXP-20260724-k4-ha-skew-stride1-batch64-"
    "three-trajectory-detached-ce-13m"
)
WIDTH = 896
HORIZONS = 4
TRAJECTORIES = 3
TEMPERATURE = 1.0
MSE_WEIGHT = 0.0
MSE_HORIZONS: tuple[int, ...] | None = None
DETACH_MSE_TARGETS = False
ANCHOR_STRIDE = 1
WINDOW_LENGTH = CONTEXT + HORIZONS
ANCHOR_COUNT = len(range(0, CONTEXT, ANCHOR_STRIDE))
EFFECTIVE_BATCH = 64
MICROBATCH = 16
STEPS = 1000
SCHEDULE_STEPS = 6000
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
    model.sigma_predictor = SigmaPredictor(WIDTH)
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


def orthogonality_error(model: K1DecoderAblationLM) -> float:
    weight = model.operator.weight.detach()
    identity = torch.eye(weight.shape[0], device=weight.device)
    return float((weight.T @ weight - identity).abs().max())


def objective(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
    *,
    noise_generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MultiTrajectoryWindowOutput,
]:
    return sparse_multi_trajectory_compounding_noise_loss(
        model,
        window,
        prefix_length=CONTEXT,
        horizons=HORIZONS,
        anchor_stride=ANCHOR_STRIDE,
        trajectories=TRAJECTORIES,
        temperature=TEMPERATURE,
        mse_weight=MSE_WEIGHT,
        mse_horizons=MSE_HORIZONS,
        detach_mse_targets=DETACH_MSE_TARGETS,
        noise_generator=noise_generator,
    )


def responsibility_metrics(
    output: MultiTrajectoryWindowOutput,
) -> dict[str, float]:
    responsibilities = output.responsibilities.float()
    entropy = -(
        responsibilities
        * responsibilities.clamp_min(1e-12).log()
    ).sum(dim=1)
    sorted_ce = (
        output.trajectory_ce.detach().float().sort(dim=1).values
    )
    return {
        "posterior_ce": float(output.posterior_ce.detach()),
        "path_mean_ce": float(
            output.trajectory_ce.detach().mean() / HORIZONS
        ),
        "monitor_mse": (
            math.nan
            if output.trajectory_mse is None
            else float(
                (
                    responsibilities
                    * output.trajectory_mse.detach().float()
                ).sum(dim=1).mean()
            )
        ),
        "responsibility_max": float(
            responsibilities.max(dim=1).values.mean()
        ),
        "responsibility_entropy": float(entropy.mean()),
        "effective_trajectories": float(entropy.exp().mean()),
        "score_gap": float(
            (sorted_ce[:, 1] - sorted_ce[:, 0]).mean()
        ),
        "sigma_mean": float(
            output.sparse.log_sigma.detach().float().exp().mean()
        ),
        "log_sigma_mean": float(
            output.sparse.log_sigma.detach().float().mean()
        ),
    }


def preflight(
    model: K1DecoderAblationLM,
    training,
) -> dict[str, float]:
    window = sampled_windows(
        training,
        2,
        torch.Generator().manual_seed(SEED + 4242),
    )
    noise_generator = torch.Generator(device="cuda").manual_seed(
        SEED + 4243
    )
    loss, marginal_ce, monitor_mse, output = objective(
        model,
        window,
        noise_generator=noise_generator,
    )
    sparse = output.sparse
    expected_shape = (
        2,
        TRAJECTORIES,
        ANCHOR_COUNT,
        HORIZONS,
    )
    if output.token_ce.shape != expected_shape:
        raise RuntimeError(
            f"unexpected token CE shape {output.token_ce.shape}"
        )
    if output.trajectory_ce.shape != expected_shape[:-1]:
        raise RuntimeError("unexpected trajectory CE shape")
    if output.responsibilities.requires_grad:
        raise RuntimeError("trajectory responsibilities are not detached")
    if not bool(
        torch.allclose(
            output.trajectory_ce,
            output.token_ce.sum(dim=-1),
        )
    ):
        raise RuntimeError("trajectory score is not four-token CE sum")
    if not bool(
        torch.allclose(
            output.responsibilities.sum(dim=1),
            torch.ones_like(output.responsibilities[:, 0]),
        )
    ):
        raise RuntimeError("trajectory responsibilities do not sum to one")
    if not bool(torch.isfinite(sparse.logits).all()):
        raise RuntimeError("preflight produced non-finite logits")

    k_gradient, sigma_gradient, target_gradient = torch.autograd.grad(
        loss,
        (
            model.operator.A,
            model.sigma_predictor.linear.weight,
            sparse.gold_states,
        ),
        retain_graph=True,
        allow_unused=True,
    )
    for name, gradient in (
        ("objective to K", k_gradient),
        ("objective to sigma predictor", sigma_gradient),
    ):
        if (
            gradient is None
            or not bool(torch.isfinite(gradient).all())
            or not bool(gradient.norm() > 0)
        ):
            raise RuntimeError(f"no finite {name} gradient")
    selected_mse_horizons = (
        tuple(range(1, HORIZONS + 1))
        if MSE_HORIZONS is None
        else MSE_HORIZONS
    )
    target_gradient_by_horizon = torch.zeros(
        HORIZONS,
        device=window.device,
    )
    if target_gradient is not None:
        target_gradient_by_horizon = (
            target_gradient.float()
            .square()
            .sum(dim=(0, 1, 3))
            .sqrt()
        )
    target_should_be_attached = (
        MSE_WEIGHT != 0.0 and not DETACH_MSE_TARGETS
    )
    for horizon in range(1, HORIZONS + 1):
        has_gradient = bool(
            target_gradient_by_horizon[horizon - 1] > 0
        )
        expected = (
            target_should_be_attached
            and horizon in selected_mse_horizons
        )
        if has_gradient != expected:
            raise RuntimeError(
                "unexpected online MSE target gradient at "
                f"horizon {horizon}: expected={expected}"
            )
    _, _, optional_mse, optional_output = (
        sparse_multi_trajectory_compounding_noise_loss(
            model,
            window,
            prefix_length=CONTEXT,
            horizons=HORIZONS,
            anchor_stride=ANCHOR_STRIDE,
            trajectories=TRAJECTORIES,
            temperature=TEMPERATURE,
            mse_weight=1.0,
            mse_horizons=(1,),
            detach_mse_targets=False,
            noise_generator=torch.Generator(
                device="cuda"
            ).manual_seed(SEED + 4244),
        )
    )
    optional_mse_target_gradient = torch.autograd.grad(
        optional_mse,
        optional_output.sparse.gold_states,
    )[0]
    if (
        not bool(torch.isfinite(optional_mse_target_gradient).all())
        or not bool(optional_mse_target_gradient.norm() > 0)
    ):
        raise RuntimeError("optional MSE path is not available")

    identity = torch.eye(WIDTH, device=window.device)
    identity_difference = float(
        (model.operator.weight.detach() - identity).abs().max()
    )
    initial_orthogonality = orthogonality_error(model)
    if identity_difference >= 1e-4 or initial_orthogonality >= 1e-4:
        raise RuntimeError("K is not identity/orthogonal at initialization")
    if not hasattr(model, "sigma_predictor"):
        raise RuntimeError("noise model has no sigma predictor")

    noise = sparse.noise.reshape(
        2,
        TRAJECTORIES,
        ANCHOR_COUNT,
        HORIZONS,
        WIDTH,
    )
    trajectory_noise_difference = float(
        (noise[:, 0] - noise[:, 1]).detach().abs().max()
    )
    if trajectory_noise_difference == 0.0:
        raise RuntimeError("different trajectories received identical noise")

    stats = responsibility_metrics(output)
    values = {
        "initial_loss": float(loss.detach()),
        "initial_marginal_ce": float(marginal_ce.detach()),
        "initial_monitor_mse": float(monitor_mse.detach()),
        "initial_objective_k_gradient_norm": float(k_gradient.norm()),
        "initial_objective_sigma_gradient_norm": float(
            sigma_gradient.norm()
        ),
        "objective_gold_target_gradient_norm": float(
            target_gradient_by_horizon.norm()
        ),
        "objective_h1_gold_target_gradient_norm": float(
            target_gradient_by_horizon[0]
        ),
        "objective_later_gold_target_gradient_norm": float(
            target_gradient_by_horizon[1:].norm()
        ),
        "optional_mse_gold_target_gradient_norm": float(
            optional_mse_target_gradient.norm()
        ),
        "operator_identity_diff": identity_difference,
        "operator_orthogonality_error": initial_orthogonality,
        "head_self_retrieval": active_head_self_retrieval(model),
        "trajectory_noise_max_difference": (
            trajectory_noise_difference
        ),
        **{f"initial_{key}": value for key, value in stats.items()},
    }
    model.zero_grad(set_to_none=True)
    return values


@torch.inference_mode()
def evaluate(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    microbatch: int,
    operator_gradient_norm: float,
) -> dict[str, float]:
    model.eval()
    path_nll_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    path_correct_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    posterior_nll_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    posterior_correct_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    oracle_nll_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    oracle_correct_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    confidence_nll_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    confidence_correct_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    state_mse_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    diversity_sum = torch.zeros(HORIZONS, dtype=torch.float64)
    responsibility_max_sum = 0.0
    responsibility_entropy_sum = 0.0
    effective_trajectories_sum = 0.0
    score_gap_sum = 0.0
    marginal_sequence_nll_sum = 0.0
    mc_marginal_sequence_nll_sum = 0.0
    usage_sum = torch.zeros(TRAJECTORIES, dtype=torch.float64)
    winner_sum = torch.zeros(TRAJECTORIES, dtype=torch.float64)
    log_sigma_sum = 0.0
    sigma_sum = 0.0
    log_sigma_count = 0
    log_sigma_min = math.inf
    log_sigma_max = -math.inf
    decisions = 0
    noise_generator = torch.Generator(device="cuda").manual_seed(
        SEED + 5151
    )

    for begin in range(0, len(starts), microbatch):
        window = windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            WINDOW_LENGTH,
        )
        _, _, _, output = objective(
            model,
            window,
            noise_generator=noise_generator,
        )
        sparse = output.sparse
        batch = window.shape[0]
        anchors = sparse.anchor_indices.numel()
        logits = sparse.logits.float()
        vocabulary = logits.shape[-1]
        logits = logits.reshape(
            batch,
            TRAJECTORIES,
            anchors,
            HORIZONS,
            vocabulary,
        )
        predictions = logits.argmax(dim=-1)
        targets = sparse.targets
        token_ce = output.token_ce.float()
        correct = predictions.eq(targets[:, None]).float()
        responsibilities = output.responsibilities.float()
        decisions_in_batch = batch * anchors

        path_nll_sum += token_ce.sum(dim=(0, 1, 2)).cpu()
        path_correct_sum += correct.sum(dim=(0, 1, 2)).cpu()
        posterior_nll_sum += (
            responsibilities[..., None] * token_ce
        ).sum(dim=(0, 1, 2)).cpu()
        posterior_correct_sum += (
            responsibilities[..., None] * correct
        ).sum(dim=(0, 1, 2)).cpu()

        oracle_index = output.trajectory_ce.argmin(dim=1)
        gather_index = oracle_index[:, None, :, None].expand(
            batch,
            1,
            anchors,
            HORIZONS,
        )
        oracle_ce = token_ce.gather(1, gather_index).squeeze(1)
        oracle_correct = correct.gather(
            1, gather_index
        ).squeeze(1)
        oracle_nll_sum += oracle_ce.sum(dim=(0, 1)).cpu()
        oracle_correct_sum += oracle_correct.sum(dim=(0, 1)).cpu()

        max_log_probability = (
            logits.max(dim=-1).values - logits.logsumexp(dim=-1)
        )
        confidence_index = max_log_probability.sum(dim=-1).argmax(
            dim=1
        )
        confidence_gather = confidence_index[
            :, None, :, None
        ].expand(batch, 1, anchors, HORIZONS)
        confidence_ce = token_ce.gather(
            1, confidence_gather
        ).squeeze(1)
        confidence_correct = correct.gather(
            1, confidence_gather
        ).squeeze(1)
        confidence_nll_sum += confidence_ce.sum(dim=(0, 1)).cpu()
        confidence_correct_sum += confidence_correct.sum(
            dim=(0, 1)
        ).cpu()

        state_mse = relative_mse_rows(
            sparse.predicted_states,
            sparse.gold_states[:, None],
        )
        state_mse_sum += state_mse.sum(dim=(0, 1, 2)).cpu()
        diversity_sum += (
            predictions.amax(dim=1).ne(predictions.amin(dim=1))
        ).sum(dim=(0, 1)).cpu()

        entropy = -(
            responsibilities
            * responsibilities.clamp_min(1e-12).log()
        ).sum(dim=1)
        responsibility_max_sum += float(
            responsibilities.max(dim=1).values.sum()
        )
        responsibility_entropy_sum += float(entropy.sum())
        effective_trajectories_sum += float(entropy.exp().sum())
        usage_sum += responsibilities.sum(dim=(0, 2)).cpu()
        winners = output.trajectory_ce.argmin(dim=1)
        winner_sum += torch.stack(
            [(winners == index).sum() for index in range(TRAJECTORIES)]
        ).cpu()
        sorted_ce = output.trajectory_ce.float().sort(dim=1).values
        score_gap_sum += float(
            (sorted_ce[:, 1] - sorted_ce[:, 0]).sum()
        )
        marginal_sequence_nll = -TEMPERATURE * (
            torch.logsumexp(
                -output.trajectory_ce.float() / TEMPERATURE,
                dim=1,
            )
            - math.log(TRAJECTORIES)
        )
        marginal_sequence_nll_sum += float(
            marginal_sequence_nll.sum()
        )
        mc_marginal_sequence_nll = -(
            torch.logsumexp(
                -output.trajectory_ce.float(),
                dim=1,
            )
            - math.log(TRAJECTORIES)
        )
        mc_marginal_sequence_nll_sum += float(
            mc_marginal_sequence_nll.sum()
        )

        log_sigma = sparse.log_sigma.float()
        log_sigma_sum += float(log_sigma.sum())
        sigma_sum += float(log_sigma.exp().sum())
        log_sigma_count += log_sigma.numel()
        log_sigma_min = min(log_sigma_min, float(log_sigma.min()))
        log_sigma_max = max(log_sigma_max, float(log_sigma.max()))
        decisions += decisions_in_batch

    path_denominator = decisions * TRAJECTORIES
    metrics = {
        "marginal_nll": (
            marginal_sequence_nll_sum / decisions / HORIZONS
        ),
        "mc_marginal_nll_tau1": (
            mc_marginal_sequence_nll_sum / decisions / HORIZONS
        ),
        "path_mean_nll": float(
            (path_nll_sum / path_denominator).mean()
        ),
        "posterior_nll": float(
            (posterior_nll_sum / decisions).mean()
        ),
        "oracle_nll": float(
            (oracle_nll_sum / decisions).mean()
        ),
        "confidence_nll": float(
            (confidence_nll_sum / decisions).mean()
        ),
        "responsibility_max": (
            responsibility_max_sum / decisions
        ),
        "responsibility_entropy": (
            responsibility_entropy_sum / decisions
        ),
        "effective_trajectories": (
            effective_trajectories_sum / decisions
        ),
        "score_gap": score_gap_sum / decisions,
        "log_sigma_mean": log_sigma_sum / log_sigma_count,
        "sigma_mean": sigma_sum / log_sigma_count,
        "log_sigma_min": log_sigma_min,
        "log_sigma_max": log_sigma_max,
        "operator_gradient_norm": operator_gradient_norm,
        "operator_orthogonality_error": orthogonality_error(model),
    }
    for trajectory in range(TRAJECTORIES):
        metrics[f"responsibility_usage_t{trajectory + 1}"] = float(
            usage_sum[trajectory] / decisions
        )
        metrics[f"winner_usage_t{trajectory + 1}"] = float(
            winner_sum[trajectory] / decisions
        )
    for horizon in range(HORIZONS):
        suffix = horizon + 1
        metrics[f"path_h{suffix}_nll"] = float(
            path_nll_sum[horizon] / path_denominator
        )
        metrics[f"path_h{suffix}_accuracy"] = float(
            path_correct_sum[horizon] / path_denominator
        )
        metrics[f"posterior_h{suffix}_nll"] = float(
            posterior_nll_sum[horizon] / decisions
        )
        metrics[f"posterior_h{suffix}_accuracy"] = float(
            posterior_correct_sum[horizon] / decisions
        )
        metrics[f"oracle_h{suffix}_nll"] = float(
            oracle_nll_sum[horizon] / decisions
        )
        metrics[f"oracle_h{suffix}_accuracy"] = float(
            oracle_correct_sum[horizon] / decisions
        )
        metrics[f"confidence_h{suffix}_nll"] = float(
            confidence_nll_sum[horizon] / decisions
        )
        metrics[f"confidence_h{suffix}_accuracy"] = float(
            confidence_correct_sum[horizon] / decisions
        )
        metrics[f"h{suffix}_state_relative_mse"] = float(
            state_mse_sum[horizon] / path_denominator
        )
        metrics[f"h{suffix}_trajectory_diversity"] = float(
            diversity_sum[horizon] / decisions
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
                "trajectories": TRAJECTORIES,
                "temperature": TEMPERATURE,
                "mse_weight": MSE_WEIGHT,
                "mse_horizons": MSE_HORIZONS,
                "mse_target_encoder": "online_shared",
                "detach_mse_targets": DETACH_MSE_TARGETS,
                "anchor_stride": ANCHOR_STRIDE,
                "effective_batch": EFFECTIVE_BATCH,
                "microbatch": MICROBATCH,
                "seed": SEED,
                "operator": "skew_matrix_exp_no_query",
                "noise": "learned_compounding_reparameterized",
                "noise_predictor_input": "clean_K_power",
                "trajectory_assignment": (
                    "detached_softmax_negative_four_token_ce"
                ),
                "trajectory_shared_computation": (
                    "encoder_clean_K_sigma_and_inverse_prefix"
                ),
                "schedule_steps": SCHEDULE_STEPS,
            },
        },
        path,
    )


def should_report(step: int, final_step: int) -> bool:
    return (
        step in (1, 50, 100, 250, 500, 750, 1000)
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
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "token_start"))
        writer.writerows(enumerate(validation_starts.tolist()))

    model = make_model().cuda().train()
    preflight_values = preflight(model, training)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    accumulation_steps = args.batch // args.microbatch
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"trajectories={TRAJECTORIES} horizons={HORIZONS} "
        f"batch={args.batch} micro={args.microbatch}x"
        f"{accumulation_steps} "
        f"initial_marginal_ce="
        f"{preflight_values['initial_marginal_ce']:.4f}",
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
            ("trajectories", TRAJECTORIES),
            ("temperature", TEMPERATURE),
            ("mse_weight", MSE_WEIGHT),
            (
                "mse_horizons",
                "all"
                if MSE_HORIZONS is None
                else ",".join(map(str, MSE_HORIZONS)),
            ),
            ("mse_target_encoder", "online_shared"),
            ("detach_mse_targets", DETACH_MSE_TARGETS),
            ("anchor_stride", ANCHOR_STRIDE),
            ("anchor_count", ANCHOR_COUNT),
            ("unique_labels_per_example", ANCHOR_COUNT * HORIZONS),
            (
                "trajectory_labels_per_example",
                ANCHOR_COUNT * HORIZONS * TRAJECTORIES,
            ),
            ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"),
            ("operator", "skew_matrix_exp_no_query"),
            ("noise", "learned_compounding_reparameterized"),
            ("noise_predictor_input", "clean_K_power"),
            ("noise_draws", "independent_per_trajectory_and_horizon"),
            ("init_sigma", INIT_SIGMA),
            (
                "trajectory_score",
                "negative_sum_of_four_gold_token_cross_entropies",
            ),
            (
                "trajectory_responsibility",
                f"stopgrad_softmax_score_temperature_{TEMPERATURE:g}",
            ),
            (
                "trajectory_shared_computation",
                "encoder_clean_K_sigma_and_inverse_prefix",
            ),
            (
                "hidden_state_mse",
                "disabled_weight_0"
                if MSE_WEIGHT == 0.0
                else (
                    "online_"
                    + (
                        "detached"
                        if DETACH_MSE_TARGETS
                        else "attached"
                    )
                    + "_relative_mse_"
                    + (
                        "all_horizons"
                        if MSE_HORIZONS is None
                        else "horizons_"
                        + "_".join(map(str, MSE_HORIZONS))
                    )
                ),
            ),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    metric_names = (
        "marginal_nll",
        "mc_marginal_nll_tau1",
        "path_mean_nll",
        "posterior_nll",
        "oracle_nll",
        "confidence_nll",
        "responsibility_max",
        "responsibility_entropy",
        "effective_trajectories",
        "score_gap",
        "log_sigma_mean",
        "sigma_mean",
        "log_sigma_min",
        "log_sigma_max",
        "operator_gradient_norm",
        "operator_orthogonality_error",
        *tuple(
            f"responsibility_usage_t{trajectory}"
            for trajectory in range(1, TRAJECTORIES + 1)
        ),
        *tuple(
            f"winner_usage_t{trajectory}"
            for trajectory in range(1, TRAJECTORIES + 1)
        ),
        *tuple(
            f"{prefix}_h{horizon}_{metric}"
            for horizon in range(1, HORIZONS + 1)
            for prefix, metric in (
                ("path", "nll"),
                ("path", "accuracy"),
                ("posterior", "nll"),
                ("posterior", "accuracy"),
                ("oracle", "nll"),
                ("oracle", "accuracy"),
                ("confidence", "nll"),
                ("confidence", "accuracy"),
            )
        ),
        *tuple(
            f"h{horizon}_state_relative_mse"
            for horizon in range(1, HORIZONS + 1)
        ),
        *tuple(
            f"h{horizon}_trajectory_diversity"
            for horizon in range(1, HORIZONS + 1)
        ),
    )
    fields = (
        "step",
        "train_marginal_ce",
        "train_posterior_ce",
        "train_path_mean_ce",
        "train_monitor_mse",
        "train_responsibility_max",
        "train_responsibility_entropy",
        "train_effective_trajectories",
        "train_score_gap",
        "train_sigma_mean",
        "train_log_sigma_mean",
        *tuple(f"val_{name}" for name in metric_names),
        "lr",
        "wall_s",
        "unique_tokens_per_s",
        "trajectory_labels_per_s",
        "peak_vram_bytes",
    )

    data_generator = torch.Generator().manual_seed(SEED)
    training_noise_generator = torch.Generator(
        device="cuda"
    ).manual_seed(SEED + 7171)
    best_nll = math.inf
    best_step = 0
    started = time.time()
    processed_unique_tokens = 0
    processed_trajectory_labels = 0
    torch.cuda.reset_peak_memory_stats()
    with (args.record_dir / "metrics.tsv").open(
        "w", newline=""
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
                "marginal_ce": 0.0,
                "posterior_ce": 0.0,
                "path_mean_ce": 0.0,
                "monitor_mse": 0.0,
                "responsibility_max": 0.0,
                "responsibility_entropy": 0.0,
                "effective_trajectories": 0.0,
                "score_gap": 0.0,
                "sigma_mean": 0.0,
                "log_sigma_mean": 0.0,
            }
            micro_windows = window.split(args.microbatch)
            for micro_window in micro_windows:
                loss, marginal_ce, _, output = objective(
                    model,
                    micro_window,
                    noise_generator=training_noise_generator,
                )
                (loss / len(micro_windows)).backward()
                stats = responsibility_metrics(output)
                train_sums["marginal_ce"] += (
                    float(marginal_ce.detach()) / len(micro_windows)
                )
                for key, value in stats.items():
                    train_sums[key] += value / len(micro_windows)
                del loss, marginal_ce, output

            operator_gradient_norm = float(
                model.operator.A.grad.float().norm()
            )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CLIP_NORM,
            )
            optimizer.step()
            processed_unique_tokens += args.batch * WINDOW_LENGTH
            processed_trajectory_labels += (
                args.batch
                * ANCHOR_COUNT
                * HORIZONS
                * TRAJECTORIES
            )
            step = index + 1
            if should_report(step, args.steps):
                metrics = evaluate(
                    model,
                    validation,
                    validation_starts,
                    args.eval_microbatch,
                    operator_gradient_norm,
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
                    "unique_tokens_per_s": (
                        processed_unique_tokens / wall
                    ),
                    "trajectory_labels_per_s": (
                        processed_trajectory_labels / wall
                    ),
                    "peak_vram_bytes": (
                        torch.cuda.max_memory_allocated()
                    ),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d} marginal="
                    f"{metrics['marginal_nll']:.4f} "
                    f"path={metrics['path_mean_nll']:.4f} "
                    f"resp_max={metrics['responsibility_max']:.3f} "
                    f"eff={metrics['effective_trajectories']:.3f} "
                    f"sigma={metrics['sigma_mean']:.4f}",
                    flush=True,
                )
                print(
                    "  confidence acc[h1-h4]: "
                    + " ".join(
                        f"{metrics[f'confidence_h{h}_accuracy']:.3f}"
                        for h in range(1, HORIZONS + 1)
                    ),
                    flush=True,
                )
                print(
                    "  oracle acc[h1-h4]:     "
                    + " ".join(
                        f"{metrics[f'oracle_h{h}_accuracy']:.3f}"
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
                if step in (100, 250, 500, 750, 1000):
                    save_checkpoint(
                        args.output_dir / f"step{step:04d}.pt",
                        model,
                        optimizer,
                        step,
                        metrics,
                    )
                if metrics["marginal_nll"] < best_nll:
                    best_nll = metrics["marginal_nll"]
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
        writer.writerow(("best_val_marginal_nll", best_nll))
        writer.writerow(("wall_s", time.time() - started))
        writer.writerow(
            ("peak_vram_bytes", torch.cuda.max_memory_allocated())
        )


if __name__ == "__main__":
    main()
