"""Train self-predicted complex KV dynamics with one online H1 closure."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from rotlm.evaluation import complex_self_predicted_kv_ar_rollouts
from rotlm.models.complex_self_prediction import (
    ComplexSelfPredictedKVTransition,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.training.ema import (
    ema_decay_at_step,
    make_ema_model,
    update_ema_model,
)
from rotlm.training.ha_skew_window import (
    relative_mse_rows,
    sparse_complex_self_predicted_kv_loss,
)
from train_byte256_simplex_tied_self_composition_h1_ce_13m import (
    DATA_ROOT,
    load_fixed_starts,
    memmap,
    sampled_windows,
)
from train_k1_decoder_inverse_ablation_13m import (
    CLIP_NORM,
    CONTEXT,
    PEAK_LR,
    SEED,
    lr_at,
    parameter_count,
    trainable_parameter_count,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length


EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "h1-online-mse-ce-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
VOCABULARY = 256
WIDTH = 1344
ENCODER_BLOCKS = 2
HEADS = 8
KEY_DIM = 16
VALUE_DIM = 31
SIMPLEX_LOGIT_SCALE = 16.0
INITIAL_FREQUENCY_RANGE = 0.05
INITIAL_BETA = 0.5
TEMPERED_READOUT = True
RESIDUAL_PRIOR = True
RECURRENT_HIDDEN_MODE = "split-query-residual"
REAL_VALUE = False
STATE_DEPENDENT_MEMORY_DECAY = False
MEMORY_DECAY_DT_MIN = 5e-2
MEMORY_DECAY_DT_MAX = 5e-1
PRE_NORMALIZE_QKV_INPUTS = True
NORMALIZE_INITIAL_RECURRENT_ROOT = False
NORMALIZE_ACCUMULATED_READS = False
POST_NORMALIZE_HIDDEN_READS = False
POST_NORMALIZE_MEMORY = False
POST_NORMALIZE_RECURRENT_STATE = False
POST_NORM_EPS = 1e-6
LATENT_WEIGHT = 1.0
DETACH_LATENT_TARGET = False
HORIZONS = 4
TRAIN_HORIZONS = 1
TRAIN_ANCHOR_STRIDE = 1
EVAL_ANCHOR_STRIDE = 4
TRAIN_WINDOW_LENGTH = CONTEXT + TRAIN_HORIZONS
EVAL_WINDOW_LENGTH = CONTEXT + HORIZONS
TRAIN_ANCHORS = len(range(0, CONTEXT, TRAIN_ANCHOR_STRIDE))
EFFECTIVE_BATCH = 64
MICROBATCH = 16
STEPS = 1000
SCHEDULE_STEPS = 6000
REPORT_STEPS = frozenset((1, 50, 100, 250, 500, 750, 1000))
EXPECTED_PARAMETERS = 13_216_353
MIN_H2_H4_TOKEN_AGREEMENT = 0.50
MIN_MEAN_AR_STATE_COSINE = 0.75
MIN_H1_POSTERIOR_TARGET_ACCURACY = None
EMA_TARGET_DECAY = None
EMA_WARM_START_STEPS = 0
EVALUATE_WITH_EMA = False
REPORT_ONLINE_WITH_EMA = False
STEP100_REFERENCE = None
STEP100_REFERENCE_TOLERANCE = 1e-5
CLOSURE_METRIC_NAME = "latent_relative_mse"
CLOSURE_DISPLAY_NAME = "mse"
CLOSURE_VALIDATION_METRIC_NAME = "h1_latent_relative_mse"
OBJECTIVE_NAME_OVERRIDE = None
TRAIN_OBJECTIVE_OVERRIDE = None
CLOSURE_TARGET_NAME_OVERRIDE = None
REQUIRE_H1_LATENT_MSE_IMPROVEMENT = True
REQUIRE_WRITE_EXCEEDS_NO_WRITE = True
TRAINING_TARGET_DESCRIPTION_OVERRIDE = None
FUSE_BRANCH_MEMORY_KERNEL = False


def make_model() -> K1DecoderAblationLM:
    model = K1DecoderAblationLM(
        VOCABULARY,
        width=WIDTH,
        encoder_blocks=ENCODER_BLOCKS,
        decoder_mode="exact-inverse",
        head_mode="simplex-raw-tied",
        simplex_logit_scale=SIMPLEX_LOGIT_SCALE,
    )
    model.operator = nn.Identity()
    transition_kwargs = {
        "heads": HEADS,
        "key_dim": KEY_DIM,
        "value_dim": VALUE_DIM,
        "initial_frequency_range": INITIAL_FREQUENCY_RANGE,
        "learned_read_beta": TEMPERED_READOUT,
        "residual_prior": RESIDUAL_PRIOR,
        "recurrent_hidden_mode": RECURRENT_HIDDEN_MODE,
        "real_value": REAL_VALUE,
        "state_dependent_memory_decay": STATE_DEPENDENT_MEMORY_DECAY,
        "memory_decay_dt_min": MEMORY_DECAY_DT_MIN,
        "memory_decay_dt_max": MEMORY_DECAY_DT_MAX,
        "pre_normalize_qkv_inputs": PRE_NORMALIZE_QKV_INPUTS,
        "normalize_initial_recurrent_root": (
            NORMALIZE_INITIAL_RECURRENT_ROOT
        ),
        "normalize_accumulated_reads": NORMALIZE_ACCUMULATED_READS,
        "post_normalize_hidden_reads": POST_NORMALIZE_HIDDEN_READS,
        "post_normalize_memory": POST_NORMALIZE_MEMORY,
        "post_normalize_recurrent_state": (
            POST_NORMALIZE_RECURRENT_STATE
        ),
        "post_norm_eps": POST_NORM_EPS,
        "fuse_branch_memory_kernel": FUSE_BRANCH_MEMORY_KERNEL,
    }
    if TEMPERED_READOUT:
        transition_kwargs["initial_beta"] = INITIAL_BETA
    model.complex_self_prediction = ComplexSelfPredictedKVTransition(
        WIDTH,
        **transition_kwargs,
    )
    return model


def objective(model, window, *, latent_target_model=None):
    return sparse_complex_self_predicted_kv_loss(
        model,
        window,
        prefix_length=CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=TRAIN_ANCHOR_STRIDE,
        latent_weight=LATENT_WEIGHT,
        detach_latent_targets=DETACH_LATENT_TARGET,
        latent_target_model=latent_target_model,
    )


def registered_objective_name() -> str:
    if OBJECTIVE_NAME_OVERRIDE is not None:
        return OBJECTIVE_NAME_OVERRIDE
    if EMA_TARGET_DECAY is not None:
        return "raw_tied_ce_plus_ema_sg_h1_relative_mse"
    if DETACH_LATENT_TARGET:
        return "raw_tied_ce_plus_online_sg_h1_relative_mse"
    return "raw_tied_ce_plus_attached_h1_relative_mse"


def registered_closure_target_name() -> str:
    if CLOSURE_TARGET_NAME_OVERRIDE is not None:
        return CLOSURE_TARGET_NAME_OVERRIDE
    return "full_model_ema" if EMA_TARGET_DECAY is not None else "online_encoder"


def additional_final_checks(
    initial_row: dict[str, float],
    final_row: dict[str, float],
    rows: list[dict[str, float]],
) -> dict[str, bool]:
    """Extension hook for objective-specific producer wrappers."""
    del initial_row, final_row, rows
    return {}


def additional_interpretation_lines(
    initial_row: dict[str, float],
    final_row: dict[str, float],
    rows: list[dict[str, float]],
) -> list[str]:
    """Extension hook for objective-specific interpretation tables."""
    del initial_row, final_row, rows
    return []


def _require_gradient(name: str, gradient: torch.Tensor | None) -> None:
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.norm() > 0)
    ):
        raise RuntimeError(f"no finite nonzero gradient to {name}")


def preflight(model, training, *, latent_target_model=None) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 9242)
    window = sampled_windows(training, 2, generator, TRAIN_WINDOW_LENGTH)
    loss, token_ce, latent_mse, output = objective(
        model,
        window,
        latent_target_model=latent_target_model,
    )
    expected = (2, TRAIN_ANCHORS, 1)
    if output.token_ce.shape != expected:
        raise RuntimeError(f"unexpected token CE shape {output.token_ce.shape}")
    torch.testing.assert_close(loss, token_ce + LATENT_WEIGHT * latent_mse)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("preflight loss is non-finite")
    ema_initial_state_error = 0.0
    if latent_target_model is not None:
        online_state = model.state_dict()
        target_state = latent_target_model.state_dict()
        if online_state.keys() != target_state.keys():
            raise RuntimeError("online and EMA state dictionaries differ")
        state_errors = []
        for name, online_value in online_state.items():
            target_value = target_state[name]
            if torch.equal(online_value, target_value):
                state_errors.append(0.0)
            elif torch.is_floating_point(online_value) or online_value.is_complex():
                state_errors.append(
                    float((online_value - target_value).abs().max())
                )
            else:
                state_errors.append(math.inf)
        ema_initial_state_error = max(state_errors)
        if ema_initial_state_error != 0.0:
            raise RuntimeError("EMA did not start as an exact online copy")
        if any(
            parameter.requires_grad
            for parameter in latent_target_model.parameters()
        ):
            raise RuntimeError("EMA target parameters require gradients")
    stop_gradient_forward_error = 0.0
    if DETACH_LATENT_TARGET:
        attached_loss, attached_ce, attached_mse, attached_output = (
            sparse_complex_self_predicted_kv_loss(
                model,
                window,
                prefix_length=CONTEXT,
                horizons=TRAIN_HORIZONS,
                anchor_stride=TRAIN_ANCHOR_STRIDE,
                latent_weight=LATENT_WEIGHT,
                detach_latent_targets=False,
            )
        )
        stop_gradient_forward_error = max(
            float((loss - attached_loss).detach().abs()),
            float((token_ce - attached_ce).detach().abs()),
            float((latent_mse - attached_mse).detach().abs()),
            float(
                (
                    output.full_states - attached_output.full_states
                ).detach().abs().max()
            ),
            float(
                (output.logits - attached_output.logits).detach().abs().max()
            ),
        )
        if stop_gradient_forward_error >= 1e-7:
            raise RuntimeError("stop-gradient changed the forward computation")

    codebook = model.embedding_weight.detach().float()
    gram = codebook @ codebook.T
    expected_gram = torch.full_like(gram, -1.0 / (VOCABULARY - 1))
    expected_gram.fill_diagonal_(1.0)
    simplex_error = float((gram - expected_gram).abs().max())
    if simplex_error >= 3e-6:
        raise RuntimeError("fixed codebook is not a regular simplex")
    if model.embedding_weight.requires_grad:
        raise RuntimeError("simplex codebook must be frozen")
    if model.encoder.head_norm.weight.requires_grad:
        raise RuntimeError("unused vocabulary head norm must be frozen")
    retrieval = float(
        model.token_logits(codebook).argmax(dim=-1).eq(
            torch.arange(VOCABULARY, device=codebook.device)
        ).float().mean()
    )
    if retrieval != 1.0:
        raise RuntimeError("raw tied codebook does not retrieve itself")
    if model.codebook_head is not None or (
        model.head_mode != "simplex-raw-tied"
    ):
        raise RuntimeError("model attached an independent vocabulary scorer")
    total_parameters = parameter_count(model)
    if total_parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"parameter total {total_parameters} != {EXPECTED_PARAMETERS}"
        )

    transition = model.complex_self_prediction
    parameters = [
        transition.query.weight,
        transition.key.weight,
        transition.value.weight,
        transition.output.weight,
        transition.hidden_phase,
        transition.memory_phase,
        model.encoder.blocks[0].attn.qkv.weight,
    ]
    parameter_names = [
        "query",
        "key",
        "value",
        "complex readout",
        "hidden phase",
        "memory phase",
        "reversible encoder",
    ]
    if TEMPERED_READOUT:
        if transition.beta_logit is None:
            raise RuntimeError("tempered run did not register beta")
        parameters.insert(-1, transition.beta_logit)
        parameter_names.insert(-1, "beta")
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
    for name, gradient in zip(
        parameter_names,
        gradients,
    ):
        _require_gradient(name, gradient)
    target_gradient = None
    if output.gold_states.requires_grad:
        target_gradient = torch.autograd.grad(
            latent_mse,
            output.gold_states,
            retain_graph=True,
            allow_unused=DETACH_LATENT_TARGET,
        )[0]
    if DETACH_LATENT_TARGET or latent_target_model is not None:
        if target_gradient is not None and bool(
            target_gradient.abs().max() > 0
        ):
            raise RuntimeError("stop-gradient latent target received a gradient")
    else:
        _require_gradient("attached online future state", target_gradient)
    if latent_target_model is not None and any(
        parameter.grad is not None
        for parameter in latent_target_model.parameters()
    ):
        raise RuntimeError("EMA target accumulated gradients")

    ema_copy_error = 0.0
    ema_affine_error = 0.0
    if latent_target_model is not None:
        tiny_online = nn.Linear(2, 2, bias=False)
        tiny_ema = make_ema_model(tiny_online)
        with torch.no_grad():
            tiny_online.weight.add_(1.0)
        update_ema_model(tiny_ema, tiny_online, 0.0)
        ema_copy_error = float(
            (tiny_ema.weight - tiny_online.weight).detach().abs().max()
        )
        with torch.no_grad():
            before = tiny_ema.weight.clone()
            tiny_online.weight.add_(1.0)
        update_ema_model(tiny_ema, tiny_online, 0.99)
        expected_ema = 0.99 * before + 0.01 * tiny_online.weight
        ema_affine_error = float(
            (tiny_ema.weight - expected_ema).detach().abs().max()
        )
        if max(ema_copy_error, ema_affine_error) >= 1e-7:
            raise RuntimeError("EMA update does not match its registered rule")

    eval_window = sampled_windows(training, 1, generator, EVAL_WINDOW_LENGTH)
    rollout = complex_self_predicted_kv_ar_rollouts(
        model,
        eval_window,
        prefix_length=CONTEXT,
        horizons=HORIZONS,
        anchor_stride=EVAL_ANCHOR_STRIDE,
    )
    h1_state_error = float(
        (
            rollout.parallel_full_states[:, :, 0]
            - rollout.ar_proposal_states[:, :, 0]
        ).abs().max()
    )
    h1_logit_error = float(
        (
            rollout.parallel_logits[:, :, 0]
            - rollout.ar_logits[:, :, 0]
        ).abs().max()
    )
    if h1_state_error >= 1e-6 or h1_logit_error >= 5e-4:
        raise RuntimeError("parallel and AR paths do not share horizon one")

    changed = eval_window.clone()
    changed[:, CONTEXT:] = (changed[:, CONTEXT:] + 1) % VOCABULARY
    changed_rollout = complex_self_predicted_kv_ar_rollouts(
        model,
        changed,
        prefix_length=CONTEXT,
        horizons=HORIZONS,
        anchor_stride=EVAL_ANCHOR_STRIDE,
    )
    future_state_error = float(
        (
            rollout.parallel_full_states
            - changed_rollout.parallel_full_states
        ).abs().max()
    )
    future_logit_error = float(
        (rollout.parallel_logits - changed_rollout.parallel_logits).abs().max()
    )
    if max(future_state_error, future_logit_error) >= 1e-6:
        raise RuntimeError("held-out future bytes leaked into central rollout")

    posterior_readout_logit_delta = float(
        (rollout.full_gain_logits - rollout.parallel_logits).abs().mean()
    )
    prior_readout_error = float(
        (
            rollout.parallel_read_states
            - rollout.parallel_preliminary_states
        ).abs().max()
    )
    if posterior_readout_logit_delta <= 0:
        raise RuntimeError("posterior and decoder readouts are identical")
    if not TEMPERED_READOUT and prior_readout_error >= 1e-6:
        raise RuntimeError("decoder readout contains current innovation")
    beta = transition.beta
    if TEMPERED_READOUT and beta is None:
        raise RuntimeError("tempered readout has no learned beta")
    if not TEMPERED_READOUT and beta is not None:
        raise RuntimeError("strict prior readout unexpectedly has beta")
    model.zero_grad(set_to_none=True)
    result = {
        "initial_loss": float(loss.detach()),
        "initial_token_ce": float(token_ce.detach()),
        "initial_latent_relative_mse": float(latent_mse.detach()),
        "initial_closure_loss": float(latent_mse.detach()),
        "simplex_gram_max_error": simplex_error,
        "simplex_self_retrieval": retrieval,
        "h1_parallel_ar_state_error": h1_state_error,
        "h1_parallel_ar_logit_error": h1_logit_error,
        "future_state_leak_error": future_state_error,
        "future_logit_leak_error": future_logit_error,
        "posterior_readout_logit_abs_delta": posterior_readout_logit_delta,
        "prior_readout_state_error": prior_readout_error,
        "future_hidden_target_gradient_norm": (
            float(target_gradient.norm()) if target_gradient is not None else 0.0
        ),
        "ema_initial_state_error": ema_initial_state_error,
        "ema_exact_copy_update_error": ema_copy_error,
        "ema_affine_update_error": ema_affine_error,
        "stop_gradient_forward_error": stop_gradient_forward_error,
        "query_gradient_norm": float(gradients[0].norm()),
        "innovation_gradient_norm": float(gradients[2].norm()),
    }
    if beta is not None:
        result["initial_beta"] = float(beta.detach())
        result["beta_gradient_norm"] = float(gradients[6].norm())
    return result


EVAL_ROW_NAMES = (
    "overall_cosine",
    "overall_relative_mse",
    "one_step_closure_cosine",
    "one_step_closure_relative_mse",
    "path_cosine",
    "path_relative_mse",
    "token_agreement",
    "no_write_token_agreement",
    "posterior_readout_token_agreement",
    "posterior_target_accuracy",
    "target_nll",
    "target_accuracy",
    "readout_entropy",
    "posterior_entropy",
    "readout_max_probability",
    "posterior_max_probability",
    "posterior_max_probability_ge_099",
    "memory_energy",
    "memory_energy_per_write",
    "innovation_energy",
    "preliminary_hidden_rms",
    "recurrent_hidden_rms",
    "parallel_logit_rms",
)


@torch.inference_mode()
def evaluate(
    model,
    validation,
    starts,
    microbatch: int,
    *,
    latent_target_model=None,
) -> dict[str, float]:
    sums = {
        name: torch.zeros(HORIZONS, dtype=torch.float64)
        for name in EVAL_ROW_NAMES
    }
    boundaries = 0
    h1_nll_sum = 0.0
    h1_correct = 0
    h1_latent_mse_sum = 0.0
    exact_blocks = 0
    posterior_readout_logit_delta_sum = 0.0
    h1_max_logit_error = 0.0

    for begin in range(0, len(starts), microbatch):
        window = windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            EVAL_WINDOW_LENGTH,
        )
        output = complex_self_predicted_kv_ar_rollouts(
            model,
            window,
            prefix_length=CONTEXT,
            horizons=HORIZONS,
            anchor_stride=EVAL_ANCHOR_STRIDE,
            latent_target_model=latent_target_model,
        )
        parallel = output.parallel_full_states.float()
        canonical = output.ar_states.float()
        proposals = output.ar_proposal_states.float()
        count = window.shape[0] * output.anchor_indices.numel()
        boundaries += count

        readout_log_probs = F.log_softmax(
            output.parallel_logits.float(), dim=-1
        )
        posterior_log_probs = F.log_softmax(
            output.full_gain_logits.float(), dim=-1
        )
        readout_entropy = -(
            readout_log_probs.exp() * readout_log_probs
        ).sum(dim=-1)
        posterior_entropy = -(
            posterior_log_probs.exp() * posterior_log_probs
        ).sum(dim=-1)
        target_nll = F.nll_loss(
            readout_log_probs.flatten(0, 2),
            output.targets.flatten(),
            reduction="none",
        ).reshape_as(output.targets)
        readout_max_probability = readout_log_probs.exp().amax(dim=-1)
        posterior_max_probability = posterior_log_probs.exp().amax(dim=-1)
        memory_write_counts = torch.arange(
            2,
            2 + output.memory_energy.shape[-1],
            device=output.memory_energy.device,
            dtype=output.memory_energy.dtype,
        )
        values = {
            "overall_cosine": F.cosine_similarity(
                parallel, canonical, dim=-1
            ),
            "overall_relative_mse": relative_mse_rows(parallel, canonical),
            "one_step_closure_cosine": F.cosine_similarity(
                proposals, canonical, dim=-1
            ),
            "one_step_closure_relative_mse": relative_mse_rows(
                proposals, canonical
            ),
            "path_cosine": F.cosine_similarity(
                parallel, proposals, dim=-1
            ),
            "path_relative_mse": relative_mse_rows(parallel, proposals),
            "token_agreement": output.parallel_tokens.eq(
                output.ar_tokens
            ).float(),
            "no_write_token_agreement": output.no_write_tokens.eq(
                output.ar_tokens
            ).float(),
            "posterior_readout_token_agreement": output.full_gain_tokens.eq(
                output.parallel_tokens
            ).float(),
            "posterior_target_accuracy": output.full_gain_tokens.eq(
                output.targets
            ).float(),
            "target_nll": target_nll,
            "target_accuracy": output.parallel_tokens.eq(
                output.targets
            ).float(),
            "readout_entropy": readout_entropy,
            "posterior_entropy": posterior_entropy,
            "readout_max_probability": readout_max_probability,
            "posterior_max_probability": posterior_max_probability,
            "posterior_max_probability_ge_099": (
                posterior_max_probability.ge(0.99).float()
            ),
            "memory_energy": output.memory_energy.float(),
            "memory_energy_per_write": (
                output.memory_energy / memory_write_counts
            ).float(),
            "innovation_energy": output.innovation_energy.float(),
            "preliminary_hidden_rms": output.parallel_preliminary_states.float()
            .square()
            .mean(dim=-1)
            .sqrt(),
            "recurrent_hidden_rms": output.parallel_full_states.float()
            .square()
            .mean(dim=-1)
            .sqrt(),
            "parallel_logit_rms": output.parallel_logits.float()
            .square()
            .mean(dim=-1)
            .sqrt(),
        }
        for name, value in values.items():
            sums[name] += value.double().sum(dim=(0, 1)).cpu()

        matches = values["token_agreement"].bool()
        exact_blocks += int(matches.all(dim=-1).sum())
        targets = output.targets[:, :, 0]
        h1_nll_sum += float(
            F.cross_entropy(
                output.parallel_logits[:, :, 0].flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )
        )
        h1_correct += int(output.parallel_tokens[:, :, 0].eq(targets).sum())
        h1_latent_mse_sum += float(
            relative_mse_rows(
                output.parallel_full_states[:, :, 0],
                output.gold_states[:, :, 0],
            ).sum()
        )
        posterior_readout_logit_delta_sum += float(
            (
                output.full_gain_logits - output.parallel_logits
            ).abs().mean(dim=-1).sum()
        )
        h1_max_logit_error = max(
            h1_max_logit_error,
            float(
                (
                    output.parallel_logits[:, :, 0]
                    - output.ar_logits[:, :, 0]
                ).abs().max()
            ),
        )

    if boundaries < 1:
        raise RuntimeError("evaluation produced no boundaries")
    means = {name: value / boundaries for name, value in sums.items()}
    metrics = {
        "evaluated_boundaries": float(boundaries),
        "h1_validation_nll": h1_nll_sum / boundaries,
        "h1_validation_accuracy": h1_correct / boundaries,
        "block_validation_nll": float(means["target_nll"].mean()),
        "block_validation_accuracy": float(
            means["target_accuracy"].mean()
        ),
        "h1_latent_relative_mse": h1_latent_mse_sum / boundaries,
        "h1_posterior_target_accuracy": float(
            means["posterior_target_accuracy"][0]
        ),
        "exact_block_token_agreement": exact_blocks / boundaries,
        "mean_overall_cosine": float(means["overall_cosine"].mean()),
        "mean_overall_relative_mse": float(
            means["overall_relative_mse"].mean()
        ),
        "h2_h4_token_agreement": float(
            means["token_agreement"][1:].mean()
        ),
        "h2_h4_no_write_token_agreement": float(
            means["no_write_token_agreement"][1:].mean()
        ),
        "posterior_readout_logit_abs_delta": posterior_readout_logit_delta_sum
        / (boundaries * HORIZONS),
        "learned_read_beta": (
            float(model.complex_self_prediction.beta)
            if model.complex_self_prediction.beta is not None
            else math.nan
        ),
        "h1_ar_parallel_max_logit_error": h1_max_logit_error,
    }
    for horizon in range(HORIZONS):
        for name, value in means.items():
            metrics[f"h{horizon + 1}_{name}"] = float(value[horizon])
    return metrics


AGGREGATE_METRIC_NAMES = (
    "evaluated_boundaries",
    "h1_validation_nll",
    "h1_validation_accuracy",
    "block_validation_nll",
    "block_validation_accuracy",
    "h1_latent_relative_mse",
    "h1_posterior_target_accuracy",
    "exact_block_token_agreement",
    "mean_overall_cosine",
    "mean_overall_relative_mse",
    "h2_h4_token_agreement",
    "h2_h4_no_write_token_agreement",
    "posterior_readout_logit_abs_delta",
    "learned_read_beta",
    "h1_ar_parallel_max_logit_error",
)


def metric_names_for_horizons(horizons: int) -> tuple[str, ...]:
    if horizons < 1:
        raise ValueError("metric horizons must be positive")
    return (
        *AGGREGATE_METRIC_NAMES,
        *tuple(
            f"h{horizon}_{name}"
            for horizon in range(1, horizons + 1)
            for name in EVAL_ROW_NAMES
        ),
    )


METRIC_NAMES = metric_names_for_horizons(HORIZONS)


def save_checkpoint(
    path,
    model,
    optimizer,
    data_generator,
    step,
    metrics,
    args,
    *,
    ema_model=None,
):
    payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_generator_state": data_generator.get_state(),
            "step": step,
            "metrics": metrics,
            "config": {
                "experiment_id": EXPERIMENT_ID,
                "vocabulary": VOCABULARY,
                "width": WIDTH,
                "encoder_blocks": ENCODER_BLOCKS,
                "heads": HEADS,
                "key_dim": KEY_DIM,
                "value_dim": VALUE_DIM,
                "simplex_logit_scale": SIMPLEX_LOGIT_SCALE,
                "initial_beta": INITIAL_BETA if TEMPERED_READOUT else None,
                "tempered_readout": TEMPERED_READOUT,
                "residual_prior": RESIDUAL_PRIOR,
                "recurrent_hidden_mode": RECURRENT_HIDDEN_MODE,
                "pre_normalize_qkv_inputs": PRE_NORMALIZE_QKV_INPUTS,
                "normalize_initial_recurrent_root": (
                    NORMALIZE_INITIAL_RECURRENT_ROOT
                ),
                "normalize_accumulated_reads": (
                    NORMALIZE_ACCUMULATED_READS
                ),
                "post_normalize_hidden_reads": (
                    POST_NORMALIZE_HIDDEN_READS
                ),
                "post_normalize_memory": POST_NORMALIZE_MEMORY,
                "post_normalize_recurrent_state": (
                    POST_NORMALIZE_RECURRENT_STATE
                ),
                "post_norm_eps": POST_NORM_EPS,
                "decoder_readout": (
                    "tempered_prior_posterior_interpolation"
                    if TEMPERED_READOUT
                    else "innovation_free_prior"
                ),
                "latent_weight": LATENT_WEIGHT,
                "closure_weight": LATENT_WEIGHT,
                "closure_metric": CLOSURE_METRIC_NAME,
                "detach_latent_target": DETACH_LATENT_TARGET,
                "latent_target": registered_closure_target_name(),
                "closure_target": registered_closure_target_name(),
                "ema_decay": EMA_TARGET_DECAY,
                "ema_warm_start_steps": EMA_WARM_START_STEPS,
                "inference_model": (
                    "full_model_ema" if EVALUATE_WITH_EMA else "online"
                ),
                "minimum_h2_h4_token_agreement": (
                    MIN_H2_H4_TOKEN_AGREEMENT
                ),
                "minimum_mean_ar_state_cosine": MIN_MEAN_AR_STATE_COSINE,
                "minimum_h1_posterior_target_accuracy": (
                    MIN_H1_POSTERIOR_TARGET_ACCURACY
                ),
                "objective": registered_objective_name(),
                "recurrence": "unitary_memory_plus_self_predicted_kv",
                "seed": SEED,
                "batch": args.batch,
                "microbatch": args.microbatch,
                "schedule_steps": args.schedule_steps,
            },
        }
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=EFFECTIVE_BATCH)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    parser.add_argument("--schedule-steps", type=int, default=SCHEDULE_STEPS)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=2)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--allow-resume-microbatch-change",
        action="store_true",
        help="allow mathematically equivalent accumulation regrouping",
    )
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
    if EMA_TARGET_DECAY is not None and not 0 <= EMA_TARGET_DECAY < 1:
        parser.error("EMA target decay must be in [0,1)")
    if EMA_WARM_START_STEPS < 0:
        parser.error("EMA warm-start steps must be non-negative")
    if EVALUATE_WITH_EMA and EMA_TARGET_DECAY is None:
        parser.error("EMA inference requires an EMA target model")

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
    starts_path = args.record_dir / "validation_starts.tsv"
    if starts_path.exists():
        stored = load_fixed_starts(starts_path, args.eval_examples)
        if not torch.equal(stored, validation_starts):
            raise RuntimeError("stored validation starts violate fixed seed")
    else:
        with starts_path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("index", "token_start"))
            writer.writerows(enumerate(validation_starts.tolist()))

    model = make_model().cuda().train()
    ema_model = (
        make_ema_model(model) if EMA_TARGET_DECAY is not None else None
    )
    preflight_values = preflight(
        model,
        training,
        latent_target_model=ema_model,
    )
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    data_generator = torch.Generator().manual_seed(SEED)
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_config = checkpoint.get("config", {})
        resume_fields = [
            ("batch", args.batch),
            ("schedule_steps", args.schedule_steps),
        ]
        if not args.allow_resume_microbatch_change:
            resume_fields.append(("microbatch", args.microbatch))
        for name, expected in resume_fields:
            if checkpoint_config.get(name) != expected:
                raise RuntimeError(
                    f"resume checkpoint {name} mismatch: "
                    f"{checkpoint_config.get(name)!r} != {expected!r}"
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        data_generator.set_state(checkpoint["data_generator_state"])
        start_step = int(checkpoint["step"])
        if not 0 < start_step < args.steps:
            raise RuntimeError(
                "resume checkpoint step must lie between zero and --steps"
            )
        if ema_model is not None:
            if "ema_model" not in checkpoint:
                raise RuntimeError("EMA resume checkpoint has no EMA state")
            ema_model.load_state_dict(checkpoint["ema_model"])
    accumulation_steps = args.batch // args.microbatch
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"batch={args.batch} micro={args.microbatch}x{accumulation_steps} "
        f"initial_ce={preflight_values['initial_token_ce']:.4f} "
        f"initial_{CLOSURE_DISPLAY_NAME}="
        f"{preflight_values['initial_closure_loss']:.4f}",
        flush=True,
    )

    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        run_items = [
            ("experiment_id", EXPERIMENT_ID),
            ("seed", SEED),
            ("steps", args.steps),
            ("start_step", start_step),
            ("resume_checkpoint", args.resume_checkpoint),
            (
                "allow_resume_microbatch_change",
                args.allow_resume_microbatch_change,
            ),
            ("schedule_steps", args.schedule_steps),
            ("effective_batch", args.batch),
            ("microbatch", args.microbatch),
            ("context", CONTEXT),
            ("vocabulary", VOCABULARY),
            ("width", WIDTH),
            ("encoder_blocks", ENCODER_BLOCKS),
            ("heads", HEADS),
            ("key_dim", KEY_DIM),
            ("value_dim", VALUE_DIM),
            ("simplex_logit_scale", SIMPLEX_LOGIT_SCALE),
            ("tempered_readout", TEMPERED_READOUT),
            ("residual_prior", RESIDUAL_PRIOR),
            ("recurrent_hidden_mode", RECURRENT_HIDDEN_MODE),
            ("pre_normalize_qkv_inputs", PRE_NORMALIZE_QKV_INPUTS),
            (
                "normalize_initial_recurrent_root",
                NORMALIZE_INITIAL_RECURRENT_ROOT,
            ),
            (
                "normalize_accumulated_reads",
                NORMALIZE_ACCUMULATED_READS,
            ),
            (
                "post_normalize_hidden_reads",
                POST_NORMALIZE_HIDDEN_READS,
            ),
            ("post_normalize_memory", POST_NORMALIZE_MEMORY),
            (
                "post_normalize_recurrent_state",
                POST_NORMALIZE_RECURRENT_STATE,
            ),
            ("post_norm_eps", POST_NORM_EPS),
            (
                "decoder_readout",
                "tempered_prior_posterior_interpolation"
                if TEMPERED_READOUT
                else "innovation_free_prior",
            ),
            ("latent_weight", LATENT_WEIGHT),
            ("closure_weight", LATENT_WEIGHT),
            ("closure_metric", CLOSURE_METRIC_NAME),
            ("detach_latent_target", DETACH_LATENT_TARGET),
            ("latent_target", registered_closure_target_name()),
            ("closure_target", registered_closure_target_name()),
            ("ema_decay", EMA_TARGET_DECAY),
            ("ema_warm_start_steps", EMA_WARM_START_STEPS),
            (
                "inference_model",
                "full_model_ema" if EVALUATE_WITH_EMA else "online",
            ),
            ("report_online_with_ema", REPORT_ONLINE_WITH_EMA),
            (
                "minimum_h2_h4_token_agreement",
                MIN_H2_H4_TOKEN_AGREEMENT,
            ),
            ("minimum_mean_ar_state_cosine", MIN_MEAN_AR_STATE_COSINE),
            (
                "minimum_h1_posterior_target_accuracy",
                MIN_H1_POSTERIOR_TARGET_ACCURACY,
            ),
            ("objective", registered_objective_name()),
            ("recurrence", "unitary_memory_plus_self_predicted_kv"),
            ("monitor_horizons", HORIZONS),
            ("train_anchor_stride", TRAIN_ANCHOR_STRIDE),
            ("eval_anchor_stride", EVAL_ANCHOR_STRIDE),
            ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"),
            ("data_root", DATA_ROOT),
        ]
        if TEMPERED_READOUT:
            run_items.append(("initial_beta", INITIAL_BETA))
        run_items.extend(preflight_values.items())
        for key, value in run_items:
            writer.writerow((key, value))

    ema_fields = (
        ("effective_ema_decay",)
        if EMA_TARGET_DECAY is not None
        else ()
    )
    online_metric_fields = (
        tuple(f"online_val_{name}" for name in METRIC_NAMES)
        if REPORT_ONLINE_WITH_EMA
        else ()
    )
    closure_train_field = f"train_{CLOSURE_METRIC_NAME}"
    fields = (
        "step",
        "train_loss",
        "train_token_ce",
        closure_train_field,
        "gradient_norm",
        *tuple(f"val_{name}" for name in METRIC_NAMES),
        *online_metric_fields,
        *ema_fields,
        "lr",
        "wall_s",
        "unique_tokens_per_s",
        "ce_labels_per_s",
        "peak_vram_bytes",
    )
    metrics_path = args.record_dir / "metrics.tsv"
    started = time.time()
    processed_tokens = 0
    processed_labels = 0
    torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, float]] = []
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        inference_model = ema_model if EVALUATE_WITH_EMA else model
        if inference_model is None:
            raise RuntimeError("configured inference model is unavailable")
        initial = evaluate(
            inference_model,
            validation,
            validation_starts,
            args.eval_microbatch,
        )
        online_initial = (
            evaluate(
                model,
                validation,
                validation_starts,
                args.eval_microbatch,
                latent_target_model=ema_model,
            )
            if REPORT_ONLINE_WITH_EMA
            else None
        )
        initial_row = {
            "step": start_step,
            "train_loss": math.nan,
            "train_token_ce": math.nan,
            closure_train_field: math.nan,
            "gradient_norm": math.nan,
            **{f"val_{name}": initial[name] for name in METRIC_NAMES},
            **(
                {
                    f"online_val_{name}": online_initial[name]
                    for name in METRIC_NAMES
                }
                if online_initial is not None
                else {}
            ),
            **(
                {"effective_ema_decay": 0.0}
                if EMA_TARGET_DECAY is not None
                else {}
            ),
            "lr": 0.0,
            "wall_s": time.time() - started,
            "unique_tokens_per_s": 0.0,
            "ce_labels_per_s": 0.0,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        writer.writerow(initial_row)
        handle.flush()
        rows.append(initial_row)
        initial_message = (
            f"step=   0 nll={initial['h1_validation_nll']:.4f} "
            f"{CLOSURE_DISPLAY_NAME}="
            f"{initial[CLOSURE_VALIDATION_METRIC_NAME]:.4f} "
            f"arcos={initial['mean_overall_cosine']:.4f} "
            f"tok_h2-h4={initial['h2_h4_token_agreement']:.3f}"
        )
        if TEMPERED_READOUT:
            initial_message += (
                f" beta={initial['learned_read_beta']:.4f}"
            )
        print(initial_message, flush=True)

        for index in range(start_step, args.steps):
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
            train_loss = train_ce = train_closure = 0.0
            micro_windows = window.split(args.microbatch)
            for micro_window in micro_windows:
                training_objective = (
                    TRAIN_OBJECTIVE_OVERRIDE
                    if TRAIN_OBJECTIVE_OVERRIDE is not None
                    else objective
                )
                loss, token_ce, closure_loss, output = training_objective(
                    model,
                    micro_window,
                    latent_target_model=ema_model,
                )
                scale = 1.0 / len(micro_windows)
                (loss * scale).backward()
                train_loss += float(loss.detach()) * scale
                train_ce += float(token_ce.detach()) * scale
                train_closure += float(closure_loss.detach()) * scale
                del loss, token_ce, closure_loss, output
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            )
            optimizer.step()
            step = index + 1
            effective_ema_decay = math.nan
            if ema_model is not None:
                effective_ema_decay = ema_decay_at_step(
                    step,
                    EMA_TARGET_DECAY,
                    EMA_WARM_START_STEPS,
                )
                update_ema_model(
                    ema_model,
                    model,
                    effective_ema_decay,
                )
            processed_tokens += args.batch * TRAIN_WINDOW_LENGTH
            processed_labels += args.batch * TRAIN_ANCHORS
            if step in REPORT_STEPS or step == args.steps:
                metrics = evaluate(
                    ema_model if EVALUATE_WITH_EMA else model,
                    validation,
                    validation_starts,
                    args.eval_microbatch,
                )
                online_metrics = (
                    evaluate(
                        model,
                        validation,
                        validation_starts,
                        args.eval_microbatch,
                        latent_target_model=ema_model,
                    )
                    if REPORT_ONLINE_WITH_EMA
                    else None
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "train_token_ce": train_ce,
                    closure_train_field: train_closure,
                    "gradient_norm": gradient_norm,
                    **{
                        f"val_{name}": metrics[name]
                        for name in METRIC_NAMES
                    },
                    **(
                        {
                            f"online_val_{name}": online_metrics[name]
                            for name in METRIC_NAMES
                        }
                        if online_metrics is not None
                        else {}
                    ),
                    **(
                        {"effective_ema_decay": effective_ema_decay}
                        if EMA_TARGET_DECAY is not None
                        else {}
                    ),
                    "lr": lr,
                    "wall_s": wall,
                    "unique_tokens_per_s": processed_tokens / wall,
                    "ce_labels_per_s": processed_labels / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                message = (
                    f"step={step:4d} ce={train_ce:.4f} "
                    f"{CLOSURE_DISPLAY_NAME}={train_closure:.4f} "
                    f"nll={metrics['h1_validation_nll']:.4f} "
                    f"arcos={metrics['mean_overall_cosine']:.4f} "
                    f"tok_h2-h4={metrics['h2_h4_token_agreement']:.3f}"
                )
                if TEMPERED_READOUT:
                    message += (
                        f" beta={metrics['learned_read_beta']:.4f}"
                    )
                if EMA_TARGET_DECAY is not None:
                    message += f" ema={effective_ema_decay:.4f}"
                print(message, flush=True)
                save_checkpoint(
                    args.output_dir / "last.pt",
                    model,
                    optimizer,
                    data_generator,
                    step,
                    metrics,
                    args,
                    ema_model=ema_model,
                )
                if step in (100, 250, 500, 750, 1000) or step == args.steps:
                    save_checkpoint(
                        args.output_dir / f"step{step:04d}.pt",
                        model,
                        optimizer,
                        data_generator,
                        step,
                        metrics,
                        args,
                        ema_model=ema_model,
                    )

    final = rows[-1]
    checks = {
        "h1_nll_improved": final["val_h1_validation_nll"]
        < initial_row["val_h1_validation_nll"],
        "h2_h4_token_agreement_threshold": final[
            "val_h2_h4_token_agreement"
        ]
        >= MIN_H2_H4_TOKEN_AGREEMENT,
        "mean_ar_state_cosine_threshold": final["val_mean_overall_cosine"]
        >= MIN_MEAN_AR_STATE_COSINE,
        "h1_path_tolerance": final["val_h1_ar_parallel_max_logit_error"]
        < 5e-4,
    }
    if REQUIRE_WRITE_EXCEEDS_NO_WRITE:
        checks["write_exceeds_no_write"] = (
            final["val_h2_h4_token_agreement"]
            > final["val_h2_h4_no_write_token_agreement"]
        )
    if REQUIRE_H1_LATENT_MSE_IMPROVEMENT:
        checks["h1_latent_mse_improved"] = (
            final["val_h1_latent_relative_mse"]
            < initial_row["val_h1_latent_relative_mse"]
        )
    if TEMPERED_READOUT:
        checks["posterior_tempered_readouts_distinct"] = (
            final["val_posterior_readout_logit_abs_delta"] > 0.0
        )
        checks["beta_strictly_interior"] = (
            0.01 < final["val_learned_read_beta"] < 0.99
        )
    else:
        checks["posterior_prior_readouts_distinct"] = (
            final["val_posterior_readout_logit_abs_delta"] > 0.0
        )
    if MIN_H1_POSTERIOR_TARGET_ACCURACY is not None:
        checks["h1_posterior_target_accuracy_threshold"] = (
            final["val_h1_posterior_target_accuracy"]
            > MIN_H1_POSTERIOR_TARGET_ACCURACY
        )
    if STEP100_REFERENCE is not None and args.steps >= 100:
        step100_rows = [row for row in rows if int(row["step"]) == 100]
        if len(step100_rows) != 1:
            raise RuntimeError("step-100 reference requested without one row")
        step100 = step100_rows[0]
        for metric_name, reference_value in STEP100_REFERENCE.items():
            checks[f"step100_matches_{metric_name}"] = (
                abs(step100[metric_name] - reference_value)
                <= STEP100_REFERENCE_TOLERANCE
            )
    checks.update(additional_final_checks(initial_row, final, rows))
    passed = all(checks.values())
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("final_step", int(final["step"])))
        writer.writerow(("mechanism_criterion_passed", str(passed).lower()))
        for name, value in checks.items():
            writer.writerow((name, str(value).lower()))
        writer.writerow(("wall_s", final["wall_s"]))
        writer.writerow(("peak_vram_bytes", int(final["peak_vram_bytes"])))

    verdict = "passed" if passed else "did not pass"
    lines = [
        "# Interpretation",
        "",
        f"The preregistered initial mechanism criterion {verdict}.",
        "",
        "The central recurrence consumed only its own continuous predictions. "
        "The greedy-AR path was used only by the evaluator and never supplied "
        "a token, embedding, hidden state, probability, or gradient to the "
        "trained recurrence.",
        (
            TRAINING_TARGET_DESCRIPTION_OVERRIDE
            if TRAINING_TARGET_DESCRIPTION_OVERRIDE is not None
            else "The H1 target came from a gradient-free full-model EMA, "
            "and all reported primary inference metrics used that same EMA "
            "snapshot."
            if EMA_TARGET_DECAY is not None
            else "The H1 online encoder target was stop-gradient."
            if DETACH_LATENT_TARGET
            else "The H1 online encoder target remained attached."
        ),
        "",
        "| Metric | Step 0 | Final |",
        "|---|---:|---:|",
        f"| H1 validation NLL | {initial_row['val_h1_validation_nll']:.6f} | {final['val_h1_validation_nll']:.6f} |",
        f"| H1 latent relative MSE | {initial_row['val_h1_latent_relative_mse']:.6f} | {final['val_h1_latent_relative_mse']:.6f} |",
        f"| H1 posterior target accuracy | {initial_row['val_h1_posterior_target_accuracy']:.6f} | {final['val_h1_posterior_target_accuracy']:.6f} |",
        f"| Mean continuous/AR state cosine | {initial_row['val_mean_overall_cosine']:.6f} | {final['val_mean_overall_cosine']:.6f} |",
        f"| H2--H4 token agreement | {initial_row['val_h2_h4_token_agreement']:.6f} | {final['val_h2_h4_token_agreement']:.6f} |",
        f"| No-write H2--H4 agreement | {initial_row['val_h2_h4_no_write_token_agreement']:.6f} | {final['val_h2_h4_no_write_token_agreement']:.6f} |",
        f"| Posterior/readout mean logit delta | {initial_row['val_posterior_readout_logit_abs_delta']:.6f} | {final['val_posterior_readout_logit_abs_delta']:.6f} |",
        "",
    ]
    lines.extend(additional_interpretation_lines(initial_row, final, rows))
    if TEMPERED_READOUT:
        lines.insert(
            -1,
            f"| Learned beta | {initial_row['val_learned_read_beta']:.6f} | {final['val_learned_read_beta']:.6f} |",
        )
    (args.record_dir / "interpretation.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
