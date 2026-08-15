"""Train post-update recurrence with EMA self-redecode forward KL."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.training.ha_skew_window import (
    forward_sparse_complex_self_prediction_self_redecode_kl,
    sparse_complex_self_prediction_self_redecode_kl_loss,
)
import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base


EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "post-update-full-read-h1-ema-self-redecode-kl-ce-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
KL_WEIGHT = 1.0
HIDDEN_MSE_WEIGHT = 0.0
DETACH_PRELIMINARY_FOR_UPDATE = False
MIN_H1_P_Z_TOP1_AGREEMENT = 0.40087890625
MIN_H1_Z_MAX_PROBABILITY = 0.46046939492225647
MIN_H1_Z_GE99_FRACTION = 0.130615234375

ORIGINAL_EVALUATE = base.evaluate
BASE_METRIC_NAMES = base.METRIC_NAMES
REDECODE_METRIC_NAMES = (
    "h1_self_redecode_forward_kl",
    "h1_raw_self_redecode_forward_kl",
    "h1_self_redecode_kl_ratio",
    "h1_self_redecode_kl_plus_hidden_mse",
    "h1_canonical_selected_probability",
    "h1_student_selected_probability",
    "h1_student_selected_top1_agreement",
    "h1_raw_ema_selected_top1_agreement",
    "h1_student_max_probability",
    "h1_student_max_probability_ge_099",
    "h1_student_entropy",
    "h1_student_canonical_relative_mse",
    "h1_student_canonical_cosine",
)


def objective(model, window, *, latent_target_model=None):
    if latent_target_model is None:
        raise ValueError("EMA self-redecode KL requires a full-model EMA")
    return sparse_complex_self_prediction_self_redecode_kl_loss(
        model,
        latent_target_model,
        window,
        prefix_length=base.CONTEXT,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        kl_weight=KL_WEIGHT,
        hidden_mse_weight=HIDDEN_MSE_WEIGHT,
        detach_preliminary_for_update=DETACH_PRELIMINARY_FOR_UPDATE,
    )


def _selected_probability(
    log_probs: torch.Tensor,
    selected_tokens: torch.Tensor,
) -> torch.Tensor:
    return log_probs.exp().gather(
        -1,
        selected_tokens.unsqueeze(-1),
    ).squeeze(-1)


@torch.inference_mode()
def evaluate(
    model,
    validation,
    starts,
    microbatch: int,
    *,
    latent_target_model=None,
) -> dict[str, float]:
    metrics = ORIGINAL_EVALUATE(
        model,
        validation,
        starts,
        microbatch,
        latent_target_model=latent_target_model,
    )
    redecode_model = latent_target_model or model
    sums = {name: 0.0 for name in REDECODE_METRIC_NAMES}
    rows = 0
    for begin in range(0, len(starts), microbatch):
        window = base.windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            base.EVAL_WINDOW_LENGTH,
        )[:, : base.TRAIN_WINDOW_LENGTH]
        output = forward_sparse_complex_self_prediction_self_redecode_kl(
            model,
            redecode_model,
            window,
            prefix_length=base.CONTEXT,
            anchor_stride=base.EVAL_ANCHOR_STRIDE,
        )
        count = output.selected_tokens.numel()
        rows += count
        canonical_probs = output.canonical_log_probs.exp()
        student_probs = output.student_log_probs.exp()
        student_max = student_probs.amax(dim=-1)
        canonical_selected = _selected_probability(
            output.canonical_log_probs,
            output.selected_tokens,
        )
        student_selected = _selected_probability(
            output.student_log_probs,
            output.selected_tokens,
        )
        student_entropy = -(
            student_probs * output.student_log_probs
        ).sum(dim=-1)
        values = {
            "h1_self_redecode_forward_kl": output.canonical_student_kl,
            "h1_raw_self_redecode_forward_kl": output.canonical_raw_kl,
            "h1_canonical_selected_probability": canonical_selected,
            "h1_student_selected_probability": student_selected,
            "h1_student_selected_top1_agreement": (
                output.student_logits.argmax(dim=-1).eq(
                    output.selected_tokens
                )
            ).float(),
            "h1_raw_ema_selected_top1_agreement": (
                output.raw_ema_logits.argmax(dim=-1).eq(
                    output.selected_tokens
                )
            ).float(),
            "h1_student_max_probability": student_max,
            "h1_student_max_probability_ge_099": student_max.ge(0.99).float(),
            "h1_student_entropy": student_entropy,
            "h1_student_canonical_relative_mse": (
                output.canonical_relative_mse
            ),
            "h1_student_canonical_cosine": F.cosine_similarity(
                output.base.full_states.float(),
                output.canonical_states.float(),
                dim=-1,
            ),
        }
        for name, value in values.items():
            sums[name] += float(value.double().sum())
        del canonical_probs

    if rows < 1:
        raise RuntimeError("EMA redecode evaluation produced no rows")
    for name in REDECODE_METRIC_NAMES:
        if name not in (
            "h1_self_redecode_kl_ratio",
            "h1_self_redecode_kl_plus_hidden_mse",
        ):
            metrics[name] = sums[name] / rows
    metrics["h1_self_redecode_kl_ratio"] = (
        metrics["h1_self_redecode_forward_kl"]
        / max(metrics["h1_raw_self_redecode_forward_kl"], 1e-12)
    )
    metrics["h1_self_redecode_kl_plus_hidden_mse"] = (
        metrics["h1_self_redecode_forward_kl"]
        + metrics["h1_student_canonical_relative_mse"]
    )
    return metrics


def _require_gradient(name: str, gradient: torch.Tensor | None) -> float:
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.norm() > 0)
    ):
        raise RuntimeError(f"no finite nonzero KL gradient to {name}")
    return float(gradient.norm())


def preflight(model, training, *, latent_target_model=None) -> dict[str, float]:
    if latent_target_model is None:
        raise RuntimeError("preflight requires an EMA model")
    generator = torch.Generator().manual_seed(base.SEED + 11831)
    window = base.sampled_windows(
        training,
        2,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    loss, token_ce, closure_loss, output = objective(
        model,
        window,
        latent_target_model=latent_target_model,
    )
    redecode_kl = output.canonical_student_kl.mean()
    hidden_mse = output.canonical_relative_mse.mean()
    finite_values = (
        loss,
        token_ce,
        closure_loss,
        redecode_kl,
        hidden_mse,
        output.base.logits,
        output.base.full_states,
        output.canonical_logits,
        output.student_logits,
        output.canonical_student_kl,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise RuntimeError("non-finite EMA self-redecode preflight value")
    if not torch.equal(
        output.selected_tokens,
        output.base.logits.detach().argmax(dim=-1),
    ):
        raise RuntimeError("self-selected token differs from online P argmax")
    if not torch.equal(
        output.canonical_logits.argmax(dim=-1),
        output.selected_tokens,
    ):
        raise RuntimeError("EMA canonical redecode changed the selected token")
    if (
        output.canonical_states.requires_grad
        or output.canonical_logits.requires_grad
        or output.ema_prefix_encoded.requires_grad
    ):
        raise RuntimeError("EMA canonical branch is not gradient-free")
    if any(
        parameter.requires_grad
        for parameter in latent_target_model.parameters()
    ):
        raise RuntimeError("EMA decoder parameters require gradients")
    if not output.student_logits.requires_grad:
        raise RuntimeError("frozen EMA decoder cut the Znext input gradient")

    parameters = [
        output.base.full_states,
        model.complex_self_prediction.query.weight,
        model.complex_self_prediction.key.weight,
        model.complex_self_prediction.value.weight,
        model.complex_self_prediction.output.weight,
        model.complex_self_prediction.memory_phase,
        model.encoder.blocks[-1].attn.qkv.weight,
    ]
    parameter_names = [
        "Znext",
        "query",
        "key",
        "value",
        "complex readout",
        "memory phase",
        "online encoder final block",
    ]
    if not DETACH_PRELIMINARY_FOR_UPDATE:
        parameters.insert(5, model.complex_self_prediction.hidden_phase)
        parameter_names.insert(5, "hidden phase")
    gradients = torch.autograd.grad(
        closure_loss,
        tuple(parameters),
        retain_graph=True,
    )
    gradient_norms = {
        f"kl_{name.replace(' ', '_')}_gradient_norm": _require_gradient(
            name,
            gradient,
        )
        for name, gradient in zip(parameter_names, gradients)
    }
    preliminary_edge_gradient_norm = float("nan")
    detach_forward_error = 0.0
    if DETACH_PRELIMINARY_FOR_UPDATE:
        hidden_phase_gradient = torch.autograd.grad(
            closure_loss,
            model.complex_self_prediction.hidden_phase,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if hidden_phase_gradient is not None and bool(
            hidden_phase_gradient.norm() > 0
        ):
            raise RuntimeError("closure bypassed the detached P boundary")

        attached = forward_sparse_complex_self_prediction_self_redecode_kl(
            model,
            latent_target_model,
            window,
            prefix_length=base.CONTEXT,
            anchor_stride=base.TRAIN_ANCHOR_STRIDE,
            detach_preliminary_for_update=False,
        )
        detach_forward_error = max(
            float((left - right).detach().abs().max())
            for left, right in (
                (output.base.logits, attached.base.logits),
                (output.base.full_states, attached.base.full_states),
                (output.student_logits, attached.student_logits),
            )
        )
        if detach_forward_error != 0.0:
            raise RuntimeError("detaching P changed forward values")

        roots = output.base.prefix_encoded.index_select(
            1,
            output.base.anchor_indices,
        ).detach()
        transition = model.complex_self_prediction
        probe_step = transition(
            transition.initialize(roots),
            detach_preliminary_for_update=True,
        )
        preliminary_edge_gradient = torch.autograd.grad(
            probe_step.state.hidden.square().mean(),
            probe_step.preliminary_hidden,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if preliminary_edge_gradient is not None and bool(
            preliminary_edge_gradient.norm() > 0
        ):
            raise RuntimeError("corrector retained a gradient edge to P")
        preliminary_edge_gradient_norm = 0.0
    if any(
        parameter.grad is not None
        for parameter in latent_target_model.parameters()
    ):
        raise RuntimeError("EMA parameter accumulated a gradient")

    changed_window = window.clone()
    changed_window[:, -1] = (
        changed_window[:, -1] + 1
    ) % base.VOCABULARY
    changed = forward_sparse_complex_self_prediction_self_redecode_kl(
        model,
        latent_target_model,
        changed_window,
        prefix_length=base.CONTEXT,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        detach_preliminary_for_update=DETACH_PRELIMINARY_FOR_UPDATE,
    )
    future_nonleakage_error = max(
        float((left - right).detach().abs().max())
        for left, right in (
            (output.base.logits, changed.base.logits),
            (output.base.full_states, changed.base.full_states),
            (output.selected_tokens, changed.selected_tokens),
            (output.canonical_states, changed.canonical_states),
            (output.canonical_logits, changed.canonical_logits),
            (output.student_logits, changed.student_logits),
        )
    )
    if future_nonleakage_error != 0.0:
        raise RuntimeError("corpus future byte leaked into self-redecode path")

    probe_anchor = 7
    literal_tokens = torch.cat(
        (
            window[:, : probe_anchor + 1],
            output.selected_tokens[:, probe_anchor, 0, None],
        ),
        dim=1,
    )
    with torch.no_grad():
        literal_encoded, _ = latent_target_model.encode(literal_tokens)
    literal_error = float(
        (
            literal_encoded[:, -1]
            - output.canonical_states[:, probe_anchor, 0]
        ).abs().max()
    )
    if literal_error >= 5e-5:
        raise RuntimeError("batched EMA self-action differs from literal encode")

    selected_probability = float(
        _selected_probability(
            output.canonical_log_probs,
            output.selected_tokens,
        ).mean().detach()
    )
    student_probability = float(
        _selected_probability(
            output.student_log_probs,
            output.selected_tokens,
        ).mean().detach()
    )
    total_parameters = base.parameter_count(model)
    if total_parameters != base.EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"parameter total {total_parameters} != {base.EXPECTED_PARAMETERS}"
        )
    if model.complex_self_prediction.beta is not None:
        raise RuntimeError("coefficient-free KL run unexpectedly has beta")
    model.zero_grad(set_to_none=True)
    return {
        "initial_loss": float(loss.detach()),
        "initial_token_ce": float(token_ce.detach()),
        "initial_closure_loss": float(closure_loss.detach()),
        "initial_latent_relative_mse": float(
            hidden_mse.detach()
        ),
        "initial_self_redecode_forward_kl": float(redecode_kl.detach()),
        "initial_raw_self_redecode_forward_kl": float(
            output.canonical_raw_kl.mean().detach()
        ),
        "initial_canonical_selected_probability": selected_probability,
        "initial_student_selected_probability": student_probability,
        "future_state_leak_error": future_nonleakage_error,
        "literal_self_action_error": literal_error,
        "detach_forward_error": detach_forward_error,
        "corrector_p_edge_gradient_norm": preliminary_edge_gradient_norm,
        **gradient_norms,
    }


def additional_final_checks(initial_row, final_row, rows):
    del initial_row, rows
    return {
        "h1_p_z_top1_agreement_improved_over_mse": final_row[
            "val_h1_student_selected_top1_agreement"
        ]
        > MIN_H1_P_Z_TOP1_AGREEMENT,
        "h1_z_max_probability_improved_over_mse": final_row[
            "val_h1_student_max_probability"
        ]
        > MIN_H1_Z_MAX_PROBABILITY,
        "h1_z_ge99_fraction_improved_over_mse": final_row[
            "val_h1_student_max_probability_ge_099"
        ]
        > MIN_H1_Z_GE99_FRACTION,
        "znext_kl_below_raw_p_kl": final_row[
            "val_h1_self_redecode_forward_kl"
        ]
        < final_row["val_h1_raw_self_redecode_forward_kl"],
    }


def additional_interpretation_lines(initial_row, final_row, rows):
    del rows
    return [
        "## EMA self-redecode closure",
        "",
        "| Metric | Step 0 | Final |",
        "|---|---:|---:|",
        f"| Canonical-to-Znext KL | {initial_row['val_h1_self_redecode_forward_kl']:.6f} | {final_row['val_h1_self_redecode_forward_kl']:.6f} |",
        f"| Canonical-to-P KL | {initial_row['val_h1_raw_self_redecode_forward_kl']:.6f} | {final_row['val_h1_raw_self_redecode_forward_kl']:.6f} |",
        f"| Znext/P selected top-1 | {initial_row['val_h1_student_selected_top1_agreement']:.6f} | {final_row['val_h1_student_selected_top1_agreement']:.6f} |",
        f"| Znext selected probability | {initial_row['val_h1_student_selected_probability']:.6f} | {final_row['val_h1_student_selected_probability']:.6f} |",
        f"| Znext max probability | {initial_row['val_h1_student_max_probability']:.6f} | {final_row['val_h1_student_max_probability']:.6f} |",
        f"| Znext max probability >= 0.99 | {initial_row['val_h1_student_max_probability_ge_099']:.6f} | {final_row['val_h1_student_max_probability_ge_099']:.6f} |",
        f"| Znext entropy | {initial_row['val_h1_student_entropy']:.6f} | {final_row['val_h1_student_entropy']:.6f} |",
        f"| Znext/canonical relative MSE (diagnostic) | {initial_row['val_h1_student_canonical_relative_mse']:.6f} | {final_row['val_h1_student_canonical_relative_mse']:.6f} |",
        "",
    ]


base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.TEMPERED_READOUT = False
base.RESIDUAL_PRIOR = False
base.RECURRENT_HIDDEN_MODE = "post-update-full-read"
base.DETACH_LATENT_TARGET = False
base.EMA_TARGET_DECAY = 0.99
base.EMA_WARM_START_STEPS = 100
base.EVALUATE_WITH_EMA = True
base.REPORT_ONLINE_WITH_EMA = True
base.REPORT_STEPS = frozenset(
    (1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
)
base.EXPECTED_PARAMETERS = 13_216_352
base.MIN_H2_H4_TOKEN_AGREEMENT = 0.0
base.MIN_MEAN_AR_STATE_COSINE = -1.0
base.MIN_H1_POSTERIOR_TARGET_ACCURACY = None
base.STEP100_REFERENCE = None
base.LATENT_WEIGHT = KL_WEIGHT
base.CLOSURE_METRIC_NAME = "self_redecode_forward_kl"
base.CLOSURE_DISPLAY_NAME = "kl"
base.CLOSURE_VALIDATION_METRIC_NAME = "h1_self_redecode_forward_kl"
base.OBJECTIVE_NAME_OVERRIDE = (
    "raw_tied_ce_plus_ema_self_selected_redecode_forward_kl"
)
base.CLOSURE_TARGET_NAME_OVERRIDE = (
    "self_selected_token_ema_encode_inverse_decode_distribution"
)
base.REQUIRE_H1_LATENT_MSE_IMPROVEMENT = False
base.REQUIRE_WRITE_EXCEEDS_NO_WRITE = False
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The closure target was the gradient-free EMA encode-and-immediate-"
    "redecode distribution of P's own greedy token. Znext was decoded through "
    "the frozen EMA inverse, which preserved gradients only to the online "
    "Znext input."
)
base.objective = objective
base.preflight = preflight
base.evaluate = evaluate
base.METRIC_NAMES = (*BASE_METRIC_NAMES, *REDECODE_METRIC_NAMES)
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
