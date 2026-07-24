"""Joint CE + stop-gradient state MSE, K=1, 13M: Householder K + ActNorm +
learned per-channel reparameterized noise (VAE-style).

Same Design B decoupling as the fixed-epsilon sibling (MSE always compares
the CLEAN state to h_(t+1); only the CE-decode path sees a perturbed state),
but instead of one global scalar epsilon, a small learned linear map predicts
a per-channel, per-example log-variance from the (RMS-normalized) state
itself, and the perturbation uses the reparameterization trick
(m = exp(log_sigma)*eps, eps~N(0,I)) so gradient flows through log_sigma via
CE alone. No explicit regularizer or KL term on log_sigma is added, per the
user-supplied design: log_sigma's weight starts at 0 and its bias starts at
log(0.05), so training starts exactly equivalent to the fixed-epsilon
sibling and is free to move from there.
See EXP-20260722-query-k1-householder-actnorm-learned-noise-mse-ce-joint-13m.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from rotlm.models.query_k1_inverse import QueryK1InverseLM
from train_k1_decoder_inverse_ablation_13m import (
    CLIP_NORM,
    CONTEXT,
    PEAK_LR,
    REPORT_STEPS,
    SEED,
    VOCABULARY,
    active_head_self_retrieval,
    lr_at,
    memmap,
    parameter_count,
    sampled_windows,
    trainable_parameter_count,
    windows,
)


STEPS = 1000
WIDTH = 896
INIT_SIGMA = 0.05
RECORD = Path("experiments/records/EXP-20260722-query-k1-householder-actnorm-learned-noise-mse-ce-joint-13m")
OUTPUT = Path("outputs/experiments/EXP-20260722-query-k1-householder-actnorm-learned-noise-mse-ce-joint-13m")


NUM_REFLECTIONS = 32


class HouseholderOperator(nn.Module):
    """K = H_1 ... H_m, each H_i a Householder reflection.

    Exactly orthogonal for any parameter values. Vectors are stored in
    identical pairs at init (v_{2i} = v_{2i+1}), so each pair's product is
    the identity (H H = I) and K = I exactly at init; the two vectors in a
    pair are independent parameters free to diverge under training.
    """

    def __init__(self, width: int, num_reflections: int = NUM_REFLECTIONS) -> None:
        super().__init__()
        if num_reflections % 2:
            raise ValueError("num_reflections must be even for identity init")
        self.width = width
        self.num_reflections = num_reflections
        base = torch.randn(num_reflections // 2, width)
        base = base / base.norm(dim=-1, keepdim=True)
        vectors_init = base.repeat_interleave(2, dim=0)
        self.vectors = nn.Parameter(vectors_init.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for index in range(self.num_reflections):
            v = self.vectors[index]
            v_hat = v / v.norm().clamp_min(1e-12)
            y = y - 2 * (y @ v_hat).unsqueeze(-1) * v_hat
        return y

    @property
    def weight(self) -> torch.Tensor:
        """Materialize the dense width x width matrix, for diagnostics only."""
        return self.forward(torch.eye(self.width, device=self.vectors.device))


class ActNorm(nn.Module):
    """Per-channel affine, exactly invertible, identity at init."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(self.log_scale) + self.bias

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.bias) * torch.exp(-self.log_scale)


class SigmaPredictor(nn.Module):
    """Predicts per-channel log-sigma from the (RMS-normalized) state.

    Weight starts at 0 and bias at log(INIT_SIGMA), so at init every channel's
    sigma equals INIT_SIGMA regardless of the input -- identical starting
    behavior to the fixed-epsilon sibling. Input is RMS-normalized to 1 per
    dimension before this layer so its weights do not have to compensate for
    ||u||, which varies by close to an order of magnitude across checkpoints.
    """

    def __init__(self, width: int, init_sigma: float = INIT_SIGMA) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=True)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(math.log(init_sigma))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        per_dim_scale = u.detach().norm(dim=-1, keepdim=True) / (u.shape[-1] ** 0.5)
        u_hat = u.detach() / per_dim_scale.clamp_min(1e-12)
        return self.linear(u_hat)


def make_model() -> QueryK1InverseLM:
    model = QueryK1InverseLM(
        VOCABULARY, width=WIDTH, encoder_blocks=2,
        head_mode="rms-tied", trainable_cosine_scale=True,
    )
    model.operator = HouseholderOperator(WIDTH)
    model.actnorm = ActNorm(WIDTH)
    model.sigma_predictor = SigmaPredictor(WIDTH)
    return model


def forward_with_actnorm(
    model: QueryK1InverseLM,
    tokens: torch.Tensor,
    *,
    inject_noise: bool = False,
):
    """Mirror QueryK1InverseLM.forward, inserting ActNorm after K.

    Design B (decoupled), learned per-channel noise: when ``inject_noise`` is
    true (training only), a per-channel, per-example log-sigma is predicted
    from the clean state (RMS-normalized first), and Gaussian noise scaled by
    exp(log_sigma) times the state's own RMS-per-dimension is added ONLY to
    the state that gets decoded for CE -- the state returned for the MSE
    comparison (and every other diagnostic) is always the clean, noise-free
    ActNorm'd state, exactly as in the fixed-epsilon sibling. Disabled at
    eval either way. ``log_sigma`` is returned (None if inject_noise=False)
    so its trajectory can be logged.
    """
    prefix_states, query_states, positions = model.dense_query_encode(tokens)
    clean_states = model.actnorm(model.operator(query_states))
    log_sigma = None
    if inject_noise:
        log_sigma = model.sigma_predictor(clean_states)
        per_dim_scale = clean_states.detach().norm(dim=-1, keepdim=True) / (
            clean_states.shape[-1] ** 0.5
        )
        decode_states = clean_states + torch.exp(log_sigma) * per_dim_scale * torch.randn_like(
            clean_states
        )
    else:
        decode_states = clean_states
    inverse_hidden = model.exact_inverse_dense_decode_states(
        prefix_states, decode_states, positions,
    )
    logits = model.token_logits(inverse_hidden)
    return logits, clean_states, prefix_states, positions, log_sigma


def relative_mse_rows(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def gold_next_states(model: QueryK1InverseLM, window: torch.Tensor) -> torch.Tensor:
    full_positions = torch.arange(window.shape[1], device=window.device)
    with torch.no_grad():
        full_encoded, _ = model.encode(window, full_positions)
    return full_encoded[:, 1:]


def loss_terms(model: QueryK1InverseLM, window: torch.Tensor):
    inputs, targets = window[:, :-1], window[:, 1:]
    logits, operator_states, prefix_states, positions, log_sigma = forward_with_actnorm(
        model, inputs, inject_noise=True,
    )
    ce = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
    gold = gold_next_states(model, window)
    mse = relative_mse_rows(operator_states, gold).mean()
    loss = ce + mse
    return loss, ce, mse, (logits, operator_states, prefix_states, positions, log_sigma), targets


@dataclass(frozen=True)
class Metrics:
    nll: float
    accuracy: float
    combined_loss: float
    ce_loss: float
    mse_loss: float
    gold_state_relative_mse: float
    gold_state_cosine: float
    action_closure_relative_mse: float
    action_closure_cosine: float
    operator_gradient_norm: float
    query_gradient_norm: float
    operator_orthogonality_error: float
    log_sigma_mean: float
    log_sigma_min: float
    log_sigma_max: float


def orthogonality_error(model: QueryK1InverseLM) -> float:
    weight = model.operator.weight.detach()
    identity = torch.eye(weight.shape[0], device=weight.device)
    return float((weight.T @ weight - identity).abs().max())


@torch.inference_mode()
def evaluate(
    model: QueryK1InverseLM,
    validation,
    starts: torch.Tensor,
    micro: int,
    operator_gradient_norm: float,
    query_gradient_norm: float,
) -> Metrics:
    model.eval()
    total_loss = total_correct = total_tokens = 0
    combined_sum = ce_sum = mse_sum = 0.0
    gold_state_mse_sum = gold_state_cos_sum = 0.0
    action_mse_sum = action_cos_sum = 0.0
    log_sigma_sum = 0.0
    log_sigma_count = 0
    log_sigma_min = math.inf
    log_sigma_max = -math.inf
    for begin in range(0, len(starts), micro):
        window = windows(validation, starts[begin: begin + micro])
        inputs, targets = window[:, :-1], window[:, 1:]
        logits, operator_states, _, positions, _ = forward_with_actnorm(model, inputs)
        # log_sigma is not part of the clean (no-noise) eval path; probe it
        # separately, purely for logging, without affecting logits/accuracy.
        probe_log_sigma = model.sigma_predictor(operator_states)
        log_sigma_sum += float(probe_log_sigma.sum())
        log_sigma_count += probe_log_sigma.numel()
        log_sigma_min = min(log_sigma_min, float(probe_log_sigma.min()))
        log_sigma_max = max(log_sigma_max, float(probe_log_sigma.max()))
        ce_row = F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), reduction="sum",
        )
        gold = gold_next_states(model, window)
        mse_rows = relative_mse_rows(operator_states, gold)
        total_loss += float(ce_row)
        combined_sum += float(ce_row) + float(mse_rows.sum())
        ce_sum += float(ce_row)
        mse_sum += float(mse_rows.sum())
        actions = logits.argmax(dim=-1)
        total_correct += int(actions.eq(targets).sum())
        total_tokens += targets.numel()

        gold_state_mse_sum += float(relative_mse_rows(operator_states, gold).sum())
        gold_state_cos_sum += float(
            F.cosine_similarity(operator_states.float(), gold.float(), dim=-1).sum()
        )
        action_states = model.dense_action_encode(inputs, actions, positions)
        action_mse = relative_mse_rows(operator_states, action_states)
        action_cos = F.cosine_similarity(
            operator_states.float(), action_states.float(), dim=-1,
        )
        action_mse_sum += float(action_mse.sum())
        action_cos_sum += float(action_cos.sum())

    model.train()
    return Metrics(
        nll=total_loss / total_tokens,
        accuracy=total_correct / total_tokens,
        combined_loss=combined_sum / total_tokens,
        ce_loss=ce_sum / total_tokens,
        mse_loss=mse_sum / total_tokens,
        gold_state_relative_mse=gold_state_mse_sum / total_tokens,
        gold_state_cosine=gold_state_cos_sum / total_tokens,
        action_closure_relative_mse=action_mse_sum / total_tokens,
        action_closure_cosine=action_cos_sum / total_tokens,
        operator_gradient_norm=operator_gradient_norm,
        query_gradient_norm=query_gradient_norm,
        operator_orthogonality_error=orthogonality_error(model),
        log_sigma_mean=log_sigma_sum / log_sigma_count,
        log_sigma_min=log_sigma_min,
        log_sigma_max=log_sigma_max,
    )


def preflight(model: QueryK1InverseLM, training, batch: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 4242)
    window = sampled_windows(training, min(batch, 2), generator)
    loss, ce, mse, output, _ = loss_terms(model, window)
    logits, operator_states, prefix_states, positions, log_sigma = output
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("preflight produced non-finite logits")

    log_sigma_init_diff = float((log_sigma - math.log(INIT_SIGMA)).abs().max())
    if log_sigma_init_diff >= 1e-5:
        raise RuntimeError(
            f"log_sigma not at log(INIT_SIGMA) at init: max abs diff={log_sigma_init_diff:.3e}"
        )

    ortho_err_init = orthogonality_error(model)
    if ortho_err_init >= 1e-4:
        raise RuntimeError(f"K not orthogonal at init: max abs error={ortho_err_init:.3e}")
    weight_init = model.operator.weight.detach()
    identity_diff = float((weight_init - torch.eye(WIDTH, device=weight_init.device)).abs().max())
    if identity_diff >= 1e-4:
        raise RuntimeError(f"K not identity at init (A=0): max abs diff={identity_diff:.3e}")

    actnorm_identity_diff = float(
        torch.cat((model.actnorm.log_scale, model.actnorm.bias)).abs().max()
    )
    if actnorm_identity_diff >= 1e-9:
        raise RuntimeError(
            f"ActNorm not identity at init: max abs param={actnorm_identity_diff:.3e}"
        )
    probe = torch.randn(4, WIDTH, device=model.actnorm.log_scale.device)
    actnorm_roundtrip = float(
        (model.actnorm.inverse(model.actnorm(probe)) - probe).abs().max()
    )
    if actnorm_roundtrip >= 1e-4:
        raise RuntimeError(f"ActNorm inverse failed: max abs error={actnorm_roundtrip:.3e}")

    ce_k_grad, ce_q_grad, ce_scale_grad, ce_bias_grad, ce_sigma_w_grad, ce_sigma_b_grad = (
        torch.autograd.grad(
            ce, (model.operator.vectors, model.query_embedding,
                 model.actnorm.log_scale, model.actnorm.bias,
                 model.sigma_predictor.linear.weight, model.sigma_predictor.linear.bias),
            retain_graph=True,
        )
    )
    mse_k_grad, mse_q_grad, mse_scale_grad, mse_bias_grad = torch.autograd.grad(
        mse, (model.operator.vectors, model.query_embedding,
              model.actnorm.log_scale, model.actnorm.bias),
        retain_graph=False,
    )
    for name, gradient in (
        ("ce->K.vectors", ce_k_grad), ("ce->query", ce_q_grad),
        ("ce->actnorm.log_scale", ce_scale_grad), ("ce->actnorm.bias", ce_bias_grad),
        ("ce->sigma_predictor.weight", ce_sigma_w_grad),
        ("ce->sigma_predictor.bias", ce_sigma_b_grad),
        ("mse->K.vectors", mse_k_grad), ("mse->query", mse_q_grad),
        ("mse->actnorm.log_scale", mse_scale_grad), ("mse->actnorm.bias", mse_bias_grad),
    ):
        if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm() > 0):
            raise RuntimeError(f"preflight produced no finite {name} gradient")

    with torch.no_grad():
        anchor = min(63, window.shape[1] - 2)
        tape = torch.cat(
            (prefix_states[:, : anchor + 1], operator_states[:, anchor: anchor + 1]),
            dim=1,
        )
        tape_positions = torch.arange(anchor + 2, device=window.device).unsqueeze(0).expand(
            window.shape[0], -1,
        )
        decoded = model.encoder.decode_hidden(tape, tape_positions)
        restored, _ = model.encoder.encode_hidden(decoded, tape_positions)
        inverse_roundtrip_rel_l2 = float(
            relative_mse_rows(restored, tape).mean().sqrt()
        )
    if inverse_roundtrip_rel_l2 >= 1e-4:
        raise RuntimeError(
            f"exact inverse failed: relative L2={inverse_roundtrip_rel_l2:.3e}"
        )

    with torch.no_grad():
        torch.manual_seed(SEED)
        logits_noisy, states_from_noisy_call, _, _, _ = forward_with_actnorm(
            model, window[:, :-1], inject_noise=True,
        )
        logits_clean, states_from_clean_call, _, _, _ = forward_with_actnorm(
            model, window[:, :-1], inject_noise=False,
        )
        # CE-relevant logits must differ (noise reaches the decode path)...
        noise_effect = float((logits_noisy - logits_clean).abs().max())
        # ...but the MSE-relevant state must NOT (Design B decoupling: MSE
        # never sees the noisy value, regardless of inject_noise).
        mse_state_decoupled = float(
            (states_from_noisy_call - states_from_clean_call).abs().max()
        )
        eval_a, eval_b = (
            forward_with_actnorm(model, window[:, :-1], inject_noise=False)[0] for _ in range(2)
        )
        eval_determinism = float((eval_a - eval_b).abs().max())
    if noise_effect <= 1e-8:
        raise RuntimeError(
            f"training-time noise had no effect on CE logits: max abs diff={noise_effect:.3e}"
        )
    if mse_state_decoupled > 0.0:
        raise RuntimeError(
            "Design B violated: MSE-relevant state changed with inject_noise: "
            f"max abs diff={mse_state_decoupled:.3e}"
        )
    if eval_determinism > 0.0:
        raise RuntimeError(
            f"eval path is not noise-free/deterministic: max abs diff={eval_determinism:.3e}"
        )

    values = {
        "init_sigma": INIT_SIGMA,
        "log_sigma_init_diff_max_abs": log_sigma_init_diff,
        "noise_effect_on_ce_logits_max_abs": noise_effect,
        "mse_state_decoupled_from_noise_max_abs": mse_state_decoupled,
        "eval_determinism_max_abs": eval_determinism,
        "initial_combined_loss": float(loss),
        "initial_ce": float(ce),
        "initial_mse": float(mse),
        "initial_ce_k_gradient_norm": float(ce_k_grad.float().norm()),
        "initial_mse_k_gradient_norm": float(mse_k_grad.float().norm()),
        "initial_ce_query_gradient_norm": float(ce_q_grad.float().norm()),
        "initial_mse_query_gradient_norm": float(mse_q_grad.float().norm()),
        "initial_ce_actnorm_gradient_norm": float(
            torch.cat((ce_scale_grad, ce_bias_grad)).norm()
        ),
        "initial_mse_actnorm_gradient_norm": float(
            torch.cat((mse_scale_grad, mse_bias_grad)).norm()
        ),
        "head_self_retrieval": active_head_self_retrieval(model),
        "inverse_roundtrip_rel_l2": inverse_roundtrip_rel_l2,
        "operator_orthogonality_error_init": ortho_err_init,
        "operator_identity_diff_init": identity_diff,
        "actnorm_identity_diff_init": actnorm_identity_diff,
        "actnorm_roundtrip_max_abs": actnorm_roundtrip,
    }
    model.zero_grad(set_to_none=True)
    return values


def save_checkpoint(path: Path, model, optimizer, step, metrics):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": asdict(metrics),
            "config": {
                "vocabulary": VOCABULARY,
                "context": CONTEXT,
                "seed": SEED,
                "width": WIDTH,
                "encoder_blocks": 2,
                "head_mode": "rms-tied",
                "objective": "ce_plus_stopgrad_state_mse_learned_reparam_noise",
                "operator": "householder_product_plus_actnorm_plus_sigma_predictor",
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-micro", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_examples, args.eval_micro) < 1:
        parser.error("steps and batch sizes must be positive")
    if not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    training, validation = memmap("train"), memmap("validation")
    validation_starts = torch.randint(
        0, len(validation) - CONTEXT - 1, (args.eval_examples,),
        generator=torch.Generator().manual_seed(SEED + 999),
    )
    RECORD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (RECORD / "validation_starts.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "token_start"))
        writer.writerows(enumerate(validation_starts.tolist()))

    model = make_model().cuda().train()
    preflight_values = preflight(model, training, args.batch)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"head_identity={preflight_values['head_self_retrieval']:.6f} "
        f"initial_ce={preflight_values['initial_ce']:.4f} "
        f"initial_mse={preflight_values['initial_mse']:.4f} "
        f"ortho_err={preflight_values['operator_orthogonality_error_init']:.2e} "
        f"identity_diff={preflight_values['operator_identity_diff_init']:.2e} "
        f"inverse_rt={preflight_values['inverse_roundtrip_rel_l2']:.2e}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95),
        weight_decay=0.0, fused=True,
    )
    with (RECORD / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in (
            ("seed", SEED), ("steps", args.steps), ("batch", args.batch),
            ("context", CONTEXT), ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"),
            ("objective", "ce_plus_stopgrad_state_mse_learned_reparam_noise_equal_weight"),
            ("init_sigma", INIT_SIGMA),
            ("operator", "householder_product_plus_actnorm_plus_sigma_predictor"),
            ("k_input", "contextualized_next_position_query"),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    fields = (
        "step", "train_combined", "train_ce", "train_mse",
        *tuple(f"val_{key}" for key in asdict(Metrics(*(0.0,) * 15))),
        "lr", "wall_s", "tokens_per_s", "peak_vram_bytes",
    )
    data_generator = torch.Generator().manual_seed(SEED)
    best_nll, best_step = math.inf, 0
    started = time.time()
    processed_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    with (RECORD / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(args.steps):
            lr = lr_at(index, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(training, args.batch, data_generator)
            optimizer.zero_grad(set_to_none=True)
            loss, ce, mse, _, _ = loss_terms(model, window)
            loss.backward()
            operator_gradient_norm = float(model.operator.vectors.grad.float().norm())
            query_gradient_norm = float(model.query_embedding.grad.float().norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            processed_tokens += args.batch * CONTEXT
            step = index + 1
            if step in REPORT_STEPS or step == args.steps:
                metrics = evaluate(
                    model, validation, validation_starts, args.eval_micro,
                    operator_gradient_norm, query_gradient_norm,
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    "train_combined": float(loss),
                    "train_ce": float(ce),
                    "train_mse": float(mse),
                    **{f"val_{key}": value for key, value in asdict(metrics).items()},
                    "lr": lr,
                    "wall_s": wall,
                    "tokens_per_s": processed_tokens / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d} train_ce={float(ce):.4f} train_mse={float(mse):.4f} "
                    f"val_nll={metrics.nll:.4f} acc={metrics.accuracy:.4f} "
                    f"gold_mse={metrics.gold_state_relative_mse:.4f} "
                    f"log_sigma[mean/min/max]={metrics.log_sigma_mean:.3f}/"
                    f"{metrics.log_sigma_min:.3f}/{metrics.log_sigma_max:.3f} "
                    f"ortho_err={metrics.operator_orthogonality_error:.2e} "
                    f"wall={wall:.1f}s",
                    flush=True,
                )
                save_checkpoint(OUTPUT / "last.pt", model, optimizer, step, metrics)
                if metrics.nll < best_nll:
                    best_nll, best_step = metrics.nll, step
                    save_checkpoint(OUTPUT / "best.pt", model, optimizer, step, metrics)

    with (RECORD / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("best_step", best_step))
        writer.writerow(("best_val_nll", best_nll))
        writer.writerow(("wall_s", time.time() - started))
        writer.writerow(("peak_vram_bytes", torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    main()
