"""K=1, no query: K operates on h_A (the real causal encoder state of the
current token, from ordinary causal encoding of the real prefix) instead of
a query-branch state. This fixes the domain mismatch confirmed this
session: query-branch states (encodings of [prefix, QUERY]) and
real-token-branch states (h_B = encodings of [prefix, real_token]) are
different objects, and K -- trained only on the former -- fails
catastrophically when fed the latter (e.g. via K applied to its own
output), independent of how precise the input is.

Here K's domain and range are the SAME kind of object (both are ordinary
causal states of real token sequences), so K(h_A) and K(applied again to
that output) are at least well-typed. Skew-symmetric matrix-exponential K
(exactly orthogonal), Design B stop-gradient MSE (K(h_A) vs real h_B) +
CE (decode(K(h_A)+eps) vs real next token), with LEARNED per-channel noise
(SigmaPredictor, same design as the collapsed learned-noise experiment
earlier this session) -- re-tested here because the earlier collapse was
observed in the query-branch (domain-mismatched) architecture, and it is
not yet established whether the same collapse happens once the domain
mismatch itself is removed.

2000 steps, inline h1..h5 extrapolation check every 250 steps.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
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
    sampled_windows,
    trainable_parameter_count,
    windows,
)


def windows_of_length(data, starts: torch.Tensor, length: int) -> torch.Tensor:
    values = np.stack([
        np.asarray(data[int(s): int(s) + length], dtype=np.int64) for s in starts.tolist()
    ])
    return torch.from_numpy(values).cuda(non_blocking=True)


STEPS = 2000
WIDTH = 896
INIT_SIGMA = 0.05
HORIZONS = 5
EXTRAPOLATION_SAMPLE = 256
REPORT_STEPS = frozenset((1, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000))
RECORD = Path("experiments/records/EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m")
OUTPUT = Path("outputs/experiments/EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m")


class SkewOrthogonalOperator(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.A = nn.Parameter(torch.zeros(width, width))

    def skew(self) -> torch.Tensor:
        return self.A - self.A.T

    @property
    def weight(self) -> torch.Tensor:
        return torch.matrix_exp(self.skew())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class SigmaPredictor(nn.Module):
    """log-sigma per channel/example, zero-init weight so sigma=INIT_SIGMA everywhere at init."""

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


def make_model() -> K1DecoderAblationLM:
    model = K1DecoderAblationLM(
        VOCABULARY, width=WIDTH, encoder_blocks=2,
        decoder_mode="exact-inverse", head_mode="rms-tied", trainable_cosine_scale=True,
    )
    model.operator = SkewOrthogonalOperator(WIDTH)
    model.sigma_predictor = SigmaPredictor(WIDTH)
    return model


def relative_mse_rows(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def forward_pass(model: K1DecoderAblationLM, window: torch.Tensor, *, inject_noise: bool = False):
    """window: [batch, T] real tokens. Returns logits for predicting
    window[:,1:], the clean K(h_A) states, gold h_B states, and log_sigma."""
    positions_full = torch.arange(window.shape[1], device=window.device)
    encoded, _ = model.encode(window, positions_full)
    h_a = encoded[:, :-1]
    h_b_gold = encoded[:, 1:]
    clean_states = model.operator(h_a)
    log_sigma = None
    if inject_noise:
        log_sigma = model.sigma_predictor(clean_states)
        per_dim_scale = clean_states.detach().norm(dim=-1, keepdim=True) / (clean_states.shape[-1] ** 0.5)
        decode_states = clean_states + torch.exp(log_sigma) * per_dim_scale * torch.randn_like(clean_states)
    else:
        decode_states = clean_states
    positions_prefix = positions_full[:-1]
    decoded = model.exact_inverse_dense_decode_states(h_a, decode_states, positions_prefix)
    logits = model.token_logits(decoded)
    return logits, clean_states, h_b_gold, log_sigma


def loss_terms(model: K1DecoderAblationLM, window: torch.Tensor):
    targets = window[:, 1:]
    logits, clean_states, h_b_gold, log_sigma = forward_pass(model, window, inject_noise=True)
    ce = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
    mse = relative_mse_rows(clean_states, h_b_gold).mean()
    loss = ce + mse
    return loss, ce, mse, log_sigma, logits, clean_states, h_b_gold, targets


@dataclass(frozen=True)
class Metrics:
    nll: float
    accuracy: float
    combined_loss: float
    ce_loss: float
    mse_loss: float
    gold_state_relative_mse: float
    gold_state_cosine: float
    operator_gradient_norm: float
    operator_orthogonality_error: float
    log_sigma_mean: float
    log_sigma_min: float
    log_sigma_max: float


def orthogonality_error(model: K1DecoderAblationLM) -> float:
    weight = model.operator.weight.detach()
    identity = torch.eye(weight.shape[0], device=weight.device)
    return float((weight.T @ weight - identity).abs().max())


@torch.inference_mode()
def evaluate(model: K1DecoderAblationLM, validation, starts: torch.Tensor, micro: int, operator_gradient_norm: float) -> Metrics:
    model.eval()
    total_loss = total_correct = total_tokens = 0
    combined_sum = ce_sum = mse_sum = 0.0
    gold_state_mse_sum = gold_state_cos_sum = 0.0
    log_sigma_sum = 0.0
    log_sigma_count = 0
    log_sigma_min, log_sigma_max = math.inf, -math.inf
    for begin in range(0, len(starts), micro):
        window = windows(validation, starts[begin: begin + micro])
        targets = window[:, 1:]
        logits, clean_states, h_b_gold, _ = forward_pass(model, window, inject_noise=False)
        probe_log_sigma = model.sigma_predictor(clean_states)
        log_sigma_sum += float(probe_log_sigma.sum())
        log_sigma_count += probe_log_sigma.numel()
        log_sigma_min = min(log_sigma_min, float(probe_log_sigma.min()))
        log_sigma_max = max(log_sigma_max, float(probe_log_sigma.max()))
        ce_row = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="sum")
        mse_rows = relative_mse_rows(clean_states, h_b_gold)
        total_loss += float(ce_row)
        combined_sum += float(ce_row) + float(mse_rows.sum())
        ce_sum += float(ce_row)
        mse_sum += float(mse_rows.sum())
        actions = logits.argmax(dim=-1)
        total_correct += int(actions.eq(targets).sum())
        total_tokens += targets.numel()
        gold_state_mse_sum += float(mse_rows.sum())
        gold_state_cos_sum += float(F.cosine_similarity(clean_states.float(), h_b_gold.float(), dim=-1).sum())
    model.train()
    return Metrics(
        nll=total_loss / total_tokens, accuracy=total_correct / total_tokens,
        combined_loss=combined_sum / total_tokens, ce_loss=ce_sum / total_tokens,
        mse_loss=mse_sum / total_tokens, gold_state_relative_mse=gold_state_mse_sum / total_tokens,
        gold_state_cosine=gold_state_cos_sum / total_tokens,
        operator_gradient_norm=operator_gradient_norm,
        operator_orthogonality_error=orthogonality_error(model),
        log_sigma_mean=log_sigma_sum / log_sigma_count,
        log_sigma_min=log_sigma_min, log_sigma_max=log_sigma_max,
    )


@torch.inference_mode()
def check_extrapolation(model: K1DecoderAblationLM, validation, starts: torch.Tensor, micro: int = 32) -> dict:
    model.eval()
    clean_correct = [0] * HORIZONS
    hard_correct = [0] * HORIZONS
    total = 0
    length = CONTEXT + HORIZONS
    for begin in range(0, len(starts), micro):
        window = windows_of_length(validation, starts[begin: begin + micro], length)
        prefix, targets = window[:, :CONTEXT], window[:, CONTEXT:]
        batch = prefix.shape[0]

        positions = torch.arange(CONTEXT, device=prefix.device)
        encoded, _ = model.encode(prefix, positions)
        h_a = encoded[:, -1]

        # direct-clean: parallel, K applied repeatedly to its own output, no re-encoding
        chain = [h_a]
        for _ in range(HORIZONS):
            chain.append(model.operator(chain[-1]))
        future = torch.stack(chain[1:], dim=1)
        full_positions = torch.arange(CONTEXT + HORIZONS, device=prefix.device).unsqueeze(0).expand(batch, -1)
        latent = torch.cat((encoded, future), dim=1)
        decoded = model.encoder.decode_hidden(latent, full_positions)
        pred_clean = model.token_logits(decoded[:, -HORIZONS:]).float().argmax(dim=-1)
        for h in range(HORIZONS):
            clean_correct[h] += int(pred_clean[:, h].eq(targets[:, h]).sum())

        # hard-token: real re-encoding at every step
        rolling = prefix
        for h in range(HORIZONS):
            step_positions = torch.arange(rolling.shape[1], device=prefix.device)
            step_encoded, _ = model.encode(rolling, step_positions)
            step_h_a = step_encoded[:, -1]
            step_operator_state = model.operator(step_h_a)
            step_latent = torch.cat((step_encoded, step_operator_state[:, None]), dim=1)
            step_full_positions = torch.arange(rolling.shape[1] + 1, device=prefix.device).unsqueeze(0).expand(batch, -1)
            step_decoded = model.encoder.decode_hidden(step_latent, step_full_positions)
            step_logits = model.token_logits(step_decoded[:, -1]).float()
            action = step_logits.argmax(dim=-1)
            hard_correct[h] += int(action.eq(targets[:, h]).sum())
            rolling = torch.cat((rolling, action[:, None]), dim=1)

        total += batch

    model.train()
    return {
        "direct_clean": [c / total for c in clean_correct],
        "hard_token": [c / total for c in hard_correct],
    }


def preflight(model: K1DecoderAblationLM, training, batch: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 4242)
    window = sampled_windows(training, min(batch, 2), generator)
    loss, ce, mse, log_sigma, logits, clean_states, h_b_gold, targets = loss_terms(model, window)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("preflight produced non-finite logits")

    log_sigma_init_diff = float((log_sigma - math.log(INIT_SIGMA)).abs().max())
    if log_sigma_init_diff >= 1e-5:
        raise RuntimeError(f"log_sigma not at log(INIT_SIGMA) at init: max abs diff={log_sigma_init_diff:.3e}")

    ortho_err_init = orthogonality_error(model)
    if ortho_err_init >= 1e-4:
        raise RuntimeError(f"K not orthogonal at init: max abs error={ortho_err_init:.3e}")
    weight_init = model.operator.weight.detach()
    identity_diff = float((weight_init - torch.eye(WIDTH, device=weight_init.device)).abs().max())
    if identity_diff >= 1e-4:
        raise RuntimeError(f"K not identity at init: max abs diff={identity_diff:.3e}")

    ce_k_grad, ce_sigma_grad = torch.autograd.grad(ce, (model.operator.A, model.sigma_predictor.linear.weight), retain_graph=True)
    mse_k_grad = torch.autograd.grad(mse, model.operator.A)[0]
    for name, gradient in (("ce->K.A", ce_k_grad), ("ce->sigma_predictor", ce_sigma_grad), ("mse->K.A", mse_k_grad)):
        if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm() > 0):
            raise RuntimeError(f"preflight produced no finite {name} gradient")

    with torch.no_grad():
        torch.manual_seed(SEED)
        logits_noisy, states_noisy, _, _ = forward_pass(model, window, inject_noise=True)
        logits_clean, states_clean, _, _ = forward_pass(model, window, inject_noise=False)
        noise_effect = float((logits_noisy - logits_clean).abs().max())
        mse_state_decoupled = float((states_noisy - states_clean).abs().max())
    if noise_effect <= 1e-8:
        raise RuntimeError(f"training-time noise had no effect on CE logits: max abs diff={noise_effect:.3e}")
    if mse_state_decoupled > 0.0:
        raise RuntimeError(f"Design B violated: max abs diff={mse_state_decoupled:.3e}")

    values = {
        "initial_ce": float(ce), "initial_mse": float(mse),
        "head_self_retrieval": active_head_self_retrieval(model),
        "operator_orthogonality_error_init": ortho_err_init,
        "operator_identity_diff_init": identity_diff,
        "log_sigma_init_diff": log_sigma_init_diff,
        "noise_effect_on_ce_logits_max_abs": noise_effect,
    }
    model.zero_grad(set_to_none=True)
    return values


def save_checkpoint(path: Path, model, optimizer, step: int, metrics: Metrics) -> None:
    torch.save(
        {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "metrics": asdict(metrics),
            "config": {"vocabulary": VOCABULARY, "width": WIDTH, "seed": SEED, "operator": "skew_matrix_exp_no_query"},
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
    extrapolation_starts = torch.randint(
        0, len(validation) - CONTEXT - HORIZONS - 1, (EXTRAPOLATION_SAMPLE,),
        generator=torch.Generator().manual_seed(SEED + 7777),
    )
    RECORD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    model = make_model().cuda().train()
    preflight_values = preflight(model, training, args.batch)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"initial_ce={preflight_values['initial_ce']:.4f} initial_mse={preflight_values['initial_mse']:.4f} "
        f"ortho_err={preflight_values['operator_orthogonality_error_init']:.2e}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    with (RECORD / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in (
            ("seed", SEED), ("steps", args.steps), ("batch", args.batch), ("context", CONTEXT),
            ("parameters", total_parameters), ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"), ("init_sigma", INIT_SIGMA),
            ("operator", "skew_matrix_exp_no_query"), ("k_input", "h_A_ordinary_causal_state"),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    fields = (
        "step", "train_combined", "train_ce", "train_mse",
        *tuple(f"val_{key}" for key in asdict(Metrics(*(0.0,) * 12))),
        *tuple(f"direct_clean_h{h}" for h in range(1, HORIZONS + 1)),
        *tuple(f"hard_token_h{h}" for h in range(1, HORIZONS + 1)),
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
            loss, ce, mse, _, _, _, _, _ = loss_terms(model, window)
            loss.backward()
            operator_gradient_norm = float(model.operator.A.grad.float().norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            processed_tokens += args.batch * CONTEXT
            step = index + 1
            if step in REPORT_STEPS or step == args.steps:
                metrics = evaluate(model, validation, validation_starts, args.eval_micro, operator_gradient_norm)
                extrapolation = check_extrapolation(model, validation, extrapolation_starts)
                wall = time.time() - started
                row = {
                    "step": step, "train_combined": float(loss), "train_ce": float(ce), "train_mse": float(mse),
                    **{f"val_{key}": value for key, value in asdict(metrics).items()},
                    **{f"direct_clean_h{h+1}": extrapolation["direct_clean"][h] for h in range(HORIZONS)},
                    **{f"hard_token_h{h+1}": extrapolation["hard_token"][h] for h in range(HORIZONS)},
                    "lr": lr, "wall_s": wall, "tokens_per_s": processed_tokens / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d} train_ce={float(ce):.4f} val_nll={metrics.nll:.4f} acc={metrics.accuracy:.4f} "
                    f"log_sigma[mean/min/max]={metrics.log_sigma_mean:.3f}/{metrics.log_sigma_min:.3f}/{metrics.log_sigma_max:.3f} "
                    f"ortho_err={metrics.operator_orthogonality_error:.2e} wall={wall:.1f}s",
                    flush=True,
                )
                print("  h1-5 direct_clean: " + " ".join(f"{v:.3f}" for v in extrapolation["direct_clean"]), flush=True)
                print("  h1-5 hard_token:   " + " ".join(f"{v:.3f}" for v in extrapolation["hard_token"]), flush=True)
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
