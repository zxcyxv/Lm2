"""ha-skew variant with FIXED noise sigma=0.01 instead of the learned
SigmaPredictor. Rationale (this session's epsilon-tolerance measurement +
theory): injected relative MSE = sigma^2, the decoder's random-direction
tolerance knee sits at eps ~0.013 (relative MSE ~1.5e-4), and the
conditional-variance floor of the operator MSE is estimated at the same
scale, so sigma=0.01 (relative MSE 1e-4) is simultaneously (a) below the
token-signal amplitude, (b) matched to the target operating point. The
previous INIT_SIGMA=0.05 injected 2.5e-3 relative MSE -- 4x the token
amplitude -- so training-time CE saw mostly token-obliterated states.

Registered predictions (user, 2026-07-23), to be distinguished by the
inline tolerance probe:
  H1 (works): direct_clean_h1 rises above the learned-noise run while the
      gold-state tolerance knee stays put.
  H2 (adaptation): the model adapts to the fixed noise instead -- the
      decoder narrows/shifts the token-signal amplitude, or learns to
      systematically distinguish K(h_A)-path states from gold h_B states;
      visible as the probe knee moving (gold+eps accuracies changing)
      and/or gold ceiling accuracy drifting.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

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
from train_k1_ha_skew_learned_noise_mse_ce_13m import (
    HORIZONS,
    WIDTH,
    Metrics,
    check_extrapolation,
    evaluate,
    loss_terms,
    make_model,
    orthogonality_error,
    save_checkpoint,
)

FIXED_SIGMA = 0.01
STEPS = 6000
PROBE_EPSILONS = (0.005, 0.01, 0.02)
PROBE_EXAMPLES = 64
RECORD = Path("experiments/records/EXP-20260723-k1-ha-skew-fixed-noise-sigma001-13m")
OUTPUT = Path("outputs/experiments/EXP-20260723-k1-ha-skew-fixed-noise-sigma001-13m")


class FixedSigma(nn.Module):
    """Drop-in for SigmaPredictor: constant log-sigma, no parameters."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.log_sigma_value = math.log(sigma)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return torch.full_like(u, self.log_sigma_value)


def make_fixed_model():
    model = make_model()
    model.sigma_predictor = FixedSigma(FIXED_SIGMA)
    return model


@torch.inference_mode()
def probe_tolerance(model, validation, starts: torch.Tensor, micro: int = 16) -> dict[str, float]:
    """Decode gold h_B (clean and with eps-scaled random noise) to track
    whether the decoder's tolerance knee moves during training (H2)."""
    model.eval()
    correct = {"gold": 0, **{f"eps_{eps}": 0 for eps in PROBE_EPSILONS}}
    total = 0
    for begin in range(0, len(starts), micro):
        window = windows(validation, starts[begin: begin + micro])
        positions_full = torch.arange(window.shape[1], device=window.device)
        encoded, _ = model.encode(window, positions_full)
        h_a, h_b = encoded[:, :-1], encoded[:, 1:]
        targets = window[:, 1:]
        norm_b = h_b.norm(dim=-1, keepdim=True)
        variants = {"gold": h_b}
        for eps in PROBE_EPSILONS:
            u = torch.randn_like(h_b)
            u_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            variants[f"eps_{eps}"] = h_b + eps * norm_b * u_hat
        for name, states in variants.items():
            decoded = model.exact_inverse_dense_decode_states(h_a, states, positions_full[:-1])
            logits = model.token_logits(decoded)
            correct[name] += int(logits.float().argmax(dim=-1).eq(targets).sum())
        total += targets.numel()
    model.train()
    return {name: value / total for name, value in correct.items()}


def preflight(model, training, batch: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 4242)
    window = sampled_windows(training, min(batch, 2), generator)
    loss, ce, mse, log_sigma, logits, clean_states, h_b_gold, targets = loss_terms(model, window)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("preflight produced non-finite logits")

    log_sigma_diff = float((log_sigma - math.log(FIXED_SIGMA)).abs().max())
    if log_sigma_diff >= 1e-6:
        raise RuntimeError(f"log_sigma not fixed at log(FIXED_SIGMA): max abs diff={log_sigma_diff:.3e}")

    ortho_err_init = orthogonality_error(model)
    if ortho_err_init >= 1e-4:
        raise RuntimeError(f"K not orthogonal at init: max abs error={ortho_err_init:.3e}")
    weight_init = model.operator.weight.detach()
    identity_diff = float((weight_init - torch.eye(WIDTH, device=weight_init.device)).abs().max())
    if identity_diff >= 1e-4:
        raise RuntimeError(f"K not identity at init: max abs diff={identity_diff:.3e}")

    ce_k_grad = torch.autograd.grad(ce, model.operator.A, retain_graph=True)[0]
    mse_k_grad = torch.autograd.grad(mse, model.operator.A)[0]
    for name, gradient in (("ce->K.A", ce_k_grad), ("mse->K.A", mse_k_grad)):
        if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm() > 0):
            raise RuntimeError(f"preflight produced no finite {name} gradient")

    from train_k1_ha_skew_learned_noise_mse_ce_13m import forward_pass

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
        "log_sigma_fixed_diff": log_sigma_diff,
        "noise_effect_on_ce_logits_max_abs": noise_effect,
    }
    model.zero_grad(set_to_none=True)
    return values


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
        0, len(validation) - CONTEXT - HORIZONS - 1, (256,),
        generator=torch.Generator().manual_seed(SEED + 7777),
    )
    probe_starts = torch.randint(
        0, len(validation) - CONTEXT - 1, (PROBE_EXAMPLES,),
        generator=torch.Generator().manual_seed(SEED + 5555),
    )
    RECORD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    model = make_fixed_model().cuda().train()
    preflight_values = preflight(model, training, args.batch)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"initial_ce={preflight_values['initial_ce']:.4f} initial_mse={preflight_values['initial_mse']:.4f} "
        f"fixed_sigma={FIXED_SIGMA}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    with (RECORD / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in (
            ("seed", SEED), ("steps", args.steps), ("batch", args.batch), ("context", CONTEXT),
            ("parameters", total_parameters), ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"), ("fixed_sigma", FIXED_SIGMA),
            ("operator", "skew_matrix_exp_no_query"), ("k_input", "h_A_ordinary_causal_state"),
            ("noise", "fixed_sigma_replaces_learned_sigma_predictor"),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    report_steps = {1, 50, 100} | set(range(250, args.steps + 1, 250)) | {args.steps}
    fields = (
        "step", "train_combined", "train_ce", "train_mse",
        *tuple(f"val_{key}" for key in asdict(Metrics(*(0.0,) * 12))),
        *tuple(f"direct_clean_h{h}" for h in range(1, HORIZONS + 1)),
        *tuple(f"hard_token_h{h}" for h in range(1, HORIZONS + 1)),
        "probe_gold", *tuple(f"probe_eps_{eps}" for eps in PROBE_EPSILONS),
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
            if step in report_steps:
                metrics = evaluate(model, validation, validation_starts, args.eval_micro, operator_gradient_norm)
                extrapolation = check_extrapolation(model, validation, extrapolation_starts)
                probe = probe_tolerance(model, validation, probe_starts)
                wall = time.time() - started
                row = {
                    "step": step, "train_combined": float(loss), "train_ce": float(ce), "train_mse": float(mse),
                    **{f"val_{key}": value for key, value in asdict(metrics).items()},
                    **{f"direct_clean_h{h+1}": extrapolation["direct_clean"][h] for h in range(HORIZONS)},
                    **{f"hard_token_h{h+1}": extrapolation["hard_token"][h] for h in range(HORIZONS)},
                    "probe_gold": probe["gold"],
                    **{f"probe_eps_{eps}": probe[f"eps_{eps}"] for eps in PROBE_EPSILONS},
                    "lr": lr, "wall_s": wall, "tokens_per_s": processed_tokens / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d} train_ce={float(ce):.4f} val_nll={metrics.nll:.4f} acc={metrics.accuracy:.4f} "
                    f"state_mse={metrics.gold_state_relative_mse:.5f} wall={wall:.1f}s",
                    flush=True,
                )
                print("  h1-5 direct_clean: " + " ".join(f"{v:.3f}" for v in extrapolation["direct_clean"]), flush=True)
                print("  h1-5 hard_token:   " + " ".join(f"{v:.3f}" for v in extrapolation["hard_token"]), flush=True)
                print(
                    "  probe gold=" + f"{probe['gold']:.4f} "
                    + " ".join(f"eps{eps}={probe[f'eps_{eps}']:.4f}" for eps in PROBE_EPSILONS),
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
