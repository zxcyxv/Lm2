"""Epsilon-reinsertion tolerance curve for the ha-skew model: how low must
the operator's relative state MSE be before exact-inverse decoding of the
predicted state yields correct tokens?

Method: take GOLD next states h_B (so the answer is architecture-limited,
not operator-limited), inject noise at controlled relative scale epsilon
(relative MSE = epsilon^2 exactly, per row), decode with the trained
exact-inverse decoder, and measure next-token accuracy vs epsilon.

Two noise directions per epsilon:
  - random: isotropic unit direction (standard tolerance curve)
  - k-error: the direction of the trained K's actual residual K(h_A)-h_B,
    to check whether the operator's error direction is more or less
    damaging than random.

Reference rows: epsilon=0 (gold ceiling) and the model's own K(h_A)
prediction (where the trained operator currently sits on the curve).
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from train_k1_ha_skew_learned_noise_mse_ce_13m import make_model
from train_k1_decoder_inverse_ablation_13m import CONTEXT, SEED, memmap, windows

CHECKPOINT = Path("outputs/experiments/EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m/best.pt")
RECORD = Path("experiments/records/EXP-20260723-ha-skew-epsilon-tolerance-13m")
EPSILONS = (0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.085, 0.12, 0.2, 0.3, 0.5, 1.0)
EXAMPLES = 256
MICRO = 16


@torch.inference_mode()
def decode_accuracy(model, window: torch.Tensor, decode_states: torch.Tensor) -> tuple[int, int]:
    positions_full = torch.arange(window.shape[1], device=window.device)
    encoded, _ = model.encode(window, positions_full)
    h_a = encoded[:, :-1]
    decoded = model.exact_inverse_dense_decode_states(h_a, decode_states, positions_full[:-1])
    logits = model.token_logits(decoded)
    targets = window[:, 1:]
    correct = int(logits.float().argmax(dim=-1).eq(targets).sum())
    return correct, targets.numel()


@torch.inference_mode()
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(SEED + 31337)

    model = make_model().cuda().eval()
    state = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"])
    print(f"loaded checkpoint step={state['step']}")

    validation = memmap("validation")
    starts = torch.randint(
        0, len(validation) - CONTEXT - 1, (EXAMPLES,),
        generator=torch.Generator().manual_seed(SEED + 999),
    )

    rows = []
    counters: dict[str, list[int]] = {"gold": [0, 0], "model_k": [0, 0]}
    for eps in EPSILONS:
        counters[f"random_{eps}"] = [0, 0]
        counters[f"kerr_{eps}"] = [0, 0]

    for begin in range(0, len(starts), MICRO):
        window = windows(validation, starts[begin: begin + MICRO])
        positions_full = torch.arange(window.shape[1], device=window.device)
        encoded, _ = model.encode(window, positions_full)
        h_a = encoded[:, :-1]
        h_b = encoded[:, 1:]
        k_pred = model.operator(h_a)

        norm_b = h_b.norm(dim=-1, keepdim=True)
        residual = k_pred - h_b
        residual_hat = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        for name, states in (("gold", h_b), ("model_k", k_pred)):
            correct, total = decode_accuracy(model, window, states)
            counters[name][0] += correct
            counters[name][1] += total

        for eps in EPSILONS:
            u = torch.randn_like(h_b)
            u_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for name, direction in ((f"random_{eps}", u_hat), (f"kerr_{eps}", residual_hat)):
                correct, total = decode_accuracy(model, window, h_b + eps * norm_b * direction)
                counters[name][0] += correct
                counters[name][1] += total

    model_rel_mse = None
    with torch.inference_mode():
        window = windows(validation, starts[:MICRO])
        positions_full = torch.arange(window.shape[1], device=window.device)
        encoded, _ = model.encode(window, positions_full)
        h_a, h_b = encoded[:, :-1], encoded[:, 1:]
        k_pred = model.operator(h_a)
        num = (k_pred.float() - h_b.float()).square().sum(dim=-1)
        den = h_b.float().square().sum(dim=-1).clamp_min(1e-8)
        model_rel_mse = float((num / den).mean())

    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "tolerance.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("condition", "relative_mse", "accuracy"))
        gold_acc = counters["gold"][0] / counters["gold"][1]
        model_acc = counters["model_k"][0] / counters["model_k"][1]
        writer.writerow(("gold", 0.0, gold_acc))
        writer.writerow(("model_k", model_rel_mse, model_acc))
        print(f"gold ceiling accuracy      : {gold_acc:.4f}")
        print(f"model K(h_A) accuracy      : {model_acc:.4f}  (relative MSE {model_rel_mse:.5f})")
        print(f"{'eps':>8} {'rel_mse':>10} {'random':>8} {'k_error':>8}")
        for eps in EPSILONS:
            r = counters[f"random_{eps}"]
            k = counters[f"kerr_{eps}"]
            writer.writerow((f"random_{eps}", eps * eps, r[0] / r[1]))
            writer.writerow((f"kerr_{eps}", eps * eps, k[0] / k[1]))
            print(f"{eps:>8.3f} {eps*eps:>10.6f} {r[0]/r[1]:>8.4f} {k[0]/k[1]:>8.4f}")


if __name__ == "__main__":
    main()
