"""Does spreading hidden_phase break the horizon coherence?

``memory_phase`` is initialized as ``linspace(-r, r)`` over distinct
frequencies and its gradient is negligible (0.0055).  ``hidden_phase`` is
initialized to ``torch.zeros`` -- every pair shares the same zero angle --
and it carries 93% of the 121M run's squared gradient norm (15.38).

If the zero initialization is what keeps every horizon looking at the same
state, then replacing the trained hidden_phase with a spread of frequencies
should decorrelate the horizons: ``||delta_h||`` should start decaying along
h instead of staying flat, and the theta gradient should collapse.  Nothing
else about the architecture changes -- the transport stays exactly unitary.
"""
from __future__ import annotations

import argparse
import math

import torch

from train_time_varying_scan import (
    configure_scan_backend,
    make_model,
    memmap,
    sampled_windows,
    scan_loss,
)
from rotlm.training.ha_skew_window import sparse_anchor_indices

CHECKPOINT = "outputs/time_varying_scan/checkpoints/step01000.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--width", type=int, default=4352)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=31)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)
    parser.add_argument(
        "--init",
        action="store_true",
        help="probe a freshly initialized model instead of the trained "
        "checkpoint. Perturbing a trained checkpoint is confounded: the rest "
        "of the model was fitted around the near-zero phase, so replacing it "
        "wrecks the loss and that dominates the gradient.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = make_model(args).cuda().train()
    configure_scan_backend(model, "eager")
    if not args.init:
        checkpoint = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)

    transition = model.complex_self_prediction
    trained_phase = transition.hidden_phase.detach().clone()
    pairs = trained_phase.numel()

    training = memmap("train")
    window = sampled_windows(
        training,
        args.microbatch,
        torch.Generator().manual_seed(args.seed),
        args.context + args.horizons,
    )
    positions = torch.arange(args.context, device=window.device)
    anchors = sparse_anchor_indices(
        args.context, args.anchor_stride, device=window.device
    )

    source = "fresh init" if args.init else "121M step-1000 checkpoint"
    print(
        f"width={args.width} H={args.horizons} pairs={pairs} "
        f"({source}, only hidden_phase replaced)\n"
    )
    print(
        f"{'hidden_phase':>22} | {'theta gnorm':>11} {'vs trained':>10} | "
        f"{'total gnorm':>11} | {'|d_16|/|d_1|':>12} | {'loss':>7}"
    )
    print("-" * 88)

    baseline_label = "zeros (as shipped)" if args.init else "trained (zero-ish)"
    variants = [(baseline_label, trained_phase)]
    for spread in (0.01, 0.05, 0.1, 0.5, 1.0):
        variants.append(
            (
                f"linspace(-{spread}, {spread})",
                torch.linspace(-spread, spread, pairs, device=trained_phase.device),
            )
        )

    for label, values in variants:
        with torch.no_grad():
            transition.hidden_phase.copy_(values)

        # coherence: how much does the innovation decay along the horizon axis?
        with torch.inference_mode():
            encoded, _ = model.encode(window[:, : args.context], positions)
            roots = encoded.index_select(1, anchors)
            rollout = transition.rollout_time_varying_scan(
                roots, args.horizons, collect_scan_diagnostics=True
            )
            deltas = rollout.innovation_deltas
            first = float(deltas[..., 0, :].float().norm(dim=-1).mean())
            last = float(deltas[..., -1, :].float().norm(dim=-1).mean())
        decay_ratio = last / first if first else float("nan")

        model.zero_grad(set_to_none=True)
        loss = scan_loss(model, window, args)
        loss_value = float(loss.detach())
        loss.backward()

        theta_norm = float(transition.hidden_phase.grad.float().norm())
        total = math.sqrt(
            sum(
                float(p.grad.float().square().sum())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        if label == baseline_label:
            reference = theta_norm
        print(
            f"{label:>22} | {theta_norm:11.4f} {theta_norm / reference:9.3f}x | "
            f"{total:11.4f} | {decay_ratio:12.3f} | {loss_value:7.4f}"
        )


if __name__ == "__main__":
    main()
