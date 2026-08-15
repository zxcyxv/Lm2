"""Do ||S_h|| and ||z_h|| actually grow along the horizon axis?

The hypothesis under test: the memory ``S`` is a unitary-transported lossless
accumulator, so its norm grows with h, and that growth propagates into the
latent state ``z``, inflating gradients.  The counter-claim from the code is
that the read is divided by ``||S||_F`` (``forward``, normalize_read_by_memory
_norm), so the delta entering ``z`` is scale invariant no matter how large the
memory gets.  This measures all three quantities per horizon on both trained
checkpoints.
"""
from __future__ import annotations

import argparse

import torch

from train_time_varying_scan import (
    configure_scan_backend,
    make_model,
    memmap,
    sampled_windows,
)
from rotlm.training.ha_skew_window import sparse_anchor_indices

CHECKPOINTS = (
    (
        1344,
        "13M",
        "outputs/experiments/"
        "EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090/"
        "step1000.pt",
    ),
    (4352, "121M", "outputs/time_varying_scan/checkpoints/step01000.pt"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=31)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)
    args = parser.parse_args()

    for width, label, checkpoint_path in CHECKPOINTS:
        namespace = argparse.Namespace(**vars(args))
        namespace.width = width
        torch.manual_seed(args.seed)
        model = make_model(namespace).cuda().eval()
        configure_scan_backend(model, "eager")
        checkpoint = torch.load(
            checkpoint_path, map_location="cuda", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=False)

        training = memmap("train")
        window = sampled_windows(
            training,
            args.microbatch,
            torch.Generator().manual_seed(args.seed),
            args.context + args.horizons,
        )

        with torch.inference_mode():
            positions = torch.arange(args.context, device=window.device)
            encoded, _ = model.encode(window[:, : args.context], positions)
            anchors = sparse_anchor_indices(
                args.context, args.anchor_stride, device=window.device
            )
            roots = encoded.index_select(1, anchors)
            rollout = model.complex_self_prediction.rollout_time_varying_scan(
                roots, args.horizons, collect_scan_diagnostics=True
            )

            memory = rollout.memory_states
            states = rollout.full_states
            deltas = rollout.innovation_deltas
            root_norm = float(roots.float().norm(dim=-1).mean())

            print(f"\n=== {label} (width={width}) step-1000 checkpoint ===")
            print(f"||z_0|| (root) = {root_norm:.4f}")
            print(
                f"{'h':>3} | {'||S_h||_F':>11} {'vs h=1':>7} | "
                f"{'||z_h||':>10} {'vs z_0':>7} | {'||delta_h||':>11}"
            )
            print("-" * 62)
            first_memory = None
            for index in range(args.horizons):
                memory_norm = float(
                    memory[..., index, :, :, :]
                    .abs()
                    .square()
                    .sum(dim=(-3, -2, -1))
                    .sqrt()
                    .float()
                    .mean()
                )
                state_norm = float(states[..., index, :].float().norm(dim=-1).mean())
                delta_norm = float(deltas[..., index, :].float().norm(dim=-1).mean())
                if first_memory is None:
                    first_memory = memory_norm
                print(
                    f"{index + 1:>3} | {memory_norm:11.4f} "
                    f"{memory_norm / first_memory:6.2f}x | "
                    f"{state_norm:10.4f} {state_norm / root_norm:6.2f}x | "
                    f"{delta_norm:11.4f}"
                )

        del model, checkpoint
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
