"""Split the phase gradient into hidden_phase vs memory_phase.

The step-1000 decomposition showed ``csp.phase`` carrying 15.38 of the
121M run's 15.97 total gradient norm from just 2,304 parameters, while the
same group contributes 0.059 of 1.083 in the 13M run.  This isolates which
of the two phase parameters is responsible and how it scales with width.
"""
from __future__ import annotations

import argparse
import math

import torch

from train_time_varying_scan import (
    make_model,
    configure_scan_backend,
    memmap,
    sampled_windows,
    scan_loss,
)

CHECKPOINTS = (
    (
        1344,
        "outputs/experiments/"
        "EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090/"
        "step1000.pt",
    ),
    (4352, "outputs/time_varying_scan/checkpoints/step01000.pt"),
)

WATCH = (
    "complex_self_prediction.hidden_phase",
    "complex_self_prediction.memory_phase",
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

    print(
        f"{'width':>6} {'horizons':>8} | "
        + " | ".join(f"{name.split('.')[-1]:>26}" for name in WATCH)
        + " |     total"
    )
    print(
        f"{'':>6} {'':>8} | "
        + " | ".join(f"{'gnorm':>9} {'n':>5} {'rms':>9}" for _ in WATCH)
        + " |     gnorm"
    )
    print("-" * 104)

    for width, checkpoint_path in CHECKPOINTS:
        for horizons in (1, 4, 16):
            namespace = argparse.Namespace(**vars(args))
            namespace.width = width
            namespace.horizons = horizons

            torch.manual_seed(args.seed)
            model = make_model(namespace).cuda().train()
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
                args.context + horizons,
            )
            model.zero_grad(set_to_none=True)
            scan_loss(model, window, namespace).backward()

            named = dict(model.named_parameters())
            cells = []
            for name in WATCH:
                parameter = named[name]
                grad = parameter.grad
                if grad is None:
                    cells.append(f"{'none':>9} {parameter.numel():5d} {'-':>9}")
                    continue
                gnorm = float(grad.float().norm())
                cells.append(
                    f"{gnorm:9.4f} {grad.numel():5d} "
                    f"{gnorm / math.sqrt(grad.numel()):9.2e}"
                )
            total = math.sqrt(
                sum(
                    float(p.grad.float().square().sum())
                    for p in model.parameters()
                    if p.grad is not None
                )
            )
            print(
                f"{width:>6} {horizons:>8} | " + " | ".join(cells) + f" | {total:9.4f}"
            )

            del model, checkpoint
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
