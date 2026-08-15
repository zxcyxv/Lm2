"""Print the largest microbatch that fits for a given central-operator config.

Removing the write-back bottleneck grows the memory tape by
``batch * anchors * horizons * heads * key_dim * value_dim`` complex entries,
which is what actually decides the microbatch.  This tries each divisor of
the effective batch from large to small and prints the first that survives a
real forward+backward, so the training launch does not have to guess.
"""
from __future__ import annotations

import argparse

import torch

from train_time_varying_scan import (
    configure_scan_backend,
    make_model,
    memmap,
    sampled_windows,
    scan_loss,
)


def fits(microbatch: int, args) -> bool:
    namespace = argparse.Namespace(**vars(args))
    torch.cuda.empty_cache()
    try:
        model = make_model(namespace).cuda().train()
        configure_scan_backend(model, "eager")
        window = sampled_windows(
            memmap("train"),
            microbatch,
            torch.Generator().manual_seed(args.seed),
            args.context + args.horizons,
        )
        model.zero_grad(set_to_none=True)
        scan_loss(model, window, namespace).backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        del model, window
        torch.cuda.empty_cache()
        print(f"# microbatch {microbatch}: ok, peak {peak:.2f} GiB", flush=True)
        return True
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"# microbatch {microbatch}: OOM", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--key-dim", type=int, required=True)
    parser.add_argument("--value-dim", type=int, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)
    parser.add_argument("--shared-kv-heads", action="store_true")
    args = parser.parse_args()

    for microbatch in (64, 32, 16, 8, 4, 2, 1):
        if args.batch % microbatch:
            continue
        torch.cuda.reset_peak_memory_stats()
        if fits(microbatch, args):
            print(microbatch)
            return
    raise SystemExit("no microbatch fits")


if __name__ == "__main__":
    main()
