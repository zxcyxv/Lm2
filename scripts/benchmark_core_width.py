"""What does removing the 496-wide bottleneck actually cost?

``read_width = 2 * heads * value_dim`` is 496 regardless of ``--width``, so at
width 4352 the central recurrence writes into a rank-496 subspace of a
4352-dimensional latent.  Standard multi-head attention instead ties
``n_heads * head_dim = d_model``, which here means ``read_width ~ width``.

This times one forward+backward of the real training objective at several
head counts so the compute and memory price of widening the core is measured
rather than estimated.
"""
from __future__ import annotations

import argparse
import time

import torch

from train_time_varying_scan import (
    configure_scan_backend,
    make_model,
    memmap,
    parameter_count,
    sampled_windows,
    scan_loss,
)


def benchmark(width: int, heads: int, args, key_dim=None, value_dim=None) -> dict:
    namespace = argparse.Namespace(**vars(args))
    namespace.width = width
    namespace.heads = heads
    if key_dim is not None:
        namespace.key_dim = key_dim
    if value_dim is not None:
        namespace.value_dim = value_dim

    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = make_model(namespace).cuda().train()
    configure_scan_backend(model, "eager")

    training = memmap("train")
    window = sampled_windows(
        training,
        args.microbatch,
        torch.Generator().manual_seed(args.seed),
        args.context + args.horizons,
    )

    for index in range(args.warmup + args.iters):
        if index == args.warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        scan_loss(model, window, namespace).backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / args.iters

    result = {
        "heads": heads,
        "read_width": 2 * heads * namespace.value_dim,
        "memory_state": heads * namespace.key_dim * namespace.value_dim,
        "key_dim": namespace.key_dim,
        "value_dim": namespace.value_dim,
        "params": parameter_count(model),
        "seconds": elapsed,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=4352)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=31)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)
    parser.add_argument("--shared-kv-heads", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--heads-list", type=int, nargs="+", default=[8, 16, 26, 44, 70]
    )
    args = parser.parse_args()

    print(
        f"width={args.width} microbatch={args.microbatch} H={args.horizons} "
        f"(one fwd+bwd of the training objective)\n"
    )
    print(
        f"{'heads':>6} {'read_w':>7} {'rw/width':>9} {'memory':>8} "
        f"{'params':>13} {'s/step':>8} {'vs h=8':>7} {'peak GiB':>9}"
    )
    print("-" * 76)

    reference = None
    for heads in args.heads_list:
        try:
            row = benchmark(args.width, heads, args)
        except torch.OutOfMemoryError:
            print(f"{heads:>6} {'':>7} {'':>9} {'':>8} {'':>13} {'OOM':>8}")
            torch.cuda.empty_cache()
            continue
        if reference is None:
            reference = row["seconds"]
        print(
            f"{row['heads']:>6} {row['read_width']:>7} "
            f"{row['read_width'] / args.width:9.3f} {row['memory_state']:>8,} "
            f"{row['params']:>13,} {row['seconds']:8.4f} "
            f"{row['seconds'] / reference:6.2f}x {row['peak_gib']:9.2f}"
        )


if __name__ == "__main__":
    main()
