"""Decompose the global gradient norm by submodule at two widths.

The 121M run clips at gnorm ~12-20 for hundreds of steps while the 13M
baseline is at ~3.6 by step 300 and ~0.99 by step 1000.  Two candidate
explanations:

  (a) pure dimension -- ``clip_grad_norm_`` measures one flat L2 over every
      parameter, so it grows like ``sqrt(parameter count)`` even when the
      per-coordinate gradient is identical.  9.3x more parameters = 3.05x
      more norm, for free.
  (b) structural imbalance -- Q/K/V (fan_in=width, fan_out fixed) and output
      (fan_in fixed, fan_out=width) scale differently from the encoder's
      width x width matrices, so one global LR cannot be right for all of
      them and some submodule is left with an outsized gradient.

(a) predicts the *per-coordinate RMS* gradient ``gnorm / sqrt(numel)`` is
flat across widths; (b) predicts specific submodules blow up in RMS terms.
This script measures both on the same batch and the same seed.
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


GROUPS = (
    ("embedding", lambda n: "embedding" in n or "token" in n),
    ("encoder", lambda n: n.startswith("encoder")),
    ("csp.query", lambda n: n.startswith("complex_self_prediction.query")),
    ("csp.key", lambda n: n.startswith("complex_self_prediction.key")),
    ("csp.value", lambda n: n.startswith("complex_self_prediction.value")),
    ("csp.output", lambda n: n.startswith("complex_self_prediction.output")),
    ("csp.phase", lambda n: "phase" in n),
    ("head", lambda n: n.startswith("head") or "logit" in n),
)


def group_of(name: str) -> str:
    for label, predicate in GROUPS:
        if predicate(name):
            return label
    return "other"


def measure(width: int, peak_lr: float, args) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    namespace = argparse.Namespace(**vars(args))
    namespace.width = width

    model = make_model(namespace).cuda().train()
    configure_scan_backend(model, "eager")

    training = memmap("train")
    window = sampled_windows(
        training,
        args.microbatch,
        torch.Generator().manual_seed(args.seed),
        args.context + args.horizons,
    )

    model.zero_grad(set_to_none=True)
    loss = scan_loss(model, window, namespace)
    loss.backward()

    per_group: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        label = group_of(name)
        squared, count = per_group.setdefault(label, [0.0, 0])
        per_group[label] = [
            squared + float(parameter.grad.float().square().sum()),
            count + parameter.numel(),
        ]

    total_squared = sum(value[0] for value in per_group.values())
    total_count = sum(value[1] for value in per_group.values())
    result = {
        "width": width,
        "loss": float(loss),
        "total_gnorm": math.sqrt(total_squared),
        "total_params": total_count,
        "groups": {
            label: {
                "gnorm": math.sqrt(squared),
                "params": count,
                "rms": math.sqrt(squared / count) if count else 0.0,
            }
            for label, (squared, count) in per_group.items()
        },
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=31)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)
    args = parser.parse_args()

    small = measure(1344, 3e-4, args)
    large = measure(4352, 1e-4, args)

    print(f"{'':>12} | {'13M (w=1344)':>28} | {'121M (w=4352)':>28} | ratio")
    print(
        f"{'group':>12} | {'gnorm':>9} {'params':>10} {'rms':>7} "
        f"| {'gnorm':>9} {'params':>10} {'rms':>7} | gnorm  rms"
    )
    print("-" * 104)

    labels = sorted(set(small["groups"]) | set(large["groups"]))
    for label in labels:
        s = small["groups"].get(label)
        l = large["groups"].get(label)
        if s is None or l is None:
            continue
        gnorm_ratio = l["gnorm"] / s["gnorm"] if s["gnorm"] else float("inf")
        rms_ratio = l["rms"] / s["rms"] if s["rms"] else float("inf")
        print(
            f"{label:>12} | {s['gnorm']:9.4f} {s['params']:10,} {s['rms']:7.2e} "
            f"| {l['gnorm']:9.4f} {l['params']:10,} {l['rms']:7.2e} "
            f"| {gnorm_ratio:5.2f}x {rms_ratio:5.2f}x"
        )

    print("-" * 104)
    s_rms = small["total_gnorm"] / math.sqrt(small["total_params"])
    l_rms = large["total_gnorm"] / math.sqrt(large["total_params"])
    print(
        f"{'TOTAL':>12} | {small['total_gnorm']:9.4f} {small['total_params']:10,} "
        f"{s_rms:7.2e} | {large['total_gnorm']:9.4f} {large['total_params']:10,} "
        f"{l_rms:7.2e} | {large['total_gnorm']/small['total_gnorm']:5.2f}x "
        f"{l_rms/s_rms:5.2f}x"
    )
    print(
        f"\nsqrt(param ratio) = "
        f"{math.sqrt(large['total_params']/small['total_params']):.2f}x "
        f"(what pure dimension alone predicts for the gnorm ratio)"
    )
    print(f"loss: 13M={small['loss']:.4f}  121M={large['loss']:.4f}")


if __name__ == "__main__":
    main()
