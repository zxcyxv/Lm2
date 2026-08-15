"""Per-submodule gradient decomposition at the two step-1000 checkpoints.

Comparing at initialization is confounded: the two widths start at very
different losses (45.0 vs 15.3), so the 13M simply has a bigger error to
backpropagate.  At step 1000 both runs sit at a comparable loss (13M block
NLL 3.054, 121M 3.159), so a gradient decomposition there isolates *where*
the 121M's residual gnorm ~10 lives while the 13M is already at ~0.99.

``rms`` is ``gnorm / sqrt(numel)``: the per-coordinate gradient magnitude.
If the whole gap is the flat-L2-over-more-parameters artifact, rms matches
across widths and only gnorm differs, by sqrt(param ratio) = 3.06x.
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

SMALL_CHECKPOINT = (
    "outputs/experiments/"
    "EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090/"
    "step1000.pt"
)
LARGE_CHECKPOINT = "outputs/time_varying_scan/checkpoints/step01000.pt"

GROUPS = (
    ("enc.embed", lambda n: n.startswith("encoder.embed")),
    ("enc.attn", lambda n: ".attn." in n),
    ("enc.ffn", lambda n: ".ffn." in n),
    ("enc.norm", lambda n: ".norm" in n and n.startswith("encoder")),
    ("csp.query", lambda n: n.startswith("complex_self_prediction.query")),
    ("csp.key", lambda n: n.startswith("complex_self_prediction.key")),
    ("csp.value", lambda n: n.startswith("complex_self_prediction.value")),
    ("csp.output", lambda n: n.startswith("complex_self_prediction.output")),
    ("csp.phase", lambda n: "phase" in n),
)


def group_of(name: str) -> str:
    for label, predicate in GROUPS:
        if predicate(name):
            return label
    return "other"


def measure(width: int, checkpoint_path: str, args) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    namespace = argparse.Namespace(**vars(args))
    namespace.width = width

    model = make_model(namespace).cuda().train()
    configure_scan_backend(model, "eager")

    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [w={width}] missing={list(missing)} unexpected={list(unexpected)}")

    training = memmap("train")
    window = sampled_windows(
        training,
        args.microbatch,
        torch.Generator().manual_seed(args.seed),
        args.context + args.horizons,
    )

    model.zero_grad(set_to_none=True)
    loss = scan_loss(model, window, namespace)
    loss_value = float(loss.detach())
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
        "loss": loss_value,
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
    del model, checkpoint, state
    torch.cuda.empty_cache()
    return result


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

    small = measure(1344, SMALL_CHECKPOINT, args)
    large = measure(4352, LARGE_CHECKPOINT, args)

    print(f"\n{'':>11} | {'13M (w=1344) step1000':>29} | {'121M (w=4352) step1000':>29} | ratio")
    print(
        f"{'group':>11} | {'gnorm':>9} {'params':>11} {'rms':>7} "
        f"| {'gnorm':>9} {'params':>11} {'rms':>7} | gnorm   rms"
    )
    print("-" * 106)

    labels = sorted(
        set(small["groups"]) | set(large["groups"]),
        key=lambda label: -large["groups"].get(label, {"gnorm": 0})["gnorm"],
    )
    for label in labels:
        s = small["groups"].get(label)
        l = large["groups"].get(label)
        if s is None or l is None:
            continue
        gnorm_ratio = l["gnorm"] / s["gnorm"] if s["gnorm"] else float("inf")
        rms_ratio = l["rms"] / s["rms"] if s["rms"] else float("inf")
        print(
            f"{label:>11} | {s['gnorm']:9.4f} {s['params']:11,} {s['rms']:7.2e} "
            f"| {l['gnorm']:9.4f} {l['params']:11,} {l['rms']:7.2e} "
            f"| {gnorm_ratio:5.2f}x {rms_ratio:5.2f}x"
        )

    print("-" * 106)
    s_rms = small["total_gnorm"] / math.sqrt(small["total_params"])
    l_rms = large["total_gnorm"] / math.sqrt(large["total_params"])
    print(
        f"{'TOTAL':>11} | {small['total_gnorm']:9.4f} {small['total_params']:11,} "
        f"{s_rms:7.2e} | {large['total_gnorm']:9.4f} {large['total_params']:11,} "
        f"{l_rms:7.2e} | {large['total_gnorm']/small['total_gnorm']:5.2f}x "
        f"{l_rms/s_rms:5.2f}x"
    )
    print(
        f"\nsqrt(param ratio) = "
        f"{math.sqrt(large['total_params']/small['total_params']):.2f}x"
        f"  <- gnorm ratio predicted by pure dimension alone"
    )
    print(f"loss on this batch: 13M={small['loss']:.4f}  121M={large['loss']:.4f}")


if __name__ == "__main__":
    main()
