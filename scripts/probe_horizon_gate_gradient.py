"""Is there a gradient mechanism driving innovations to decay along h?

The 13M run learned ``|delta_16|/|delta_1| = 0.081`` while the 121M stayed at
0.967.  That decay is what decorrelates the horizons and collapses the
hidden_phase gradient, so the question is whether the objective itself
supplies a force toward it -- and whether that force weakens with width.

Probe: insert a per-horizon gate ``g_h`` (all ones, so the forward pass is
bit-identical) multiplying ``delta_h`` before the latent scan.  Then

    dL/dg_h > 0  ->  the loss wants horizon h's innovation smaller
    dL/dg_h < 0  ->  it wants it larger

so the sign pattern along h *is* the driving force, and its magnitude
relative to horizon 1 says how strong the pressure to decay is.  Comparing
the profile at both widths, at init and at step 1000, shows whether the
mechanism exists and whether width attenuates it.
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

CHECKPOINTS = {
    1344: (
        "outputs/experiments/"
        "EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090/"
        "step1000.pt"
    ),
    4352: "outputs/time_varying_scan/checkpoints/step01000.pt",
}


def gated_rollout(transition, gate):
    """Patch the scan so ``delta_h`` is scaled by ``gate[h]`` (all ones)."""
    original = transition.rollout_time_varying_scan

    def replacement(root, horizons, *, collect_scan_diagnostics=True, **kwargs):
        result = original(
            root,
            horizons,
            collect_scan_diagnostics=True,
            **kwargs,
        )
        gated = result.innovation_deltas * gate[:horizons, None]
        full_states, _, _ = transition._scan_hidden_innovation_tape(
            root, gated, collect_scan_diagnostics=False
        )
        return result.__class__(
            **{
                **{
                    field: getattr(result, field)
                    for field in result.__dataclass_fields__
                },
                "full_states": full_states,
                "read_states": transition._fixed_rms_normalize_hidden(full_states),
            }
        )

    transition.rollout_time_varying_scan = replacement
    return original


def profile(width: int, trained: bool, args) -> list[float]:
    namespace = argparse.Namespace(**vars(args))
    namespace.width = width

    torch.manual_seed(args.seed)
    model = make_model(namespace).cuda().train()
    configure_scan_backend(model, "eager")
    if trained:
        checkpoint = torch.load(
            CHECKPOINTS[width], map_location="cuda", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    training = memmap("train")
    window = sampled_windows(
        training,
        args.microbatch,
        torch.Generator().manual_seed(args.seed),
        args.context + args.horizons,
    )

    transition = model.complex_self_prediction
    gate = torch.ones(args.horizons, device="cuda", requires_grad=True)
    original = gated_rollout(transition, gate)
    try:
        model.zero_grad(set_to_none=True)
        loss = scan_loss(model, window, namespace)
        loss.backward()
    finally:
        transition.rollout_time_varying_scan = original

    values = [float(v) for v in gate.grad]
    loss_value = float(loss.detach())
    del model
    torch.cuda.empty_cache()
    return values, loss_value


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

    columns = [
        ("13M init", profile(1344, False, args)),
        ("121M init", profile(4352, False, args)),
        ("13M step1000", profile(1344, True, args)),
        ("121M step1000", profile(4352, True, args)),
    ]

    print("\ndL/dg_h  (>0 = loss wants horizon h's innovation SMALLER)\n")
    header = f"{'h':>3} | " + " | ".join(f"{label:>14}" for label, _ in columns)
    print(header)
    print("-" * len(header))
    for index in range(args.horizons):
        cells = " | ".join(f"{values[index]:14.6f}" for _, (values, _) in columns)
        print(f"{index + 1:>3} | {cells}")

    print("-" * len(header))
    ratios = " | ".join(
        f"{values[-1] / values[0]:14.3f}" if values[0] else f"{'n/a':>14}"
        for _, (values, _) in columns
    )
    print(f"{'16/1':>3} | {ratios}")
    signs = " | ".join(
        f"{sum(1 for v in values if v > 0):>7}/{len(values):<6}"
        for _, (values, _) in columns
    )
    print(f"{'>0':>3} | {signs}")
    print(f"{'loss':>3} | " + " | ".join(f"{loss:14.4f}" for _, (_, loss) in columns))
    # dL/dg is a directional derivative along delta itself, so its raw size
    # inherits delta's scale.  Dividing by the loss makes the per-horizon
    # differentiation force comparable across widths.
    print(
        f"\n{'|dg1|/L':>7} | "
        + " | ".join(f"{abs(values[0]) / loss:14.3e}" for _, (values, loss) in columns)
    )
    print(
        f"{'spread':>7} | "
        + " | ".join(
            f"{abs(values[0] - values[-1]) / loss:14.3e}"
            for _, (values, loss) in columns
        )
        + "   <- (dg_1 - dg_16)/L: force differentiating early vs late horizons"
    )


if __name__ == "__main__":
    main()
