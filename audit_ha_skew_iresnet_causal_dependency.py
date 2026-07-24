"""Intervention and gradient audit for the K/K^2 causal decoder tape."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.nn.utils import parametrize

from train_k1_decoder_inverse_ablation_13m import CONTEXT, SEED, memmap
from train_k1_ha_skew_iresnet_spectral_learned_noise_13m import (
    EXPERIMENT_ID,
    make_model,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import (
    HORIZONS,
    windows_of_length,
)


DEFAULT_CHECKPOINT = (
    Path("outputs/experiments") / EXPERIMENT_ID / "step1000.pt"
)
DEFAULT_OUTPUT = (
    Path("experiments/records")
    / EXPERIMENT_ID
    / "step1000_causal_dependency_audit"
)


def load_model(path: Path) -> tuple[torch.nn.Module, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model()
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    model.requires_grad_(False)
    return model, int(payload["step"])


def max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max())


def mean_relative_mse(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = (first.float() - second.float()).square().flatten(1).sum(-1)
    denominator = second.float().square().flatten(1).sum(-1).clamp_min(1e-12)
    return float((numerator / denominator).mean())


def decode(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    state1: torch.Tensor,
    state2: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = [encoded, state1[:, None]]
    if state2 is not None:
        states.append(state2[:, None])
    tape = torch.cat(states, dim=1)
    positions = torch.arange(
        tape.shape[1], device=tape.device
    ).unsqueeze(0).expand(tape.shape[0], -1)
    hidden = model.encoder.decode_hidden(tape, positions)
    logits = model.token_logits(hidden)
    return hidden, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--expected-step", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this audit requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model, step = load_model(args.checkpoint)
    if step != args.expected_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_step}, found {step}"
        )
    validation = memmap("validation")
    starts = torch.randint(
        0,
        len(validation) - CONTEXT - HORIZONS - 1,
        (args.examples,),
        generator=torch.Generator().manual_seed(SEED + 7777),
    )
    window = windows_of_length(validation, starts, CONTEXT + 2)
    prefix = window[:, :CONTEXT]
    positions = torch.arange(CONTEXT, device=prefix.device)
    with torch.inference_mode(), parametrize.cached():
        encoded, _ = model.encode(prefix, positions)
        state1 = model.operator(encoded[:, -1])
        state2 = model.operator(state1)
        hidden_short, logits_short = decode(model, encoded, state1, None)
        hidden_joint, logits_joint = decode(
            model, encoded, state1, state2
        )

        generator = torch.Generator(device=state1.device).manual_seed(
            SEED + 9090
        )
        noise1 = torch.randn(
            state1.shape,
            device=state1.device,
            dtype=state1.dtype,
            generator=generator,
        )
        noise2 = torch.randn(
            state2.shape,
            device=state2.device,
            dtype=state2.dtype,
            generator=generator,
        )
        rms1 = state1.norm(dim=-1, keepdim=True) / state1.shape[-1] ** 0.5
        rms2 = state2.norm(dim=-1, keepdim=True) / state2.shape[-1] ** 0.5
        alternate_state1 = state1 + 10.0 * rms1 * noise1
        alternate_state2 = state2 + 10.0 * rms2 * noise2
        hidden_future_changed, logits_future_changed = decode(
            model, encoded, state1, alternate_state2
        )
        hidden_first_changed, logits_first_changed = decode(
            model, encoded, alternate_state1, state2
        )

    rows = [
        (
            "short_vs_joint_h1_hidden",
            max_abs(hidden_short[:, -1], hidden_joint[:, -2]),
            mean_relative_mse(hidden_short[:, -1], hidden_joint[:, -2]),
        ),
        (
            "short_vs_joint_h1_logits",
            max_abs(logits_short[:, -1], logits_joint[:, -2]),
            mean_relative_mse(logits_short[:, -1], logits_joint[:, -2]),
        ),
        (
            "change_s2_prefix_plus_h1_hidden",
            max_abs(
                hidden_joint[:, :-1], hidden_future_changed[:, :-1]
            ),
            mean_relative_mse(
                hidden_joint[:, :-1], hidden_future_changed[:, :-1]
            ),
        ),
        (
            "change_s2_h1_logits",
            max_abs(
                logits_joint[:, -2], logits_future_changed[:, -2]
            ),
            mean_relative_mse(
                logits_joint[:, -2], logits_future_changed[:, -2]
            ),
        ),
        (
            "change_s2_h2_hidden",
            max_abs(
                hidden_joint[:, -1], hidden_future_changed[:, -1]
            ),
            mean_relative_mse(
                hidden_joint[:, -1], hidden_future_changed[:, -1]
            ),
        ),
        (
            "change_s2_h2_logits",
            max_abs(
                logits_joint[:, -1], logits_future_changed[:, -1]
            ),
            mean_relative_mse(
                logits_joint[:, -1], logits_future_changed[:, -1]
            ),
        ),
        (
            "change_s1_prefix_hidden",
            max_abs(
                hidden_joint[:, :-2], hidden_first_changed[:, :-2]
            ),
            mean_relative_mse(
                hidden_joint[:, :-2], hidden_first_changed[:, :-2]
            ),
        ),
        (
            "change_s1_h1_hidden",
            max_abs(
                hidden_joint[:, -2], hidden_first_changed[:, -2]
            ),
            mean_relative_mse(
                hidden_joint[:, -2], hidden_first_changed[:, -2]
            ),
        ),
        (
            "change_s1_h2_hidden",
            max_abs(
                hidden_joint[:, -1], hidden_first_changed[:, -1]
            ),
            mean_relative_mse(
                hidden_joint[:, -1], hidden_first_changed[:, -1]
            ),
        ),
        (
            "change_s1_h2_logits",
            max_abs(
                logits_joint[:, -1], logits_first_changed[:, -1]
            ),
            mean_relative_mse(
                logits_joint[:, -1], logits_first_changed[:, -1]
            ),
        ),
    ]

    # Independent autograd check on one prefix. A random scalar projection
    # avoids a special all-ones direction while testing the actual dependency.
    # inference_mode tensors cannot be marked requires_grad directly. Cloning
    # outside that context produces ordinary tensors for the Jacobian audit.
    encoded_grad = encoded[:1].detach().clone()
    state1_grad = state1[:1].detach().clone().requires_grad_(True)
    state2_grad = state2[:1].detach().clone().requires_grad_(True)
    probe_generator = torch.Generator(device=encoded.device).manual_seed(
        SEED + 9191
    )
    with torch.enable_grad(), parametrize.cached():
        hidden_grad, _ = decode(
            model, encoded_grad, state1_grad, state2_grad
        )
        probe1 = torch.randn(
            hidden_grad[:, -2].shape,
            device=encoded.device,
            generator=probe_generator,
        )
        probe2 = torch.randn(
            hidden_grad[:, -1].shape,
            device=encoded.device,
            generator=probe_generator,
        )
        scalar1 = (hidden_grad[:, -2] * probe1).sum()
        scalar2 = (hidden_grad[:, -1] * probe2).sum()
        grad_h1_s1 = torch.autograd.grad(
            scalar1, state1_grad, retain_graph=True
        )[0]
        grad_h1_s2 = torch.autograd.grad(
            scalar1, state2_grad, retain_graph=True
        )[0]
        grad_h2_s1 = torch.autograd.grad(
            scalar2, state1_grad, retain_graph=True
        )[0]
        grad_h2_s2 = torch.autograd.grad(
            scalar2, state2_grad
        )[0]

    gradient_rows = [
        ("d_h1_probe_d_s1_norm", float(grad_h1_s1.norm())),
        ("d_h1_probe_d_s2_norm", float(grad_h1_s2.norm())),
        ("d_h2_probe_d_s1_norm", float(grad_h2_s1.norm())),
        ("d_h2_probe_d_s2_norm", float(grad_h2_s2.norm())),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "interventions.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("intervention", "max_abs", "mean_relative_mse"))
        writer.writerows(rows)
    with (args.output_dir / "gradients.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("gradient", "l2_norm"))
        writer.writerows(gradient_rows)

    report = [
        "# Step-1000 causal dependency audit",
        "",
        f"- Checkpoint step: {step}",
        f"- Examples: {args.examples}",
        "",
        "## Interventions",
        "",
        "| intervention | max abs | mean relative MSE |",
        "|---|---:|---:|",
        *[
            f"| {name} | {maximum:.9g} | {relative:.9g} |"
            for name, maximum, relative in rows
        ],
        "",
        "## Autograd dependency probes",
        "",
        "| gradient | L2 norm |",
        "|---|---:|",
        *[
            f"| {name} | {value:.9g} |"
            for name, value in gradient_rows
        ],
    ]
    (args.output_dir / "analysis.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
