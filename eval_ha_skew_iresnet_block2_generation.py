"""Matched block-K^2 generation audit for i-ResNet spectral checkpoints."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.nn.utils import parametrize
from tokenizers import Tokenizer

from eval_ha_skew_block2_generation import generate_block2
from eval_ha_skew_generation_quality import generate_sequential
from rotlm.evaluation import SEED, memmap, token_stats as stats, windows
from train_k1_ha_skew_iresnet_spectral_learned_noise_13m import (
    EXPERIMENT_ID,
    make_model,
)


GENERATION_PROMPT = 64
GENERATION_NEW = 64
GENERATION_SAMPLES = 5
DEFAULT_CHECKPOINT = (
    Path("outputs/experiments") / EXPERIMENT_ID / "step1000.pt"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID


def load_model(path: Path) -> tuple[torch.nn.Module, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model()
    model.load_state_dict(payload["model"], strict=True)
    return model.cuda().eval(), int(payload["step"])


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("distinct_1", "distinct_2", "immediate_repeat")
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in keys
    }


def learned_epsilon(
    model: torch.nn.Module,
    mean_state: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample epsilon with the exact state-dependent scale used in training."""
    log_sigma = model.sigma_predictor(mean_state)
    per_dim_scale = (
        mean_state.detach().norm(dim=-1, keepdim=True)
        / math.sqrt(mean_state.shape[-1])
    )
    standard_normal = torch.randn(
        mean_state.shape,
        dtype=mean_state.dtype,
        device=mean_state.device,
        generator=generator,
    )
    epsilon = torch.exp(log_sigma) * per_dim_scale * standard_normal
    return epsilon, log_sigma


@torch.inference_mode()
def generate_block2_cumulative_noise(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Decode z1=K hA+eps1 and z2=K z1+eps2 in two-token blocks."""
    if new_tokens % 2:
        raise ValueError("block-2 generation requires an even token count")
    rolling = prompt
    batch = prompt.shape[0]
    log_sigmas: list[torch.Tensor] = []
    relative_noises: list[torch.Tensor] = []
    expansion_residual = 0.0
    for _ in range(new_tokens // 2):
        positions = torch.arange(rolling.shape[1], device=prompt.device)
        encoded, _ = model.encode(rolling, positions)
        h_a = encoded[:, -1]

        clean1 = model.operator(h_a)
        epsilon1, log_sigma1 = learned_epsilon(
            model, clean1, generator
        )
        state1 = clean1 + epsilon1

        # K is linear, so this is simultaneously
        # K(K hA + eps1) and K^2 hA + K eps1.
        state2_mean = model.operator(state1)
        clean2 = model.operator(clean1)
        propagated_epsilon1 = model.operator(epsilon1)
        epsilon2, log_sigma2 = learned_epsilon(
            model, state2_mean, generator
        )
        state2 = state2_mean + epsilon2
        expanded_state2 = clean2 + propagated_epsilon1 + epsilon2
        expansion_residual = max(
            expansion_residual,
            float((state2 - expanded_state2).abs().max()),
        )

        for epsilon, mean_state, log_sigma in (
            (epsilon1, clean1, log_sigma1),
            (epsilon2, state2_mean, log_sigma2),
        ):
            log_sigmas.append(log_sigma.float().reshape(-1).cpu())
            relative_noises.append(
                (
                    epsilon.float().norm(dim=-1)
                    / mean_state.float().norm(dim=-1).clamp_min(1e-12)
                ).reshape(-1).cpu()
            )

        future = torch.stack((state1, state2), dim=1)
        latent = torch.cat((encoded, future), dim=1)
        full_positions = torch.arange(
            rolling.shape[1] + 2, device=prompt.device
        ).unsqueeze(0).expand(batch, -1)
        decoded = model.encoder.decode_hidden(latent, full_positions)
        logits = model.token_logits(decoded[:, -2:]).float()
        actions = logits.argmax(dim=-1)
        rolling = torch.cat((rolling, actions), dim=1)

    all_log_sigma = torch.cat(log_sigmas)
    all_relative_noise = torch.cat(relative_noises)
    diagnostics = {
        "log_sigma_mean": float(all_log_sigma.mean()),
        "log_sigma_min": float(all_log_sigma.min()),
        "log_sigma_max": float(all_log_sigma.max()),
        "relative_noise_norm_mean": float(all_relative_noise.mean()),
        "relative_noise_norm_max": float(all_relative_noise.max()),
        "k_linearity_expansion_max_abs": expansion_residual,
    }
    return rolling[:, prompt.shape[1]:], diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--expected-step", type=int, default=1000)
    parser.add_argument("--cumulative-noise", action="store_true")
    parser.add_argument("--noise-seed", type=int, default=SEED + 3030)
    parser.add_argument("--noise-draws", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this evaluation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    tokenizer = Tokenizer.from_file("data/wikitext103/tokenizer.json")
    validation = memmap("validation")
    starts = torch.randint(
        0,
        len(validation) - GENERATION_PROMPT - GENERATION_NEW - 1,
        (GENERATION_SAMPLES,),
        generator=torch.Generator().manual_seed(SEED + 2024),
    )
    window = windows(
        validation, starts, GENERATION_PROMPT + GENERATION_NEW
    )
    prompt = window[:, :GENERATION_PROMPT]
    reference = window[:, GENERATION_PROMPT:]

    model, step = load_model(args.checkpoint)
    if step != args.expected_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_step}, found {step}"
        )
    with parametrize.cached():
        sequential = generate_sequential(model, prompt, GENERATION_NEW)
        block2 = generate_block2(model, prompt, GENERATION_NEW)
        noisy_draws = []
        noisy_diagnostics = []
        if args.cumulative_noise:
            for draw in range(args.noise_draws):
                generator = torch.Generator(device=prompt.device).manual_seed(
                    args.noise_seed + draw
                )
                tokens, diagnostics = generate_block2_cumulative_noise(
                    model,
                    prompt,
                    GENERATION_NEW,
                    generator=generator,
                )
                noisy_draws.append(tokens)
                noisy_diagnostics.append(diagnostics)

    sequential_stats = [stats(tokens) for tokens in sequential]
    block2_stats = [stats(tokens) for tokens in block2]
    aggregate_sequential = aggregate(sequential_stats)
    aggregate_block2 = aggregate(block2_stats)
    noisy_stats = [
        [stats(tokens) for tokens in draw] for draw in noisy_draws
    ]
    aggregate_noisy = [
        aggregate(draw_stats) for draw_stats in noisy_stats
    ]

    report = [
        "# i-ResNet spectral checkpoint: block-K^2 generation at step 1000",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Step: {step}",
        f"- Prompts: {GENERATION_SAMPLES} fixed validation samples, "
        f"{GENERATION_PROMPT} BPE tokens each",
        f"- Continuation: {GENERATION_NEW} greedy BPE tokens",
        f"- Sequential aggregate: `{aggregate_sequential}`",
        f"- Block-K^2 aggregate: `{aggregate_block2}`",
    ]
    if args.cumulative_noise:
        report.extend(
            [
                f"- Noise draws: {args.noise_draws}; seeds "
                f"`{args.noise_seed}..{args.noise_seed + args.noise_draws - 1}`",
                "- Noisy recurrence: "
                "`z1 = K hA + eps1`; "
                "`z2 = K z1 + eps2 = K^2 hA + K eps1 + eps2`",
                f"- First noisy aggregate: `{aggregate_noisy[0]}`",
                f"- First noisy diagnostics: `{noisy_diagnostics[0]}`",
            ]
        )
    report.append("")
    for index in range(GENERATION_SAMPLES):
        prompt_text = tokenizer.decode(prompt[index].tolist())
        reference_text = tokenizer.decode(reference[index].tolist())
        sequential_text = tokenizer.decode(sequential[index].tolist())
        block2_text = tokenizer.decode(block2[index].tolist())
        report.extend(
            [
                f"## Sample {index + 1}",
                "",
                f"**Prompt:** {prompt_text}",
                "",
                f"**Reference:** {reference_text}",
                "",
                "**Sequential (1 token/re-grounding)** "
                f"(`{sequential_stats[index]}`): {sequential_text}",
                "",
                "**Block-K^2 (2 tokens/re-grounding)** "
                f"(`{block2_stats[index]}`): {block2_text}",
                "",
            ]
        )
        if args.cumulative_noise:
            noisy_text = tokenizer.decode(noisy_draws[0][index].tolist())
            report.extend(
                [
                    "**Block-K^2 cumulative learned noise, first draw** "
                    f"(`{noisy_stats[0][index]}`): {noisy_text}",
                    "",
                ]
            )

    args.record_dir.mkdir(parents=True, exist_ok=True)
    if args.cumulative_noise:
        output_dir = args.record_dir / "step1000_block2_cumulative_noise"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "generation.md"
        metrics_path = output_dir / "metrics.tsv"
    else:
        report_path = args.record_dir / "block2_generation_step1000.md"
        metrics_path = args.record_dir / "block2_generation_step1000.tsv"
    report_path.write_text("\n".join(report))
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ("method", "distinct_1", "distinct_2", "immediate_repeat")
        )
        for method, values in (
            ("sequential", aggregate_sequential),
            ("block_k2", aggregate_block2),
        ):
            writer.writerow(
                (
                    method,
                    values["distinct_1"],
                    values["distinct_2"],
                    values["immediate_repeat"],
                )
            )
        for draw, values in enumerate(aggregate_noisy):
            writer.writerow(
                (
                    f"block_k2_cumulative_noise_draw_{draw}",
                    values["distinct_1"],
                    values["distinct_2"],
                    values["immediate_repeat"],
                )
            )
    if args.cumulative_noise:
        diagnostics_path = output_dir / "diagnostics.tsv"
        with diagnostics_path.open("w", newline="") as handle:
            keys = tuple(noisy_diagnostics[0])
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("draw", "seed", *keys))
            for draw, values in enumerate(noisy_diagnostics):
                writer.writerow(
                    (
                        draw,
                        args.noise_seed + draw,
                        *(values[key] for key in keys),
                    )
                )
    print(f"checkpoint step {step}")
    print(f"aggregate sequential: {aggregate_sequential}")
    print(f"aggregate block-K^2: {aggregate_block2}")
    for draw, (values, diagnostics) in enumerate(
        zip(aggregate_noisy, noisy_diagnostics)
    ):
        print(
            f"aggregate cumulative-noise draw {draw}: {values}; "
            f"diagnostics={diagnostics}"
        )
    print(f"wrote {report_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
