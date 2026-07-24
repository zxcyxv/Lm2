"""Standard dense one-step teacher-forced PPL for a window-3 checkpoint."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from train_k1_decoder_inverse_ablation_13m import CONTEXT, memmap, windows
from train_k1_ha_skew_learned_noise_mse_ce_13m import forward_pass
from train_k3_ha_skew_clean_window3_mse_ce_13m import (
    EXPERIMENT_ID,
    make_model,
)


DEFAULT_CHECKPOINT = (
    Path("outputs/experiments") / EXPERIMENT_ID / "step1750.pt"
)
DEFAULT_STARTS = (
    Path("experiments/records")
    / EXPERIMENT_ID
    / "validation_starts.tsv"
)
DEFAULT_OUTPUT = (
    Path("experiments/records")
    / EXPERIMENT_ID
    / "step1750_sequential_ppl"
)


def load_starts(path: Path) -> torch.Tensor:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return torch.tensor(
            [int(row["token_start"]) for row in rows],
            dtype=torch.long,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--starts", type=Path, default=DEFAULT_STARTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-step", type=int, default=1750)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this evaluation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    step = int(payload["step"])
    if step != args.expected_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_step}, found {step}"
        )
    model = make_model()
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    validation = memmap("validation")
    starts = load_starts(args.starts)

    nll_sum = 0.0
    correct = 0
    decisions = 0
    with torch.inference_mode():
        for begin in range(0, len(starts), args.micro):
            window = windows(
                validation, starts[begin : begin + args.micro]
            )
            targets = window[:, 1:]
            logits, _, _, _ = forward_pass(
                model, window, inject_noise=False
            )
            logits = logits.float()
            nll_sum += float(
                F.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten(),
                    reduction="sum",
                )
            )
            correct += int(logits.argmax(-1).eq(targets).sum())
            decisions += targets.numel()
    nll = nll_sum / decisions
    perplexity = math.exp(nll)
    accuracy = correct / decisions

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.tsv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("step", "examples", "tokens", "nll", "ppl", "accuracy"))
        writer.writerow(
            (step, len(starts), decisions, nll, perplexity, accuracy)
        )
    analysis_path = args.output_dir / "analysis.md"
    analysis_path.write_text(
        "\n".join(
            [
                "# Step-1750 standard sequential PPL",
                "",
                f"- Examples: {len(starts)}",
                f"- Decisions: {decisions}",
                f"- NLL: {nll:.9f}",
                f"- PPL: {perplexity:.9f}",
                f"- Accuracy: {accuracy:.9f}",
            ]
        )
    )
    print(f"checkpoint step {step}")
    print(f"examples={len(starts)} decisions={decisions}")
    print(f"nll={nll:.9f} ppl={perplexity:.9f} accuracy={accuracy:.9f}")
    print(f"wrote {metrics_path}")
    print(f"wrote {analysis_path}")


if __name__ == "__main__":
    main()
