"""Matched sequential versus clean block-3 generation at step 1000."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_ha_skew_block2_generation import generate_block
from eval_ha_skew_generation_quality import generate_sequential
from rotlm.evaluation import SEED, memmap, token_stats as stats, windows
from train_k3_ha_skew_clean_window3_mse_ce_13m import (
    EXPERIMENT_ID,
    make_model,
)


PROMPT_TOKENS = 64
NEW_TOKENS = 63
SAMPLES = 5
BLOCK = 3
DEFAULT_CHECKPOINT = (
    Path("outputs/experiments") / EXPERIMENT_ID / "step1000.pt"
)
DEFAULT_OUTPUT = (
    Path("experiments/records")
    / EXPERIMENT_ID
    / "step1000_block3_generation"
)


def load_model(path: Path) -> tuple[torch.nn.Module, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model()
    model.load_state_dict(payload["model"], strict=True)
    return model.cuda().eval(), int(payload["step"])


def repetition_split(tokens: torch.Tensor) -> dict[str, float]:
    equal = tokens[:, 1:].eq(tokens[:, :-1])
    destinations = torch.arange(1, tokens.shape[1], device=tokens.device)
    within = destinations.remainder(BLOCK).ne(0)
    boundary = ~within
    return {
        "within_block_repeat": float(equal[:, within].float().mean()),
        "boundary_repeat": float(equal[:, boundary].float().mean()),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("distinct_1", "distinct_2", "immediate_repeat")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-step", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this evaluation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    tokenizer = Tokenizer.from_file("data/wikitext103/tokenizer.json")
    validation = memmap("validation")
    # Preserve the previous audits' exact prompt-start sampling range.
    starts = torch.randint(
        0,
        len(validation) - PROMPT_TOKENS - 64 - 1,
        (SAMPLES,),
        generator=torch.Generator().manual_seed(SEED + 2024),
    )
    window = windows(validation, starts, PROMPT_TOKENS + NEW_TOKENS)
    prompt = window[:, :PROMPT_TOKENS]
    reference = window[:, PROMPT_TOKENS:]
    model, step = load_model(args.checkpoint)
    if step != args.expected_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_step}, found {step}"
        )
    sequential = generate_sequential(model, prompt, NEW_TOKENS)
    block3 = generate_block(
        model, prompt, NEW_TOKENS, block=BLOCK
    )

    sequential_stats = [stats(row) for row in sequential]
    block3_stats = [stats(row) for row in block3]
    sequential_aggregate = aggregate(sequential_stats)
    block3_aggregate = aggregate(block3_stats)
    sequential_repetition = repetition_split(sequential)
    block3_repetition = repetition_split(block3)

    report = [
        f"# Step-{step} clean K^1..K^3 block generation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Sequential aggregate: `{sequential_aggregate}`",
        f"- Sequential repetition split: `{sequential_repetition}`",
        f"- Block-3 aggregate: `{block3_aggregate}`",
        f"- Block-3 repetition split: `{block3_repetition}`",
        "",
    ]
    for index in range(SAMPLES):
        report.extend(
            [
                f"## Sample {index + 1}",
                "",
                "**Prompt:** "
                + tokenizer.decode(prompt[index].tolist()),
                "",
                "**Reference:** "
                + tokenizer.decode(reference[index].tolist()),
                "",
                "**Sequential:** "
                + tokenizer.decode(sequential[index].tolist()),
                "",
                "**Block-3:** "
                + tokenizer.decode(block3[index].tolist()),
                "",
            ]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "generation.md"
    report_path.write_text("\n".join(report))
    metrics_path = args.output_dir / "metrics.tsv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "method",
                "distinct_1",
                "distinct_2",
                "immediate_repeat",
                "within_block_repeat",
                "boundary_repeat",
            )
        )
        for method, values, repetition in (
            (
                "sequential",
                sequential_aggregate,
                sequential_repetition,
            ),
            ("block3", block3_aggregate, block3_repetition),
        ):
            writer.writerow(
                (
                    method,
                    values["distinct_1"],
                    values["distinct_2"],
                    values["immediate_repeat"],
                    repetition["within_block_repeat"],
                    repetition["boundary_repeat"],
                )
            )
    print(f"checkpoint step {step}")
    print(f"sequential: {sequential_aggregate} {sequential_repetition}")
    print(f"block-3: {block3_aggregate} {block3_repetition}")
    print(f"wrote {report_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
