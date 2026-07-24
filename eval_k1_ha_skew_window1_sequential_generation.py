"""Matched 63-token sequential generation for the final window-1 model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_ha_skew_generation_quality import (
    generate_sequential,
    load_model,
)
from rotlm.evaluation import SEED, memmap, token_stats as stats, windows


PROMPT_TOKENS = 64
NEW_TOKENS = 63
SAMPLES = 5
DEFAULT_CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m/last.pt"
)
DEFAULT_OUTPUT = Path(
    "experiments/records/"
    "EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m/"
    "matched_sequential_generation_63tok"
)


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("distinct_1", "distinct_2", "immediate_repeat")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-step", type=int, default=6000)
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
    generated = generate_sequential(model, prompt, NEW_TOKENS)
    sample_stats = [stats(row) for row in generated]
    aggregate_stats = aggregate(sample_stats)

    report = [
        "# Window-1 final matched sequential generation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Step: {step}",
        f"- Aggregate: `{aggregate_stats}`",
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
                f"**Sequential** (`{sample_stats[index]}`): "
                + tokenizer.decode(generated[index].tolist()),
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
            ("method", "distinct_1", "distinct_2", "immediate_repeat")
        )
        writer.writerow(
            (
                "window1_sequential",
                aggregate_stats["distinct_1"],
                aggregate_stats["distinct_2"],
                aggregate_stats["immediate_repeat"],
            )
        )
    print(f"checkpoint step {step}")
    print(f"aggregate: {aggregate_stats}")
    print(f"wrote {report_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
