"""Audit greedy AR versus four-byte composition at the step-5000 snapshot."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from rotlm.dataset import token_memmap
from rotlm.evaluation import (
    continuation_metrics,
    generate_input_conditioned_composition_blocks,
    token_stats,
)
from train_byte256_simplex_tied_self_composition_h1_ce_13m import (
    CONTEXT,
    DATA_ROOT,
    EXPERIMENT_ID as TRAIN_EXPERIMENT_ID,
    SEED,
    make_model,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length


EXPERIMENT_ID = "EXP-20260801-byte256-simplex-step5000-block4-generation-audit"
PROMPT_BYTES = CONTEXT
NEW_BYTES = 64
SAMPLES = 3
BLOCK = 4
DEFAULT_CHECKPOINT = (
    Path("outputs/experiments") / TRAIN_EXPERIMENT_ID / "step5000.pt"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID


def render_bytes(tokens: torch.Tensor) -> str:
    return bytes(tokens.tolist()).decode("utf-8", errors="backslashreplace")


def byte_metrics(tokens: torch.Tensor) -> dict[str, float]:
    values = bytes(tokens.tolist())
    decoded = values.decode("utf-8", errors="replace")
    printable = sum(
        byte in (9, 10, 13) or 32 <= byte <= 126 for byte in values
    )
    try:
        values.decode("utf-8", errors="strict")
        valid_utf8 = 1.0
    except UnicodeDecodeError:
        valid_utf8 = 0.0
    return {
        **token_stats(tokens),
        "ascii_printable_fraction": printable / len(values),
        "strict_utf8_valid": valid_utf8,
        "replacement_characters": float(decoded.count("\ufffd")),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--expected-step", type=int, default=5000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this audit requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = int(payload["step"])
    if step != args.expected_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_step}, found {step}"
        )
    model = make_model()
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    del payload

    validation = token_memmap("validation", root=DATA_ROOT)
    starts = torch.randint(
        0,
        len(validation) - PROMPT_BYTES - NEW_BYTES,
        (SAMPLES,),
        generator=torch.Generator().manual_seed(SEED + 2024),
    )
    windows = windows_of_length(
        validation, starts, PROMPT_BYTES + NEW_BYTES
    )
    prompt = windows[:, :PROMPT_BYTES]
    reference = windows[:, PROMPT_BYTES:]
    ar = generate_input_conditioned_composition_blocks(
        model,
        prompt,
        NEW_BYTES,
        block=1,
        context_length=CONTEXT,
    )
    block4 = generate_input_conditioned_composition_blocks(
        model,
        prompt,
        NEW_BYTES,
        block=BLOCK,
        context_length=CONTEXT,
    )

    ar_rows = [byte_metrics(row) for row in ar]
    block_rows = [byte_metrics(row) for row in block4]
    ar_aggregate = aggregate(ar_rows)
    block_aggregate = aggregate(block_rows)
    agreement_rows = [
        continuation_metrics(block4[index], ar[index])
        for index in range(SAMPLES)
    ]
    agreement = aggregate(agreement_rows)

    args.record_dir.mkdir(parents=True, exist_ok=True)
    with (args.record_dir / "metrics.tsv").open("w", newline="") as handle:
        fields = ("sample", "method", *ar_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(SAMPLES):
            for method, rows in (("ar", ar_rows), ("block4", block_rows)):
                writer.writerow(
                    {"sample": index, "method": method, **rows[index]}
                )
    with (args.record_dir / "agreement.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in agreement.items():
            writer.writerow((key, value))

    lines = [
        f"# Step-{step} byte AR versus block-4 generation",
        "",
        f"- AR aggregate: `{ar_aggregate}`",
        f"- Block-4 aggregate: `{block_aggregate}`",
        f"- Mean per-sample block-4/AR agreement: `{agreement}`",
        "",
    ]
    for index in range(SAMPLES):
        lines.extend(
            (
                f"## Sample {index + 1}",
                "",
                "**Prompt**",
                "",
                "```text",
                render_bytes(prompt[index]),
                "```",
                "",
                "**Reference**",
                "",
                "```text",
                render_bytes(reference[index]),
                "```",
                "",
                "**Greedy AR**",
                "",
                "```text",
                render_bytes(ar[index]),
                "```",
                "",
                "**Four-byte composition**",
                "",
                "```text",
                render_bytes(block4[index]),
                "```",
                "",
            )
        )
    (args.record_dir / "interpretation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"checkpoint_step={step}")
    print(f"ar={ar_aggregate}")
    print(f"block4={block_aggregate}")
    print(f"agreement={agreement}")
    print(f"wrote={args.record_dir}")


if __name__ == "__main__":
    main()
