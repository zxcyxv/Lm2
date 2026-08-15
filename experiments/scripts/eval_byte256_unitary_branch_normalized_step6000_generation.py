"""Long H1-reanchored versus H16-block generation at step 6000."""
from __future__ import annotations

import csv
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
CHECKPOINT = Path(
    "outputs/experiments/EXP-20260802-byte256-unitary-branch-normalized-"
    "h16-6000step-micro64/step6000.pt"
)
RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-interpretability-audit"
)
SAMPLES = 4
GENERATE = 128


def printable(values: list[int]) -> str:
    return bytes(values).decode("utf-8", errors="replace").replace("\n", "\\n")


@torch.inference_mode()
def next_block(model, contexts: torch.Tensor, block: int) -> torch.Tensor:
    dummy = torch.zeros(
        contexts.shape[0], base.TRAIN_HORIZONS,
        dtype=contexts.dtype, device=contexts.device,
    )
    window = torch.cat((contexts[:, -base.CONTEXT:], dummy), dim=1)
    _, _, _, output = base.objective(model, window)
    predictions = output.logits[:, -1].argmax(dim=-1)
    return predictions[:, :block]


def generate(model, prompts: torch.Tensor, block: int) -> torch.Tensor:
    sequences = prompts.clone()
    remaining = GENERATE
    while remaining:
        take = min(block, remaining)
        predictions = next_block(model, sequences, take)
        sequences = torch.cat((sequences, predictions), dim=1)
        remaining -= take
    return sequences[:, -GENERATE:]


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = base.make_model().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    validation = base.memmap("validation")
    starts = torch.randint(
        0,
        len(validation) - base.CONTEXT - GENERATE,
        (64,),
        generator=torch.Generator().manual_seed(base.SEED + 999),
    )[:SAMPLES]
    prompts = base.windows_of_length(validation, starts, base.CONTEXT).cuda()
    gold = base.windows_of_length(
        validation, starts + base.CONTEXT, GENERATE
    ).cuda()
    h1 = generate(model, prompts, 1)
    h16 = generate(model, prompts, 16)
    rows = []
    for sample in range(SAMPLES):
        prompt_values = prompts[sample, -96:].cpu().tolist()
        gold_values = gold[sample].cpu().tolist()
        h1_values = h1[sample].cpu().tolist()
        h16_values = h16[sample].cpu().tolist()
        rows.append({
            "sample": sample,
            "validation_start": int(starts[sample]),
            "prompt_suffix_text": printable(prompt_values),
            "gold_text": printable(gold_values),
            "h1_reanchored_text": printable(h1_values),
            "h16_block_text": printable(h16_values),
            "h1_exact_byte_accuracy": sum(a == b for a, b in zip(h1_values, gold_values)) / GENERATE,
            "h16_exact_byte_accuracy": sum(a == b for a, b in zip(h16_values, gold_values)) / GENERATE,
            "h1_unique_bytes": len(set(h1_values)),
            "h16_unique_bytes": len(set(h16_values)),
        })
    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "generation.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
