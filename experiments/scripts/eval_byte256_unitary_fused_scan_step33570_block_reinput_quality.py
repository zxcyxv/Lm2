"""Compare 1/2/4/8/16-byte re-input generation at epoch 10."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import torch

import eval_byte256_unitary_fused_scan_step10071_flow_generation as flow
import eval_byte256_unitary_fused_scan_step10071_generation_quality as quality
from train_byte256_unitary_time_varying_scan_h16_1000 import (
    configure_scan_backend,
)


base = flow.base
CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-"
    "13m-rtx5090/step33570.pt"
)
RECORD = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-step33570-"
    "block-reinput-quality"
)
BLOCKS = (1, 2, 4, 8, 16)
SAMPLES = 64
REPORT_SAMPLES = 12
GENERATE = 128


def phase_rows(
    mode: str,
    block: int,
    values: torch.Tensor,
    gold: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    for phase in range(block):
        selected = values[:, phase::block]
        flat = selected.flatten().cpu().tolist()
        counts = Counter(flat)
        modal_byte, modal_count = counts.most_common(1)[0]
        rows.append({
            "mode": mode,
            "block_size": block,
            "phase_1based": phase + 1,
            "aligned_byte_accuracy": float(
                (selected == gold[:, phase::block]).float().mean()
            ),
            "byte_entropy_bits": quality.byte_entropy(flat),
            "unique_bytes": len(counts),
            "space_fraction": counts[32] / len(flat),
            "modal_byte": modal_byte,
            "modal_byte_fraction": modal_count / len(flat),
        })
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 33570:
        raise RuntimeError("expected step 33570")
    model = base.make_model().cuda().eval()
    configure_scan_backend(model, "fused-triton-rotating-frame")
    model.load_state_dict(checkpoint["model"])

    validation = base.memmap("validation")
    starts = base.load_fixed_starts(flow.BASELINE_STARTS, SAMPLES)
    prompts = base.windows_of_length(validation, starts, base.CONTEXT)
    gold = base.windows_of_length(
        validation,
        starts + base.CONTEXT,
        GENERATE,
    )
    outputs: dict[str, torch.Tensor] = {"gold": gold}
    for block in BLOCKS:
        outputs[f"b{block}_greedy"] = flow.generate(
            model,
            prompts,
            block,
            generate_count=GENERATE,
        )
        generator = torch.Generator(device="cuda").manual_seed(
            base.SEED + 41000 + block
        )
        outputs[f"b{block}_top_p"] = flow.generate(
            model,
            prompts,
            block,
            selector=quality.top_p_selector(generator),
            generate_count=GENERATE,
        )

    metric_rows = [
        quality.aggregate_mode(mode, values, gold)
        for mode, values in outputs.items()
    ]
    phases = [
        row
        for block in BLOCKS
        for suffix in ("greedy", "top_p")
        for row in phase_rows(
            f"b{block}_{suffix}",
            block,
            outputs[f"b{block}_{suffix}"],
            gold,
        )
    ]
    sample_rows = []
    ordered_modes = ["gold"] + [
        f"b{block}_{suffix}"
        for block in BLOCKS
        for suffix in ("greedy", "top_p")
    ]
    for sample in range(REPORT_SAMPLES):
        prompt_values = prompts[sample, -96:].cpu().tolist()
        for mode in ordered_modes:
            values = outputs[mode][sample].cpu().tolist()
            sample_rows.append({
                "sample": sample,
                "validation_start": int(starts[sample]),
                "mode": mode,
                "prompt_suffix_escaped": quality.escaped(prompt_values),
                "continuation_escaped": quality.escaped(values),
                "continuation_hex": bytes(values).hex(),
            })

    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "metrics.tsv", metric_rows)
    write_rows(RECORD / "block_phase.tsv", phases)
    write_rows(RECORD / "samples.tsv", sample_rows)
    for row in metric_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
