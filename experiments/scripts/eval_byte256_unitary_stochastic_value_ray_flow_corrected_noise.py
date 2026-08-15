"""Correct noise sensitivity using identical full-encode evaluation paths."""
from __future__ import annotations

import csv
from pathlib import Path

import torch

import train_byte256_unitary_stochastic_value_ray_flow_h16_300 as experiment


RECORD = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-stochastic-value-ray-flow-corrected-"
    "noise-audit"
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    validation = experiment.base.memmap("validation")
    starts = experiment.base.load_fixed_starts(
        experiment.VALIDATION_STARTS,
        64,
    )
    windows = experiment.base.windows_of_length(
        validation,
        starts,
        experiment.base.EVAL_WINDOW_LENGTH,
    )
    checkpoints = [
        (
            "parent_step33570",
            experiment.CHECKPOINT,
            experiment.VALUE_NOISE_SCALE,
        ),
        *[
            (
                arm.name,
                experiment.OUTPUT / f"{arm.name}_step300.pt",
                arm.value_noise_scale,
            )
            for arm in experiment.ARMS
        ],
    ]
    rows = []
    for name, checkpoint, value_noise_scale in checkpoints:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model = experiment.base.make_model()
        model.load_state_dict(payload["model"])
        experiment.configure_scan_backend(
            model,
            "fused-triton-rotating-frame",
        )
        model = model.cuda().eval()
        metrics = experiment.evaluate(
            model,
            windows,
            value_noise_scale=value_noise_scale,
            microbatch=16,
        )
        row = {
            "checkpoint": name,
            "value_noise_scale": value_noise_scale,
            **metrics,
        }
        rows.append(row)
        print(
            f"{name} scale={value_noise_scale:.5f} "
            f"det_nll={metrics['deterministic_block_nll']:.6f} "
            f"stoch_nll={metrics['stochastic_block_nll']:.6f} "
            f"logit_rms={metrics['noise_logit_rms']:.6f} "
            f"top1_disagree={metrics['noise_top1_disagreement']:.6f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
