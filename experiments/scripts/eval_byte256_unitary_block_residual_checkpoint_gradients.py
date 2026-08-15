"""Fine-grained next-batch gradient audit for saved unitary-block checkpoints."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_unitary_block_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
SOURCE_OUTPUT = run.DEFAULT_OUTPUT
RECORD = Path("experiments/records/EXP-20260802-byte256-unitary-block-residual-checkpoint-gradient-audit")
CHECKPOINTS = (SOURCE_OUTPUT / "step0100.pt", SOURCE_OUTPUT / "last.pt")


def group(name: str) -> str:
    if name == "encoder.embed.weight":
        return "encoder.embedding_tied_head"
    if name.startswith("encoder.head_norm."):
        return "encoder.head_norm"
    for block in (0, 1):
        prefix = f"encoder.blocks.{block}."
        if name.startswith(prefix):
            leaf = name.removeprefix(prefix)
            if leaf.startswith("attn.qkv."):
                part = "attention_qkv"
            elif leaf.startswith("attn.proj."):
                part = "attention_output"
            elif leaf.startswith("ffn.0."):
                part = "ffn_input"
            elif leaf.startswith("ffn.2."):
                part = "ffn_output"
            elif leaf.startswith("norm1."):
                part = "attention_norm"
            elif leaf.startswith("norm2."):
                part = "ffn_norm"
            else:
                part = leaf.replace(".", "_")
            return f"encoder.block{block}.{part}"
    if name.startswith("complex_self_prediction."):
        leaf = name.removeprefix("complex_self_prediction.")
        return f"central.{leaf.rsplit('.', 1)[0]}"
    return "other"


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    training = base.memmap("train")
    parameter_rows = []
    group_rows = []
    check_rows = []
    RECORD.mkdir(parents=True, exist_ok=True)

    for path in CHECKPOINTS:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        step = int(checkpoint["step"])
        model = base.make_model().cuda().train()
        model.load_state_dict(checkpoint["model"])
        generator = torch.Generator()
        generator.set_state(checkpoint["data_generator_state"])
        window = base.sampled_windows(
            training, base.EFFECTIVE_BATCH, generator, base.TRAIN_WINDOW_LENGTH
        )
        model.zero_grad(set_to_none=True)
        micro_windows = window.split(base.MICROBATCH)
        loss_value = 0.0
        for micro in micro_windows:
            loss, _, _, output = base.objective(model, micro)
            scale = 1.0 / len(micro_windows)
            (loss * scale).backward()
            loss_value += float(loss.detach()) * scale
            del loss, output

        global_square = 0.0
        group_squares: dict[str, float] = {}
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                square = maximum = 0.0
            else:
                gradient = gradient.detach().float()
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError(f"non-finite gradient: step {step} {name}")
                square = float(gradient.square().sum(dtype=torch.float64))
                maximum = float(gradient.abs().max())
            norm = math.sqrt(square)
            parameter_norm = float(parameter.detach().float().norm())
            label = group(name)
            global_square += square
            group_squares[label] = group_squares.get(label, 0.0) + square
            parameter_rows.append({
                "checkpoint_step": step,
                "group": label,
                "parameter": name,
                "numel": parameter.numel(),
                "gradient_norm": norm,
                "gradient_max_abs": maximum,
                "parameter_norm": parameter_norm,
                "gradient_to_parameter_norm": norm / parameter_norm if parameter_norm else math.nan,
            })
        global_norm = math.sqrt(global_square)
        for label, square in sorted(group_squares.items()):
            group_rows.append({
                "checkpoint_step": step,
                "group": label,
                "gradient_norm": math.sqrt(square),
                "gradient_square_fraction": square / global_square if global_square else 0.0,
                "global_gradient_norm": global_norm,
            })
        group_sum = sum(group_squares.values())
        check_rows.append({
            "checkpoint_step": step,
            "next_batch_loss": loss_value,
            "global_gradient_norm": global_norm,
            "relative_group_square_sum_error": abs(group_sum - global_square) / max(global_square, torch.finfo(torch.float64).tiny),
        })
        del model, checkpoint, window
        torch.cuda.empty_cache()

    write_rows(RECORD / "parameter_gradients.tsv", parameter_rows)
    write_rows(RECORD / "group_gradients.tsv", group_rows)
    write_rows(RECORD / "checks.tsv", check_rows)


if __name__ == "__main__":
    main()
