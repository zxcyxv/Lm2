"""Exact seed replay with disjoint pre-clip gradient attribution."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_input_prenorm_sqrtn_hidden_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-input-prenorm-sqrtn-hidden-residual-gradient-"
    "layer-replay-audit"
)
RECORD_DIR = Path("experiments/records") / EXPERIMENT_ID
SOURCE_RECORD = run.DEFAULT_RECORD
TARGET_STEPS = (1, 50, 100, 200, 300)


def gradient_group(name: str) -> str:
    if name == "encoder.embed.weight":
        return "encoder_embedding_and_tied_head"
    if name.startswith("encoder.blocks.0."):
        return "encoder_block_0"
    if name.startswith("encoder.blocks.1."):
        return "encoder_block_1"
    if name.startswith("encoder.head_norm."):
        return "encoder_head_norm"
    if name.startswith("complex_self_prediction."):
        leaf = name.removeprefix("complex_self_prediction.")
        return f"central_{leaf.replace('.', '_')}"
    return "other"


def reference_rows() -> dict[int, dict[str, str]]:
    with (SOURCE_RECORD / "metrics.tsv").open(newline="") as handle:
        return {
            int(row["step"]): row
            for row in csv.DictReader(handle, delimiter="\t")
            if int(row["step"]) in TARGET_STEPS
        }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("exact replay requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    training = base.memmap("train")
    model = base.make_model().cuda().train()
    preflight = base.preflight(model, training)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=base.PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    data_generator = torch.Generator().manual_seed(base.SEED)
    references = reference_rows()
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    parameter_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []

    for index in range(max(TARGET_STEPS)):
        lr = base.lr_at(index, base.SCHEDULE_STEPS)
        for optimizer_group in optimizer.param_groups:
            optimizer_group["lr"] = lr
        window = base.sampled_windows(
            training,
            base.EFFECTIVE_BATCH,
            data_generator,
            base.TRAIN_WINDOW_LENGTH,
        )
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        micro_windows = window.split(base.MICROBATCH)
        for micro_window in micro_windows:
            loss, _, _, output = base.objective(model, micro_window)
            scale = 1.0 / len(micro_windows)
            (loss * scale).backward()
            train_loss += float(loss.detach()) * scale
            del loss, output

        step = index + 1
        if step in TARGET_STEPS:
            group_squares: dict[str, float] = {}
            global_square = 0.0
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    grad_square = 0.0
                    grad_norm = 0.0
                    grad_max = 0.0
                else:
                    if not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError(
                            f"non-finite replay gradient at step {step}: {name}"
                        )
                    grad_float = gradient.detach().float()
                    grad_square = float(grad_float.square().sum(dtype=torch.float64))
                    grad_norm = math.sqrt(grad_square)
                    grad_max = float(grad_float.abs().max())
                global_square += grad_square
                group = gradient_group(name)
                group_squares[group] = group_squares.get(group, 0.0) + grad_square
                parameter_norm = float(parameter.detach().float().norm())
                parameter_rows.append({
                    "step": step,
                    "group": group,
                    "parameter": name,
                    "numel": parameter.numel(),
                    "gradient_norm": grad_norm,
                    "gradient_max_abs": grad_max,
                    "parameter_norm": parameter_norm,
                    "gradient_to_parameter_norm": (
                        grad_norm / parameter_norm if parameter_norm else math.nan
                    ),
                })

            global_norm = math.sqrt(global_square)
            group_square_sum = sum(group_squares.values())
            for group, square in sorted(group_squares.items()):
                group_rows.append({
                    "step": step,
                    "group": group,
                    "gradient_norm": math.sqrt(square),
                    "gradient_square_fraction": (
                        square / global_square if global_square else 0.0
                    ),
                    "global_gradient_norm": global_norm,
                })
            reference = float(references[step]["gradient_norm"])
            replay_rows.append({
                "step": step,
                "train_loss": train_loss,
                "reference_gradient_norm": reference,
                "replay_gradient_norm": global_norm,
                "absolute_gradient_norm_error": abs(global_norm - reference),
                "relative_group_square_sum_error": abs(
                    group_square_sum - global_square
                ) / max(global_square, torch.finfo(torch.float64).tiny),
            })

        torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP_NORM)
        optimizer.step()

    def write_dicts(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)

    write_dicts(RECORD_DIR / "parameter_gradients.tsv", parameter_rows)
    write_dicts(RECORD_DIR / "group_gradients.tsv", group_rows)
    write_dicts(RECORD_DIR / "replay_checks.tsv", replay_rows)
    with (RECORD_DIR / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("source_experiment", run.EXPERIMENT_ID))
        writer.writerow(("seed", base.SEED))
        writer.writerow(("batch", base.EFFECTIVE_BATCH))
        writer.writerow(("microbatch", base.MICROBATCH))
        writer.writerow(("target_steps", ",".join(map(str, TARGET_STEPS))))
        writer.writerow(("preflight_initial_loss", preflight["initial_loss"]))


if __name__ == "__main__":
    main()
