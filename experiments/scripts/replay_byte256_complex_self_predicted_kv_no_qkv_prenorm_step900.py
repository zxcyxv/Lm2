"""Exactly replay updates 501--900 of the no-QKV-pre-norm H16 run.

This producer reconstructs the missing historical step-900 state from the
preserved step-500 model, optimizer, and data-generator state.  It deliberately
uses the registered common model, objective, sampler, and learning-rate code;
it does not reimplement any training computation.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_h16_attached_stride16_ce_only_13m as registered


EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "no-qkv-prenorm-step900-gradient-forensics"
)
RECORD_DIR = Path("experiments/records") / EXPERIMENT_ID
OUTPUT_DIR = Path("outputs/experiments") / EXPERIMENT_ID
SOURCE_EXPERIMENT_ID = registered.EXPERIMENT_ID
SOURCE_RECORD_DIR = Path("experiments/records") / SOURCE_EXPERIMENT_ID
SOURCE_OUTPUT_DIR = Path("outputs/experiments") / SOURCE_EXPERIMENT_ID
SOURCE_CHECKPOINT = SOURCE_OUTPUT_DIR / "step0500.pt"
SOURCE_METRICS = SOURCE_RECORD_DIR / "metrics.tsv"
START_STEP = 500
END_STEP = 900
VERIFICATION_STEPS = frozenset((600, 700, 800, 900))
FIRST_CROSSING_AFTER_STEP = 800
FIRST_CROSSING_GRADIENT_NORM = 100.0
RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-6


def _load_source_rows() -> dict[int, dict[str, str]]:
    with SOURCE_METRICS.open(newline="") as handle:
        return {
            int(row["step"]): row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def _agrees(replayed: float, source: float) -> bool:
    return math.isclose(
        replayed,
        source,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    )


def _save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    completed_step: int,
    next_step: int | None,
    next_window: torch.Tensor | None,
    generator_state_before_next_window: torch.Tensor | None,
    train_ce: float | None,
    gradient_norm: float | None,
    config: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_generator_state": data_generator.get_state(),
        "step": completed_step,
        "next_step": next_step,
        "train_ce": train_ce,
        "gradient_norm": gradient_norm,
        "config": config,
        "reconstruction_source": str(SOURCE_CHECKPOINT),
    }
    if next_window is not None:
        payload["next_window"] = next_window.detach().cpu()
    if generator_state_before_next_window is not None:
        payload["data_generator_state_before_next_window"] = (
            generator_state_before_next_window.detach().cpu()
        )
    torch.save(payload, path)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty TSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("exact replay requires CUDA")
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    if not SOURCE_METRICS.exists():
        raise FileNotFoundError(SOURCE_METRICS)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["step"]) != START_STEP:
        raise RuntimeError(
            f"source checkpoint is step {checkpoint['step']}, expected "
            f"{START_STEP}"
        )
    config = dict(checkpoint["config"])
    if config.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise RuntimeError("source checkpoint experiment id does not match")
    if bool(config.get("pre_normalize_qkv_inputs", True)):
        raise RuntimeError("source checkpoint unexpectedly enables QKV pre-norm")

    batch = int(config["batch"])
    microbatch = int(config["microbatch"])
    schedule_steps = int(config["schedule_steps"])
    if batch % microbatch:
        raise RuntimeError("checkpoint batch is not divisible by microbatch")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = registered.base.make_model().cuda().train()
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=registered.base.PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    data_generator = torch.Generator()
    data_generator.set_state(checkpoint["data_generator_state"])
    training = registered.base.memmap("train")
    source_rows = _load_source_rows()

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory: list[dict[str, object]] = []
    verification: list[dict[str, object]] = []
    first_crossing_step: int | None = None

    for step in range(START_STEP + 1, END_STEP + 1):
        index = step - 1
        lr = registered.base.lr_at(index, schedule_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        generator_state_before_window = data_generator.get_state()
        window = registered.base.sampled_windows(
            training,
            batch,
            data_generator,
            registered.base.TRAIN_WINDOW_LENGTH,
        )
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_ce = 0.0
        train_auxiliary = 0.0
        micro_windows = window.split(microbatch)
        for micro_window in micro_windows:
            loss, token_ce, auxiliary_loss, output = (
                registered.base.objective(model, micro_window)
            )
            scale = 1.0 / len(micro_windows)
            (loss * scale).backward()
            train_loss += float(loss.detach()) * scale
            train_ce += float(token_ce.detach()) * scale
            train_auxiliary += float(auxiliary_loss.detach()) * scale
            del loss, token_ce, auxiliary_loss, output

        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                registered.base.CLIP_NORM,
            )
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")

        trajectory.append(
            {
                "step": step,
                "train_loss": train_loss,
                "train_token_ce": train_ce,
                "train_no_auxiliary_loss": train_auxiliary,
                "preclip_gradient_norm": gradient_norm,
                "lr": lr,
                "first_gradient_100_crossing": False,
            }
        )

        if step == END_STEP:
            _save_snapshot(
                OUTPUT_DIR / "pre_update0900.pt",
                model=model,
                optimizer=optimizer,
                data_generator=data_generator,
                completed_step=step - 1,
                next_step=step,
                next_window=window,
                generator_state_before_next_window=(
                    generator_state_before_window
                ),
                train_ce=train_ce,
                gradient_norm=gradient_norm,
                config=config,
            )

        if (
            first_crossing_step is None
            and step > FIRST_CROSSING_AFTER_STEP
            and gradient_norm >= FIRST_CROSSING_GRADIENT_NORM
        ):
            first_crossing_step = step
            trajectory[-1]["first_gradient_100_crossing"] = True
            crossing_path = OUTPUT_DIR / f"pre_update{step:04d}_crossing.pt"
            if step != END_STEP:
                _save_snapshot(
                    crossing_path,
                    model=model,
                    optimizer=optimizer,
                    data_generator=data_generator,
                    completed_step=step - 1,
                    next_step=step,
                    next_window=window,
                    generator_state_before_next_window=(
                        generator_state_before_window
                    ),
                    train_ce=train_ce,
                    gradient_norm=gradient_norm,
                    config=config,
                )

        if step in VERIFICATION_STEPS:
            source = source_rows[step]
            source_ce = float(source["train_token_ce"])
            source_gradient_norm = float(source["gradient_norm"])
            ce_agrees = _agrees(train_ce, source_ce)
            gradient_agrees = _agrees(
                gradient_norm,
                source_gradient_norm,
            )
            verification.append(
                {
                    "step": step,
                    "metric": "train_token_ce",
                    "source_value": source_ce,
                    "replayed_value": train_ce,
                    "absolute_error": abs(train_ce - source_ce),
                    "relative_error": abs(train_ce - source_ce)
                    / max(abs(source_ce), 1e-30),
                    "agrees": ce_agrees,
                }
            )
            verification.append(
                {
                    "step": step,
                    "metric": "preclip_gradient_norm",
                    "source_value": source_gradient_norm,
                    "replayed_value": gradient_norm,
                    "absolute_error": abs(
                        gradient_norm - source_gradient_norm
                    ),
                    "relative_error": abs(
                        gradient_norm - source_gradient_norm
                    )
                    / max(abs(source_gradient_norm), 1e-30),
                    "agrees": gradient_agrees,
                }
            )
            print(
                f"step={step} ce={train_ce:.9g} "
                f"source_ce={source_ce:.9g} gnorm={gradient_norm:.9g} "
                f"source_gnorm={source_gradient_norm:.9g}",
                flush=True,
            )

        optimizer.step()

        if step == END_STEP:
            _save_snapshot(
                OUTPUT_DIR / "post_update0900.pt",
                model=model,
                optimizer=optimizer,
                data_generator=data_generator,
                completed_step=step,
                next_step=None,
                next_window=None,
                generator_state_before_next_window=None,
                train_ce=train_ce,
                gradient_norm=gradient_norm,
                config=config,
            )

    _write_tsv(RECORD_DIR / "replay_trajectory.tsv", trajectory)
    _write_tsv(RECORD_DIR / "replay_verification.tsv", verification)
    if not all(bool(row["agrees"]) for row in verification):
        raise RuntimeError(
            "deterministic replay did not match the preserved source metrics"
        )
    if first_crossing_step is None:
        print(
            "no post-step-800 raw gradient norm crossed the diagnostic "
            f"threshold {FIRST_CROSSING_GRADIENT_NORM}; no crossing "
            "snapshot was written",
            flush=True,
        )


if __name__ == "__main__":
    main()
