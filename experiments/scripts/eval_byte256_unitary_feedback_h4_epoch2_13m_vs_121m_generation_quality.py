"""Compare epoch-2 H4 feedback quality at widths 1344 and 4352.

The 121M checkpoint is evaluated through the same shared rollout and natural
generation functions that produced the preserved 13M epoch-2 records.  The
13M tensors are not recomputed: their completed TSV evidence is copied into
the matched comparison after protocol and validation-start checks.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import math
from pathlib import Path
import time

import torch

import eval_byte256_unitary_epoch2_natural_chunk_generation_quality as natural
from eval_byte256_unitary_fused_scan_step10071_generation_quality import (
    aggregate_mode,
    escaped,
)
from rotlm.evaluation import greedy_block_reinput_rollout_metrics
import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch_121m as run


base = run.base
ROOT = Path("experiments/records")
OUTPUT = Path("outputs/experiments")
SMALL_ID = (
    "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "10epoch-13m"
)
LARGE_ID = run.EXPERIMENT_ID
RECORD = ROOT / (
    "EXP-20260806-byte256-unitary-feedback-h4-epoch2-"
    "13m-vs-121m-generation-quality"
)
SMALL_BLOCK_RECORD = ROOT / (
    "EXP-20260805-byte256-unitary-feedback-h1-h4loss-epoch2-"
    "greedy-reinput-h16-rollout-ce"
)
SMALL_GENERATION_RECORD = ROOT / (
    "EXP-20260805-byte256-unitary-epoch2-natural-chunk-generation-quality"
)
CHECKPOINT = OUTPUT / LARGE_ID / "last.pt"
STEP = 6_714
SAMPLES = 64
REPORT_SAMPLES = 12
HORIZONS = 16
BLOCK_SIZE = 4
ANCHOR_STRIDE = 16
# Preserve the completed 13M rollout producer's physical batching.  A separate
# full-context reproduction below uses microbatch 4, matching the widened
# training producer, so checkpoint identity and actual prefix re-encoding are
# audited independently.
ROLLOUT_MICROBATCH = 8
REGISTERED_EVAL_MICROBATCH = 4
GENERATION_MICROBATCH = 16
GENERATE = 128
SMALL_PARAMETERS = 13_215_008
LARGE_PARAMETERS = run.EXPECTED_PARAMETERS
OBJECTIVE = (
    "four_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_"
    "attached_stride4_h16_monitor"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_step(path: Path) -> dict[str, str]:
    rows = [row for row in read_rows(path) if int(row["step"]) == STEP]
    if len(rows) != 1:
        raise RuntimeError(f"expected one step-{STEP} row in {path}")
    return rows[0]


def load_large_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["step"]) != STEP:
        raise RuntimeError(f"wrong checkpoint step: {checkpoint['step']}")
    config = checkpoint.get("config", {})
    expected = {
        "experiment_id": LARGE_ID,
        "objective": OBJECTIVE,
        "schedule_steps": 33_570,
        "batch": 64,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"checkpoint {key} mismatch: {config.get(key)!r} != {value!r}"
            )

    model = base.make_model().cuda().eval()
    parameters = base.parameter_count(model)
    if parameters != LARGE_PARAMETERS:
        raise RuntimeError(
            f"parameter count changed: {parameters} != {LARGE_PARAMETERS}"
        )
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    gc.collect()
    return model, config


def preserved_small_horizons() -> list[dict[str, object]]:
    rows = [
        row
        for row in read_rows(SMALL_BLOCK_RECORD / "horizon_metrics.tsv")
        if row["mode"] == "feedback_h4loss_greedy_block4"
    ]
    if [int(row["horizon"]) for row in rows] != list(range(1, 17)):
        raise RuntimeError("preserved 13M H4 horizon rows are incomplete")
    return [
        {
            "model": "feedback_h4_13m_epoch2",
            "width": 1344,
            "parameters": SMALL_PARAMETERS,
            "horizon": int(row["horizon"]),
            "gold_aligned_rollout_ce": float(
                row["gold_aligned_rollout_ce"]
            ),
            "gold_target_accuracy": float(row["gold_target_accuracy"]),
            "labels": int(row["labels"]),
            "source": SMALL_BLOCK_RECORD.name,
        }
        for row in rows
    ]


def block_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model": rows[0]["model"],
        "width": rows[0]["width"],
        "parameters": rows[0]["parameters"],
        "chunk_size": BLOCK_SIZE,
        "gold_reset_interval": HORIZONS,
        "h1_h16_mean_rollout_ce": sum(
            float(row["gold_aligned_rollout_ce"]) for row in rows
        )
        / HORIZONS,
        "h1_h16_mean_accuracy": sum(
            float(row["gold_target_accuracy"]) for row in rows
        )
        / HORIZONS,
        "labels": sum(int(row["labels"]) for row in rows),
        "source": rows[0]["source"],
    }


def evaluate_large_rollout(model, windows: torch.Tensor):
    metrics = greedy_block_reinput_rollout_metrics(
        model,
        windows,
        context_length=base.CONTEXT,
        horizons=HORIZONS,
        block_size=BLOCK_SIZE,
        anchor_stride=ANCHOR_STRIDE,
        microbatch=ROLLOUT_MICROBATCH,
    )
    if metrics.total_labels != SAMPLES * 16 * HORIZONS:
        raise RuntimeError(f"unexpected rollout label count: {metrics.total_labels}")
    if not all(
        math.isfinite(value)
        for vector in (
            metrics.horizon_target_cross_entropy,
            metrics.horizon_target_accuracy,
        )
        for value in vector
    ):
        raise RuntimeError("non-finite 121M rollout metric")
    return [
        {
            "model": "feedback_h4_121m_epoch2",
            "width": run.WIDTH,
            "parameters": LARGE_PARAMETERS,
            "horizon": horizon,
            "gold_aligned_rollout_ce": (
                metrics.horizon_target_cross_entropy[horizon - 1]
            ),
            "gold_target_accuracy": (
                metrics.horizon_target_accuracy[horizon - 1]
            ),
            "labels": metrics.labels_per_horizon,
            "source": LARGE_ID,
        }
        for horizon in range(1, HORIZONS + 1)
    ]


def generate_large(model, prompts: torch.Tensor):
    outputs = []
    weighted_margin = 0.0
    weighted_entropy = 0.0
    count = 0
    for micro_prompts in prompts.split(GENERATION_MICROBATCH):
        generated, margin, entropy = natural.generate(
            model,
            micro_prompts,
            BLOCK_SIZE,
            scan=False,
        )
        outputs.append(generated)
        batch = generated.shape[0]
        weighted_margin += margin * batch
        weighted_entropy += entropy * batch
        count += batch
    return (
        torch.cat(outputs, dim=0),
        weighted_margin / count,
        weighted_entropy / count,
    )


def generation_row(
    model: str,
    width: int,
    parameters: int,
    source: str,
    values: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model,
        "width": width,
        "parameters": parameters,
        "source": source,
        **{key: value for key, value in values.items() if key != "mode"},
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    small_starts_path = ROOT / SMALL_ID / "validation_starts.tsv"
    large_starts_path = ROOT / LARGE_ID / "validation_starts.tsv"
    if small_starts_path.read_bytes() != large_starts_path.read_bytes():
        raise RuntimeError("13M and 121M validation starts differ")
    starts = base.load_fixed_starts(small_starts_path, SAMPLES)
    validation = base.memmap("validation")
    windows = base.windows_of_length(
        validation,
        starts,
        base.CONTEXT + HORIZONS,
    )
    prompts = windows[:, : base.CONTEXT]
    gold = base.windows_of_length(
        validation,
        starts + base.CONTEXT,
        GENERATE,
    )

    model, config = load_large_model()
    open_row = read_step(ROOT / LARGE_ID / "metrics.tsv")
    registered_metrics = base.evaluate(
        model,
        validation,
        starts,
        REGISTERED_EVAL_MICROBATCH,
    )
    registered_reproduction_error = max(
        abs(
            float(registered_metrics[f"h{horizon}_target_nll"])
            - float(open_row[f"val_h{horizon}_target_nll"])
        )
        for horizon in range(1, BLOCK_SIZE + 1)
    )
    if registered_reproduction_error >= 2e-3:
        raise RuntimeError(
            "121M checkpoint does not reproduce its registered evaluator: "
            f"{registered_reproduction_error}"
        )

    small_horizons = preserved_small_horizons()
    large_horizons = evaluate_large_rollout(model, windows)
    prefix_reencode_vs_full_context_error = max(
        abs(
            float(large_horizons[horizon - 1]["gold_aligned_rollout_ce"])
            - float(registered_metrics[f"h{horizon}_target_nll"])
        )
        for horizon in range(1, BLOCK_SIZE + 1)
    )

    generated, margin, predictive_entropy = generate_large(model, prompts)
    large_generation = aggregate_mode(
        "feedback_h4_121m_epoch2",
        generated,
        gold,
    )
    large_generation["mean_top1_top2_logit_margin"] = margin
    large_generation["mean_predictive_entropy_nats"] = predictive_entropy

    old_generation_rows = read_rows(SMALL_GENERATION_RECORD / "metrics.tsv")
    old_small = next(
        row for row in old_generation_rows if row["mode"] == "h4_greedy_block4"
    )
    gold_generation = aggregate_mode("gold", gold, gold)
    gold_generation["mean_top1_top2_logit_margin"] = float("nan")
    gold_generation["mean_predictive_entropy_nats"] = float("nan")
    generation_rows = [
        generation_row("gold", 0, 0, "validation", gold_generation),
        generation_row(
            "feedback_h4_13m_epoch2",
            1344,
            SMALL_PARAMETERS,
            SMALL_GENERATION_RECORD.name,
            old_small,
        ),
        generation_row(
            "feedback_h4_121m_epoch2",
            run.WIDTH,
            LARGE_PARAMETERS,
            LARGE_ID,
            large_generation,
        ),
    ]

    small_block = block_summary(small_horizons)
    large_block = block_summary(large_horizons)
    block_rows = [small_block, large_block]
    comparison_rows = [
        {
            "metric": "h1_h16_mean_rollout_ce",
            "value_13m": small_block["h1_h16_mean_rollout_ce"],
            "value_121m": large_block["h1_h16_mean_rollout_ce"],
            "value_121m_minus_13m": (
                float(large_block["h1_h16_mean_rollout_ce"])
                - float(small_block["h1_h16_mean_rollout_ce"])
            ),
        },
        {
            "metric": "h1_h16_mean_accuracy",
            "value_13m": small_block["h1_h16_mean_accuracy"],
            "value_121m": large_block["h1_h16_mean_accuracy"],
            "value_121m_minus_13m": (
                float(large_block["h1_h16_mean_accuracy"])
                - float(small_block["h1_h16_mean_accuracy"])
            ),
        },
    ]
    for metric in (
        "aligned_accuracy_first_128",
        "corpus_byte_entropy_bits",
        "mean_sample_unique_bytes",
        "mean_repeated_4gram_fraction",
        "mean_longest_identical_byte_run",
        "strict_utf8_sample_fraction",
        "mean_top1_top2_logit_margin",
        "mean_predictive_entropy_nats",
    ):
        comparison_rows.append({
            "metric": metric,
            "value_13m": float(old_small[metric]),
            "value_121m": float(large_generation[metric]),
            "value_121m_minus_13m": (
                float(large_generation[metric]) - float(old_small[metric])
            ),
        })

    old_samples = {
        int(row["sample"]): row
        for row in read_rows(SMALL_GENERATION_RECORD / "samples.tsv")
        if row["mode"] == "h4_greedy_block4"
    }
    sample_rows: list[dict[str, object]] = []
    for sample in range(REPORT_SAMPLES):
        common = {
            "sample": sample,
            "validation_start": int(starts[sample]),
            "prompt_suffix_escaped": escaped(
                prompts[sample, -96:].cpu().tolist()
            ),
        }
        sample_rows.extend((
            {
                **common,
                "model": "gold",
                "continuation_escaped": escaped(gold[sample].cpu().tolist()),
                "continuation_hex": bytes(gold[sample].cpu().tolist()).hex(),
            },
            {
                **common,
                "model": "feedback_h4_13m_epoch2",
                "continuation_escaped": old_samples[sample][
                    "continuation_escaped"
                ],
                "continuation_hex": old_samples[sample]["continuation_hex"],
            },
            {
                **common,
                "model": "feedback_h4_121m_epoch2",
                "continuation_escaped": escaped(
                    generated[sample].cpu().tolist()
                ),
                "continuation_hex": bytes(
                    generated[sample].cpu().tolist()
                ).hex(),
            },
        ))

    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "horizon_metrics.tsv", small_horizons + large_horizons)
    write_rows(RECORD / "block_metrics.tsv", block_rows)
    write_rows(RECORD / "generation_metrics.tsv", generation_rows)
    write_rows(RECORD / "comparisons.tsv", comparison_rows)
    write_rows(RECORD / "samples.tsv", sample_rows)
    elapsed = time.monotonic() - started
    with (RECORD / "run.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows((
            ("key", "value"),
            ("step", STEP),
            ("seed", base.SEED),
            ("split", "wikitext103_bytes_validation"),
            ("samples", SAMPLES),
            ("anchors_per_sample", 16),
            ("rollout_labels_per_model", SAMPLES * 16 * HORIZONS),
            ("natural_chunk_size", BLOCK_SIZE),
            ("gold_reset_interval", HORIZONS),
            ("generated_bytes", GENERATE),
            ("policy", "greedy_argmax"),
            ("rollout_microbatch", ROLLOUT_MICROBATCH),
            ("registered_eval_microbatch", REGISTERED_EVAL_MICROBATCH),
            ("generation_microbatch", GENERATION_MICROBATCH),
            ("large_parameter_count", LARGE_PARAMETERS),
            ("large_checkpoint", CHECKPOINT),
            ("large_checkpoint_sha256", sha256(CHECKPOINT)),
            ("validation_starts_sha256", sha256(small_starts_path)),
            (
                "registered_h1_h4_reproduction_max_error",
                registered_reproduction_error,
            ),
            (
                "prefix_reencode_vs_full_context_h1_h4_max_error",
                prefix_reencode_vs_full_context_error,
            ),
            ("checkpoint_objective", config["objective"]),
            ("elapsed_seconds", elapsed),
            ("peak_vram_bytes", torch.cuda.max_memory_allocated()),
            ("small_block_source", SMALL_BLOCK_RECORD),
            ("small_generation_source", SMALL_GENERATION_RECORD),
        ))

    del model
    torch.cuda.empty_cache()
    print(small_block, flush=True)
    print(large_block, flush=True)
    for row in comparison_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
