"""Inspect epoch-2 H1/H4/scan generation at their natural chunk sizes."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_byte256_unitary_fused_scan_step10071_generation_quality import (
    aggregate_mode,
    escaped,
)
from train_byte256_unitary_time_varying_scan_h16_1000 import (
    configure_scan_backend,
)
import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
ROOT = Path("experiments/records")
OUTPUT = Path("outputs/experiments")
H1_ID = "EXP-20260805-byte256-unitary-feedback-h1-stride1-h16-monitor-2epoch-13m"
H4_ID = "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-10epoch-13m"
SCAN_ID = "EXP-20260803-byte256-unitary-fused-triton-scan-h16-10epoch-13m-rtx5090"
RECORD = ROOT / "EXP-20260805-byte256-unitary-epoch2-natural-chunk-generation-quality"
SAMPLES = 64
REPORT_SAMPLES = 12
GENERATE = 128
STEP = 6_714
OBJECTIVES = {
    H1_ID: "one_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_attached_stride1_h16_monitor",
    H4_ID: "four_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_attached_stride4_h16_monitor",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_model(experiment_id: str, *, scan: bool = False):
    path = OUTPUT / experiment_id / "step6714.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != STEP:
        raise RuntimeError(f"wrong checkpoint step: {path}")
    if not scan:
        config = checkpoint.get("config", {})
        if (
            config.get("experiment_id") != experiment_id
            or config.get("objective") != OBJECTIVES[experiment_id]
            or config.get("schedule_steps") != 33_570
        ):
            raise RuntimeError(f"wrong feedback checkpoint config: {path}")
    model = base.make_model().cuda().eval()
    if scan:
        configure_scan_backend(model, "fused-triton-rotating-frame")
    model.load_state_dict(checkpoint["model"])
    return model, path


@torch.inference_mode()
def next_logits(model, contexts: torch.Tensor, block: int, *, scan: bool):
    contexts = contexts[:, -base.CONTEXT :]
    positions = torch.arange(contexts.shape[1], device=contexts.device)
    encoded, expanded_positions = model.encode(contexts, positions)
    roots = encoded[:, -1:, :]
    transition = model.complex_self_prediction
    if scan:
        states, _ = transition.rollout_time_varying_scan_read_states(
            roots, block
        )
    else:
        states, _ = transition.rollout_read_states(roots, block)
    anchor = torch.tensor(
        [contexts.shape[1] - 1], device=contexts.device, dtype=torch.long
    )
    decoded = model.exact_inverse_selected_decode_tape_states(
        encoded, states, expanded_positions, anchor
    )[:, 0]
    return model.token_logits(decoded).float()


@torch.inference_mode()
def generate(model, prompts, block: int, *, scan: bool):
    rolling = prompts.clone()
    margins = []
    entropies = []
    for _ in range(GENERATE // block):
        logits = next_logits(model, rolling, block, scan=scan)
        top = logits.topk(2, dim=-1).values
        margins.append((top[..., 0] - top[..., 1]).reshape(-1))
        entropies.append(
            (-(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1)).reshape(-1)
        )
        rolling = torch.cat((rolling, logits.argmax(dim=-1)), dim=1)
    return (
        rolling[:, -GENERATE:],
        float(torch.cat(margins).mean()),
        float(torch.cat(entropies).mean()),
    )


def write_rows(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    starts_path = ROOT / H1_ID / "validation_starts.tsv"
    starts = base.load_fixed_starts(starts_path, SAMPLES)
    validation = base.memmap("validation")
    prompts = base.windows_of_length(validation, starts, base.CONTEXT)
    gold = base.windows_of_length(
        validation, starts + base.CONTEXT, GENERATE
    )
    outputs = {"gold": gold}
    confidence = {"gold": (float("nan"), float("nan"))}
    checkpoints = {}
    for mode, experiment_id, block, scan in (
        ("h1_greedy_block1", H1_ID, 1, False),
        ("h4_greedy_block4", H4_ID, 4, False),
        ("scan_greedy_block16", SCAN_ID, 16, True),
    ):
        model, checkpoint_path = load_model(experiment_id, scan=scan)
        outputs[mode], margin, entropy = generate(
            model, prompts, block, scan=scan
        )
        confidence[mode] = (margin, entropy)
        checkpoints[mode] = checkpoint_path
        del model
        torch.cuda.empty_cache()

    metric_rows = []
    for mode, values in outputs.items():
        row = aggregate_mode(mode, values, gold)
        row["mean_top1_top2_logit_margin"] = confidence[mode][0]
        row["mean_predictive_entropy_nats"] = confidence[mode][1]
        metric_rows.append(row)
    sample_rows = []
    for sample in range(REPORT_SAMPLES):
        for mode, values in outputs.items():
            sample_rows.append({
                "sample": sample,
                "validation_start": int(starts[sample]),
                "mode": mode,
                "prompt_suffix_escaped": escaped(prompts[sample, -96:].cpu().tolist()),
                "continuation_escaped": escaped(values[sample].cpu().tolist()),
                "continuation_hex": bytes(values[sample].cpu().tolist()).hex(),
            })
    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "metrics.tsv", metric_rows)
    write_rows(RECORD / "samples.tsv", sample_rows)
    with (RECORD / "run.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows([
            ("key", "value"), ("step", STEP), ("samples", SAMPLES),
            ("generated_bytes", GENERATE), ("policy", "greedy_argmax"),
            ("validation_starts_sha256", sha256(starts_path)),
            *[(f"{mode}_checkpoint_sha256", sha256(path)) for mode, path in checkpoints.items()],
        ])
    for row in metric_rows:
        print({key: row[key] for key in (
            "mode", "aligned_accuracy_first_128", "corpus_byte_entropy_bits",
            "mean_sample_unique_bytes", "mean_repeated_4gram_fraction",
            "mean_longest_identical_byte_run", "strict_utf8_sample_fraction",
            "mean_top1_top2_logit_margin", "mean_predictive_entropy_nats",
        )}, flush=True)


if __name__ == "__main__":
    main()
