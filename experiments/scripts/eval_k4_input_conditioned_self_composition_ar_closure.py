"""Audit u[j+1]=K(u[j])u[j] against canonical greedy AR generation."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.evaluation import (
    input_conditioned_spectral_composition_rollouts,
)
from rotlm.training.ha_skew_window import relative_mse_rows
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length
from train_k4_input_conditioned_kpower_h1_ce_only_tied_ar_closure_13m import (
    CONTEXT,
    EVAL_ANCHOR_STRIDE,
    EVAL_WINDOW_LENGTH,
    HORIZONS,
    SEED,
    calibrate_interface,
    make_model,
    memmap,
)


SOURCE_ID = (
    "EXP-20260801-k4-input-conditioned-kpower-h1-ce-only-"
    "tied-ar-closure-13m"
)
EXPERIMENT_ID = (
    "EXP-20260801-k4-input-conditioned-self-composition-"
    "ar-audit-13m"
)
SOURCE_RECORD = Path("experiments/records") / SOURCE_ID
SOURCE_OUTPUT = Path("outputs/experiments") / SOURCE_ID
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
CHECKPOINTS = {
    500: SOURCE_OUTPUT / "step0500.pt",
    1000: SOURCE_OUTPUT / "step1000.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_starts(path: Path) -> torch.Tensor:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("validation start file is empty")
    return torch.tensor(
        [int(row["token_start"]) for row in rows],
        dtype=torch.long,
    )


@torch.inference_mode()
def evaluate(
    model,
    validation,
    starts: torch.Tensor,
    microbatch: int,
) -> dict[str, float]:
    overall_cosine = torch.zeros(HORIZONS, dtype=torch.float64)
    overall_mse = torch.zeros(HORIZONS, dtype=torch.float64)
    closure_cosine = torch.zeros(HORIZONS, dtype=torch.float64)
    closure_mse = torch.zeros(HORIZONS, dtype=torch.float64)
    path_cosine = torch.zeros(HORIZONS, dtype=torch.float64)
    path_mse = torch.zeros(HORIZONS, dtype=torch.float64)
    token_agreement = torch.zeros(HORIZONS, dtype=torch.float64)
    forward_kl = torch.zeros(HORIZONS, dtype=torch.float64)
    h1_nll_sum = 0.0
    h1_correct = 0
    exact_blocks = 0
    boundaries = 0
    h1_composition_proposal_error = 0.0
    h1_logit_error = 0.0

    for begin in range(0, len(starts), microbatch):
        window = windows_of_length(
            validation,
            starts[begin : begin + microbatch],
            EVAL_WINDOW_LENGTH,
        )
        output = input_conditioned_spectral_composition_rollouts(
            model,
            window,
            prefix_length=CONTEXT,
            horizons=HORIZONS,
            anchor_stride=EVAL_ANCHOR_STRIDE,
        )
        composition = output.composition_states.float()
        canonical = output.boundary_ar_states.float()
        proposals = output.boundary_ar_proposal_states.float()
        count = window.shape[0] * output.anchor_indices.numel()
        boundaries += count

        overall_cosine += F.cosine_similarity(
            composition,
            canonical,
            dim=-1,
        ).double().sum(dim=(0, 1)).cpu()
        overall_mse += relative_mse_rows(
            composition,
            canonical,
        ).double().sum(dim=(0, 1)).cpu()
        closure_cosine += F.cosine_similarity(
            proposals,
            canonical,
            dim=-1,
        ).double().sum(dim=(0, 1)).cpu()
        closure_mse += relative_mse_rows(
            proposals,
            canonical,
        ).double().sum(dim=(0, 1)).cpu()
        path_cosine += F.cosine_similarity(
            composition,
            proposals,
            dim=-1,
        ).double().sum(dim=(0, 1)).cpu()
        path_mse += relative_mse_rows(
            composition,
            proposals,
        ).double().sum(dim=(0, 1)).cpu()

        matches = output.composition_tokens.eq(output.boundary_ar_tokens)
        token_agreement += matches.double().sum(dim=(0, 1)).cpu()
        exact_blocks += int(matches.all(dim=-1).sum())
        for horizon in range(HORIZONS):
            ar_log_probs = F.log_softmax(
                output.boundary_ar_logits[:, :, horizon],
                dim=-1,
            )
            composition_log_probs = F.log_softmax(
                output.composition_logits[:, :, horizon],
                dim=-1,
            )
            forward_kl[horizon] += float(
                (
                    ar_log_probs.exp()
                    * (ar_log_probs - composition_log_probs)
                ).sum(dim=-1).sum()
            )

        targets = output.targets[:, :, 0]
        h1_nll_sum += float(
            F.cross_entropy(
                output.composition_logits[:, :, 0].flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )
        )
        h1_correct += int(output.composition_tokens[:, :, 0].eq(targets).sum())
        h1_composition_proposal_error = max(
            h1_composition_proposal_error,
            float(
                (
                    output.composition_states[:, :, 0]
                    - output.boundary_ar_proposal_states[:, :, 0]
                ).abs().max()
            ),
        )
        h1_logit_error = max(
            h1_logit_error,
            float(
                (
                    output.composition_logits[:, :, 0]
                    - output.boundary_ar_logits[:, :, 0]
                ).abs().max()
            ),
        )

    if boundaries < 1:
        raise RuntimeError("audit evaluated no boundaries")
    aggregates = {
        "overall_cosine": overall_cosine / boundaries,
        "overall_relative_mse": overall_mse / boundaries,
        "one_step_closure_cosine": closure_cosine / boundaries,
        "one_step_closure_relative_mse": closure_mse / boundaries,
        "path_cosine": path_cosine / boundaries,
        "path_relative_mse": path_mse / boundaries,
        "token_agreement": token_agreement / boundaries,
        "ar_to_composition_kl": forward_kl / boundaries,
    }
    metrics = {
        "evaluated_boundaries": float(boundaries),
        "h1_validation_nll": h1_nll_sum / boundaries,
        "h1_validation_accuracy": h1_correct / boundaries,
        "exact_block_token_agreement": exact_blocks / boundaries,
        "h1_composition_proposal_max_error": h1_composition_proposal_error,
        "h1_ar_composition_max_logit_error": h1_logit_error,
        "mean_overall_cosine": float(aggregates["overall_cosine"].mean()),
        "mean_overall_relative_mse": float(
            aggregates["overall_relative_mse"].mean()
        ),
        "h2_h4_token_agreement": float(
            aggregates["token_agreement"][1:].mean()
        ),
    }
    for horizon in range(HORIZONS):
        for name, values in aggregates.items():
            metrics[f"h{horizon + 1}_{name}"] = float(values[horizon])
    return metrics


METRIC_NAMES = (
    "evaluated_boundaries",
    "h1_validation_nll",
    "h1_validation_accuracy",
    "exact_block_token_agreement",
    "h1_composition_proposal_max_error",
    "h1_ar_composition_max_logit_error",
    "mean_overall_cosine",
    "mean_overall_relative_mse",
    "h2_h4_token_agreement",
    *tuple(
        f"h{horizon}_{metric}"
        for horizon in range(1, HORIZONS + 1)
        for metric in (
            "overall_cosine",
            "overall_relative_mse",
            "one_step_closure_cosine",
            "one_step_closure_relative_mse",
            "path_cosine",
            "path_relative_mse",
            "token_agreement",
            "ar_to_composition_kl",
        )
    ),
)


def initialized_model(training):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = make_model().cuda().eval()
    calibrate_interface(model, training)
    return model


def checkpoint_model(path: Path):
    model = make_model().cuda().eval()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if int(payload["step"]) not in CHECKPOINTS:
        raise RuntimeError(f"unexpected checkpoint step {payload['step']}")
    del payload
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--eval-microbatch", type=int, default=4)
    args = parser.parse_args()
    if args.eval_microbatch < 1:
        parser.error("eval microbatch must be positive")
    if not torch.cuda.is_available():
        parser.error("this audit requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    training, validation = memmap("train"), memmap("validation")
    starts_path = SOURCE_RECORD / "validation_starts.tsv"
    starts = read_starts(starts_path)
    args.record_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hashes = {
        step: sha256(path) for step, path in CHECKPOINTS.items()
    }
    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("experiment_id", EXPERIMENT_ID))
        writer.writerow(("source_experiment_id", SOURCE_ID))
        writer.writerow(("validation_starts", starts_path))
        writer.writerow(("validation_starts_sha256", sha256(starts_path)))
        writer.writerow(("validation_examples", len(starts)))
        writer.writerow(("recurrence", "u_next=K(u_current)@u_current"))
        writer.writerow(("horizons", HORIZONS))
        writer.writerow(("anchor_stride", EVAL_ANCHOR_STRIDE))
        for step, path in CHECKPOINTS.items():
            writer.writerow((f"step{step}_checkpoint", path))
            writer.writerow((f"step{step}_checkpoint_sha256", checkpoint_hashes[step]))

    rows = []
    for step in (0, 500, 1000):
        model = (
            initialized_model(training)
            if step == 0
            else checkpoint_model(CHECKPOINTS[step])
        )
        metrics = evaluate(
            model,
            validation,
            starts,
            args.eval_microbatch,
        )
        row = {"step": step, **metrics}
        rows.append(row)
        print(
            f"step={step:4d} h1_nll={metrics['h1_validation_nll']:.4f} "
            f"overall_cos={metrics['mean_overall_cosine']:.4f} "
            f"overall_mse={metrics['mean_overall_relative_mse']:.4f} "
            f"tok_h2-h4={metrics['h2_h4_token_agreement']:.3f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    with (args.record_dir / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", *METRIC_NAMES),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    initial, final = rows[0], rows[-1]
    cosine_improved = sum(
        final[f"h{h}_overall_cosine"] > initial[f"h{h}_overall_cosine"]
        for h in range(1, HORIZONS + 1)
    )
    mse_improved = sum(
        final[f"h{h}_overall_relative_mse"]
        < initial[f"h{h}_overall_relative_mse"]
        for h in range(1, HORIZONS + 1)
    )
    passed = cosine_improved >= 3 and mse_improved >= 3
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("criterion_passed", str(passed).lower()))
        writer.writerow(("cosine_horizons_improved", cosine_improved))
        writer.writerow(("relative_mse_horizons_improved", mse_improved))

    lines = [
        "# Interpretation",
        "",
        (
            "The corrected state-dependent self-composition criterion "
            + ("passed." if passed else "did not pass.")
        ),
        "",
        "The evaluated recurrence is `u_(j+1) = K(u_j) u_j`; no matrix is "
        "frozen across horizons. The AR state is obtained only from the "
        "model's own greedy tokens.",
        "",
        "| Metric | Step 0 | Step 1000 |",
        "|---|---:|---:|",
        f"| Mean overall cosine | {initial['mean_overall_cosine']:.6f} | {final['mean_overall_cosine']:.6f} |",
        f"| Mean overall relative MSE | {initial['mean_overall_relative_mse']:.6f} | {final['mean_overall_relative_mse']:.6f} |",
        f"| H2--H4 token agreement | {initial['h2_h4_token_agreement']:.6f} | {final['h2_h4_token_agreement']:.6f} |",
        f"| H1 validation NLL | {initial['h1_validation_nll']:.6f} | {final['h1_validation_nll']:.6f} |",
        "",
    ]
    (args.record_dir / "interpretation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
