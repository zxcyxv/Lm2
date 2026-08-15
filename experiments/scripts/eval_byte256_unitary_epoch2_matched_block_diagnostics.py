"""Margin-neutral diagnostics on the matched epoch-2 sixteen-byte blocks."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

import eval_byte256_unitary_epoch2_natural_chunk_generation_quality as gen
from eval_byte256_unitary_fused_scan_step10071_generation_quality import escaped
from rotlm.evaluation import greedy_chunk_reinput_rollouts


base = gen.base
ROOT = Path("experiments/records")
RECORD = ROOT / "EXP-20260805-byte256-unitary-epoch2-matched-block-diagnostics"
TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def write_rows(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metrics(mode, block, rollout):
    logits, targets, actions = rollout.logits, rollout.targets, rollout.actions
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ranks = logits.gt(target_logits.unsqueeze(-1)).sum(-1) + 1
    correct = actions.eq(targets)
    other = logits.clone()
    other.scatter_(-1, targets.unsqueeze(-1), -torch.inf)
    top2 = logits.topk(2, dim=-1).values
    probabilities = logits.softmax(-1)
    temperature_rows = [
        {
            "mode": mode,
            "temperature": temperature,
            "rollout_ce": float(F.cross_entropy(flat_logits / temperature, flat_targets)),
        }
        for temperature in TEMPERATURES
    ]
    best = min(temperature_rows, key=lambda row: row["rollout_ce"])
    row = {
        "mode": mode,
        "chunk_size": block,
        "labels": targets.numel(),
        "rollout_ce_t1": next(r["rollout_ce"] for r in temperature_rows if r["temperature"] == 1.0),
        "best_grid_temperature": best["temperature"],
        "best_grid_rollout_ce": best["rollout_ce"],
        "top1_accuracy": float(correct.float().mean()),
        "top5_accuracy": float(ranks.le(5).float().mean()),
        "top10_accuracy": float(ranks.le(10).float().mean()),
        "mean_target_rank": float(ranks.float().mean()),
        "mean_reciprocal_rank": float(ranks.float().reciprocal().mean()),
        "mean_target_vs_best_other_margin": float((target_logits - other.max(-1).values).mean()),
        "mean_selected_top1_top2_margin": float((top2[..., 0] - top2[..., 1]).mean()),
        "mean_predictive_entropy_nats": float((-(probabilities * logits.log_softmax(-1)).sum(-1)).mean()),
        "predicted_space_fraction": float(actions.eq(32).float().mean()),
        "gold_space_fraction": float(targets.eq(32).float().mean()),
        "exact_16byte_block_fraction": float(correct.all(-1).float().mean()),
        "mean_exact_prefix_bytes": float(correct.long().cumprod(-1).sum(-1).float().mean()),
        "predicted_unique_bytes": actions.unique().numel(),
    }
    return row, temperature_rows


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    starts_paths = (
        ROOT / gen.H1_ID / "validation_starts.tsv",
        ROOT / gen.H4_ID / "validation_starts.tsv",
        ROOT / "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-normalized-residual-h16-ce-only-attached-stride16-1000step-13m/validation_starts.tsv",
    )
    if len({path.read_bytes() for path in starts_paths}) != 1:
        raise RuntimeError("validation starts differ")
    starts = base.load_fixed_starts(starts_paths[0], 64)
    windows = base.windows_of_length(base.memmap("validation"), starts, 272)
    specs = (
        ("h1_greedy_block1", gen.H1_ID, 1, False, "feedback"),
        ("h4_greedy_block4", gen.H4_ID, 4, False, "feedback"),
        ("scan_greedy_block16", gen.SCAN_ID, 16, True, "time-varying-scan"),
    )
    metric_rows, temperature_rows, sample_rows = [], [], []
    for mode, experiment_id, block, scan, central in specs:
        model, _ = gen.load_model(experiment_id, scan=scan)
        rollout = greedy_chunk_reinput_rollouts(
            model, windows, context_length=256, horizons=16,
            block_size=block, anchor_stride=16, microbatch=8,
            central_rollout=central,
        )
        row, temperatures = metrics(mode, block, rollout)
        metric_rows.append(row)
        temperature_rows.extend(temperatures)
        anchor_slot = 15
        for sample in range(12):
            sample_rows.append({
                "sample": sample,
                "validation_start": int(starts[sample]),
                "anchor": int(rollout.anchor_indices[anchor_slot]),
                "mode": mode,
                "prompt_suffix_escaped": escaped(windows[sample, 145:241].cpu().tolist()),
                "gold_16_escaped": escaped(rollout.targets[sample, anchor_slot].tolist()),
                "generated_16_escaped": escaped(rollout.actions[sample, anchor_slot].tolist()),
            })
        del model, rollout
        torch.cuda.empty_cache()
    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "metrics.tsv", metric_rows)
    write_rows(RECORD / "temperature_sweep.tsv", temperature_rows)
    write_rows(RECORD / "samples.tsv", sample_rows)
    for row in metric_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
