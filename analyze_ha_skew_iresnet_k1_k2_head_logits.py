"""Audit causal K h_A / K^2 h_A decoding at the head interface.

The first future slot is decoded twice:

1. ``[encoded_prefix, K h_A]``
2. ``[encoded_prefix, K h_A, K^2 h_A]``

A correct causal decoder must give the same first-slot hidden state and logits
in both tapes.  The joint tape's first and second slots are then compared at
the raw decoder hidden, RMSNorm head-input, and vocabulary-logit levels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from tokenizers import Tokenizer

from train_k1_decoder_inverse_ablation_13m import CONTEXT, SEED, memmap
from train_k1_ha_skew_iresnet_spectral_learned_noise_13m import (
    EXPERIMENT_ID,
    make_model,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import (
    HORIZONS,
    windows_of_length,
)


DEFAULT_CHECKPOINT = (
    Path("outputs/experiments") / EXPERIMENT_ID / "step1000.pt"
)
DEFAULT_OUTPUT = (
    Path("experiments/records")
    / EXPERIMENT_ID
    / "step1000_k1_k2_head_logits"
)


def relative_mse_rows(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-12)
    return numerator / denominator


def load_model(path: Path) -> tuple[torch.nn.Module, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model()
    model.load_state_dict(payload["model"], strict=True)
    return model.cuda().eval(), int(payload["step"])


def append_metric(
    storage: dict[tuple[str, str], list[torch.Tensor]],
    section: str,
    metric: str,
    value: torch.Tensor,
) -> None:
    storage.setdefault((section, metric), []).append(
        value.detach().float().reshape(-1).cpu()
    )


def record_slot(
    storage: dict[tuple[str, str], list[torch.Tensor]],
    section: str,
    hidden: torch.Tensor,
    head_input: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    target_embedding: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    log_probabilities = F.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    target_logits = logits.gather(-1, target[:, None]).squeeze(-1)
    target_log_probabilities = log_probabilities.gather(
        -1, target[:, None]
    ).squeeze(-1)
    top2_values, top2_ids = logits.topk(2, dim=-1)
    prediction = top2_ids[:, 0]
    best_wrong = torch.where(
        prediction.eq(target), top2_values[:, 1], top2_values[:, 0]
    )
    target_rank = logits.gt(target_logits[:, None]).sum(dim=-1) + 1

    for metric, value in (
        ("accuracy", prediction.eq(target)),
        ("nll", -target_log_probabilities),
        ("target_probability", target_log_probabilities.exp()),
        ("target_rank", target_rank),
        ("target_logit", target_logits),
        ("best_wrong_logit", best_wrong),
        ("target_margin", target_logits - best_wrong),
        ("top1_probability", probabilities.max(dim=-1).values),
        ("entropy", -(probabilities * log_probabilities).sum(dim=-1)),
        ("logits_mean", logits.mean(dim=-1)),
        ("logits_std", logits.std(dim=-1)),
        ("logits_l2_norm", logits.norm(dim=-1)),
        ("raw_hidden_mean", hidden.float().mean(dim=-1)),
        ("raw_hidden_std", hidden.float().std(dim=-1)),
        ("raw_hidden_l2_norm", hidden.float().norm(dim=-1)),
        ("head_input_mean", head_input.float().mean(dim=-1)),
        ("head_input_std", head_input.float().std(dim=-1)),
        ("head_input_l2_norm", head_input.float().norm(dim=-1)),
        (
            "hidden_to_target_embedding_cosine",
            F.cosine_similarity(
                hidden.float(), target_embedding.float(), dim=-1
            ),
        ),
        (
            "hidden_to_target_embedding_relative_mse",
            relative_mse_rows(hidden, target_embedding),
        ),
    ):
        append_metric(storage, section, metric, value)
    return prediction


def record_pair(
    storage: dict[tuple[str, str], list[torch.Tensor]],
    section: str,
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    append_metric(
        storage,
        section,
        "max_abs",
        (first.float() - second.float()).abs().flatten(1).max(dim=-1).values,
    )
    append_metric(
        storage,
        section,
        "relative_mse",
        relative_mse_rows(first, second),
    )
    append_metric(
        storage,
        section,
        "cosine",
        F.cosine_similarity(first.float(), second.float(), dim=-1),
    )


def record_logit_distribution_pair(
    storage: dict[tuple[str, str], list[torch.Tensor]],
    section: str,
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    first = first.float()
    second = second.float()
    centered_first = first - first.mean(dim=-1, keepdim=True)
    centered_second = second - second.mean(dim=-1, keepdim=True)
    record_pair(storage, section + "_centered", centered_first, centered_second)

    log_probability_first = F.log_softmax(first, dim=-1)
    log_probability_second = F.log_softmax(second, dim=-1)
    probability_first = log_probability_first.exp()
    probability_second = log_probability_second.exp()
    mixture = 0.5 * (probability_first + probability_second)
    log_mixture = mixture.clamp_min(1e-30).log()
    js = 0.5 * (
        (
            probability_first
            * (log_probability_first - log_mixture)
        ).sum(dim=-1)
        + (
            probability_second
            * (log_probability_second - log_mixture)
        ).sum(dim=-1)
    )
    append_metric(storage, section, "jensen_shannon", js)
    append_metric(
        storage,
        section,
        "probability_cosine",
        F.cosine_similarity(
            probability_first, probability_second, dim=-1
        ),
    )
    top5_first = first.topk(5, dim=-1).indices
    top5_second = second.topk(5, dim=-1).indices
    overlap = (
        top5_first.unsqueeze(-1)
        .eq(top5_second.unsqueeze(-2))
        .any(dim=-1)
        .sum(dim=-1)
        .float()
        / 5.0
    )
    append_metric(storage, section, "top5_overlap_fraction", overlap)


def summarize(
    storage: dict[tuple[str, str], list[torch.Tensor]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (section, metric), chunks in sorted(storage.items()):
        values = torch.cat(chunks).double()
        rows.append(
            {
                "section": section,
                "metric": metric,
                "count": values.numel(),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p10": float(torch.quantile(values, 0.10)),
                "p90": float(torch.quantile(values, 0.90)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return rows


def row_value(
    rows: list[dict[str, object]], section: str, metric: str, statistic: str
) -> float:
    for row in rows:
        if row["section"] == section and row["metric"] == metric:
            return float(row[statistic])
    raise KeyError((section, metric, statistic))


def token_label(tokenizer: Tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id]).replace("\n", "\\n")
    return f"{text!r}({token_id})"


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    validation,
    starts: torch.Tensor,
    tokenizer: Tokenizer,
    *,
    micro: int,
    sample_count: int,
) -> tuple[list[dict[str, object]], list[str]]:
    storage: dict[tuple[str, str], list[torch.Tensor]] = {}
    examples = [
        "# Step-1000 causal K / K^2 head and logit examples",
        "",
    ]
    examples_written = 0
    for begin in range(0, len(starts), micro):
        window = windows_of_length(
            validation, starts[begin : begin + micro], CONTEXT + 2
        )
        prefix, targets = window[:, :CONTEXT], window[:, CONTEXT:]
        batch = prefix.shape[0]
        positions = torch.arange(CONTEXT, device=prefix.device)
        encoded, _ = model.encode(prefix, positions)
        h_a = encoded[:, -1]
        state1 = model.operator(h_a)
        state2 = model.operator(state1)

        positions1 = torch.arange(
            CONTEXT + 1, device=prefix.device
        ).unsqueeze(0).expand(batch, -1)
        positions2 = torch.arange(
            CONTEXT + 2, device=prefix.device
        ).unsqueeze(0).expand(batch, -1)
        tape1 = torch.cat((encoded, state1[:, None]), dim=1)
        tape2 = torch.cat(
            (encoded, state1[:, None], state2[:, None]), dim=1
        )
        decoded1 = model.encoder.decode_hidden(tape1, positions1)
        decoded2 = model.encoder.decode_hidden(tape2, positions2)
        hidden1_short = decoded1[:, -1]
        hidden1_joint = decoded2[:, -2]
        hidden2_joint = decoded2[:, -1]
        head1_short = model.encoder.head_norm(hidden1_short)
        head1_joint = model.encoder.head_norm(hidden1_joint)
        head2_joint = model.encoder.head_norm(hidden2_joint)
        logits1_short = head1_short @ model.embedding_weight.T
        logits1_joint = head1_joint @ model.embedding_weight.T
        logits2_joint = head2_joint @ model.embedding_weight.T

        target_embedding1 = model.encoder.embed(targets[:, 0])
        target_embedding2 = model.encoder.embed(targets[:, 1])
        prediction1_short = record_slot(
            storage,
            "h1_short_tape",
            hidden1_short,
            head1_short,
            logits1_short,
            targets[:, 0],
            target_embedding1,
        )
        prediction1_joint = record_slot(
            storage,
            "h1_joint_tape",
            hidden1_joint,
            head1_joint,
            logits1_joint,
            targets[:, 0],
            target_embedding1,
        )
        prediction2_joint = record_slot(
            storage,
            "h2_joint_tape",
            hidden2_joint,
            head2_joint,
            logits2_joint,
            targets[:, 1],
            target_embedding2,
        )

        record_pair(
            storage,
            "causal_invariance_raw_hidden",
            hidden1_short,
            hidden1_joint,
        )
        record_pair(
            storage,
            "causal_invariance_head_input",
            head1_short,
            head1_joint,
        )
        record_pair(
            storage,
            "causal_invariance_logits",
            logits1_short,
            logits1_joint,
        )
        record_logit_distribution_pair(
            storage,
            "causal_invariance_logits",
            logits1_short,
            logits1_joint,
        )
        append_metric(
            storage,
            "causal_invariance",
            "top1_agreement",
            prediction1_short.eq(prediction1_joint),
        )

        record_pair(
            storage, "h1_vs_h2_raw_hidden", hidden1_joint, hidden2_joint
        )
        record_pair(
            storage, "h1_vs_h2_head_input", head1_joint, head2_joint
        )
        record_pair(
            storage, "h1_vs_h2_logits", logits1_joint, logits2_joint
        )
        record_logit_distribution_pair(
            storage, "h1_vs_h2_logits", logits1_joint, logits2_joint
        )
        append_metric(
            storage,
            "h1_vs_h2",
            "head_difference_l2",
            (head2_joint.float() - head1_joint.float()).norm(dim=-1),
        )
        append_metric(
            storage,
            "h1_vs_h2",
            "logit_difference_l2",
            (logits2_joint.float() - logits1_joint.float()).norm(dim=-1),
        )
        append_metric(
            storage,
            "h1_vs_h2",
            "empirical_head_projection_gain",
            (
                (logits2_joint.float() - logits1_joint.float()).norm(dim=-1)
                / (head2_joint.float() - head1_joint.float())
                .norm(dim=-1)
                .clamp_min(1e-12)
            ),
        )
        append_metric(
            storage,
            "h1_vs_h2",
            "top1_same_token",
            prediction1_joint.eq(prediction2_joint),
        )

        full_tokens = torch.cat((prefix, targets), dim=1)
        full_encoded, full_positions = model.encode(full_tokens, positions2)
        gold_state1 = full_encoded[:, -2]
        gold_state2 = full_encoded[:, -1]
        record_pair(storage, "state1_vs_gold_hB", state1, gold_state1)
        record_pair(storage, "state2_vs_gold_hC", state2, gold_state2)
        gold_decoded = model.encoder.decode_hidden(
            full_encoded, full_positions
        )
        gold_hidden1, gold_hidden2 = gold_decoded[:, -2], gold_decoded[:, -1]
        gold_head1 = model.encoder.head_norm(gold_hidden1)
        gold_head2 = model.encoder.head_norm(gold_hidden2)
        gold_logits1 = gold_head1 @ model.embedding_weight.T
        gold_logits2 = gold_head2 @ model.embedding_weight.T
        record_slot(
            storage,
            "gold_hB_control",
            gold_hidden1,
            gold_head1,
            gold_logits1,
            targets[:, 0],
            target_embedding1,
        )
        record_slot(
            storage,
            "gold_hC_control",
            gold_hidden2,
            gold_head2,
            gold_logits2,
            targets[:, 1],
            target_embedding2,
        )

        for local in range(batch):
            if examples_written >= sample_count:
                break
            global_index = begin + local
            top1 = torch.topk(
                F.softmax(logits1_joint[local].float(), dim=-1), 5
            )
            top2 = torch.topk(
                F.softmax(logits2_joint[local].float(), dim=-1), 5
            )
            top1_text = ", ".join(
                f"{token_label(tokenizer, int(token))}: {float(probability):.4f}"
                for probability, token in zip(top1.values, top1.indices)
            )
            top2_text = ", ".join(
                f"{token_label(tokenizer, int(token))}: {float(probability):.4f}"
                for probability, token in zip(top2.values, top2.indices)
            )
            examples.extend(
                [
                    f"## Example {global_index + 1}",
                    "",
                    "**Prefix tail:** "
                    + tokenizer.decode(prefix[local, -32:].tolist()),
                    "",
                    "**Gold:** "
                    + token_label(tokenizer, int(targets[local, 0]))
                    + " → "
                    + token_label(tokenizer, int(targets[local, 1])),
                    "",
                    "**Predicted:** "
                    + token_label(tokenizer, int(prediction1_joint[local]))
                    + " → "
                    + token_label(tokenizer, int(prediction2_joint[local])),
                    "",
                    f"**h1 top-5:** {top1_text}",
                    "",
                    f"**h2 top-5:** {top2_text}",
                    "",
                ]
            )
            examples_written += 1

    return summarize(storage), examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--expected-step", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this analysis requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    validation = memmap("validation")
    starts = torch.randint(
        0,
        # Match the inline h1..h5 evaluator's exact sampling range even
        # though this audit reads only its first two target tokens.
        len(validation) - CONTEXT - HORIZONS - 1,
        (args.examples,),
        generator=torch.Generator().manual_seed(SEED + 7777),
    )
    tokenizer = Tokenizer.from_file("data/wikitext103/tokenizer.json")
    model, step = load_model(args.checkpoint)
    if step != args.expected_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_step}, found {step}"
        )
    with parametrize.cached():
        rows, examples = evaluate(
            model,
            validation,
            starts,
            tokenizer,
            micro=args.micro,
            sample_count=args.sample_count,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "section",
        "metric",
        "count",
        "mean",
        "median",
        "p10",
        "p90",
        "min",
        "max",
    )
    with (args.output_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "examples.md").write_text("\n".join(examples))
    with (args.output_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("checkpoint", args.checkpoint))
        writer.writerow(("checkpoint_step", step))
        writer.writerow(("checkpoint_sha256", hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()))
        writer.writerow(("examples", args.examples))
        writer.writerow(("starts_seed", SEED + 7777))
        writer.writerow(("precision", "strict_float32_no_tf32"))

    key_lines = [
        "# Step-1000 causal K / K^2 head-logit audit",
        "",
        f"- Checkpoint step: {step}",
        f"- Fixed validation prefixes: {args.examples}",
        "- h1 accuracy (short tape): "
        f"{row_value(rows, 'h1_short_tape', 'accuracy', 'mean'):.6f}",
        "- h1 accuracy (joint K,K^2 tape): "
        f"{row_value(rows, 'h1_joint_tape', 'accuracy', 'mean'):.6f}",
        "- h2 accuracy (joint tape): "
        f"{row_value(rows, 'h2_joint_tape', 'accuracy', 'mean'):.6f}",
        "- h1 short-vs-joint hidden max-abs (global max): "
        f"{row_value(rows, 'causal_invariance_raw_hidden', 'max_abs', 'max'):.6e}",
        "- h1 short-vs-joint logits max-abs (global max): "
        f"{row_value(rows, 'causal_invariance_logits', 'max_abs', 'max'):.6e}",
        "- h1-vs-h2 head-input cosine (mean): "
        f"{row_value(rows, 'h1_vs_h2_head_input', 'cosine', 'mean'):.6f}",
        "- h1-vs-h2 logits cosine (mean): "
        f"{row_value(rows, 'h1_vs_h2_logits', 'cosine', 'mean'):.6f}",
        "- h1-vs-h2 centered-logits cosine (mean): "
        f"{row_value(rows, 'h1_vs_h2_logits_centered', 'cosine', 'mean'):.6f}",
        "- h1-vs-h2 probability cosine / JS (mean): "
        f"{row_value(rows, 'h1_vs_h2_logits', 'probability_cosine', 'mean'):.6f} / "
        f"{row_value(rows, 'h1_vs_h2_logits', 'jensen_shannon', 'mean'):.6f}",
        "- h1-vs-h2 top-5 overlap (mean fraction): "
        f"{row_value(rows, 'h1_vs_h2_logits', 'top5_overlap_fraction', 'mean'):.6f}",
        "- h1/h2 same top-1 token rate: "
        f"{row_value(rows, 'h1_vs_h2', 'top1_same_token', 'mean'):.6f}",
        "- h1 target-rank median: "
        f"{row_value(rows, 'h1_joint_tape', 'target_rank', 'median'):.1f}",
        "- h2 target-rank median: "
        f"{row_value(rows, 'h2_joint_tape', 'target_rank', 'median'):.1f}",
        "- exact-gold hB/hC head accuracy: "
        f"{row_value(rows, 'gold_hB_control', 'accuracy', 'mean'):.6f} / "
        f"{row_value(rows, 'gold_hC_control', 'accuracy', 'mean'):.6f}",
        "",
        "Full distributions are in `summary.tsv`; token examples are in `examples.md`.",
    ]
    (args.output_dir / "analysis.md").write_text("\n".join(key_lines))
    print("\n".join(key_lines))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
