"""Detailed epoch-3 free-generation audit on fixed validation prompts."""
from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path
import re
import statistics

import torch

import eval_byte256_unitary_fused_scan_step10071_flow_generation as prior
from train_byte256_unitary_time_varying_scan_h16_1000 import (
    configure_scan_backend,
)


base = prior.base
RECORD = Path(
    "experiments/records/"
    "EXP-20260803-byte256-unitary-fused-triton-scan-step10071-"
    "generation-quality-64prompt"
)
SAMPLES = 64
GENERATE = 128
TOP_P = 0.9
TEMPERATURE = 0.8
REPORT_SAMPLES = 12


def top_p_selector(generator: torch.Generator):
    def select(logits: torch.Tensor) -> torch.Tensor:
        probabilities = (logits.float() / TEMPERATURE).softmax(dim=-1)
        sorted_probabilities, sorted_indices = probabilities.sort(
            dim=-1,
            descending=True,
        )
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative > TOP_P
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered = sorted_probabilities.masked_fill(remove, 0.0)
        samples = torch.multinomial(
            filtered.reshape(-1, filtered.shape[-1]),
            1,
            generator=generator,
        ).reshape(*filtered.shape[:-1], 1)
        return sorted_indices.gather(-1, samples).squeeze(-1)

    return select


def escaped(values: list[int]) -> str:
    decoded = bytes(values).decode("utf-8", errors="backslashreplace")
    return decoded.encode("unicode_escape").decode("ascii")


def byte_entropy(values: list[int]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def repeated_ngram_fraction(values: list[int], n: int) -> float:
    grams = [tuple(values[index : index + n]) for index in range(len(values) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def longest_identical_run(values: list[int]) -> int:
    longest = current = 1
    for previous, value in zip(values, values[1:]):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest


def word_metrics(values: list[int]) -> tuple[int, float, float]:
    text = bytes(values).decode("utf-8", errors="replace").lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    bigrams = list(zip(words, words[1:]))
    type_token = len(set(words)) / len(words) if words else 0.0
    repeated_bigrams = (
        1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    )
    return len(words), type_token, repeated_bigrams


def invalid_utf8_replacements(values: list[int]) -> int:
    return bytes(values).decode("utf-8", errors="replace").count("\ufffd")


def prefix_match_length(prediction: list[int], target: list[int]) -> int:
    count = 0
    for predicted, gold in zip(prediction, target):
        if predicted != gold:
            break
        count += 1
    return count


def aggregate_mode(
    name: str,
    values: torch.Tensor,
    gold: torch.Tensor,
) -> dict[str, object]:
    value_rows = values.cpu().tolist()
    gold_rows = gold.cpu().tolist()
    flat = [value for row in value_rows for value in row]
    prefix_lengths = [
        prefix_match_length(row, target)
        for row, target in zip(value_rows, gold_rows)
    ]
    words = [word_metrics(row) for row in value_rows]
    position_modal = []
    for position in range(values.shape[1]):
        counts = Counter(int(value) for value in values[:, position].cpu())
        position_modal.append(max(counts.values()) / values.shape[0])
    row: dict[str, object] = {
        "mode": name,
        "samples": values.shape[0],
        "bytes_per_sample": values.shape[1],
        "corpus_byte_entropy_bits": byte_entropy(flat),
        "corpus_unique_bytes": len(set(flat)),
        "mean_sample_unique_bytes": statistics.mean(
            len(set(sample)) for sample in value_rows
        ),
        "space_fraction": flat.count(32) / len(flat),
        "newline_fraction": flat.count(10) / len(flat),
        "ascii_printable_fraction": statistics.mean(
            32 <= value <= 126 or value in (9, 10, 13) for value in flat
        ),
        "ascii_alpha_fraction": statistics.mean(
            65 <= value <= 90 or 97 <= value <= 122 for value in flat
        ),
        "strict_utf8_sample_fraction": statistics.mean(
            invalid_utf8_replacements(sample) == 0 for sample in value_rows
        ),
        "mean_utf8_replacements": statistics.mean(
            invalid_utf8_replacements(sample) for sample in value_rows
        ),
        "mean_repeated_4gram_fraction": statistics.mean(
            repeated_ngram_fraction(sample, 4) for sample in value_rows
        ),
        "mean_longest_identical_byte_run": statistics.mean(
            longest_identical_run(sample) for sample in value_rows
        ),
        "max_longest_identical_byte_run": max(
            longest_identical_run(sample) for sample in value_rows
        ),
        "mean_words": statistics.mean(item[0] for item in words),
        "mean_word_type_token_ratio": statistics.mean(
            item[1] for item in words
        ),
        "mean_repeated_word_bigram_fraction": statistics.mean(
            item[2] for item in words
        ),
        "mean_position_modal_fraction": statistics.mean(position_modal),
        "mean_exact_prefix_length": statistics.mean(prefix_lengths),
        "max_exact_prefix_length": max(prefix_lengths),
    }
    for length in (1, 4, 8, 16, 32, 64, 128):
        row[f"aligned_accuracy_first_{length}"] = float(
            (values[:, :length] == gold[:, :length]).float().mean()
        )
    return row


def position_buckets(
    name: str,
    values: torch.Tensor,
    gold: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    for start, end in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)):
        selected = values[:, start:end]
        flat = selected.flatten().cpu().tolist()
        rows.append({
            "mode": name,
            "start_offset_1based": start + 1,
            "end_offset_1based": end,
            "aligned_byte_accuracy": float(
                (selected == gold[:, start:end]).float().mean()
            ),
            "byte_entropy_bits": byte_entropy(flat),
            "unique_bytes": len(set(flat)),
            "space_fraction": flat.count(32) / len(flat),
        })
    return rows


def h16_phases(
    name: str,
    values: torch.Tensor,
    gold: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    for phase in range(16):
        selected = values[:, phase::16]
        flat = selected.flatten().cpu().tolist()
        counts = Counter(flat)
        modal_byte, modal_count = counts.most_common(1)[0]
        rows.append({
            "mode": name,
            "phase_1based": phase + 1,
            "aligned_byte_accuracy": float(
                (selected == gold[:, phase::16]).float().mean()
            ),
            "byte_entropy_bits": byte_entropy(flat),
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
    checkpoint = torch.load(
        prior.CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint["step"]) != 10071:
        raise RuntimeError("expected the epoch-3 checkpoint")
    model = base.make_model().cuda().eval()
    configure_scan_backend(model, "fused-triton-rotating-frame")
    model.load_state_dict(checkpoint["model"])

    validation = base.memmap("validation")
    starts = base.load_fixed_starts(prior.BASELINE_STARTS, SAMPLES)
    prompts = base.windows_of_length(validation, starts, base.CONTEXT)
    gold = base.windows_of_length(
        validation,
        starts + base.CONTEXT,
        GENERATE,
    )
    generators = {
        "h1_top_p": torch.Generator(device="cuda").manual_seed(
            base.SEED + 31001
        ),
        "h16_top_p": torch.Generator(device="cuda").manual_seed(
            base.SEED + 31016
        ),
    }
    outputs = {
        "gold": gold,
        "all_space": torch.full_like(gold, 32),
        "h1_greedy": prior.generate(
            model,
            prompts,
            1,
            generate_count=GENERATE,
        ),
        "h16_greedy": prior.generate(
            model,
            prompts,
            16,
            generate_count=GENERATE,
        ),
        "h1_top_p": prior.generate(
            model,
            prompts,
            1,
            selector=top_p_selector(generators["h1_top_p"]),
            generate_count=GENERATE,
        ),
        "h16_top_p": prior.generate(
            model,
            prompts,
            16,
            selector=top_p_selector(generators["h16_top_p"]),
            generate_count=GENERATE,
        ),
    }

    metric_rows = [
        aggregate_mode(name, values, gold)
        for name, values in outputs.items()
    ]
    bucket_rows = [
        row
        for name, values in outputs.items()
        for row in position_buckets(name, values, gold)
    ]
    phase_rows = [
        row
        for name in ("h16_greedy", "h16_top_p")
        for row in h16_phases(name, outputs[name], gold)
    ]
    sample_rows = []
    for sample in range(REPORT_SAMPLES):
        prompt_values = prompts[sample, -96:].cpu().tolist()
        for name in ("gold", "h1_greedy", "h1_top_p", "h16_greedy", "h16_top_p"):
            values = outputs[name][sample].cpu().tolist()
            sample_rows.append({
                "sample": sample,
                "validation_start": int(starts[sample]),
                "mode": name,
                "prompt_suffix_escaped": escaped(prompt_values),
                "continuation_escaped": escaped(values),
                "continuation_hex": bytes(values).hex(),
            })

    RECORD.mkdir(parents=True, exist_ok=True)
    write_rows(RECORD / "metrics.tsv", metric_rows)
    write_rows(RECORD / "position_buckets.tsv", bucket_rows)
    write_rows(RECORD / "h16_phase.tsv", phase_rows)
    write_rows(RECORD / "samples.tsv", sample_rows)
    for row in metric_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
