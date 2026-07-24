"""Compare K hA error with equal-size random perturbations around gold hB."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from analyze_ha_skew_iresnet_k1_k2_head_logits import (
    append_metric,
    load_model,
    relative_mse_rows,
    row_value,
    summarize,
)
from train_k1_decoder_inverse_ablation_13m import CONTEXT, SEED, memmap
from train_k1_ha_skew_iresnet_spectral_learned_noise_13m import (
    EXPERIMENT_ID,
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
    / "step1000_hb_matched_noise"
)


def decode_last(
    model: torch.nn.Module,
    prefix_encoded: torch.Tensor,
    state: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    tape = torch.cat((prefix_encoded, state[:, None]), dim=1)
    return model.encoder.decode_hidden(tape, positions)[:, -1]


def record_variant(
    storage: dict[tuple[str, str], list[torch.Tensor]],
    section: str,
    *,
    state: torch.Tensor,
    h_b: torch.Tensor,
    hidden: torch.Tensor,
    gold_hidden: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> None:
    input_delta = state.float() - h_b.float()
    output_delta = hidden.float() - gold_hidden.float()
    input_norm = input_delta.norm(dim=-1)
    output_norm = output_delta.norm(dim=-1)
    target_logits = logits.gather(-1, target[:, None]).squeeze(-1)
    top2_values, top2_ids = logits.topk(2, dim=-1)
    prediction = top2_ids[:, 0]
    best_wrong = torch.where(
        prediction.eq(target), top2_values[:, 1], top2_values[:, 0]
    )
    log_probabilities = F.log_softmax(logits, dim=-1)
    target_log_probability = log_probabilities.gather(
        -1, target[:, None]
    ).squeeze(-1)

    for metric, values in (
        ("input_relative_mse", relative_mse_rows(state, h_b)),
        (
            "input_cosine",
            F.cosine_similarity(state.float(), h_b.float(), dim=-1),
        ),
        ("input_error_l2", input_norm),
        ("decoded_relative_mse", relative_mse_rows(hidden, gold_hidden)),
        (
            "decoded_cosine",
            F.cosine_similarity(
                hidden.float(), gold_hidden.float(), dim=-1
            ),
        ),
        ("decoded_error_l2", output_norm),
        ("local_l2_gain", output_norm / input_norm.clamp_min(1e-12)),
        ("accuracy", prediction.eq(target)),
        ("nll", -target_log_probability),
        ("target_rank", logits.gt(target_logits[:, None]).sum(-1) + 1),
        ("target_margin", target_logits - best_wrong),
    ):
        append_metric(storage, section, metric, values)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    validation,
    starts: torch.Tensor,
    *,
    micro: int,
    noise_draws: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    storage: dict[tuple[str, str], list[torch.Tensor]] = {}
    generator = torch.Generator(device="cuda").manual_seed(SEED + 8811)
    geometry_distance_residual = 0.0
    geometry_norm_residual = 0.0

    for begin in range(0, len(starts), micro):
        window = windows_of_length(
            validation, starts[begin : begin + micro], CONTEXT + 1
        )
        target = window[:, -1]
        batch = window.shape[0]
        positions = torch.arange(
            CONTEXT + 1, device=window.device
        ).unsqueeze(0).expand(batch, -1)
        full_encoded, _ = model.encode(window, positions)
        prefix_encoded = full_encoded[:, :-1]
        h_b = full_encoded[:, -1]
        h_a = prefix_encoded[:, -1]
        k_h_a = model.operator(h_a)

        gold_hidden = model.encoder.decode_hidden(
            full_encoded, positions
        )[:, -1]
        actual_hidden = decode_last(
            model, prefix_encoded, k_h_a, positions
        )
        actual_logits = model.token_logits(actual_hidden).float()
        record_variant(
            storage,
            "actual_kha",
            state=k_h_a,
            h_b=h_b,
            hidden=actual_hidden,
            gold_hidden=gold_hidden,
            logits=actual_logits,
            target=target,
        )

        gold_logits = model.token_logits(gold_hidden).float()
        append_metric(
            storage,
            "gold_hb",
            "accuracy",
            gold_logits.argmax(-1).eq(target),
        )
        append_metric(
            storage,
            "gold_hb",
            "embedding_roundtrip_max_abs",
            (
                gold_hidden.float() - model.encoder.embed(target).float()
            ).abs().max(dim=-1).values,
        )

        actual_error = k_h_a - h_b
        error_norm = actual_error.float().norm(dim=-1, keepdim=True)
        h_b_unit = h_b.float() / h_b.float().norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        parallel_scalar = (
            actual_error.float() * h_b_unit
        ).sum(dim=-1, keepdim=True)
        parallel_error = parallel_scalar * h_b_unit
        orthogonal_norm = (
            actual_error.float() - parallel_error
        ).norm(dim=-1, keepdim=True)

        for _ in range(noise_draws):
            random = torch.randn(
                h_b.shape,
                dtype=h_b.dtype,
                device=h_b.device,
                generator=generator,
            )
            norm_noise = random * (
                error_norm / random.float().norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            )
            norm_state = h_b + norm_noise
            norm_hidden = decode_last(
                model, prefix_encoded, norm_state, positions
            )
            record_variant(
                storage,
                "matched_norm_noise",
                state=norm_state,
                h_b=h_b,
                hidden=norm_hidden,
                gold_hidden=gold_hidden,
                logits=model.token_logits(norm_hidden).float(),
                target=target,
            )

            random_orthogonal = (
                random.float()
                - (random.float() * h_b_unit).sum(
                    dim=-1, keepdim=True
                )
                * h_b_unit
            )
            random_orthogonal = random_orthogonal * (
                orthogonal_norm
                / random_orthogonal.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            )
            geometry_error = parallel_error + random_orthogonal
            geometry_state = h_b.float() + geometry_error
            geometry_distance_residual = max(
                geometry_distance_residual,
                float(
                    (
                        geometry_error.norm(dim=-1, keepdim=True)
                        - error_norm
                    ).abs().max()
                ),
            )
            geometry_norm_residual = max(
                geometry_norm_residual,
                float(
                    (
                        geometry_state.norm(dim=-1, keepdim=True)
                        - k_h_a.float().norm(dim=-1, keepdim=True)
                    ).abs().max()
                ),
            )
            geometry_hidden = decode_last(
                model,
                prefix_encoded,
                geometry_state.to(h_b.dtype),
                positions,
            )
            record_variant(
                storage,
                "matched_geometry_noise",
                state=geometry_state,
                h_b=h_b,
                hidden=geometry_hidden,
                gold_hidden=gold_hidden,
                logits=model.token_logits(geometry_hidden).float(),
                target=target,
            )

    return summarize(storage), {
        "geometry_distance_match_max_abs": geometry_distance_residual,
        "geometry_state_norm_match_max_abs": geometry_norm_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--micro", type=int, default=8)
    parser.add_argument("--noise-draws", type=int, default=8)
    parser.add_argument("--expected-step", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this audit requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    validation = memmap("validation")
    starts = torch.randint(
        0,
        len(validation) - CONTEXT - HORIZONS - 1,
        (args.examples,),
        generator=torch.Generator().manual_seed(SEED + 7777),
    )
    model, step = load_model(args.checkpoint)
    if step != args.expected_step:
        raise RuntimeError(
            f"expected checkpoint step {args.expected_step}, found {step}"
        )
    with parametrize.cached():
        rows, diagnostics = evaluate(
            model,
            validation,
            starts,
            micro=args.micro,
            noise_draws=args.noise_draws,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.tsv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = (
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
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    diagnostics_path = args.output_dir / "diagnostics.tsv"
    with diagnostics_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("metric", "value"))
        writer.writerows(diagnostics.items())

    sections = (
        "actual_kha",
        "matched_norm_noise",
        "matched_geometry_noise",
    )
    metrics = (
        "input_relative_mse",
        "input_cosine",
        "decoded_relative_mse",
        "decoded_cosine",
        "local_l2_gain",
        "accuracy",
        "nll",
        "target_rank",
        "target_margin",
    )
    report = [
        "# Step-1000 matched hB-noise decoder audit",
        "",
        f"- Checkpoint step: {step}",
        f"- Examples: {args.examples}",
        f"- Random draws: {args.noise_draws}",
        "",
        "| variant | " + " | ".join(metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for section in sections:
        values = [
            row_value(rows, section, metric, "mean")
            for metric in metrics
        ]
        report.append(
            f"| {section} | "
            + " | ".join(f"{value:.8g}" for value in values)
            + " |"
        )
    actual_decoded = row_value(
        rows, "actual_kha", "decoded_relative_mse", "mean"
    )
    geometry_decoded = row_value(
        rows,
        "matched_geometry_noise",
        "decoded_relative_mse",
        "mean",
    )
    report.extend(
        [
            "",
            "- Geometry-matched / actual decoded relative-MSE ratio: "
            f"{geometry_decoded / actual_decoded:.8g}",
            "- Gold hB accuracy: "
            f"{row_value(rows, 'gold_hb', 'accuracy', 'mean'):.8g}",
            "- Gold embedding roundtrip max: "
            f"{row_value(rows, 'gold_hb', 'embedding_roundtrip_max_abs', 'max'):.8g}",
            "- Geometry distance-match max residual: "
            f"{diagnostics['geometry_distance_match_max_abs']:.8g}",
            "- Geometry state-norm-match max residual: "
            f"{diagnostics['geometry_state_norm_match_max_abs']:.8g}",
        ]
    )
    analysis_path = args.output_dir / "analysis.md"
    analysis_path.write_text("\n".join(report))
    print("\n".join(report))
    print(f"wrote {summary_path}")
    print(f"wrote {diagnostics_path}")
    print(f"wrote {analysis_path}")


if __name__ == "__main__":
    main()
