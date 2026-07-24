"""Matched i-ResNet spectral-control run for the strongest h_A K=1 model.

Everything in ``train_k1_ha_skew_learned_noise_mse_ce_13m.py`` is retained,
including the attached h_B target, CE + relative MSE objective, learned noise,
orthogonal K, exact-inverse decoder, and rms-tied head.  The sole training
intervention is i-ResNet Eq. (2) soft spectral normalization on the eight
attention/FFN linear maps in the two shared reversible blocks.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.utils import parametrize

from rotlm.models.iresnet_spectral import (
    IResNetSpectralStats,
    apply_iresnet_soft_spectral_norm,
    exact_effective_spectral_norms,
    iter_iresnet_spectral_layers,
    update_iresnet_power_vectors,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from train_k1_decoder_inverse_ablation_13m import (
    CLIP_NORM,
    CONTEXT,
    PEAK_LR,
    SEED,
    VOCABULARY,
    lr_at,
    memmap,
    parameter_count,
    sampled_windows,
    trainable_parameter_count,
    windows,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import (
    HORIZONS,
    INIT_SIGMA,
    WIDTH,
    Metrics,
    check_extrapolation as baseline_check_extrapolation,
    evaluate as baseline_evaluate,
    loss_terms as baseline_loss_terms,
    make_model as make_baseline_model,
    preflight as baseline_preflight,
)


STEPS = 6000
SPECTRAL_COEFFICIENT = 0.9
SPECTRAL_POWER_ITERATIONS = 5
SPECTRAL_VECTOR_SEED = SEED + 240724
EXTRAPOLATION_SAMPLE = 256
EXPERIMENT_ID = "EXP-20260724-k1-ha-skew-iresnet-spectral-learned-noise-13m"
RECORD = Path("experiments/records") / EXPERIMENT_ID
OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID


def make_model() -> K1DecoderAblationLM:
    """Build the matched baseline, then constrain only reversible-block linears."""
    model = make_baseline_model()
    generator = torch.Generator().manual_seed(SPECTRAL_VECTOR_SEED)
    for block in model.encoder.blocks:
        for linear in (
            block.attn.qkv,
            block.attn.proj,
            block.ffn[0],
            block.ffn[2],
        ):
            apply_iresnet_soft_spectral_norm(
                linear,
                coeff=SPECTRAL_COEFFICIENT,
                n_power_iterations=SPECTRAL_POWER_ITERATIONS,
                generator=generator,
            )
    return model


def loss_terms(model: K1DecoderAblationLM, window: torch.Tensor):
    # Cache each effective W_tilde once so encode and its analytic inverse use
    # bit-identical shared weights throughout this top-level computation.
    with parametrize.cached():
        return baseline_loss_terms(model, window)


def evaluate(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    micro: int,
    operator_gradient_norm: float,
) -> Metrics:
    with parametrize.cached():
        return baseline_evaluate(
            model, validation, starts, micro, operator_gradient_norm
        )


def check_extrapolation(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    micro: int = 32,
) -> dict:
    with parametrize.cached():
        return baseline_check_extrapolation(model, validation, starts, micro)


@torch.inference_mode()
def inverse_roundtrip_max_abs(
    model: K1DecoderAblationLM,
    data,
    starts: torch.Tensor,
) -> float:
    window = windows(data, starts)
    positions = torch.arange(window.shape[1], device=window.device)
    embedded = model.encoder.embed(window)
    with parametrize.cached():
        encoded, encoded_positions = model.encode(window, positions)
        reconstructed = model.encoder.decode_hidden(encoded, encoded_positions)
    return float((reconstructed - embedded).abs().max())


def preflight(
    model: K1DecoderAblationLM,
    training,
    batch: int,
) -> tuple[dict[str, float], IResNetSpectralStats]:
    spectral_stats = update_iresnet_power_vectors(model)
    with parametrize.cached():
        values = baseline_preflight(model, training, batch)
    starts = torch.randint(
        0,
        len(training) - CONTEXT - 1,
        (min(batch, 2),),
        generator=torch.Generator().manual_seed(SEED + 5151),
    )
    roundtrip = inverse_roundtrip_max_abs(model, training, starts)
    if roundtrip >= 1e-4:
        raise RuntimeError(
            "spectral parameterization broke the analytic inverse: "
            f"max abs roundtrip error={roundtrip:.3e}"
        )
    exact = exact_effective_spectral_norms(model)
    values.update(
        {
            "inverse_roundtrip_max_abs_init": roundtrip,
            "spectral_layer_count": spectral_stats.layer_count,
            "spectral_active_layer_count_init": spectral_stats.active_layer_count,
            "spectral_raw_sigma_max_estimate_init": (
                spectral_stats.raw_sigma_max_estimate
            ),
            "spectral_effective_sigma_max_estimate_init": (
                spectral_stats.effective_sigma_max_estimate
            ),
            "spectral_effective_sigma_max_exact_init": max(exact.values()),
        }
    )
    return values, spectral_stats


def spectral_audit_rows(
    model: K1DecoderAblationLM,
    *,
    checkpoint: str,
    step: int,
) -> list[dict[str, object]]:
    exact = exact_effective_spectral_norms(model)
    rows: list[dict[str, object]] = []
    for name, _, spectral in iter_iresnet_spectral_layers(model):
        raw = float(spectral.sigma_estimate)
        estimated_effective = raw / max(1.0, raw / spectral.coeff)
        rows.append(
            {
                "checkpoint": checkpoint,
                "step": step,
                "layer": name,
                "coefficient": spectral.coeff,
                "raw_sigma_estimate": raw,
                "effective_sigma_estimate": estimated_effective,
                "effective_sigma_exact": exact[name],
            }
        )
    return rows


def save_checkpoint(
    path: Path,
    model: K1DecoderAblationLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Metrics,
    extrapolation: dict,
    spectral_stats: IResNetSpectralStats,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": asdict(metrics),
            "extrapolation": extrapolation,
            "spectral_stats": asdict(spectral_stats),
            "config": {
                "experiment_id": EXPERIMENT_ID,
                "vocabulary": VOCABULARY,
                "width": WIDTH,
                "seed": SEED,
                "operator": "skew_matrix_exp_no_query",
                "spectral_method": "iresnet_eq2_soft_spectral_normalization",
                "spectral_coefficient": SPECTRAL_COEFFICIENT,
                "spectral_power_iterations": SPECTRAL_POWER_ITERATIONS,
                "spectral_targets": (
                    "encoder.blocks.*.{attn.qkv,attn.proj,ffn.0,ffn.2}"
                ),
            },
        },
        path,
    )


def write_spectral_audit(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "checkpoint",
        "step",
        "layer",
        "coefficient",
        "raw_sigma_estimate",
        "effective_sigma_estimate",
        "effective_sigma_exact",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-micro", type=int, default=16)
    parser.add_argument(
        "--extrapolation-examples", type=int, default=EXTRAPOLATION_SAMPLE
    )
    parser.add_argument("--record-dir", type=Path, default=RECORD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    training, validation = memmap("train"), memmap("validation")
    validation_starts = torch.randint(
        0,
        len(validation) - CONTEXT - 1,
        (args.eval_examples,),
        generator=torch.Generator().manual_seed(SEED + 999),
    )
    extrapolation_starts = torch.randint(
        0,
        len(validation) - CONTEXT - HORIZONS - 1,
        (args.extrapolation_examples,),
        generator=torch.Generator().manual_seed(SEED + 7777),
    )
    roundtrip_starts = torch.randint(
        0,
        len(validation) - CONTEXT - 1,
        (2,),
        generator=torch.Generator().manual_seed(SEED + 5151),
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model().cuda().train()
    preflight_values, spectral_stats = preflight(model, training, args.batch)
    initial_audit = spectral_audit_rows(model, checkpoint="initial", step=0)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"initial_ce={preflight_values['initial_ce']:.4f} "
        f"initial_mse={preflight_values['initial_mse']:.4f} "
        f"spectral_exact_max={preflight_values['spectral_effective_sigma_max_exact_init']:.4f} "
        f"roundtrip={preflight_values['inverse_roundtrip_max_abs_init']:.2e}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    with (args.record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in (
            ("experiment_id", EXPERIMENT_ID),
            ("authorization", "user_requested"),
            ("seed", SEED),
            ("steps", args.steps),
            ("batch", args.batch),
            ("context", CONTEXT),
            ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"),
            ("init_sigma", INIT_SIGMA),
            ("operator", "skew_matrix_exp_no_query"),
            ("k_input", "h_A_ordinary_causal_state"),
            ("h_b_gradient", "attached_matches_executed_baseline"),
            ("spectral_method", "iresnet_eq2_soft_spectral_normalization"),
            ("spectral_coefficient", SPECTRAL_COEFFICIENT),
            ("spectral_power_iterations", SPECTRAL_POWER_ITERATIONS),
            ("spectral_update", "once_per_optimizer_step_then_cached"),
            (
                "spectral_targets",
                "encoder.blocks.*.{attn.qkv,attn.proj,ffn.0,ffn.2}",
            ),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    report_steps = (
        {1, 50, 100}
        | set(range(250, args.steps + 1, 250))
        | {args.steps}
    )
    fields = (
        "step",
        "train_combined",
        "train_ce",
        "train_mse",
        *tuple(f"val_{key}" for key in asdict(Metrics(*(0.0,) * 12))),
        *tuple(f"direct_clean_h{h}" for h in range(1, HORIZONS + 1)),
        *tuple(f"hard_token_h{h}" for h in range(1, HORIZONS + 1)),
        "spectral_active_layers",
        "spectral_raw_sigma_max_estimate",
        "spectral_effective_sigma_max_estimate",
        "inverse_roundtrip_max_abs",
        "lr",
        "wall_s",
        "tokens_per_s",
        "peak_vram_bytes",
    )
    data_generator = torch.Generator().manual_seed(SEED)
    best_nll, best_step = math.inf, 0
    started = time.time()
    processed_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    with (args.record_dir / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(args.steps):
            lr = lr_at(index, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(training, args.batch, data_generator)
            optimizer.zero_grad(set_to_none=True)
            loss, ce, mse, _, _, _, _, _ = loss_terms(model, window)
            loss.backward()
            operator_gradient_norm = float(model.operator.A.grad.float().norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            spectral_stats = update_iresnet_power_vectors(model)

            processed_tokens += args.batch * CONTEXT
            step = index + 1
            if step in report_steps:
                metrics = evaluate(
                    model,
                    validation,
                    validation_starts,
                    args.eval_micro,
                    operator_gradient_norm,
                )
                extrapolation = check_extrapolation(
                    model, validation, extrapolation_starts
                )
                roundtrip = inverse_roundtrip_max_abs(
                    model, validation, roundtrip_starts
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    "train_combined": float(loss),
                    "train_ce": float(ce),
                    "train_mse": float(mse),
                    **{
                        f"val_{key}": value
                        for key, value in asdict(metrics).items()
                    },
                    **{
                        f"direct_clean_h{h + 1}": extrapolation["direct_clean"][h]
                        for h in range(HORIZONS)
                    },
                    **{
                        f"hard_token_h{h + 1}": extrapolation["hard_token"][h]
                        for h in range(HORIZONS)
                    },
                    "spectral_active_layers": spectral_stats.active_layer_count,
                    "spectral_raw_sigma_max_estimate": (
                        spectral_stats.raw_sigma_max_estimate
                    ),
                    "spectral_effective_sigma_max_estimate": (
                        spectral_stats.effective_sigma_max_estimate
                    ),
                    "inverse_roundtrip_max_abs": roundtrip,
                    "lr": lr,
                    "wall_s": wall,
                    "tokens_per_s": processed_tokens / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d} val_nll={metrics.nll:.4f} "
                    f"acc={metrics.accuracy:.4f} "
                    f"spec_raw/effective={spectral_stats.raw_sigma_max_estimate:.3f}/"
                    f"{spectral_stats.effective_sigma_max_estimate:.3f} "
                    f"roundtrip={roundtrip:.2e} wall={wall:.1f}s",
                    flush=True,
                )
                print(
                    "  h1-5 direct_clean: "
                    + " ".join(
                        f"{value:.3f}"
                        for value in extrapolation["direct_clean"]
                    ),
                    flush=True,
                )
                print(
                    "  h1-5 hard_token:   "
                    + " ".join(
                        f"{value:.3f}"
                        for value in extrapolation["hard_token"]
                    ),
                    flush=True,
                )
                save_checkpoint(
                    args.output_dir / "last.pt",
                    model,
                    optimizer,
                    step,
                    metrics,
                    extrapolation,
                    spectral_stats,
                )
                if metrics.nll < best_nll:
                    best_nll, best_step = metrics.nll, step
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        step,
                        metrics,
                        extrapolation,
                        spectral_stats,
                    )

    final_audit = spectral_audit_rows(
        model, checkpoint="last", step=args.steps
    )
    if best_step == args.steps:
        best_audit = [
            {**row, "checkpoint": "best"} for row in final_audit
        ]
    else:
        payload = torch.load(
            args.output_dir / "best.pt",
            map_location="cuda",
            weights_only=False,
        )
        model.load_state_dict(payload["model"])
        best_audit = spectral_audit_rows(
            model, checkpoint="best", step=best_step
        )
        del payload
    audit_rows = initial_audit + final_audit + best_audit
    write_spectral_audit(args.record_dir / "spectral_audit.tsv", audit_rows)
    last_exact_max = max(
        float(row["effective_sigma_exact"]) for row in final_audit
    )
    best_exact_max = max(
        float(row["effective_sigma_exact"]) for row in best_audit
    )

    wall = time.time() - started
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("best_step", best_step))
        writer.writerow(("best_val_nll", best_nll))
        writer.writerow(("best_spectral_effective_sigma_max_exact", best_exact_max))
        writer.writerow(("last_spectral_effective_sigma_max_exact", last_exact_max))
        writer.writerow(("wall_s", wall))
        writer.writerow(("peak_vram_bytes", torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    main()
