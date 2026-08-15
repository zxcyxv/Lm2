"""Train byte-256 simplex-tied h1 CE and audit continuous/AR closure."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from rotlm.dataset import token_memmap
from rotlm.evaluation import (
    input_conditioned_spectral_composition_rollouts,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.spectral_collapse import InputConditionedSpectralCollapse
from rotlm.training.ha_skew_window import (
    SpectralCollapseWindowOutput,
    forward_sparse_spectral_collapse_window,
    relative_mse_rows,
)
from train_k1_decoder_inverse_ablation_13m import (
    CLIP_NORM,
    CONTEXT,
    PEAK_LR,
    SEED,
    lr_at,
    parameter_count,
    trainable_parameter_count,
)
from train_k1_ha_skew_learned_noise_mse_ce_13m import windows_of_length


EXPERIMENT_ID = (
    "EXP-20260801-byte256-simplex-tied-self-composition-h1-ce-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
DATA_ROOT = Path("data/wikitext103_bytes")
VOCABULARY = 256
WIDTH = 1344
ENCODER_BLOCKS = 2
CONTROLLER_BOTTLENECK = 73
SIMPLEX_LOGIT_SCALE = 16.0
NORM_WEIGHT = 10.0
CODEBOOK_RADIUS = 1.0
HORIZONS = 4
TRAIN_HORIZONS = 1
TRAIN_ANCHOR_STRIDE = 1
EVAL_ANCHOR_STRIDE = 4
TRAIN_WINDOW_LENGTH = CONTEXT + TRAIN_HORIZONS
EVAL_WINDOW_LENGTH = CONTEXT + HORIZONS
TRAIN_ANCHORS = len(range(0, CONTEXT, TRAIN_ANCHOR_STRIDE))
INITIAL_FREQUENCY_RANGE = 0.05
MAX_PHASE_DELTA = 0.10
INITIAL_RADIUS = 0.99
MAX_DECAY_LOGIT_DELTA = 2.0
ALPHA = 0.0
CALIBRATION_BATCH = 16
CALIBRATION_SEED = SEED + 9100
EFFECTIVE_BATCH = 64
MICROBATCH = 16
STEPS = 1000
SCHEDULE_STEPS = 6000
REPORT_STEPS = frozenset(
    (
        1,
        50,
        100,
        250,
        500,
        750,
        1000,
        1250,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6000,
    )
)
EXPECTED_PARAMETERS = 13_200_096


def memmap(split: str):
    return token_memmap(split, root=DATA_ROOT)


def make_model() -> K1DecoderAblationLM:
    model = K1DecoderAblationLM(
        VOCABULARY,
        width=WIDTH,
        encoder_blocks=ENCODER_BLOCKS,
        decoder_mode="exact-inverse",
        head_mode="simplex-tied",
        simplex_logit_scale=SIMPLEX_LOGIT_SCALE,
    )
    model.operator = nn.Identity()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        WIDTH,
        bottleneck=CONTROLLER_BOTTLENECK,
        initial_frequency_range=INITIAL_FREQUENCY_RANGE,
        max_phase_delta=MAX_PHASE_DELTA,
        initial_radius=INITIAL_RADIUS,
        max_decay_logit_delta=MAX_DECAY_LOGIT_DELTA,
        alpha=ALPHA,
    )
    model.spectral_collapse.shell_raw.requires_grad_(False)
    return model


def sampled_windows(
    data,
    batch: int,
    generator: torch.Generator,
    length: int,
) -> torch.Tensor:
    starts = torch.randint(
        0,
        len(data) - length,
        (batch,),
        generator=generator,
    )
    return windows_of_length(data, starts, length)


@torch.no_grad()
def calibrate_interface(model, training) -> torch.Tensor:
    starts = torch.randint(
        0,
        len(training) - CONTEXT,
        (CALIBRATION_BATCH,),
        generator=torch.Generator().manual_seed(CALIBRATION_SEED),
    )
    tokens = windows_of_length(training, starts, CONTEXT)
    encoded, _ = model.encode(tokens)
    model.spectral_collapse.calibrate_shell(encoded)
    return starts


def objective(
    model: K1DecoderAblationLM,
    window: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SpectralCollapseWindowOutput,
]:
    output = forward_sparse_spectral_collapse_window(
        model,
        window,
        prefix_length=CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=TRAIN_ANCHOR_STRIDE,
    )
    token_ce = output.token_ce[..., 0].mean()
    norm_error = (
        output.decoded_hidden.float().norm(dim=-1) - CODEBOOK_RADIUS
    ).square().mean()
    return token_ce + NORM_WEIGHT * norm_error, token_ce, norm_error, output


def _require_gradient(name: str, gradient: torch.Tensor | None) -> None:
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.norm() > 0)
    ):
        raise RuntimeError(f"no finite nonzero gradient to {name}")


def preflight(model, training) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 9242)
    window = sampled_windows(
        training,
        2,
        generator,
        TRAIN_WINDOW_LENGTH,
    )
    loss, token_ce, norm_error, output = objective(model, window)
    if output.logits.shape != (2, TRAIN_ANCHORS, 1, VOCABULARY):
        raise RuntimeError(f"unexpected logits {tuple(output.logits.shape)}")
    torch.testing.assert_close(
        loss,
        token_ce + NORM_WEIGHT * norm_error,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("preflight loss is non-finite")

    codebook = model.embedding_weight.detach().float()
    gram = codebook @ codebook.T
    expected_gram = torch.full_like(gram, -1.0 / (VOCABULARY - 1))
    expected_gram.fill_diagonal_(1.0)
    simplex_error = float((gram - expected_gram).abs().max())
    if simplex_error >= 3e-6:
        raise RuntimeError("fixed codebook is not a regular simplex")
    if model.embedding_weight.requires_grad:
        raise RuntimeError("simplex embedding/codebook must be frozen")
    if model.encoder.head_norm.weight.requires_grad:
        raise RuntimeError("legacy learned RMSNorm must be frozen")
    retrieval = float(
        model.token_logits(codebook).argmax(dim=-1).eq(
            torch.arange(VOCABULARY, device=codebook.device)
        ).float().mean()
    )
    if retrieval != 1.0:
        raise RuntimeError("simplex codebook does not retrieve itself")
    total_parameters = parameter_count(model)
    if total_parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"parameter total {total_parameters} != {EXPECTED_PARAMETERS}"
        )
    vocabulary_linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == VOCABULARY
    ]
    if vocabulary_linears:
        raise RuntimeError(
            "independent vocabulary projection found: "
            + ", ".join(vocabulary_linears)
        )

    spectral = model.spectral_collapse
    ce_parameters = (
        spectral.basis_generator,
        spectral.context.weight,
        spectral.phase.weight,
        spectral.decay.weight,
        model.encoder.blocks[0].attn.qkv.weight,
    )
    gradients = torch.autograd.grad(
        token_ce,
        ce_parameters,
        retain_graph=True,
    )
    for name, gradient in zip(
        ("basis", "K controller", "phase", "radius", "reversible stack"),
        gradients,
    ):
        _require_gradient(name, gradient)

    norm_gradient = torch.autograd.grad(
        norm_error,
        spectral.phase.weight,
        retain_graph=True,
    )[0]
    _require_gradient("radial constraint", norm_gradient)
    target_gradient = torch.autograd.grad(
        loss,
        output.gold_states,
        allow_unused=True,
        retain_graph=True,
    )[0]
    if target_gradient is not None and bool(target_gradient.norm() > 0):
        raise RuntimeError("loss reached future hidden states")

    eval_window = sampled_windows(
        training,
        1,
        generator,
        EVAL_WINDOW_LENGTH,
    )
    rollout = input_conditioned_spectral_composition_rollouts(
        model,
        eval_window,
        prefix_length=CONTEXT,
        horizons=HORIZONS,
        anchor_stride=EVAL_ANCHOR_STRIDE,
    )
    h1_state_error = float(
        (
            rollout.composition_states[:, :, 0]
            - rollout.boundary_ar_proposal_states[:, :, 0]
        ).abs().max()
    )
    h1_logit_error = float(
        (
            rollout.composition_logits[:, :, 0]
            - rollout.boundary_ar_logits[:, :, 0]
        ).abs().max()
    )
    if max(h1_state_error, h1_logit_error) >= 3e-4:
        raise RuntimeError("composition and AR do not share h1")

    changed_future = eval_window.clone()
    changed_future[:, CONTEXT:] = (
        changed_future[:, CONTEXT:] + 1
    ) % VOCABULARY
    with torch.no_grad():
        changed = input_conditioned_spectral_composition_rollouts(
            model,
            changed_future,
            prefix_length=CONTEXT,
            horizons=HORIZONS,
            anchor_stride=EVAL_ANCHOR_STRIDE,
        )
    future_state_error = float(
        (
            rollout.composition_states
            - changed.composition_states
        ).abs().max()
    )
    future_logit_error = float(
        (
            rollout.composition_logits
            - changed.composition_logits
        ).abs().max()
    )
    if max(future_state_error, future_logit_error) >= 1e-6:
        raise RuntimeError("future bytes leaked into continuous rollout")

    y = rollout.composition_decoded.float()
    selected = model.embedding_weight[rollout.composition_tokens].float()
    initial_quantization_cosine = float(
        F.cosine_similarity(y, selected, dim=-1).mean()
    )
    initial_norm_abs_error = float(
        (y.norm(dim=-1) - CODEBOOK_RADIUS).abs().mean()
    )
    model.zero_grad(set_to_none=True)
    return {
        "initial_loss": float(loss.detach()),
        "initial_token_ce": float(token_ce.detach()),
        "initial_norm_loss": float(norm_error.detach()),
        "simplex_gram_max_error": simplex_error,
        "simplex_self_retrieval": retrieval,
        "initial_quantization_cosine": initial_quantization_cosine,
        "initial_decoder_norm_abs_error": initial_norm_abs_error,
        "h1_composition_state_error": h1_state_error,
        "h1_composition_logit_error": h1_logit_error,
        "future_state_leak_error": future_state_error,
        "future_logit_leak_error": future_logit_error,
        "future_hidden_target_gradient_norm": (
            0.0
            if target_gradient is None
            else float(target_gradient.norm())
        ),
        "ce_basis_gradient_norm": float(gradients[0].norm()),
        "ce_controller_gradient_norm": float(gradients[1].norm()),
        "norm_constraint_gradient_norm": float(norm_gradient.norm()),
    }


@torch.inference_mode()
def evaluate(
    model,
    validation,
    starts: torch.Tensor,
    microbatch: int,
) -> dict[str, float]:
    names = (
        "overall_cosine",
        "overall_relative_mse",
        "one_step_closure_cosine",
        "one_step_closure_relative_mse",
        "path_cosine",
        "path_relative_mse",
        "token_agreement",
        "quantization_cosine",
        "quantization_squared_distance",
        "decoder_norm",
        "decoder_norm_abs_error",
    )
    sums = {
        name: torch.zeros(HORIZONS, dtype=torch.float64) for name in names
    }
    h1_nll_sum = 0.0
    h1_correct = 0
    exact_blocks = 0
    boundaries = 0

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
        y = output.composition_decoded.float()
        selected = model.embedding_weight[output.composition_tokens].float()
        count = window.shape[0] * output.anchor_indices.numel()
        boundaries += count

        values = {
            "overall_cosine": F.cosine_similarity(
                composition, canonical, dim=-1
            ),
            "overall_relative_mse": relative_mse_rows(
                composition, canonical
            ),
            "one_step_closure_cosine": F.cosine_similarity(
                proposals, canonical, dim=-1
            ),
            "one_step_closure_relative_mse": relative_mse_rows(
                proposals, canonical
            ),
            "path_cosine": F.cosine_similarity(
                composition, proposals, dim=-1
            ),
            "path_relative_mse": relative_mse_rows(
                composition, proposals
            ),
            "token_agreement": output.composition_tokens.eq(
                output.boundary_ar_tokens
            ).float(),
            "quantization_cosine": F.cosine_similarity(y, selected, dim=-1),
            "quantization_squared_distance": (y - selected).square().sum(
                dim=-1
            ),
            "decoder_norm": y.norm(dim=-1),
            "decoder_norm_abs_error": (
                y.norm(dim=-1) - CODEBOOK_RADIUS
            ).abs(),
        }
        for name, value in values.items():
            sums[name] += value.double().sum(dim=(0, 1)).cpu()

        matches = values["token_agreement"].bool()
        exact_blocks += int(matches.all(dim=-1).sum())
        targets = output.targets[:, :, 0]
        h1_nll_sum += float(
            F.cross_entropy(
                output.composition_logits[:, :, 0].flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )
        )
        h1_correct += int(
            output.composition_tokens[:, :, 0].eq(targets).sum()
        )

    if boundaries < 1:
        raise RuntimeError("evaluation produced no boundaries")
    means = {name: values / boundaries for name, values in sums.items()}
    metrics = {
        "evaluated_boundaries": float(boundaries),
        "h1_validation_nll": h1_nll_sum / boundaries,
        "h1_validation_accuracy": h1_correct / boundaries,
        "exact_block_token_agreement": exact_blocks / boundaries,
        "mean_overall_cosine": float(means["overall_cosine"].mean()),
        "mean_overall_relative_mse": float(
            means["overall_relative_mse"].mean()
        ),
        "mean_quantization_cosine": float(
            means["quantization_cosine"].mean()
        ),
        "mean_decoder_norm_abs_error": float(
            means["decoder_norm_abs_error"].mean()
        ),
        "h2_h4_token_agreement": float(
            means["token_agreement"][1:].mean()
        ),
    }
    for horizon in range(HORIZONS):
        for name, values in means.items():
            metrics[f"h{horizon + 1}_{name}"] = float(values[horizon])
    return metrics


METRIC_NAMES = (
    "evaluated_boundaries",
    "h1_validation_nll",
    "h1_validation_accuracy",
    "exact_block_token_agreement",
    "mean_overall_cosine",
    "mean_overall_relative_mse",
    "mean_quantization_cosine",
    "mean_decoder_norm_abs_error",
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
            "quantization_cosine",
            "quantization_squared_distance",
            "decoder_norm",
            "decoder_norm_abs_error",
        )
    ),
)


def save_checkpoint(
    path,
    model,
    optimizer,
    data_generator,
    step,
    metrics,
    *,
    batch: int,
    microbatch: int,
    schedule_steps: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_generator_state": data_generator.get_state(),
            "step": step,
            "metrics": metrics,
            "config": {
                "experiment_id": EXPERIMENT_ID,
                "vocabulary": VOCABULARY,
                "width": WIDTH,
                "encoder_blocks": ENCODER_BLOCKS,
                "controller_bottleneck": CONTROLLER_BOTTLENECK,
                "simplex_logit_scale": SIMPLEX_LOGIT_SCALE,
                "norm_weight": NORM_WEIGHT,
                "objective": "h1_byte_ce_plus_decoder_radius",
                "recurrence": "u_next=K(u_current)@u_current",
                "seed": SEED,
                "batch": batch,
                "microbatch": microbatch,
                "schedule_steps": schedule_steps,
            },
        },
        path,
    )


def load_fixed_starts(path: Path, expected_count: int) -> torch.Tensor:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["index", "token_start"]:
            raise RuntimeError(f"unexpected start-file columns in {path}")
        rows = list(reader)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"{path} has {len(rows)} starts, expected {expected_count}"
        )
    if [int(row["index"]) for row in rows] != list(range(expected_count)):
        raise RuntimeError(f"non-contiguous indices in {path}")
    return torch.tensor(
        [int(row["token_start"]) for row in rows], dtype=torch.long
    )


def load_metric_rows(path: Path, fields: tuple[str, ...]) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError(f"unexpected metric columns in {path}")
        rows = [
            {name: float(value) for name, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise RuntimeError(f"no metric rows in {path}")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(set(steps)):
        raise RuntimeError(f"metric steps are not strictly increasing in {path}")
    return rows


def restore_data_generator(
    checkpoint: dict,
    *,
    training_size: int,
    batch: int,
    window_length: int,
) -> tuple[torch.Generator, str]:
    generator = torch.Generator().manual_seed(SEED)
    state = checkpoint.get("data_generator_state")
    if state is not None:
        generator.set_state(state)
        return generator, "checkpoint_state"

    # The original step-1000 checkpoint predates generator-state storage.
    # One CPU randint call was made per optimizer update, so replaying those
    # calls reconstructs the exact next training window without loading data.
    for _ in range(int(checkpoint["step"])):
        torch.randint(
            0,
            training_size - window_length,
            (batch,),
            generator=generator,
        )
    return generator, "deterministic_replay"


def validate_resume_checkpoint(checkpoint: dict, args) -> int:
    config = checkpoint.get("config", {})
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "vocabulary": VOCABULARY,
        "width": WIDTH,
        "encoder_blocks": ENCODER_BLOCKS,
        "controller_bottleneck": CONTROLLER_BOTTLENECK,
        "simplex_logit_scale": SIMPLEX_LOGIT_SCALE,
        "norm_weight": NORM_WEIGHT,
        "objective": "h1_byte_ce_plus_decoder_radius",
        "recurrence": "u_next=K(u_current)@u_current",
        "seed": SEED,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"resume checkpoint config mismatch: {mismatches}")
    for key, value in (
        ("batch", args.batch),
        ("microbatch", args.microbatch),
        ("schedule_steps", args.schedule_steps),
    ):
        if key in config and config[key] != value:
            raise RuntimeError(
                f"resume checkpoint {key}={config[key]} but requested {value}"
            )
    step = int(checkpoint["step"])
    if step < 1 or step >= args.steps:
        raise RuntimeError(
            f"resume step {step} must be in [1, requested steps {args.steps})"
        )
    return step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=EFFECTIVE_BATCH)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    parser.add_argument("--schedule-steps", type=int, default=SCHEDULE_STEPS)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=4)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(
        args.steps,
        args.batch,
        args.microbatch,
        args.schedule_steps,
        args.eval_examples,
        args.eval_microbatch,
    ) < 1:
        parser.error("all counts must be positive")
    if args.batch % args.microbatch:
        parser.error("batch must be divisible by microbatch")
    if not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    training, validation = memmap("train"), memmap("validation")
    expected_validation_starts = torch.randint(
        0,
        len(validation) - EVAL_WINDOW_LENGTH,
        (args.eval_examples,),
        generator=torch.Generator().manual_seed(SEED + 999),
    )
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_starts_path = args.record_dir / "validation_starts.tsv"
    if args.resume is None:
        validation_starts = expected_validation_starts
        with validation_starts_path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("index", "token_start"))
            writer.writerows(enumerate(validation_starts.tolist()))
    else:
        validation_starts = load_fixed_starts(
            validation_starts_path, args.eval_examples
        )
        if not torch.equal(validation_starts, expected_validation_starts):
            raise RuntimeError(
                "stored validation starts differ from the fixed-seed protocol"
            )

    model = make_model().cuda().train()
    if args.resume is None:
        calibration_starts = calibrate_interface(model, training)
        with (args.record_dir / "calibration_starts.tsv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("index", "token_start"))
            writer.writerows(enumerate(calibration_starts.tolist()))
        preflight_values = preflight(model, training)
    else:
        # Calibration and preflight mutate/check a newly initialized model.
        # The resumed model already contains the calibrated interface.
        load_fixed_starts(
            args.record_dir / "calibration_starts.tsv", CALIBRATION_BATCH
        )
        preflight_values = {}
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    if args.resume is None:
        start_step = 0
        data_generator = torch.Generator().manual_seed(SEED)
        generator_restoration = "fresh_seed"
    else:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        start_step = validate_resume_checkpoint(checkpoint, args)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        data_generator, generator_restoration = restore_data_generator(
            checkpoint,
            training_size=len(training),
            batch=args.batch,
            window_length=TRAIN_WINDOW_LENGTH,
        )
        checkpoint_metrics = checkpoint["metrics"]
        del checkpoint
    accumulation_steps = args.batch // args.microbatch
    if args.resume is None:
        print(
            f"params={total_parameters:,} trainable={trainable_parameters:,} "
            f"width={WIDTH} blocks={ENCODER_BLOCKS} "
            f"batch={args.batch} micro={args.microbatch}x{accumulation_steps} "
            f"initial_ce={preflight_values['initial_token_ce']:.4f}",
            flush=True,
        )
    else:
        print(
            f"resuming step={start_step} -> {args.steps} "
            f"params={total_parameters:,} batch={args.batch} "
            f"micro={args.microbatch}x{accumulation_steps} "
            f"data_rng={generator_restoration}",
            flush=True,
        )

    if args.resume is None:
        with (args.record_dir / "run.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("key", "value"))
            for key, value in (
                ("experiment_id", EXPERIMENT_ID),
                ("seed", SEED),
                ("steps", args.steps),
                ("schedule_steps", args.schedule_steps),
                ("effective_batch", args.batch),
                ("microbatch", args.microbatch),
                ("context", CONTEXT),
                ("vocabulary", VOCABULARY),
                ("width", WIDTH),
                ("encoder_blocks", ENCODER_BLOCKS),
                ("controller_bottleneck", CONTROLLER_BOTTLENECK),
                ("simplex_logit_scale", SIMPLEX_LOGIT_SCALE),
                ("norm_weight", NORM_WEIGHT),
                ("codebook_radius", CODEBOOK_RADIUS),
                ("objective", "h1_byte_ce_plus_decoder_radius"),
                ("recurrence", "u_next=K(u_current)@u_current"),
                ("monitor_horizons", HORIZONS),
                ("train_anchor_stride", TRAIN_ANCHOR_STRIDE),
                ("eval_anchor_stride", EVAL_ANCHOR_STRIDE),
                ("parameters", total_parameters),
                ("trainable_parameters", trainable_parameters),
                ("precision", "strict_float32_no_tf32"),
                *preflight_values.items(),
            ):
                writer.writerow((key, value))
    else:
        continuation_path = args.record_dir / "continuations.tsv"
        write_header = not continuation_path.exists()
        with continuation_path.open("a", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            if write_header:
                writer.writerow(
                    (
                        "start_step",
                        "target_step",
                        "schedule_steps",
                        "batch",
                        "microbatch",
                        "checkpoint",
                        "data_rng_restoration",
                    )
                )
            writer.writerow(
                (
                    start_step,
                    args.steps,
                    args.schedule_steps,
                    args.batch,
                    args.microbatch,
                    str(args.resume),
                    generator_restoration,
                )
            )

    fields = (
        "step",
        "train_loss",
        "train_token_ce",
        "train_norm_loss",
        "gradient_norm",
        *tuple(f"val_{name}" for name in METRIC_NAMES),
        "lr",
        "wall_s",
        "unique_tokens_per_s",
        "ce_labels_per_s",
        "peak_vram_bytes",
    )
    metrics_path = args.record_dir / "metrics.tsv"
    segment_started = time.time()
    processed_tokens = start_step * args.batch * TRAIN_WINDOW_LENGTH
    processed_labels = start_step * args.batch * TRAIN_ANCHORS
    torch.cuda.reset_peak_memory_stats()
    if args.resume is None:
        rows = []
        base_wall = 0.0
        prior_peak_vram = 0
        metrics_mode = "w"
    else:
        rows = load_metric_rows(metrics_path, fields)
        if int(rows[-1]["step"]) != start_step:
            raise RuntimeError(
                f"metrics end at step {int(rows[-1]['step'])}, "
                f"checkpoint is step {start_step}"
            )
        for name in METRIC_NAMES:
            recorded = rows[-1][f"val_{name}"]
            checkpoint_value = float(checkpoint_metrics[name])
            if not math.isclose(
                recorded, checkpoint_value, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise RuntimeError(
                    f"checkpoint/metrics mismatch for {name}: "
                    f"{checkpoint_value} != {recorded}"
                )
        base_wall = rows[-1]["wall_s"]
        prior_peak_vram = int(
            max(row["peak_vram_bytes"] for row in rows)
        )
        metrics_mode = "a"
    with metrics_path.open(metrics_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        if args.resume is None:
            writer.writeheader()
            initial = evaluate(
                model, validation, validation_starts, args.eval_microbatch
            )
            initial_wall = time.time() - segment_started
            initial_row = {
                "step": 0,
                "train_loss": math.nan,
                "train_token_ce": math.nan,
                "train_norm_loss": math.nan,
                "gradient_norm": math.nan,
                **{f"val_{name}": initial[name] for name in METRIC_NAMES},
                "lr": 0.0,
                "wall_s": initial_wall,
                "unique_tokens_per_s": 0.0,
                "ce_labels_per_s": 0.0,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
            writer.writerow(initial_row)
            handle.flush()
            rows.append(initial_row)
            print(
                f"step=   0 nll={initial['h1_validation_nll']:.4f} "
                f"h1_qcos={initial['h1_quantization_cosine']:.4f} "
                f"mean_qcos={initial['mean_quantization_cosine']:.4f} "
                f"normerr={initial['mean_decoder_norm_abs_error']:.4f} "
                f"arcos={initial['mean_overall_cosine']:.4f} "
                f"tok_h2-h4={initial['h2_h4_token_agreement']:.3f}",
                flush=True,
            )
        else:
            initial_row = rows[0]
            print(
                f"verified checkpoint metrics at step={start_step}; "
                f"next_lr={lr_at(start_step, args.schedule_steps):.9f}",
                flush=True,
            )

        for index in range(start_step, args.steps):
            lr = lr_at(index, args.schedule_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(
                training,
                args.batch,
                data_generator,
                TRAIN_WINDOW_LENGTH,
            )
            optimizer.zero_grad(set_to_none=True)
            train_loss = train_ce = train_norm = 0.0
            micro_windows = window.split(args.microbatch)
            for micro_window in micro_windows:
                loss, token_ce, norm_error, output = objective(
                    model, micro_window
                )
                (loss / len(micro_windows)).backward()
                scale = 1.0 / len(micro_windows)
                train_loss += float(loss.detach()) * scale
                train_ce += float(token_ce.detach()) * scale
                train_norm += float(norm_error.detach()) * scale
                del loss, token_ce, norm_error, output
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            )
            optimizer.step()
            processed_tokens += args.batch * TRAIN_WINDOW_LENGTH
            processed_labels += args.batch * TRAIN_ANCHORS
            step = index + 1
            if step in REPORT_STEPS or step == args.steps:
                metrics = evaluate(
                    model,
                    validation,
                    validation_starts,
                    args.eval_microbatch,
                )
                wall = base_wall + (time.time() - segment_started)
                peak_vram = max(
                    prior_peak_vram, torch.cuda.max_memory_allocated()
                )
                row = {
                    "step": step,
                    "train_loss": train_loss,
                    "train_token_ce": train_ce,
                    "train_norm_loss": train_norm,
                    "gradient_norm": gradient_norm,
                    **{f"val_{name}": metrics[name] for name in METRIC_NAMES},
                    "lr": lr,
                    "wall_s": wall,
                    "unique_tokens_per_s": processed_tokens / wall,
                    "ce_labels_per_s": processed_labels / wall,
                    "peak_vram_bytes": peak_vram,
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(
                    f"step={step:4d} ce={train_ce:.4f} "
                    f"nll={metrics['h1_validation_nll']:.4f} "
                    f"h1_qcos={metrics['h1_quantization_cosine']:.4f} "
                    f"mean_qcos={metrics['mean_quantization_cosine']:.4f} "
                    f"normerr={metrics['mean_decoder_norm_abs_error']:.4f} "
                    f"arcos={metrics['mean_overall_cosine']:.4f} "
                    f"tok_h2-h4={metrics['h2_h4_token_agreement']:.3f}",
                    flush=True,
                )
                save_checkpoint(
                    args.output_dir / "last.pt",
                    model,
                    optimizer,
                    data_generator,
                    step,
                    metrics,
                    batch=args.batch,
                    microbatch=args.microbatch,
                    schedule_steps=args.schedule_steps,
                )
                if step % 1000 == 0 or step == args.steps:
                    save_checkpoint(
                        args.output_dir / f"step{step:04d}.pt",
                        model,
                        optimizer,
                        data_generator,
                        step,
                        metrics,
                        batch=args.batch,
                        microbatch=args.microbatch,
                        schedule_steps=args.schedule_steps,
                    )

    final = rows[-1]
    strict_passed = (
        final["val_h1_validation_nll"] < initial_row["val_h1_validation_nll"]
        and final["val_mean_quantization_cosine"] >= 0.90
        and final["val_mean_decoder_norm_abs_error"] <= 0.10
        and final["val_h2_h4_token_agreement"] >= 0.80
        and final["val_mean_overall_cosine"] >= 0.90
    )
    continuation_baseline = (
        next(row for row in rows if int(row["step"]) == start_step)
        if start_step > 0
        else initial_row
    )
    continuation_checks = {
        "h1_nll_nonworse": (
            final["val_h1_validation_nll"]
            <= continuation_baseline["val_h1_validation_nll"]
        ),
        "h1_quantization_cosine_increased": (
            final["val_h1_quantization_cosine"]
            > continuation_baseline["val_h1_quantization_cosine"]
        ),
        "mean_quantization_cosine_increased": (
            final["val_mean_quantization_cosine"]
            > continuation_baseline["val_mean_quantization_cosine"]
        ),
        "mean_ar_state_cosine_increased": (
            final["val_mean_overall_cosine"]
            > continuation_baseline["val_mean_overall_cosine"]
        ),
        "h2_h4_token_agreement_increased": (
            final["val_h2_h4_token_agreement"]
            > continuation_baseline["val_h2_h4_token_agreement"]
        ),
        "decoder_norm_error_bounded": (
            final["val_mean_decoder_norm_abs_error"] <= 0.10
        ),
    }
    continuation_passed = all(continuation_checks.values())
    with (args.record_dir / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("final_step", args.steps))
        writer.writerow(("strict_absolute_criterion_passed", str(strict_passed).lower()))
        writer.writerow(("continuation_trend_criterion_passed", str(continuation_passed).lower()))
        for name, value in continuation_checks.items():
            writer.writerow((f"continuation_{name}", str(value).lower()))
        writer.writerow(("wall_s", final["wall_s"]))
        writer.writerow(("peak_vram_bytes", int(final["peak_vram_bytes"])))

    strict_verdict = "passed" if strict_passed else "did not pass"
    trend_verdict = "passed" if continuation_passed else "did not pass"
    lines = [
        "# Interpretation",
        "",
        f"The original strict absolute closure criterion {strict_verdict}.",
        f"The preregistered step-{start_step} continuation trend criterion "
        f"{trend_verdict}.",
        "",
        "The continuous recurrence regenerates K after every continuous "
        "output. The AR reference uses only the model's own greedy bytes.",
        "",
        f"| Metric | Step 0 | Step {start_step} | Step {int(final['step'])} |",
        "|---|---:|---:|---:|",
        f"| H1 validation NLL | {initial_row['val_h1_validation_nll']:.6f} | {continuation_baseline['val_h1_validation_nll']:.6f} | {final['val_h1_validation_nll']:.6f} |",
        f"| H1 decoder/codebook cosine | {initial_row['val_h1_quantization_cosine']:.6f} | {continuation_baseline['val_h1_quantization_cosine']:.6f} | {final['val_h1_quantization_cosine']:.6f} |",
        f"| Mean decoder/codebook cosine | {initial_row['val_mean_quantization_cosine']:.6f} | {continuation_baseline['val_mean_quantization_cosine']:.6f} | {final['val_mean_quantization_cosine']:.6f} |",
        f"| Decoder norm absolute error | {initial_row['val_mean_decoder_norm_abs_error']:.6f} | {continuation_baseline['val_mean_decoder_norm_abs_error']:.6f} | {final['val_mean_decoder_norm_abs_error']:.6f} |",
        f"| Continuous/AR state cosine | {initial_row['val_mean_overall_cosine']:.6f} | {continuation_baseline['val_mean_overall_cosine']:.6f} | {final['val_mean_overall_cosine']:.6f} |",
        f"| H2--H4 token agreement | {initial_row['val_h2_h4_token_agreement']:.6f} | {continuation_baseline['val_h2_h4_token_agreement']:.6f} | {final['val_h2_h4_token_agreement']:.6f} |",
        "",
    ]
    (args.record_dir / "interpretation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
