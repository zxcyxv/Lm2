"""Self-contained trainer for the *correct* parallel-scan central recurrence
-- model, data loading, and training loop in one file, every hyperparameter
on the command line. Defaults reproduce the 121M run this file replaces
(see below): just run with no flags to repeat it on another GPU.

Architecture: exactly the byte256 state-dependent lineage's encoder
(K1DecoderAblationLM, reversible causal encoder / exact-inverse decoder,
``unitary-branch-normalized-residual`` recurrent hidden mode) and central
operator (rotlm.models.complex_self_prediction.ComplexSelfPredictedKVTransition
-- complex unitary KV memory), but rolled out with
``rollout_time_varying_scan`` instead of the default serial ``rollout``:
Q/K/V are computed from a *forcing tape* ``c_h = R^h z_0`` (a fixed rotation
power of the root state, computed once, not from the evolving successor
state), so the whole central recurrence factors into two associative
prefix scans (a complex-memory scan, then a latent-state scan) instead of
H sequential steps. This is exactly the architecture specified in
parallel_scan_central_recurrence.md, and it is provably behaviorally
identical to the serial recurrence at horizon 1 (checked below at every
run) and near-identical in practice out to H=16 (see
EXP-20260803-byte256-unitary-time-varying-scan-h16-1000step-rtx5090 vs the
serial EXP-20260802 baseline: H1 NLL 2.034 vs 2.049, H16 NLL 3.145 vs
3.148 at step 1000, 13M).

This replaces train_prefix_orthogonal_scan.py, which implemented an
unrelated, simpler architecture (PrefixConditionedOrthogonalAffineScan --
no complex memory, params don't match the state-dependent lineage at all)
that does not match parallel_scan_central_recurrence.md.

Default flags reproduce the 121M comparison run (width 4352 -> 121,336,064
parameters, exactly matching the byte256 state-dependent 121M count; peak
LR cut to 1/3 of the 13M baseline's 3e-4; effective batch 64 via
microbatch 16 x4 accumulation, sized for a 20GB card -- raise
--microbatch if you have more VRAM, e.g. a single microbatch=64 forward
fit on the original RTX 5090 run with room to spare):

    python train_time_varying_scan.py

To reproduce the 13M baseline instead:

    python train_time_varying_scan.py --width 1344 --peak-lr 3e-4 \\
        --microbatch 64
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rotlm.dataset import token_memmap
from rotlm.models.complex_self_prediction import ComplexSelfPredictedKVTransition
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.training.ha_skew_window import (
    evaluate_sparse_complex_self_predicted_kv_ce_only,
    sparse_complex_self_predicted_kv_ce_only_fast_logits,
    sparse_complex_self_predicted_kv_ce_only_fast_loss,
)


VOCABULARY = 256
DATA_ROOT = Path("data/wikitext103_bytes")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def memmap(split: str):
    return token_memmap(split, root=DATA_ROOT)


def windows_of_length(data, starts: torch.Tensor, length: int) -> torch.Tensor:
    values = np.stack(
        [
            np.asarray(data[int(s) : int(s) + length], dtype=np.int64)
            for s in starts.tolist()
        ]
    )
    return torch.from_numpy(values).cuda(non_blocking=True)


def sampled_windows(
    data, batch: int, generator: torch.Generator, length: int
) -> torch.Tensor:
    starts = torch.randint(0, len(data) - length, (batch,), generator=generator)
    return windows_of_length(data, starts, length)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def make_model(args: argparse.Namespace) -> K1DecoderAblationLM:
    model = K1DecoderAblationLM(
        VOCABULARY,
        width=args.width,
        encoder_blocks=args.encoder_blocks,
        decoder_mode="exact-inverse",
        head_mode="simplex-raw-tied",
        simplex_logit_scale=args.simplex_logit_scale,
    )
    model.operator = nn.Identity()
    model.complex_self_prediction = ComplexSelfPredictedKVTransition(
        args.width,
        heads=args.heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        initial_frequency_range=args.initial_frequency_range,
        learned_read_beta=False,
        residual_prior=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        real_value=False,
        normalize_read_by_memory_norm=not args.no_read_norm,
        qkv_prenorm=not args.no_qkv_prenorm,
        scan_memory_decay=args.memory_decay,
        scan_memory_decay_init=args.memory_decay_init,
        scan_memory_decay_max=args.memory_decay_max,
        state_dependent_memory_decay=False,
        memory_decay_dt_min=0.05,
        memory_decay_dt_max=0.5,
        pre_normalize_qkv_inputs=True,
        normalize_initial_recurrent_root=False,
        normalize_accumulated_reads=False,
        post_normalize_hidden_reads=False,
        post_normalize_memory=False,
        post_normalize_recurrent_state=False,
        post_norm_eps=1e-6,
        shared_kv_heads=args.shared_kv_heads,
        fuse_branch_memory_kernel=False,
    )
    return model


def configure_scan_backend(model: K1DecoderAblationLM, backend: str) -> None:
    transition = model.complex_self_prediction
    transition.fuse_time_varying_memory_scan = False
    transition.fuse_time_varying_hidden_scan = False
    transition.use_triton_time_varying_scan = False
    transition.use_rotating_frame_time_varying_scan = False
    transition.use_fused_triton_rotating_frame_time_varying_scan = False
    if backend == "eager":
        return
    if backend == "generated-associative":
        transition.fuse_time_varying_memory_scan = True
        transition.fuse_time_varying_hidden_scan = True
        return
    if backend == "triton-associative":
        transition.use_triton_time_varying_scan = True
        return
    if backend == "rotating-frame-cumsum":
        transition.use_rotating_frame_time_varying_scan = True
        return
    if backend == "fused-triton-rotating-frame":
        transition.use_fused_triton_rotating_frame_time_varying_scan = True
        return
    raise ValueError(f"unknown scan backend: {backend}")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# --------------------------------------------------------------------------
# Schedule / objective / correctness check
# --------------------------------------------------------------------------

def lr_at(index: int, steps: int, *, peak_lr: float, warmup: int) -> float:
    if index < warmup:
        return peak_lr * (index + 1) / warmup
    progress = (index - warmup) / max(1, steps - warmup)
    return peak_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def scan_loss(model: K1DecoderAblationLM, window: torch.Tensor, args) -> torch.Tensor:
    return sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window,
        prefix_length=args.context,
        horizons=args.horizons,
        anchor_stride=args.anchor_stride,
        central_rollout="time-varying-scan",
    )


def scan_metrics(model, windows, microbatch, args) -> dict[str, float]:
    return evaluate_sparse_complex_self_predicted_kv_ce_only(
        model,
        windows,
        prefix_length=args.context,
        horizons=args.horizons,
        anchor_stride=args.anchor_stride,
        central_rollout="time-varying-scan",
        microbatch=microbatch,
    )


def h1_feedback_equivalence(model, training, args) -> float:
    """The forcing-tape scan must reproduce horizon-1 feedback exactly --
    at H=1 the two rollouts are the same computation by construction."""
    window = sampled_windows(
        training,
        2,
        torch.Generator().manual_seed(args.seed + 8303),
        args.context + 1,
    )
    with torch.inference_mode():
        feedback, feedback_targets = sparse_complex_self_predicted_kv_ce_only_fast_logits(
            model,
            window,
            prefix_length=args.context,
            horizons=1,
            anchor_stride=args.anchor_stride,
            central_rollout="feedback",
        )
        scanned, scanned_targets = sparse_complex_self_predicted_kv_ce_only_fast_logits(
            model,
            window,
            prefix_length=args.context,
            horizons=1,
            anchor_stride=args.anchor_stride,
            central_rollout="time-varying-scan",
        )
    if not torch.equal(feedback_targets, scanned_targets):
        raise RuntimeError("H1 scan and feedback targets differ")
    return float((feedback.float() - scanned.float()).abs().max())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--width", type=int, default=4352)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=31)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)
    parser.add_argument(
        "--shared-kv-heads",
        action="store_true",
        help="Mamba multi-value-attention layout: project one query/key "
        "shared across all heads (width*key_dim for the layer) instead of "
        "one per head. Only the value stays per head.",
    )

    parser.add_argument(
        "--memory-decay",
        action="store_true",
        help="contract the forcing-tape memory scan by a learned per-(head,"
        "key) lambda instead of transporting it on the unit circle. With the "
        "QKV prenorm bounding every write the state is then bounded by "
        "B/(1-lambda) uniformly in the horizon count.",
    )
    parser.add_argument("--memory-decay-init", type=float, default=0.95)
    parser.add_argument("--memory-decay-max", type=float, default=0.999)
    parser.add_argument(
        "--no-qkv-prenorm",
        action="store_true",
        help="feed the forcing tape to Q/K/V raw instead of RMS-normalizing "
        "it first. The branch-normalized hidden mode hardwires that "
        "normalization, so --pre-normalize flags cannot reach it.",
    )
    parser.add_argument(
        "--no-read-norm",
        action="store_true",
        help="drop the ||S||_F division and leave the memory read linear. "
        "Only bounded when the memory itself contracts, so pair it with "
        "--memory-decay.",
    )

    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument(
        "--scan-backend",
        choices=(
            "eager",
            "generated-associative",
            "triton-associative",
            "rotating-frame-cumsum",
            "fused-triton-rotating-frame",
        ),
        default="eager",
    )

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64, help="effective batch")
    parser.add_argument(
        "--microbatch",
        type=int,
        default=16,
        help="physical batch per forward/backward; gradients accumulate "
        "over batch/microbatch steps. 16 is sized for a 20GB card; the "
        "original run used a single microbatch=64 pass on an RTX 5090.",
    )
    parser.add_argument("--schedule-steps", type=int, default=6000)
    parser.add_argument(
        "--peak-lr",
        type=float,
        default=1e-4,
        help="1e-4 = the 13M baseline's 3e-4 cut to 1/3 for this 121M run",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument(
        "--record-dir", type=Path, default=Path("outputs/time_varying_scan")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/time_varying_scan/checkpoints"),
    )
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    if args.width < 2 or args.width % 16:
        parser.error("--width must be positive and divisible by 16")
    if args.batch % args.microbatch:
        parser.error("--batch must be divisible by --microbatch")
    if args.horizons < 1:
        parser.error("--horizons must be positive")
    if not torch.cuda.is_available():
        parser.error("this trainer requires CUDA")
    return args


def main() -> None:
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    training, validation = memmap("train"), memmap("validation")
    window_length = args.context + args.horizons
    validation_starts = torch.randint(
        0,
        len(validation) - window_length,
        (args.eval_examples,),
        generator=torch.Generator().manual_seed(args.seed + 999),
    )
    validation_windows = windows_of_length(validation, validation_starts, window_length)

    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args).cuda().train()
    configure_scan_backend(model, args.scan_backend)
    h1_error = h1_feedback_equivalence(model, training, args)
    if h1_error > 5e-5:
        raise RuntimeError(f"H1 scan/feedback equivalence failed: {h1_error}")

    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    accumulation_steps = args.batch // args.microbatch
    print(
        f"width={args.width} params={total_parameters:,} "
        f"trainable={trainable_parameters:,} horizons={args.horizons} "
        f"anchor_stride={args.anchor_stride} "
        f"batch={args.batch} micro={args.microbatch}x{accumulation_steps} "
        f"peak_lr={args.peak_lr:g} warmup={args.warmup} "
        f"h1_scan_feedback_error={h1_error:.3e}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )

    with (args.record_dir / "config.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in sorted(vars(args).items()):
            writer.writerow((key, value))
        writer.writerow(("parameters", total_parameters))
        writer.writerow(("trainable_parameters", trainable_parameters))
        writer.writerow(("h1_scan_feedback_max_error", h1_error))

    horizon_metric_names = tuple(
        f"h{h}_{m}" for h in range(1, args.horizons + 1) for m in ("nll", "accuracy")
    )
    metric_names = ("block_nll", "block_accuracy", *horizon_metric_names)
    fields = (
        "step",
        "train_ce",
        "gradient_norm",
        "phase_gradient_norm",
        "lr",
        *(f"val_{name}" for name in metric_names),
        "wall_s",
        "eta_s",
        "peak_vram_bytes",
    )

    data_generator = torch.Generator().manual_seed(args.seed)
    best_nll = math.inf
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    initial_metrics = scan_metrics(model, validation_windows, args.eval_microbatch, args)
    print(
        f"step=0/{args.steps} block_nll={initial_metrics['block_nll']:.4f} "
        f"h1={initial_metrics['h1_nll']:.4f} "
        f"h{args.horizons}={initial_metrics[f'h{args.horizons}_nll']:.4f}",
        flush=True,
    )

    with (args.record_dir / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(args.steps):
            lr = lr_at(index, args.schedule_steps, peak_lr=args.peak_lr, warmup=args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr

            window = sampled_windows(training, args.batch, data_generator, window_length)
            optimizer.zero_grad(set_to_none=True)
            micro_windows = window.split(args.microbatch)
            train_ce = 0.0
            for micro_window in micro_windows:
                micro_loss = scan_loss(model, micro_window, args)
                if not bool(torch.isfinite(micro_loss)):
                    raise RuntimeError(f"non-finite loss at step {index + 1}")
                (micro_loss / len(micro_windows)).backward()
                train_ce += float(micro_loss.detach()) / len(micro_windows)

            # Measured before clipping: hidden_phase carried 93% of the 121M
            # run's squared gradient norm, so the raw split is what tells us
            # whether a layout change actually moved the pathology.
            phase_gradient_norm = float(
                model.complex_self_prediction.hidden_phase.grad.norm()
            )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            )
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient at step {index + 1}")
            optimizer.step()
            step = index + 1

            report = (
                step in (1, 50)
                or step % args.report_every == 0
                or (step > 1000 and step % max(args.report_every, 500) == 0)
                or step == args.steps
            )
            if report:
                metrics = scan_metrics(model, validation_windows, args.eval_microbatch, args)
                wall = time.time() - started
                eta = (wall / step) * (args.steps - step)
                row = {
                    "step": step,
                    "train_ce": train_ce,
                    "gradient_norm": gradient_norm,
                    "phase_gradient_norm": phase_gradient_norm,
                    "lr": lr,
                    **{f"val_{name}": metrics[name] for name in metric_names},
                    "wall_s": wall,
                    "eta_s": eta,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d}/{args.steps} ce={train_ce:.4f} "
                    f"gnorm={gradient_norm:.4f} "
                    f"pgnorm={phase_gradient_norm:.4f} "
                    f"block_nll={metrics['block_nll']:.4f} "
                    f"h1={metrics['h1_nll']:.4f} "
                    f"h{args.horizons}={metrics[f'h{args.horizons}_nll']:.4f} "
                    f"elapsed={wall/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
                if args.checkpoint_every and (
                    step % args.checkpoint_every == 0 or step == args.steps
                ):
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "metrics": metrics,
                            "args": vars(args),
                        },
                        args.output_dir / f"step{step:05d}.pt",
                    )
                if metrics["block_nll"] < best_nll:
                    best_nll = metrics["block_nll"]
                    torch.save(
                        {"model": model.state_dict(), "step": step, "metrics": metrics},
                        args.output_dir / "best.pt",
                    )

    print(f"done best_val_block_nll={best_nll:.4f}", flush=True)


if __name__ == "__main__":
    main()
