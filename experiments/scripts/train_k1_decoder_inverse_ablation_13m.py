"""Run the preregistered 13M K=1 decoder inverse-constraint comparison."""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rotlm.dataset import token_memmap
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM


VOCABULARY = 8192
CONTEXT = 256
SEED = 1337
STEPS = 1000
PEAK_LR = 3e-4
WARMUP = 100
CLIP_NORM = 1.0
REPORT_STEPS = frozenset((1, 50, 100, 250, 500, 750, 1000))
RECORD = Path("experiments/records/EXP-20260721-k1-decoder-inverse-ablation-13m")
OUTPUT = Path("outputs/experiments/EXP-20260721-k1-decoder-inverse-ablation-13m")
VARIANTS = {
    "inverse_cosine": dict(
        width=896, decoder_mode="exact-inverse", head_mode="cosine"
    ),
    "independent_cosine": dict(
        width=736, decoder_mode="independent", head_mode="cosine"
    ),
    "inverse_rms": dict(
        width=896, decoder_mode="exact-inverse", head_mode="rms-tied"
    ),
}


@dataclass(frozen=True)
class Metrics:
    nll: float
    accuracy: float
    state_relative_mse: float
    state_cosine: float
    operator_gradient_norm: float


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def memmap(split: str):
    return token_memmap(split)


def windows(data, starts: torch.Tensor) -> torch.Tensor:
    length = CONTEXT + 1
    values = np.stack(
        [
            np.asarray(data[int(start) : int(start) + length], dtype=np.int64)
            for start in starts
        ]
    )
    return torch.from_numpy(values).cuda(non_blocking=True)


def sampled_windows(data, batch: int, generator: torch.Generator) -> torch.Tensor:
    starts = torch.randint(0, len(data) - CONTEXT - 1, (batch,), generator=generator)
    return windows(data, starts)


def relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-8)
    return (numerator / denominator).mean()


def active_head_self_retrieval(
    model: K1DecoderAblationLM, chunk: int = 256
) -> float:
    weight = model.embedding_weight
    correct = 0
    with torch.inference_mode():
        for begin in range(0, weight.shape[0], chunk):
            end = min(begin + chunk, weight.shape[0])
            logits = model.token_logits(weight[begin:end])
            expected = torch.arange(begin, end, device=weight.device)
            correct += int(logits.argmax(dim=-1).eq(expected).sum())
    return correct / weight.shape[0]


def make_model(variant: str) -> K1DecoderAblationLM:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return K1DecoderAblationLM(
        VOCABULARY,
        encoder_blocks=2,
        independent_decoder_blocks=2,
        **VARIANTS[variant],
    )


def lr_at(index: int, steps: int) -> float:
    if index < WARMUP:
        return PEAK_LR * (index + 1) / WARMUP
    progress = (index - WARMUP) / max(1, steps - WARMUP)
    return PEAK_LR * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def loss_terms(model: K1DecoderAblationLM, window: torch.Tensor):
    inputs, targets = window[:, :-1], window[:, 1:]
    logits, operator_states, _ = model(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
    return loss, logits, operator_states


@torch.inference_mode()
def evaluate(
    model: K1DecoderAblationLM,
    validation,
    starts: torch.Tensor,
    micro: int,
    operator_gradient_norm: float,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    state_mse_sum = 0.0
    state_cos_sum = 0.0
    examples = 0
    for begin in range(0, len(starts), micro):
        window = windows(validation, starts[begin : begin + micro])
        full_positions = torch.arange(window.shape[1], device=window.device)
        full_encoded, _ = model.encode(window, full_positions)
        source = full_encoded[:, :-1]
        hidden, operator_states = model.dense_decode_hidden(
            source, full_positions[:-1]
        )
        logits = model.token_logits(hidden)
        targets = window[:, 1:]
        loss_sum = F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), reduction="sum"
        )
        total_loss += float(loss_sum)
        total_correct += int(logits.argmax(dim=-1).eq(targets).sum())
        total_tokens += targets.numel()
        target_states = full_encoded[:, 1:]
        batch = window.shape[0]
        state_mse_sum += float(relative_mse(operator_states, target_states)) * batch
        state_cos_sum += float(
            F.cosine_similarity(
                operator_states.float(), target_states.float(), dim=-1
            ).mean()
        ) * batch
        examples += batch
    model.train()
    return Metrics(
        nll=total_loss / total_tokens,
        accuracy=total_correct / total_tokens,
        state_relative_mse=state_mse_sum / examples,
        state_cosine=state_cos_sum / examples,
        operator_gradient_norm=operator_gradient_norm,
    )


def preflight(model: K1DecoderAblationLM, training, batch: int) -> dict[str, float]:
    window = sampled_windows(
        training,
        min(batch, 2),
        torch.Generator().manual_seed(SEED + 4242),
    )
    loss, logits, _ = loss_terms(model, window)
    gradient = torch.autograd.grad(loss, model.operator.weight)[0]
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("preflight produced non-finite logits")
    if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm() > 0):
        raise RuntimeError("preflight produced no finite operator gradient")
    inverse_roundtrip_rel_l2 = math.nan
    if model.decoder_mode == "exact-inverse":
        inputs = window[:, :-1]
        encoded, positions = model.encode(inputs)
        anchor = min(63, inputs.shape[1] - 1)
        state = model.operator(encoded[:, anchor])
        tape = torch.cat((encoded[:, : anchor + 1], state[:, None]), dim=1)
        tape_positions = torch.cat(
            (positions[:, : anchor + 1], positions[:, anchor : anchor + 1] + 1),
            dim=1,
        )
        decoded = model.encoder.decode_hidden(tape, tape_positions)
        restored, _ = model.encoder.encode_hidden(decoded, tape_positions)
        inverse_roundtrip_rel_l2 = float(
            relative_mse(restored, tape).sqrt().detach()
        )
        if inverse_roundtrip_rel_l2 >= 1e-4:
            raise RuntimeError(
                "exact inverse preflight failed: "
                f"relative L2={inverse_roundtrip_rel_l2:.3e}"
            )
    values = {
        "initial_nll": float(loss),
        "initial_k_gradient_norm": float(gradient.norm()),
        "head_self_retrieval": active_head_self_retrieval(model),
        "inverse_roundtrip_rel_l2": inverse_roundtrip_rel_l2,
    }
    model.zero_grad(set_to_none=True)
    return values


def save_checkpoint(
    path: Path, model, optimizer, step: int, metrics: Metrics, variant: str
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": asdict(metrics),
            "config": {
                "variant": variant,
                "vocabulary": VOCABULARY,
                "context": CONTEXT,
                "seed": SEED,
                **VARIANTS[variant],
            },
        },
        path,
    )


def train_variant(args, variant: str, training, validation, validation_starts):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = make_model(variant).cuda().train()
    preflight_values = preflight(model, training, args.batch)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    print(
        f"[{variant}] params={total_parameters:,} trainable={trainable_parameters:,} "
        f"head_identity={preflight_values['head_self_retrieval']:.6f} "
        f"initial_nll={preflight_values['initial_nll']:.4f}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    record_dir = RECORD / variant
    output_dir = OUTPUT / variant
    record_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (record_dir / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in (
            ("variant", variant),
            ("seed", SEED),
            ("steps", args.steps),
            ("batch", args.batch),
            ("context", CONTEXT),
            ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("precision", "float32_tf32_matmul"),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    fields = (
        "step",
        "train_nll",
        *tuple(f"val_{key}" for key in asdict(Metrics(0, 0, 0, 0, 0))),
        "lr",
        "wall_s",
        "tokens_per_s",
        "peak_vram_bytes",
    )
    data_generator = torch.Generator().manual_seed(SEED)
    best_nll = math.inf
    best_step = 0
    started = time.time()
    processed_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    with (record_dir / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(args.steps):
            lr = lr_at(index, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(training, args.batch, data_generator)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_terms(model, window)
            loss.backward()
            operator_gradient_norm = float(model.operator.weight.grad.float().norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            processed_tokens += args.batch * CONTEXT
            step = index + 1
            if step in REPORT_STEPS or step == args.steps:
                metrics = evaluate(
                    model,
                    validation,
                    validation_starts,
                    args.eval_micro,
                    operator_gradient_norm,
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    "train_nll": float(loss),
                    **{f"val_{key}": value for key, value in asdict(metrics).items()},
                    "lr": lr,
                    "wall_s": wall,
                    "tokens_per_s": processed_tokens / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"[{variant}] step={step:4d} train={float(loss):.4f} "
                    f"val={metrics.nll:.4f} acc={metrics.accuracy:.4f} "
                    f"state_mse={metrics.state_relative_mse:.4f} "
                    f"wall={wall:.1f}s",
                    flush=True,
                )
                save_checkpoint(
                    output_dir / "last.pt", model, optimizer, step, metrics, variant
                )
                if metrics.nll < best_nll:
                    best_nll = metrics.nll
                    best_step = step
                    save_checkpoint(
                        output_dir / "best.pt", model, optimizer, step, metrics, variant
                    )
    return {
        "variant": variant,
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "best_step": best_step,
        "best_val_nll": best_nll,
        "final_head_self_retrieval": active_head_self_retrieval(model),
        "wall_s": time.time() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS)
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-micro", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_examples, args.eval_micro) < 1:
        parser.error("steps and batch sizes must be positive")
    if not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    training, validation = memmap("train"), memmap("validation")
    validation_starts = torch.randint(
        0,
        len(validation) - CONTEXT - 1,
        (args.eval_examples,),
        generator=torch.Generator().manual_seed(SEED + 999),
    )
    RECORD.mkdir(parents=True, exist_ok=True)
    with (RECORD / "validation_starts.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "token_start"))
        writer.writerows(enumerate(validation_starts.tolist()))

    summaries = []
    for variant in args.variants:
        summaries.append(
            train_variant(args, variant, training, validation, validation_starts)
        )
        torch.cuda.empty_cache()
    with (RECORD / "summary.tsv").open("w", newline="") as handle:
        fields = tuple(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
