"""Run the preregistered joint stop-gradient state MSE + token CE, K=1, 13M."""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.models.query_k1_inverse import QueryK1InverseLM
from train_k1_decoder_inverse_ablation_13m import (
    CLIP_NORM,
    CONTEXT,
    PEAK_LR,
    REPORT_STEPS,
    SEED,
    VOCABULARY,
    active_head_self_retrieval,
    lr_at,
    memmap,
    parameter_count,
    sampled_windows,
    trainable_parameter_count,
    windows,
)


STEPS = 1000
WIDTH = 896
RECORD = Path("experiments/records/EXP-20260722-query-k1-mse-ce-joint-13m")
OUTPUT = Path("outputs/experiments/EXP-20260722-query-k1-mse-ce-joint-13m")


@dataclass(frozen=True)
class Metrics:
    nll: float
    accuracy: float
    combined_loss: float
    ce_loss: float
    mse_loss: float
    gold_state_relative_mse: float
    gold_state_cosine: float
    action_closure_relative_mse: float
    action_closure_cosine: float
    operator_gradient_norm: float
    query_gradient_norm: float


def make_model() -> QueryK1InverseLM:
    return QueryK1InverseLM(
        VOCABULARY, width=WIDTH, encoder_blocks=2,
        head_mode="rms-tied", trainable_cosine_scale=True,
    )


def relative_mse_rows(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - target.float()).square().sum(dim=-1)
    denominator = target.float().square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def gold_next_states(model: QueryK1InverseLM, window: torch.Tensor) -> torch.Tensor:
    """Stop-gradient h_(t+1) from the SAME online encoder over the real
    gold continuation (no separate EMA copy)."""
    full_positions = torch.arange(window.shape[1], device=window.device)
    with torch.no_grad():
        full_encoded, _ = model.encode(window, full_positions)
    return full_encoded[:, 1:]


def loss_terms(model: QueryK1InverseLM, window: torch.Tensor):
    inputs, targets = window[:, :-1], window[:, 1:]
    output = model(inputs)
    ce = F.cross_entropy(output.logits.flatten(0, 1).float(), targets.flatten())
    gold = gold_next_states(model, window)
    # Relative (target-norm-normalized) MSE, matching this project's existing
    # gold_state_relative_mse diagnostic, instead of raw unnormalized squared
    # L2. Raw L2 sits on a wildly different scale than CE (hundreds vs ~9
    # nats) and rewards inflating hidden-state norm rather than direction;
    # normalizing keeps both terms O(1) and removes that degenerate incentive.
    mse = relative_mse_rows(output.operator_states, gold).mean()
    loss = ce + mse
    return loss, ce, mse, output, targets


@torch.inference_mode()
def evaluate(
    model: QueryK1InverseLM,
    validation,
    starts: torch.Tensor,
    micro: int,
    operator_gradient_norm: float,
    query_gradient_norm: float,
) -> Metrics:
    model.eval()
    total_loss = total_correct = total_tokens = 0
    combined_sum = ce_sum = mse_sum = 0.0
    gold_state_mse_sum = gold_state_cos_sum = 0.0
    action_mse_sum = action_cos_sum = 0.0
    for begin in range(0, len(starts), micro):
        window = windows(validation, starts[begin: begin + micro])
        inputs, targets = window[:, :-1], window[:, 1:]
        output = model(inputs)
        ce_row = F.cross_entropy(
            output.logits.flatten(0, 1).float(), targets.flatten(), reduction="sum",
        )
        gold = gold_next_states(model, window)
        mse_rows = relative_mse_rows(output.operator_states, gold)
        total_loss += float(ce_row)
        combined_sum += float(ce_row) + float(mse_rows.sum())
        ce_sum += float(ce_row)
        mse_sum += float(mse_rows.sum())
        actions = output.logits.argmax(dim=-1)
        total_correct += int(actions.eq(targets).sum())
        total_tokens += targets.numel()

        gold_state_mse_sum += float(relative_mse_rows(output.operator_states, gold).sum())
        gold_state_cos_sum += float(
            F.cosine_similarity(output.operator_states.float(), gold.float(), dim=-1).sum()
        )
        action_states = model.dense_action_encode(inputs, actions, output.positions)
        action_mse = relative_mse_rows(output.operator_states, action_states)
        action_cos = F.cosine_similarity(
            output.operator_states.float(), action_states.float(), dim=-1,
        )
        action_mse_sum += float(action_mse.sum())
        action_cos_sum += float(action_cos.sum())

    model.train()
    return Metrics(
        nll=total_loss / total_tokens,
        accuracy=total_correct / total_tokens,
        combined_loss=combined_sum / total_tokens,
        ce_loss=ce_sum / total_tokens,
        mse_loss=mse_sum / total_tokens,
        gold_state_relative_mse=gold_state_mse_sum / total_tokens,
        gold_state_cosine=gold_state_cos_sum / total_tokens,
        action_closure_relative_mse=action_mse_sum / total_tokens,
        action_closure_cosine=action_cos_sum / total_tokens,
        operator_gradient_norm=operator_gradient_norm,
        query_gradient_norm=query_gradient_norm,
    )


def preflight(model: QueryK1InverseLM, training, batch: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(SEED + 4242)
    window = sampled_windows(training, min(batch, 2), generator)
    loss, ce, mse, output, _ = loss_terms(model, window)
    if not bool(torch.isfinite(output.logits).all()):
        raise RuntimeError("preflight produced non-finite logits")

    ce_k_grad, ce_q_grad = torch.autograd.grad(
        ce, (model.operator.weight, model.query_embedding), retain_graph=True,
    )
    mse_k_grad, mse_q_grad = torch.autograd.grad(
        mse, (model.operator.weight, model.query_embedding), retain_graph=False,
    )
    for name, gradient in (
        ("ce->K", ce_k_grad), ("ce->query", ce_q_grad),
        ("mse->K", mse_k_grad), ("mse->query", mse_q_grad),
    ):
        if not bool(torch.isfinite(gradient).all()) or not bool(gradient.norm() > 0):
            raise RuntimeError(f"preflight produced no finite {name} gradient")

    with torch.no_grad():
        anchor = min(63, window.shape[1] - 2)
        tape = torch.cat(
            (output.prefix_states[:, : anchor + 1], output.operator_states[:, anchor: anchor + 1]),
            dim=1,
        )
        tape_positions = torch.arange(anchor + 2, device=window.device).unsqueeze(0).expand(
            window.shape[0], -1,
        )
        decoded = model.encoder.decode_hidden(tape, tape_positions)
        restored, _ = model.encoder.encode_hidden(decoded, tape_positions)
        inverse_roundtrip_rel_l2 = float(
            relative_mse_rows(restored, tape).mean().sqrt()
        )
    if inverse_roundtrip_rel_l2 >= 1e-4:
        raise RuntimeError(
            f"exact inverse failed: relative L2={inverse_roundtrip_rel_l2:.3e}"
        )

    values = {
        "initial_combined_loss": float(loss),
        "initial_ce": float(ce),
        "initial_mse": float(mse),
        "initial_ce_k_gradient_norm": float(ce_k_grad.float().norm()),
        "initial_mse_k_gradient_norm": float(mse_k_grad.float().norm()),
        "initial_ce_query_gradient_norm": float(ce_q_grad.float().norm()),
        "initial_mse_query_gradient_norm": float(mse_q_grad.float().norm()),
        "head_self_retrieval": active_head_self_retrieval(model),
        "inverse_roundtrip_rel_l2": inverse_roundtrip_rel_l2,
    }
    model.zero_grad(set_to_none=True)
    return values


def save_checkpoint(path: Path, model, optimizer, step, metrics):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": asdict(metrics),
            "config": {
                "vocabulary": VOCABULARY,
                "context": CONTEXT,
                "seed": SEED,
                "width": WIDTH,
                "encoder_blocks": 2,
                "head_mode": "rms-tied",
                "objective": "ce_plus_stopgrad_state_mse",
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--eval-micro", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_examples, args.eval_micro) < 1:
        parser.error("steps and batch sizes must be positive")
    if not torch.cuda.is_available():
        parser.error("this experiment requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    training, validation = memmap("train"), memmap("validation")
    validation_starts = torch.randint(
        0, len(validation) - CONTEXT - 1, (args.eval_examples,),
        generator=torch.Generator().manual_seed(SEED + 999),
    )
    RECORD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (RECORD / "validation_starts.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "token_start"))
        writer.writerows(enumerate(validation_starts.tolist()))

    model = make_model().cuda().train()
    preflight_values = preflight(model, training, args.batch)
    total_parameters = parameter_count(model)
    trainable_parameters = trainable_parameter_count(model)
    print(
        f"params={total_parameters:,} trainable={trainable_parameters:,} "
        f"head_identity={preflight_values['head_self_retrieval']:.6f} "
        f"initial_ce={preflight_values['initial_ce']:.4f} "
        f"initial_mse={preflight_values['initial_mse']:.4f} "
        f"inverse_rt={preflight_values['inverse_roundtrip_rel_l2']:.2e}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95),
        weight_decay=0.0, fused=True,
    )
    with (RECORD / "run.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        for key, value in (
            ("seed", SEED), ("steps", args.steps), ("batch", args.batch),
            ("context", CONTEXT), ("parameters", total_parameters),
            ("trainable_parameters", trainable_parameters),
            ("precision", "strict_float32_no_tf32"),
            ("objective", "ce_plus_stopgrad_state_mse_equal_weight"),
            ("k_input", "contextualized_next_position_query"),
            *preflight_values.items(),
        ):
            writer.writerow((key, value))

    fields = (
        "step", "train_combined", "train_ce", "train_mse",
        *tuple(f"val_{key}" for key in asdict(Metrics(*(0.0,) * 11))),
        "lr", "wall_s", "tokens_per_s", "peak_vram_bytes",
    )
    data_generator = torch.Generator().manual_seed(SEED)
    best_nll, best_step = math.inf, 0
    started = time.time()
    processed_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    with (RECORD / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(args.steps):
            lr = lr_at(index, args.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            window = sampled_windows(training, args.batch, data_generator)
            optimizer.zero_grad(set_to_none=True)
            loss, ce, mse, _, _ = loss_terms(model, window)
            loss.backward()
            operator_gradient_norm = float(model.operator.weight.grad.float().norm())
            query_gradient_norm = float(model.query_embedding.grad.float().norm())
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            processed_tokens += args.batch * CONTEXT
            step = index + 1
            if step in REPORT_STEPS or step == args.steps:
                metrics = evaluate(
                    model, validation, validation_starts, args.eval_micro,
                    operator_gradient_norm, query_gradient_norm,
                )
                wall = time.time() - started
                row = {
                    "step": step,
                    "train_combined": float(loss),
                    "train_ce": float(ce),
                    "train_mse": float(mse),
                    **{f"val_{key}": value for key, value in asdict(metrics).items()},
                    "lr": lr,
                    "wall_s": wall,
                    "tokens_per_s": processed_tokens / wall,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"step={step:4d} train_ce={float(ce):.4f} train_mse={float(mse):.4f} "
                    f"val_nll={metrics.nll:.4f} acc={metrics.accuracy:.4f} "
                    f"gold_mse={metrics.gold_state_relative_mse:.4f} "
                    f"action_mse={metrics.action_closure_relative_mse:.4f} "
                    f"wall={wall:.1f}s",
                    flush=True,
                )
                save_checkpoint(OUTPUT / "last.pt", model, optimizer, step, metrics)
                if metrics.nll < best_nll:
                    best_nll, best_step = metrics.nll, step
                    save_checkpoint(OUTPUT / "best.pt", model, optimizer, step, metrics)

    with (RECORD / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerow(("best_step", best_step))
        writer.writerow(("best_val_nll", best_nll))
        writer.writerow(("wall_s", time.time() - started))
        writer.writerow(("peak_vram_bytes", torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    main()
