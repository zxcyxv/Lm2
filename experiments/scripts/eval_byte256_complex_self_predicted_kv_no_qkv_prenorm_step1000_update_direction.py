"""Measure raw-gradient alignment with the registered native update 1001."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from train_k1_decoder_inverse_ablation_13m import CLIP_NORM, lr_at
from eval_byte256_complex_self_predicted_kv_no_qkv_prenorm_step1000_one_update_forensics import (
    OUTPUT_ROOT,
    RECORD_ROOT,
    SOURCE_OUTPUT,
    checkpoint_model,
    exact_next_batch,
    gradient_norm,
    load_payload,
    parameter_group,
    run_gradient,
    trainable_parameters,
    write_tsv,
)


SOURCE_STEP = 1000
SCHEDULE_STEPS = 6000
MICROBATCH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--microbatch", type=int, default=MICROBATCH)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def registered_references(record_dir: Path) -> dict[str, object]:
    counterfactuals = read_tsv(record_dir / "counterfactuals.tsv")
    pre = next(
        row
        for row in counterfactuals
        if row["phase"] == "pre_update1001_train_batch"
        and row["condition"] == "attached_raw"
        and row["horizon"] == "0"
        and row["group"] == "all"
    )
    post = next(
        row
        for row in counterfactuals
        if row["phase"] == "post_update1001_same_batch"
        and row["condition"] == "attached_raw"
        and row["horizon"] == "0"
        and row["group"] == "all"
    )
    update_rows = read_tsv(record_dir / "parameter_updates.tsv")
    global_update = next(row for row in update_rows if row["scope"] == "global")
    parameter_updates = {
        row["name"]: float(row["adam_parameter_delta_l2"])
        for row in update_rows
        if row["scope"] == "parameter"
    }
    return {
        "pre_ce": float(pre["loss"]),
        "post_ce": float(post["loss"]),
        "raw_gradient_l2": float(pre["gradient_l2"]),
        "delta_l2": float(global_update["adam_parameter_delta_l2"]),
        "parameter_delta_l2": parameter_updates,
    }


def vector_metrics(
    names: list[str],
    gradients: dict[str, torch.Tensor],
    deltas: dict[str, torch.Tensor],
) -> tuple[int, float, float, float]:
    numel = 0
    gradient_squared = 0.0
    delta_squared = 0.0
    dot = 0.0
    for name in names:
        gradient = gradients[name].double()
        delta = deltas[name].double()
        numel += gradient.numel()
        gradient_squared += float(gradient.square().sum())
        delta_squared += float(delta.square().sum())
        dot += float((gradient * delta).sum())
    return numel, math.sqrt(gradient_squared), math.sqrt(delta_squared), dot


def close_enough(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=2e-5, abs_tol=5e-7)


def main() -> None:
    args = parse_args()
    if args.microbatch < 1 or 64 % args.microbatch:
        raise ValueError("microbatch must divide the native batch of 64")
    if not torch.cuda.is_available():
        raise RuntimeError("update-direction audit requires CUDA")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    references = registered_references(args.record_dir)
    payload = load_payload(SOURCE_OUTPUT / "step1000.pt")
    if int(payload["step"]) != SOURCE_STEP:
        raise RuntimeError("source checkpoint is not native step 1000")
    window, _ = exact_next_batch(payload)
    model = checkpoint_model(payload, device)

    current = run_gradient(
        model,
        window,
        microbatch=args.microbatch,
        detach=False,
        sqrt_reads=False,
    )
    raw_gradient_l2 = gradient_norm(current.gradients)
    parameters = trainable_parameters(model)
    pre_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in parameters.items()
    }

    optimizer = torch.optim.AdamW(
        parameters.values(),
        lr=lr_at(SOURCE_STEP, SCHEDULE_STEPS),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    optimizer.load_state_dict(payload["optimizer"])
    for optimizer_group in optimizer.param_groups:
        optimizer_group["lr"] = lr_at(SOURCE_STEP, SCHEDULE_STEPS)
    for name, parameter in parameters.items():
        parameter.grad = current.gradients[name].to(device).clone()
    clipped_from = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM))
    if not close_enough(clipped_from, raw_gradient_l2):
        raise RuntimeError(
            f"clip norm mismatch: clip={clipped_from}, raw={raw_gradient_l2}"
        )
    optimizer.step()
    post_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in parameters.items()
    }
    deltas = {
        name: post_parameters[name] - pre_parameters[name]
        for name in pre_parameters
    }

    post = run_gradient(
        model,
        window,
        microbatch=args.microbatch,
        detach=False,
        sqrt_reads=False,
    )

    if not close_enough(current.loss, float(references["pre_ce"])):
        raise RuntimeError("pre-update CE did not reproduce the parent audit")
    if not close_enough(post.loss, float(references["post_ce"])):
        raise RuntimeError("post-update CE did not reproduce the parent audit")
    if not close_enough(raw_gradient_l2, float(references["raw_gradient_l2"])):
        raise RuntimeError("raw gradient norm did not reproduce the parent audit")

    expected_parameter_deltas = references["parameter_delta_l2"]
    assert isinstance(expected_parameter_deltas, dict)
    maximum_parameter_delta_relative_error = 0.0
    for name, delta in deltas.items():
        actual = float(delta.double().norm())
        expected = float(expected_parameter_deltas[name])
        relative_error = abs(actual - expected) / max(abs(expected), 1e-30)
        maximum_parameter_delta_relative_error = max(
            maximum_parameter_delta_relative_error, relative_error
        )
        if not close_enough(actual, expected):
            raise RuntimeError(
                f"parameter delta mismatch for {name}: {actual} vs {expected}"
            )

    groups: dict[str, list[str]] = {}
    for name in parameters:
        groups.setdefault(parameter_group(name), []).append(name)
    all_names = list(parameters)
    global_numel, global_gradient_l2, global_delta_l2, global_dot = vector_metrics(
        all_names, current.gradients, deltas
    )
    if not close_enough(global_delta_l2, float(references["delta_l2"])):
        raise RuntimeError("global delta norm did not reproduce the parent audit")

    group_dot_sum = sum(
        vector_metrics(names, current.gradients, deltas)[3]
        for names in groups.values()
    )
    group_dot_sum_relative_error = abs(group_dot_sum - global_dot) / max(
        abs(global_dot), 1e-30
    )
    if group_dot_sum_relative_error > 1e-10:
        raise RuntimeError(
            "parameter-group dot products do not sum to the global dot: "
            f"relative error {group_dot_sum_relative_error}"
        )

    actual_ce_change = post.loss - current.loss

    def row(scope: str, name: str, selected: list[str]) -> dict[str, object]:
        numel, gradient_l2, delta_l2, dot = vector_metrics(
            selected, current.gradients, deltas
        )
        cosine = dot / max(gradient_l2 * delta_l2, 1e-30)
        return {
            "scope": scope,
            "name": name,
            "numel": numel,
            "raw_gradient_l2": gradient_l2,
            "actual_parameter_delta_l2": delta_l2,
            "gradient_dot_actual_delta": dot,
            "predicted_first_order_ce_change": dot,
            "cosine_gradient_actual_delta": cosine,
            "cosine_gradient_negative_delta": -cosine,
            "first_order_direction": (
                "descent" if dot < 0.0 else "ascent" if dot > 0.0 else "neutral"
            ),
            "squared_gradient_fraction": gradient_l2**2
            / max(global_gradient_l2**2, 1e-30),
            "squared_delta_fraction": delta_l2**2
            / max(global_delta_l2**2, 1e-30),
            "dot_fraction_of_global": (
                dot / global_dot if abs(global_dot) > 1e-30 else math.nan
            ),
            "same_batch_ce_before": current.loss,
            "same_batch_ce_after": post.loss,
            "actual_same_batch_ce_change": actual_ce_change,
            "actual_minus_first_order_change": (
                actual_ce_change - dot if scope == "global" else math.nan
            ),
        }

    rows = [row("group", name, groups[name]) for name in sorted(groups)]
    rows.append(row("global", "all", all_names))
    write_tsv(args.record_dir / "update_direction_alignment.tsv", rows)

    global_row = rows[-1]
    strongest_descent = min(rows[:-1], key=lambda item: float(item["gradient_dot_actual_delta"]))
    strongest_ascent = max(rows[:-1], key=lambda item: float(item["gradient_dot_actual_delta"]))
    first_order_label = str(global_row["first_order_direction"])
    interpretation = (
        "# Update-direction alignment\n\n"
        "This audit exactly reproduced the native step-1000 current gradient "
        "and its already registered clipped-AdamW update 1001. `delta` means "
        "the actual parameter change `theta_1001 - theta_1000`; therefore "
        "`g dot delta < 0` is first-order descent. Global clipping is only a "
        "positive scalar rescaling and cannot rotate the gradient.\n\n"
        f"- Same-batch CE: `{current.loss:.9f}` -> `{post.loss:.9f}` "
        f"(actual change `{actual_ce_change:+.9g}`).\n"
        f"- Raw gradient L2: `{global_gradient_l2:.9g}`.\n"
        f"- Actual Adam parameter-delta L2: `{global_delta_l2:.9g}`.\n"
        f"- Global `g dot delta`: `{global_dot:+.9g}` ({first_order_label}).\n"
        f"- Cosine(`g`, `delta`): "
        f"`{float(global_row['cosine_gradient_actual_delta']):+.9g}`; "
        f"cosine(`g`, `-delta`): "
        f"`{float(global_row['cosine_gradient_negative_delta']):+.9g}`.\n"
        f"- Actual minus first-order CE change: "
        f"`{float(global_row['actual_minus_first_order_change']):+.9g}`.\n\n"
        f"The most negative group contribution was "
        f"`{strongest_descent['name']}` "
        f"(`{float(strongest_descent['gradient_dot_actual_delta']):+.9g}`); "
        f"the most positive was `{strongest_ascent['name']}` "
        f"(`{float(strongest_ascent['gradient_dot_actual_delta']):+.9g}`). "
        "Group contributions are disjoint and sum to the global dot product.\n\n"
        + (
            "The actual delta is first-order descent even though the finite "
            "same-batch update increased CE. Thus current-gradient/Adam "
            "direction misalignment does not explain the sign reversal; "
            "higher-order curvature or other finite-step nonlinearity must "
            "dominate along this update. This audit does not identify which."
            if global_dot < 0.0
            else
            "The actual delta is already first-order ascent for the current "
            "batch. Since clipping does not rotate the gradient, this local "
            "misalignment comes from Adam's stored state and coordinate-wise "
            "preconditioning, not clipping alone."
        )
        + "\n\n"
        f"Reproduction checks: maximum per-parameter delta-L2 relative error "
        f"`{maximum_parameter_delta_relative_error:.3g}`; group-dot sum "
        f"relative error `{group_dot_sum_relative_error:.3g}`. This is a "
        "one-update local result, not a historical step-900 or long-run claim.\n"
    )
    (args.record_dir / "update_direction_interpretation.md").write_text(
        interpretation
    )
    print(
        f"CE {current.loss:.9f}->{post.loss:.9f}; "
        f"g_dot_delta={global_dot:+.9g}; "
        f"cos(g,delta)={float(global_row['cosine_gradient_actual_delta']):+.9g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
