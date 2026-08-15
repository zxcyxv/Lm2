"""Replay step 100 and measure every central-operation backward VJP gain."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.models.complex_self_prediction import (
    ComplexMemoryState,
    rotate_pairwise,
)
import train_byte256_detached_postnorm_raw_read_h4_step100_gradient_localization_13m as run


base = run.base
LOSSES = ("H1", "H2", "H3", "H4", "mean_h1_h4")
DEFAULT_CHECKPOINT = run.DEFAULT_OUTPUT / "step0100.pt"


def tensor_stats(value: torch.Tensor | None) -> tuple[float, float, int, bool]:
    if value is None:
        return 0.0, 0.0, 0, True
    magnitude = value.detach().abs().float()
    real_dof = value.numel() * (2 if value.is_complex() else 1)
    l2 = float(torch.linalg.vector_norm(magnitude))
    return l2, l2 / math.sqrt(real_dof), real_dof, bool(
        torch.isfinite(magnitude).all()
    )


def combined_stats(values: list[torch.Tensor | None]) -> tuple[float, float, int, bool]:
    stats = [tensor_stats(value) for value in values if value is not None]
    l2 = math.sqrt(sum(item[0] ** 2 for item in stats))
    dof = sum(item[2] for item in stats)
    return l2, (l2 / math.sqrt(dof) if dof else 0.0), dof, all(
        item[3] for item in stats
    )


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def add_gain(
    rows: list[dict],
    *,
    loss: str,
    step: int,
    operation: str,
    output_gradients: list[torch.Tensor | None],
    input_gradients: list[torch.Tensor | None],
    category: str = "local_vjp",
) -> None:
    output_l2, output_rms, output_dof, output_finite = combined_stats(
        output_gradients
    )
    input_l2, input_rms, input_dof, input_finite = combined_stats(
        input_gradients
    )
    rows.append(
        {
            "loss_source": loss,
            "recurrent_step": step,
            "operation": operation,
            "category": category,
            "output_adjoint_l2": output_l2,
            "input_adjoint_l2": input_l2,
            "local_l2_gain": input_l2 / output_l2 if output_l2 else "nan",
            "output_adjoint_rms": output_rms,
            "input_adjoint_rms": input_rms,
            "local_rms_gain": input_rms / output_rms if output_rms else "nan",
            "output_real_dof": output_dof,
            "input_real_dof": input_dof,
            "finite": int(output_finite and input_finite),
        }
    )


def local_vjp(
    output: torch.Tensor,
    input_value: torch.Tensor,
    output_gradient: torch.Tensor | None,
) -> torch.Tensor | None:
    if output_gradient is None:
        return None
    return torch.autograd.grad(
        output,
        input_value,
        grad_outputs=output_gradient,
        retain_graph=True,
        allow_unused=True,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--record-dir", type=Path, default=run.DEFAULT_RECORD)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--diagnostic-seed", type=int, default=base.SEED + 43_100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != 100:
        raise RuntimeError("expected step-100 checkpoint")
    model = base.make_model().cuda().train()
    model.load_state_dict(checkpoint["model"], strict=True)
    transition = model.complex_self_prediction

    generator = torch.Generator().manual_seed(args.diagnostic_seed)
    window = base.sampled_windows(
        base.memmap("train"), args.batch, generator, base.TRAIN_WINDOW_LENGTH
    ).cuda()
    positions = torch.arange(base.CONTEXT, device="cuda")
    with torch.no_grad():
        encoded_value, _ = model.encode(window[:, : base.CONTEXT], positions)
    prefix = encoded_value.detach().requires_grad_(True)
    anchors = torch.arange(
        0, base.CONTEXT, base.TRAIN_ANCHOR_STRIDE, device="cuda"
    )
    roots = prefix.index_select(1, anchors)
    offsets = torch.arange(1, 5, device="cuda")
    target_positions = anchors[:, None] + offsets[None, :]
    targets = window.index_select(1, target_positions.reshape(-1)).reshape(
        args.batch, anchors.numel(), 4
    )

    initial = transition.initialize(roots)
    initial_memory = initial.memory.detach().requires_grad_(True)
    state = ComplexMemoryState(
        hidden=initial.hidden,
        memory=initial_memory,
        write_count=initial.write_count,
    )
    packed_weight = transition.packed_qkv_weight()
    step_records = []
    for recurrent_step in range(1, 5):
        predecessor_hidden = state.hidden
        external_hidden = predecessor_hidden.detach().requires_grad_(True)
        hidden_angles = transition.hidden_phase.expand(
            *external_hidden.shape[:-1], -1
        )
        rotated_hidden = rotate_pairwise(external_hidden, hidden_angles)
        memory_multiplier = torch.polar(
            torch.ones_like(transition.memory_phase), -transition.memory_phase
        )
        predecessor_memory = state.memory
        rotated_memory = predecessor_memory * memory_multiplier[..., None]
        query, key, value = transition.project_qkv(rotated_hidden, packed_weight)
        write = torch.einsum("...hk,...hv->...hkv", key, value.conj())
        updated_memory = rotated_memory + write
        complex_read = transition.read(query, updated_memory)
        read_flat = torch.cat(
            (
                complex_read.real.flatten(-2),
                complex_read.imag.flatten(-2),
            ),
            dim=-1,
        )
        delta = transition.output(read_flat)
        raw_hidden = rotated_hidden + delta
        successor_hidden = transition._fixed_rms_normalize_hidden(raw_hidden)
        state = ComplexMemoryState(
            hidden=successor_hidden,
            memory=updated_memory,
            write_count=state.write_count + 1,
        )
        step_records.append(
            {
                "step": recurrent_step,
                "predecessor_hidden": predecessor_hidden,
                "external_hidden": external_hidden,
                "rotated_hidden": rotated_hidden,
                "query": query,
                "key": key,
                "value": value,
                "predecessor_memory": predecessor_memory,
                "rotated_memory": rotated_memory,
                "write": write,
                "updated_memory": updated_memory,
                "complex_read": complex_read,
                "read_flat": read_flat,
                "delta": delta,
                "raw_hidden": raw_hidden,
                "successor_hidden": successor_hidden,
            }
        )

    read_states = torch.stack(
        [record["successor_hidden"] for record in step_records], dim=-2
    )
    decoded = model.exact_inverse_selected_decode_tape_states(
        prefix, read_states, positions, anchors
    )
    logits = model.token_logits(decoded)
    token_ce = F.cross_entropy(
        logits.reshape(-1, base.VOCABULARY),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    losses = {f"H{index + 1}": token_ce[..., index].mean() for index in range(4)}
    losses["mean_h1_h4"] = token_ce.mean()

    with torch.no_grad():
        reference_states, _ = transition.rollout_read_states(roots.detach(), 4)
    forward_error = float((reference_states - read_states.detach()).abs().max())
    if forward_error != 0.0:
        raise RuntimeError(f"central trace changed forward values: {forward_error}")

    stage_items: list[tuple[int, str, torch.Tensor]] = []
    for record in step_records:
        for name in (
            "external_hidden",
            "rotated_hidden",
            "query",
            "key",
            "value",
            "predecessor_memory",
            "rotated_memory",
            "write",
            "updated_memory",
            "complex_read",
            "read_flat",
            "delta",
            "raw_hidden",
            "successor_hidden",
        ):
            stage_items.append((record["step"], name, record[name]))
    unique_tensors: list[torch.Tensor] = []
    unique_ids: set[int] = set()
    for _, _, tensor in stage_items:
        if id(tensor) not in unique_ids:
            unique_ids.add(id(tensor))
            unique_tensors.append(tensor)

    activation_rows: list[dict] = []
    gain_rows: list[dict] = []
    for loss_name in LOSSES:
        gradients = torch.autograd.grad(
            losses[loss_name],
            unique_tensors,
            retain_graph=True,
            allow_unused=True,
        )
        gradient_by_id = {
            id(tensor): gradient
            for tensor, gradient in zip(unique_tensors, gradients)
        }
        for step, stage, tensor in stage_items:
            gradient = gradient_by_id[id(tensor)]
            l2, rms, real_dof, finite = tensor_stats(gradient)
            activation_rows.append(
                {
                    "loss_source": loss_name,
                    "recurrent_step": step,
                    "stage": stage,
                    "adjoint_l2": l2,
                    "adjoint_rms": rms,
                    "real_dof": real_dof,
                    "finite": int(finite),
                }
            )

        for index, record in enumerate(step_records):
            step = record["step"]
            grad = {
                name: gradient_by_id[id(record[name])]
                for name in record
                if name != "step" and id(record[name]) in gradient_by_id
            }
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="hidden_unitary_rotation",
                output_gradients=[grad["rotated_hidden"]],
                input_gradients=[grad["external_hidden"]],
            )

            projection_inputs = []
            for projection_name in ("query", "key", "value"):
                projected_input = local_vjp(
                    record[projection_name],
                    record["rotated_hidden"],
                    grad[projection_name],
                )
                projection_inputs.append(projected_input)
                add_gain(
                    gain_rows,
                    loss=loss_name,
                    step=step,
                    operation=f"{projection_name}_projection",
                    output_gradients=[grad[projection_name]],
                    input_gradients=[projected_input],
                )
            combined_projection_input = sum(projection_inputs)
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="fused_qkv_projection",
                output_gradients=[grad["query"], grad["key"], grad["value"]],
                input_gradients=[combined_projection_input],
            )

            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="rank1_write",
                output_gradients=[grad["write"]],
                input_gradients=[grad["key"], grad["value"]],
            )
            rotated_memory_input = local_vjp(
                record["rotated_memory"],
                record["predecessor_memory"],
                grad["rotated_memory"],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="memory_unitary_rotation",
                output_gradients=[grad["rotated_memory"]],
                input_gradients=[rotated_memory_input],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="memory_add_rotated_edge",
                output_gradients=[grad["updated_memory"]],
                input_gradients=[grad["rotated_memory"]],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="memory_add_write_edge",
                output_gradients=[grad["updated_memory"]],
                input_gradients=[grad["write"]],
            )

            read_query_input = local_vjp(
                record["complex_read"],
                record["query"],
                grad["complex_read"],
            )
            read_memory_input = local_vjp(
                record["complex_read"],
                record["updated_memory"],
                grad["complex_read"],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="complex_read_query_edge",
                output_gradients=[grad["complex_read"]],
                input_gradients=[read_query_input],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="complex_read_memory_edge",
                output_gradients=[grad["complex_read"]],
                input_gradients=[read_memory_input],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="complex_read_combined_inputs",
                output_gradients=[grad["complex_read"]],
                input_gradients=[read_query_input, read_memory_input],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="complex_to_real_flatten",
                output_gradients=[grad["read_flat"]],
                input_gradients=[grad["complex_read"]],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="output_projection",
                output_gradients=[grad["delta"]],
                input_gradients=[grad["read_flat"]],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="residual_rotated_edge",
                output_gradients=[grad["raw_hidden"]],
                input_gradients=[grad["raw_hidden"]],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="residual_delta_edge",
                output_gradients=[grad["raw_hidden"]],
                input_gradients=[grad["delta"]],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="rms_postnorm",
                output_gradients=[grad["successor_hidden"]],
                input_gradients=[grad["raw_hidden"]],
            )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="rotated_hidden_branch_merge",
                output_gradients=[grad["raw_hidden"], combined_projection_input],
                input_gradients=[grad["rotated_hidden"]],
                category="branch_alignment",
            )
            future_memory_input = None
            if index + 1 < len(step_records):
                next_record = step_records[index + 1]
                future_memory_input = local_vjp(
                    next_record["rotated_memory"],
                    record["updated_memory"],
                    gradient_by_id[id(next_record["rotated_memory"])],
                )
            add_gain(
                gain_rows,
                loss=loss_name,
                step=step,
                operation="updated_memory_branch_merge",
                output_gradients=[read_memory_input, future_memory_input],
                input_gradients=[grad["updated_memory"]],
                category="branch_alignment",
            )

    args.record_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.record_dir / "central_activation_adjoint_norms.tsv", activation_rows)
    write_tsv(args.record_dir / "central_operation_vjp_gains.tsv", gain_rows)
    finite_gains = [
        row
        for row in gain_rows
        if row["category"] == "local_vjp"
        and row["local_l2_gain"] != "nan"
        and int(row["finite"])
    ]
    ranked = sorted(finite_gains, key=lambda row: float(row["local_l2_gain"]), reverse=True)
    lines = [
        "# Central operation VJP amplification",
        "",
        f"Checkpoint step: 100; batch: {args.batch}; forward max error: {forward_error}",
        "",
        "Local L2 gain is `input adjoint L2 / output adjoint L2` for the same "
        "executed operation. Branch-alignment rows are kept separate.",
        "",
        "| loss | step | operation | local L2 gain | local RMS gain |",
        "|---|---:|---|---:|---:|",
    ]
    for row in ranked[:80]:
        lines.append(
            f"| {row['loss_source']} | {row['recurrent_step']} | "
            f"{row['operation']} | {float(row['local_l2_gain']):.8f} | "
            f"{float(row['local_rms_gain']):.8f} |"
        )
    (args.record_dir / "central_vjp_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("TOP_LOCAL_VJP_GAINS")
    for row in ranked[:30]:
        print(
            row["loss_source"],
            row["recurrent_step"],
            row["operation"],
            f"L2={float(row['local_l2_gain']):.6f}",
            f"RMS={float(row['local_rms_gain']):.6f}",
        )
    print(f"forward_error={forward_error:.3e} rows={len(gain_rows)}")


if __name__ == "__main__":
    main()
