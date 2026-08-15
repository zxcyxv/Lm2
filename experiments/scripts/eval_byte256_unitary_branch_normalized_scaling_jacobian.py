"""Checkpoint audit of scaling and J=D+E assumptions."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from rotlm.models.complex_self_prediction import ComplexMemoryState
from rotlm.models.prefix_orthogonal_scan import rotate_pairwise
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
RECORD = Path("experiments/records/EXP-20260802-byte256-unitary-branch-normalized-scaling-jacobian-audit")
CONTINUATION_OUTPUT = Path(
    "outputs/experiments/EXP-20260802-byte256-complex-self-predicted-kv-"
    "unitary-branch-normalized-residual-h16-ce-only-attached-stride16-"
    "1000step-13m"
)
CHECKPOINTS = (
    run.DEFAULT_OUTPUT / "step0100.pt",
    run.DEFAULT_OUTPUT / "step0300.pt",
    CONTINUATION_OUTPUT / "step0500.pt",
    CONTINUATION_OUTPUT / "step1000.pt",
)
PROBES = 4


def combined_norm(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (z.float().square().sum() + s.abs().float().square().sum()).sqrt()


def exponent(xs: list[float], ys: list[float]) -> float:
    x = torch.tensor(xs, dtype=torch.float64).log()
    y = torch.tensor(ys, dtype=torch.float64).log()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / x.square().sum())


def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().flatten() - x.double().mean()
    y = y.double().flatten() - y.double().mean()
    denominator = x.norm() * y.norm()
    return float((x @ y) / denominator) if denominator > 0 else math.nan


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    validation = base.memmap("validation")
    starts = torch.randint(
        0,
        len(validation) - base.EVAL_WINDOW_LENGTH,
        (4,),
        generator=torch.Generator().manual_seed(base.SEED + 999),
    )
    window = base.windows_of_length(
        validation, starts, base.EVAL_WINDOW_LENGTH
    ).cuda()
    trajectory_rows = []
    jvp_rows = []
    summary_rows = []
    RECORD.mkdir(parents=True, exist_ok=True)

    for checkpoint_path in CHECKPOINTS:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        step_number = int(checkpoint["step"])
        model = base.make_model().cuda().eval()
        model.load_state_dict(checkpoint["model"])
        transition = model.complex_self_prediction
        with torch.no_grad():
            _, _, _, output = base.objective(model, window)
            roots = output.prefix_encoded.index_select(1, output.anchor_indices)
            state = transition.initialize(roots)
            states = [state]
            z_means, centered_means, s_means, angle_means, perp_means = [], [], [], [], []
            homogeneous = roots
            memory_rows = []
            for horizon in range(1, base.TRAIN_HORIZONS + 1):
                result = transition(state)
                homogeneous = rotate_pairwise(
                    homogeneous,
                    transition.hidden_phase.expand(
                        *homogeneous.shape[:-1], -1
                    ),
                )
                zbar = result.rotated_hidden.float()
                delta = result.innovation_delta.float()
                successor = result.state.hidden.float()
                z_norm = successor.norm(dim=-1)
                zbar_norm = zbar.norm(dim=-1)
                delta_norm = delta.norm(dim=-1)
                projection = (delta * zbar).sum(dim=-1, keepdim=True) / zbar.square().sum(dim=-1, keepdim=True).clamp_min(1e-30)
                perpendicular = (delta - projection * zbar).norm(dim=-1)
                cosine = (zbar * successor).sum(dim=-1) / (zbar_norm * z_norm).clamp_min(1e-30)
                angle = cosine.clamp(-1, 1).acos()
                memory_norm = result.state.memory.abs().float().square().sum(dim=(-3, -2, -1)).sqrt()
                memory_rows.append(memory_norm)
                z_means.append(float(z_norm.mean()))
                centered_means.append(
                    float((successor - homogeneous).norm(dim=-1).mean())
                )
                s_means.append(float(memory_norm.mean()))
                angle_means.append(float(angle.mean()))
                perp_means.append(float((perpendicular / zbar_norm.clamp_min(1e-30)).mean()))
                trajectory_rows.append({
                    "checkpoint_step": step_number,
                    "horizon": horizon,
                    "mean_latent_norm": z_means[-1],
                    "mean_centered_residual_norm": centered_means[-1],
                    "mean_memory_frobenius": s_means[-1],
                    "mean_correction_norm": float(delta_norm.mean()),
                    "mean_direction_angle_rad": angle_means[-1],
                    "mean_perpendicular_ratio": perp_means[-1],
                })
                state = result.state
                states.append(state)
            memory_tensor = torch.stack(memory_rows, dim=-1)
            memory_loss_correlation = correlation(memory_tensor, output.token_ce)

        # Directional local and product Jacobian evidence on one fixed root.
        root_state = ComplexMemoryState(
            hidden=states[0].hidden[0, 0].detach(),
            memory=states[0].memory[0, 0].detach(),
            write_count=0,
        )
        probe_generator = torch.Generator(device="cuda").manual_seed(88000 + step_number)
        product_gains = []
        for probe in range(PROBES):
            dz = torch.randn(root_state.hidden.shape, device="cuda", generator=probe_generator)
            ds = torch.complex(
                torch.randn(root_state.memory.shape, device="cuda", generator=probe_generator),
                torch.randn(root_state.memory.shape, device="cuda", generator=probe_generator),
            )
            scale = combined_norm(dz, ds)
            dz, ds = dz / scale, ds / scale
            tangent_z, tangent_s = dz, ds
            state = root_state
            for horizon in range(1, base.TRAIN_HORIZONS + 1):
                count = state.write_count

                def local(z, s):
                    result = transition(ComplexMemoryState(z, s, count))
                    return result.state.hidden, result.state.memory

                (_, _), (jz, js) = torch.func.jvp(
                    local,
                    (state.hidden, state.memory),
                    (tangent_z, tangent_s),
                )
                angles = transition.hidden_phase.expand(*tangent_z.shape[:-1], -1)
                dz_unitary = rotate_pairwise(tangent_z, angles)
                memory_phase = torch.polar(
                    torch.ones_like(transition.memory_phase),
                    -transition.memory_phase,
                )
                ds_unitary = tangent_s * memory_phase[..., None]
                ez, es = jz - dz_unitary, js - ds_unitary
                j_norm = float(combined_norm(jz, js))
                d_norm = float(combined_norm(dz_unitary, ds_unitary))
                e_norm = float(combined_norm(ez, es))
                inner = (
                    (jz.float() * dz_unitary.float()).sum()
                    + (js.conj() * ds_unitary).real.float().sum()
                )
                alignment = float(inner / (combined_norm(jz, js) * combined_norm(dz_unitary, ds_unitary)).clamp_min(1e-30))
                jvp_rows.append({
                    "checkpoint_step": step_number,
                    "probe": probe,
                    "horizon": horizon,
                    "input_tangent_norm": float(combined_norm(tangent_z, tangent_s)),
                    "jv_norm": j_norm,
                    "dv_norm": d_norm,
                    "ev_norm": e_norm,
                    "ev_to_dv": e_norm / d_norm,
                    "jv_dv_alignment": alignment,
                })
                with torch.no_grad():
                    next_result = transition(state)
                    state = ComplexMemoryState(
                        next_result.state.hidden.detach(),
                        next_result.state.memory.detach(),
                        next_result.state.write_count,
                    )
                tangent_z, tangent_s = jz.detach(), js.detach()
            product_gains.append(float(combined_norm(tangent_z, tangent_s)))

        local = [r for r in jvp_rows if r["checkpoint_step"] == step_number]
        by_horizon_e = [
            sum(float(r["ev_to_dv"]) for r in local if r["horizon"] == h) / PROBES
            for h in range(1, base.TRAIN_HORIZONS + 1)
        ]
        summary_rows.append({
            "checkpoint_step": step_number,
            "latent_norm_loglog_exponent_h2_h16": exponent(list(range(2, 17)), z_means[1:]),
            "centered_residual_loglog_exponent_h2_h16": exponent(list(range(2, 17)), centered_means[1:]),
            "memory_norm_loglog_exponent_h2_h16": exponent(list(range(2, 17)), s_means[1:]),
            "direction_angle_loglog_exponent_h2_h16": exponent(list(range(2, 17)), angle_means[1:]),
            "perpendicular_ratio_loglog_exponent_h2_h16": exponent(list(range(2, 17)), perp_means[1:]),
            "e_to_d_loglog_exponent_h2_h16": exponent(list(range(2, 17)), by_horizon_e[1:]),
            "mean_e_to_d": sum(by_horizon_e) / len(by_horizon_e),
            "min_product_directional_gain_h16": min(product_gains),
            "mean_product_directional_gain_h16": sum(product_gains) / len(product_gains),
            "max_product_directional_gain_h16": max(product_gains),
            "memory_norm_token_ce_pearson": memory_loss_correlation,
        })
        del model, checkpoint
        torch.cuda.empty_cache()

    write_rows(RECORD / "trajectory_scaling.tsv", trajectory_rows)
    write_rows(RECORD / "jvp.tsv", jvp_rows)
    write_rows(RECORD / "summary.tsv", summary_rows)


if __name__ == "__main__":
    main()
