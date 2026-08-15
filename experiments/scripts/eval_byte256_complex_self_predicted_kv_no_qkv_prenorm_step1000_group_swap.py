"""Isolate one native update by swapping post-step parameter groups."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_h16_attached_stride16_ce_only_13m as source


EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-no-qkv-prenorm-"
    "step1000-one-update-forensics"
)
RECORD_ROOT = Path("experiments/records") / EXPERIMENT_ID
OUTPUT_ROOT = Path("outputs/experiments") / EXPERIMENT_ID
SOURCE_CHECKPOINT = (
    Path("outputs/experiments")
    / source.EXPERIMENT_ID
    / "step1000.pt"
)
SOURCE_STEP = 1000
SOURCE_BATCH = 64
SOURCE_MICROBATCH = 16
SOURCE_SCHEDULE_STEPS = 6000

base = source.base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--record-dir", type=Path, default=RECORD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--microbatch", type=int, default=SOURCE_MICROBATCH)
    return parser.parse_args()


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def trainable_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def atomic_group(name: str) -> str:
    if name.startswith("encoder.blocks."):
        parts = name.split(".")
        block = parts[2]
        family = parts[3]
        if family == "attn":
            family = f"attn.{parts[4]}"
        elif family == "ffn":
            family = f"ffn.{parts[4]}"
        elif family not in ("norm1", "norm2"):
            raise RuntimeError(f"unregistered encoder family for {name}")
        return f"encoder.block{block}.{family}"
    prefix = "complex_self_prediction."
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.startswith("query."):
            return "central.query"
        if suffix.startswith("key."):
            return "central.key"
        if suffix.startswith("value."):
            return "central.value"
        if suffix.startswith("output."):
            return "central.output"
        if suffix == "hidden_phase":
            return "central.hidden_phase"
        if suffix == "memory_phase":
            return "central.memory_phase"
    raise RuntimeError(f"trainable parameter has no registered group: {name}")


def build_groups(
    parameters: dict[str, torch.nn.Parameter],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    atomic_lists: dict[str, list[str]] = {}
    for name in parameters:
        atomic_lists.setdefault(atomic_group(name), []).append(name)
    atomic = {
        group: tuple(sorted(names))
        for group, names in sorted(atomic_lists.items())
    }
    flattened = [name for names in atomic.values() for name in names]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(parameters):
        raise RuntimeError("atomic groups do not partition trainable parameters")

    def selected(prefixes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name
            for group, names in atomic.items()
            if group.startswith(prefixes)
            for name in names
        )

    groups = dict(atomic)
    groups.update(
        {
            "aggregate.encoder.block0": selected(("encoder.block0.",)),
            "aggregate.encoder.block1": selected(("encoder.block1.",)),
            "aggregate.encoder.all": selected(("encoder.",)),
            "aggregate.central.projections": selected(
                (
                    "central.query",
                    "central.key",
                    "central.value",
                    "central.output",
                )
            ),
            "aggregate.central.phases": selected(
                ("central.hidden_phase", "central.memory_phase")
            ),
            "aggregate.central.all": selected(("central.",)),
        }
    )
    for group, names in groups.items():
        if not names:
            raise RuntimeError(f"empty parameter group: {group}")
    return groups, tuple(atomic)


class TensorScale:
    def __init__(self) -> None:
        self.squared_sum = 0.0
        self.elements = 0
        self.max_abs = 0.0
        self.row_rms_sum = 0.0
        self.rows = 0
        self.max_row_rms = 0.0

    def add(self, values: torch.Tensor) -> None:
        detached = values.detach()
        squared = (
            detached.abs().float().square()
            if detached.is_complex()
            else detached.float().square()
        )
        self.squared_sum += float(squared.sum())
        self.elements += squared.numel()
        self.max_abs = max(self.max_abs, float(squared.amax().sqrt()))
        row_rms = squared.flatten(0, -2).mean(dim=-1).sqrt()
        self.row_rms_sum += float(row_rms.sum())
        self.rows += row_rms.numel()
        self.max_row_rms = max(self.max_row_rms, float(row_rms.amax()))

    def metrics(self, prefix: str) -> dict[str, float]:
        if self.elements < 1 or self.rows < 1:
            raise RuntimeError(f"empty tensor scale accumulator: {prefix}")
        return {
            f"{prefix}_rms": math.sqrt(self.squared_sum / self.elements),
            f"{prefix}_max_abs": self.max_abs,
            f"{prefix}_mean_row_rms": self.row_rms_sum / self.rows,
            f"{prefix}_max_row_rms": self.max_row_rms,
        }


class EnergyScale:
    """Accumulate rowwise mean-square energy already reduced by the model."""

    def __init__(self) -> None:
        self.energy_sum = 0.0
        self.rows = 0
        self.row_rms_sum = 0.0
        self.max_row_rms = 0.0

    def add(self, energy: torch.Tensor) -> None:
        detached = energy.detach().float()
        row_rms = detached.sqrt()
        self.energy_sum += float(detached.sum())
        self.rows += detached.numel()
        self.row_rms_sum += float(row_rms.sum())
        self.max_row_rms = max(self.max_row_rms, float(row_rms.amax()))

    def metrics(self, prefix: str) -> dict[str, float]:
        if self.rows < 1:
            raise RuntimeError(f"empty energy accumulator: {prefix}")
        return {
            f"{prefix}_rms": math.sqrt(self.energy_sum / self.rows),
            f"{prefix}_mean_row_rms": self.row_rms_sum / self.rows,
            f"{prefix}_max_row_rms": self.max_row_rms,
        }


@torch.inference_mode()
def measure_same_batch(
    model: torch.nn.Module,
    window: torch.Tensor,
    microbatch: int,
    *,
    include_horizon_ce: bool = False,
) -> dict[str, float]:
    model.eval()
    ce_sum = 0.0
    ce_count = 0
    h1_ce_sum = 0.0
    h16_ce_sum = 0.0
    horizon_count = 0
    horizon_ce_sum = torch.zeros(
        base.TRAIN_HORIZONS, dtype=torch.float64
    )
    scales = {
        "root": TensorScale(),
        "h1_p": TensorScale(),
        "h1_z": TensorScale(),
        "h1_decoded": TensorScale(),
        "h1_logit": TensorScale(),
        "h16_p": TensorScale(),
        "h16_z": TensorScale(),
        "h16_decoded": TensorScale(),
        "h16_logit": TensorScale(),
    }
    energies = {"h1_w": EnergyScale(), "h16_w": EnergyScale()}
    for micro_window in window.split(microbatch):
        _, _, _, output = base.objective(model, micro_window)
        token_ce = output.token_ce.detach().float()
        ce_sum += float(
            token_ce.double().sum() if include_horizon_ce else token_ce.sum()
        )
        ce_count += token_ce.numel()
        h1_ce_sum += float(
            token_ce[:, :, 0].double().sum()
            if include_horizon_ce
            else token_ce[:, :, 0].sum()
        )
        h16_ce_sum += float(
            token_ce[:, :, -1].double().sum()
            if include_horizon_ce
            else token_ce[:, :, -1].sum()
        )
        horizon_count += token_ce[:, :, 0].numel()
        if include_horizon_ce:
            horizon_ce_sum += token_ce.sum(dim=(0, 1)).double().cpu()
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        scales["root"].add(roots)
        for label, horizon in (("h1", 0), ("h16", -1)):
            scales[f"{label}_p"].add(output.preliminary_states[:, :, horizon])
            scales[f"{label}_z"].add(output.full_states[:, :, horizon])
            scales[f"{label}_decoded"].add(output.decoded_hidden[:, :, horizon])
            scales[f"{label}_logit"].add(output.logits[:, :, horizon])
            energies[f"{label}_w"].add(
                output.innovation_energy[:, :, horizon]
            )
    if ce_count < 1 or horizon_count < 1:
        raise RuntimeError("same-batch measurement produced no CE labels")
    result = {
        "same_batch_ce": ce_sum / ce_count,
        "same_batch_h1_ce": h1_ce_sum / horizon_count,
        "same_batch_h16_ce": h16_ce_sum / horizon_count,
    }
    if include_horizon_ce:
        result.update(
            {
                f"same_batch_h{horizon}_ce": float(
                    horizon_ce_sum[horizon - 1] / horizon_count
                )
                for horizon in range(1, base.TRAIN_HORIZONS + 1)
            }
        )
    for prefix, accumulator in scales.items():
        result.update(accumulator.metrics(prefix))
    for prefix, accumulator in energies.items():
        result.update(accumulator.metrics(prefix))
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite same-batch counterfactual metric")
    return result


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty TSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise RuntimeError(f"inconsistent TSV fields for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_key_values(path: Path, values: dict[str, object]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        writer.writerows(values.items())


def delta_statistics(
    names: tuple[str, ...],
    pre_state: dict[str, torch.Tensor],
    post_state: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    squared_delta = 0.0
    squared_weight = 0.0
    max_delta = 0.0
    count = 0
    for name in names:
        before = pre_state[name].double()
        delta = post_state[name].double() - before
        squared_delta += float(delta.square().sum())
        squared_weight += float(before.square().sum())
        max_delta = max(max_delta, float(delta.abs().amax()))
        count += delta.numel()
    delta_l2 = math.sqrt(squared_delta)
    return {
        "parameter_count": count,
        "parameter_delta_l2": delta_l2,
        "parameter_delta_rms": math.sqrt(squared_delta / count),
        "parameter_delta_max_abs": max_delta,
        "parameter_delta_to_weight_l2": delta_l2
        / max(math.sqrt(squared_weight), 1e-30),
    }


def state_max_error(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    if left.keys() != right.keys():
        return math.inf
    error = 0.0
    for name in left:
        if torch.equal(left[name], right[name]):
            continue
        if left[name].is_floating_point() or left[name].is_complex():
            error = max(error, float((left[name] - right[name]).abs().amax()))
        else:
            return math.inf
    return error


def main() -> None:
    args = parse_args()
    if args.microbatch < 1 or SOURCE_BATCH % args.microbatch:
        raise SystemExit("microbatch must divide the native batch of 64")
    if not torch.cuda.is_available():
        raise SystemExit("this audit requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)

    payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != SOURCE_STEP:
        raise RuntimeError("source checkpoint is not native step 1000")
    if int(payload["config"]["schedule_steps"]) != SOURCE_SCHEDULE_STEPS:
        raise RuntimeError("source checkpoint has a different LR schedule")
    if bool(payload["config"]["pre_normalize_qkv_inputs"]):
        raise RuntimeError("source checkpoint unexpectedly enables QKV pre-norm")

    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = base.make_model().cuda().train()
    model.load_state_dict(payload["model"], strict=True)
    parameters = trainable_parameters(model)
    groups, atomic_groups = build_groups(parameters)

    optimizer = torch.optim.AdamW(
        parameters.values(),
        lr=base.PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    optimizer.load_state_dict(payload["optimizer"])
    update_lr = base.lr_at(SOURCE_STEP, SOURCE_SCHEDULE_STEPS)
    for optimizer_group in optimizer.param_groups:
        optimizer_group["lr"] = update_lr

    data_generator = torch.Generator()
    data_generator.set_state(payload["data_generator_state"])
    generator_before = data_generator.get_state().clone()
    training = base.memmap("train")
    window = base.sampled_windows(
        training,
        SOURCE_BATCH,
        data_generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    if tuple(window.shape) != (SOURCE_BATCH, base.CONTEXT + base.TRAIN_HORIZONS):
        raise RuntimeError(f"unexpected native next-batch shape: {tuple(window.shape)}")
    generator_after = data_generator.get_state().clone()

    pre_state = cpu_state_dict(model)
    pre_metrics = measure_same_batch(model, window, args.microbatch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_windows = window.split(args.microbatch)
    training_ce = 0.0
    for micro_window in micro_windows:
        loss, token_ce, auxiliary, output = base.objective(model, micro_window)
        if float(auxiliary.detach()) != 0.0:
            raise RuntimeError("source CE-only objective produced auxiliary loss")
        scale = 1.0 / len(micro_windows)
        (loss * scale).backward()
        training_ce += float(token_ce.detach()) * scale
        del loss, token_ce, auxiliary, output

    group_raw_gradient_l2: dict[str, float] = {}
    for group, names in groups.items():
        squared = sum(
            float(parameters[name].grad.detach().float().square().sum())
            for name in names
            if parameters[name].grad is not None
        )
        group_raw_gradient_l2[group] = math.sqrt(squared)
    raw_gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP_NORM)
    )
    if not math.isfinite(raw_gradient_norm):
        raise RuntimeError("native update1001 gradient is non-finite")
    optimizer.step()
    post_state = cpu_state_dict(model)
    post_metrics = measure_same_batch(model, window, args.microbatch)

    torch.save(pre_state, args.output_dir / "group_swap_pre_update1001_model.pt")
    torch.save(post_state, args.output_dir / "group_swap_post_update1001_model.pt")

    total_delta = delta_statistics(
        tuple(parameters),
        pre_state,
        post_state,
    )
    total_delta_squared = float(total_delta["parameter_delta_l2"]) ** 2
    metric_names = tuple(pre_metrics)

    def result_row(
        variant: str,
        variant_type: str,
        names: tuple[str, ...],
        metrics: dict[str, float],
        raw_group_gradient: float,
    ) -> dict[str, object]:
        statistics = (
            delta_statistics(names, pre_state, post_state)
            if names
            else {
                "parameter_count": 0,
                "parameter_delta_l2": 0.0,
                "parameter_delta_rms": 0.0,
                "parameter_delta_max_abs": 0.0,
                "parameter_delta_to_weight_l2": 0.0,
            }
        )
        delta_fraction = (
            float(statistics["parameter_delta_l2"]) ** 2
            / max(total_delta_squared, 1e-30)
        )
        row: dict[str, object] = {
            "variant": variant,
            "variant_type": variant_type,
            **statistics,
            "squared_total_parameter_delta_fraction": delta_fraction,
            "preclip_group_gradient_l2": raw_group_gradient,
        }
        for metric in metric_names:
            row[metric] = metrics[metric]
            row[f"delta_{metric}_vs_pre"] = metrics[metric] - pre_metrics[metric]
        full_ce_effect = post_metrics["same_batch_ce"] - pre_metrics["same_batch_ce"]
        row["same_batch_ce_effect_fraction_of_full_update"] = (
            (metrics["same_batch_ce"] - pre_metrics["same_batch_ce"])
            / full_ce_effect
            if full_ce_effect != 0.0
            else math.nan
        )
        return row

    result_rows = [
        result_row("pre_update", "control", (), pre_metrics, 0.0),
        result_row(
            "full_post_update",
            "control",
            tuple(parameters),
            post_metrics,
            raw_gradient_norm,
        ),
    ]
    member_rows: list[dict[str, object]] = []
    hybrid = base.make_model().cuda().eval()
    hybrid_parameters = dict(hybrid.named_parameters())
    for group, names in groups.items():
        hybrid.load_state_dict(pre_state, strict=True)
        with torch.no_grad():
            for name in names:
                hybrid_parameters[name].copy_(post_state[name].to("cuda"))
        metrics = measure_same_batch(hybrid, window, args.microbatch)
        variant_type = "atomic" if group in atomic_groups else "aggregate"
        result_rows.append(
            result_row(
                group,
                variant_type,
                names,
                metrics,
                group_raw_gradient_l2[group],
            )
        )
        member_rows.extend(
            {
                "group": group,
                "variant_type": variant_type,
                "parameter_name": name,
                "parameter_count": pre_state[name].numel(),
            }
            for name in names
        )

    hybrid.load_state_dict(pre_state, strict=True)
    with torch.no_grad():
        for group in atomic_groups:
            for name in groups[group]:
                hybrid_parameters[name].copy_(post_state[name].to("cuda"))
    union_state = cpu_state_dict(hybrid)
    union_state_error = state_max_error(union_state, post_state)
    union_metrics = measure_same_batch(hybrid, window, args.microbatch)
    union_metric_error = max(
        abs(union_metrics[name] - post_metrics[name]) for name in metric_names
    )
    if union_state_error != 0.0 or union_metric_error > 1e-7:
        raise RuntimeError(
            "atomic group union did not reproduce full post update: "
            f"state={union_state_error}, metric={union_metric_error}"
        )

    write_tsv(args.record_dir / "update_group_counterfactual.tsv", result_rows)
    write_tsv(args.record_dir / "group_swap_members.tsv", member_rows)
    write_key_values(
        args.record_dir / "group_swap_run.tsv",
        {
            "experiment_id": EXPERIMENT_ID,
            "source_checkpoint": str(args.source_checkpoint),
            "source_step": SOURCE_STEP,
            "update_step": SOURCE_STEP + 1,
            "seed": base.SEED,
            "split": "train",
            "effective_batch": SOURCE_BATCH,
            "microbatch": args.microbatch,
            "window_length": base.TRAIN_WINDOW_LENGTH,
            "anchors": base.TRAIN_ANCHORS,
            "horizons": base.TRAIN_HORIZONS,
            "schedule_steps": SOURCE_SCHEDULE_STEPS,
            "learning_rate": update_lr,
            "clip_norm": base.CLIP_NORM,
            "raw_gradient_norm": raw_gradient_norm,
            "gradient_clip_multiplier": min(1.0, base.CLIP_NORM / raw_gradient_norm),
            "training_ce_before_update": training_ce,
            "measured_ce_before_update": pre_metrics["same_batch_ce"],
            "measured_ce_after_update": post_metrics["same_batch_ce"],
            "generator_state_before_bytes": generator_before.numel(),
            "generator_state_after_bytes": generator_after.numel(),
            "atomic_group_count": len(atomic_groups),
            "atomic_union_state_max_error": union_state_error,
            "atomic_union_metric_max_error": union_metric_error,
            "pre_model_state_output": str(
                args.output_dir / "group_swap_pre_update1001_model.pt"
            ),
            "post_model_state_output": str(
                args.output_dir / "group_swap_post_update1001_model.pt"
            ),
        },
    )

    atomic_rows = [row for row in result_rows if row["variant_type"] == "atomic"]
    strongest_ce = max(
        atomic_rows,
        key=lambda row: abs(float(row["delta_same_batch_ce_vs_pre"])),
    )
    strongest_p = max(
        atomic_rows,
        key=lambda row: abs(float(row["delta_h1_p_rms_vs_pre"])),
    )
    strongest_w = max(
        atomic_rows,
        key=lambda row: abs(float(row["delta_h1_w_rms_vs_pre"])),
    )
    interpretation = (
        "# One-update parameter-group swap\n\n"
        "This audit restored the native step-1000 Adam and training-data "
        "generator states, drew exactly the next training batch, and applied "
        "one source update. Each counterfactual starts from the pre-update "
        "model and receives only the named group's actual post-step delta.\n\n"
        f"- Raw pre-clip gradient norm: `{raw_gradient_norm:.9g}`.\n"
        f"- Same-batch CE: `{pre_metrics['same_batch_ce']:.9f}` -> "
        f"`{post_metrics['same_batch_ce']:.9f}`.\n"
        f"- Largest atomic absolute CE effect: `{strongest_ce['variant']}` "
        f"(`{float(strongest_ce['delta_same_batch_ce_vs_pre']):+.9g}`).\n"
        f"- Largest atomic H1 P-RMS effect: `{strongest_p['variant']}` "
        f"(`{float(strongest_p['delta_h1_p_rms_vs_pre']):+.9g}`).\n"
        f"- Largest atomic H1 W-RMS effect: `{strongest_w['variant']}` "
        f"(`{float(strongest_w['delta_h1_w_rms_vs_pre']):+.9g}`).\n"
        f"- Atomic-union state/metric errors: `{union_state_error:.3g}` / "
        f"`{union_metric_error:.3g}`.\n\n"
        "These are instantaneous, same-batch effects around step 1000. "
        "Interactions between simultaneous Adam deltas can make isolated "
        "effects non-additive, and no row establishes long-run stability.\n"
    )
    (args.record_dir / "group_swap_interpretation.md").write_text(
        interpretation
    )


if __name__ == "__main__":
    main()
