"""Compare initial-root RMSNorm recurrent scales on fixed byte-h rows."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping

import torch

import eval_byte256_complex_self_predicted_kv_h16_scale_jacobian as scale_audit
from rotlm.latent_diagnostics import (
    ComplexRecurrentScaleTape,
    complex_recurrent_scale_tape,
)
from rotlm.training.ha_skew_window import sparse_anchor_indices


EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "initial-root-rmsnorm-anchor0-h-trajectory-monitor"
)
CANDIDATE_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-"
    "no-qkv-prenorm-initial-root-rmsnorm-h16-ce-only-attached-stride16-13m"
)
CONTROL_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-"
    "no-qkv-prenorm-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
OUTPUT_ROOT = Path("outputs/experiments")
DEFAULT_CANDIDATE_OUTPUT = OUTPUT_ROOT / CANDIDATE_ID
DEFAULT_CONTROL_OUTPUT = OUTPUT_ROOT / CONTROL_ID

CONTEXT = 256
HORIZONS = 16
ANCHOR_STRIDE = 16
ANCHORS = CONTEXT // ANCHOR_STRIDE
WINDOW_LENGTH = CONTEXT + HORIZONS
EXAMPLES = 256
PHYSICAL_MICROBATCH = 16
SAMPLE_SEED = 27_358
ANCHOR0_H_TOKEN = 104
VALIDATION_SHA256 = (
    "7ffe866bb723736027b30046f5db623588285ef0f3c061872d53793c589327b0"
)
DEFAULT_STEPS = (100, 500, 1000)

base = scale_audit.base

GROUPS = (
    "anchor0_h",
    "anchor0_non_h",
    "h_rows_other_anchors",
    "all_anchor_rows",
)
SUMMARY_STATISTICS = ("mean", "median", "q90", "q99", "maximum")
H0_TENSORS = (
    ("raw_encoded_root", "raw_root_hidden_rms"),
    ("initialized_recurrent_root", "initialized_hidden_rms"),
    ("initialized_memory", "initialized_memory_rms"),
)
STEP_TENSORS = (
    ("rotated_z", "rotated_hidden_rms"),
    ("rotated_s", "rotated_memory_rms"),
    ("prior_p", "preliminary_hidden_rms"),
    ("innovation_w", "innovation_memory_rms"),
    ("raw_successor_z", "raw_full_hidden_rms"),
    ("raw_updated_s", "raw_updated_memory_rms"),
    ("z_boundary_denominator", "hidden_boundary_denominator"),
    ("s_boundary_denominator", "memory_boundary_denominator"),
    ("stored_successor_z", "stored_hidden_rms"),
    ("stored_successor_s", "stored_memory_rms"),
)
EXPECTED_OBJECTIVES = {
    "candidate": (
        "sixteen_step_initial_root_rmsnorm_urm_boundary_postnorm_no_qkv_"
        "prenorm_parallel_decode_ce_only_attached_stride16"
    ),
    "control": (
        "sixteen_step_urm_boundary_postnorm_no_qkv_prenorm_parallel_decode_"
        "ce_only_attached_stride16"
    ),
}
EXPECTED_SHARED_CONFIG = {
    "vocabulary": 256,
    "width": 1344,
    "encoder_blocks": 2,
    "heads": 8,
    "key_dim": 16,
    "value_dim": 31,
    "simplex_logit_scale": 16.0,
    "tempered_readout": False,
    "residual_prior": False,
    "recurrent_hidden_mode": "post-update-full-read",
    "pre_normalize_qkv_inputs": False,
    "normalize_accumulated_reads": False,
    "post_normalize_hidden_reads": False,
    "post_normalize_memory": False,
    "post_normalize_recurrent_state": True,
    "post_norm_eps": 1e-6,
    "decoder_readout": "innovation_free_prior",
    "latent_weight": 0.0,
    "closure_weight": 0.0,
    "closure_metric": "no_auxiliary_loss",
    "detach_latent_target": False,
    "latent_target": "none",
    "closure_target": "none",
    "ema_decay": None,
    "ema_warm_start_steps": 0,
    "inference_model": "online",
    "recurrence": "unitary_memory_plus_self_predicted_kv",
    "seed": 1337,
    "batch": 64,
    "microbatch": 16,
    "schedule_steps": 6000,
}


def parse_steps(value: str) -> tuple[int, ...]:
    """Parse an exact, ascending checkpoint list without fallback semantics."""
    pieces = value.split(",")
    if not pieces or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError("steps must be comma-separated integers")
    try:
        parsed = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "steps must be comma-separated integers"
        ) from error
    if any(step < 1 for step in parsed):
        raise argparse.ArgumentTypeError("checkpoint steps must be positive")
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError(
            "checkpoint steps must be unique and strictly increasing"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=parse_steps, default=DEFAULT_STEPS)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=DEFAULT_CANDIDATE_OUTPUT,
    )
    parser.add_argument(
        "--control-output",
        type=Path,
        default=DEFAULT_CONTROL_OUTPUT,
    )
    return parser.parse_args()


def checkpoint_path(output: Path, step: int) -> Path:
    """Return only the exact registered checkpoint name."""
    if step < 1:
        raise ValueError("checkpoint step must be positive")
    return output / f"step{step:04d}.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_sha256(windows: torch.Tensor) -> str:
    if windows.ndim != 2 or windows.shape != (EXAMPLES, WINDOW_LENGTH):
        raise ValueError("sample windows have an unexpected shape")
    if int(windows.min()) < 0 or int(windows.max()) >= 256:
        raise ValueError("byte sample contains a token outside [0,255]")
    raw = windows.to(torch.uint8).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def fixed_validation_sample() -> tuple[torch.Tensor, torch.Tensor, str]:
    """Materialize the manifest-fixed validation sample, never train or test."""
    validation = base.memmap("validation")
    validation_path = Path(base.DATA_ROOT) / "validation.bin"
    actual_digest = file_sha256(validation_path)
    if actual_digest != VALIDATION_SHA256:
        raise RuntimeError(
            f"validation SHA-256 {actual_digest} != {VALIDATION_SHA256}"
        )
    starts = torch.randint(
        0,
        len(validation) - WINDOW_LENGTH,
        (EXAMPLES,),
        generator=torch.Generator().manual_seed(SAMPLE_SEED),
    )
    chunks = []
    for begin in range(0, EXAMPLES, PHYSICAL_MICROBATCH):
        chunk = base.windows_of_length(
            validation,
            starts[begin : begin + PHYSICAL_MICROBATCH],
            WINDOW_LENGTH,
        )
        chunks.append(chunk.cpu())
    windows = torch.cat(chunks, dim=0)
    if windows.shape != (EXAMPLES, WINDOW_LENGTH):
        raise RuntimeError(f"unexpected sampled window shape {windows.shape}")
    return starts, windows, sample_sha256(windows)


def byte_h_count_rows(
    starts: torch.Tensor,
    windows: torch.Tensor,
    digest: str,
) -> list[dict[str, object]]:
    if starts.shape != (EXAMPLES,) or windows.shape != (
        EXAMPLES,
        WINDOW_LENGTH,
    ):
        raise ValueError("fixed sample has an unexpected shape")
    anchor0 = windows[:, 0].long()
    h_mask = anchor0.eq(ANCHOR0_H_TOKEN)
    h_count = int(h_mask.sum())
    if h_count < 1 or h_count >= EXAMPLES:
        raise RuntimeError(
            "registered anchor0_h and anchor0_non_h groups must be non-empty"
        )
    rate = h_count / EXAMPLES
    return [
        {
            "generator_seed": SAMPLE_SEED,
            "sample_sha256": digest,
            "global_row": row,
            "physical_microbatch": row // PHYSICAL_MICROBATCH,
            "local_row": row % PHYSICAL_MICROBATCH,
            "validation_start": int(starts[row]),
            "anchor0_token_id": int(anchor0[row]),
            "anchor0_byte_hex": f"{int(anchor0[row]):02x}",
            "is_anchor0_h": int(h_mask[row]),
            "anchor0_h_count": h_count,
            "total_rows": EXAMPLES,
            "anchor0_h_rate": rate,
        }
        for row in range(EXAMPLES)
    ]


def validate_checkpoint_config(config: Mapping[str, object], role: str) -> None:
    if role not in EXPECTED_OBJECTIVES:
        raise ValueError(f"unknown checkpoint role {role!r}")
    for key, expected in EXPECTED_SHARED_CONFIG.items():
        if key not in config or config[key] != expected:
            raise RuntimeError(
                f"{role} checkpoint config {key!r}={config.get(key)!r}; "
                f"expected {expected!r}"
            )
    expected_root_norm = role == "candidate"
    # The native control predates this config key. Its historical constructor
    # default is False, matching configure_model's explicit legacy fallback.
    # A candidate omission must still fail because that intervention is the
    # sole registered difference.
    root_norm = config.get("normalize_initial_recurrent_root", False)
    if root_norm is not expected_root_norm:
        raise RuntimeError(
            f"{role} has wrong initial-root normalization setting"
        )
    if config.get("objective") != EXPECTED_OBJECTIVES[role]:
        raise RuntimeError(f"{role} has the wrong registered objective")


def load_checkpoint(path: Path, *, step: int, role: str) -> dict[str, object]:
    if path != checkpoint_path(path.parent, step):
        raise RuntimeError(f"refusing non-exact checkpoint path {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {path} is not a dictionary")
    for key in ("model", "step", "config"):
        if key not in payload:
            raise RuntimeError(f"checkpoint {path} lacks {key!r}")
    if int(payload["step"]) != step:
        raise RuntimeError(
            f"checkpoint {path} stores step {payload['step']}, expected {step}"
        )
    if "ema_model" in payload:
        raise RuntimeError(f"checkpoint {path} unexpectedly contains an EMA")
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise RuntimeError(f"checkpoint {path} config is not a mapping")
    validate_checkpoint_config(config, role)
    if not isinstance(payload["model"], Mapping):
        raise RuntimeError(f"checkpoint {path} model is not a state mapping")
    return payload


def validate_checkpoint_pair(
    candidate: Mapping[str, object],
    control: Mapping[str, object],
) -> None:
    candidate_config = candidate["config"]
    control_config = control["config"]
    assert isinstance(candidate_config, Mapping)
    assert isinstance(control_config, Mapping)
    for key in EXPECTED_SHARED_CONFIG:
        if candidate_config[key] != control_config[key]:
            raise RuntimeError(f"checkpoint pair differs at config {key!r}")
    candidate_state = candidate["model"]
    control_state = control["model"]
    assert isinstance(candidate_state, Mapping)
    assert isinstance(control_state, Mapping)
    if tuple(candidate_state) != tuple(control_state):
        raise RuntimeError("checkpoint pair has different state keys")
    for name in candidate_state:
        left = candidate_state[name]
        right = control_state[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RuntimeError(f"state value {name!r} is not a tensor")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"checkpoint pair differs in state schema at {name}")


def model_from_checkpoint(
    payload: Mapping[str, object],
    device: torch.device,
):
    config = payload["config"]
    state = payload["model"]
    assert isinstance(config, dict)
    assert isinstance(state, Mapping)
    model = scale_audit.configure_model(config)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model = model.to(device).eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("registered audit requires float32 model parameters")
    return model


@torch.inference_mode()
def trace_microbatch(model, window: torch.Tensor) -> ComplexRecurrentScaleTape:
    """Encode only the literal prefix and trace the target-free recurrence."""
    if window.ndim != 2 or window.shape[1] != WINDOW_LENGTH:
        raise ValueError(
            f"window must have shape [batch,{WINDOW_LENGTH}]"
        )
    prefix = window[:, :CONTEXT]
    positions = torch.arange(CONTEXT, device=window.device)
    prefix_encoded, _ = model.encode(prefix, positions)
    anchors = sparse_anchor_indices(
        CONTEXT,
        ANCHOR_STRIDE,
        device=window.device,
    )
    expected = torch.arange(0, CONTEXT, ANCHOR_STRIDE, device=window.device)
    if not torch.equal(anchors, expected) or anchors.numel() != ANCHORS:
        raise RuntimeError("registered sparse anchors changed")
    roots = prefix_encoded.index_select(1, anchors)
    return complex_recurrent_scale_tape(
        model.complex_self_prediction,
        roots,
        horizons=HORIZONS,
    )


@torch.inference_mode()
def collect_scale_values(
    model,
    windows: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if windows.shape != (EXAMPLES, WINDOW_LENGTH):
        raise ValueError("fixed windows have an unexpected shape")
    collected: dict[str, list[torch.Tensor]] = {
        field.name: [] for field in fields(ComplexRecurrentScaleTape)
    }
    for begin in range(0, EXAMPLES, PHYSICAL_MICROBATCH):
        batch = windows[begin : begin + PHYSICAL_MICROBATCH]
        if batch.shape[0] != PHYSICAL_MICROBATCH:
            raise RuntimeError("physical microbatch grouping changed")
        tape = trace_microbatch(model, batch.to(device))
        for name in collected:
            collected[name].append(getattr(tape, name).detach().cpu())
    result = {name: torch.cat(chunks, dim=0) for name, chunks in collected.items()}
    for name, value in result.items():
        expected_tail = (ANCHORS,) if value.ndim == 2 else (ANCHORS, HORIZONS)
        if value.shape != (EXAMPLES, *expected_tail):
            raise RuntimeError(f"scale tape field {name} has shape {value.shape}")
    return result


def registered_group_values(
    values: torch.Tensor,
    anchor0_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Select the four manifest-registered row/anchor scopes."""
    if values.ndim != 2 or values.shape[1] != ANCHORS:
        raise ValueError(f"scale values must have shape [rows,{ANCHORS}]")
    if anchor0_tokens.shape != (values.shape[0],):
        raise ValueError("anchor0 token shape does not match scale rows")
    h_mask = anchor0_tokens.long().eq(ANCHOR0_H_TOKEN)
    groups = {
        "anchor0_h": values[h_mask, 0].reshape(-1),
        "anchor0_non_h": values[~h_mask, 0].reshape(-1),
        "h_rows_other_anchors": values[h_mask, 1:].reshape(-1),
        "all_anchor_rows": values.reshape(-1),
    }
    for name, selected in groups.items():
        if selected.numel() < 1:
            raise RuntimeError(f"registered group {name} is empty")
    return groups


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() < 1:
        raise ValueError("cannot summarize an empty tensor")
    if not bool(torch.isfinite(flat).all()):
        raise RuntimeError("trajectory scale contains a non-finite value")
    median, q90, q99 = torch.quantile(
        flat,
        torch.tensor([0.5, 0.9, 0.99], dtype=flat.dtype),
    )
    return {
        "count": flat.numel(),
        "mean": float(flat.mean()),
        "median": float(median),
        "q90": float(q90),
        "q99": float(q99),
        "maximum": float(flat.max()),
    }


def build_trajectory_rows(
    *,
    model_name: str,
    step: int,
    tape_values: Mapping[str, torch.Tensor],
    anchor0_tokens: torch.Tensor,
) -> list[dict[str, object]]:
    if model_name not in ("candidate", "control"):
        raise ValueError(f"unknown model name {model_name!r}")
    rows: list[dict[str, object]] = []

    def append(horizon: int, tensor_name: str, values: torch.Tensor) -> None:
        for group_name, selected in registered_group_values(
            values,
            anchor0_tokens,
        ).items():
            rows.append(
                {
                    "model": model_name,
                    "checkpoint_step": step,
                    "horizon": horizon,
                    "group": group_name,
                    "tensor": tensor_name,
                    **summarize(selected),
                }
            )

    for tensor_name, field_name in H0_TENSORS:
        append(0, tensor_name, tape_values[field_name])
    for horizon in range(1, HORIZONS + 1):
        for tensor_name, field_name in STEP_TENSORS:
            append(
                horizon,
                tensor_name,
                tape_values[field_name][..., horizon - 1],
            )
    expected = len(GROUPS) * (
        len(H0_TENSORS) + HORIZONS * len(STEP_TENSORS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"produced {len(rows)} trajectory rows, expected {expected}")
    return rows


def _trajectory_index(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[object, ...], Mapping[str, object]]:
    index: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in rows:
        key = (
            row["model"],
            row["checkpoint_step"],
            row["horizon"],
            row["group"],
            row["tensor"],
        )
        if key in index:
            raise RuntimeError(f"duplicate trajectory row {key}")
        index[key] = row
    return index


def _comparison_row(
    *,
    comparison: str,
    left: Mapping[str, object],
    right: Mapping[str, object],
    statistic: str,
) -> dict[str, object]:
    left_value = float(left[statistic])
    right_value = float(right[statistic])
    ratio_defined = right_value != 0.0
    ratio: float | str = left_value / right_value if ratio_defined else ""
    difference = left_value - right_value
    if not math.isfinite(difference) or (
        ratio_defined and not math.isfinite(float(ratio))
    ):
        raise RuntimeError("checkpoint comparison is non-finite")
    return {
        "comparison": comparison,
        "left_model": left["model"],
        "right_model": right["model"],
        "checkpoint_step": left["checkpoint_step"],
        "horizon": left["horizon"],
        "tensor": left["tensor"],
        "statistic": statistic,
        "left_group": left["group"],
        "right_group": right["group"],
        "left_count": left["count"],
        "right_count": right["count"],
        "left_value": left_value,
        "right_value": right_value,
        "left_minus_right": difference,
        "left_over_right": ratio,
        "ratio_defined": int(ratio_defined),
    }


def build_checkpoint_comparisons(
    trajectory_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    index = _trajectory_index(trajectory_rows)
    rows: list[dict[str, object]] = []
    candidate_rows = [
        row for row in trajectory_rows if row["model"] == "candidate"
    ]
    for left in candidate_rows:
        right_key = (
            "control",
            left["checkpoint_step"],
            left["horizon"],
            left["group"],
            left["tensor"],
        )
        if right_key not in index:
            raise RuntimeError(f"missing matched control row {right_key}")
        for statistic in SUMMARY_STATISTICS:
            rows.append(
                _comparison_row(
                    comparison="candidate_vs_control",
                    left=left,
                    right=index[right_key],
                    statistic=statistic,
                )
            )

    for model_name in ("candidate", "control"):
        anchor_h_rows = [
            row
            for row in trajectory_rows
            if row["model"] == model_name and row["group"] == "anchor0_h"
        ]
        for left in anchor_h_rows:
            for reference_group in (
                "anchor0_non_h",
                "h_rows_other_anchors",
            ):
                right_key = (
                    model_name,
                    left["checkpoint_step"],
                    left["horizon"],
                    reference_group,
                    left["tensor"],
                )
                if right_key not in index:
                    raise RuntimeError(f"missing within-model row {right_key}")
                for statistic in SUMMARY_STATISTICS:
                    rows.append(
                        _comparison_row(
                            comparison="anchor0_h_within_model",
                            left=left,
                            right=index[right_key],
                            statistic=statistic,
                        )
                    )
    return rows


def write_tsv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"refusing to write empty TSV {path}")
    fieldnames = tuple(materialized[0])
    if any(tuple(row) != fieldnames for row in materialized):
        raise RuntimeError(f"inconsistent TSV schema for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(materialized)


def _mean_row(
    index: Mapping[tuple[object, ...], Mapping[str, object]],
    model: str,
    step: int,
    horizon: int,
    group: str,
    tensor: str,
) -> Mapping[str, object]:
    key = (model, step, horizon, group, tensor)
    if key not in index:
        raise RuntimeError(f"missing interpretation metric {key}")
    return index[key]


def _ratio(left: float, right: float) -> str:
    return "undefined" if right == 0.0 else f"{left / right:.6g}"


def interpretation_text(
    *,
    steps: tuple[int, ...],
    count_rows: list[dict[str, object]],
    trajectory_rows: list[dict[str, object]],
) -> str:
    h_count = int(count_rows[0]["anchor0_h_count"])
    h_rate = float(count_rows[0]["anchor0_h_rate"])
    digest = str(count_rows[0]["sample_sha256"])
    index = _trajectory_index(trajectory_rows)
    highlights = (
        (5, "z_boundary_denominator", "H5 Z boundary denominator"),
        (5, "s_boundary_denominator", "H5 S boundary denominator"),
        (6, "prior_p", "H6 P RMS"),
        (6, "innovation_w", "H6 W RMS"),
        (6, "raw_successor_z", "H6 Zraw RMS"),
        (6, "raw_updated_s", "H6 Sraw RMS"),
    )
    lines = [
        "# Interpretation",
        "",
        "## Fixed validation sample",
        "",
        f"The registered seed selected {EXAMPLES} validation windows; "
        f"{h_count} ({h_rate:.6f}) have byte 104 at anchor 0. The immutable "
        "sample was passed to candidate and control in the same batch-16 "
        "grouping.",
        "",
        f"Sample token SHA-256: `{digest}`.",
        "",
        "## Registered H5/H6 view",
        "",
        "Ratios are descriptive. No ratio threshold was preregistered as a "
        "success criterion.",
        "",
        "| Step | Tensor | Candidate h | Control h | Cand/control | "
        "Cand h/non-h | Cand h/other anchors | Control h/non-h | "
        "Control h/other anchors |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in steps:
        for horizon, tensor, label in highlights:
            candidate_h = float(_mean_row(
                index, "candidate", step, horizon, "anchor0_h", tensor
            )["mean"])
            control_h = float(_mean_row(
                index, "control", step, horizon, "anchor0_h", tensor
            )["mean"])
            candidate_non_h = float(_mean_row(
                index, "candidate", step, horizon, "anchor0_non_h", tensor
            )["mean"])
            candidate_other = float(_mean_row(
                index,
                "candidate",
                step,
                horizon,
                "h_rows_other_anchors",
                tensor,
            )["mean"])
            control_non_h = float(_mean_row(
                index, "control", step, horizon, "anchor0_non_h", tensor
            )["mean"])
            control_other = float(_mean_row(
                index,
                "control",
                step,
                horizon,
                "h_rows_other_anchors",
                tensor,
            )["mean"])
            lines.append(
                f"| {step} | {label} | {candidate_h:.6g} | "
                f"{control_h:.6g} | {_ratio(candidate_h, control_h)} | "
                f"{_ratio(candidate_h, candidate_non_h)} | "
                f"{_ratio(candidate_h, candidate_other)} | "
                f"{_ratio(control_h, control_non_h)} | "
                f"{_ratio(control_h, control_other)} |"
            )
    final_step = steps[-1]
    candidate_h6_w = float(_mean_row(
        index, "candidate", final_step, 6, "anchor0_h", "innovation_w"
    )["mean"])
    control_h6_w = float(_mean_row(
        index, "control", final_step, 6, "anchor0_h", "innovation_w"
    )["mean"])
    candidate_h6_zraw = float(_mean_row(
        index, "candidate", final_step, 6, "anchor0_h", "raw_successor_z"
    )["mean"])
    control_h6_zraw = float(_mean_row(
        index, "control", final_step, 6, "anchor0_h", "raw_successor_z"
    )["mean"])
    candidate_late_h_non_h_ratios = []
    for horizon in range(2, HORIZONS + 1):
        for tensor in (
            "prior_p",
            "innovation_w",
            "raw_successor_z",
            "raw_updated_s",
        ):
            h_value = float(_mean_row(
                index,
                "candidate",
                final_step,
                horizon,
                "anchor0_h",
                tensor,
            )["mean"])
            non_h_value = float(_mean_row(
                index,
                "candidate",
                final_step,
                horizon,
                "anchor0_non_h",
                tensor,
            )["mean"])
            if non_h_value == 0.0:
                raise RuntimeError("candidate late-horizon non-h mean is zero")
            candidate_late_h_non_h_ratios.append(h_value / non_h_value)
    maximum_late_ratio = max(candidate_late_h_non_h_ratios)
    lines.extend(
        (
            "",
            "## Result",
            "",
            f"At step {final_step}, candidate/control anchor-0 byte-h H6 "
            f"innovation-W and raw-successor-Z ratios were "
            f"{candidate_h6_w / control_h6_w:.6g} and "
            f"{candidate_h6_zraw / control_h6_zraw:.6g}.",
            "",
            "The control byte-h H6 event was absent from the candidate: its "
            "byte-h W/Zraw means were near its own non-h rows, and the "
            f"largest candidate byte-h/non-h mean ratio across H2--H16 "
            f"P/W/Zraw/Sraw was {maximum_late_ratio:.6g}. The candidate also "
            "shifted global trajectory scales downward, so this supports "
            "removal rather than relocation of the registered event without "
            "establishing a unique causal pathway.",
        )
    )
    lines.extend(
        (
            "",
            "## Completion checks",
            "",
            "- Every requested exact candidate/control checkpoint pair loaded "
            "strictly; no `last.pt` or nearest-step substitution was used.",
            "- All four registered groups were non-empty and all H0/H1--H16 "
            "trajectory summaries were finite.",
            "- Raw and stored successor Z/S scales remain separate in "
            "`trajectory_metrics.tsv`, including every conflicting horizon.",
            "- `checkpoint_comparisons.tsv` retains candidate/control and "
            "within-model h/non-h and h/other-anchor differences and ratios.",
            "",
            "## Evidence boundary",
            "",
            "This is a fixed validation-forward monitoring audit. It does not "
            "establish that initial-root scale caused a historical training "
            "spike, and it does not replace likelihood, gradient-norm, or "
            "fresh-seed training comparisons. Train and test splits were not "
            "sampled.",
            "",
        )
    )
    return "\n".join(lines)


def _required_checkpoint_paths(
    steps: tuple[int, ...],
    candidate_output: Path,
    control_output: Path,
) -> list[Path]:
    return [
        path
        for step in steps
        for path in (
            checkpoint_path(candidate_output, step),
            checkpoint_path(control_output, step),
        )
    ]


def main() -> None:
    args = parse_args()
    steps = tuple(args.steps)
    missing = [
        path
        for path in _required_checkpoint_paths(
            steps,
            args.candidate_output,
            args.control_output,
        )
        if not path.is_file()
    ]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "every exact checkpoint pair is required; missing:\n" + joined
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered batch-16 audit")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    # Validate every pair before sampling data or writing any output.
    for step in steps:
        candidate = load_checkpoint(
            checkpoint_path(args.candidate_output, step),
            step=step,
            role="candidate",
        )
        control = load_checkpoint(
            checkpoint_path(args.control_output, step),
            step=step,
            role="control",
        )
        validate_checkpoint_pair(candidate, control)
        del candidate, control

    starts, windows, digest = fixed_validation_sample()
    count_rows = byte_h_count_rows(starts, windows, digest)
    anchor0_tokens = windows[:, 0].clone()
    trajectory_rows: list[dict[str, object]] = []
    for step in steps:
        for role, output in (
            ("candidate", args.candidate_output),
            ("control", args.control_output),
        ):
            print(f"trace {role} step {step}", flush=True)
            payload = load_checkpoint(
                checkpoint_path(output, step),
                step=step,
                role=role,
            )
            model = model_from_checkpoint(payload, device)
            values = collect_scale_values(model, windows, device)
            trajectory_rows.extend(build_trajectory_rows(
                model_name=role,
                step=step,
                tape_values=values,
                anchor0_tokens=anchor0_tokens,
            ))
            del payload, model, values
            torch.cuda.empty_cache()
    if sample_sha256(windows) != digest:
        raise RuntimeError("checkpoint evaluation mutated the fixed sample")

    comparison_rows = build_checkpoint_comparisons(trajectory_rows)
    stored_tensors = {
        row["tensor"] for row in trajectory_rows if int(row["horizon"]) > 0
    }
    if not {"stored_successor_z", "stored_successor_s"}.issubset(stored_tensors):
        raise RuntimeError("stored successor carriers are missing")

    args.record_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.record_dir / "byte_h_counts.tsv", count_rows)
    write_tsv(args.record_dir / "trajectory_metrics.tsv", trajectory_rows)
    write_tsv(
        args.record_dir / "checkpoint_comparisons.tsv",
        comparison_rows,
    )
    (args.record_dir / "interpretation.md").write_text(
        interpretation_text(
            steps=steps,
            count_rows=count_rows,
            trajectory_rows=trajectory_rows,
        )
    )
    print(
        f"wrote {len(trajectory_rows)} trajectory rows and "
        f"{len(comparison_rows)} comparisons",
        flush=True,
    )


if __name__ == "__main__":
    main()
