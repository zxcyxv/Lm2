from dataclasses import fields

import pytest
import torch
from torch import nn

import eval_byte256_complex_self_predicted_kv_initial_root_rmsnorm_anchor0_h_monitor as monitor
from rotlm.models.complex_self_prediction import (
    ComplexSelfPredictedKVTransition,
)


def _tape_values(scale: float) -> dict[str, torch.Tensor]:
    step_fields = {field_name for _, field_name in monitor.STEP_TENSORS}
    return {
        field.name: torch.full(
            (4, monitor.ANCHORS, monitor.HORIZONS)
            if field.name in step_fields
            else (4, monitor.ANCHORS),
            scale,
        )
        for field in fields(monitor.ComplexRecurrentScaleTape)
    }


def _registered_config(role: str) -> dict[str, object]:
    config = dict(monitor.EXPECTED_SHARED_CONFIG)
    config["normalize_initial_recurrent_root"] = role == "candidate"
    config["objective"] = monitor.EXPECTED_OBJECTIVES[role]
    return config


def test_parse_steps_requires_exact_strictly_increasing_steps():
    assert monitor.parse_steps("100,500,1000") == (100, 500, 1000)
    with pytest.raises(Exception, match="strictly increasing"):
        monitor.parse_steps("100,100")
    with pytest.raises(Exception, match="positive"):
        monitor.parse_steps("0,100")
    with pytest.raises(Exception, match="comma-separated"):
        monitor.parse_steps("100,")


def test_checkpoint_path_never_substitutes_last_or_nearest(tmp_path):
    assert monitor.checkpoint_path(tmp_path, 100) == tmp_path / "step0100.pt"
    assert monitor.checkpoint_path(tmp_path, 1000) == tmp_path / "step1000.pt"


def test_checkpoint_config_enforces_the_sole_intervention():
    monitor.validate_checkpoint_config(
        _registered_config("candidate"),
        "candidate",
    )
    monitor.validate_checkpoint_config(
        _registered_config("control"),
        "control",
    )
    legacy_control = _registered_config("control")
    del legacy_control["normalize_initial_recurrent_root"]
    monitor.validate_checkpoint_config(legacy_control, "control")

    missing_candidate = _registered_config("candidate")
    del missing_candidate["normalize_initial_recurrent_root"]
    with pytest.raises(RuntimeError, match="initial-root normalization"):
        monitor.validate_checkpoint_config(missing_candidate, "candidate")

    wrong = _registered_config("candidate")
    wrong["normalize_initial_recurrent_root"] = False
    with pytest.raises(RuntimeError, match="initial-root normalization"):
        monitor.validate_checkpoint_config(wrong, "candidate")


def test_registered_groups_preserve_exact_row_and_anchor_scopes():
    values = torch.arange(4 * monitor.ANCHORS, dtype=torch.float32).reshape(
        4,
        monitor.ANCHORS,
    )
    anchor0_tokens = torch.tensor([104, 7, 104, 8])
    groups = monitor.registered_group_values(values, anchor0_tokens)

    assert groups["anchor0_h"].tolist() == [0.0, 32.0]
    assert groups["anchor0_non_h"].tolist() == [16.0, 48.0]
    assert groups["h_rows_other_anchors"].numel() == 30
    assert groups["h_rows_other_anchors"][:3].tolist() == [1.0, 2.0, 3.0]
    assert groups["h_rows_other_anchors"][-3:].tolist() == [45.0, 46.0, 47.0]
    assert groups["all_anchor_rows"].numel() == 64


def test_summary_retains_registered_quantiles():
    summary = monitor.summarize(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5
    assert summary["q90"] == pytest.approx(3.7)
    assert summary["q99"] == pytest.approx(3.97)
    assert summary["maximum"] == 4.0


def test_trajectory_and_comparison_schemas_cover_all_registered_scopes():
    anchor0_tokens = torch.tensor([104, 7, 104, 8])
    candidate = monitor.build_trajectory_rows(
        model_name="candidate",
        step=100,
        tape_values=_tape_values(2.0),
        anchor0_tokens=anchor0_tokens,
    )
    control = monitor.build_trajectory_rows(
        model_name="control",
        step=100,
        tape_values=_tape_values(1.0),
        anchor0_tokens=anchor0_tokens,
    )
    per_model = len(monitor.GROUPS) * (
        len(monitor.H0_TENSORS)
        + monitor.HORIZONS * len(monitor.STEP_TENSORS)
    )
    assert len(candidate) == per_model
    assert len(control) == per_model

    comparisons = monitor.build_checkpoint_comparisons(candidate + control)
    assert len(comparisons) == per_model * len(monitor.SUMMARY_STATISTICS) * 2
    selected = next(
        row
        for row in comparisons
        if row["comparison"] == "candidate_vs_control"
        and row["horizon"] == 0
        and row["tensor"] == "raw_encoded_root"
        and row["left_group"] == "anchor0_h"
        and row["statistic"] == "mean"
    )
    assert selected["left_minus_right"] == 1.0
    assert selected["left_over_right"] == 2.0
    assert selected["ratio_defined"] == 1

    within = next(
        row
        for row in comparisons
        if row["comparison"] == "anchor0_h_within_model"
        and row["left_model"] == "candidate"
        and row["horizon"] == 6
        and row["tensor"] == "innovation_w"
        and row["right_group"] == "h_rows_other_anchors"
        and row["statistic"] == "q99"
    )
    assert within["left_over_right"] == 1.0


class _PrefixOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.complex_self_prediction = ComplexSelfPredictedKVTransition(
            8,
            heads=2,
            key_dim=2,
            value_dim=2,
            learned_read_beta=False,
            recurrent_hidden_mode="post-update-full-read",
            pre_normalize_qkv_inputs=False,
            normalize_initial_recurrent_root=True,
            post_normalize_recurrent_state=True,
        )
        self.seen_tokens: torch.Tensor | None = None

    def encode(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        assert positions.shape == (monitor.CONTEXT,)
        self.seen_tokens = tokens.detach().clone()
        basis = torch.arange(1, 9, device=tokens.device, dtype=torch.float32)
        encoded = (tokens.float().unsqueeze(-1) + 1.0) * basis
        return encoded, None

    def exact_inverse_selected_decode_tape_states(self, *args, **kwargs):
        raise AssertionError("the monitoring evaluator must not call a decoder")


def test_trace_microbatch_uses_only_prefix_and_never_decodes():
    model = _PrefixOnlyModel().eval()
    window = torch.zeros(2, monitor.WINDOW_LENGTH, dtype=torch.long)
    window[:, monitor.CONTEXT :] = 255
    tape = monitor.trace_microbatch(model, window)

    assert model.seen_tokens is not None
    torch.testing.assert_close(model.seen_tokens, window[:, : monitor.CONTEXT])
    assert tape.raw_root_hidden_rms.shape == (2, monitor.ANCHORS)
    assert tape.raw_full_hidden_rms.shape == (
        2,
        monitor.ANCHORS,
        monitor.HORIZONS,
    )
    assert bool(torch.isfinite(tape.raw_full_hidden_rms).all())
