import copy

import torch
from torch import nn

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.training.ha_skew_window import (
    forward_sparse_compounding_noise_window,
    forward_sparse_clean_window,
    forward_sparse_multi_trajectory_compounding_noise_window,
    operator_orbit,
    relative_mse_rows,
    sparse_compounding_noise_window_loss,
    sparse_clean_window_loss,
    sparse_multi_trajectory_compounding_noise_loss,
)


def make_model():
    torch.manual_seed(47)
    return K1DecoderAblationLM(
        43,
        width=32,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
    )


def test_sparse_window_uses_nonoverlapping_three_token_targets():
    model = make_model().eval()
    window = torch.randint(0, 43, (2, 12))
    output = forward_sparse_clean_window(
        model,
        window,
        prefix_length=9,
        horizons=3,
        anchor_stride=3,
    )
    assert output.anchor_indices.tolist() == [0, 3, 6]
    expected_positions = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    assert torch.equal(
        output.targets,
        window[:, expected_positions],
    )
    assert output.logits.shape == (2, 3, 3, 43)
    assert output.predicted_states.shape == (2, 3, 3, 32)
    assert output.gold_states.requires_grad


def test_sparse_window_ce_and_each_mse_horizon_reach_operator():
    model = make_model()
    window = torch.randint(0, 43, (2, 12))
    _, ce, _, mse_by_horizon, output = sparse_clean_window_loss(
        model,
        window,
        prefix_length=9,
        horizons=3,
        anchor_stride=3,
    )
    ce_gradient = torch.autograd.grad(
        ce, model.operator.weight, retain_graph=True
    )[0]
    assert torch.isfinite(ce_gradient).all()
    assert ce_gradient.norm() > 0
    for horizon in range(3):
        gradient = torch.autograd.grad(
            mse_by_horizon[horizon],
            model.operator.weight,
            retain_graph=True,
        )[0]
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    target_gradient = torch.autograd.grad(
        mse_by_horizon.mean(), output.gold_states
    )[0]
    assert torch.isfinite(target_gradient).all()
    assert target_gradient.norm() > 0


def test_truncated_operator_orbit_stops_cross_horizon_state_gradient():
    model = make_model()
    roots = torch.randn(2, 3, model.width, requires_grad=True)

    full_orbit = operator_orbit(model, roots, 3)
    full_root_gradient = torch.autograd.grad(
        full_orbit[:, :, 1].square().sum(),
        roots,
    )[0]
    assert full_root_gradient.norm() > 0

    truncated_orbit = operator_orbit(
        model,
        roots,
        3,
        detach_between_horizons=True,
    )
    horizon_two_loss = truncated_orbit[:, :, 1].square().sum()
    truncated_root_gradient = torch.autograd.grad(
        horizon_two_loss,
        roots,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert (
        truncated_root_gradient is None
        or torch.count_nonzero(truncated_root_gradient) == 0
    )
    operator_gradient = torch.autograd.grad(
        horizon_two_loss,
        model.operator.weight,
    )[0]
    assert torch.isfinite(operator_gradient).all()
    assert operator_gradient.norm() > 0


def test_detached_gold_targets_keep_operator_gradient_only():
    model = make_model()
    window = torch.randint(0, 43, (2, 12))
    _, _, mse, _, output = sparse_clean_window_loss(
        model,
        window,
        prefix_length=9,
        horizons=3,
        anchor_stride=3,
        detach_gold_targets=True,
    )
    target_gradient = torch.autograd.grad(
        mse,
        output.gold_states,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert target_gradient is None
    operator_gradient = torch.autograd.grad(
        mse,
        model.operator.weight,
    )[0]
    assert torch.isfinite(operator_gradient).all()
    assert operator_gradient.norm() > 0


class RecordingSigmaPredictor(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((width,), -3.0))
        self.last_input = None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.last_input = states
        return self.log_sigma.view(1, 1, 1, -1).expand_as(states)


def test_compounding_noise_matches_propagation_formula():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    with torch.no_grad():
        orthogonal, _ = torch.linalg.qr(
            torch.randn(model.width, model.width)
        )
        model.operator.weight.copy_(orthogonal)
    window = torch.randint(0, 43, (2, 12))
    output = forward_sparse_compounding_noise_window(
        model,
        window,
        prefix_length=9,
        horizons=3,
        anchor_stride=3,
        noise_generator=torch.Generator().manual_seed(88),
    )
    assert torch.equal(
        model.sigma_predictor.last_input,
        output.clean_states,
    )
    weight = model.operator.weight
    expected_h2 = output.clean_states[:, :, 1] + torch.nn.functional.linear(
        output.noise[:, :, 0],
        weight,
    )
    expected_h3 = output.clean_states[:, :, 2] + torch.nn.functional.linear(
        torch.nn.functional.linear(output.noise[:, :, 0], weight)
        + output.noise[:, :, 1],
        weight,
    )
    torch.testing.assert_close(output.predicted_states[:, :, 0], output.clean_states[:, :, 0])
    torch.testing.assert_close(output.predicted_states[:, :, 1], expected_h2)
    torch.testing.assert_close(output.predicted_states[:, :, 2], expected_h3)
    torch.testing.assert_close(
        output.decode_states,
        output.predicted_states + output.noise,
    )


def test_compounding_noise_loss_reaches_sigma_k_and_attached_targets():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    window = torch.randint(0, 43, (2, 12))
    _, _, mse, _, output = sparse_compounding_noise_window_loss(
        model,
        window,
        prefix_length=9,
        horizons=3,
        anchor_stride=3,
        noise_generator=torch.Generator().manual_seed(89),
    )
    sigma_gradient, operator_gradient, target_gradient = (
        torch.autograd.grad(
            mse,
            (
                model.sigma_predictor.log_sigma,
                model.operator.weight,
                output.gold_states,
            ),
        )
    )
    for gradient in (
        sigma_gradient,
        operator_gradient,
        target_gradient,
    ):
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_multi_trajectory_score_is_four_token_ce_sum_and_selector_is_detached():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    window = torch.randint(0, 43, (2, 13))
    loss, marginal_ce, _, output = (
        sparse_multi_trajectory_compounding_noise_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            temperature=0.7,
            mse_weight=0.0,
            noise_generator=torch.Generator().manual_seed(90),
        )
    )
    assert output.token_ce.shape == (2, 3, 3, 4)
    assert output.trajectory_ce.shape == (2, 3, 3)
    assert output.responsibilities.shape == (2, 3, 3)
    torch.testing.assert_close(
        output.trajectory_ce,
        output.token_ce.sum(dim=-1),
    )
    torch.testing.assert_close(
        output.responsibilities,
        torch.softmax(-output.trajectory_ce / 0.7, dim=1),
    )
    torch.testing.assert_close(
        output.responsibilities.sum(dim=1),
        torch.ones_like(output.responsibilities[:, 0]),
    )
    assert not output.responsibilities.requires_grad
    torch.testing.assert_close(loss, marginal_ce)


def test_multi_trajectory_ce_only_reaches_noise_and_k_but_not_mse_target():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    window = torch.randint(0, 43, (2, 13))
    loss, _, _, output = sparse_multi_trajectory_compounding_noise_loss(
        model,
        window,
        prefix_length=9,
        horizons=4,
        anchor_stride=3,
        trajectories=3,
        mse_weight=0.0,
        noise_generator=torch.Generator().manual_seed(91),
    )
    sigma_gradient, operator_gradient, target_gradient = torch.autograd.grad(
        loss,
        (
            model.sigma_predictor.log_sigma,
            model.operator.weight,
            output.sparse.gold_states,
        ),
        allow_unused=True,
    )
    for gradient in (sigma_gradient, operator_gradient):
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    assert target_gradient is None


def test_ce_only_objective_is_unchanged_by_inactive_h1_mse_options():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    window = torch.randint(0, 43, (2, 13))
    baseline = sparse_multi_trajectory_compounding_noise_loss(
        model,
        window,
        prefix_length=9,
        horizons=4,
        anchor_stride=3,
        trajectories=3,
        temperature=0.05,
        mse_weight=0.0,
        noise_generator=torch.Generator().manual_seed(911),
    )
    explicit_h1 = sparse_multi_trajectory_compounding_noise_loss(
        model,
        window,
        prefix_length=9,
        horizons=4,
        anchor_stride=3,
        trajectories=3,
        temperature=0.05,
        mse_weight=0.0,
        mse_horizons=(1,),
        detach_mse_targets=False,
        noise_generator=torch.Generator().manual_seed(911),
    )
    torch.testing.assert_close(baseline[0], explicit_h1[0], rtol=0, atol=0)
    torch.testing.assert_close(
        baseline[3].sparse.logits,
        explicit_h1[3].sparse.logits,
        rtol=0,
        atol=0,
    )
    assert baseline[3].trajectory_mse is None
    assert explicit_h1[3].trajectory_mse is None


def test_multi_trajectory_optional_mse_uses_same_responsibilities():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    window = torch.randint(0, 43, (2, 13))
    loss, ce, mse, output = (
        sparse_multi_trajectory_compounding_noise_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            mse_weight=0.25,
            noise_generator=torch.Generator().manual_seed(92),
        )
    )
    torch.testing.assert_close(loss, ce + 0.25 * mse)
    expected_mse = (
        output.responsibilities * output.trajectory_mse
    ).sum(dim=1).mean()
    torch.testing.assert_close(mse, expected_mse)
    target_gradient = torch.autograd.grad(
        loss,
        output.sparse.gold_states,
    )[0]
    assert torch.isfinite(target_gradient).all()
    assert target_gradient.norm() > 0


def test_multi_trajectory_h1_mse_uses_online_attached_clean_state_only():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    window = torch.randint(0, 43, (2, 13))
    loss, ce, mse, output = (
        sparse_multi_trajectory_compounding_noise_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            temperature=0.05,
            mse_weight=1.0,
            mse_horizons=(1,),
            detach_mse_targets=False,
            noise_generator=torch.Generator().manual_seed(921),
        )
    )
    expected = relative_mse_rows(
        output.sparse.clean_states[:, :, 0],
        output.sparse.gold_states[:, :, 0],
    ).mean()
    torch.testing.assert_close(loss, ce + mse)
    torch.testing.assert_close(mse, expected)

    target_gradient, sigma_gradient = torch.autograd.grad(
        mse,
        (
            output.sparse.gold_states,
            model.sigma_predictor.log_sigma,
        ),
        allow_unused=True,
    )
    assert target_gradient[:, :, 0].norm() > 0
    assert torch.count_nonzero(target_gradient[:, :, 1:]) == 0
    assert (
        sigma_gradient is None
        or torch.count_nonzero(sigma_gradient) == 0
    )


def test_shared_multi_trajectory_matches_expanded_prefix_reference():
    shared_model = make_model().eval()
    shared_model.sigma_predictor = RecordingSigmaPredictor(
        shared_model.width
    )
    expanded_model = copy.deepcopy(shared_model)
    window = torch.randint(0, 43, (2, 13))
    trajectories = 3
    expanded_window = (
        window[:, None]
        .expand(2, trajectories, window.shape[1])
        .reshape(2 * trajectories, window.shape[1])
    )
    expanded = forward_sparse_compounding_noise_window(
        expanded_model,
        expanded_window,
        prefix_length=9,
        horizons=4,
        anchor_stride=3,
        noise_generator=torch.Generator().manual_seed(93),
    )
    shared = forward_sparse_multi_trajectory_compounding_noise_window(
        shared_model,
        window,
        prefix_length=9,
        horizons=4,
        anchor_stride=3,
        trajectories=trajectories,
        noise_generator=torch.Generator().manual_seed(93),
    )
    for shared_value, expanded_value in (
        (shared.logits.reshape_as(expanded.logits), expanded.logits),
        (
            shared.decoded_hidden.reshape_as(expanded.decoded_hidden),
            expanded.decoded_hidden,
        ),
        (
            shared.predicted_states.reshape_as(
                expanded.predicted_states
            ),
            expanded.predicted_states,
        ),
        (
            shared.decode_states.reshape_as(expanded.decode_states),
            expanded.decode_states,
        ),
        (
            shared.clean_states[:, None]
            .expand(2, trajectories, 3, 4, shared_model.width)
            .reshape_as(expanded.clean_states),
            expanded.clean_states,
        ),
        (
            shared.gold_states[:, None]
            .expand(2, trajectories, 3, 4, shared_model.width)
            .reshape_as(expanded.gold_states),
            expanded.gold_states,
        ),
        (shared.noise.reshape_as(expanded.noise), expanded.noise),
        (
            shared.log_sigma[:, None]
            .expand(2, trajectories, 3, 4, shared_model.width)
            .reshape_as(expanded.log_sigma),
            expanded.log_sigma,
        ),
    ):
        torch.testing.assert_close(
            shared_value,
            expanded_value,
            rtol=2e-5,
            atol=2e-6,
        )
    shared_targets = (
        shared.targets[:, None]
        .expand(2, trajectories, 3, 4)
        .reshape_as(expanded.targets)
    )
    assert torch.equal(shared_targets, expanded.targets)

    shared_loss = torch.nn.functional.cross_entropy(
        shared.logits.reshape(-1, shared.logits.shape[-1]),
        shared_targets.flatten(),
    )
    expanded_loss = torch.nn.functional.cross_entropy(
        expanded.logits.flatten(0, 2),
        expanded.targets.flatten(),
    )
    shared_gradients = torch.autograd.grad(
        shared_loss,
        (
            shared_model.operator.weight,
            shared_model.sigma_predictor.log_sigma,
            shared_model.embedding_weight,
        ),
    )
    expanded_gradients = torch.autograd.grad(
        expanded_loss,
        (
            expanded_model.operator.weight,
            expanded_model.sigma_predictor.log_sigma,
            expanded_model.embedding_weight,
        ),
    )
    for shared_gradient, expanded_gradient in zip(
        shared_gradients,
        expanded_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            shared_gradient,
            expanded_gradient,
            rtol=3e-4,
            atol=2e-6,
        )
