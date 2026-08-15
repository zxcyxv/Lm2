import copy

import torch
from torch import nn
import torch.nn.functional as F

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.prefix_orthogonal_scan import (
    PrefixConditionedOrthogonalAffineScan,
    apply_pairwise_affine_scan,
    rotate_pairwise,
    sequential_pairwise_affine,
)
from rotlm.models.token_conditioned_transition import (
    TokenConditionedTransition,
)
from rotlm.models.spectral_rotation import SpectralRotationOperator
from rotlm.models.trajectory_initializer import InitialConditionTrajectory
from rotlm.models.trajectory_innovation import (
    SelectiveInnovationTrajectory,
    diagonal_affine_scan,
    sequential_diagonal_affine,
)
from rotlm.models.trajectory_operator import BranchPlaneRotationTrajectory
from rotlm.training.ha_skew_window import (
    forward_sparse_compounding_noise_window,
    forward_sparse_clean_window,
    forward_sparse_multi_trajectory_compounding_noise_window,
    forward_sparse_multi_trajectory_initial_condition_window,
    forward_sparse_multi_trajectory_operator_commitment_window,
    forward_sparse_multi_trajectory_selective_innovation_window,
    operator_orbit,
    relative_mse_rows,
    sparse_compounding_noise_window_loss,
    sparse_clean_window_loss,
    sparse_multi_trajectory_compounding_noise_loss,
    sparse_multi_trajectory_initial_condition_loss,
    sparse_multi_trajectory_operator_commitment_loss,
    sparse_multi_trajectory_selective_innovation_loss,
    sparse_selective_innovation_joint_nll_behavioral_closure_loss,
    sparse_spectral_initial_condition_joint_nll_loss,
    sparse_prefix_orthogonal_affine_scan_loss,
    sparse_token_conditioned_sequential_loss,
    symmetric_state_info_nce,
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


def test_explicit_spectral_rotation_matches_recurrence_and_preserves_geometry():
    operator = SpectralRotationOperator(
        32,
        initial_frequency_range=0.05,
    )
    values = torch.randn(3, 5, 32)
    other = torch.randn_like(values)
    direct = operator.apply_power(values, 7)
    recurrent = values
    for _ in range(7):
        recurrent = operator(recurrent)
    torch.testing.assert_close(direct, recurrent, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        direct.square().sum(dim=-1),
        values.square().sum(dim=-1),
        atol=2e-5,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        (operator(values) * operator(other)).sum(dim=-1),
        (values * other).sum(dim=-1),
        atol=2e-5,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        operator(values),
        F.linear(values, operator.weight),
        atol=1e-6,
        rtol=1e-6,
    )
    loss = direct.square().mul(torch.randn_like(direct)).sum()
    gradient = torch.autograd.grad(loss, operator.A)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_spectral_hypothesis_joint_nll_is_exact_and_has_no_prior_kl():
    model = make_model()
    model.operator = SpectralRotationOperator(
        model.width,
        initial_frequency_range=0.05,
    )
    model.trajectory_initializer = InitialConditionTrajectory(
        model.width,
        3,
        code_width=8,
        bottleneck=16,
        init_scale=0.05,
    )
    window = torch.randint(0, 43, (2, 12))
    loss, joint_nll, h1_closure, output = (
        sparse_spectral_initial_condition_joint_nll_loss(
            model,
            window,
            prefix_length=9,
            horizons=3,
            anchor_stride=3,
            trajectories=3,
            h1_closure_weight=1.0,
        )
    )
    explicit = -torch.log(
        (
            output.sparse.prior_probs.float()
            * (-output.trajectory_ce.float()).exp()
        ).sum(dim=1)
    ).mean() / 3
    torch.testing.assert_close(joint_nll, explicit)
    torch.testing.assert_close(loss, joint_nll + h1_closure)
    assert output.prior_loss is None
    assert not output.responsibilities.requires_grad
    torch.testing.assert_close(
        output.responsibilities.sum(dim=1),
        torch.ones_like(output.responsibilities[:, 0]),
    )
    frequency_gradient, prior_gradient = torch.autograd.grad(
        joint_nll,
        (
            model.operator.A,
            model.trajectory_initializer.prior.weight,
        ),
    )
    assert torch.isfinite(frequency_gradient).all()
    assert frequency_gradient.norm() > 0
    assert torch.isfinite(prior_gradient).all()
    assert prior_gradient.norm() > 0


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


def test_symmetric_state_info_nce_prefers_matched_attached_pairs():
    torch.manual_seed(922)
    target = torch.randn(4, 3, 16, requires_grad=True)
    prediction = (
        target.detach() + 0.05 * torch.randn_like(target)
    ).requires_grad_()
    loss, positive, negative, accuracy = symmetric_state_info_nce(
        prediction,
        target,
        temperature=0.2,
    )
    assert torch.isfinite(loss)
    assert positive < negative
    assert accuracy == 1
    prediction_gradient, target_gradient = torch.autograd.grad(
        loss, (prediction, target)
    )
    for gradient in (prediction_gradient, target_gradient):
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_multi_trajectory_h1_state_nce_is_attached_without_mse():
    model = make_model()
    model.sigma_predictor = RecordingSigmaPredictor(model.width)
    window = torch.randint(0, 43, (3, 13))
    loss, ce, mse, output = (
        sparse_multi_trajectory_compounding_noise_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            temperature=0.05,
            mse_weight=0.0,
            state_nce_weight=1.0,
            state_nce_temperature=0.2,
            state_nce_horizon=1,
            noise_generator=torch.Generator().manual_seed(923),
        )
    )
    torch.testing.assert_close(loss, ce + output.state_nce)
    torch.testing.assert_close(mse, torch.zeros_like(mse))
    assert output.trajectory_mse is None

    target_gradient = torch.autograd.grad(
        output.state_nce,
        output.sparse.gold_states,
        retain_graph=True,
    )[0]
    assert target_gradient[:, :, 0].norm() > 0
    assert torch.count_nonzero(target_gradient[:, :, 1:]) == 0
    operator_gradient = torch.autograd.grad(
        output.state_nce,
        model.operator.weight,
    )[0]
    assert torch.isfinite(operator_gradient).all()
    assert operator_gradient.norm() > 0


def test_initial_condition_branch_distance_is_preserved_by_orthogonal_k():
    model = make_model()
    model.trajectory_initializer = InitialConditionTrajectory(
        model.width,
        3,
        code_width=8,
        bottleneck=12,
    )
    with torch.no_grad():
        orthogonal, _ = torch.linalg.qr(
            torch.randn(model.width, model.width)
        )
        model.operator.weight.copy_(orthogonal)
    window = torch.randint(0, 43, (2, 13))
    output, distance_error = (
        forward_sparse_multi_trajectory_initial_condition_window(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
        )
    )
    assert output.prior_probs is not None
    assert output.branch_residuals is not None
    assert output.prior_probs.shape == (2, 3, 3)
    assert output.predicted_states.shape == (2, 3, 3, 4, 32)
    weighted_centroid = (
        output.prior_probs[..., None]
        * output.branch_residuals[:, :, :, 0]
    ).sum(dim=1)
    torch.testing.assert_close(
        weighted_centroid,
        torch.zeros_like(weighted_centroid),
        atol=2e-6,
        rtol=0,
    )
    assert distance_error < 2e-5


def test_initial_condition_loss_reaches_prior_initializer_k_and_h1_target():
    model = make_model()
    model.trajectory_initializer = InitialConditionTrajectory(
        model.width,
        3,
        code_width=8,
        bottleneck=12,
    )
    window = torch.randint(0, 43, (3, 13))
    loss, marginal_ce, mse, output = (
        sparse_multi_trajectory_initial_condition_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            temperature=0.05,
            mse_weight=1.0,
            mse_horizons=(1,),
            prior_weight=1.0,
        )
    )
    assert output.sparse.prior_probs is not None
    expected_responsibilities = torch.softmax(
        output.sparse.prior_probs.float().clamp_min(1e-8).log()
        - output.trajectory_ce / 0.05,
        dim=1,
    )
    torch.testing.assert_close(
        output.responsibilities,
        expected_responsibilities,
    )
    torch.testing.assert_close(
        loss,
        marginal_ce + mse + output.prior_loss,
    )
    assert not output.responsibilities.requires_grad

    gradients = torch.autograd.grad(
        loss,
        (
            model.operator.weight,
            model.trajectory_initializer.output.weight,
            model.trajectory_initializer.prior.weight,
            output.sparse.gold_states,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    target_gradient = gradients[-1]
    assert target_gradient[:, :, 0].norm() > 0
    assert torch.count_nonzero(target_gradient[:, :, 1:]) == 0


def test_operator_commitment_reuses_one_orthogonal_transition():
    model = make_model()
    model.trajectory_operator = BranchPlaneRotationTrajectory(
        model.width,
        3,
        code_width=8,
        bottleneck=12,
    )
    with torch.no_grad():
        orthogonal, _ = torch.linalg.qr(
            torch.randn(model.width, model.width)
        )
        model.operator.weight.copy_(orthogonal)
    window = torch.randint(0, 43, (2, 13))
    output, distance_error = (
        forward_sparse_multi_trajectory_operator_commitment_window(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
        )
    )
    assert distance_error is None
    assert output.prior_probs is not None
    assert output.branch_residuals is not None
    assert output.branch_operator_orthogonality_error is not None
    assert output.branch_operator_orthogonality_error < 2e-5
    assert output.predicted_states.shape == (2, 3, 3, 4, 32)

    positions = torch.arange(window.shape[1])
    encoded, _ = model.encode(window, positions)
    roots = encoded[:, :9].index_select(1, output.anchor_indices)
    first, second, angles, _ = model.trajectory_operator(roots)
    current = roots[:, None].expand(2, 3, 3, model.width)
    manual = []
    for _ in range(4):
        before_rotation = model.operator(current)
        current = model.trajectory_operator.rotate(
            before_rotation,
            first,
            second,
            angles,
        )
        torch.testing.assert_close(
            current.float().norm(dim=-1),
            before_rotation.float().norm(dim=-1),
            atol=2e-5,
            rtol=2e-5,
        )
        manual.append(current)
    torch.testing.assert_close(
        output.predicted_states,
        torch.stack(manual, dim=3),
    )


def test_operator_commitment_loss_reaches_prior_plane_angle_k_and_h1_target():
    model = make_model()
    model.trajectory_operator = BranchPlaneRotationTrajectory(
        model.width,
        3,
        code_width=8,
        bottleneck=12,
    )
    window = torch.randint(0, 43, (3, 13))
    loss, marginal_ce, mse, output = (
        sparse_multi_trajectory_operator_commitment_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            temperature=0.05,
            mse_weight=1.0,
            mse_horizons=(1,),
            prior_weight=1.0,
        )
    )
    torch.testing.assert_close(
        loss,
        marginal_ce + mse + output.prior_loss,
    )
    assert not output.responsibilities.requires_grad
    gradients = torch.autograd.grad(
        loss,
        (
            model.operator.weight,
            model.trajectory_operator.plane.weight,
            model.trajectory_operator.angle.weight,
            model.trajectory_operator.prior.weight,
            output.sparse.gold_states,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    target_gradient = gradients[-1]
    assert target_gradient[:, :, 0].norm() > 0
    assert torch.count_nonzero(target_gradient[:, :, 1:]) == 0


def test_diagonal_affine_scan_matches_sequential_recurrence():
    torch.manual_seed(20260725)
    multipliers = torch.sigmoid(torch.randn(2, 3, 4, 7))
    increments = torch.randn(2, 3, 4, 7)
    scanned = diagonal_affine_scan(multipliers, increments)
    sequential = sequential_diagonal_affine(
        multipliers,
        increments,
    )
    torch.testing.assert_close(scanned, sequential)


def test_selective_innovation_is_seeded_distinct_and_centered():
    model = make_model()
    module = SelectiveInnovationTrajectory(
        model.width,
        3,
        noise_width=8,
        state_width=12,
    )
    roots = torch.randn(2, 3, model.width)
    first = module(
        roots,
        4,
        noise_generator=torch.Generator().manual_seed(101),
    )
    repeated = module(
        roots,
        4,
        noise_generator=torch.Generator().manual_seed(101),
    )
    changed = module(
        roots,
        4,
        noise_generator=torch.Generator().manual_seed(102),
    )
    torch.testing.assert_close(first[0], repeated[0], rtol=0, atol=0)
    assert not torch.equal(first[0], changed[0])
    weighted_centroid = (
        first[1][..., None, None] * first[0]
    ).sum(dim=1)
    torch.testing.assert_close(
        weighted_centroid,
        torch.zeros_like(weighted_centroid),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        diagonal_affine_scan(first[4], first[5]),
        sequential_diagonal_affine(first[4], first[5]),
    )


def test_selective_innovation_loss_uses_ku_plus_r_and_reaches_parameters():
    model = make_model()
    model.trajectory_innovation = SelectiveInnovationTrajectory(
        model.width,
        3,
        noise_width=8,
        state_width=12,
    )
    window = torch.randint(0, 43, (3, 13))
    generator_seed = 103
    forward, distance_error = (
        forward_sparse_multi_trajectory_selective_innovation_window(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            noise_generator=torch.Generator().manual_seed(
                generator_seed
            ),
        )
    )
    assert distance_error is None
    positions = torch.arange(window.shape[1])
    encoded, _ = model.encode(window, positions)
    roots = encoded[:, :9].index_select(1, forward.anchor_indices)
    current = roots[:, None].expand(3, 3, 3, model.width)
    manual = []
    for horizon in range(4):
        current = (
            model.operator(current)
            + forward.branch_residuals[:, :, :, horizon]
        )
        manual.append(current)
    torch.testing.assert_close(
        forward.predicted_states,
        torch.stack(manual, dim=3),
    )

    loss, marginal_ce, mse, output = (
        sparse_multi_trajectory_selective_innovation_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            temperature=0.05,
            mse_weight=1.0,
            mse_horizons=(1,),
            prior_weight=1.0,
            noise_generator=torch.Generator().manual_seed(
                generator_seed
            ),
        )
    )
    torch.testing.assert_close(
        loss,
        marginal_ce + mse + output.prior_loss,
    )
    gradients = torch.autograd.grad(
        loss,
        (
            model.operator.weight,
            model.trajectory_innovation.noise.weight,
            model.trajectory_innovation.decay.weight,
            model.trajectory_innovation.output.weight,
            model.trajectory_innovation.prior.weight,
            output.sparse.gold_states,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    target_gradient = gradients[-1]
    assert target_gradient[:, :, 0].norm() > 0
    assert torch.count_nonzero(target_gradient[:, :, 1:]) == 0


def test_selective_joint_nll_closes_future_behavior_on_same_tape():
    model = make_model()
    model.trajectory_innovation = SelectiveInnovationTrajectory(
        model.width,
        3,
        noise_width=8,
        state_width=12,
    )
    window = torch.randint(0, 43, (3, 13))
    loss, joint_nll, closure, output = (
        sparse_selective_innovation_joint_nll_behavioral_closure_loss(
            model,
            window,
            prefix_length=9,
            horizons=4,
            anchor_stride=3,
            trajectories=3,
            closure_weight=1.0,
            noise_generator=torch.Generator().manual_seed(107),
        )
    )
    sparse = output.sparse
    explicit_joint_nll = -torch.log(
        (
            sparse.prior_probs.float()
            * (-output.trajectory_ce.float()).exp()
        ).sum(dim=1)
    ).mean() / 4
    torch.testing.assert_close(joint_nll, explicit_joint_nll)
    torch.testing.assert_close(loss, joint_nll + closure)
    assert not output.responsibilities.requires_grad
    torch.testing.assert_close(
        output.responsibilities.sum(dim=1),
        torch.ones_like(output.responsibilities[:, 0]),
    )
    assert output.prior_loss is None
    assert output.trajectory_mse is None
    assert closure.isfinite() and closure >= -1e-6
    assert output.behavioral_closure_by_horizon.shape == (3,)

    assert output.reanchored_states is not None
    assert not output.reanchored_states.requires_grad
    expected_h1 = sparse.gold_states[:, :, 0][:, None].expand(
        3,
        3,
        3,
        model.width,
    )
    torch.testing.assert_close(
        output.reanchored_states[:, :, :, 0],
        expected_h1,
    )
    current = expected_h1
    manual_reanchored = [current]
    for horizon in range(1, 4):
        current = (
            model.operator(current)
            + sparse.branch_residuals[:, :, :, horizon]
        )
        manual_reanchored.append(current)
    torch.testing.assert_close(
        output.reanchored_states,
        torch.stack(manual_reanchored, dim=3),
    )

    joint_gradients = torch.autograd.grad(
        joint_nll,
        (
            model.operator.weight,
            model.trajectory_innovation.noise.weight,
            model.trajectory_innovation.decay.weight,
            model.trajectory_innovation.output.weight,
            model.trajectory_innovation.scale.weight,
            model.trajectory_innovation.prior.weight,
        ),
        retain_graph=True,
    )
    for gradient in joint_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0

    closure_gradients = torch.autograd.grad(
        closure,
        (
            model.operator.weight,
            model.trajectory_innovation.noise.weight,
            model.trajectory_innovation.output.weight,
        ),
        retain_graph=True,
    )
    for gradient in closure_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    gold_gradient = torch.autograd.grad(
        closure,
        sparse.gold_states,
        allow_unused=True,
    )[0]
    assert gold_gradient is None


def test_token_conditioned_sequential_transition_has_no_current_label_leak():
    model = make_model()
    model.token_conditioned_transition = TokenConditionedTransition(
        32,
        bottleneck=16,
    )
    window = torch.randint(0, 43, (2, 12))
    loss, token_nll, closure, output = (
        sparse_token_conditioned_sequential_loss(
            model,
            window,
            prefix_length=9,
            horizons=3,
            anchor_stride=3,
        )
    )

    assert output.logits.shape == (2, 3, 3, 43)
    assert output.token_ce.shape == (2, 3, 3)
    assert output.proposal_states.shape == (2, 3, 3, 32)
    assert output.conditioned_states.shape == (2, 3, 3, 32)
    assert output.student_readout_tapes.shape == (2, 3, 3, 3, 32)
    assert output.behavioral_closure_by_horizon.shape == (2,)
    assert not output.teacher_logits.requires_grad
    assert not output.canonical_readout_tapes.requires_grad
    torch.testing.assert_close(loss, token_nll + closure)
    assert torch.isfinite(closure)
    assert closure >= -1e-6

    # Tape 1 predicts B from z_B. Tape 2 exposes conditioned s_B only as
    # history and predicts C from the unconditioned z_C slot.
    torch.testing.assert_close(
        output.student_readout_tapes[:, 0, :, 0],
        output.proposal_states[:, :, 0],
    )
    torch.testing.assert_close(
        output.student_readout_tapes[:, 1, :, 0],
        output.conditioned_states[:, :, 0],
    )
    torch.testing.assert_close(
        output.student_readout_tapes[:, 1, :, 1],
        output.proposal_states[:, :, 1],
    )

    positions = torch.arange(window.shape[1])
    full_encoded, _ = model.encode(window, positions)
    roots = full_encoded[:, :9].index_select(1, output.anchor_indices)
    current = roots
    for horizon in range(3):
        proposal = model.operator(current)
        torch.testing.assert_close(
            proposal,
            output.proposal_states[:, :, horizon],
        )
        embedding = torch.nn.functional.embedding(
            output.targets[:, :, horizon],
            model.embedding_weight,
        )
        current, innovation, log_scale = (
            model.token_conditioned_transition(proposal, embedding)
        )
        torch.testing.assert_close(
            current,
            output.conditioned_states[:, :, horizon],
        )
        torch.testing.assert_close(
            innovation,
            output.innovations[:, :, horizon],
        )
        torch.testing.assert_close(
            log_scale,
            output.log_scale[:, :, horizon],
        )

    transition_parameters = tuple(
        model.token_conditioned_transition.parameters()
    )
    h1_gradients = torch.autograd.grad(
        output.token_ce[:, :, 0].mean(),
        transition_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    assert all(
        gradient is None or torch.count_nonzero(gradient) == 0
        for gradient in h1_gradients
    )

    later_gradients = torch.autograd.grad(
        output.token_ce[:, :, 1:].mean(),
        (
            model.token_conditioned_transition.proposal.weight,
            model.token_conditioned_transition.token.weight,
            model.token_conditioned_transition.gate.weight,
            model.token_conditioned_transition.output.weight,
            model.token_conditioned_transition.scale.weight,
        ),
        retain_graph=True,
    )
    for gradient in later_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0

    closure_gradient = torch.autograd.grad(
        closure,
        model.token_conditioned_transition.output.weight,
        retain_graph=True,
    )[0]
    assert torch.isfinite(closure_gradient).all()
    assert closure_gradient.norm() > 0
    gold_gradient = torch.autograd.grad(
        closure,
        output.gold_states,
        allow_unused=True,
    )[0]
    assert gold_gradient is None

    proposal = output.proposal_states[:, :, 0].detach()
    first_embedding = model.embedding_weight[
        output.targets[:, :, 0]
    ].detach()
    alternative_embedding = model.embedding_weight[
        (output.targets[:, :, 0] + 1) % model.vocabulary
    ].detach()
    first_state = model.token_conditioned_transition(
        proposal,
        first_embedding,
    )[0]
    alternative_state = model.token_conditioned_transition(
        proposal,
        alternative_embedding,
    )[0]
    assert (first_state - alternative_state).abs().max() > 0


def test_pairwise_orthogonal_affine_scan_matches_literal_recurrence():
    torch.manual_seed(113)
    root = torch.randn(2, 3, 16)
    angles = 0.2 * torch.randn(2, 3, 9, 8)
    increments = 0.1 * torch.randn(2, 3, 9, 16)
    scanned, cumulative_angles, cumulative_increments = (
        apply_pairwise_affine_scan(root, angles, increments)
    )
    sequential = sequential_pairwise_affine(
        root,
        angles,
        increments,
    )
    torch.testing.assert_close(scanned, sequential, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        scanned,
        rotate_pairwise(
            root.unsqueeze(2).expand_as(increments),
            cumulative_angles,
        )
        + cumulative_increments,
    )
    rotated = rotate_pairwise(root, angles[:, :, 0])
    torch.testing.assert_close(
        rotated.square().sum(dim=-1),
        root.square().sum(dim=-1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_prefix_orthogonal_scan_is_prefix_only_and_closes_behavior():
    model = make_model()
    model.operator = nn.Identity()
    model.prefix_orthogonal_scan = (
        PrefixConditionedOrthogonalAffineScan(
            model.width,
            bottleneck=16,
            position_width=8,
        )
    )
    window = torch.randint(0, 43, (2, 12))
    loss, token_nll, closure, output = (
        sparse_prefix_orthogonal_affine_scan_loss(
            model,
            window,
            prefix_length=9,
            horizons=3,
            anchor_stride=3,
        )
    )
    assert output.logits.shape == (2, 3, 3, 43)
    assert output.predicted_states.shape == (2, 3, 3, 32)
    assert output.local_angles.shape == (2, 3, 3, 16)
    assert output.innovations.shape == (2, 3, 3, 32)
    assert output.behavioral_closure_by_horizon.shape == (2,)
    assert not output.teacher_logits.requires_grad
    assert not output.reanchored_states.requires_grad
    torch.testing.assert_close(loss, token_nll + closure)
    assert torch.isfinite(closure)
    assert closure >= -1e-6

    positions = torch.arange(window.shape[1])
    full_encoded, _ = model.encode(window, positions)
    roots = full_encoded[:, :9].index_select(1, output.anchor_indices)
    coordinate_root = torch.nn.functional.linear(
        roots,
        output.basis.T,
    )
    sequential = sequential_pairwise_affine(
        coordinate_root,
        output.local_angles,
        output.innovations,
    )
    torch.testing.assert_close(
        output.coordinate_states,
        sequential,
        atol=2e-5,
        rtol=2e-5,
    )

    changed_future = window.clone()
    changed_future[:, 9:] = (changed_future[:, 9:] + 1) % 43
    changed = sparse_prefix_orthogonal_affine_scan_loss(
        model,
        changed_future,
        prefix_length=9,
        horizons=3,
        anchor_stride=3,
    )[3]
    torch.testing.assert_close(
        output.predicted_states,
        changed.predicted_states,
    )
    torch.testing.assert_close(output.local_angles, changed.local_angles)
    torch.testing.assert_close(output.innovations, changed.innovations)
    torch.testing.assert_close(output.logits, changed.logits)

    scan_module = model.prefix_orthogonal_scan
    ce_parameters = (
        scan_module.basis_generator,
        scan_module.context.weight,
        scan_module.position.weight,
        scan_module.angle.weight,
        scan_module.innovation.weight,
        scan_module.scale.weight,
        model.encoder.blocks[0].attn.qkv.weight,
        model.embedding_weight,
    )
    ce_gradients = torch.autograd.grad(
        token_nll,
        ce_parameters,
        retain_graph=True,
    )
    for gradient in ce_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0

    closure_gradients = torch.autograd.grad(
        closure,
        (
            scan_module.basis_generator,
            scan_module.angle.weight,
            scan_module.innovation.weight,
            scan_module.scale.weight,
        ),
        retain_graph=True,
    )
    for gradient in closure_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    gold_gradient = torch.autograd.grad(
        closure,
        output.gold_states,
        allow_unused=True,
    )[0]
    assert gold_gradient is None


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
