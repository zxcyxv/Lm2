import torch
import torch.nn.functional as F

from rotlm.models.complex_self_prediction import (
    ComplexSelfPredictedKVTransition,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.latent_flow_matching import (
    ConditionalRayFlowTimeConditioner,
    conditional_tangent_source,
    shortest_spherical_bridge,
)
from rotlm.training.conditional_flow_matching import (
    sparse_scan_conditional_ray_flow_loss,
)


def make_model() -> K1DecoderAblationLM:
    torch.manual_seed(283)
    model = K1DecoderAblationLM(
        17,
        width=32,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="simplex-raw-tied",
        simplex_logit_scale=8.0,
    )
    model.operator = torch.nn.Identity()
    model.complex_self_prediction = ComplexSelfPredictedKVTransition(
        model.width,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    )
    return model


def test_spherical_bridge_has_exact_endpoints_and_analytic_velocity():
    torch.manual_seed(281)
    source = F.normalize(torch.randn(2, 3, 4, 32), dim=-1)
    target = F.normalize(torch.randn(2, 3, 4, 32), dim=-1)
    at_zero = shortest_spherical_bridge(source, target, 0.0)
    at_one = shortest_spherical_bridge(source, target, 1.0)
    torch.testing.assert_close(at_zero.state, source, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(at_one.state, target, atol=2e-6, rtol=2e-6)

    time = 0.37
    epsilon = 1e-3
    center = shortest_spherical_bridge(source, target, time)
    before = shortest_spherical_bridge(source, target, time - epsilon)
    after = shortest_spherical_bridge(source, target, time + epsilon)
    finite_difference = (after.state - before.state) / (2 * epsilon)
    torch.testing.assert_close(
        center.target_velocity,
        finite_difference,
        atol=4e-4,
        rtol=8e-4,
    )
    tangent_dot = (center.state * center.target_velocity).sum(dim=-1)
    torch.testing.assert_close(tangent_dot, torch.zeros_like(tangent_dot), atol=2e-6, rtol=0)


def test_equal_scale_tangent_source_retains_independent_horizon_draws():
    torch.manual_seed(282)
    roots = torch.randn(64, 128)
    source, noise = conditional_tangent_source(roots, 8, 0.8)
    assert source.shape == (64, 8, 128)
    assert noise.shape == source.shape
    torch.testing.assert_close(
        source.norm(dim=-1),
        torch.ones(64, 8),
        atol=2e-6,
        rtol=2e-6,
    )
    root_unit = F.normalize(roots, dim=-1).unsqueeze(1)
    dot = (source * root_unit).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(dot)
    assert abs(float(angles.square().mean().sqrt()) - 0.8) < 0.08
    assert not torch.equal(noise[:, 0], noise[:, 1])


def test_forcing_velocity_is_the_shared_scan_measurement_primitive():
    torch.manual_seed(284)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    )
    roots = torch.randn(2, 3, 32)
    rollout = transition.rollout_time_varying_scan(roots, 4)
    forcing = transition.forcing_velocity_tape(rollout.schedule_states)
    torch.testing.assert_close(
        forcing.velocity,
        rollout.innovation_deltas,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        forcing.memory_states,
        rollout.memory_states,
        atol=2e-6,
        rtol=2e-6,
    )


def test_sparse_conditional_flow_is_finite_and_detaches_endpoints():
    model = make_model()
    conditioner = ConditionalRayFlowTimeConditioner(model.width)
    window = torch.randint(0, 17, (2, 10))
    total, token_nll, flow_loss, output = (
        sparse_scan_conditional_ray_flow_loss(
            model,
            conditioner,
            window,
            prefix_length=6,
            horizons=4,
            ce_anchor_stride=2,
            flow_anchor_stride=4,
            noise_scale=0.8,
            flow_weight=0.1,
            flow_time=torch.full((2, 2, 1, 1), 0.4),
        )
    )
    torch.testing.assert_close(total, token_nll + 0.1 * flow_loss)
    assert torch.isfinite(total)
    assert output.flow.loss_by_horizon.shape == (4,)
    assert output.flow.relative_loss_by_horizon.shape == (4,)
    assert output.flow.cosine_by_horizon.shape == (4,)
    target_gradient = torch.autograd.grad(
        flow_loss,
        output.gold_states,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert target_gradient is None

    conditioner_parameters = tuple(conditioner.parameters())
    parameters = tuple(
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ) + conditioner_parameters
    gradients = torch.autograd.grad(total, parameters, allow_unused=True)
    assert all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )
    conditioner_gradients = gradients[-len(conditioner_parameters):]
    assert all(
        gradient is not None and bool(gradient.norm() > 0)
        for gradient in conditioner_gradients
    )
