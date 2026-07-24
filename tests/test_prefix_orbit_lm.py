import torch
import torch.nn.functional as F

from rotlm.geometry.horizon import cyclic_shift_channels, real_dft_basis
from rotlm.models.prefix_orbit_lm import (
    PrefixOrbitLM,
    _LinearWithDetachedInputView,
)


def test_dual_gradient_linear_matches_explicit_detached_input():
    torch.manual_seed(9)
    inputs = torch.randn(2, 3, 7, requires_grad=True)
    weight = torch.randn(11, 7, requires_grad=True)
    bias = torch.randn(11, requires_grad=True)
    live_scale = torch.randn(2, 3, 11)
    detached_scale = torch.randn(2, 3, 11)

    live, detached_input = _LinearWithDetachedInputView.apply(
        inputs, weight, bias
    )
    assert live.data_ptr() == detached_input.data_ptr()
    loss = (
        (live * live_scale).sum()
        + (detached_input * detached_scale).sum()
    )
    gradients = torch.autograd.grad(loss, (inputs, weight, bias))

    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    reference_loss = (
        F.linear(
            reference_inputs,
            reference_weight,
            reference_bias,
        )
        * live_scale
    ).sum() + (
        F.linear(
            reference_inputs.detach(),
            reference_weight,
            reference_bias,
        )
        * detached_scale
    ).sum()
    reference_gradients = torch.autograd.grad(
        reference_loss,
        (reference_inputs, reference_weight, reference_bias),
    )
    for gradient, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference)


def test_prefix_orbit_is_one_causal_inverse_sequence():
    torch.manual_seed(10)
    model = PrefixOrbitLM(19, d=32, n_blocks=1).eval()
    tokens = torch.randint(0, 19, (2, 12))
    positions = torch.arange(12)
    out = model.decode_orbit(tokens, positions, anchor=11, horizons=4)
    back, _ = model.encode_hidden(out.hidden, out.positions)
    assert out.latent.shape == (2, 16, 32)
    assert torch.allclose(back, out.latent, atol=3e-5, rtol=3e-5)


def test_future_slot_cannot_change_earlier_inverse_outputs():
    torch.manual_seed(11)
    model = PrefixOrbitLM(19, d=32, n_blocks=1).eval()
    tokens = torch.randint(0, 19, (2, 10)); positions = torch.arange(10)
    latent, positions = model.encode(tokens, positions)
    extended, extended_pos = model.append_orbit(latent, positions, anchor=9, horizons=4)
    first = model.decode_hidden(extended, extended_pos)
    changed = extended.clone(); changed[:, -1] += torch.randn_like(changed[:, -1])
    second = model.decode_hidden(changed, extended_pos)
    assert torch.equal(first[:, :-1], second[:, :-1])


def test_earlier_orbit_slot_affects_later_output():
    torch.manual_seed(12)
    model = PrefixOrbitLM(19, d=32, n_blocks=1).eval()
    tokens = torch.randint(0, 19, (2, 10)); positions = torch.arange(10)
    latent, positions = model.encode(tokens, positions)
    extended, extended_pos = model.append_orbit(latent, positions, anchor=9, horizons=4)
    first = model.decode_hidden(extended, extended_pos)
    changed = extended.clone(); changed[:, -3] += torch.randn_like(changed[:, -3])
    second = model.decode_hidden(changed, extended_pos)
    assert (first[:, -1] - second[:, -1]).abs().max() > 0


def test_dense_shared_prefix_matches_individual_anchor_decodes_and_roundtrip():
    torch.manual_seed(13)
    model = PrefixOrbitLM(19, d=32, n_blocks=2).eval()
    tokens = torch.randint(0, 19, (2, 7)); positions = torch.arange(7)
    latent, positions = model.encode(tokens, positions)
    prefix_h, branch_h, branch_pos = model.dense_decode_hidden(latent, positions, horizons=3)
    for anchor in range(7):
        individual = model.decode_orbit_from_latent(latent, positions, anchor, horizons=3)
        assert torch.allclose(branch_h[:, anchor], individual.hidden[:, -3:], atol=4e-5, rtol=4e-5)
    prefix_back, branch_back = model.dense_encode_hidden(prefix_h, branch_h, positions, branch_pos)
    branch_latent, _ = model.dense_orbit_latent(latent, positions, horizons=3)
    assert torch.allclose(prefix_back, latent, atol=5e-5, rtol=5e-5)
    assert torch.allclose(branch_back, branch_latent, atol=5e-5, rtol=5e-5)


def test_k8_cyclic_operator_is_orthogonal_periodic_and_compositional():
    torch.manual_seed(14)
    states = torch.randn(3, 5, 32)
    once_then_twice = cyclic_shift_channels(cyclic_shift_channels(states, 1, 8), 2, 8)
    assert torch.equal(once_then_twice, cyclic_shift_channels(states, 3, 8))
    assert torch.equal(cyclic_shift_channels(states, 8, 8), states)
    assert torch.equal(cyclic_shift_channels(states, -1, 8), cyclic_shift_channels(states, 7, 8))
    assert torch.allclose(
        cyclic_shift_channels(states, 3, 8).square().sum(-1),
        states.square().sum(-1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_real_k8_dft_basis_is_orthonormal():
    basis = real_dft_basis(8)
    assert basis.shape == (8, 8)
    assert torch.allclose(basis.T @ basis, torch.eye(8, dtype=basis.dtype), atol=1e-12, rtol=0)


def test_prefix_orbit_accepts_k8_cyclic_operator():
    torch.manual_seed(15)
    model = PrefixOrbitLM(19, d=32, n_blocks=1, latent_operator="cyclic", orbit_period=8)
    tokens = torch.randint(0, 19, (2, 10)); positions = torch.arange(10)
    latent, positions = model.encode(tokens, positions)
    branch, _ = model.dense_orbit_latent(latent, positions, horizons=8)
    assert torch.equal(branch[:, :, -1], latent)
