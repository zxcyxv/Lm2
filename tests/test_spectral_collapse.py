import torch
from torch import nn
import torch.nn.functional as F

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.prefix_orthogonal_scan import rotate_pairwise
from rotlm.models.spectral_collapse import (
    InputConditionedSpectralCollapse,
)
from rotlm.training.ha_skew_window import (
    sparse_spectral_collapse_loss,
)


def _calibrated_module(*, alpha: float) -> InputConditionedSpectralCollapse:
    torch.manual_seed(7)
    module = InputConditionedSpectralCollapse(
        16,
        bottleneck=8,
        alpha=alpha,
    )
    module.calibrate_shell(torch.randn(3, 5, 16))
    return module


def test_spectral_collapse_closed_form_matches_literal_and_shell() -> None:
    module = _calibrated_module(alpha=1.0)
    roots = torch.randn(2, 4, 16)
    output = module(roots, 5)
    literal = module.literal_rollout(roots, 5)

    torch.testing.assert_close(output.states, literal, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        output.radii,
        output.shell.view(1, 1, 1, -1).expand_as(output.radii),
        atol=1e-6,
        rtol=1e-6,
    )
    assert bool((output.shell > 0).all())
    assert float(output.rho.detach().min()) > 0
    assert float(output.rho.detach().max()) < 1
    assert float(output.rho.detach().var(dim=(0, 1)).sum()) > 0
    assert float(output.phi.detach().var(dim=(0, 1)).sum()) > 0

    identity = torch.eye(16)
    torch.testing.assert_close(
        output.basis.T @ output.basis,
        identity,
        atol=1e-6,
        rtol=1e-6,
    )


def test_alpha_zero_is_raw_block_frozen_conditional_operator_power() -> None:
    module = _calibrated_module(alpha=0.0)
    roots = torch.randn(2, 3, 16)
    output = module(roots, 4)

    current = output.root_coordinates
    expected = []
    for _ in range(4):
        current = rotate_pairwise(current, output.phi)
        pairs = current.reshape(2, 3, 8, 2)
        current = (output.rho.unsqueeze(-1) * pairs).flatten(-2)
        expected.append(F.linear(current, output.basis.T))
    expected = torch.stack(expected, dim=2)

    torch.testing.assert_close(
        output.states,
        expected,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        output.states,
        output.unprojected_states,
        atol=2e-5,
        rtol=2e-5,
    )


def test_project_to_shell_retracts_arbitrary_states() -> None:
    module = _calibrated_module(alpha=0.0)
    states = torch.randn(2, 3, 4, 16)
    projected = module.project_to_shell(states)
    coordinates = F.linear(projected, module.basis)
    radii = coordinates.reshape(2, 3, 4, 8, 2).square().sum(
        dim=-1
    ).sqrt()
    torch.testing.assert_close(
        radii,
        module.shell.view(1, 1, 1, -1).expand_as(radii),
        atol=2e-5,
        rtol=2e-5,
    )


def test_fixed_prefix_operator_is_normal() -> None:
    module = _calibrated_module(alpha=1.0)
    root = torch.randn(1, 1, 16)
    output = module(root, 1)
    rho = output.rho[0, 0]
    phi = output.phi[0, 0]
    cosine, sine = torch.cos(phi), torch.sin(phi)
    blocks = torch.stack(
        (
            torch.stack((rho * cosine, -rho * sine), dim=-1),
            torch.stack((rho * sine, rho * cosine), dim=-1),
        ),
        dim=-2,
    )
    diagonal = torch.block_diag(*blocks.unbind(dim=0))
    operator = output.basis.T @ diagonal @ output.basis
    torch.testing.assert_close(
        operator @ operator.T,
        operator.T @ operator,
        atol=2e-5,
        rtol=2e-5,
    )


def test_spectral_collapse_loss_paths_and_future_nonleakage() -> None:
    torch.manual_seed(11)
    model = K1DecoderAblationLM(
        31,
        width=16,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
    )
    model.operator = nn.Identity()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        16,
        bottleneck=8,
        alpha=1.0,
    )
    window = torch.randint(0, 31, (2, 10))
    with torch.no_grad():
        encoded, _ = model.encode(window, torch.arange(10))
        model.spectral_collapse.calibrate_shell(encoded[:, :7])

    loss, token_nll, one_step_mse, output = (
        sparse_spectral_collapse_loss(
            model,
            window,
            prefix_length=7,
            horizons=3,
            anchor_stride=1,
        )
    )
    torch.testing.assert_close(loss, token_nll + one_step_mse)
    assert output.logits.shape == (2, 7, 3, 31)

    ce_gradients = torch.autograd.grad(
        token_nll,
        (
            model.spectral_collapse.basis_generator,
            model.spectral_collapse.phase.weight,
            model.spectral_collapse.shell_raw,
            model.encoder.blocks[0].attn.qkv.weight,
            model.embedding_weight,
        ),
        retain_graph=True,
    )
    for gradient in ce_gradients:
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.norm()) > 0

    mse_gradients = torch.autograd.grad(
        one_step_mse,
        (
            model.spectral_collapse.phase.weight,
            model.spectral_collapse.decay.weight,
        ),
        retain_graph=True,
    )
    for gradient in mse_gradients:
        assert float(gradient.norm()) > 0
    gold_gradient = torch.autograd.grad(
        one_step_mse,
        output.gold_states,
        allow_unused=True,
    )[0]
    assert gold_gradient is None or int(torch.count_nonzero(gold_gradient)) == 0

    changed = window.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 31
    with torch.no_grad():
        changed_output = sparse_spectral_collapse_loss(
            model,
            changed,
            prefix_length=7,
            horizons=3,
            anchor_stride=1,
        )[3]
    torch.testing.assert_close(
        output.predicted_states,
        changed_output.predicted_states,
    )
    torch.testing.assert_close(output.logits, changed_output.logits)


def test_h1_only_collapse_ce_has_no_h2_h3_logit_gradient() -> None:
    torch.manual_seed(17)
    model = K1DecoderAblationLM(
        31,
        width=16,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
    )
    model.operator = nn.Identity()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        16,
        bottleneck=8,
        alpha=1.0,
    )
    window = torch.randint(0, 31, (2, 10))
    with torch.no_grad():
        encoded, _ = model.encode(window, torch.arange(10))
        model.spectral_collapse.calibrate_shell(encoded[:, :7])

    _, token_nll, _, output = sparse_spectral_collapse_loss(
        model,
        window,
        prefix_length=7,
        horizons=3,
        anchor_stride=1,
        ce_horizons=(1,),
    )
    torch.testing.assert_close(token_nll, output.token_ce[:, :, 0].mean())
    logit_gradient = torch.autograd.grad(
        token_nll,
        output.logits,
        retain_graph=True,
    )[0]
    assert logit_gradient[:, :, 0].norm() > 0
    assert int(torch.count_nonzero(logit_gradient[:, :, 1:])) == 0

    changed = window.clone()
    changed[:, 8:] = (changed[:, 8:] + 3) % 31
    changed_nll = sparse_spectral_collapse_loss(
        model,
        changed,
        prefix_length=7,
        horizons=3,
        anchor_stride=1,
        ce_horizons=(1,),
    )[1]
    torch.testing.assert_close(token_nll, changed_nll, atol=0, rtol=0)
