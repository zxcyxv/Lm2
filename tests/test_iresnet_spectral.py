import torch
from torch import nn
from torch.nn.utils import parametrize

from rotlm.models.iresnet_spectral import (
    apply_iresnet_soft_spectral_norm,
    exact_effective_spectral_norms,
    update_iresnet_power_vectors,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM


def test_soft_normalization_matches_iresnet_equation_and_keeps_raw_weight():
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.diag(torch.tensor([2.0, 0.25])))
    raw_before = linear.weight.detach().clone()
    apply_iresnet_soft_spectral_norm(
        linear,
        coeff=0.9,
        n_power_iterations=20,
        generator=torch.Generator().manual_seed(7),
    )
    stats = update_iresnet_power_vectors(linear)

    torch.testing.assert_close(
        linear.parametrizations.weight.original,
        raw_before,
    )
    assert stats.active_layer_count == 1
    torch.testing.assert_close(
        torch.linalg.matrix_norm(linear.weight, ord=2),
        torch.tensor(0.9),
        atol=1e-6,
        rtol=1e-6,
    )


def test_soft_normalization_does_not_expand_a_weight_below_the_coefficient():
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.diag(torch.tensor([0.5, 0.25])))
    raw_before = linear.weight.detach().clone()
    apply_iresnet_soft_spectral_norm(
        linear,
        coeff=0.9,
        n_power_iterations=20,
        generator=torch.Generator().manual_seed(11),
    )
    stats = update_iresnet_power_vectors(linear)

    assert stats.active_layer_count == 0
    torch.testing.assert_close(linear.weight, raw_before)


def test_parameterization_is_differentiable_to_the_raw_weight():
    linear = nn.Linear(3, 2, bias=False)
    apply_iresnet_soft_spectral_norm(
        linear,
        coeff=0.9,
        n_power_iterations=5,
        generator=torch.Generator().manual_seed(13),
    )
    update_iresnet_power_vectors(linear)
    linear(torch.randn(4, 3)).square().mean().backward()

    gradient = linear.parametrizations.weight.original.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_shared_spectral_weights_preserve_additive_coupling_inverse():
    torch.manual_seed(17)
    model = K1DecoderAblationLM(
        41,
        width=32,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
    )
    generator = torch.Generator().manual_seed(19)
    block = model.encoder.blocks[0]
    for linear in (block.attn.qkv, block.attn.proj, block.ffn[0], block.ffn[2]):
        apply_iresnet_soft_spectral_norm(
            linear,
            coeff=0.9,
            n_power_iterations=5,
            generator=generator,
        )
    stats = update_iresnet_power_vectors(model)

    tokens = torch.randint(0, 41, (2, 12))
    positions = torch.arange(tokens.shape[1])
    embedded = model.encoder.embed(tokens)
    with parametrize.cached():
        encoded, positions = model.encode(tokens, positions)
        reconstructed = model.encoder.decode_hidden(encoded, positions)

    assert stats.layer_count == 4
    torch.testing.assert_close(reconstructed, embedded, atol=2e-5, rtol=2e-5)
    exact = exact_effective_spectral_norms(model)
    assert set(exact) == {
        "encoder.blocks.0.attn.qkv",
        "encoder.blocks.0.attn.proj",
        "encoder.blocks.0.ffn.0",
        "encoder.blocks.0.ffn.2",
    }
