import torch

from rotlm.latent_diagnostics import (
    additive_residual_diagnostics,
    complex_recurrent_scale_tape,
    decision_margin_diagnostics,
    jacobian_spectral_norm_power_iteration,
    orthogonal_drift_diagnostics,
    recurrent_step_scale_diagnostics,
)
from rotlm.models.complex_self_prediction import (
    ComplexSelfPredictedKVTransition,
)


def test_orthogonal_drift_removes_one_global_rotation():
    torch.manual_seed(17)
    source = torch.randn(32, 8)
    orthogonal, _ = torch.linalg.qr(torch.randn(8, 8))
    target = source @ orthogonal

    diagnostics = orthogonal_drift_diagnostics(source, target)
    assert diagnostics.raw_relative_mse.mean() > 0.1
    assert diagnostics.aligned_relative_mse.max() < 1e-10
    torch.testing.assert_close(
        diagnostics.centered_linear_cka,
        torch.ones_like(diagnostics.centered_linear_cka),
        atol=2e-6,
        rtol=2e-6,
    )
    assert diagnostics.alignment_orthogonality_error < 2e-6


def test_decision_margin_decomposition_matches_observed_flips():
    reference = torch.tensor(
        [
            [4.0, 2.0, 0.0],
            [4.0, 3.5, 0.0],
            [1.0, 4.0, 2.0],
        ]
    )
    candidate = torch.tensor(
        [
            [3.0, 2.5, 0.0],
            [3.0, 4.0, 0.0],
            [3.5, 3.0, 2.0],
        ]
    )
    diagnostics = decision_margin_diagnostics(reference, candidate)

    torch.testing.assert_close(
        diagnostics.reference_top1_margin,
        torch.tensor([2.0, 0.5, 2.0]),
    )
    assert diagnostics.matches.tolist() == [True, False, False]
    assert diagnostics.crossing_excess.tolist() == [-0.5, 1.0, 0.5]
    torch.testing.assert_close(
        diagnostics.crossing_excess.gt(0),
        ~diagnostics.matches,
    )


def test_additive_residual_shapley_values_sum_to_exact_error_reduction():
    prior = torch.tensor([[0.0, 0.0]])
    components = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[2.0, 1.0]])
    diagnostics = additive_residual_diagnostics(prior, components, target)

    torch.testing.assert_close(diagnostics.prior_relative_mse, torch.tensor([1.0]))
    torch.testing.assert_close(diagnostics.full_relative_mse, torch.tensor([0.2]))
    torch.testing.assert_close(diagnostics.error_reduction, torch.tensor([0.8]))
    torch.testing.assert_close(
        diagnostics.head_shapley_error_reduction,
        torch.tensor([[0.6, 0.2]]),
    )
    torch.testing.assert_close(
        diagnostics.head_shapley_error_reduction.sum(dim=-1),
        diagnostics.error_reduction,
    )
    torch.testing.assert_close(
        diagnostics.optimal_innovation_scale,
        torch.tensor([1.5]),
    )
    torch.testing.assert_close(
        diagnostics.innovation_target_energy_ratio,
        torch.tensor([0.4]),
    )
    torch.testing.assert_close(
        diagnostics.innovation_residual_dot_ratio,
        torch.tensor([0.6]),
    )
    torch.testing.assert_close(
        diagnostics.oracle_relative_mse,
        torch.tensor([0.1]),
    )


def test_recurrent_step_scale_diagnostics_uses_real_complex_inner_product():
    rotated = torch.ones(2, 1, 2, 3, dtype=torch.complex64)
    innovation = 2j * rotated
    raw_updated = rotated + innovation
    stored_memory = raw_updated / raw_updated.abs().square().mean(
        dim=(-3, -2, -1), keepdim=True
    ).sqrt()
    raw_preliminary = torch.full((2, 4), 0.5)
    preliminary = raw_preliminary * 2.0
    raw_full = torch.full((2, 4), 2.0)
    stored_hidden = raw_full * 0.5

    diagnostics = recurrent_step_scale_diagnostics(
        rotated_memory=rotated,
        innovation_memory=innovation,
        raw_updated_memory=raw_updated,
        stored_memory=stored_memory,
        raw_preliminary_hidden=raw_preliminary,
        preliminary_hidden=preliminary,
        raw_full_hidden=raw_full,
        stored_hidden=stored_hidden,
    )
    torch.testing.assert_close(
        diagnostics.innovation_to_memory_ratio,
        torch.full((2,), 2.0),
    )
    torch.testing.assert_close(
        diagnostics.memory_innovation_cosine,
        torch.zeros(2),
    )
    torch.testing.assert_close(
        diagnostics.raw_updated_memory_rms,
        torch.full((2,), 5.0**0.5),
    )
    torch.testing.assert_close(
        diagnostics.stored_memory_rms,
        torch.ones(2),
    )


def test_complex_recurrent_scale_tape_separates_raw_and_stored_carriers():
    torch.manual_seed(29)
    transition = ComplexSelfPredictedKVTransition(
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
    roots = 3.0 * torch.randn(2, 8)
    tape = complex_recurrent_scale_tape(transition, roots, horizons=3)

    assert tape.rotated_hidden_rms.shape == (2, 3)
    assert tape.raw_full_hidden_rms.shape == (2, 3)
    torch.testing.assert_close(
        tape.raw_root_hidden_rms,
        roots.square().mean(dim=-1).sqrt(),
    )
    torch.testing.assert_close(
        tape.initialized_hidden_rms,
        torch.ones(2),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        tape.initialized_hidden_cosine_to_raw,
        torch.ones(2),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        tape.stored_hidden_rms,
        tape.raw_full_hidden_rms / tape.hidden_boundary_denominator,
    )
    torch.testing.assert_close(
        tape.stored_memory_rms,
        tape.raw_updated_memory_rms / tape.memory_boundary_denominator,
    )
    assert bool((tape.stored_hidden_rms <= 1.0).all())
    assert bool((tape.stored_memory_rms <= 1.0).all())
    torch.testing.assert_close(
        tape.hidden_boundary_denominator,
        (tape.raw_full_hidden_rms.square() + transition.post_norm_eps).sqrt(),
    )
    torch.testing.assert_close(
        tape.memory_boundary_denominator,
        (
            tape.raw_updated_memory_rms.square()
            + transition.post_norm_eps
        ).sqrt(),
    )


def test_jacobian_power_iteration_recovers_largest_singular_value():
    matrix = torch.diag(torch.tensor([0.5, 2.0, 4.0]))
    point = torch.randn(3)
    estimate = jacobian_spectral_norm_power_iteration(
        lambda value: matrix @ value,
        point,
        iterations=12,
        seed=31,
    )
    assert abs(estimate.singular_value - 4.0) < 1e-4
    assert len(estimate.history) == 13
