import pytest
import torch

from rotlm.evaluation import complex_self_predicted_kv_ar_rollouts
from rotlm.models.complex_self_prediction import (
    ComplexMemoryState,
    ComplexSelfPredictedKVTransition,
    complex_memory_affine_scan,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.prefix_orthogonal_scan import rotate_pairwise
from rotlm.training.ema import make_ema_model
from rotlm.training.ha_skew_window import (
    forward_sparse_complex_self_prediction_self_redecode_kl,
    forward_sparse_complex_self_predicted_kv_window,
    sparse_complex_self_prediction_self_redecode_kl_loss,
    sparse_complex_self_predicted_kv_ce_only_fast_loss,
    sparse_complex_self_predicted_kv_loss,
    sparse_complex_stochastic_ray_flow_loss,
)


def make_model() -> K1DecoderAblationLM:
    torch.manual_seed(93)
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
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        initial_beta=0.5,
    )
    return model


def test_complex_memory_affine_scan_matches_literal_time_varying_recurrence():
    torch.manual_seed(91)
    multipliers = torch.polar(
        torch.rand(2, 3, 5, 7),
        torch.randn(2, 3, 5, 7),
    )
    increments = torch.complex(
        torch.randn(2, 3, 5, 7, 4),
        torch.randn(2, 3, 5, 7, 4),
    )
    _, scanned = complex_memory_affine_scan(multipliers, increments)

    state = torch.zeros_like(increments[..., 0, :, :])
    sequential = []
    for horizon in range(increments.shape[-3]):
        state = (
            multipliers[..., horizon, :, None] * state
            + increments[..., horizon, :, :]
        )
        sequential.append(state)
    sequential = torch.stack(sequential, dim=-3)
    torch.testing.assert_close(scanned, sequential, atol=3e-6, rtol=3e-6)


def test_time_varying_scan_is_exact_prefix_composition_and_shares_h1():
    torch.manual_seed(92)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    )
    root = torch.randn(2, 4, 32)
    scanned = transition.rollout_time_varying_scan(root, 5)

    conditioning = root
    hidden = root
    memory = torch.zeros(
        2, 4, 2, 3, 5, dtype=torch.complex64
    )
    memory_multiplier = torch.polar(
        torch.ones_like(transition.memory_phase),
        -transition.memory_phase,
    )[..., None]
    sequential_hidden = []
    sequential_read = []
    sequential_memory = []
    sequential_delta = []
    for _ in range(5):
        angles = transition.hidden_phase.expand(2, 4, -1)
        conditioning = rotate_pairwise(conditioning, angles)
        query, key, value = transition.project_qkv(conditioning)
        write = torch.einsum(
            "...hk,...hv->...hkv", key, value.conj()
        )
        memory = memory_multiplier * memory + write
        read = transition.read(query, memory)
        frobenius = memory.abs().square().sum(
            dim=(-3, -2, -1)
        ).sqrt()
        read = read / (
            frobenius[..., None, None] + transition.post_norm_eps
        )
        delta = transition.read_to_hidden(read)
        hidden = rotate_pairwise(hidden, angles) + delta
        sequential_hidden.append(hidden)
        sequential_read.append(
            transition._fixed_rms_normalize_hidden(hidden)
        )
        sequential_memory.append(memory)
        sequential_delta.append(delta)

    torch.testing.assert_close(
        scanned.full_states,
        torch.stack(sequential_hidden, dim=-2),
        atol=8e-6,
        rtol=8e-6,
    )
    torch.testing.assert_close(
        scanned.read_states,
        torch.stack(sequential_read, dim=-2),
        atol=8e-6,
        rtol=8e-6,
    )
    torch.testing.assert_close(
        scanned.memory_states,
        torch.stack(sequential_memory, dim=-4),
        atol=8e-6,
        rtol=8e-6,
    )
    torch.testing.assert_close(
        scanned.innovation_deltas,
        torch.stack(sequential_delta, dim=-2),
        atol=8e-6,
        rtol=8e-6,
    )

    feedback = transition.rollout(root, 1)
    torch.testing.assert_close(
        scanned.full_states[..., :1, :],
        feedback.full_states,
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        scanned.read_states[..., :1, :],
        feedback.read_states,
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        scanned.memory_states[..., :1, :, :, :],
        feedback.final_state.memory.unsqueeze(-4),
        atol=3e-6,
        rtol=3e-6,
    )

    loss = scanned.read_states.square().mul(
        torch.randn_like(scanned.read_states)
    ).mean()
    gradients = torch.autograd.grad(
        loss,
        (
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
            transition.hidden_phase,
            transition.memory_phase,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_iterative_feedback_scan_extends_exact_prefix_and_converges():
    torch.manual_seed(9201)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    )
    root = torch.randn(2, 3, 32, requires_grad=True)
    horizons = 5
    compiled = transition.rollout_time_varying_scan(root, horizons)
    feedback = transition.rollout(root, horizons)

    rescanned_zero = transition.rollout_iterative_feedback_scan(
        root,
        horizons,
        feedback_rescans=0,
    )
    for rescanned_value, compiled_value in (
        (rescanned_zero.full_states, compiled.full_states),
        (rescanned_zero.read_states, compiled.read_states),
        (rescanned_zero.forcing_states, compiled.schedule_states),
        (rescanned_zero.innovation_deltas, compiled.innovation_deltas),
        (rescanned_zero.memory_states, compiled.memory_states),
    ):
        torch.testing.assert_close(
            rescanned_value,
            compiled_value,
            atol=0,
            rtol=0,
        )

    rollouts = []
    for feedback_rescans in range(horizons):
        rescanned = transition.rollout_iterative_feedback_scan(
            root,
            horizons,
            feedback_rescans=feedback_rescans,
        )
        exact_horizons = feedback_rescans + 1
        torch.testing.assert_close(
            rescanned.full_states[..., :exact_horizons, :],
            feedback.full_states[..., :exact_horizons, :],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            rescanned.read_states[..., :exact_horizons, :],
            feedback.read_states[..., :exact_horizons, :],
            atol=2e-5,
            rtol=2e-5,
        )
        rollouts.append(rescanned)

    converged = rollouts[-1]
    torch.testing.assert_close(
        converged.full_states,
        feedback.full_states,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        converged.read_states,
        feedback.read_states,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        converged.final_state.memory,
        feedback.final_state.memory,
        atol=2e-5,
        rtol=2e-5,
    )

    loss = rollouts[1].read_states.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
            transition.hidden_phase,
            transition.memory_phase,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_stochastic_value_write_preserves_literal_and_rotating_scans():
    torch.manual_seed(9200)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    )
    with torch.no_grad():
        transition.hidden_phase.uniform_(-0.2, 0.2)
        transition.memory_phase.uniform_(-0.2, 0.2)
    root = torch.randn(2, 4, 32)
    noise = torch.complex(
        torch.randn(2, 4, 5, 2, 5),
        torch.randn(2, 4, 5, 2, 5),
    ) * (2.0 ** -0.5)

    deterministic = transition.rollout_time_varying_scan(root, 5)
    zero_scale = transition.rollout_time_varying_scan(
        root,
        5,
        value_noise=noise,
        value_noise_scale=0.0,
    )
    torch.testing.assert_close(zero_scale.full_states, deterministic.full_states)
    torch.testing.assert_close(
        zero_scale.memory_states,
        deterministic.memory_states,
    )

    noise_scale = 0.07
    stochastic = transition.rollout_time_varying_scan(
        root,
        5,
        value_noise=noise,
        value_noise_scale=noise_scale,
    )
    conditioning = root
    hidden = root
    memory = torch.zeros(2, 4, 2, 3, 5, dtype=torch.complex64)
    memory_multiplier = torch.polar(
        torch.ones_like(transition.memory_phase),
        -transition.memory_phase,
    )[..., None]
    sequential_hidden = []
    sequential_memory = []
    for horizon in range(5):
        angles = transition.hidden_phase.expand(2, 4, -1)
        conditioning = rotate_pairwise(conditioning, angles)
        query, key, value = transition.project_qkv(conditioning)
        value_rms = value.abs().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt()
        value = value + noise_scale * value_rms * noise[:, :, horizon]
        write = torch.einsum("...hk,...hv->...hkv", key, value.conj())
        memory = memory_multiplier * memory + write
        read = transition.read(query, memory)
        frobenius = memory.abs().square().sum(dim=(-3, -2, -1)).sqrt()
        read = read / (
            frobenius[..., None, None] + transition.post_norm_eps
        )
        hidden = (
            rotate_pairwise(hidden, angles)
            + transition.read_to_hidden(read)
        )
        sequential_hidden.append(hidden)
        sequential_memory.append(memory)

    torch.testing.assert_close(
        stochastic.full_states,
        torch.stack(sequential_hidden, dim=-2),
        atol=8e-6,
        rtol=8e-6,
    )
    torch.testing.assert_close(
        stochastic.memory_states,
        torch.stack(sequential_memory, dim=-4),
        atol=8e-6,
        rtol=8e-6,
    )
    transition.use_rotating_frame_time_varying_scan = True
    rotating = transition.rollout_time_varying_scan(
        root,
        5,
        value_noise=noise,
        value_noise_scale=noise_scale,
    )
    torch.testing.assert_close(
        rotating.full_states,
        stochastic.full_states,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        rotating.memory_states,
        stochastic.memory_states,
        atol=2e-5,
        rtol=2e-5,
    )


def test_rotating_frame_scan_matches_eager_states_and_gradients():
    torch.manual_seed(9201)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    )
    with torch.no_grad():
        transition.hidden_phase.uniform_(-0.2, 0.2)
        transition.memory_phase.uniform_(-0.2, 0.2)
    root = torch.randn(2, 4, 32, requires_grad=True)

    eager = transition.rollout_time_varying_scan(root, 8)
    transition.use_rotating_frame_time_varying_scan = True
    rotating = transition.rollout_time_varying_scan(root, 8)

    for eager_value, rotating_value in (
        (eager.full_states, rotating.full_states),
        (eager.read_states, rotating.read_states),
        (eager.schedule_states, rotating.schedule_states),
        (eager.innovation_deltas, rotating.innovation_deltas),
        (eager.memory_states, rotating.memory_states),
        (
            eager.cumulative_hidden_angles,
            rotating.cumulative_hidden_angles,
        ),
        (
            eager.cumulative_hidden_increments,
            rotating.cumulative_hidden_increments,
        ),
        (
            eager.cumulative_memory_multipliers,
            rotating.cumulative_memory_multipliers,
        ),
    ):
        torch.testing.assert_close(
            rotating_value,
            eager_value,
            atol=2e-5,
            rtol=2e-5,
        )

    full_probe = torch.randn_like(eager.full_states)
    memory_probe = torch.randn_like(eager.memory_states)

    def objective(rollout):
        return (
            (rollout.full_states * full_probe).mean()
            + (rollout.memory_states.conj() * memory_probe).real.mean()
        )

    parameters = (
        root,
        transition.query.weight,
        transition.key.weight,
        transition.value.weight,
        transition.output.weight,
        transition.hidden_phase,
        transition.memory_phase,
    )
    eager_gradients = torch.autograd.grad(objective(eager), parameters)
    rotating_gradients = torch.autograd.grad(objective(rotating), parameters)
    for eager_gradient, rotating_gradient in zip(
        eager_gradients,
        rotating_gradients,
        strict=True,
    ):
        assert torch.isfinite(rotating_gradient).all()
        torch.testing.assert_close(
            rotating_gradient,
            eager_gradient,
            atol=3e-5,
            rtol=3e-4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_triton_rotating_frame_matches_eager_scan_and_gradients():
    torch.manual_seed(9202)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    ).cuda()
    with torch.no_grad():
        transition.hidden_phase.uniform_(-0.2, 0.2)
        transition.memory_phase.uniform_(-0.2, 0.2)
    root = torch.randn(2, 4, 32, device="cuda", requires_grad=True)

    eager = transition.rollout_time_varying_scan(root, 8)
    eager_refined = transition.rollout_iterative_feedback_scan(
        root,
        8,
        feedback_rescans=2,
    )
    transition.use_fused_triton_rotating_frame_time_varying_scan = True
    fused = transition.rollout_time_varying_scan(root, 8)
    fused_refined = transition.rollout_iterative_feedback_scan(
        root,
        8,
        feedback_rescans=2,
    )
    for eager_value, fused_value in (
        (eager.full_states, fused.full_states),
        (eager.read_states, fused.read_states),
        (eager.innovation_deltas, fused.innovation_deltas),
        (eager.memory_states, fused.memory_states),
        (
            eager.cumulative_hidden_angles,
            fused.cumulative_hidden_angles,
        ),
        (
            eager.cumulative_hidden_increments,
            fused.cumulative_hidden_increments,
        ),
        (
            eager.cumulative_memory_multipliers,
            fused.cumulative_memory_multipliers,
        ),
    ):
        assert eager_value is not None
        assert fused_value is not None
        torch.testing.assert_close(
            fused_value,
            eager_value,
            atol=3e-5,
            rtol=3e-5,
        )
    for eager_value, fused_value in (
        (eager_refined.full_states, fused_refined.full_states),
        (eager_refined.read_states, fused_refined.read_states),
        (eager_refined.forcing_states, fused_refined.forcing_states),
        (eager_refined.innovation_deltas, fused_refined.innovation_deltas),
        (eager_refined.memory_states, fused_refined.memory_states),
    ):
        torch.testing.assert_close(
            fused_value,
            eager_value,
            atol=6e-5,
            rtol=6e-5,
        )

    full_probe = torch.randn_like(eager.full_states)
    memory_probe = torch.randn_like(eager.memory_states)

    def objective(rollout):
        return (
            (rollout.full_states * full_probe).mean()
            + (rollout.memory_states.conj() * memory_probe).real.mean()
        )

    parameters = (
        root,
        transition.query.weight,
        transition.key.weight,
        transition.value.weight,
        transition.output.weight,
        transition.hidden_phase,
        transition.memory_phase,
    )
    eager_gradients = torch.autograd.grad(objective(eager), parameters)
    fused_gradients = torch.autograd.grad(objective(fused), parameters)
    for eager_gradient, fused_gradient in zip(
        eager_gradients,
        fused_gradients,
        strict=True,
    ):
        assert torch.isfinite(fused_gradient).all()
        torch.testing.assert_close(
            fused_gradient,
            eager_gradient,
            atol=5e-5,
            rtol=6e-4,
        )


def test_complex_rollout_is_self_predicted_and_beta_only_tempers_readout():
    transition = make_model().complex_self_prediction
    root = torch.randn(2, 4, 32)
    rollout = transition.rollout(root, 3)
    assert rollout.full_states.shape == (2, 4, 3, 32)
    assert rollout.read_states.shape == (2, 4, 3, 32)
    assert rollout.memory_energy.shape == (2, 4, 3)
    assert rollout.final_state.memory.shape == (2, 4, 2, 3, 5)
    assert rollout.final_state.memory.is_complex()

    first_full_delta = (
        rollout.full_states[:, :, 0]
        - rollout.preliminary_states[:, :, 0]
    )
    first_read_delta = (
        rollout.read_states[:, :, 0]
        - rollout.preliminary_states[:, :, 0]
    )
    torch.testing.assert_close(
        first_read_delta,
        rollout.beta * first_full_delta,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        rollout.rotated_states[:, :, 0].norm(dim=-1),
        root.norm(dim=-1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_complex_window_has_no_future_input_and_attaches_both_latent_sides():
    model = make_model()
    window = torch.randint(0, 17, (2, 8))
    output = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=7,
        horizons=1,
        anchor_stride=1,
    )
    changed = window.clone()
    changed[:, 7] = (changed[:, 7] + 1) % 17
    changed_output = forward_sparse_complex_self_predicted_kv_window(
        model,
        changed,
        prefix_length=7,
        horizons=1,
        anchor_stride=1,
    )
    torch.testing.assert_close(output.full_states, changed_output.full_states)
    torch.testing.assert_close(output.read_states, changed_output.read_states)
    torch.testing.assert_close(output.logits, changed_output.logits)

    loss, token_nll, latent_mse, output = (
        sparse_complex_self_predicted_kv_loss(
            model,
            window,
            prefix_length=7,
            horizons=1,
            anchor_stride=1,
        )
    )
    torch.testing.assert_close(loss, token_nll + latent_mse)
    target_gradient = torch.autograd.grad(
        latent_mse,
        output.gold_states,
        retain_graph=True,
    )[0]
    assert target_gradient.norm() > 0
    gradients = torch.autograd.grad(
        loss,
        (
            model.complex_self_prediction.query.weight,
            model.complex_self_prediction.key.weight,
            model.complex_self_prediction.value.weight,
            model.complex_self_prediction.output.weight,
            model.complex_self_prediction.hidden_phase,
            model.complex_self_prediction.memory_phase,
            model.complex_self_prediction.beta_logit,
            model.encoder.blocks[0].attn.qkv.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_complex_parallel_and_ar_share_first_proposal_before_feedback():
    model = make_model().eval()
    window = torch.randint(0, 17, (2, 10))
    rollout = complex_self_predicted_kv_ar_rollouts(
        model,
        window,
        prefix_length=7,
        horizons=3,
        anchor_stride=2,
    )
    torch.testing.assert_close(
        rollout.parallel_full_states[:, :, 0],
        rollout.ar_proposal_states[:, :, 0],
    )
    torch.testing.assert_close(
        rollout.parallel_logits[:, :, 0],
        rollout.ar_logits[:, :, 0],
        atol=5e-5,
        rtol=5e-5,
    )
    torch.testing.assert_close(
        rollout.parallel_read_states[:, :, 0],
        rollout.ar_read_states[:, :, 0],
    )
    assert rollout.parallel_tokens.shape == rollout.ar_tokens.shape


def test_untempered_mode_decodes_prior_and_recurs_on_full_innovation():
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
    )
    assert transition.beta_logit is None
    root = torch.randn(2, 4, 32)
    rollout = transition.rollout(root, 3)
    assert rollout.beta is None
    torch.testing.assert_close(
        rollout.read_states,
        rollout.preliminary_states,
    )
    assert not torch.allclose(rollout.read_states, rollout.full_states)
    torch.testing.assert_close(
        rollout.full_states,
        rollout.preliminary_states + rollout.innovation_deltas,
    )


def test_qkv_pre_normalization_can_be_removed_without_changing_projections():
    torch.manual_seed(109)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        pre_normalize_qkv_inputs=False,
    )
    hidden = torch.randn(2, 4, 32) * 7.0

    projected = transition.query(hidden)
    real, imaginary = projected.chunk(2, dim=-1)
    expected = torch.complex(
        real.reshape(2, 4, 2, 3),
        imaginary.reshape(2, 4, 2, 3),
    )

    assert isinstance(transition.input_norm, torch.nn.Identity)
    assert not any(
        name.startswith("input_norm")
        for name, _ in transition.named_parameters()
    )
    torch.testing.assert_close(transition.queries(hidden), expected)


def test_post_update_full_read_reuses_new_query_on_complete_memory():
    torch.manual_seed(117)
    old_transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="split-query-residual",
    )
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
    )
    transition.load_state_dict(old_transition.state_dict())
    root = torch.randn(2, 4, 32)
    state = transition.initialize(root)
    step = transition(state)
    old_step = old_transition(old_transition.initialize(root))
    updated_query = transition.queries(step.preliminary_hidden)
    expected = transition.read_to_hidden(
        transition.read(updated_query, step.state.memory)
    )
    innovation_values = transition.read(
        updated_query,
        transition.write(step.preliminary_hidden),
    )
    head_contributions = transition.read_head_contributions(innovation_values)

    torch.testing.assert_close(
        step.raw_preliminary_hidden,
        step.preliminary_hidden,
    )
    torch.testing.assert_close(step.rotated_memory, state.memory * torch.polar(
        torch.ones_like(transition.memory_phase),
        -transition.memory_phase,
    )[..., None])
    torch.testing.assert_close(
        step.raw_updated_memory,
        step.rotated_memory + step.innovation_memory,
    )
    torch.testing.assert_close(step.raw_full_hidden, step.state.hidden)

    # This mode changes only the recurrent hidden.  At identical weights the
    # causal CE readout, innovation, and updated memory are exactly unchanged.
    torch.testing.assert_close(step.read_hidden, old_step.read_hidden)
    torch.testing.assert_close(step.preliminary_hidden, old_step.preliminary_hidden)
    torch.testing.assert_close(step.innovation_delta, old_step.innovation_delta)
    torch.testing.assert_close(step.state.memory, old_step.state.memory)
    torch.testing.assert_close(step.state.hidden, expected)
    torch.testing.assert_close(step.read_hidden, step.preliminary_hidden)
    torch.testing.assert_close(
        head_contributions.sum(dim=-2),
        step.innovation_delta,
    )
    assert not torch.allclose(step.state.hidden, old_step.state.hidden)


def test_rotated_hidden_innovation_residual_is_explicit_boundary_delta():
    torch.manual_seed(167)
    full_read = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
        post_normalize_recurrent_state=True,
    )
    residual = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="rotated-hidden-innovation-residual",
        post_normalize_recurrent_state=True,
    )
    residual.load_state_dict(full_read.state_dict())
    root = torch.randn(2, 4, 32, requires_grad=True)

    full_step = full_read(full_read.initialize(root))
    residual_step = residual(residual.initialize(root))
    expected_raw = (
        residual_step.rotated_hidden + residual_step.innovation_delta
    )
    expected_hidden = expected_raw * torch.rsqrt(
        expected_raw.float().square().mean(dim=-1, keepdim=True)
        + residual.post_norm_eps
    )

    # At identical weights, this intervention changes only the successor
    # hidden construction.  P, its write, and the complete memory carrier are
    # untouched.
    for residual_values, control_values in (
        (residual_step.read_hidden, full_step.read_hidden),
        (residual_step.preliminary_hidden, full_step.preliminary_hidden),
        (residual_step.innovation_delta, full_step.innovation_delta),
        (residual_step.innovation_memory, full_step.innovation_memory),
        (residual_step.raw_updated_memory, full_step.raw_updated_memory),
        (residual_step.state.memory, full_step.state.memory),
    ):
        torch.testing.assert_close(residual_values, control_values)
    torch.testing.assert_close(residual_step.raw_full_hidden, expected_raw)
    torch.testing.assert_close(residual_step.state.hidden, expected_hidden)
    assert not torch.allclose(
        residual_step.raw_full_hidden,
        full_step.raw_full_hidden,
    )

    rollout = residual.rollout(root, 4)
    loss = rollout.read_states[:, :, -1].square().mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            residual.query.weight,
            residual.key.weight,
            residual.value.weight,
            residual.output.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_unitary_block_residual_is_single_write_read_closure():
    torch.manual_seed(169)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-block-residual",
        normalize_accumulated_reads=True,
        residual_step_scale=0.1,
    )
    root = torch.randn(2, 4, 32, requires_grad=True)
    initial = transition.initialize(root)
    assert initial.write_count == 0
    torch.testing.assert_close(initial.memory, torch.zeros_like(initial.memory))

    step = transition(initial)
    expected_write = transition.write(step.rotated_hidden)
    expected_read = transition.read(
        transition.queries(step.rotated_hidden),
        expected_write,
    )
    expected_delta = transition.read_to_hidden(expected_read)
    expected_hidden = step.rotated_hidden + 0.1 * expected_delta
    torch.testing.assert_close(step.innovation_memory, expected_write)
    torch.testing.assert_close(step.innovation_delta, expected_delta)
    torch.testing.assert_close(step.state.hidden, expected_hidden)
    torch.testing.assert_close(step.read_hidden, expected_hidden)
    assert step.state.write_count == 1

    second = transition(step.state)
    expected_second_read = transition.read(
        transition.queries(second.rotated_hidden),
        second.state.memory,
    ) / (2.0**0.5)
    expected_second_delta = transition.read_to_hidden(expected_second_read)
    torch.testing.assert_close(second.innovation_delta, expected_second_delta)
    loss = second.read_hidden.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_detached_input_raw_read_postnorm_residual_has_exact_boundary():
    torch.manual_seed(171)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode=(
            "unitary-detached-input-raw-read-postnorm-residual"
        ),
        pre_normalize_qkv_inputs=False,
        normalize_initial_recurrent_root=True,
    )
    assert isinstance(transition.input_norm, torch.nn.Identity)

    root = torch.randn(2, 4, 32, requires_grad=True)
    initial = transition.initialize(root)
    expected_root = transition._fixed_rms_normalize_hidden(root)
    torch.testing.assert_close(initial.hidden, expected_root)
    torch.testing.assert_close(initial.memory, torch.zeros_like(initial.memory))
    assert initial.write_count == 0

    step = transition(initial)
    expected_write = transition.write(step.rotated_hidden)
    expected_read = transition.read(
        transition.queries(step.rotated_hidden),
        expected_write,
    )
    expected_delta = transition.read_to_hidden(expected_read)
    expected_raw = step.rotated_hidden + expected_delta
    expected_hidden = transition._fixed_rms_normalize_hidden(expected_raw)
    torch.testing.assert_close(step.innovation_memory, expected_write)
    torch.testing.assert_close(step.state.memory, expected_write)
    torch.testing.assert_close(step.innovation_delta, expected_delta)
    torch.testing.assert_close(step.raw_full_hidden, expected_raw)
    torch.testing.assert_close(step.state.hidden, expected_hidden)
    torch.testing.assert_close(step.read_hidden, expected_hidden)
    assert step.state.write_count == 1

    # The predecessor hidden is an external value at this recurrent boundary,
    # but the predecessor complex memory is still a differentiable carrier.
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    memory = torch.complex(
        torch.randn(2, 4, 2, 3, 5),
        torch.randn(2, 4, 2, 3, 5),
    ).requires_grad_()
    live_step = transition(
        ComplexMemoryState(hidden=hidden, memory=memory, write_count=1)
    )
    probe = torch.randn_like(live_step.state.hidden)
    loss = (live_step.state.hidden * probe).sum()
    hidden_gradient, memory_gradient = torch.autograd.grad(
        loss,
        (hidden, memory),
        allow_unused=True,
    )
    assert hidden_gradient is None
    assert memory_gradient is not None
    assert torch.isfinite(memory_gradient).all()
    assert memory_gradient.norm() > 0


def test_full_bptt_raw_read_joint_postnorm_has_exact_attached_boundary():
    torch.manual_seed(172)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode=(
            "unitary-full-bptt-raw-read-joint-postnorm-residual"
        ),
        pre_normalize_qkv_inputs=False,
        normalize_initial_recurrent_root=True,
    )
    assert isinstance(transition.input_norm, torch.nn.Identity)

    root = torch.randn(2, 4, 32, requires_grad=True)
    initial = transition.initialize(root)
    expected_root = transition._fixed_rms_normalize_hidden(root)
    torch.testing.assert_close(initial.hidden, expected_root)
    torch.testing.assert_close(initial.memory, torch.zeros_like(initial.memory))
    assert initial.write_count == 0

    first = transition(initial)
    first_write = transition.write(first.rotated_hidden)
    expected_first_memory = transition._fixed_rms_normalize_complex_memory(
        first_write
    )
    expected_first_read = transition.read(
        transition.queries(first.rotated_hidden),
        expected_first_memory,
    )
    expected_first_delta = transition.read_to_hidden(expected_first_read)
    expected_first_raw = first.rotated_hidden + expected_first_delta
    expected_first_hidden = transition._fixed_rms_normalize_hidden(
        expected_first_raw
    )
    torch.testing.assert_close(first.raw_updated_memory, first_write)
    torch.testing.assert_close(first.state.memory, expected_first_memory)
    torch.testing.assert_close(first.innovation_delta, expected_first_delta)
    torch.testing.assert_close(first.raw_full_hidden, expected_first_raw)
    torch.testing.assert_close(first.state.hidden, expected_first_hidden)
    torch.testing.assert_close(first.read_hidden, expected_first_hidden)
    assert first.state.write_count == 1

    second = transition(first.state)
    second_write = transition.write(second.rotated_hidden)
    expected_second_raw_memory = second.rotated_memory + second_write
    expected_second_memory = transition._fixed_rms_normalize_complex_memory(
        expected_second_raw_memory
    )
    expected_second_read = transition.read(
        transition.queries(second.rotated_hidden),
        expected_second_memory,
    )
    expected_second_delta = transition.read_to_hidden(expected_second_read)
    expected_second_raw = second.rotated_hidden + expected_second_delta
    expected_second_hidden = transition._fixed_rms_normalize_hidden(
        expected_second_raw
    )
    torch.testing.assert_close(
        second.raw_updated_memory,
        expected_second_raw_memory,
    )
    torch.testing.assert_close(second.state.memory, expected_second_memory)
    torch.testing.assert_close(second.innovation_delta, expected_second_delta)
    torch.testing.assert_close(second.state.hidden, expected_second_hidden)
    assert second.state.write_count == 2

    for step in (first, second):
        hidden_rms = step.state.hidden.float().square().mean(dim=-1).sqrt()
        memory_rms = step.state.memory.abs().square().float().mean(
            dim=(-3, -2, -1)
        ).sqrt()
        assert hidden_rms.min() > 0.99
        assert hidden_rms.max() <= 1.0
        assert memory_rms.min() > 0.99
        assert memory_rms.max() <= 1.0

    loss = second.state.hidden.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_unitary_branch_normalized_residual_keeps_carriers_raw():
    torch.manual_seed(173)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        pre_normalize_qkv_inputs=True,
    )
    assert isinstance(transition.input_norm, torch.nn.Identity)
    root = torch.randn(2, 4, 32, requires_grad=True)
    initial = transition.initialize(root)
    step = transition(initial)

    normalized = root * torch.rsqrt(
        root.float().square().mean(dim=-1, keepdim=True)
        + transition.post_norm_eps
    )
    expected_write = transition.write(root)
    torch.testing.assert_close(expected_write, transition.write(normalized))
    expected_read = transition.read(
        transition.queries(root), expected_write
    )
    frobenius = expected_write.abs().square().sum(
        dim=(-3, -2, -1)
    ).sqrt()
    expected_read = expected_read / (
        frobenius[..., None, None] + transition.post_norm_eps
    )
    expected_delta = transition.read_to_hidden(expected_read)
    expected_carrier = step.rotated_hidden + expected_delta
    expected_decode = expected_carrier * torch.rsqrt(
        expected_carrier.float().square().mean(dim=-1, keepdim=True)
        + transition.post_norm_eps
    )
    torch.testing.assert_close(step.state.memory, expected_write)
    torch.testing.assert_close(step.innovation_delta, expected_delta)
    torch.testing.assert_close(step.state.hidden, expected_carrier)
    torch.testing.assert_close(step.read_hidden, expected_decode)
    assert not torch.allclose(step.state.hidden, step.read_hidden)

    second = transition(step.state)
    loss = second.read_hidden.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_unitary_branch_normalized_no_residual_uses_only_measurement_delta():
    torch.manual_seed(177)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-no-residual",
        pre_normalize_qkv_inputs=True,
    )
    root = torch.randn(2, 4, 32, requires_grad=True)
    initial = transition.initialize(root)
    step = transition(initial)

    expected_write = transition.write(step.rotated_hidden)
    expected_read = transition.read(
        transition.queries(step.rotated_hidden), expected_write
    )
    frobenius = expected_write.abs().square().sum(
        dim=(-3, -2, -1)
    ).sqrt()
    expected_read = expected_read / (
        frobenius[..., None, None] + transition.post_norm_eps
    )
    expected_delta = transition.read_to_hidden(expected_read)
    expected_decode = transition._fixed_rms_normalize_hidden(expected_delta)

    torch.testing.assert_close(step.state.memory, expected_write)
    torch.testing.assert_close(step.innovation_delta, expected_delta)
    torch.testing.assert_close(step.state.hidden, expected_delta)
    torch.testing.assert_close(step.read_hidden, expected_decode)
    assert not torch.allclose(
        step.state.hidden,
        step.rotated_hidden + expected_delta,
    )

    second = transition(step.state)
    loss = second.read_hidden.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_ce_only_fast_path_matches_full_loss_and_gradients():
    torch.manual_seed(179)
    full_model = make_model()
    full_model.complex_self_prediction = ComplexSelfPredictedKVTransition(
        full_model.width,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        residual_prior=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        pre_normalize_qkv_inputs=True,
    )
    fast_model = make_model()
    fast_model.complex_self_prediction = ComplexSelfPredictedKVTransition(
        fast_model.width,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        residual_prior=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        pre_normalize_qkv_inputs=True,
    )
    fast_model.load_state_dict(full_model.state_dict())
    window = torch.randint(0, 17, (2, 10))
    full_output = forward_sparse_complex_self_predicted_kv_window(
        full_model,
        window,
        prefix_length=6,
        horizons=4,
        anchor_stride=2,
    )
    full_loss = full_output.token_ce.mean()
    fast_loss = sparse_complex_self_predicted_kv_ce_only_fast_loss(
        fast_model,
        window,
        prefix_length=6,
        horizons=4,
        anchor_stride=2,
    )
    torch.testing.assert_close(fast_loss, full_loss)
    full_parameters = tuple(
        parameter for parameter in full_model.parameters()
        if parameter.requires_grad
    )
    fast_parameters = tuple(
        parameter for parameter in fast_model.parameters()
        if parameter.requires_grad
    )
    full_gradients = torch.autograd.grad(
        full_loss,
        full_parameters,
        allow_unused=True,
    )
    fast_gradients = torch.autograd.grad(
        fast_loss,
        fast_parameters,
        allow_unused=True,
    )
    for full_gradient, fast_gradient in zip(full_gradients, fast_gradients):
        if full_gradient is None or fast_gradient is None:
            assert full_gradient is None and fast_gradient is None
        else:
            torch.testing.assert_close(fast_gradient, full_gradient)


def test_time_varying_scan_ce_is_finite_and_h1_matches_feedback():
    torch.manual_seed(181)
    model = make_model()
    model.complex_self_prediction = ComplexSelfPredictedKVTransition(
        model.width,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        residual_prior=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        pre_normalize_qkv_inputs=True,
    )
    window = torch.randint(0, 17, (2, 10))
    feedback_h1 = sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window[:, :7],
        prefix_length=6,
        horizons=1,
        anchor_stride=2,
        central_rollout="feedback",
    )
    scan_h1 = sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window[:, :7],
        prefix_length=6,
        horizons=1,
        anchor_stride=2,
        central_rollout="time-varying-scan",
    )
    torch.testing.assert_close(scan_h1, feedback_h1, atol=2e-6, rtol=2e-6)

    scan_h4 = sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window,
        prefix_length=6,
        horizons=4,
        anchor_stride=2,
        central_rollout="time-varying-scan",
    )
    assert torch.isfinite(scan_h4)
    rescan_h4 = sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window,
        prefix_length=6,
        horizons=4,
        anchor_stride=2,
        central_rollout="time-varying-scan",
        feedback_rescans=1,
    )
    assert torch.isfinite(rescan_h4)
    gradients = torch.autograd.grad(
        scan_h4,
        tuple(parameter for parameter in model.parameters()
              if parameter.requires_grad),
        allow_unused=True,
    )
    nonzero = [
        gradient for gradient in gradients
        if gradient is not None and bool(gradient.norm() > 0)
    ]
    assert nonzero
    assert all(torch.isfinite(gradient).all() for gradient in nonzero)


def test_stochastic_ray_flow_loss_is_finite_and_detaches_sample_targets():
    torch.manual_seed(182)
    model = make_model()
    model.complex_self_prediction = ComplexSelfPredictedKVTransition(
        model.width,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        residual_prior=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        pre_normalize_qkv_inputs=True,
    )
    window = torch.randint(0, 17, (2, 10))
    total, token_nll, ray_flow, output = (
        sparse_complex_stochastic_ray_flow_loss(
            model,
            window,
            prefix_length=6,
            horizons=4,
            anchor_stride=2,
            value_noise_scale=0.05,
            ray_flow_weight=0.01,
            ray_flow_horizons=4,
        )
    )
    torch.testing.assert_close(total, token_nll + 0.01 * ray_flow)
    assert torch.isfinite(total)
    assert torch.isfinite(ray_flow)
    assert output.ray_velocity_loss_by_horizon.shape == (4,)
    assert output.ray_velocity_cosine_by_horizon.shape == (4,)
    target_gradient = torch.autograd.grad(
        ray_flow,
        output.gold_states,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert target_gradient is None

    gradients = torch.autograd.grad(
        total,
        tuple(
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ),
        allow_unused=True,
    )
    nonzero = [
        gradient for gradient in gradients
        if gradient is not None and bool(gradient.norm() > 0)
    ]
    assert nonzero
    assert all(torch.isfinite(gradient).all() for gradient in nonzero)


def test_post_update_full_read_normalizes_only_accumulated_memory_reads():
    torch.manual_seed(171)
    control = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
    )
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
        normalize_accumulated_reads=True,
    )
    transition.load_state_dict(control.state_dict())
    root = torch.randn(2, 4, 32)
    state = transition.initialize(root)
    control_first = control(control.initialize(root))
    assert state.write_count == 1

    first = transition(state)
    torch.testing.assert_close(
        first.preliminary_hidden,
        control_first.preliminary_hidden,
    )
    torch.testing.assert_close(first.innovation_delta, control_first.innovation_delta)
    torch.testing.assert_close(first.state.memory, control_first.state.memory)
    torch.testing.assert_close(
        first.state.hidden,
        control_first.state.hidden / (2.0**0.5),
    )
    first_query = transition.queries(first.preliminary_hidden)
    expected_first_full = transition.read_to_hidden(
        transition.read(first_query, first.state.memory) / (2.0**0.5)
    )
    torch.testing.assert_close(first.state.hidden, expected_first_full)
    torch.testing.assert_close(first.read_hidden, first.preliminary_hidden)
    assert first.state.write_count == 2

    second = transition(first.state)
    rotated_memory = first.state.memory * torch.polar(
        torch.ones_like(transition.memory_phase),
        -transition.memory_phase,
    )[..., None]
    expected_second_prior = transition.read_to_hidden(
        transition.read(
            transition.queries(second.rotated_hidden),
            rotated_memory,
        )
        / (2.0**0.5)
    )
    second_query = transition.queries(second.preliminary_hidden)
    expected_second_full = transition.read_to_hidden(
        transition.read(second_query, second.state.memory) / (3.0**0.5)
    )
    torch.testing.assert_close(second.preliminary_hidden, expected_second_prior)
    torch.testing.assert_close(second.state.hidden, expected_second_full)
    assert second.state.write_count == 3


def test_urm_post_norm_bounds_complete_recurrent_carrier_and_gradients():
    torch.manual_seed(181)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
        normalize_accumulated_reads=False,
        post_normalize_hidden_reads=True,
        post_normalize_memory=True,
    )
    root = torch.randn(2, 4, 32, requires_grad=True)
    initial = transition.initialize(root)
    rollout = transition.rollout(root, 4)

    initial_memory_energy = initial.memory.abs().square().mean(
        dim=(-3, -2, -1)
    )
    assert initial_memory_energy.min() > 0.9
    assert initial_memory_energy.max() <= 1.0
    assert rollout.memory_energy.min() > 0.9
    assert rollout.memory_energy.max() <= 1.0
    for hidden in (
        rollout.preliminary_states,
        rollout.read_states,
        rollout.full_states,
    ):
        hidden_rms = hidden.float().square().mean(dim=-1).sqrt()
        # RMSNorm is bounded by one, and is only approximately one when the
        # deliberately tiny initial readout variance is comparable to eps.
        assert hidden_rms.min() > 0.8
        assert hidden_rms.max() <= 1.0

    probe = torch.linspace(
        -1.0,
        1.0,
        rollout.read_states.shape[-1],
        dtype=rollout.read_states.dtype,
    )
    loss = (rollout.read_states[:, :, -1] * probe).mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_final_carrier_post_norm_occurs_after_complete_raw_transition():
    torch.manual_seed(191)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
        normalize_accumulated_reads=False,
        post_normalize_recurrent_state=True,
    )
    root = torch.randn(2, 4, 32, requires_grad=True)
    initial = transition.initialize(root)
    torch.testing.assert_close(initial.memory, transition.write(root))

    step = transition(initial)
    hidden_angles = transition.hidden_phase.expand(*root.shape[:-1], -1)
    rotated_hidden = rotate_pairwise(root, hidden_angles)
    memory_phase = torch.polar(
        torch.ones_like(transition.memory_phase),
        -transition.memory_phase,
    )
    rotated_memory = initial.memory * memory_phase[..., None]
    raw_preliminary = transition.read_to_hidden(
        transition.read(transition.queries(rotated_hidden), rotated_memory)
    )
    raw_memory = rotated_memory + transition.write(raw_preliminary)
    raw_hidden = transition.read_to_hidden(
        transition.read(transition.queries(raw_preliminary), raw_memory)
    )
    expected_hidden = raw_hidden * torch.rsqrt(
        raw_hidden.float().square().mean(dim=-1, keepdim=True)
        + transition.post_norm_eps
    )
    expected_memory = raw_memory * torch.rsqrt(
        raw_memory.abs().square().float().mean(
            dim=(-3, -2, -1),
            keepdim=True,
        )
        + transition.post_norm_eps
    )

    torch.testing.assert_close(step.preliminary_hidden, raw_preliminary)
    torch.testing.assert_close(step.read_hidden, raw_preliminary)
    torch.testing.assert_close(step.state.hidden, expected_hidden)
    torch.testing.assert_close(step.state.memory, expected_memory)
    assert step.preliminary_hidden.float().square().mean().sqrt() < 0.1

    rollout = transition.rollout(root, 4)
    loss = rollout.read_states.square().mean() + rollout.full_states.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (
            root,
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_initial_recurrent_root_norm_precedes_hidden_and_first_kv_write():
    torch.manual_seed(193)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        pre_normalize_qkv_inputs=False,
        normalize_initial_recurrent_root=True,
        post_norm_eps=1e-6,
    )
    root = (7.0 * torch.randn(2, 4, 32)).requires_grad_()
    state = transition.initialize(root)
    expected_root = root * torch.rsqrt(
        root.float().square().mean(dim=-1, keepdim=True)
        + transition.post_norm_eps
    )

    torch.testing.assert_close(state.hidden, expected_root)
    torch.testing.assert_close(state.memory, transition.write(expected_root))
    assert not torch.allclose(state.memory, transition.write(root))
    root_rms = state.hidden.float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(root_rms, torch.ones_like(root_rms), atol=1e-6, rtol=1e-6)

    loss = state.memory.abs().square().mean() + state.hidden.square().mean()
    gradients = torch.autograd.grad(
        loss,
        (root, transition.key.weight, transition.value.weight),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_initial_recurrent_root_norm_default_false_is_exactly_legacy():
    torch.manual_seed(194)
    legacy = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
    )
    explicit_false = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
        normalize_initial_recurrent_root=False,
    )
    explicit_false.load_state_dict(legacy.state_dict())
    root = torch.randn(2, 4, 32)

    legacy_initial = legacy.initialize(root)
    explicit_initial = explicit_false.initialize(root)
    torch.testing.assert_close(legacy_initial.hidden, root, rtol=0, atol=0)
    torch.testing.assert_close(legacy_initial.memory, legacy.write(root), rtol=0, atol=0)
    torch.testing.assert_close(
        explicit_initial.hidden,
        legacy_initial.hidden,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        explicit_initial.memory,
        legacy_initial.memory,
        rtol=0,
        atol=0,
    )
    legacy_rollout = legacy.rollout(root, 3)
    explicit_rollout = explicit_false.rollout(root, 3)
    for legacy_values, explicit_values in (
        (legacy_rollout.read_states, explicit_rollout.read_states),
        (legacy_rollout.full_states, explicit_rollout.full_states),
        (legacy_rollout.final_state.memory, explicit_rollout.final_state.memory),
    ):
        torch.testing.assert_close(
            explicit_values,
            legacy_values,
            rtol=0,
            atol=0,
        )


def test_initial_recurrent_root_norm_does_not_modify_raw_encoder_inverse_path():
    model = make_model().eval()
    tokens = torch.randint(0, 17, (2, 7))
    positions = torch.arange(tokens.shape[1])
    anchors = torch.tensor([1, 3, 5])
    encoded_before, _ = model.encode(tokens, positions)
    branch_before = encoded_before.index_select(1, anchors).unsqueeze(2)
    decoded_before = model.exact_inverse_selected_decode_tape_states(
        encoded_before,
        branch_before,
        positions,
        anchors,
    )

    model.complex_self_prediction.normalize_initial_recurrent_root = True
    encoded_after, _ = model.encode(tokens, positions)
    branch_after = encoded_after.index_select(1, anchors).unsqueeze(2)
    decoded_after = model.exact_inverse_selected_decode_tape_states(
        encoded_after,
        branch_after,
        positions,
        anchors,
    )

    torch.testing.assert_close(encoded_after, encoded_before, rtol=0, atol=0)
    torch.testing.assert_close(decoded_after, decoded_before, rtol=0, atol=0)
    recurrent = model.complex_self_prediction.initialize(
        encoded_after.index_select(1, anchors)
    )
    assert not torch.allclose(recurrent.hidden, branch_after.squeeze(2))
    recurrent_rms = recurrent.hidden.float().square().mean(dim=-1).sqrt()
    assert recurrent_rms.min() > 0.99
    assert recurrent_rms.max() <= 1.0


def test_common_training_factory_exposes_initial_recurrent_root_norm(monkeypatch):
    import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base

    monkeypatch.setattr(base, "VOCABULARY", 17)
    monkeypatch.setattr(base, "WIDTH", 32)
    monkeypatch.setattr(base, "ENCODER_BLOCKS", 1)
    monkeypatch.setattr(base, "HEADS", 2)
    monkeypatch.setattr(base, "KEY_DIM", 3)
    monkeypatch.setattr(base, "VALUE_DIM", 5)
    monkeypatch.setattr(base, "NORMALIZE_INITIAL_RECURRENT_ROOT", True)
    model = base.make_model()

    assert model.complex_self_prediction.normalize_initial_recurrent_root


def test_final_carrier_and_intermediate_post_norms_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        ComplexSelfPredictedKVTransition(
            32,
            heads=2,
            key_dim=3,
            value_dim=5,
            post_normalize_hidden_reads=True,
            post_normalize_recurrent_state=True,
        )


def test_complex_rollout_detaches_only_cross_horizon_state_gradients():
    torch.manual_seed(197)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
        normalize_accumulated_reads=True,
    )
    root = torch.randn(2, 4, 32, requires_grad=True)
    attached = transition.rollout(root, 3)
    detached = transition.rollout(
        root,
        3,
        detach_state_between_horizons=True,
    )

    for attached_values, detached_values in (
        (attached.read_states, detached.read_states),
        (attached.full_states, detached.full_states),
        (attached.preliminary_states, detached.preliminary_states),
        (attached.rotated_states, detached.rotated_states),
        (attached.innovation_deltas, detached.innovation_deltas),
        (attached.memory_energy, detached.memory_energy),
        (attached.final_state.hidden, detached.final_state.hidden),
        (attached.final_state.memory, detached.final_state.memory),
    ):
        torch.testing.assert_close(attached_values, detached_values)

    h2_loss = detached.read_states[:, :, 1].square().mean()
    root_gradient = torch.autograd.grad(
        h2_loss,
        root,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert root_gradient is None or torch.count_nonzero(root_gradient) == 0
    query_gradient = torch.autograd.grad(
        h2_loss,
        transition.query.weight,
    )[0]
    assert torch.isfinite(query_gradient).all()
    assert query_gradient.norm() > 0


def test_complex_window_detaches_later_ce_from_earlier_decoder_slots():
    model = make_model()
    model.complex_self_prediction.recurrent_hidden_mode = (
        "post-update-full-read"
    )
    model.complex_self_prediction.beta_logit = None
    model.complex_self_prediction.normalize_accumulated_reads = True
    window = torch.randint(0, 17, (2, 10))

    attached = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=7,
        horizons=3,
        anchor_stride=2,
    )
    detached = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=7,
        horizons=3,
        anchor_stride=2,
        detach_cross_horizon_gradients=True,
    )
    for attached_values, detached_values in (
        (attached.read_states, detached.read_states),
        (attached.full_states, detached.full_states),
        (attached.preliminary_states, detached.preliminary_states),
        (attached.rotated_states, detached.rotated_states),
        (attached.innovation_deltas, detached.innovation_deltas),
        (attached.memory_energy, detached.memory_energy),
        (attached.logits, detached.logits),
    ):
        torch.testing.assert_close(
            attached_values,
            detached_values,
            atol=5e-5,
            rtol=5e-5,
        )

    h3_loss = detached.token_ce[:, :, 2].mean()
    read_state_gradient = torch.autograd.grad(
        h3_loss,
        detached.read_states,
    )[0]
    assert torch.count_nonzero(read_state_gradient[:, :, :2]) == 0
    assert read_state_gradient[:, :, 2].norm() > 0


def test_detached_preliminary_preserves_forward_and_cuts_corrector_edge():
    torch.manual_seed(211)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="post-update-full-read",
    )
    root = torch.randn(2, 4, 32, requires_grad=True)
    state = transition.initialize(root)
    attached = transition(state)
    detached = transition(
        state,
        detach_preliminary_for_update=True,
    )

    torch.testing.assert_close(attached.read_hidden, detached.read_hidden)
    torch.testing.assert_close(attached.state.hidden, detached.state.hidden)
    torch.testing.assert_close(attached.state.memory, detached.state.memory)

    attached_gradient = torch.autograd.grad(
        attached.state.hidden.square().mean(),
        attached.preliminary_hidden,
        retain_graph=True,
    )[0]
    detached_gradient = torch.autograd.grad(
        detached.state.hidden.square().mean(),
        detached.preliminary_hidden,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert attached_gradient.norm() > 0
    assert detached_gradient is None

    # The corrector still learns from the fixed P value.
    parameter_gradients = torch.autograd.grad(
        detached.state.hidden.square().mean(),
        (
            transition.query.weight,
            transition.key.weight,
            transition.value.weight,
            transition.output.weight,
        ),
    )
    for gradient in parameter_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_online_stop_gradient_latent_target_changes_only_the_backward_path():
    model = make_model()
    window = torch.randint(0, 17, (2, 8))
    attached = sparse_complex_self_predicted_kv_loss(
        model,
        window,
        prefix_length=7,
        horizons=1,
        anchor_stride=1,
    )
    detached = sparse_complex_self_predicted_kv_loss(
        model,
        window,
        prefix_length=7,
        horizons=1,
        anchor_stride=1,
        detach_latent_targets=True,
    )
    torch.testing.assert_close(attached[0], detached[0])
    torch.testing.assert_close(attached[1], detached[1])
    torch.testing.assert_close(attached[2], detached[2])
    torch.testing.assert_close(attached[3].full_states, detached[3].full_states)

    target_gradient = torch.autograd.grad(
        detached[2],
        detached[3].gold_states,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert target_gradient is None
    prediction_gradients = torch.autograd.grad(
        detached[2],
        (
            model.complex_self_prediction.value.weight,
            model.complex_self_prediction.output.weight,
            model.encoder.blocks[0].attn.qkv.weight,
        ),
    )
    for gradient in prediction_gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_ema_latent_target_changes_only_target_values_and_has_no_gradient():
    model = make_model()
    ema_model = make_ema_model(model)
    with torch.no_grad():
        model.encoder.blocks[0].attn.qkv.weight.add_(0.01)
    window = torch.randint(0, 17, (2, 8))

    online_target = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=7,
        horizons=1,
        anchor_stride=1,
        detach_latent_targets=True,
    )
    ema_target = sparse_complex_self_predicted_kv_loss(
        model,
        window,
        prefix_length=7,
        horizons=1,
        anchor_stride=1,
        latent_target_model=ema_model,
    )
    output = ema_target[3]

    torch.testing.assert_close(output.full_states, online_target.full_states)
    torch.testing.assert_close(output.logits, online_target.logits)
    assert not torch.allclose(output.gold_states, online_target.gold_states)
    assert not output.gold_states.requires_grad
    assert all(not parameter.requires_grad for parameter in ema_model.parameters())

    gradients = torch.autograd.grad(
        ema_target[2],
        (
            model.complex_self_prediction.value.weight,
            model.complex_self_prediction.output.weight,
            model.encoder.blocks[0].attn.qkv.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    assert all(parameter.grad is None for parameter in ema_model.parameters())


def test_ema_self_redecode_kl_uses_own_token_and_frozen_decoder():
    model = make_model()
    ema_model = make_ema_model(model)
    window = torch.randint(0, 17, (2, 8))
    loss, token_ce, redecode_kl, output = (
        sparse_complex_self_prediction_self_redecode_kl_loss(
            model,
            ema_model,
            window,
            prefix_length=7,
            anchor_stride=1,
        )
    )

    torch.testing.assert_close(loss, token_ce + redecode_kl)
    torch.testing.assert_close(
        output.selected_tokens,
        output.base.logits.detach().argmax(dim=-1),
    )
    torch.testing.assert_close(
        output.canonical_logits.argmax(dim=-1),
        output.selected_tokens,
    )
    assert not output.canonical_states.requires_grad
    assert not output.canonical_logits.requires_grad
    assert output.student_logits.requires_grad
    assert all(not parameter.requires_grad for parameter in ema_model.parameters())

    gradients = torch.autograd.grad(
        redecode_kl,
        (
            output.base.full_states,
            model.complex_self_prediction.query.weight,
            model.complex_self_prediction.key.weight,
            model.complex_self_prediction.value.weight,
            model.complex_self_prediction.output.weight,
            model.encoder.blocks[0].attn.qkv.weight,
        ),
        retain_graph=True,
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    assert all(parameter.grad is None for parameter in ema_model.parameters())

    changed = window.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 17
    changed_output = forward_sparse_complex_self_prediction_self_redecode_kl(
        model,
        ema_model,
        changed,
        prefix_length=7,
        anchor_stride=1,
    )
    for original, modified in (
        (output.base.logits, changed_output.base.logits),
        (output.base.full_states, changed_output.base.full_states),
        (output.selected_tokens, changed_output.selected_tokens),
        (output.canonical_states, changed_output.canonical_states),
        (output.canonical_logits, changed_output.canonical_logits),
        (output.student_logits, changed_output.student_logits),
    ):
        torch.testing.assert_close(original, modified)


def test_p_detached_joint_hidden_and_decode_closure_share_canonical_target():
    model = make_model()
    model.complex_self_prediction.recurrent_hidden_mode = (
        "post-update-full-read"
    )
    model.complex_self_prediction.beta_logit = None
    ema_model = make_ema_model(model)
    window = torch.randint(0, 17, (2, 8))

    loss, token_ce, closure, output = (
        sparse_complex_self_prediction_self_redecode_kl_loss(
            model,
            ema_model,
            window,
            prefix_length=7,
            anchor_stride=1,
            kl_weight=1.0,
            hidden_mse_weight=1.0,
            detach_preliminary_for_update=True,
        )
    )
    expected_closure = (
        output.canonical_student_kl.mean()
        + output.canonical_relative_mse.mean()
    )
    torch.testing.assert_close(closure, expected_closure)
    torch.testing.assert_close(loss, token_ce + expected_closure)

    ordinary = forward_sparse_complex_self_prediction_self_redecode_kl(
        model,
        ema_model,
        window,
        prefix_length=7,
        anchor_stride=1,
    )
    torch.testing.assert_close(output.base.logits, ordinary.base.logits)
    torch.testing.assert_close(
        output.base.full_states,
        ordinary.base.full_states,
    )
    torch.testing.assert_close(output.student_logits, ordinary.student_logits)

    gradients = torch.autograd.grad(
        closure,
        (
            output.base.full_states,
            model.complex_self_prediction.query.weight,
            model.complex_self_prediction.key.weight,
            model.complex_self_prediction.value.weight,
            model.complex_self_prediction.output.weight,
            model.encoder.blocks[0].attn.qkv.weight,
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
    assert all(parameter.grad is None for parameter in ema_model.parameters())


def test_shared_kv_heads_projects_one_query_key_broadcast_over_heads():
    """Mamba's multi-value-attention layout: B/C shared, only X per head."""
    heads, key_dim, value_dim = 4, 3, 5
    shared = ComplexSelfPredictedKVTransition(
        32,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        shared_kv_heads=True,
    )
    per_head = ComplexSelfPredictedKVTransition(
        32,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
    )

    # Q/K cost drops from width*heads*key_dim to width*key_dim for the layer;
    # the per-head value projection is untouched.
    assert shared.query.weight.shape == (2 * key_dim, 32)
    assert per_head.query.weight.shape == (2 * heads * key_dim, 32)
    assert shared.value.weight.shape == per_head.value.weight.shape

    hidden = torch.randn(2, 4, 32)
    query, key, value = shared.project_qkv(hidden)
    assert query.shape == (2, 4, heads, key_dim)
    assert key.shape == (2, 4, heads, key_dim)
    assert value.shape == (2, 4, heads, value_dim)
    for head in range(1, heads):
        torch.testing.assert_close(query[..., head, :], query[..., 0, :])
        torch.testing.assert_close(key[..., head, :], key[..., 0, :])
    # Values must stay independent across heads, otherwise this is not MVA.
    assert not torch.allclose(value[..., 1, :], value[..., 0, :])

    # The dedicated accessors must agree with the packed projection.
    torch.testing.assert_close(shared.queries(hidden), query)
    torch.testing.assert_close(shared.keys(hidden), key)


def test_shared_kv_heads_scan_still_matches_literal_recurrence():
    """The MVA layout must not disturb the exact prefix composition."""
    torch.manual_seed(94)
    transition = ComplexSelfPredictedKVTransition(
        32,
        heads=2,
        key_dim=3,
        value_dim=5,
        learned_read_beta=False,
        recurrent_hidden_mode="unitary-branch-normalized-residual",
        shared_kv_heads=True,
    )
    root = torch.randn(2, 4, 32)
    scanned = transition.rollout_time_varying_scan(root, 5)

    conditioning = root
    hidden = root
    memory = torch.zeros(2, 4, 2, 3, 5, dtype=torch.complex64)
    memory_multiplier = torch.polar(
        torch.ones_like(transition.memory_phase),
        -transition.memory_phase,
    )[..., None]
    sequential_hidden = []
    for _ in range(5):
        angles = transition.hidden_phase.expand(2, 4, -1)
        conditioning = rotate_pairwise(conditioning, angles)
        query, key, value = transition.project_qkv(conditioning)
        memory = memory_multiplier * memory + torch.einsum(
            "...hk,...hv->...hkv", key, value.conj()
        )
        frobenius = memory.abs().square().sum(dim=(-3, -2, -1)).sqrt()
        read = transition.read(query, memory) / (
            frobenius[..., None, None] + transition.post_norm_eps
        )
        hidden = rotate_pairwise(hidden, angles) + transition.read_to_hidden(read)
        sequential_hidden.append(hidden)

    torch.testing.assert_close(
        scanned.full_states,
        torch.stack(sequential_hidden, dim=-2),
        atol=8e-6,
        rtol=8e-6,
    )
