import torch

from rotlm.models import (
    AttachedSpectralStateCorrector,
    DetachedSpectralStateCorrector,
    InputConditionedSpectralCollapse,
    K1DecoderAblationLM,
)
from rotlm.training import (
    forward_sparse_attached_spectral_one_step,
    forward_sparse_self_canonical_redecode_kl_one_step,
    forward_sparse_spectral_corrector_window,
    sparse_attached_spectral_one_step_loss,
    sparse_self_canonical_redecode_kl_one_step_loss,
    sparse_spectral_corrector_loss,
)


def make_model() -> K1DecoderAblationLM:
    torch.manual_seed(47)
    model = K1DecoderAblationLM(
        43,
        width=32,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
    )
    model.spectral_collapse = InputConditionedSpectralCollapse(
        32,
        bottleneck=16,
        alpha=0.0,
    )
    model.spectral_corrector = DetachedSpectralStateCorrector(
        32,
        bottleneck=16,
    )
    calibration = torch.randn(2, 4, 32)
    model.spectral_collapse.calibrate_shell(calibration)
    return model


def nonzero_or_none(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool = True,
) -> list[torch.Tensor | None]:
    return list(
        torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=retain_graph,
        )
    )


def test_spectral_corrector_is_identity_and_detaches_source_and_basis():
    torch.manual_seed(53)
    corrector = DetachedSpectralStateCorrector(32, bottleneck=16)
    source = torch.randn(2, 3, 4, 32, requires_grad=True)
    generator = torch.randn(32, 32)
    skew = generator - generator.T
    basis = torch.matrix_exp(skew).requires_grad_()
    output = corrector(source, basis)

    torch.testing.assert_close(
        output.states,
        source,
        atol=2e-5,
        rtol=2e-5,
    )
    loss = output.states.square().sum()
    source_gradient, basis_gradient = torch.autograd.grad(
        loss,
        (source, basis),
        allow_unused=True,
    )
    assert source_gradient is None
    assert basis_gradient is None


def test_spectral_corrector_target_matches_literal_selected_tape():
    model = make_model().eval()
    window = torch.randint(0, 43, (2, 9))
    output = forward_sparse_spectral_corrector_window(
        model,
        window,
        prefix_length=6,
        horizons=3,
        anchor_stride=2,
    )
    for selected_index, anchor in enumerate(
        output.anchor_indices.tolist()
    ):
        tape = torch.cat(
            (
                window[:, : anchor + 1],
                output.selected_tokens[:, selected_index],
            ),
            dim=1,
        )
        expected, _ = model.encode(tape)
        torch.testing.assert_close(
            output.canonical_states[:, selected_index],
            expected[:, -3:],
            atol=4e-5,
            rtol=4e-5,
        )


def test_spectral_corrector_loss_has_disjoint_gradient_ownership():
    model = make_model()
    window = torch.randint(0, 43, (2, 9))
    loss, token_nll, corrector_mse, output = (
        sparse_spectral_corrector_loss(
            model,
            window,
            prefix_length=6,
            horizons=3,
            anchor_stride=1,
        )
    )
    base_parameters = [
        model.spectral_collapse.basis_generator,
        model.spectral_collapse.phase.weight,
        model.spectral_collapse.decay.weight,
        model.encoder.embed.weight,
    ]
    corrector_parameters = [
        model.spectral_corrector.log_scale_head.weight,
        model.spectral_corrector.phase_head.weight,
    ]

    ce_base = nonzero_or_none(token_nll, base_parameters)
    ce_corrector = nonzero_or_none(token_nll, corrector_parameters)
    mse_base = nonzero_or_none(corrector_mse, base_parameters)
    mse_corrector = nonzero_or_none(
        corrector_mse,
        corrector_parameters,
    )
    mse_proposal = torch.autograd.grad(
        corrector_mse,
        output.proposal_states,
        allow_unused=True,
        retain_graph=True,
    )[0]
    total_base = nonzero_or_none(loss, base_parameters)

    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.norm() > 0
        for gradient in ce_base
    )
    assert all(gradient is None for gradient in ce_corrector)
    assert all(gradient is None for gradient in mse_base)
    assert mse_proposal is None
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.norm() > 0
        for gradient in mse_corrector
    )
    for total_gradient, ce_gradient in zip(total_base, ce_base):
        torch.testing.assert_close(
            total_gradient,
            ce_gradient,
            atol=0,
            rtol=0,
        )


def test_spectral_corrector_does_not_read_future_corpus_tokens():
    model = make_model().eval()
    window = torch.randint(0, 43, (2, 9))
    changed = window.clone()
    changed[:, 6:] = (changed[:, 6:] + 7) % 43
    first = forward_sparse_spectral_corrector_window(
        model,
        window,
        prefix_length=6,
        horizons=3,
        anchor_stride=1,
    )
    second = forward_sparse_spectral_corrector_window(
        model,
        changed,
        prefix_length=6,
        horizons=3,
        anchor_stride=1,
    )
    for left, right in (
        (first.logits, second.logits),
        (first.proposal_states, second.proposal_states),
        (first.selected_tokens, second.selected_tokens),
        (first.canonical_states, second.canonical_states),
        (first.corrected_states, second.corrected_states),
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_attached_spectral_corrector_propagates_source_and_basis_gradients():
    torch.manual_seed(59)
    corrector = AttachedSpectralStateCorrector(32, bottleneck=16)
    source = torch.randn(2, 3, 1, 32, requires_grad=True)
    generator = torch.randn(32, 32)
    basis = torch.matrix_exp(generator - generator.T).requires_grad_()
    output = corrector(source, basis)
    torch.testing.assert_close(
        output.states,
        source,
        atol=2e-5,
        rtol=2e-5,
    )
    source_gradient, basis_gradient = torch.autograd.grad(
        output.states.square().sum(),
        (source, basis),
    )
    assert source_gradient.norm() > 0
    assert basis_gradient.norm() > 0


def test_attached_one_step_t_regression_updates_base_but_not_ce_readout():
    model = make_model()
    model.spectral_corrector = AttachedSpectralStateCorrector(
        32,
        bottleneck=16,
    )
    window = torch.randint(0, 43, (2, 7))
    loss, token_nll, corrector_mse, output = (
        sparse_attached_spectral_one_step_loss(
            model,
            window,
            prefix_length=6,
            anchor_stride=1,
        )
    )
    base_parameters = [
        model.spectral_collapse.basis_generator,
        model.spectral_collapse.phase.weight,
        model.encoder.embed.weight,
    ]
    corrector_parameters = [
        model.spectral_corrector.log_scale_head.weight,
        model.spectral_corrector.phase_head.weight,
    ]
    ce_corrector = nonzero_or_none(token_nll, corrector_parameters)
    mse_base = nonzero_or_none(corrector_mse, base_parameters)
    mse_corrector = nonzero_or_none(corrector_mse, corrector_parameters)
    proposal_gradient = torch.autograd.grad(
        corrector_mse,
        output.proposal_states,
        retain_graph=True,
    )[0]
    assert all(gradient is None for gradient in ce_corrector)
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.norm() > 0
        for gradient in (*mse_base, *mse_corrector)
    )
    assert proposal_gradient.norm() > 0
    torch.testing.assert_close(loss, token_nll + corrector_mse)

    changed = window.clone()
    changed[:, -1] = (changed[:, -1] + 5) % 43
    first = forward_sparse_attached_spectral_one_step(
        model,
        window,
        prefix_length=6,
        anchor_stride=1,
    )
    second = forward_sparse_attached_spectral_one_step(
        model,
        changed,
        prefix_length=6,
        anchor_stride=1,
    )
    for left, right in (
        (first.logits, second.logits),
        (first.proposal_states, second.proposal_states),
        (first.selected_tokens, second.selected_tokens),
        (first.canonical_states, second.canonical_states),
        (first.corrected_states, second.corrected_states),
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_self_redecode_kl_teacher_is_canonical_and_stopped():
    model = make_model()
    model.spectral_corrector = AttachedSpectralStateCorrector(
        32,
        bottleneck=16,
    )
    window = torch.randint(0, 43, (2, 7))
    output = forward_sparse_self_canonical_redecode_kl_one_step(
        model,
        window,
        prefix_length=6,
        anchor_stride=1,
    )

    assert not output.canonical_states.requires_grad
    assert not output.teacher_logits.requires_grad
    assert output.corrected_logits.requires_grad
    torch.testing.assert_close(
        output.corrected_states,
        output.proposal_states,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        output.corrected_logits,
        output.logits,
        atol=5e-5,
        rtol=5e-5,
    )
    torch.testing.assert_close(
        output.corrected_teacher_kl,
        output.raw_teacher_kl,
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.equal(
        output.teacher_logits.argmax(dim=-1),
        output.selected_tokens,
    )

    for selected_index, anchor in enumerate(
        output.anchor_indices.tolist()
    ):
        literal = torch.cat(
            (
                window[:, : anchor + 1],
                output.selected_tokens[:, selected_index],
            ),
            dim=1,
        )
        expected, _ = model.encode(literal)
        torch.testing.assert_close(
            output.canonical_states[:, selected_index, 0],
            expected[:, -1],
            atol=4e-5,
            rtol=4e-5,
        )


def test_self_redecode_kl_gradient_routing_and_future_nonleakage():
    model = make_model()
    model.spectral_corrector = AttachedSpectralStateCorrector(
        32,
        bottleneck=16,
    )
    window = torch.randint(0, 43, (2, 7))
    loss, token_nll, corrector_kl, output = (
        sparse_self_canonical_redecode_kl_one_step_loss(
            model,
            window,
            prefix_length=6,
            anchor_stride=1,
        )
    )
    corrector_parameters = [
        model.spectral_corrector.log_scale_head.weight,
        model.spectral_corrector.phase_head.weight,
    ]
    student_parameters = [
        model.spectral_collapse.basis_generator,
        model.spectral_collapse.phase.weight,
        model.encoder.blocks[0].attn.qkv.weight,
        model.encoder.embed.weight,
    ]

    ce_corrector = nonzero_or_none(token_nll, corrector_parameters)
    kl_corrector = nonzero_or_none(corrector_kl, corrector_parameters)
    kl_student = nonzero_or_none(corrector_kl, student_parameters)
    proposal_gradient = torch.autograd.grad(
        corrector_kl,
        output.proposal_states,
        retain_graph=True,
    )[0]
    assert all(gradient is None for gradient in ce_corrector)
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.norm() > 0
        for gradient in (*kl_corrector, *kl_student)
    )
    assert proposal_gradient.norm() > 0
    torch.testing.assert_close(loss, token_nll + corrector_kl)

    changed = window.clone()
    changed[:, -1] = (changed[:, -1] + 5) % 43
    changed_output = forward_sparse_self_canonical_redecode_kl_one_step(
        model,
        changed,
        prefix_length=6,
        anchor_stride=1,
    )
    for left, right in (
        (output.logits, changed_output.logits),
        (output.proposal_states, changed_output.proposal_states),
        (output.selected_tokens, changed_output.selected_tokens),
        (output.canonical_states, changed_output.canonical_states),
        (output.teacher_logits, changed_output.teacher_logits),
        (output.corrected_states, changed_output.corrected_states),
        (output.corrected_logits, changed_output.corrected_logits),
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_self_redecode_kl_can_attach_teacher_branch():
    model = make_model()
    model.spectral_corrector = AttachedSpectralStateCorrector(
        32,
        bottleneck=16,
    )
    window = torch.randint(0, 43, (2, 7))
    loss, token_nll, corrector_kl, output = (
        sparse_self_canonical_redecode_kl_one_step_loss(
            model,
            window,
            prefix_length=6,
            anchor_stride=1,
            detach_teacher=False,
        )
    )

    assert output.canonical_states.requires_grad
    assert output.teacher_logits.requires_grad
    assert output.corrected_logits.requires_grad
    assert torch.equal(
        output.teacher_logits.argmax(dim=-1),
        output.selected_tokens,
    )
    teacher_logit_gradient, canonical_state_gradient = torch.autograd.grad(
        corrector_kl,
        (output.teacher_logits, output.canonical_states),
        retain_graph=True,
    )
    assert torch.isfinite(teacher_logit_gradient).all()
    assert teacher_logit_gradient.norm() > 0
    assert torch.isfinite(canonical_state_gradient).all()
    assert canonical_state_gradient.norm() > 0

    corrector_parameters = [
        model.spectral_corrector.log_scale_head.weight,
        model.spectral_corrector.phase_head.weight,
    ]
    assert all(
        gradient is None
        for gradient in nonzero_or_none(token_nll, corrector_parameters)
    )
    assert all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.norm() > 0
        for gradient in nonzero_or_none(
            corrector_kl,
            corrector_parameters,
        )
    )
    torch.testing.assert_close(loss, token_nll + corrector_kl)
