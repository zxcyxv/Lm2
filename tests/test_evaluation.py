import torch

from rotlm.evaluation import (
    attached_t_block_rollouts,
    continuation_metrics,
    generate_input_conditioned_composition_blocks,
    generate_reanchored,
    generate_token_conditioned,
    input_conditioned_spectral_composition_rollouts,
    spectral_block_boundary_rollouts,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.models.spectral_corrector import AttachedSpectralStateCorrector
from rotlm.models.spectral_collapse import InputConditionedSpectralCollapse
from rotlm.models.token_conditioned_transition import (
    TokenConditionedTransition,
)


def make_model() -> K1DecoderAblationLM:
    torch.manual_seed(73)
    model = K1DecoderAblationLM(
        41,
        width=32,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="rms-tied",
    )
    model.token_conditioned_transition = TokenConditionedTransition(
        32,
        bottleneck=8,
    )
    return model.eval()


def test_token_conditioned_generation_is_canonical_at_first_decision():
    model = make_model()
    prompt = torch.randint(0, 41, (2, 5))
    output = generate_token_conditioned(
        model,
        prompt,
        3,
        policy="greedy",
    )
    canonical = generate_reanchored(
        model,
        prompt,
        1,
        policy="greedy",
    )
    assert output.tokens.shape == (2, 3)
    assert output.canonical_forward_kl.shape == (2, 3)
    assert output.canonical_top1_agreement.shape == (2, 3)
    assert output.canonical_max_logit_error.shape == (2, 3)
    torch.testing.assert_close(output.tokens[:, :1], canonical)
    torch.testing.assert_close(
        output.canonical_max_logit_error[:, 0],
        torch.zeros(2),
        atol=1e-6,
        rtol=0,
    )
    assert output.canonical_top1_agreement[:, 0].all()
    assert torch.isfinite(output.canonical_forward_kl).all()


def test_token_conditioned_sampling_uses_the_supplied_generator():
    model = make_model()
    prompt = torch.randint(0, 41, (2, 5))
    first = generate_token_conditioned(
        model,
        prompt,
        4,
        policy="sample",
        generator=torch.Generator().manual_seed(91),
    )
    second = generate_token_conditioned(
        model,
        prompt,
        4,
        policy="sample",
        generator=torch.Generator().manual_seed(91),
    )
    torch.testing.assert_close(first.tokens, second.tokens)
    torch.testing.assert_close(
        first.canonical_forward_kl,
        second.canonical_forward_kl,
    )


def test_continuation_metrics_separate_first_token_and_later_drift():
    generated = torch.tensor([1, 2, 2, 2, 3, 4, 1, 2])
    reference = torch.tensor([1, 2, 9, 2, 3, 0, 1, 8])
    metrics = continuation_metrics(generated, reference)
    assert metrics["first_token_accuracy"] == 1.0
    assert metrics["matching_prefix_tokens"] == 2.0
    assert metrics["reference_accuracy"] == 5 / 8
    assert metrics["longest_identical_run"] == 3.0
    assert metrics["collapsed"] == 0.0


def test_spectral_block_boundary_ar_matches_literal_self_feedback():
    model = make_model()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        32,
        bottleneck=8,
        alpha=0.0,
    )
    window = torch.randint(0, 41, (2, 7))
    encoded, _ = model.encode(window[:, :5])
    model.spectral_collapse.calibrate_shell(encoded)

    output = spectral_block_boundary_rollouts(
        model,
        window,
        prefix_length=5,
        horizons=2,
        anchor_stride=2,
    )
    assert output.boundary_ar_logits.shape == (2, 3, 2, 41)
    assert output.parallel_logits.shape == (2, 3, 2, 41)
    assert output.boundary_ar_decoded.shape == (2, 3, 2, 32)
    assert output.parallel_decoded.shape == (2, 3, 2, 32)
    assert output.boundary_ar_states.shape == (2, 3, 2, 32)
    assert output.boundary_ar_proposal_states.shape == (2, 3, 2, 32)
    assert output.parallel_states.shape == (2, 3, 2, 32)
    torch.testing.assert_close(
        output.boundary_ar_logits[:, :, 0],
        output.parallel_logits[:, :, 0],
        atol=1e-5,
        rtol=1e-5,
    )

    for batch in range(window.shape[0]):
        for slot, anchor in enumerate(output.anchor_indices.tolist()):
            rolling = window[batch : batch + 1, : anchor + 1]
            for horizon in range(2):
                positions = torch.arange(rolling.shape[1])
                prefix, expanded_positions = model.encode(
                    rolling,
                    positions,
                )
                proposal = model.spectral_collapse(
                    prefix[:, -1, None],
                    1,
                ).states
                decoded = model.exact_inverse_selected_decode_tape_states(
                    prefix,
                    proposal,
                    expanded_positions,
                    torch.tensor([rolling.shape[1] - 1]),
                )[:, 0, 0]
                literal_logits = model.token_logits(decoded).float()
                torch.testing.assert_close(
                    output.boundary_ar_logits[
                        batch,
                        slot,
                        horizon,
                    ],
                    literal_logits[0],
                    atol=2e-5,
                    rtol=1e-5,
                )
                action = literal_logits.argmax(dim=-1)
                rolling = torch.cat((rolling, action[:, None]), dim=1)
                canonical, _ = model.encode(rolling)
                torch.testing.assert_close(
                    output.boundary_ar_states[
                        batch,
                        slot,
                        horizon,
                    ],
                    canonical[0, -1],
                    atol=2e-5,
                    rtol=1e-5,
                )


def test_input_conditioned_composition_regenerates_k_from_each_output():
    model = make_model()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        32,
        bottleneck=8,
        alpha=0.0,
    )
    window = torch.randint(0, 41, (2, 7))
    encoded, _ = model.encode(window[:, :5])
    model.spectral_collapse.calibrate_shell(encoded)

    output = input_conditioned_spectral_composition_rollouts(
        model,
        window,
        prefix_length=5,
        horizons=2,
        anchor_stride=2,
    )
    assert output.composition_states.shape == (2, 3, 2, 32)
    assert output.boundary_ar_proposal_states.shape == (2, 3, 2, 32)
    assert output.composition_logits.shape == (2, 3, 2, 41)
    assert output.boundary_ar_decoded.shape == (2, 3, 2, 32)
    assert output.composition_decoded.shape == (2, 3, 2, 32)
    torch.testing.assert_close(
        output.composition_logits[:, :, 0],
        output.boundary_ar_logits[:, :, 0],
        atol=2e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        output.composition_states[:, :, 0],
        output.boundary_ar_proposal_states[:, :, 0],
        atol=2e-5,
        rtol=1e-5,
    )

    anchors = torch.tensor(output.anchor_indices.tolist())
    with torch.inference_mode():
        current = encoded.index_select(1, anchors)
        expected = []
        for _ in range(2):
            current = model.spectral_collapse(current, 1).states[:, :, 0]
            expected.append(current)
    torch.testing.assert_close(
        output.composition_states,
        torch.stack(expected, dim=2),
        atol=2e-5,
        rtol=1e-5,
    )


def test_input_conditioned_block_one_is_literal_greedy_ar():
    model = make_model()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        32,
        bottleneck=8,
        alpha=0.0,
    )
    prompt = torch.randint(0, 41, (2, 5))
    encoded, _ = model.encode(prompt)
    model.spectral_collapse.calibrate_shell(encoded)

    generated = generate_input_conditioned_composition_blocks(
        model,
        prompt,
        3,
        block=1,
        context_length=5,
    )
    rolling = prompt
    expected = []
    with torch.inference_mode():
        for _ in range(3):
            context = rolling[:, -5:]
            encoded, positions = model.encode(context)
            proposal = model.spectral_collapse(
                encoded[:, -1:], 1
            ).states
            decoded = model.exact_inverse_selected_decode_tape_states(
                encoded,
                proposal,
                positions,
                torch.tensor([context.shape[1] - 1]),
            )[:, 0, 0]
            action = model.token_logits(decoded).argmax(dim=-1)
            expected.append(action)
            rolling = torch.cat((rolling, action[:, None]), dim=1)
    torch.testing.assert_close(generated, torch.stack(expected, dim=1))


def test_attached_t_block_rollouts_share_h1_and_project_to_shell():
    model = make_model()
    model.spectral_collapse = InputConditionedSpectralCollapse(
        32,
        bottleneck=8,
        alpha=0.0,
    )
    model.spectral_corrector = AttachedSpectralStateCorrector(
        32,
        bottleneck=8,
    )
    window = torch.randint(0, 41, (2, 7))
    encoded, _ = model.encode(window[:, :5])
    model.spectral_collapse.calibrate_shell(encoded)
    output = attached_t_block_rollouts(
        model,
        window,
        prefix_length=5,
        horizons=2,
        anchor_stride=2,
    )
    assert output.ar_logits.shape == (2, 3, 2, 41)
    assert output.t_logits.shape == (2, 3, 2, 41)
    assert output.projected_t_logits.shape == (2, 3, 2, 41)
    torch.testing.assert_close(
        output.ar_logits[:, :, 0],
        output.t_logits[:, :, 0],
        atol=3e-5,
        rtol=2e-5,
    )
    assert float(output.projected_shell_error) < 2e-5
