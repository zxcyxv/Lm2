import torch
import torch.nn.functional as F

from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM


def make_model(decoder_mode="exact-inverse", head_mode="cosine"):
    torch.manual_seed(31)
    return K1DecoderAblationLM(
        41,
        width=32,
        encoder_blocks=1,
        decoder_mode=decoder_mode,
        independent_decoder_blocks=1,
        head_mode=head_mode,
    )


def test_dense_action_encode_matches_literal_prefix_reencoding():
    model = make_model().eval()
    tokens = torch.randint(0, 41, (2, 6))
    actions = torch.randint(0, 41, (2, 6))
    dense = model.dense_action_encode(tokens, actions)
    for anchor in range(tokens.shape[1]):
        tape = torch.cat(
            (tokens[:, : anchor + 1], actions[:, anchor : anchor + 1]),
            dim=1,
        )
        expected, _ = model.encode(tape)
        torch.testing.assert_close(
            dense[:, anchor],
            expected[:, -1],
            atol=3e-5,
            rtol=3e-5,
        )


def test_dense_action_tape_encode_matches_literal_prefix_reencoding():
    model = make_model().eval()
    tokens = torch.randint(0, 41, (2, 6))
    actions = torch.randint(0, 41, (2, 6, 3))
    dense = model.dense_action_tape_encode(tokens, actions)
    for anchor in range(tokens.shape[1]):
        tape = torch.cat(
            (tokens[:, : anchor + 1], actions[:, anchor]),
            dim=1,
        )
        expected, _ = model.encode(tape)
        torch.testing.assert_close(
            dense[:, anchor],
            expected[:, -actions.shape[2] :],
            atol=4e-5,
            rtol=4e-5,
        )


def test_exact_dense_path_matches_independent_prefix_tapes():
    model = make_model().eval()
    tokens = torch.randint(0, 41, (2, 7))
    encoded, positions = model.encode(tokens)
    dense_hidden, dense_states = model.dense_decode_hidden(encoded, positions)
    for anchor in range(tokens.shape[1]):
        state = model.operator(encoded[:, anchor])
        tape = torch.cat((encoded[:, : anchor + 1], state[:, None]), dim=1)
        tape_positions = torch.arange(anchor + 2).unsqueeze(0).expand(
            tokens.shape[0], -1
        )
        expected = model.encoder.decode_hidden(tape, tape_positions)[:, -1]
        torch.testing.assert_close(dense_states[:, anchor], state)
        torch.testing.assert_close(
            dense_hidden[:, anchor], expected, atol=3e-5, rtol=3e-5
        )


def test_selected_joint_tapes_match_individual_prefix_decodes():
    model = make_model().eval()
    tokens = torch.randint(0, 41, (2, 8))
    encoded, positions = model.encode(tokens)
    anchor_indices = torch.tensor([0, 3, 6], dtype=torch.long)
    anchor_states = encoded.index_select(1, anchor_indices)
    chain = []
    current = anchor_states
    for _ in range(3):
        current = model.operator(current)
        chain.append(current)
    branch_states = torch.stack(chain, dim=2)
    selected = model.exact_inverse_selected_decode_tape_states(
        encoded,
        branch_states,
        positions,
        anchor_indices,
    )
    for selected_index, anchor in enumerate(anchor_indices.tolist()):
        tape = torch.cat(
            (
                encoded[:, : anchor + 1],
                branch_states[:, selected_index],
            ),
            dim=1,
        )
        tape_positions = torch.arange(anchor + 4).unsqueeze(0).expand(
            tokens.shape[0], -1
        )
        expected = model.encoder.decode_hidden(
            tape, tape_positions
        )[:, -3:]
        torch.testing.assert_close(
            selected[:, selected_index],
            expected,
            atol=4e-5,
            rtol=4e-5,
        )


def test_selected_tape_can_detach_past_branch_state_gradients():
    model = make_model().eval()
    tokens = torch.randint(0, 41, (2, 7))
    encoded, positions = model.encode(tokens)
    encoded = encoded.detach()
    anchor_indices = torch.tensor([2, 5], dtype=torch.long)
    branch_states = torch.randn(
        2,
        anchor_indices.numel(),
        3,
        model.width,
        requires_grad=True,
    )

    full = model.exact_inverse_selected_decode_tape_states(
        encoded,
        branch_states,
        positions,
        anchor_indices,
    )
    truncated = model.exact_inverse_selected_decode_tape_states(
        encoded,
        branch_states,
        positions,
        anchor_indices,
        detach_past_branch_states=True,
    )
    torch.testing.assert_close(truncated, full, atol=4e-5, rtol=4e-5)

    gradient = torch.autograd.grad(
        truncated[:, :, 1].square().sum(),
        branch_states,
    )[0]
    assert torch.count_nonzero(gradient[:, :, 0]) == 0
    assert gradient[:, :, 1].norm() > 0
    assert torch.count_nonzero(gradient[:, :, 2]) == 0


def test_independent_decoder_does_not_share_encoder_parameters():
    model = make_model(decoder_mode="independent")
    encoder_ids = {id(parameter) for parameter in model.encoder.blocks.parameters()}
    decoder_ids = {id(parameter) for parameter in model.decoder_blocks.parameters()}
    assert encoder_ids
    assert decoder_ids
    assert encoder_ids.isdisjoint(decoder_ids)


def test_all_variants_send_ce_gradient_to_operator():
    tokens = torch.randint(0, 41, (2, 8))
    targets = torch.randint(0, 41, (2, 8))
    for decoder_mode, head_mode in (
        ("exact-inverse", "cosine"),
        ("independent", "cosine"),
        ("exact-inverse", "rms-tied"),
    ):
        model = make_model(decoder_mode, head_mode)
        logits, _, _ = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        gradient = model.operator.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0


def test_simplex_tied_head_is_fixed_equal_norm_nearest_codebook():
    model = K1DecoderAblationLM(
        17,
        width=32,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="simplex-tied",
        simplex_logit_scale=12.0,
    )
    codebook = model.embedding_weight.detach()
    gram = codebook @ codebook.T
    expected = torch.full((17, 17), -1.0 / 16)
    expected.fill_diagonal_(1.0)
    torch.testing.assert_close(gram, expected, atol=2e-6, rtol=2e-6)
    assert not model.embedding_weight.requires_grad
    assert not model.encoder.head_norm.weight.requires_grad
    assert model.token_logits(codebook).argmax(dim=-1).equal(
        torch.arange(17)
    )

    query = F.normalize(torch.randn(5, 32), dim=-1)
    dot_choice = model.token_logits(query).argmax(dim=-1)
    distance_choice = torch.cdist(query, codebook).argmin(dim=-1)
    assert torch.equal(dot_choice, distance_choice)


def test_simplex_raw_tied_head_has_no_hidden_normalization():
    model = K1DecoderAblationLM(
        17,
        width=32,
        encoder_blocks=1,
        decoder_mode="exact-inverse",
        head_mode="simplex-raw-tied",
        simplex_logit_scale=12.0,
    )
    codebook = model.embedding_weight.detach()
    assert model.token_logits(codebook).argmax(dim=-1).equal(
        torch.arange(17)
    )
    torch.testing.assert_close(
        model.token_logits(2.0 * codebook),
        2.0 * model.token_logits(codebook),
    )
