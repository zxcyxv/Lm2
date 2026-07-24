import torch
import torch.nn.functional as F

from rotlm.models.query_k1_inverse import QueryK1InverseLM


def make_model(head_mode="cosine"):
    torch.manual_seed(47)
    return QueryK1InverseLM(
        43,
        width=32,
        encoder_blocks=1,
        head_mode=head_mode,
    )


def test_dense_query_is_the_last_row_of_each_independent_causal_prefix():
    model = make_model().eval()
    tokens = torch.randint(0, 43, (2, 6))
    prefix_states, query_states, positions = model.dense_query_encode(tokens)
    ordinary, _ = model.encode(tokens, positions)
    torch.testing.assert_close(prefix_states, ordinary, atol=2e-6, rtol=2e-6)

    for anchor in range(tokens.shape[1]):
        content = model.encoder.embed(tokens[:, : anchor + 1])
        query = model.query_embedding.view(1, 1, -1).expand(tokens.shape[0], 1, -1)
        tape = torch.cat((content, query), dim=1)
        tape_positions = torch.arange(anchor + 2).unsqueeze(0).expand(
            tokens.shape[0], -1
        )
        expected, _ = model.encoder.encode_hidden(tape, tape_positions)
        torch.testing.assert_close(
            query_states[:, anchor], expected[:, -1], atol=3e-5, rtol=3e-5
        )


def test_operator_input_is_contextual_query_not_last_token_state():
    model = make_model().eval()
    tokens = torch.randint(0, 43, (2, 6))
    output = model(tokens)
    expected = F.linear(output.query_states, model.operator.weight)
    wrong_contract = F.linear(output.prefix_states, model.operator.weight)
    torch.testing.assert_close(output.operator_states, expected)
    assert not torch.allclose(output.operator_states, wrong_contract)


def test_supplied_query_branch_is_exactly_inverted_per_anchor():
    model = make_model().eval()
    tokens = torch.randint(0, 43, (2, 6))
    output = model(tokens)
    for anchor in range(tokens.shape[1]):
        tape = torch.cat(
            (
                output.prefix_states[:, : anchor + 1],
                output.operator_states[:, anchor : anchor + 1],
            ),
            dim=1,
        )
        tape_positions = torch.arange(anchor + 2).unsqueeze(0).expand(
            tokens.shape[0], -1
        )
        expected = model.encoder.decode_hidden(tape, tape_positions)[:, -1]
        torch.testing.assert_close(
            output.inverse_hidden[:, anchor], expected, atol=3e-5, rtol=3e-5
        )


def test_dense_action_reinsertion_matches_literal_prefix_encoding():
    model = make_model().eval()
    tokens = torch.randint(0, 43, (2, 6))
    actions = torch.randint(0, 43, (2, 6))
    dense_states = model.dense_action_encode(tokens, actions)
    for anchor in range(tokens.shape[1]):
        tape = torch.cat(
            (tokens[:, : anchor + 1], actions[:, anchor : anchor + 1]), dim=1
        )
        positions = torch.arange(anchor + 2).unsqueeze(0).expand(
            tokens.shape[0], -1
        )
        expected, _ = model.encode(tape, positions)
        torch.testing.assert_close(
            dense_states[:, anchor], expected[:, -1], atol=3e-5, rtol=3e-5
        )


def test_epsilon_augmented_embedding_recovers_supplied_operator_state():
    model = make_model().eval()
    tokens = torch.randint(0, 43, (2, 6))
    output = model(tokens)
    reinsertion = model.dense_epsilon_reinsert(
        tokens, output.inverse_hidden, output.positions,
    )
    torch.testing.assert_close(
        reinsertion.continuous_embeddings,
        output.inverse_hidden,
        atol=2e-7,
        rtol=2e-7,
    )
    torch.testing.assert_close(
        reinsertion.restored_states,
        output.operator_states,
        atol=4e-5,
        rtol=4e-5,
    )

    token_only = model.dense_action_encode(
        tokens, reinsertion.actions, output.positions,
    )
    assert not torch.allclose(token_only, output.operator_states)


def test_continuous_query_step_matches_dense_last_anchor_and_closes_with_epsilon():
    model = make_model().eval()
    tokens = torch.randint(0, 43, (2, 6))
    dense = model(tokens)
    prefix_embeddings = model.encoder.embed(tokens)
    step = model.continuous_query_step(prefix_embeddings)
    torch.testing.assert_close(step.logits, dense.logits[:, -1], atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(
        step.operator_state, dense.operator_states[:, -1], atol=4e-5, rtol=4e-5
    )
    augmented = step.action_embedding + step.epsilon
    restored, _ = model.encoder.encode_hidden(
        torch.cat((prefix_embeddings, augmented[:, None]), dim=1),
        step.positions,
    )
    torch.testing.assert_close(
        restored[:, -1], step.operator_state, atol=4e-5, rtol=4e-5
    )


def test_query_orbit_is_shared_k_powers_and_epsilon_tape_roundtrips():
    model = make_model().eval()
    tokens = torch.randint(0, 43, (2, 6))
    output = model.forward_orbit(tokens, horizons=3)
    expected = output.query_states
    for horizon in range(3):
        expected = model.operator(expected)
        torch.testing.assert_close(output.orbit_states[:, :, horizon], expected)

    reinsertion = model.dense_epsilon_tape_reinsert(
        tokens, output.inverse_hidden, output.positions,
    )
    torch.testing.assert_close(
        reinsertion.continuous_embeddings,
        output.inverse_hidden,
        atol=2e-7,
        rtol=2e-7,
    )
    torch.testing.assert_close(
        reinsertion.restored_states,
        output.orbit_states,
        atol=5e-5,
        rtol=5e-5,
    )

    # A literal prefix and future tape must agree with the dense branch path.
    anchor = 3
    tape = torch.cat(
        (
            output.prefix_states[:, : anchor + 1],
            output.orbit_states[:, anchor],
        ),
        dim=1,
    )
    positions = torch.arange(anchor + 4).unsqueeze(0).expand(tokens.shape[0], -1)
    literal = model.encoder.decode_hidden(tape, positions)[:, -3:]
    torch.testing.assert_close(
        output.inverse_hidden[:, anchor], literal, atol=5e-5, rtol=5e-5
    )


def test_one_step_credit_matches_forward_and_cuts_earlier_orbit_gradients():
    model = make_model().train()
    tokens = torch.randint(0, 43, (2, 6))
    ordinary = model.forward_orbit(tokens, horizons=3)
    detached = model.forward_orbit_one_step_credit(tokens, horizons=3)
    torch.testing.assert_close(detached.logits, ordinary.logits, atol=6e-5, rtol=6e-5)
    torch.testing.assert_close(detached.orbit_states, ordinary.orbit_states)

    # Rebuild the third row from the graph's actual s1/s2/s3 nodes.  Slices of
    # QueryOrbitOutput.orbit_states are views of a later stack operation and
    # are not the nodes consumed by the decoder.
    prefix, query, positions = model.dense_query_encode(tokens)
    s1 = model.operator(query)
    s2 = model.operator(s1.detach())
    s3 = model.operator(s2.detach())
    third_branch = torch.stack((s1.detach(), s2.detach(), s3), dim=2)
    third_hidden = model.exact_inverse_dense_decode_tape_states(
        prefix, third_branch, positions,
    )[:, :, -1]
    third_loss = model.token_logits(third_hidden).square().mean()
    q_grad, s1_grad, s2_grad, s3_grad, k_grad = torch.autograd.grad(
        third_loss,
        (query, s1, s2, s3, model.operator.weight),
        allow_unused=True,
    )
    assert q_grad is None or q_grad.norm() == 0
    assert s1_grad is None
    assert s2_grad is None
    assert s3_grad is not None and s3_grad.norm() > 0
    assert k_grad is not None and k_grad.norm() > 0

    # The second-slot decoder likewise receives a detached s1 and live s2.
    prefix, query, positions = model.dense_query_encode(tokens)
    s1 = model.operator(query)
    s2 = model.operator(s1.detach())
    second_branch = torch.stack((s1.detach(), s2), dim=2)
    second_hidden = model.exact_inverse_dense_decode_tape_states(
        prefix, second_branch, positions,
    )[:, :, -1]
    second_loss = model.token_logits(second_hidden).square().mean()
    s1_grad, s2_grad, k_grad = torch.autograd.grad(
        second_loss, (s1, s2, model.operator.weight),
        allow_unused=True,
    )
    assert s1_grad is None
    assert s2_grad is not None and s2_grad.norm() > 0
    assert k_grad is not None and k_grad.norm() > 0


def test_pure_ce_reaches_query_operator_and_encoder_for_both_heads():
    tokens = torch.randint(0, 43, (2, 7))
    targets = torch.randint(0, 43, (2, 7))
    for head_mode in ("cosine", "rms-tied"):
        model = make_model(head_mode)
        output = model(tokens)
        loss = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten())
        loss.backward()
        for parameter in (
            model.query_embedding,
            model.operator.weight,
            model.encoder.blocks[0].attn.qkv.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.norm() > 0
