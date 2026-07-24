"""Dense teacher-forced K=1 model bootstrapped by contextual queries.

For every causal anchor ``t`` this model evaluates the conceptual tape

    [embedding(x_0), ..., embedding(x_t), learned QUERY]

through one shared reversible encoder.  The final contextualized query state
``q_(t+1)`` -- not the token state ``h_t`` -- is the sole input to the learned
linear operator.  The exact inverse decoder then reads

    [h_0, ..., h_t, K q_(t+1)]

and a tied token head predicts the gold next token.  All anchors are evaluated
dense in one forward pass while remaining causally independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .k1_decoder_ablation import HeadMode, K1DecoderAblationLM
from .prefix_orbit_lm import PrefixOrbitLM


@dataclass
class QueryK1InverseOutput:
    logits: torch.Tensor
    prefix_states: torch.Tensor
    query_states: torch.Tensor
    operator_states: torch.Tensor
    inverse_hidden: torch.Tensor
    positions: torch.Tensor


@dataclass
class EpsilonReinsertionOutput:
    """Hard token plus the exact continuous residual discarded by argmax.

    ``continuous_embedding`` is the exact-inverse decoder output ``y``.
    Splitting it as ``embedding(action) + epsilon`` preserves that continuous
    row exactly.  Re-encoding the augmented row behind the same causal prefix
    must therefore recover the supplied operator state up to numerical error.
    """

    actions: torch.Tensor
    token_embeddings: torch.Tensor
    epsilon: torch.Tensor
    continuous_embeddings: torch.Tensor
    restored_states: torch.Tensor


@dataclass
class ContinuousQueryStepOutput:
    """One literal query/K/inverse step from a continuous prefix tape."""

    prefix_states: torch.Tensor
    query_state: torch.Tensor
    operator_state: torch.Tensor
    decoder_embedding: torch.Tensor
    logits: torch.Tensor
    action: torch.Tensor
    action_embedding: torch.Tensor
    epsilon: torch.Tensor
    positions: torch.Tensor


@dataclass
class QueryOrbitOutput:
    """A dense all-anchor query bootstrap followed by shared K powers."""

    logits: torch.Tensor
    prefix_states: torch.Tensor
    query_states: torch.Tensor
    orbit_states: torch.Tensor
    inverse_hidden: torch.Tensor
    positions: torch.Tensor


class QueryK1InverseLM(K1DecoderAblationLM):
    """Exact-inverse K=1 LM whose transition starts from a blank query."""

    def __init__(
        self,
        vocabulary: int,
        *,
        width: int,
        encoder_blocks: int = 2,
        head_mode: HeadMode = "cosine",
        trainable_cosine_scale: bool = True,
    ) -> None:
        super().__init__(
            vocabulary,
            width=width,
            encoder_blocks=encoder_blocks,
            decoder_mode="exact-inverse",
            head_mode=head_mode,
            trainable_cosine_scale=trainable_cosine_scale,
        )
        self.query_embedding = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query_embedding, std=0.02)

    def _dense_append_hidden_encode(
        self,
        tokens: torch.Tensor,
        appended_hidden: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one supplied next-position row behind every causal prefix."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,length]")
        batch, length = tokens.shape
        if appended_hidden.shape != (batch, length, self.width):
            raise ValueError(
                "appended_hidden must have shape [batch,length,width], got "
                f"{tuple(appended_hidden.shape)}"
            )
        prefix_states, branch_states, positions = self._dense_append_hidden_tape_encode(
            tokens, appended_hidden.unsqueeze(2), positions,
        )
        return prefix_states, branch_states.squeeze(2), positions

    def _dense_append_hidden_tape_encode(
        self,
        tokens: torch.Tensor,
        appended_hidden: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a supplied causal future hidden tape per prefix anchor."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,length]")
        batch, length = tokens.shape
        if appended_hidden.ndim != 4 or appended_hidden.shape[:2] != (batch, length):
            raise ValueError(
                "appended_hidden must have shape [batch,length,horizon,width]"
            )
        if appended_hidden.shape[-1] != self.width:
            raise ValueError(
                f"appended hidden width {appended_hidden.shape[-1]} != {self.width}"
            )
        if positions is None:
            positions = torch.arange(length, device=tokens.device)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        if positions.shape != tokens.shape:
            raise ValueError(
                "positions must match token shape, got "
                f"{tuple(positions.shape)} and {tuple(tokens.shape)}"
            )

        prefix = self.encoder.embed(tokens)
        branch = appended_hidden.to(prefix.dtype)
        horizons = branch.shape[2]
        branch_positions = positions.unsqueeze(-1) + torch.arange(
            1,
            horizons + 1,
            device=positions.device,
            dtype=positions.dtype,
        ).view(1, 1, -1)

        prefix_first, prefix_second = prefix.chunk(2, dim=-1)
        branch_first, branch_second = branch.chunk(2, dim=-1)
        for block in self.encoder.blocks:
            attn_scale = getattr(block, "attn_scale", None)
            ffn_scale = getattr(block, "ffn_scale", None)
            if attn_scale is None:
                attn_scale = block.residual_scale
            if ffn_scale is None:
                ffn_scale = block.residual_scale
            prefix_attention, branch_attention = PrefixOrbitLM._branch_attention(
                block.attn,
                block.norm1(prefix_second),
                block.norm1(branch_second),
                positions,
                branch_positions,
            )
            next_prefix_first = prefix_first + attn_scale * prefix_attention
            next_branch_first = branch_first + attn_scale * branch_attention
            next_prefix_second = prefix_second + ffn_scale * block.ffn(
                block.norm2(next_prefix_first)
            )
            next_branch_second = branch_second + ffn_scale * block.ffn(
                block.norm2(next_branch_first)
            )
            prefix_first, prefix_second = next_prefix_first, next_prefix_second
            branch_first, branch_second = next_branch_first, next_branch_second

        prefix_states = torch.cat((prefix_first, prefix_second), dim=-1)
        branch_states = torch.cat((branch_first, branch_second), dim=-1)
        return prefix_states, branch_states, positions

    def dense_query_encode(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contextualize one next-position query behind every causal prefix.

        The returned ``query_states[:, t]`` is exactly the final row obtained
        by separately encoding ``[tokens[:, :t+1], QUERY]``.  Query branches
        for different anchors never attend to one another.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,length]")
        batch, length = tokens.shape
        query = self.query_embedding.view(1, 1, -1).expand(batch, length, -1)
        return self._dense_append_hidden_encode(tokens, query, positions)

    def dense_action_encode(
        self,
        prefix_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode each hard action after its corresponding literal prefix.

        ``action_tokens[:, t]`` is embedded and encoded behind
        ``prefix_tokens[:, :t+1]``.  This is the actual discrete reinsertion
        state used to audit action closure of ``K q_(t+1)``.
        """
        if action_tokens.shape != prefix_tokens.shape:
            raise ValueError(
                "action_tokens must match prefix_tokens shape, got "
                f"{tuple(action_tokens.shape)} and {tuple(prefix_tokens.shape)}"
            )
        action_hidden = self.encoder.embed(action_tokens)
        _, action_states, _ = self._dense_append_hidden_encode(
            prefix_tokens, action_hidden, positions,
        )
        return action_states

    def dense_action_tape_encode(
        self,
        prefix_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a causal hard-action tape behind every literal prefix.

        ``action_tokens`` must have shape ``[batch,anchor,horizon]``.  The
        returned slot states are the literal encoder states of
        ``[prefix[:t+1], action_tokens[t,:]]`` for every anchor ``t``.
        """
        if action_tokens.ndim != 3 or action_tokens.shape[:2] != prefix_tokens.shape:
            raise ValueError(
                "action_tokens must have shape [batch,anchor,horizon]"
            )
        action_hidden = self.encoder.embed(action_tokens)
        _, action_states, _ = self._dense_append_hidden_tape_encode(
            prefix_tokens, action_hidden, positions,
        )
        return action_states

    def dense_epsilon_reinsert(
        self,
        prefix_tokens: torch.Tensor,
        continuous_embeddings: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> EpsilonReinsertionOutput:
        """Reinsert ``argmax token embedding + discarded residual`` densely.

        This is deliberately distinct from :meth:`dense_action_encode`, which
        discards the residual and encodes the hard token alone.  Branches for
        different anchors remain causally independent.
        """
        if continuous_embeddings.shape != (
            prefix_tokens.shape[0], prefix_tokens.shape[1], self.width,
        ):
            raise ValueError(
                "continuous_embeddings must have shape [batch,length,width]"
            )
        actions = self.token_logits(continuous_embeddings).argmax(dim=-1)
        token_embeddings = self.encoder.embed(actions)
        epsilon = continuous_embeddings - token_embeddings
        augmented = token_embeddings + epsilon
        _, restored_states, _ = self._dense_append_hidden_encode(
            prefix_tokens, augmented, positions,
        )
        return EpsilonReinsertionOutput(
            actions=actions,
            token_embeddings=token_embeddings,
            epsilon=epsilon,
            continuous_embeddings=augmented,
            restored_states=restored_states,
        )

    def dense_epsilon_tape_reinsert(
        self,
        prefix_tokens: torch.Tensor,
        continuous_embeddings: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> EpsilonReinsertionOutput:
        """Reinsert a causal future tape as hard embeddings plus epsilon."""
        if continuous_embeddings.ndim != 4 or continuous_embeddings.shape[:2] != (
            prefix_tokens.shape[0], prefix_tokens.shape[1]
        ) or continuous_embeddings.shape[-1] != self.width:
            raise ValueError(
                "continuous_embeddings must have shape "
                "[batch,anchor,horizon,width]"
            )
        actions = self.token_logits(continuous_embeddings).argmax(dim=-1)
        token_embeddings = self.encoder.embed(actions)
        epsilon = continuous_embeddings - token_embeddings
        augmented = token_embeddings + epsilon
        _, restored_states, _ = self._dense_append_hidden_tape_encode(
            prefix_tokens, augmented, positions,
        )
        return EpsilonReinsertionOutput(
            actions=actions,
            token_embeddings=token_embeddings,
            epsilon=epsilon,
            continuous_embeddings=augmented,
            restored_states=restored_states,
        )

    def forward_orbit(
        self,
        tokens: torch.Tensor,
        *,
        horizons: int,
        positions: torch.Tensor | None = None,
    ) -> QueryOrbitOutput:
        """Decode ``Kq, K^2q, ...`` jointly behind every causal prefix."""
        steps = int(horizons)
        if steps < 1:
            raise ValueError("horizons must be positive")
        prefix_states, query_states, positions = self.dense_query_encode(
            tokens, positions,
        )
        state = query_states
        orbit = []
        for _ in range(steps):
            state = self.operator(state)
            orbit.append(state)
        orbit_states = torch.stack(orbit, dim=2)
        inverse_hidden = self.exact_inverse_dense_decode_tape_states(
            prefix_states, orbit_states, positions,
        )
        return QueryOrbitOutput(
            logits=self.token_logits(inverse_hidden),
            prefix_states=prefix_states,
            query_states=query_states,
            orbit_states=orbit_states,
            inverse_hidden=inverse_hidden,
            positions=positions,
        )

    def forward_orbit_one_step_credit(
        self,
        tokens: torch.Tensor,
        *,
        horizons: int,
        positions: torch.Tensor | None = None,
    ) -> QueryOrbitOutput:
        """Use inference-identical K powers with one-step semi-gradients.

        Each transition consumes a detached preceding state.  In addition,
        the loss-bearing decoder tape for horizon ``j`` receives detached
        earlier orbit rows, preventing causal attention from providing an
        alternate gradient path into those rows.  Forward values remain equal
        to :meth:`forward_orbit`; only the backward graph differs.
        """
        steps = int(horizons)
        if steps < 1:
            raise ValueError("horizons must be positive")
        prefix_states, query_states, positions = self.dense_query_encode(
            tokens, positions,
        )
        state = query_states
        orbit = []
        for step in range(steps):
            state = self.operator(state if step == 0 else state.detach())
            orbit.append(state)

        inverse_rows = []
        for horizon in range(steps):
            branch = torch.stack(
                [
                    state if index == horizon else state.detach()
                    for index, state in enumerate(orbit[: horizon + 1])
                ],
                dim=2,
            )
            decoded = self.exact_inverse_dense_decode_tape_states(
                prefix_states, branch, positions,
            )
            inverse_rows.append(decoded[:, :, -1])
        inverse_hidden = torch.stack(inverse_rows, dim=2)
        return QueryOrbitOutput(
            logits=self.token_logits(inverse_hidden),
            prefix_states=prefix_states,
            query_states=query_states,
            orbit_states=torch.stack(orbit, dim=2),
            inverse_hidden=inverse_hidden,
            positions=positions,
        )

    def continuous_query_step(
        self,
        prefix_embeddings: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> ContinuousQueryStepOutput:
        """Run one greedy step while preserving a continuous prefix.

        ``prefix_embeddings`` may contain ordinary token embeddings or rows
        retained from previous exact-inverse outputs.  The next-position query
        is encoded causally, transformed by K, and exactly inverted behind the
        same prefix.  The returned ``decoder_embedding`` can either be kept
        intact (epsilon path) or replaced by ``action_embedding`` (hard-token
        path) before the next call.
        """
        if prefix_embeddings.ndim != 3 or prefix_embeddings.shape[-1] != self.width:
            raise ValueError(
                "prefix_embeddings must have shape [batch,length,width]"
            )
        batch, length, _ = prefix_embeddings.shape
        if positions is None:
            positions = torch.arange(
                length + 1, device=prefix_embeddings.device,
            ).unsqueeze(0).expand(batch, -1)
        elif positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        if positions.shape != (batch, length + 1):
            raise ValueError("positions must cover prefix plus one future row")
        query = self.query_embedding.to(prefix_embeddings.dtype).view(
            1, 1, -1,
        ).expand(batch, 1, -1)
        query_tape = torch.cat((prefix_embeddings, query), dim=1)
        encoded, encoded_positions = self.encoder.encode_hidden(
            query_tape, positions,
        )
        prefix_states, query_state = encoded[:, :-1], encoded[:, -1]
        operator_state = self.operator(query_state)
        latent_tape = torch.cat((prefix_states, operator_state[:, None]), dim=1)
        decoded = self.encoder.decode_hidden(latent_tape, encoded_positions)
        decoder_embedding = decoded[:, -1]
        logits = self.token_logits(decoder_embedding)
        action = logits.argmax(dim=-1)
        action_embedding = self.encoder.embed(action)
        epsilon = decoder_embedding - action_embedding
        return ContinuousQueryStepOutput(
            prefix_states=prefix_states,
            query_state=query_state,
            operator_state=operator_state,
            decoder_embedding=decoder_embedding,
            logits=logits,
            action=action,
            action_embedding=action_embedding,
            epsilon=epsilon,
            positions=encoded_positions,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> QueryK1InverseOutput:
        prefix_states, query_states, positions = self.dense_query_encode(
            tokens, positions,
        )
        operator_states = self.operator(query_states)
        inverse_hidden = self.exact_inverse_dense_decode_states(
            prefix_states, operator_states, positions,
        )
        return QueryK1InverseOutput(
            logits=self.token_logits(inverse_hidden),
            prefix_states=prefix_states,
            query_states=query_states,
            operator_states=operator_states,
            inverse_hidden=inverse_hidden,
            positions=positions,
        )


__all__ = [
    "ContinuousQueryStepOutput",
    "EpsilonReinsertionOutput",
    "QueryOrbitOutput",
    "QueryK1InverseLM",
    "QueryK1InverseOutput",
]
