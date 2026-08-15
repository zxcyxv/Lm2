"""K=1 teacher-forced ablations for the decoder inverse constraint.

Every variant implements the same conditional prediction contract for anchor
``t``::

    Z_<=t = encoder(token_embeddings(x_<=t))
    u_t   = K Z_t
    y_t   = decoder([Z_<=t, u_t])[-1]
    p(x_{t+1} | x_<=t) = token_head(y_t)

The exact variant uses the analytic inverse of the encoder as ``decoder``.
The independent variant uses a separately parameterized causal reversible
stack in its ordinary forward direction. Both dense paths are equivalent to
evaluating one independent prefix tape per anchor; branches from different
anchors never attend to one another.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from shift_lm import RevBlock

from .cosine_codebook import CosineCodebookHead
from .prefix_orbit_lm import PrefixOrbitLM


DecoderMode = Literal["exact-inverse", "independent"]
HeadMode = Literal[
    "cosine",
    "rms-tied",
    "simplex-tied",
    "simplex-raw-tied",
]


class K1DecoderAblationLM(nn.Module):
    """One-step latent-transition LM with interchangeable decoder contracts."""

    def __init__(
        self,
        vocabulary: int,
        *,
        width: int,
        encoder_blocks: int = 2,
        decoder_mode: DecoderMode = "exact-inverse",
        independent_decoder_blocks: int = 2,
        head_mode: HeadMode = "cosine",
        trainable_cosine_scale: bool = True,
        simplex_logit_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 16:
            raise ValueError("width must be positive and divisible by 16")
        if encoder_blocks < 1:
            raise ValueError("encoder_blocks must be positive")
        if decoder_mode not in ("exact-inverse", "independent"):
            raise ValueError(f"unknown decoder_mode: {decoder_mode}")
        if head_mode not in (
            "cosine",
            "rms-tied",
            "simplex-tied",
            "simplex-raw-tied",
        ):
            raise ValueError(f"unknown head_mode: {head_mode}")
        if simplex_logit_scale <= 0:
            raise ValueError("simplex logit scale must be positive")
        if head_mode in ("simplex-tied", "simplex-raw-tied") and (
            width < vocabulary
        ):
            raise ValueError(
                "simplex-tied head requires width >= vocabulary"
            )
        if decoder_mode == "independent" and independent_decoder_blocks < 1:
            raise ValueError("independent_decoder_blocks must be positive")

        self.vocabulary = int(vocabulary)
        self.width = int(width)
        self.decoder_mode = decoder_mode
        self.head_mode = head_mode
        self.encoder = PrefixOrbitLM(
            vocabulary,
            d=width,
            n_blocks=encoder_blocks,
            latent_operator="geometric",
        )
        self.operator = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.operator.weight.copy_(torch.eye(width))

        if decoder_mode == "independent":
            self.decoder_blocks = nn.ModuleList(
                RevBlock(width // 2) for _ in range(independent_decoder_blocks)
            )
        else:
            self.decoder_blocks = nn.ModuleList()

        if head_mode == "cosine":
            self.codebook_head = CosineCodebookHead.from_embedding(
                self.encoder.embed.weight,
                trainable_scale=trainable_cosine_scale,
            )
            # Registered for checkpoint compatibility, absent from this head.
            self.encoder.head_norm.requires_grad_(False)
        else:
            self.codebook_head = None
            if head_mode in ("simplex-tied", "simplex-raw-tied"):
                # V equal-norm vertices with pairwise inner product
                # -1/(V-1), embedded in the first V coordinates. This makes
                # normalized dot-product argmax a nearest-code decision.
                identity = torch.eye(vocabulary)
                simplex = identity - torch.full(
                    (vocabulary, vocabulary),
                    1.0 / vocabulary,
                )
                simplex *= (vocabulary / (vocabulary - 1)) ** 0.5
                with torch.no_grad():
                    self.encoder.embed.weight.zero_()
                    self.encoder.embed.weight[:, :vocabulary].copy_(
                        simplex
                    )
                self.encoder.embed.weight.requires_grad_(False)
                self.encoder.head_norm.requires_grad_(False)
                self.register_buffer(
                    "simplex_logit_scale_value",
                    torch.tensor(float(simplex_logit_scale)),
                )

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.encoder.embed.weight

    def encode(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch,length]")
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.encoder.encode(tokens, positions)

    def dense_action_encode(
        self,
        prefix_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one hard action behind every literal causal prefix.

        ``action_tokens[:, t]`` is contextualized as the next token after
        ``prefix_tokens[:, :t+1]``. Branches for different anchors are
        independent, so the result is exactly the stack of literal
        ``encode([prefix[:t+1], action[t]])[:, -1]`` calls.
        """
        if prefix_tokens.ndim != 2:
            raise ValueError("prefix_tokens must have shape [batch,length]")
        if action_tokens.shape != prefix_tokens.shape:
            raise ValueError(
                "action_tokens must match prefix_tokens shape, got "
                f"{tuple(action_tokens.shape)} and {tuple(prefix_tokens.shape)}"
            )
        return self.dense_action_tape_encode(
            prefix_tokens,
            action_tokens.unsqueeze(-1),
            positions,
        ).squeeze(2)

    def dense_action_tape_encode(
        self,
        prefix_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one causal hard-action tape behind every literal prefix.

        ``action_tokens[:, t, :j+1]`` is contextualized behind
        ``prefix_tokens[:, :t+1]``. The returned state at ``[b,t,j]`` is
        exactly the last encoder state of that literal concatenated tape.
        Different anchors remain causally independent.
        """
        if prefix_tokens.ndim != 2:
            raise ValueError("prefix_tokens must have shape [batch,length]")
        if action_tokens.ndim != 3 or (
            action_tokens.shape[:2] != prefix_tokens.shape
        ):
            raise ValueError(
                "action_tokens must have shape [batch,length,horizon]"
            )
        batch, length = prefix_tokens.shape
        if positions is None:
            positions = torch.arange(length, device=prefix_tokens.device)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        if positions.shape != prefix_tokens.shape:
            raise ValueError("positions must match prefix token shape")

        prefix = self.encoder.embed(prefix_tokens)
        branch = self.encoder.embed(action_tokens)
        horizons = action_tokens.shape[2]
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
            next_prefix_first = (
                prefix_first + attn_scale * prefix_attention
            )
            next_branch_first = (
                branch_first + attn_scale * branch_attention
            )
            next_prefix_second = prefix_second + ffn_scale * block.ffn(
                block.norm2(next_prefix_first)
            )
            next_branch_second = branch_second + ffn_scale * block.ffn(
                block.norm2(next_branch_first)
            )
            prefix_first, prefix_second = (
                next_prefix_first,
                next_prefix_second,
            )
            branch_first, branch_second = (
                next_branch_first,
                next_branch_second,
            )
        return torch.cat((branch_first, branch_second), dim=-1)

    @staticmethod
    def _positions_for_branches(
        prefix_latent: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(prefix_latent.shape[0], -1)
        return positions, positions.unsqueeze(-1) + 1

    def _exact_inverse_dense_decode(
        self,
        prefix_latent: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operator_states = self.operator(prefix_latent)
        hidden = self.exact_inverse_dense_decode_states(
            prefix_latent, operator_states, positions,
        )
        return hidden, operator_states

    def exact_inverse_dense_decode_states(
        self,
        prefix_latent: torch.Tensor,
        branch_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Invert one supplied next-position latent state per causal anchor.

        ``branch_states[:, t]`` is decoded behind ``prefix_latent[:, :t+1]``.
        Keeping this primitive separate from ``self.operator`` lets query-based
        models feed ``K q_(t+1)`` without silently falling back to ``K h_t``.
        """
        if branch_states.shape != prefix_latent.shape:
            raise ValueError(
                "branch_states must match prefix_latent shape, got "
                f"{tuple(branch_states.shape)} and {tuple(prefix_latent.shape)}"
            )
        hidden = self.exact_inverse_dense_decode_tape_states(
            prefix_latent, branch_states.unsqueeze(2), positions,
        )
        return hidden.squeeze(2)

    def exact_inverse_dense_decode_tape_states(
        self,
        prefix_latent: torch.Tensor,
        branch_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Invert a supplied causal future tape behind every prefix anchor.

        ``branch_states`` has shape ``[batch,anchor,horizon,width]``.  Slot
        ``j`` sees the corresponding literal prefix and branch slots through
        ``j`` only.  Different anchors remain independent.
        """
        if branch_states.ndim != 4:
            raise ValueError(
                "branch_states must have shape [batch,anchor,horizon,width]"
            )
        if branch_states.shape[:2] != prefix_latent.shape[:2] or (
            branch_states.shape[-1] != prefix_latent.shape[-1]
        ):
            raise ValueError(
                "branch_states batch/anchor/width must match prefix_latent, got "
                f"{tuple(branch_states.shape)} and {tuple(prefix_latent.shape)}"
            )
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(prefix_latent.shape[0], -1)
        horizons = branch_states.shape[2]
        branch_positions = positions.unsqueeze(-1) + torch.arange(
            1,
            horizons + 1,
            device=positions.device,
            dtype=positions.dtype,
        ).view(1, 1, -1)
        prefix_first, prefix_second = prefix_latent.chunk(2, dim=-1)
        branch_first, branch_second = branch_states.chunk(2, dim=-1)
        for block in reversed(self.encoder.blocks):
            attn_scale = getattr(block, "attn_scale", None)
            ffn_scale = getattr(block, "ffn_scale", None)
            if attn_scale is None:
                attn_scale = block.residual_scale
            if ffn_scale is None:
                ffn_scale = block.residual_scale
            next_prefix_second = prefix_second - ffn_scale * block.ffn(
                block.norm2(prefix_first)
            )
            next_branch_second = branch_second - ffn_scale * block.ffn(
                block.norm2(branch_first)
            )
            prefix_attention, branch_attention = PrefixOrbitLM._branch_attention(
                block.attn,
                block.norm1(next_prefix_second),
                block.norm1(next_branch_second),
                positions,
                branch_positions,
            )
            next_prefix_first = prefix_first - attn_scale * prefix_attention
            next_branch_first = branch_first - attn_scale * branch_attention
            prefix_first, prefix_second = next_prefix_first, next_prefix_second
            branch_first, branch_second = next_branch_first, next_branch_second
        return torch.cat((branch_first, branch_second), dim=-1)

    def exact_inverse_selected_decode_tape_states(
        self,
        prefix_latent: torch.Tensor,
        branch_states: torch.Tensor,
        positions: torch.Tensor,
        anchor_indices: torch.Tensor,
        *,
        detach_past_branch_states: bool = False,
    ) -> torch.Tensor:
        """Invert joint future tapes for selected causal prefix anchors.

        ``branch_states`` has shape ``[batch,anchors,horizon,width]``.
        Branch ``a`` is appended behind the literal prefix ending at
        ``anchor_indices[a]``. Within a branch, future slot ``j`` can attend
        to slots through ``j`` but cannot see later future slots.
        """
        if branch_states.ndim != 4:
            raise ValueError(
                "branch_states must have shape "
                "[batch,anchors,horizon,width]"
            )
        if (
            branch_states.shape[0] != prefix_latent.shape[0]
            or branch_states.shape[-1] != prefix_latent.shape[-1]
        ):
            raise ValueError(
                "branch_states batch/width must match prefix_latent"
            )
        if anchor_indices.ndim != 1 or (
            anchor_indices.numel() != branch_states.shape[1]
        ):
            raise ValueError(
                "anchor_indices must have one entry per branch anchor"
            )
        if anchor_indices.dtype != torch.long:
            raise ValueError("anchor_indices must have dtype torch.long")
        if anchor_indices.numel() and (
            int(anchor_indices.min()) < 0
            or int(anchor_indices.max()) >= prefix_latent.shape[1]
        ):
            raise ValueError("anchor index outside prefix sequence")
        if anchor_indices.numel() > 1 and not bool(
            torch.all(anchor_indices[1:] > anchor_indices[:-1])
        ):
            raise ValueError("anchor_indices must be strictly increasing")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(
                prefix_latent.shape[0], -1
            )
        if positions.shape != prefix_latent.shape[:2]:
            raise ValueError("positions must match prefix batch and length")

        horizons = branch_states.shape[2]
        anchor_positions = positions.index_select(1, anchor_indices)
        branch_positions = anchor_positions.unsqueeze(-1) + torch.arange(
            1,
            horizons + 1,
            device=positions.device,
            dtype=positions.dtype,
        ).view(1, 1, -1)
        prefix_first, prefix_second = prefix_latent.chunk(2, dim=-1)
        branch_first, branch_second = branch_states.chunk(2, dim=-1)
        for block in reversed(self.encoder.blocks):
            attn_scale = getattr(block, "attn_scale", None)
            ffn_scale = getattr(block, "ffn_scale", None)
            if attn_scale is None:
                attn_scale = block.residual_scale
            if ffn_scale is None:
                ffn_scale = block.residual_scale
            next_prefix_second = prefix_second - ffn_scale * block.ffn(
                block.norm2(prefix_first)
            )
            next_branch_second = branch_second - ffn_scale * block.ffn(
                block.norm2(branch_first)
            )
            prefix_attention, branch_attention = (
                PrefixOrbitLM._selected_branch_attention(
                    block.attn,
                    block.norm1(next_prefix_second),
                    block.norm1(next_branch_second),
                    positions,
                    branch_positions,
                    anchor_indices,
                    detach_past_branch_states=(
                        detach_past_branch_states
                    ),
                )
            )
            next_prefix_first = (
                prefix_first - attn_scale * prefix_attention
            )
            next_branch_first = (
                branch_first - attn_scale * branch_attention
            )
            prefix_first, prefix_second = (
                next_prefix_first,
                next_prefix_second,
            )
            branch_first, branch_second = (
                next_branch_first,
                next_branch_second,
            )
        return torch.cat((branch_first, branch_second), dim=-1)

    def exact_inverse_selected_decode_multi_trajectory_states(
        self,
        prefix_latent: torch.Tensor,
        branch_states: torch.Tensor,
        positions: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Invert several future tapes while sharing their real prefix.

        ``branch_states`` has shape
        ``[batch,trajectories,anchors,horizons,width]``.  This is numerically
        equivalent to copying each prefix once per trajectory and calling
        :meth:`exact_inverse_selected_decode_tape_states`, but deterministic
        prefix FFN/attention/QKV work is executed only once.
        """
        if branch_states.ndim != 5:
            raise ValueError(
                "branch_states must have shape "
                "[batch,trajectories,anchors,horizons,width]"
            )
        if (
            branch_states.shape[0] != prefix_latent.shape[0]
            or branch_states.shape[-1] != prefix_latent.shape[-1]
        ):
            raise ValueError(
                "branch_states batch/width must match prefix_latent"
            )
        anchors = branch_states.shape[2]
        if anchor_indices.shape != (anchors,):
            raise ValueError(
                "anchor_indices must have one entry per branch anchor"
            )
        if anchor_indices.dtype != torch.long:
            raise ValueError("anchor_indices must have dtype torch.long")
        if anchor_indices.numel() and (
            int(anchor_indices.min()) < 0
            or int(anchor_indices.max()) >= prefix_latent.shape[1]
        ):
            raise ValueError("anchor index outside prefix sequence")
        if anchor_indices.numel() > 1 and not bool(
            torch.all(anchor_indices[1:] > anchor_indices[:-1])
        ):
            raise ValueError("anchor_indices must be strictly increasing")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(
                prefix_latent.shape[0], -1
            )
        if positions.shape != prefix_latent.shape[:2]:
            raise ValueError("positions must match prefix batch and length")

        horizons = branch_states.shape[3]
        anchor_positions = positions.index_select(1, anchor_indices)
        branch_positions = anchor_positions.unsqueeze(-1) + torch.arange(
            1,
            horizons + 1,
            device=positions.device,
            dtype=positions.dtype,
        ).view(1, 1, -1)
        prefix_first, prefix_second = prefix_latent.chunk(2, dim=-1)
        branch_first, branch_second = branch_states.chunk(2, dim=-1)
        for block in reversed(self.encoder.blocks):
            attn_scale = getattr(block, "attn_scale", None)
            ffn_scale = getattr(block, "ffn_scale", None)
            if attn_scale is None:
                attn_scale = block.residual_scale
            if ffn_scale is None:
                ffn_scale = block.residual_scale
            next_prefix_second = prefix_second - ffn_scale * block.ffn(
                block.norm2(prefix_first)
            )
            next_branch_second = branch_second - ffn_scale * block.ffn(
                block.norm2(branch_first)
            )
            prefix_attention, branch_attention = (
                PrefixOrbitLM._selected_multi_trajectory_attention(
                    block.attn,
                    block.norm1(next_prefix_second),
                    block.norm1(next_branch_second),
                    positions,
                    branch_positions,
                    anchor_indices,
                )
            )
            next_prefix_first = (
                prefix_first - attn_scale * prefix_attention
            )
            next_branch_first = (
                branch_first - attn_scale * branch_attention
            )
            prefix_first, prefix_second = (
                next_prefix_first,
                next_prefix_second,
            )
            branch_first, branch_second = (
                next_branch_first,
                next_branch_second,
            )
        return torch.cat((branch_first, branch_second), dim=-1)

    def _independent_dense_decode(
        self,
        prefix_latent: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate independent ``[Z_<=t, K Z_t]`` decoder tapes densely."""
        positions, branch_positions = self._positions_for_branches(
            prefix_latent, positions
        )
        operator_states = self.operator(prefix_latent)
        prefix_first, prefix_second = prefix_latent.chunk(2, dim=-1)
        branch_first, branch_second = operator_states.unsqueeze(2).chunk(2, dim=-1)
        for block in self.decoder_blocks:
            prefix_attention, branch_attention = PrefixOrbitLM._branch_attention(
                block.attn,
                block.norm1(prefix_second),
                block.norm1(branch_second),
                positions,
                branch_positions,
            )
            next_prefix_first = prefix_first + prefix_attention
            next_branch_first = branch_first + branch_attention
            next_prefix_second = prefix_second + block.ffn(
                block.norm2(next_prefix_first)
            )
            next_branch_second = branch_second + block.ffn(
                block.norm2(next_branch_first)
            )
            prefix_first, prefix_second = next_prefix_first, next_prefix_second
            branch_first, branch_second = next_branch_first, next_branch_second
        hidden = torch.cat((branch_first, branch_second), dim=-1).squeeze(2)
        return hidden, operator_states

    def dense_decode_hidden(
        self,
        prefix_latent: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one decoder output and one ``K h_t`` state per anchor."""
        if self.decoder_mode == "exact-inverse":
            return self._exact_inverse_dense_decode(prefix_latent, positions)
        return self._independent_dense_decode(prefix_latent, positions)

    def token_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.codebook_head is not None:
            return self.codebook_head(hidden, self.embedding_weight)
        if self.head_mode == "simplex-tied":
            query = F.normalize(hidden.float(), dim=-1, eps=1e-6)
            return self.simplex_logit_scale_value * (
                query @ self.embedding_weight.float().T
            )
        if self.head_mode == "simplex-raw-tied":
            return self.simplex_logit_scale_value * (
                hidden.float() @ self.embedding_weight.float().T
            )
        return self.encoder.head_norm(hidden) @ self.embedding_weight.T

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, positions = self.encode(tokens, positions)
        decoded, operator_states = self.dense_decode_hidden(encoded, positions)
        return self.token_logits(decoded), operator_states, decoded


__all__ = ["K1DecoderAblationLM"]
