"""Causal joint-orbit ShiftLM promoted from the legacy experiment script.

For an encoded prefix ``Z_0..Z_t``, the inverse decoder receives

    [Z_0..Z_t, R Z_t, R^2 Z_t, ..., R^K Z_t]

on one sequence axis.  Consequently future slot ``j`` can attend to the prefix
and earlier orbit slots in the same exact inverse pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from protected_model import rope_angles
from shift_lm import ShiftLM, rotate_pairs


class _LinearWithDetachedInputView(torch.autograd.Function):
    """One linear forward with live-input and detached-input gradient views."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(inputs, weight)
        ctx.has_bias = bias is not None
        output = F.linear(inputs, weight, bias)
        # The view gives autograd two distinct output edges without allocating
        # a second QKV activation.
        return output, output.view_as(output)

    @staticmethod
    def backward(
        ctx,
        live_gradient: torch.Tensor | None,
        detached_input_gradient: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        inputs, weight = ctx.saved_tensors
        if live_gradient is None and detached_input_gradient is None:
            return None, None, None
        if live_gradient is None:
            live_gradient = torch.zeros_like(detached_input_gradient)
        if detached_input_gradient is None:
            detached_input_gradient = torch.zeros_like(live_gradient)

        input_gradient = None
        if ctx.needs_input_grad[0]:
            # Only consumers of the live view can update the state input.
            input_gradient = live_gradient.matmul(weight)

        combined_gradient = live_gradient + detached_input_gradient
        weight_gradient = None
        if ctx.needs_input_grad[1]:
            flat_gradient = combined_gradient.reshape(
                -1, combined_gradient.shape[-1]
            )
            flat_inputs = inputs.reshape(-1, inputs.shape[-1])
            weight_gradient = flat_gradient.T @ flat_inputs

        bias_gradient = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduction_dimensions = tuple(
                range(combined_gradient.ndim - 1)
            )
            bias_gradient = combined_gradient.sum(
                dim=reduction_dimensions
            )
        return input_gradient, weight_gradient, bias_gradient


@dataclass
class OrbitOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    latent: torch.Tensor
    positions: torch.Tensor


class PrefixOrbitLM(ShiftLM):
    """ShiftLM with a first-class, causally joint prefix-orbit decoder."""

    def append_orbit_from_root(
        self,
        prefix_latent: torch.Tensor,
        positions: torch.Tensor,
        anchor: int,
        root: torch.Tensor,
        horizons: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `R root, ..., R^K root` to an encoded prefix.

        Separating the root from the encoded anchor lets a small external plan
        module select a stochastic trajectory without duplicating the shared
        causal inverse implementation.
        """
        if not 0 <= anchor < prefix_latent.shape[1]:
            raise ValueError(f"anchor {anchor} outside sequence length {prefix_latent.shape[1]}")
        if root.shape != (prefix_latent.shape[0], self.d):
            raise ValueError(
                f"expected root {(prefix_latent.shape[0], self.d)}, got {tuple(root.shape)}"
            )
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(prefix_latent.shape[0], -1)
        orbit = torch.cat(
            [self.shift(root[:, None], step) for step in range(1, horizons + 1)],
            dim=1,
        )
        future_pos = positions[:, anchor:anchor + 1] + torch.arange(
            1, horizons + 1, device=positions.device, dtype=positions.dtype
        ).view(1, -1)
        return (
            torch.cat((prefix_latent[:, :anchor + 1], orbit), dim=1),
            torch.cat((positions[:, :anchor + 1], future_pos), dim=1),
        )

    def append_orbit(self, prefix_latent: torch.Tensor, positions: torch.Tensor,
                     anchor: int, horizons: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.append_orbit_from_root(
            prefix_latent,
            positions,
            anchor,
            prefix_latent[:, anchor],
            horizons,
        )

    def decode_orbit_from_root(
        self,
        prefix_latent: torch.Tensor,
        positions: torch.Tensor,
        anchor: int,
        root: torch.Tensor,
        horizons: int,
    ) -> OrbitOutput:
        extended, extended_pos = self.append_orbit_from_root(
            prefix_latent, positions, anchor, root, horizons
        )
        hidden = self.decode_hidden(extended, extended_pos, horizon=0)
        logits = self.head_norm(hidden[:, -horizons:]) @ self.embed.weight.T
        return OrbitOutput(
            logits=logits,
            hidden=hidden,
            latent=extended,
            positions=extended_pos,
        )

    @staticmethod
    def _independent_branch_attention(
        attn,
        prefix: torch.Tensor,
        branches: torch.Tensor,
        prefix_pos: torch.Tensor,
        branch_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode many independent one-slot branches over one shared prefix.

        Every branch sees the complete prefix and itself, but never another
        sampled branch. This is exactly equivalent to repeating the prefix for
        every branch and is used to evaluate hundreds of random tapes without
        redundantly recomputing the prefix.
        """
        b, t, d = prefix.shape
        samples = branches.shape[1]
        heads = attn.n_heads
        hd = d // heads
        prefix_out = attn(prefix, prefix_pos)
        pqkv = attn.qkv(prefix).reshape(
            b, t, 3, heads, hd
        ).permute(2, 0, 3, 1, 4)
        bqkv = attn.qkv(branches).reshape(
            b, samples, 3, heads, hd
        ).permute(2, 0, 3, 1, 4)
        _, prefix_key, prefix_value = pqkv
        branch_query, branch_key, branch_value = bqkv
        prefix_phase = rope_angles(prefix_pos, hd // 2).unsqueeze(1)
        branch_phase = rope_angles(branch_pos, hd // 2).unsqueeze(1)
        prefix_key = rotate_pairs(prefix_key, prefix_phase)
        branch_query = rotate_pairs(branch_query, branch_phase)
        branch_key = rotate_pairs(branch_key, branch_phase)
        scale = hd ** -0.5
        prefix_score = torch.einsum(
            "bhsd,bhtd->bhst", branch_query, prefix_key
        ) * scale
        self_score = (
            branch_query * branch_key
        ).sum(dim=-1, keepdim=True) * scale
        weight = torch.softmax(
            torch.cat((prefix_score, self_score), dim=-1).float(), dim=-1
        ).to(branch_query.dtype)
        prefix_weight = weight[..., :t]
        self_weight = weight[..., t:]
        output = (
            torch.einsum(
                "bhst,bhtd->bhsd", prefix_weight, prefix_value
            )
            + self_weight * branch_value
        )
        output = attn.proj(
            output.transpose(1, 2).reshape(b, samples, d)
        )
        return prefix_out, output

    def decode_independent_one_step_roots(
        self,
        prefix_latent: torch.Tensor,
        positions: torch.Tensor,
        roots: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for independent `R root` branches sharing a prefix.

        Args:
            prefix_latent: encoded prefixes, shape `[B,T,D]`.
            positions: prefix positions, shape `[B,T]` or `[T]`.
            roots: stochastic roots, shape `[B,S,D]`.
        Returns:
            Logits with shape `[B,S,V]`.
        """
        if roots.ndim != 3 or roots.shape[0] != prefix_latent.shape[0]:
            raise ValueError("roots must have shape [prefix_batch,samples,width]")
        if roots.shape[2] != self.d:
            raise ValueError(f"root width {roots.shape[2]} != model width {self.d}")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(prefix_latent.shape[0], -1)
        branch_pos = positions[:, -1:] + 1
        branch_pos = branch_pos.expand(-1, roots.shape[1])
        branch = self.shift(roots, 1)
        py1, py2 = (
            prefix_latent[..., : self.d // 2],
            prefix_latent[..., self.d // 2 :],
        )
        by1, by2 = branch[..., : self.d // 2], branch[..., self.d // 2 :]
        for block in reversed(self.blocks):
            px2 = py2 - block.ffn(block.norm2(py1))
            bx2 = by2 - block.ffn(block.norm2(by1))
            prefix_attention, branch_attention = self._independent_branch_attention(
                block.attn,
                block.norm1(px2),
                block.norm1(bx2),
                positions,
                branch_pos,
            )
            px1 = py1 - prefix_attention
            bx1 = by1 - branch_attention
            py1, py2, by1, by2 = px1, px2, bx1, bx2
        hidden = torch.cat((by1, by2), dim=-1)
        return self.head_norm(hidden) @ self.embed.weight.T

    def decode_orbit_from_latent(self, prefix_latent: torch.Tensor, positions: torch.Tensor,
                                 anchor: int, horizons: int) -> OrbitOutput:
        return self.decode_orbit_from_root(
            prefix_latent,
            positions,
            anchor,
            prefix_latent[:, anchor],
            horizons,
        )

    def decode_orbit(self, prefix_tokens: torch.Tensor, positions: torch.Tensor,
                     anchor: int | None = None, horizons: int = 8) -> OrbitOutput:
        latent, positions = self.encode(prefix_tokens, positions)
        if anchor is None:
            anchor = prefix_tokens.shape[1] - 1
        return self.decode_orbit_from_latent(latent, positions, anchor, horizons)

    @staticmethod
    def _branch_attention(attn, prefix: torch.Tensor, branch: torch.Tensor,
                          prefix_pos: torch.Tensor, branch_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention on a prefix plus independent per-anchor causal branches.

        ``branch[b,t,j]`` sees ``prefix[b,:t+1]`` and ``branch[b,t,:j+1]``;
        it cannot see another anchor's branch. Prefix attention remains ordinary
        causal attention. This is the dense shared-prefix equivalent of decoding
        one ``[prefix[:t+1], orbit[t,:]]`` sequence for every anchor.
        """
        b, t, d = prefix.shape; k = branch.shape[2]; heads = attn.n_heads; hd = d // heads
        prefix_out = attn(prefix, prefix_pos)
        pqkv = attn.qkv(prefix).reshape(b, t, 3, heads, hd).permute(2, 0, 3, 1, 4)
        bqkv = attn.qkv(branch).reshape(b, t, k, 3, heads, hd).permute(3, 0, 4, 1, 2, 5)
        _, pk, pv = pqkv; bq, bk, bv = bqkv
        pphi = rope_angles(prefix_pos, hd // 2).unsqueeze(1)
        bphi = rope_angles(branch_pos, hd // 2).unsqueeze(1)
        pk = rotate_pairs(pk, pphi); bq = rotate_pairs(bq, bphi); bk = rotate_pairs(bk, bphi)
        scale = hd ** -0.5
        prefix_scores = torch.einsum("bhtkd,bhsd->bhtks", bq, pk) * scale
        branch_scores = torch.einsum("bhtkd,bhtrd->bhtkr", bq, bk) * scale
        ti = torch.arange(t, device=prefix.device)
        prefix_mask = ti.view(1, 1, t, 1, 1) >= ti.view(1, 1, 1, 1, t)
        hi = torch.arange(k, device=prefix.device)
        branch_mask = hi.view(1, 1, 1, k, 1) >= hi.view(1, 1, 1, 1, k)
        prefix_scores = prefix_scores.masked_fill(~prefix_mask, float("-inf"))
        branch_scores = branch_scores.masked_fill(~branch_mask, float("-inf"))
        weights = torch.softmax(torch.cat((prefix_scores, branch_scores), dim=-1).float(), dim=-1).to(bq.dtype)
        wp, wb = weights[..., :t], weights[..., t:]
        out = (torch.einsum("bhtks,bhsd->bhtkd", wp, pv)
               + torch.einsum("bhtkr,bhtrd->bhtkd", wb, bv))
        out = attn.proj(out.permute(0, 2, 3, 1, 4).reshape(b, t, k, d))
        return prefix_out, out

    @staticmethod
    def _selected_branch_attention(
        attn,
        prefix: torch.Tensor,
        branch: torch.Tensor,
        prefix_pos: torch.Tensor,
        branch_pos: torch.Tensor,
        anchor_indices: torch.Tensor,
        detach_past_branch_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention for a sparse set of causal prefix anchors.

        ``branch[b,a,j]`` sees ``prefix[b,:anchor_indices[a]+1]`` and the
        earlier slots ``branch[b,a,:j+1]`` from its own anchor only.  This is
        the sparse-anchor counterpart of :meth:`_branch_attention`; it avoids
        materializing unused branch tapes between selected anchors.  When
        ``detach_past_branch_states`` is true, a later slot sees identical
        key/value numbers from earlier branch slots, but its gradient cannot
        flow into those earlier state tensors.
        """
        b, t, d = prefix.shape
        anchors, horizons = branch.shape[1:3]
        if anchor_indices.shape != (anchors,):
            raise ValueError(
                "anchor_indices must have one entry per branch anchor"
            )
        heads = attn.n_heads
        head_dim = d // heads
        prefix_out = attn(prefix, prefix_pos)
        prefix_qkv = attn.qkv(prefix).reshape(
            b, t, 3, heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        detached_branch_qkv = None
        if detach_past_branch_states:
            branch_qkv, detached_branch_qkv = (
                _LinearWithDetachedInputView.apply(
                    branch,
                    attn.qkv.weight,
                    attn.qkv.bias,
                )
            )
        else:
            branch_qkv = attn.qkv(branch)
        branch_qkv = branch_qkv.reshape(
            b, anchors, horizons, 3, heads, head_dim
        ).permute(3, 0, 4, 1, 2, 5)
        _, prefix_key, prefix_value = prefix_qkv
        branch_query, branch_key, branch_value = branch_qkv
        detached_branch_key = None
        detached_branch_value = None
        if detached_branch_qkv is not None:
            detached_branch_qkv = detached_branch_qkv.reshape(
                b, anchors, horizons, 3, heads, head_dim
            ).permute(3, 0, 4, 1, 2, 5)
            (
                _,
                detached_branch_key,
                detached_branch_value,
            ) = detached_branch_qkv

        prefix_phase = rope_angles(prefix_pos, head_dim // 2).unsqueeze(1)
        branch_phase = rope_angles(
            branch_pos, head_dim // 2
        ).unsqueeze(1)
        prefix_key = rotate_pairs(prefix_key, prefix_phase)
        branch_query = rotate_pairs(branch_query, branch_phase)
        branch_key = rotate_pairs(branch_key, branch_phase)
        if detached_branch_key is not None:
            detached_branch_key = rotate_pairs(
                detached_branch_key, branch_phase
            )

        scale = head_dim ** -0.5
        prefix_scores = torch.einsum(
            "bhakd,bhsd->bhaks", branch_query, prefix_key
        ) * scale
        live_branch_scores = torch.einsum(
            "bhakd,bhard->bhakr", branch_query, branch_key
        ) * scale
        prefix_slot = torch.arange(t, device=prefix.device)
        prefix_mask = anchor_indices.view(
            1, 1, anchors, 1, 1
        ) >= prefix_slot.view(1, 1, 1, 1, t)
        horizon_slot = torch.arange(horizons, device=prefix.device)
        branch_mask = horizon_slot.view(
            1, 1, 1, horizons, 1
        ) >= horizon_slot.view(1, 1, 1, 1, horizons)
        past_branch_mask = horizon_slot.view(
            1, 1, 1, horizons, 1
        ) > horizon_slot.view(1, 1, 1, 1, horizons)
        if detached_branch_key is None:
            branch_scores = live_branch_scores
        else:
            detached_branch_scores = torch.einsum(
                "bhakd,bhard->bhakr",
                branch_query,
                detached_branch_key,
            ) * scale
            branch_scores = torch.where(
                past_branch_mask,
                detached_branch_scores,
                live_branch_scores,
            )
        prefix_scores = prefix_scores.masked_fill(
            ~prefix_mask, float("-inf")
        )
        branch_scores = branch_scores.masked_fill(
            ~branch_mask, float("-inf")
        )
        weights = torch.softmax(
            torch.cat((prefix_scores, branch_scores), dim=-1).float(),
            dim=-1,
        ).to(branch_query.dtype)
        prefix_weight = weights[..., :t]
        branch_weight = weights[..., t:]
        output = torch.einsum(
            "bhaks,bhsd->bhakd", prefix_weight, prefix_value
        )
        if detached_branch_value is None:
            output = output + torch.einsum(
                "bhakr,bhard->bhakd", branch_weight, branch_value
            )
        else:
            past_weight = branch_weight * past_branch_mask.to(
                branch_weight.dtype
            )
            live_weight = branch_weight * (~past_branch_mask).to(
                branch_weight.dtype
            )
            output = (
                output
                + torch.einsum(
                    "bhakr,bhard->bhakd",
                    past_weight,
                    detached_branch_value,
                )
                + torch.einsum(
                    "bhakr,bhard->bhakd",
                    live_weight,
                    branch_value,
                )
            )
        output = attn.proj(
            output.permute(0, 2, 3, 1, 4).reshape(
                b, anchors, horizons, d
            )
        )
        return prefix_out, output

    @staticmethod
    def _selected_multi_trajectory_attention(
        attn,
        prefix: torch.Tensor,
        branch: torch.Tensor,
        prefix_pos: torch.Tensor,
        branch_pos: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse branch attention with one shared prefix and many tapes.

        ``branch`` has shape
        ``[batch,trajectories,anchors,horizons,width]``.  Prefix attention,
        prefix QKV, and prefix keys/values are evaluated once per original
        example.  Only branch QKV, branch-to-prefix scores, and branch-local
        causal attention carry the trajectory dimension.
        """
        if prefix.ndim != 3 or branch.ndim != 5:
            raise ValueError(
                "prefix/branch must have shapes [b,t,d] and [b,m,a,k,d]"
            )
        b, t, d = prefix.shape
        branch_batch, trajectories, anchors, horizons, branch_width = (
            branch.shape
        )
        if branch_batch != b or branch_width != d:
            raise ValueError("branch batch/width must match prefix")
        if anchor_indices.shape != (anchors,):
            raise ValueError(
                "anchor_indices must have one entry per branch anchor"
            )
        heads = attn.n_heads
        head_dim = d // heads

        prefix_out = attn(prefix, prefix_pos)
        prefix_qkv = attn.qkv(prefix).reshape(
            b, t, 3, heads, head_dim
        ).permute(2, 0, 3, 1, 4)
        branch_qkv = attn.qkv(branch).reshape(
            b,
            trajectories,
            anchors,
            horizons,
            3,
            heads,
            head_dim,
        ).permute(4, 0, 1, 5, 2, 3, 6)
        _, prefix_key, prefix_value = prefix_qkv
        branch_query, branch_key, branch_value = branch_qkv

        prefix_phase = rope_angles(
            prefix_pos, head_dim // 2
        ).unsqueeze(1)
        branch_phase = (
            rope_angles(branch_pos, head_dim // 2)
            .unsqueeze(1)
            .unsqueeze(2)
        )
        prefix_key = rotate_pairs(prefix_key, prefix_phase)
        branch_query = rotate_pairs(branch_query, branch_phase)
        branch_key = rotate_pairs(branch_key, branch_phase)

        scale = head_dim ** -0.5
        prefix_scores = torch.einsum(
            "bmhakd,bhsd->bmhaks",
            branch_query,
            prefix_key,
        ) * scale
        branch_scores = torch.einsum(
            "bmhakd,bmhard->bmhakr",
            branch_query,
            branch_key,
        ) * scale
        prefix_slot = torch.arange(t, device=prefix.device)
        prefix_mask = anchor_indices.view(
            1, 1, 1, anchors, 1, 1
        ) >= prefix_slot.view(1, 1, 1, 1, 1, t)
        horizon_slot = torch.arange(horizons, device=prefix.device)
        branch_mask = horizon_slot.view(
            1, 1, 1, 1, horizons, 1
        ) >= horizon_slot.view(1, 1, 1, 1, 1, horizons)
        prefix_scores = prefix_scores.masked_fill(
            ~prefix_mask, float("-inf")
        )
        branch_scores = branch_scores.masked_fill(
            ~branch_mask, float("-inf")
        )
        weights = torch.softmax(
            torch.cat((prefix_scores, branch_scores), dim=-1).float(),
            dim=-1,
        ).to(branch_query.dtype)
        prefix_weight = weights[..., :t]
        branch_weight = weights[..., t:]
        output = torch.einsum(
            "bmhaks,bhsd->bmhakd",
            prefix_weight,
            prefix_value,
        )
        output = output + torch.einsum(
            "bmhakr,bmhard->bmhakd",
            branch_weight,
            branch_value,
        )
        output = attn.proj(
            output.permute(0, 1, 3, 4, 2, 5).reshape(
                b,
                trajectories,
                anchors,
                horizons,
                d,
            )
        )
        return prefix_out, output

    def dense_orbit_latent(self, prefix_latent: torch.Tensor, positions: torch.Tensor,
                           horizons: int) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(prefix_latent.shape[0], -1)
        branch = torch.stack([self.shift(prefix_latent, step)
                              for step in range(1, horizons + 1)], dim=2)
        branch_pos = positions.unsqueeze(-1) + torch.arange(
            1, horizons + 1, device=positions.device, dtype=positions.dtype).view(1, 1, -1)
        return branch, branch_pos

    def dense_decode_hidden(self, prefix_latent: torch.Tensor, positions: torch.Tensor,
                            horizons: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jointly invert every prefix anchor with shared prefix computation."""
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(prefix_latent.shape[0], -1)
        branch, branch_pos = self.dense_orbit_latent(prefix_latent, positions, horizons)
        py1, py2 = prefix_latent[..., :self.d//2], prefix_latent[..., self.d//2:]
        by1, by2 = branch[..., :self.d//2], branch[..., self.d//2:]
        for block in reversed(self.blocks):
            px2 = py2 - block.ffn(block.norm2(py1)); bx2 = by2 - block.ffn(block.norm2(by1))
            pa, ba = self._branch_attention(block.attn, block.norm1(px2), block.norm1(bx2),
                                            positions, branch_pos)
            px1, bx1 = py1 - pa, by1 - ba
            py1, py2, by1, by2 = px1, px2, bx1, bx2
        return torch.cat((py1, py2), -1), torch.cat((by1, by2), -1), branch_pos

    def dense_encode_hidden(self, prefix_hidden: torch.Tensor, branch_hidden: torch.Tensor,
                            positions: torch.Tensor, branch_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward counterpart of :meth:`dense_decode_hidden`, used as an exactness gate."""
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(prefix_hidden.shape[0], -1)
        px1, px2 = prefix_hidden[..., :self.d//2], prefix_hidden[..., self.d//2:]
        bx1, bx2 = branch_hidden[..., :self.d//2], branch_hidden[..., self.d//2:]
        for block in self.blocks:
            pa, ba = self._branch_attention(block.attn, block.norm1(px2), block.norm1(bx2),
                                            positions, branch_pos)
            py1, by1 = px1 + pa, bx1 + ba
            py2, by2 = px2 + block.ffn(block.norm2(py1)), bx2 + block.ffn(block.norm2(by1))
            px1, px2, bx1, bx2 = py1, py2, by1, by2
        return torch.cat((px1, px2), -1), torch.cat((bx1, bx2), -1)

    def dense_orbit_logits(self, prefix_latent: torch.Tensor, positions: torch.Tensor,
                           horizons: int) -> torch.Tensor:
        _, branch_hidden, _ = self.dense_decode_hidden(prefix_latent, positions, horizons)
        return self.head_norm(branch_hidden) @ self.embed.weight.T
