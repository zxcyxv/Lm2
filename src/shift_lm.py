"""Latent-shift LM: next-token prediction as a one-step rotation in latent space.

E(x) -> f (reversible causal blocks; RoPE only inside attention on Q/K)
     -> R_omega: rotate every channel pair by one position step (constant operator)
     -> f^{-1} (exact inverse, same weights, executed as the generator)
     -> tied-embedding head.

The network does all semantics; RoPE does only positional bookkeeping:
relative matching inside attention, and the deterministic +1 alignment in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba2 import RMSNorm
from protected_model import rope_angles
from rotlm.geometry.horizon import cyclic_shift_channels


def rotate_pairs(h, phi):
    a, b = h[..., 0::2], h[..., 1::2]
    cos, sin = torch.cos(phi), torch.sin(phi)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class CausalAttentionQKRope(nn.Module):
    """Causal MHA with RoPE applied to Q and K only (values untouched)."""

    def __init__(self, d, n_heads=4):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, positions, horizon=0):
        b, t, d = x.shape
        hd = d // self.n_heads
        q, k, v = self.qkv(x).reshape(b, t, 3, self.n_heads, hd).permute(2, 0, 3, 1, 4)
        phi = rope_angles(positions, hd // 2).unsqueeze(1)  # [b,1,t,hd/2]
        # horizon stamp on queries only: score phase becomes (t - s + horizon)*omega,
        # i.e. attention retargets by `horizon` positions structurally
        phi_q = phi if horizon == 0 else rope_angles(positions + horizon, hd // 2).unsqueeze(1)
        q, k = rotate_pairs(q, phi_q), rotate_pairs(k, phi)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class RevBlock(nn.Module):
    """Additive coupling: Y1 = X1 + Attn(norm(X2)); Y2 = X2 + FFN(norm(Y1))."""

    def __init__(self, d_half, residual_scale=1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.norm1, self.norm2 = RMSNorm(d_half), RMSNorm(d_half)
        self.attn = CausalAttentionQKRope(d_half)
        self.ffn = nn.Sequential(nn.Linear(d_half, 4 * d_half), nn.GELU(),
                                 nn.Linear(4 * d_half, d_half))
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2, positions, horizon=0):
        scale = self.residual_scale
        y1 = x1 + scale * self.attn(self.norm1(x2), positions, horizon)
        y2 = x2 + scale * self.ffn(self.norm2(y1))
        return y1, y2

    def inverse(self, y1, y2, positions, horizon=0):
        scale = self.residual_scale
        x2 = y2 - scale * self.ffn(self.norm2(y1))
        x1 = y1 - scale * self.attn(self.norm1(x2), positions, horizon)
        return x1, x2


class ShiftLM(nn.Module):
    def __init__(self, vocab, d=112, n_blocks=2, mechanism="latent",
                 latent_operator="geometric", orbit_period=8,
                 residual_scale=1.0):
        super().__init__()
        assert d % 4 == 0
        assert mechanism in ("latent", "query")
        if latent_operator not in ("geometric", "cyclic"):
            raise ValueError(f"unknown latent operator: {latent_operator}")
        if latent_operator == "cyclic" and d % orbit_period:
            raise ValueError(f"d={d} must be divisible by orbit_period={orbit_period}")
        self.mechanism = mechanism
        self.latent_operator = latent_operator
        self.orbit_period = orbit_period
        self.residual_scale = float(residual_scale)
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(
            RevBlock(d // 2, residual_scale=self.residual_scale)
            for _ in range(n_blocks)
        )
        self.head_norm = RMSNorm(d)
        # tied head: f^{-1} output is read in token-embedding space

    def encode(self, x, positions):
        """f: token embeddings -> latent Z (before the alignment shift)."""
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(x.shape[0], -1)
        h = self.embed(x)
        return self.encode_hidden(h, positions)

    def encode_hidden(self, h, positions):
        """Apply f to continuous token-space states (no embedding lookup)."""
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(h.shape[0], -1)
        x1, x2 = h[..., : self.d // 2], h[..., self.d // 2:]
        for blk in self.blocks:
            x1, x2 = blk(x1, x2, positions)
        return torch.cat([x1, x2], dim=-1), positions

    def shift(self, z, k=1):
        """Advance by a fixed geometric rotation or finite cyclic operator."""
        if self.latent_operator == "cyclic":
            return cyclic_shift_channels(z, int(k), self.orbit_period)
        k_t = torch.full((1, 1), float(k), device=z.device)
        return rotate_pairs(z, rope_angles(k_t, self.d // 2))

    def decode_hidden(self, z, positions, horizon=0):
        """f^{-1}: latent -> continuous token-space representation.

        horizon stamps the decoder attention queries (structural retargeting);
        0 keeps the exact inverse of encode.
        """
        y1, y2 = z[..., : self.d // 2], z[..., self.d // 2:]
        for blk in reversed(self.blocks):
            y1, y2 = blk.inverse(y1, y2, positions, horizon)
        return torch.cat([y1, y2], dim=-1)

    def decode(self, z, positions, horizon=0):
        """f^{-1} followed by the tied token readout."""
        y = self.decode_hidden(z, positions, horizon)
        return self.head_norm(y) @ self.embed.weight.T

    def forward(self, x, positions, horizon=1):
        z, positions = self.encode(x, positions)
        if self.mechanism == "query":
            return self.decode(z, positions, horizon=horizon)
        return self.decode(self.shift(z, horizon), positions)

    def rollout_logits(self, z, positions, horizons):
        """Decode several latent-rotation horizons in one batched inverse pass.

        No generated token is fed back through encode: every future is read as
        f^{-1}(R^k z) from the same encoded prefix state. Returns [B,K,T,V].
        """
        assert self.mechanism == "latent"
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(z.shape[0], -1)
        hs = tuple(int(k) for k in horizons)
        rotated = torch.cat([self.shift(z, k) for k in hs], dim=0)
        pos = positions.repeat(len(hs), 1)
        logits = self.decode(rotated, pos)
        return logits.reshape(len(hs), z.shape[0], z.shape[1], -1).permute(1, 0, 2, 3)
