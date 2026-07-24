"""Protected-channel inverse-free LM.

Embedding [B,T,2d] -> input hidden-state RoPE R_P -> involution f(r,c) = (r, U(r)-c)
-> middle RoPE R_{P+1} (strict: no phase bias / quadrature: pi/2 on half the channels)
-> LM head reads protected channel r only. The nominal back-end f^{-1} = f is skipped;
P f^{-1} = P holds exactly because f leaves r untouched.
"""

import math

import torch
import torch.nn as nn

from mamba2 import RMSNorm, Tower


def rope_angles(positions, d, base=10000.0, quadrature=False, paired=False):
    """positions [T] or [B,T] -> phase [.., T, d].

    paired: d/2 distinct frequencies, each used twice with a pi/2 offset on the
    second copy, so per frequency band sin^2 + cos^2 = 1 at every position.
    """
    if paired:
        j = torch.arange(d // 2, device=positions.device, dtype=torch.float32)
        omega = (base ** (-2 * j / d)).repeat(2)
    else:
        j = torch.arange(d, device=positions.device, dtype=torch.float32)
        omega = base ** (-2 * j / (2 * d))
    phi = positions.unsqueeze(-1).float() * omega
    if quadrature or paired:
        delta = torch.zeros(d, device=positions.device)
        delta[d // 2:] = math.pi / 2
        phi = phi + delta
    return phi


def rope_rotate(r, c, phi):
    cos, sin = torch.cos(phi), torch.sin(phi)
    return r * cos - c * sin, r * sin + c * cos


class ProtectedLM(nn.Module):
    def __init__(self, vocab, d=96, n_layers=1, phase_mode="quadrature", premod=False,
                 context_sees_c=False, mlp_head=False, raw_context_input=False):
        super().__init__()
        assert phase_mode in ("strict", "quadrature", "paired")
        self.d = d
        self.premod = premod
        self.paired = phase_mode == "paired"
        self.quadrature = phase_mode == "quadrature"
        self.embed = nn.Embedding(vocab, 2 * d)
        nn.init.normal_(self.embed.weight, std=0.02)
        # optional: context model reads (r, sg(c)) instead of r alone. Gradient to c
        # is cut so c's training signal still comes only through the involution's -c
        # term. Skip/full logits equality is untouched (f never writes to r); the
        # closed-form self-inverse property of f, however, no longer holds.
        # optional: U reads the un-rotated embedding channel r instead of r_p, so the
        # context model never sees absolute-position-modulated content ("value RoPE"
        # removed). Same caveat as context_sees_c: skip/full equality holds, the
        # closed-form involution interpretation weakens.
        self.raw_context_input = raw_context_input
        self.ctx_in = None
        if context_sees_c:
            self.ctx_in = nn.Linear(2 * d, d, bias=False)
            nn.init.normal_(self.ctx_in.weight, std=0.02)
        self.context = Tower(d, n_layers=n_layers, out_proj=True)
        # optional nonlinear readout: fusion into r stays linear (required for the
        # skip), but any function of r_m keeps skip/full logits identical, and a
        # nonlinearity applied to the superposition creates token-context cross terms
        self.head_mlp = None
        if mlp_head:
            self.mlp_norm = RMSNorm(d)
            self.head_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            for m in self.head_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
        self.head_norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x, positions, full_nominal=False):
        h = self.embed(x)
        r, c = h[..., 0::2], h[..., 1::2]
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(x.shape[0], -1)
        r_p, c_p = rope_rotate(r, c, rope_angles(positions, self.d, paired=self.paired))
        if self.ctx_in is not None:
            u = self.context(self.ctx_in(torch.cat([r_p, c_p.detach()], dim=-1)))
        elif self.raw_context_input:
            u = self.context(r)
        else:
            u = self.context(r_p)
        phi_mid = rope_angles(positions + 1, self.d, quadrature=self.quadrature,
                              paired=self.paired)
        if self.premod:
            # pre-modulation: cancel the middle-RoPE mixing coefficient so context
            # reaches r_m as u_raw * sin^2(phi) instead of u_raw * (-sin(phi))
            u = u * (-torch.sin(phi_mid))
        r_t, c_t = r_p, u - c_p                      # involution f (u is a function of r only)
        r_m, c_m = rope_rotate(r_t, c_t, phi_mid)
        if full_nominal:                             # nominal back-end f, r unchanged
            if self.ctx_in is not None:
                u2 = self.context(self.ctx_in(torch.cat([r_m, c_m.detach()], dim=-1)))
            else:
                u2 = self.context(r_m)
            if self.premod:
                u2 = u2 * (-torch.sin(phi_mid))
            r_m, c_m = r_m, u2 - c_m
        if self.head_mlp is not None:
            r_m = r_m + self.head_mlp(self.mlp_norm(r_m))
        return self.head(self.head_norm(r_m))
