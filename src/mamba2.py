"""Pure-PyTorch Mamba-2 (SSD algorithm) — same math as the official kernel, unfused."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class RMSNormGated(nn.Module):
    """Mamba-2's gated norm: RMSNorm(y * silu(z))."""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, y, z):
        y = y * F.silu(z)
        return self.weight * y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)


def segsum(x):
    """Stable segment-sum: [..., T] -> [..., T, T] with entries sum(x[j+1..i]), -inf above diag."""
    T = x.size(-1)
    csum = torch.cumsum(x, dim=-1)
    seg = csum[..., :, None] - csum[..., None, :]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    return seg.masked_fill(~mask, -torch.inf)


def ssd_scan(x, dt, A, B, C, chunk):
    """Exact Mamba-2 SSD scan, matmul-based, causal.

    x: [b,l,h,p]  dt: [b,l,h]  A: [h] (negative)  B,C: [b,l,n] (ngroups=1).
    Returns y: [b,l,h,p]. Compute in float32 for stability.
    """
    b, l, h, p = x.shape
    pad = (-l) % chunk
    if pad:
        x, dt = F.pad(x, (0, 0, 0, 0, 0, pad)), F.pad(dt, (0, 0, 0, pad))
        B, C = F.pad(B, (0, 0, 0, pad)), F.pad(C, (0, 0, 0, pad))
    L = x.shape[1]
    c = L // chunk
    dA = dt * A
    xc = x.reshape(b, c, chunk, h, p)
    dtc = dt.reshape(b, c, chunk, h)
    dAc = dA.reshape(b, c, chunk, h).permute(0, 3, 1, 2)  # [b,h,c,k]
    Bc = B.reshape(b, c, chunk, -1)
    Cc = C.reshape(b, c, chunk, -1)
    # 1. intra-chunk (diagonal blocks)
    Lmat = torch.exp(segsum(dAc))
    Y = torch.einsum("bcln,bcsn,bhcls,bcshp->bclhp", Cc, Bc, Lmat, dtc.unsqueeze(-1) * xc)
    # 2. states at chunk ends
    decay_states = torch.exp(dAc.sum(-1, keepdim=True) - torch.cumsum(dAc, -1))
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", Bc, decay_states, dtc.unsqueeze(-1) * xc)
    # 3. inter-chunk recurrence over chunk states (initial state 0 prepended)
    chunk_decay = dAc.sum(-1)
    dec = torch.exp(segsum(F.pad(chunk_decay, (1, 0))))
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    states = torch.einsum("bhzc,bchpn->bzhpn", dec, states)[:, :-1]
    # 4. state -> output contribution
    state_decay_out = torch.exp(torch.cumsum(dAc, -1))
    Y = Y + torch.einsum("bcln,bchpn,bhcl->bclhp", Cc, states, state_decay_out)
    return Y.reshape(b, L, h, p)[:, :l]


class Mamba2Block(nn.Module):
    """Faithful Mamba-2 block: in_proj -> causal depthwise conv+silu on xBC ->
    SSD scan (per-head scalar A, softplus dt, D skip) -> gated RMSNorm -> out_proj."""

    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, headdim=32, chunk_size=64):
        super().__init__()
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.headdim, self.d_state, self.chunk = headdim, d_state, chunk_size
        d_in_proj = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim,
                                padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        b, l, _ = u.shape
        z, xBC, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C = xBC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        A = -torch.exp(self.A_log.float())
        xh = x.float().reshape(b, l, self.nheads, self.headdim)
        y = ssd_scan(xh, dt, A, B.float(), C.float(), self.chunk)
        y = y + self.D.view(1, 1, -1, 1) * xh
        y = y.reshape(b, l, self.d_inner).to(u.dtype)
        return self.out_proj(self.norm(y, z))


def make_mixer(d, headdim=32):
    assert (2 * d) % headdim == 0
    return Mamba2Block(d, d_state=32, d_conv=4, expand=2, headdim=headdim, chunk_size=64)


class Tower(nn.Module):
    """n_layers pre-norm residual Mamba-2 blocks + final norm (+ optional out_proj)."""

    def __init__(self, d, n_layers, out_proj=True, alpha=0.5, headdim=32):
        super().__init__()
        self.alpha = alpha
        self.norms = nn.ModuleList(RMSNorm(d) for _ in range(n_layers))
        self.mixers = nn.ModuleList(make_mixer(d, headdim) for _ in range(n_layers))
        self.final_norm = RMSNorm(d)
        self.out_proj = None
        if out_proj:
            self.out_proj = nn.Linear(d, d)
            nn.init.normal_(self.out_proj.weight, std=0.02)
            nn.init.zeros_(self.out_proj.bias)
        self.call_count = 0

    def forward(self, h):
        self.call_count += 1
        for norm, mixer in zip(self.norms, self.mixers):
            h = h + self.alpha * mixer(norm(h))
        h = self.final_norm(h)
        return self.out_proj(h) if self.out_proj is not None else h
