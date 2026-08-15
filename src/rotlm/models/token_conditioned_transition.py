"""Observed-token-conditioned recurrent latent-state updates."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TokenConditionedTransition(nn.Module):
    """Turn a next-token proposal into its branch-conditioned successor.

    The proposal predicts the token before this module is called.  Training
    supplies the observed token embedding and inference must supply the
    embedding of the token actually sampled from that proposal.  The returned
    state is therefore available only to later predictions.

    The update is a bounded relative-RMS residual:

    ``state = proposal + scale * RMS(proposal) * unitRMS(residual)``.
    """

    def __init__(
        self,
        width: int,
        *,
        bottleneck: int = 128,
        init_scale: float = 0.05,
        min_scale: float = 1e-3,
        max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if min(width, bottleneck) < 1:
            raise ValueError("width and bottleneck must be positive")
        if not 0 < min_scale < init_scale < max_scale:
            raise ValueError(
                "require 0 < min_scale < init_scale < max_scale"
            )

        self.width = int(width)
        self.bottleneck = int(bottleneck)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        self.proposal_norm = nn.RMSNorm(width, eps=1e-6)
        self.token_norm = nn.RMSNorm(width, eps=1e-6)
        self.proposal = nn.Linear(width, bottleneck, bias=False)
        self.token = nn.Linear(width, bottleneck, bias=False)
        self.gate = nn.Linear(bottleneck, bottleneck)
        self.drive = nn.Linear(bottleneck, bottleneck)
        self.output = nn.Linear(bottleneck, width, bias=False)
        self.scale = nn.Linear(bottleneck, 1)

        for layer in (
            self.proposal,
            self.token,
            self.gate,
            self.drive,
            self.output,
        ):
            nn.init.normal_(layer.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.drive.bias)
        nn.init.zeros_(self.scale.weight)
        scale_fraction = (
            (init_scale - min_scale) / (max_scale - min_scale)
        )
        nn.init.constant_(
            self.scale.bias,
            math.log(scale_fraction / (1.0 - scale_fraction)),
        )

    def forward(
        self,
        proposal: torch.Tensor,
        token_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return conditioned state, innovation, and observed log scale."""
        if proposal.shape != token_embedding.shape:
            raise ValueError(
                "proposal and token embedding must share shape, got "
                f"{tuple(proposal.shape)} and {tuple(token_embedding.shape)}"
            )
        if proposal.shape[-1] != self.width:
            raise ValueError(
                f"expected trailing width {self.width}, got "
                f"{proposal.shape[-1]}"
            )

        features = F.silu(
            self.proposal(self.proposal_norm(proposal))
            + self.token(self.token_norm(token_embedding))
        )
        selected = torch.sigmoid(self.gate(features)) * torch.tanh(
            self.drive(features)
        )
        raw_innovation = self.output(selected)
        direction = raw_innovation / raw_innovation.float().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt().clamp_min(1e-8).to(raw_innovation.dtype)

        scale = self.min_scale + (
            self.max_scale - self.min_scale
        ) * torch.sigmoid(self.scale(features).float())
        proposal_rms = proposal.float().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt()
        innovation = (
            scale * proposal_rms * direction.float()
        ).to(proposal.dtype)
        conditioned = proposal + innovation
        observed_scale = (
            innovation.float().square().mean(dim=-1, keepdim=True).sqrt()
            / proposal_rms.clamp_min(1e-8)
        )
        return (
            conditioned,
            innovation,
            observed_scale.clamp_min(1e-8).log().squeeze(-1),
        )


__all__ = ["TokenConditionedTransition"]
