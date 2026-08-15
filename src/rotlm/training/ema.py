"""Shared gradient-free exponential-moving-average teacher utilities."""
from __future__ import annotations

import copy
from typing import TypeVar

import torch
from torch import nn


ModelT = TypeVar("ModelT", bound=nn.Module)


def make_ema_model(model: ModelT) -> ModelT:
    """Return an exact, evaluation-mode, gradient-free model copy."""
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


@torch.no_grad()
def update_ema_model(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    """Update a complete model EMA after one online optimizer step."""
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0,1)")

    online_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        online_parameter = online_parameters[name]
        ema_parameter.mul_(decay).add_(
            online_parameter.detach(),
            alpha=1.0 - decay,
        )

    online_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        online_buffer = online_buffers[name].detach()
        if torch.is_floating_point(ema_buffer):
            ema_buffer.mul_(decay).add_(
                online_buffer,
                alpha=1.0 - decay,
            )
        else:
            ema_buffer.copy_(online_buffer)


def ema_decay_at_step(
    step: int,
    target_decay: float,
    warm_start_steps: int,
) -> float:
    """Return the post-update EMA decay registered for one-based ``step``."""
    if step < 1:
        raise ValueError("EMA step must be positive")
    if not 0 <= target_decay < 1:
        raise ValueError("EMA target decay must be in [0,1)")
    if warm_start_steps < 0:
        raise ValueError("EMA warm-start steps must be non-negative")
    return 0.0 if step <= warm_start_steps else target_decay


__all__ = ["ema_decay_at_step", "make_ema_model", "update_ema_model"]
