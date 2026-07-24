"""Shared cosine-codebook readout for tied token embeddings.

The head normalizes both continuous hidden states and frozen or trainable
embedding rows before their dot product.  This can make embedding rows a
token-level left inverse when their directions are unique.  It does not make
arbitrary latent states invertible and does not differentiate token selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_EPS = 1e-12


def _validate_embedding_weight(embedding_weight: torch.Tensor) -> None:
    if embedding_weight.ndim != 2:
        raise ValueError(
            "embedding_weight must have shape [vocabulary,width], got "
            f"{tuple(embedding_weight.shape)}"
        )
    if embedding_weight.shape[0] < 1 or embedding_weight.shape[1] < 1:
        raise ValueError("embedding_weight dimensions must be positive")
    if not embedding_weight.is_floating_point():
        raise TypeError("embedding_weight must have a floating-point dtype")


def checkpoint_cosine_scale(embedding_weight: torch.Tensor) -> float:
    """Return a data-free scale initialized solely from checkpoint weights.

    The exact definition is

    ``sqrt(width) * median(row_l2_norm(embedding_weight.float()))``.

    No token samples, logits, labels, or calibration dataset are consulted.
    The returned Python float is detached from the checkpoint graph and can be
    used for either a fixed buffer or a trainable log-scale initialization.
    """
    _validate_embedding_weight(embedding_weight)
    with torch.no_grad():
        row_norm = embedding_weight.detach().float().norm(p=2, dim=-1)
        scale = math.sqrt(embedding_weight.shape[1]) * row_norm.median()
        value = float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "checkpoint-derived cosine scale must be finite and positive, "
            f"got {value}"
        )
    return value


def _positive_scalar(
    scale: float | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        value = scale.to(device=device, dtype=torch.float32)
    else:
        value = torch.tensor(float(scale), device=device, dtype=torch.float32)
    if value.ndim != 0:
        raise ValueError(f"scale must be scalar, got shape {tuple(value.shape)}")
    detached = value.detach()
    if not bool(torch.isfinite(detached)) or not bool(detached > 0):
        raise ValueError(
            "cosine scale must be finite and strictly positive, "
            f"got {float(detached)}"
        )
    return value


def cosine_codebook_logits(
    hidden: torch.Tensor,
    embedding_weight: torch.Tensor,
    scale: float | torch.Tensor = 1.0,
    *,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Return float32 scaled-cosine logits with shape ``[..., vocabulary]``.

    Gradients propagate through ``hidden``, ``embedding_weight``, and a tensor
    ``scale``.  Normalization and matmul are deliberately evaluated in
    float32, including when model activations or embeddings use a lower
    precision dtype.
    """
    _validate_embedding_weight(embedding_weight)
    if hidden.ndim < 1:
        raise ValueError("hidden must have at least one dimension")
    if not hidden.is_floating_point():
        raise TypeError("hidden must have a floating-point dtype")
    if hidden.shape[-1] != embedding_weight.shape[-1]:
        raise ValueError(
            f"hidden width {hidden.shape[-1]} does not match embedding width "
            f"{embedding_weight.shape[-1]}"
        )
    if hidden.device != embedding_weight.device:
        raise ValueError(
            f"hidden and embedding_weight must share a device, got "
            f"{hidden.device} and {embedding_weight.device}"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    positive_scale = _positive_scalar(scale, device=hidden.device)
    normalized_hidden = F.normalize(
        hidden.float(), p=2, dim=-1, eps=eps
    )
    normalized_codebook = F.normalize(
        embedding_weight.float(), p=2, dim=-1, eps=eps
    )
    return (normalized_hidden @ normalized_codebook.T) * positive_scale


class CosineCodebookHead(nn.Module):
    """Tied cosine readout with a fixed or trainable positive scale.

    The module does not own an embedding table.  Pass the tied embedding weight
    on every forward call so gradients and state ownership remain with the
    language model.
    """

    def __init__(
        self,
        initial_scale: float | torch.Tensor = 1.0,
        *,
        trainable_scale: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}")
        if isinstance(initial_scale, torch.Tensor):
            if initial_scale.numel() != 1:
                raise ValueError("initial_scale must contain exactly one value")
            initial = float(initial_scale.detach())
        else:
            initial = float(initial_scale)
        if not math.isfinite(initial) or initial <= 0.0:
            raise ValueError(
                "initial_scale must be finite and strictly positive, "
                f"got {initial}"
            )
        self.eps = float(eps)
        self.trainable_scale = bool(trainable_scale)
        initial_tensor = torch.tensor(initial, dtype=torch.float32)
        if not bool(torch.isfinite(initial_tensor)) or not bool(initial_tensor > 0):
            raise ValueError(
                "initial_scale must be representable as a positive finite float32, "
                f"got {initial}"
            )
        if self.trainable_scale:
            self.log_scale = nn.Parameter(initial_tensor.log())
            self.register_buffer("fixed_scale", None)
        else:
            self.register_parameter("log_scale", None)
            self.register_buffer("fixed_scale", initial_tensor)

    @classmethod
    def from_embedding(
        cls,
        embedding_weight: torch.Tensor,
        *,
        trainable_scale: bool = False,
        eps: float = DEFAULT_EPS,
    ) -> "CosineCodebookHead":
        """Construct using :func:`checkpoint_cosine_scale` only."""
        return cls(
            checkpoint_cosine_scale(embedding_weight),
            trainable_scale=trainable_scale,
            eps=eps,
        )

    @property
    def scale(self) -> torch.Tensor:
        """Current strictly positive scalar scale."""
        if self.log_scale is not None:
            # The clamp preserves positivity even if an unconstrained optimizer
            # drives log_scale below float32's exponential range.
            return self.log_scale.float().exp().clamp_min(
                torch.finfo(torch.float32).tiny
            )
        if self.fixed_scale is None:  # pragma: no cover - state invariant
            raise RuntimeError("fixed cosine scale is unexpectedly absent")
        return self.fixed_scale.float()

    def forward(
        self,
        hidden: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        return cosine_codebook_logits(
            hidden,
            embedding_weight,
            self.scale,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"scale={float(self.scale.detach()):.6g}, "
            f"trainable_scale={self.trainable_scale}, eps={self.eps:g}"
        )


@dataclass(frozen=True)
class CosineCodebookAudit:
    """Aggregate positive-scale-invariant embedding-row identity audit."""

    vocabulary_size: int
    self_identity_count: int
    directional_tie_count: int
    minimum_nearest_other_margin: float
    output_image_size: int

    @property
    def self_identity_rate(self) -> float:
        return self.self_identity_count / self.vocabulary_size


@torch.no_grad()
def audit_cosine_codebook(
    embedding_weight: torch.Tensor,
    *,
    chunk_size: int = 256,
    eps: float = DEFAULT_EPS,
) -> CosineCodebookAudit:
    """Audit row self-identity and directional ties without an ``V x V`` save.

    A directional tie is counted when the nearest other cosine is greater
    than or equal to the evaluated float32 self cosine.  Since every allowed
    scale is positive, scale is intentionally absent from this audit.
    """
    _validate_embedding_weight(embedding_weight)
    vocabulary = embedding_weight.shape[0]
    if vocabulary < 2:
        raise ValueError("codebook audit requires at least two embedding rows")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    codebook = F.normalize(
        embedding_weight.float(), p=2, dim=-1, eps=eps
    )
    self_identity_count = 0
    directional_tie_count = 0
    minimum_margin = math.inf
    output_ids: set[int] = set()
    for begin in range(0, vocabulary, chunk_size):
        end = min(vocabulary, begin + chunk_size)
        score = codebook[begin:end] @ codebook.T
        local_id = torch.arange(end - begin, device=score.device)
        token_id = torch.arange(begin, end, device=score.device)
        top1 = score.argmax(dim=-1)
        self_cosine = score[local_id, token_id].clone()
        score[local_id, token_id] = -torch.inf
        nearest_other = score.amax(dim=-1)
        margin = self_cosine - nearest_other
        self_identity_count += int(top1.eq(token_id).sum())
        directional_tie_count += int(nearest_other.ge(self_cosine).sum())
        minimum_margin = min(minimum_margin, float(margin.amin()))
        output_ids.update(top1.cpu().tolist())
    return CosineCodebookAudit(
        vocabulary_size=vocabulary,
        self_identity_count=self_identity_count,
        directional_tie_count=directional_tie_count,
        minimum_nearest_other_margin=minimum_margin,
        output_image_size=len(output_ids),
    )


__all__ = [
    "DEFAULT_EPS",
    "CosineCodebookAudit",
    "CosineCodebookHead",
    "audit_cosine_codebook",
    "checkpoint_cosine_scale",
    "cosine_codebook_logits",
]
