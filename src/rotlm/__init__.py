"""Minimal reusable package for the MSE+CE latent-orbit lineage."""
from __future__ import annotations

from typing import Any


__all__ = ["K1DecoderAblationLM"]


def __getattr__(name: str) -> Any:
    # Avoid importing the reversible model stack while low-level compatibility
    # modules (notably ``shift_lm``) are still being initialized.
    if name == "K1DecoderAblationLM":
        from .models.k1_decoder_ablation import K1DecoderAblationLM

        return K1DecoderAblationLM
    raise AttributeError(name)
