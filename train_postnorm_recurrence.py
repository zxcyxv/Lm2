"""Post-normalized central recurrence -- data loading, central operator and
training loop written out here rather than assembled from the flag surface of
``ComplexSelfPredictedKVTransition``.

The registered structure, fixed (not optional) in this file:

* **The read is linear.**  ``r_h = q_h^dagger S_h / sqrt(d_k)`` with no
  ``||S||_F`` division.  No linear-attention model normalizes the read by the
  state's own matrix norm; the division also projects the gradient onto the
  sphere, which sets the component along ``S`` to exactly zero and leaves the
  write magnitude unlearnable.

* **The recurrence stays affine; normalization sits after it.**
  ``z_h = R z_(h-1) + delta_h``, then ``z~_h = N(z_h)`` for the decoder.
  Normalization is behind the recurrence rather than in front of Q/K/V, but
  it is *not* folded into the recurrence itself.

  Folding it in -- ``z_h = N(R z_(h-1) + delta_h)`` -- was tried and rejected.
  ``N`` is a per-state scalar divide: it adds no routing, no mixing, and no
  path by which horizon ``h`` sees anything it could not see before.  It would
  cost the affine composition that lets the latent chain factor into a prefix
  scan, and buy a rescale.  It would also not turn this into the
  state-dependent lineage, because Q/K/V still read the forcing tape and never
  the evolving state, so no feedback edge appears either way.

* **The root enters normalized.**  ``z_0 = N(encoder_root)``, so the forcing
  tape starts at a fixed scale.

* **Everything the decoder sees is post-normalized.**  ``z_h`` is already the
  output of ``N`` in the line above, so the decode tape needs no further
  normalization and none is applied.

* **No Q/K/V pre-normalization.**  It would be redundant: ``z_0`` is normalized
  and ``c_h = R^h z_0`` is a unitary orbit, so ``||c_h||`` is already constant
  in ``h``.

* **Multi-value attention, no read bottleneck.**  One query and one key shared
  across heads (Mamba's B/C layout); only the value stays per head.
  ``read_width = 2 * heads * value_dim`` is held equal to ``width``, so the
  central operator writes into the full latent rather than a fixed subspace.

Consequences worth stating, because they are not accidents of the code:

Both recurrences keep an ``h``-independent multiplier, so both close in a
rotating frame and evaluate as one cumulative sum apiece:

    z_h = R^h * (z_0 + sum_(i<=h) R^(-i) delta_i)
    S_h = U^h * sum_(i<=h) U^(-i) W_i

``R`` is orthogonal and ``|U| = 1``, so neither inverse frame amplifies.
Nothing in the horizon axis runs sequentially unless a variant below asks for
it.

The memory itself has no decay here.  ``||S_h||`` therefore grows with ``h``
and so does ``delta_h``; the post-normalization is what keeps ``z`` bounded
regardless.  If that growth turns out to matter, decay belongs on ``S``, not
a division on the read.

Cross entropy covers **every** horizon with equal weight, matching
``train_time_varying_scan.py``.  There is no H4-train/H16-monitor split here,
so ``block_nll`` is the mean of the per-horizon NLLs -- the same quantity as
``train_ce`` on the other split.

Encoder and exact-inverse decoder are imported rather than re-derived: they
are shared with the existing lineage and unchanged by this experiment.

    python train_postnorm_recurrence.py            # 13M defaults
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rotlm.dataset import token_memmap
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM

VOCABULARY = 256
DATA_ROOT = Path("data/wikitext103_bytes")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def memmap(split: str):
    return token_memmap(split, root=DATA_ROOT)


def windows_of_length(data, starts: torch.Tensor, length: int) -> torch.Tensor:
    values = np.stack(
        [
            np.asarray(data[int(s) : int(s) + length], dtype=np.int64)
            for s in starts.tolist()
        ]
    )
    return torch.from_numpy(values).cuda(non_blocking=True)


def sampled_windows(
    data, batch: int, generator: torch.Generator, length: int
) -> torch.Tensor:
    starts = torch.randint(0, len(data) - length, (batch,), generator=generator)
    return windows_of_length(data, starts, length)


def anchor_indices_for(prefix_length: int, stride: int, device) -> torch.Tensor:
    return torch.arange(0, prefix_length, stride, device=device, dtype=torch.long)


# --------------------------------------------------------------------------
# Central operator
# --------------------------------------------------------------------------

def rms_normalize(hidden: torch.Tensor, eps: float) -> torch.Tensor:
    """Fixed, non-affine RMS normalization."""
    original = hidden.dtype
    promoted = hidden.float()
    variance = promoted.square().mean(dim=-1, keepdim=True)
    return (promoted * torch.rsqrt(variance + eps)).to(original)


def rotate_pairwise(hidden: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive coordinate pairs, one angle per pair.

    Orthogonal by construction, so the transport preserves ``||z||`` exactly
    for any angle the optimizer reaches.
    """
    pairs = hidden.reshape(*hidden.shape[:-1], -1, 2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    first, second = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (first * cos - second * sin, first * sin + second * cos), dim=-1
    )
    return rotated.reshape(hidden.shape)


class PostNormCentralRecurrence(nn.Module):
    """``z_h = R z_(h-1) + O(read(q_h, S_h))``, post-normalized for the decoder.

    The recurrence stays affine, so it factors into a prefix scan; the
    normalization is applied to the finished tape.
    """

    def __init__(
        self,
        width: int,
        *,
        heads: int = 8,
        key_dim: int = 41,
        initial_frequency_range: float = 0.05,
        hidden_phase_init: str = "zeros",
        rope_horizons: int = 16,
        rope_turns_min: float = 1.0 / 16.0,
        rope_turns_max: float = 1.0,
        memory_decay: bool = False,
        memory_decay_init: float = 0.95,
        memory_decay_max: float = 0.999,
        memory_postnorm: bool = False,
        read_normalization: str = "none",
        qkv_prenorm: bool = False,
        root_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if width < 2 or width % 2:
            raise ValueError("width must be positive and even")
        if heads < 1 or key_dim < 1:
            raise ValueError("heads and key_dim must be positive")
        if width % (2 * heads):
            raise ValueError(
                "width must be divisible by 2*heads so the read fills it"
            )

        self.width = int(width)
        self.heads = int(heads)
        self.key_dim = int(key_dim)
        # No bottleneck: the read is exactly as wide as the latent it writes.
        self.value_dim = width // (2 * heads)
        self.read_width = 2 * self.heads * self.value_dim
        assert self.read_width == self.width
        self.eps = float(eps)

        # Multi-value attention: one query and one key for the whole layer,
        # values per head.  Packed into a single projection.
        self.query_features = 2 * self.key_dim
        self.key_features = 2 * self.key_dim
        self.value_features = 2 * self.heads * self.value_dim
        self.qkv = nn.Linear(
            width,
            self.query_features + self.key_features + self.value_features,
            bias=False,
        )
        self.output = nn.Linear(self.read_width, width, bias=False)

        # Latent transport: one angle per coordinate pair.
        #
        # ``zeros`` is the inherited initialization and is degenerate: theta=0
        # makes R the identity, so c_1 = ... = c_H = z_0 and every horizon is
        # handed the same forcing state.  Every write is then the same W, the
        # memory accumulates coherently (||S_h|| ~ h rather than ~sqrt(h)), and
        # every horizon's contribution to dL/dtheta arrives in phase.
        #
        # ``rope`` spreads the pairs over a log-spaced band instead, which is
        # what c_h = R^h z_0 actually is: RoPE applied to a step index.  The
        # band is set in *turns completed by horizon H* -- the slowest pair
        # turns ``rope_turns_min`` of a circle and so mostly preserves z_0, the
        # fastest turns ``rope_turns_max`` and resolves adjacent horizons.
        # Past one turn the angle h*theta wraps and horizons alias, so
        # ``rope_turns_max = 1`` is the useful ceiling.  Signs alternate so the
        # two rotation directions are both represented.
        if hidden_phase_init == "zeros":
            hidden_angles = torch.zeros(width // 2)
        elif hidden_phase_init == "rope":
            if rope_horizons < 1:
                raise ValueError("rope_horizons must be positive")
            if not 0 < rope_turns_min <= rope_turns_max:
                raise ValueError("rope turn band must satisfy 0 < min <= max")
            magnitudes = torch.logspace(
                math.log10(2.0 * math.pi * rope_turns_min / rope_horizons),
                math.log10(2.0 * math.pi * rope_turns_max / rope_horizons),
                width // 2,
            )
            signs = torch.ones(width // 2)
            signs[1::2] = -1.0
            hidden_angles = magnitudes * signs
        else:
            raise ValueError(
                f"unknown hidden_phase_init: {hidden_phase_init}"
            )
        self.hidden_phase_init = str(hidden_phase_init)
        self.hidden_phase = nn.Parameter(hidden_angles)
        # Memory transport: distinct frequencies per (head, key) entry.
        frequencies = torch.linspace(
            -initial_frequency_range,
            initial_frequency_range,
            self.heads * self.key_dim,
        ).reshape(self.heads, self.key_dim)
        self.memory_phase = nn.Parameter(frequencies)

        # Variant 1: contract the memory instead of transporting it on the
        # unit circle.  ``lambda = max * sigmoid(logit)`` stays strictly below
        # ``max`` with a gradient everywhere, so the bound B/(1-lambda) cannot
        # be lost by the parameter drifting to one.
        self.memory_decay = bool(memory_decay)
        self.memory_decay_max = float(memory_decay_max)
        if self.memory_decay:
            if not 0 < memory_decay_max < 1:
                raise ValueError("memory_decay_max must lie in (0, 1)")
            if not 0 < memory_decay_init < memory_decay_max:
                raise ValueError("memory_decay_init must be below the maximum")
            ratio = memory_decay_init / memory_decay_max
            self.memory_decay_logit = nn.Parameter(
                torch.full(
                    (self.heads, self.key_dim), math.log(ratio / (1.0 - ratio))
                )
            )
        else:
            self.register_parameter("memory_decay_logit", None)

        # Variant 2: post-normalize the memory itself.  Applied per head over
        # its (key, value) block rather than as one scalar over every head, so
        # one head's growth cannot attenuate another head's read -- the
        # coupling that the global ||S||_F division introduced.
        self.memory_postnorm = bool(memory_postnorm)

        # Switches that exist only so the predecessor structure
        # (``train_time_varying_scan.py``) can be reproduced inside this file
        # and compared one variable at a time.  All three default to the new
        # structure, so leaving them alone changes nothing.
        # "frobenius" is the predecessor's division by the whole memory's
        # matrix norm.  "output" is the RetNet/GLA placement instead: leave the
        # read linear in S and normalize the *result* per head, which is where
        # that literature puts its GroupNorm.  It pairs with a contracting
        # memory -- decay bounds the state, the output norm conditions the
        # read -- and avoids what the Frobenius division costs: no projection
        # onto the sphere in S (write magnitude stays learnable), no single
        # scalar coupling every head, and no 1/||S_h|| factor that attenuates
        # later horizons more than earlier ones.
        if read_normalization not in ("none", "frobenius", "output"):
            raise ValueError(f"unknown read_normalization: {read_normalization}")
        self.read_normalization = str(read_normalization)
        self.qkv_prenorm = bool(qkv_prenorm)
        # The predecessor leaves the recurrent root raw
        # (``normalize_initial_recurrent_root=False``) and relies on the QKV
        # prenorm instead, so the latent baseline ``R^h z_0`` carries the
        # encoder's own scale.
        self.root_norm = bool(root_norm)

    def project(self, tape: torch.Tensor):
        """Q/K/V from the forcing tape."""
        if self.qkv_prenorm:
            tape = rms_normalize(tape, self.eps)
        projected = self.qkv(tape)
        query_raw, key_raw, value_raw = projected.split(
            (self.query_features, self.key_features, self.value_features),
            dim=-1,
        )

        def to_complex(raw: torch.Tensor, heads: int, features: int):
            real, imaginary = raw.chunk(2, dim=-1)
            shape = (*raw.shape[:-1], heads, features)
            return torch.complex(real.reshape(shape), imaginary.reshape(shape))

        query = to_complex(query_raw, 1, self.key_dim).squeeze(-2)
        key = to_complex(key_raw, 1, self.key_dim).squeeze(-2)
        value = to_complex(value_raw, self.heads, self.value_dim)
        return query, key, value

    @property
    def memory_decay_magnitude(self) -> torch.Tensor | None:
        if self.memory_decay_logit is None:
            return None
        return self.memory_decay_max * torch.sigmoid(self.memory_decay_logit)

    def normalize_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """Per-head RMS normalization over the ``(key, value)`` block."""
        power = memory.abs().square().mean(dim=(-2, -1), keepdim=True)
        return memory * torch.rsqrt(power + self.eps)

    def memory_states(self, key: torch.Tensor, value: torch.Tensor):
        """``S_h = m S_(h-1) + k_h v_h^dagger`` for every prefix.

        With a unit-modulus multiplier and no memory post-normalization the
        recurrence is affine with an ``h``-independent multiplier, so it closes
        in the rotating frame: divide the writes by ``U^i``, accumulate,
        multiply the running sum by ``U^h``.  The inverse frame sits on the
        unit circle there, so it never amplifies.

        Decay or memory post-normalization break that closed form -- the first
        makes the inverse frame grow like ``lambda^-i``, the second is
        nonlinear -- so those run the recurrence out over the horizon axis.
        """
        horizons = key.shape[-2]
        # W_h = k_h v_h^dagger, key shared across heads.
        writes = (
            key[..., None, :, None] * value.conj()[..., None, :]
        )  # [..., H, heads, key, value]

        real_dtype = key.real.dtype
        phase = self.memory_phase.to(real_dtype)
        decay = self.memory_decay_magnitude

        if decay is None and not self.memory_postnorm:
            steps = torch.arange(
                1, horizons + 1, device=key.device, dtype=real_dtype
            ).reshape(-1, 1, 1)
            angles = -phase[None] * steps  # [H, heads, key]
            forward_frame = torch.polar(torch.ones_like(angles), angles)
            framed = forward_frame.conj()[..., None] * writes
            return forward_frame[..., None] * framed.cumsum(dim=-4)

        rotation = torch.polar(torch.ones_like(phase), -phase)
        multiplier = rotation if decay is None else decay.to(real_dtype) * rotation
        state = torch.zeros_like(writes[..., 0, :, :, :])
        states = []
        for step in range(horizons):
            state = multiplier[..., None] * state + writes[..., step, :, :, :]
            if self.memory_postnorm:
                state = self.normalize_memory(state)
            states.append(state)
        return torch.stack(states, dim=-4)

    def read(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Read the memory. Linear unless the predecessor's division is on."""
        measurement = torch.einsum(
            "...k,...hkv->...hv", query.conj(), memory
        ) / math.sqrt(self.key_dim)
        if self.read_normalization == "output":
            # Per head, over its own value block -- one head's memory growth
            # cannot rescale another head's read.
            power = measurement.abs().square().mean(dim=-1, keepdim=True)
            measurement = measurement * torch.rsqrt(power + self.eps)
        flattened = torch.cat(
            (measurement.real.flatten(-2), measurement.imag.flatten(-2)),
            dim=-1,
        )
        if self.read_normalization == "frobenius":
            # One scalar over every head, as the predecessor computes it.
            frobenius = memory.abs().square().sum(dim=(-3, -2, -1)).sqrt()
            flattened = flattened / (frobenius[..., None] + self.eps)
        return flattened

    def forward(self, root: torch.Tensor, horizons: int) -> torch.Tensor:
        """Return the post-normalized state tape ``[..., horizons, width]``."""
        if horizons < 1:
            raise ValueError("horizons must be positive")
        if root.shape[-1] != self.width:
            raise ValueError(f"root width must be {self.width}")

        state = rms_normalize(root, self.eps) if self.root_norm else root

        # Unitary forcing tape c_h = R^h z_0, computed in parallel.
        steps = torch.arange(
            1, horizons + 1, device=root.device, dtype=root.dtype
        ).reshape(-1, 1)
        tape_angles = steps * self.hidden_phase
        tape = rotate_pairwise(
            state.unsqueeze(-2).expand(*state.shape[:-1], horizons, self.width),
            tape_angles,
        )

        query, key, value = self.project(tape)
        memory = self.memory_states(key, value)
        innovations = self.output(
            self.read(query, memory).to(root.dtype)
        )  # [..., horizons, width]

        # Affine latent recurrence z_h = R z_(h-1) + delta_h.  R does not
        # depend on h, so the same rotating frame closes it:
        #   z_h = R^h (z_0 + sum_(i<=h) R^(-i) delta_i)
        # R is orthogonal, so the inverse frame is a rotation and cannot
        # amplify.  One cumulative sum, no sequential pass.
        framed = rotate_pairwise(innovations, -tape_angles)
        accumulated = framed.cumsum(dim=-2) + state.unsqueeze(-2)
        states = rotate_pairwise(accumulated, tape_angles)

        # Everything the decoder sees is post-normalized.
        return rms_normalize(states, self.eps)


# --------------------------------------------------------------------------
# Model, loss, evaluation
# --------------------------------------------------------------------------

class TiedLMHead(nn.Module):
    """The ordinary readout: a final norm, then the tied unembedding.

    ``K1DecoderAblationLM``'s ``simplex-raw-tied`` mode multiplies an
    *unnormalized* hidden by a fixed 16.0 and calls that the logits.  There is
    no final norm anywhere on that path, so whatever scale the inverse decoder
    produces goes straight into the softmax: measured at initialization the
    logits reach |60-70| and cross entropy sits at 41.8 (width 1264) and 44.6
    (width 3872) against ``ln(256) = 5.545`` for a uniform prediction.  A model
    that knows nothing should score 5.545, not 8x worse.

    This is the standard arrangement instead -- RMSNorm with a learned gain,
    then ``@ E^T``, no fixed multiplier.
    """

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(
        self, hidden: torch.Tensor, embedding_weight: torch.Tensor
    ) -> torch.Tensor:
        normalized = rms_normalize(hidden, self.eps) * self.gain
        return normalized.float() @ embedding_weight.float().T


def reinitialize_encoder(model: K1DecoderAblationLM, scheme: str) -> None:
    """Re-init the reversible encoder's linears.

    ``shift_lm.RevBlock`` initializes every linear at ``std=0.02`` regardless
    of ``fan_in``.  That is GPT-2's constant, and it stops being the right
    scale once the width moves: at ``d_half=632`` the fan-in scale is 0.0398,
    at ``d_half=1936`` it is 0.0227, so a fixed 0.02 makes the residual
    branches relatively *stronger* as the model widens.  Measured at
    initialization, the branch RMS grows 3.4-3.8x between width 1264 and 3872
    where ``sqrt`` scaling predicts 1.75x, and the encoder output inherits it.

    That matters twice over here, because the decoder inverts these same
    blocks: a stronger residual branch makes ``x = y - f(...)`` amplify more,
    and the tape it inverts is a *predicted* latent that never sits exactly on
    the encoder's manifold.
    """
    if scheme == "gpt2":
        return
    if scheme != "fan-in":
        raise ValueError(f"unknown encoder init scheme: {scheme}")
    for module in model.encoder.blocks.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, std=1.0 / math.sqrt(module.weight.shape[1])
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def make_model(args: argparse.Namespace) -> K1DecoderAblationLM:
    model = K1DecoderAblationLM(
        VOCABULARY,
        width=args.width,
        encoder_blocks=args.encoder_blocks,
        decoder_mode="exact-inverse",
        head_mode="simplex-raw-tied",
        simplex_logit_scale=args.simplex_logit_scale,
    )
    model.operator = nn.Identity()
    model.complex_self_prediction = PostNormCentralRecurrence(
        args.width,
        heads=args.heads,
        key_dim=args.key_dim,
        initial_frequency_range=args.initial_frequency_range,
        hidden_phase_init=args.hidden_phase_init,
        rope_horizons=args.horizons,
        rope_turns_min=args.rope_turns_min,
        rope_turns_max=args.rope_turns_max,
        memory_decay=args.memory_decay,
        memory_decay_init=args.memory_decay_init,
        memory_decay_max=args.memory_decay_max,
        memory_postnorm=args.memory_postnorm,
        read_normalization=args.read_normalization,
        qkv_prenorm=args.qkv_prenorm,
        root_norm=not args.no_root_norm,
    )
    reinitialize_encoder(model, args.encoder_init)
    model.lm_head = TiedLMHead(args.width) if args.head == "standard" else None
    return model


def horizon_logits(model: K1DecoderAblationLM, window: torch.Tensor, args):
    required = args.context + args.horizons
    if window.ndim != 2 or window.shape[1] != required:
        raise ValueError(f"window must have shape [batch,{required}]")

    positions = torch.arange(args.context, device=window.device)
    prefix_encoded, _ = model.encode(window[:, : args.context], positions)
    anchors = anchor_indices_for(args.context, args.anchor_stride, window.device)
    roots = prefix_encoded.index_select(1, anchors)

    read_states = model.complex_self_prediction(roots, args.horizons)

    offsets = torch.arange(
        1, args.horizons + 1, device=window.device, dtype=torch.long
    )
    target_positions = anchors[:, None] + offsets[None, :]
    targets = window.index_select(1, target_positions.reshape(-1)).reshape(
        window.shape[0], anchors.numel(), args.horizons
    )

    # The tape is already post-normalized, so it goes to the decoder as is.
    decoded = model.exact_inverse_selected_decode_tape_states(
        prefix_encoded, read_states, positions, anchors
    )
    if model.lm_head is not None:
        return model.lm_head(decoded, model.embedding_weight), targets
    return model.token_logits(decoded), targets


def horizon_loss(model, window: torch.Tensor, args) -> torch.Tensor:
    logits, targets = horizon_logits(model, window, args)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
    )


@torch.no_grad()
def horizon_metrics(model, windows: torch.Tensor, microbatch: int, args):
    was_training = model.training
    model.eval()
    nll_sum = torch.zeros(args.horizons, dtype=torch.float64)
    correct_sum = torch.zeros(args.horizons, dtype=torch.float64)
    labels = 0
    try:
        for begin in range(0, windows.shape[0], microbatch):
            logits, targets = horizon_logits(
                model, windows[begin : begin + microbatch], args
            )
            token_nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            nll_sum += token_nll.double().sum(dim=(0, 1)).cpu()
            correct_sum += (
                logits.argmax(dim=-1).eq(targets).double().sum(dim=(0, 1)).cpu()
            )
            labels += targets.shape[0] * targets.shape[1]
    finally:
        model.train(was_training)

    horizon_nll = nll_sum / labels
    horizon_accuracy = correct_sum / labels
    metrics = {
        "block_nll": float(horizon_nll.mean()),
        "block_accuracy": float(horizon_accuracy.mean()),
    }
    for index in range(args.horizons):
        metrics[f"h{index + 1}_nll"] = float(horizon_nll[index])
        metrics[f"h{index + 1}_accuracy"] = float(horizon_accuracy[index])
    return metrics


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def lr_at(index: int, steps: int, *, peak_lr: float, warmup: int) -> float:
    if index < warmup:
        return peak_lr * (index + 1) / warmup
    progress = (index - warmup) / max(1, steps - warmup)
    return peak_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--head",
        choices=("simplex-raw-tied", "standard"),
        default="simplex-raw-tied",
        help="'standard' is a final RMSNorm with a learned gain followed by "
        "the tied unembedding, with no fixed logit multiplier",
    )
    parser.add_argument(
        "--encoder-init",
        choices=("gpt2", "fan-in"),
        default="gpt2",
        help="'gpt2' keeps shift_lm's constant std=0.02; 'fan-in' rescales "
        "every encoder linear to 1/sqrt(fan_in) so the residual branches stop "
        "growing relative to the stream as the width increases",
    )
    parser.add_argument("--width", type=int, default=1264)
    parser.add_argument("--encoder-blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=41)
    parser.add_argument("--simplex-logit-scale", type=float, default=16.0)
    parser.add_argument("--initial-frequency-range", type=float, default=0.05)

    # Predecessor-reproduction switches. The three together turn this file
    # into train_time_varying_scan.py's structure.
    parser.add_argument(
        "--read-normalization",
        choices=("none", "frobenius", "output"),
        default="none",
        help="'frobenius' divides the read by ||S||_F (the predecessor); "
        "'output' leaves the read linear and RMS-normalizes it per head "
        "afterwards, the RetNet/GLA placement, meant to pair with "
        "--memory-decay",
    )
    parser.add_argument("--qkv-prenorm", action="store_true")
    parser.add_argument("--no-root-norm", action="store_true")

    parser.add_argument(
        "--hidden-phase-init",
        choices=("zeros", "rope"),
        default="zeros",
        help="'zeros' leaves R at the identity, so every horizon starts from "
        "the same forcing state; 'rope' log-spaces the pair frequencies over "
        "a band measured in turns completed by the last horizon.",
    )
    parser.add_argument("--rope-turns-min", type=float, default=1.0 / 16.0)
    parser.add_argument("--rope-turns-max", type=float, default=1.0)
    parser.add_argument(
        "--memory-decay",
        action="store_true",
        help="contract S by a learned per-(head,key) lambda instead of "
        "transporting it on the unit circle",
    )
    parser.add_argument("--memory-decay-init", type=float, default=0.95)
    parser.add_argument("--memory-decay-max", type=float, default=0.999)
    parser.add_argument(
        "--memory-postnorm",
        action="store_true",
        help="RMS-normalize S per head after every write",
    )

    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizons", type=int, default=16)
    parser.add_argument("--anchor-stride", type=int, default=16)

    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--schedule-steps", type=int, default=6000)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--eval-microbatch", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument(
        "--record-dir", type=Path, default=Path("outputs/postnorm_recurrence")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/postnorm_recurrence/checkpoints"),
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()

    if args.width < 2 or args.width % (2 * args.heads):
        parser.error("--width must be divisible by 2*--heads")
    if args.batch % args.microbatch:
        parser.error("--batch must be divisible by --microbatch")
    if args.horizons < 1:
        parser.error("--horizons must be positive")
    if not torch.cuda.is_available():
        parser.error("this trainer requires CUDA")
    return args


def main() -> None:
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    training, validation = memmap("train"), memmap("validation")
    window_length = args.context + args.horizons
    validation_starts = torch.randint(
        0,
        len(validation) - window_length,
        (args.eval_examples,),
        generator=torch.Generator().manual_seed(args.seed + 999),
    )
    validation_windows = windows_of_length(
        validation, validation_starts, window_length
    )

    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args).cuda().train()
    central = model.complex_self_prediction
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    with (args.record_dir / "config.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "value"))
        recorded = {
            key: value
            for key, value in vars(args).items()
            if key not in ("record_dir", "output_dir")
        }
        recorded.update(
            {
                "value_dim": central.value_dim,
                "read_width": central.read_width,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "read_normalization": central.read_normalization,
                "recurrence_normalization": "post",
                "root_normalization": "rms" if central.root_norm else "none",
                "qkv_prenorm": "rms" if central.qkv_prenorm else "none",
                "shared_kv_heads": True,
                "memory_normalization": (
                    "post-per-head" if central.memory_postnorm else "none"
                ),
                "memory_transport": (
                    "contracting" if central.memory_decay else "unitary"
                ),
                "hidden_phase_abs_min": float(
                    central.hidden_phase.detach().abs().min()
                ),
                "hidden_phase_abs_max": float(
                    central.hidden_phase.detach().abs().max()
                ),
            }
        )
        for key in sorted(recorded):
            writer.writerow((key, recorded[key]))

    metric_names = ["block_nll", "block_accuracy"] + [
        f"h{index + 1}_{name}"
        for index in range(args.horizons)
        for name in ("nll", "accuracy")
    ]
    fields = (
        ["step", "train_ce", "gradient_norm", "phase_gradient_norm", "lr"]
        + [f"val_{name}" for name in metric_names]
        + ["wall_s", "eta_s", "peak_vram_bytes"]
    )

    print(
        f"width={args.width} heads={args.heads} key_dim={args.key_dim} "
        f"value_dim={central.value_dim} read_width={central.read_width} "
        f"params={parameters:,} trainable={trainable:,} "
        f"horizons={args.horizons} peak_lr={args.peak_lr}",
        flush=True,
    )

    data_generator = torch.Generator().manual_seed(args.seed)
    best_nll = float("inf")
    started = time.time()

    with (args.record_dir / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()

        initial = horizon_metrics(
            model, validation_windows, args.eval_microbatch, args
        )
        print(
            f"step=0/{args.steps} block_nll={initial['block_nll']:.4f} "
            f"h1={initial['h1_nll']:.4f} "
            f"h{args.horizons}={initial[f'h{args.horizons}_nll']:.4f}",
            flush=True,
        )

        for index in range(args.steps):
            lr = lr_at(
                index, args.schedule_steps, peak_lr=args.peak_lr, warmup=args.warmup
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            window = sampled_windows(
                training, args.batch, data_generator, window_length
            )
            optimizer.zero_grad(set_to_none=True)
            micro_windows = window.split(args.microbatch)
            train_ce = 0.0
            for micro_window in micro_windows:
                micro_loss = horizon_loss(model, micro_window, args)
                if not bool(torch.isfinite(micro_loss)):
                    raise RuntimeError(f"non-finite loss at step {index + 1}")
                (micro_loss / len(micro_windows)).backward()
                train_ce += float(micro_loss.detach()) / len(micro_windows)

            phase_gradient_norm = float(central.hidden_phase.grad.norm())
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            )
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient at step {index + 1}")
            optimizer.step()
            step = index + 1

            report = (
                step in (1, 50)
                or step % args.report_every == 0
                or step == args.steps
            )
            if not report:
                continue

            metrics = horizon_metrics(
                model, validation_windows, args.eval_microbatch, args
            )
            wall = time.time() - started
            eta = (wall / step) * (args.steps - step)
            writer.writerow(
                {
                    "step": step,
                    "train_ce": train_ce,
                    "gradient_norm": gradient_norm,
                    "phase_gradient_norm": phase_gradient_norm,
                    "lr": lr,
                    **{f"val_{name}": metrics[name] for name in metric_names},
                    "wall_s": wall,
                    "eta_s": eta,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            handle.flush()
            print(
                f"step={step:4d}/{args.steps} ce={train_ce:.4f} "
                f"gnorm={gradient_norm:.4f} pgnorm={phase_gradient_norm:.4f} "
                f"block_nll={metrics['block_nll']:.4f} "
                f"h1={metrics['h1_nll']:.4f} "
                f"h{args.horizons}={metrics[f'h{args.horizons}_nll']:.4f} "
                f"elapsed={wall/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )

            if args.checkpoint_every and (
                step % args.checkpoint_every == 0 or step == args.steps
            ):
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "metrics": metrics,
                        "args": vars(args),
                    },
                    args.output_dir / f"step{step:05d}.pt",
                )
            if metrics["block_nll"] < best_nll:
                best_nll = metrics["block_nll"]
                torch.save(
                    {"model": model.state_dict(), "step": step, "metrics": metrics},
                    args.output_dir / "best.pt",
                )

    print(f"done best_val_block_nll={best_nll:.4f}", flush=True)


if __name__ == "__main__":
    main()
