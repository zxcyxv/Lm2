"""Generation quality for the h_A-based (no query) skew-K + learned-noise
checkpoint: sequential (hard-token, 128 tokens, with repetition stats) vs
parallel (direct-K^5, 5 tokens), on the same prompts used throughout this
session, for direct comparison against the query-based checkpoints' samples.
"""
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from train_k1_ha_skew_learned_noise_mse_ce_13m import (
    SkewOrthogonalOperator, SigmaPredictor, VOCABULARY, WIDTH,
)
from rotlm.models.k1_decoder_ablation import K1DecoderAblationLM
from rotlm.evaluation import SEED, memmap, token_stats as stats, windows

CHECKPOINT = Path("outputs/experiments/EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m/last.pt")
GENERATION_PROMPT = 64
GENERATION_NEW = 64
DIRECT_HORIZON = 5
GENERATION_SAMPLES = 5


def load_model(path: Path) -> tuple[K1DecoderAblationLM, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = K1DecoderAblationLM(
        VOCABULARY, width=WIDTH, encoder_blocks=2,
        decoder_mode="exact-inverse", head_mode="rms-tied", trainable_cosine_scale=True,
    )
    model.operator = SkewOrthogonalOperator(WIDTH)
    model.sigma_predictor = SigmaPredictor(WIDTH)
    model.load_state_dict(payload["model"], strict=True)
    return model.cuda().eval(), payload.get("step")


@torch.inference_mode()
def generate_sequential(model: K1DecoderAblationLM, prompt: torch.Tensor, new_tokens: int) -> torch.Tensor:
    rolling = prompt
    batch = prompt.shape[0]
    for _ in range(new_tokens):
        positions = torch.arange(rolling.shape[1], device=prompt.device)
        encoded, _ = model.encode(rolling, positions)
        h_a = encoded[:, -1]
        operator_state = model.operator(h_a)
        latent = torch.cat((encoded, operator_state[:, None]), dim=1)
        full_positions = torch.arange(rolling.shape[1] + 1, device=prompt.device).unsqueeze(0).expand(batch, -1)
        decoded = model.encoder.decode_hidden(latent, full_positions)
        logits = model.token_logits(decoded[:, -1]).float()
        action = logits.argmax(dim=-1)
        rolling = torch.cat((rolling, action[:, None]), dim=1)
    return rolling[:, prompt.shape[1]:]


@torch.inference_mode()
def generate_parallel(model: K1DecoderAblationLM, prompt: torch.Tensor, horizon: int) -> torch.Tensor:
    batch = prompt.shape[0]
    positions = torch.arange(prompt.shape[1], device=prompt.device)
    encoded, _ = model.encode(prompt, positions)
    h_a = encoded[:, -1]
    chain = [h_a]
    for _ in range(horizon):
        chain.append(model.operator(chain[-1]))
    future = torch.stack(chain[1:], dim=1)
    latent = torch.cat((encoded, future), dim=1)
    full_positions = torch.arange(prompt.shape[1] + horizon, device=prompt.device).unsqueeze(0).expand(batch, -1)
    decoded = model.encoder.decode_hidden(latent, full_positions)
    logits = model.token_logits(decoded[:, -horizon:]).float()
    return logits.argmax(dim=-1)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    tokenizer = Tokenizer.from_file("data/wikitext103/tokenizer.json")
    validation = memmap("validation")
    starts = torch.randint(
        0, len(validation) - GENERATION_PROMPT - GENERATION_NEW - 1, (GENERATION_SAMPLES,),
        generator=torch.Generator().manual_seed(SEED + 2024),
    )
    window = windows(validation, starts, GENERATION_PROMPT + GENERATION_NEW)
    prompt, reference = window[:, :GENERATION_PROMPT], window[:, GENERATION_PROMPT:]

    model, step = load_model(CHECKPOINT)
    print(f"checkpoint step {step}\n")

    sequential = generate_sequential(model, prompt, GENERATION_NEW)
    parallel = generate_parallel(model, prompt, DIRECT_HORIZON)

    report = ["# h_A-based (no query) skew-K generation quality", ""]
    seq_stats_all = []
    for i in range(GENERATION_SAMPLES):
        seq_text = tokenizer.decode(sequential[i].tolist())
        seq_stat = stats(sequential[i])
        seq_stats_all.append(seq_stat)
        par_text = tokenizer.decode(parallel[i].tolist())
        print(f"=== Sample {i + 1} ===")
        print("PROMPT:   ", tokenizer.decode(prompt[i].tolist()))
        print("REFERENCE:", tokenizer.decode(reference[i].tolist()))
        print(f"sequential(64tok, distinct_1={seq_stat['distinct_1']:.2f} immediate_repeat={seq_stat['immediate_repeat']:.2f}):")
        print("  ", seq_text)
        print(f"parallel(direct-K^5, 5tok):", par_text)
        print()
        report += [
            f"## Sample {i + 1}", "",
            "**Prompt:** " + tokenizer.decode(prompt[i].tolist()), "",
            "**Reference:** " + tokenizer.decode(reference[i].tolist()), "",
            f"**Sequential (64 tok)** (`{seq_stat}`): {seq_text}", "",
            f"**Parallel (direct-K^5, 5 tok):** {par_text}", "",
        ]

    agg = {k: sum(s[k] for s in seq_stats_all) / GENERATION_SAMPLES for k in ("distinct_1", "distinct_2", "immediate_repeat")}
    print(f"Aggregate sequential repetition stats: {agg}")

    out = Path("experiments/records/EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m/generation_quality.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
