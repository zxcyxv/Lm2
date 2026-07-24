"""Block-K^2 generation: commit 2 tokens at a time instead of 1 (sequential)
or all H at once (full parallel). Within a block, s1=K(h_a), s2=K(s1) are
decoded jointly with no re-grounding between them (like direct-K^2); between
blocks, the 2 committed tokens are re-embedded and the sequence is
re-encoded from scratch (real re-grounding), exactly like ordinary
sequential rollout but advancing 2 tokens per re-encoding instead of 1.

Compared against pure sequential (1-token blocks) and pure parallel
(one-shot H-token blocks, no re-grounding at all) on the same prompts used
throughout this session, with repetition stats.
"""
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_ha_skew_generation_quality import (
    CHECKPOINT, GENERATION_NEW, GENERATION_PROMPT, GENERATION_SAMPLES,
    generate_parallel, generate_sequential, load_model,
)
from rotlm.evaluation import SEED, memmap, token_stats as stats, windows

BLOCK = 2


@torch.inference_mode()
def generate_block(
    model,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    block: int,
) -> torch.Tensor:
    if block < 1 or new_tokens % block:
        raise ValueError("new_tokens must be divisible by a positive block")
    rolling = prompt
    batch = prompt.shape[0]
    for _ in range(new_tokens // block):
        positions = torch.arange(rolling.shape[1], device=prompt.device)
        encoded, _ = model.encode(rolling, positions)
        h_a = encoded[:, -1]
        future_states = []
        state = h_a
        for _ in range(block):
            state = model.operator(state)
            future_states.append(state)
        future = torch.stack(future_states, dim=1)
        latent = torch.cat((encoded, future), dim=1)
        full_positions = torch.arange(
            rolling.shape[1] + block, device=prompt.device
        ).unsqueeze(0).expand(batch, -1)
        decoded = model.encoder.decode_hidden(latent, full_positions)
        logits = model.token_logits(decoded[:, -block:]).float()
        actions = logits.argmax(dim=-1)
        rolling = torch.cat((rolling, actions), dim=1)
    return rolling[:, prompt.shape[1]:]


def generate_block2(
    model, prompt: torch.Tensor, new_tokens: int
) -> torch.Tensor:
    return generate_block(model, prompt, new_tokens, block=BLOCK)


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

    seq = generate_sequential(model, prompt, GENERATION_NEW)
    block2 = generate_block2(model, prompt, GENERATION_NEW)

    report = ["# Block-K^2 (2 tokens/re-grounding) vs sequential (1 token/re-grounding)", ""]
    seq_stats, block2_stats = [], []
    for i in range(GENERATION_SAMPLES):
        seq_text, block2_text = tokenizer.decode(seq[i].tolist()), tokenizer.decode(block2[i].tolist())
        seq_stat, block2_stat = stats(seq[i]), stats(block2[i])
        seq_stats.append(seq_stat)
        block2_stats.append(block2_stat)
        print(f"=== Sample {i + 1} ===")
        print("PROMPT:   ", tokenizer.decode(prompt[i].tolist()))
        print("REFERENCE:", tokenizer.decode(reference[i].tolist()))
        print(f"sequential (1tok/step, distinct_1={seq_stat['distinct_1']:.2f} immediate_repeat={seq_stat['immediate_repeat']:.2f}):")
        print("  ", seq_text)
        print(f"block-K^2 (2tok/step, distinct_1={block2_stat['distinct_1']:.2f} immediate_repeat={block2_stat['immediate_repeat']:.2f}):")
        print("  ", block2_text)
        print()
        report += [
            f"## Sample {i + 1}", "",
            "**Prompt:** " + tokenizer.decode(prompt[i].tolist()), "",
            "**Reference:** " + tokenizer.decode(reference[i].tolist()), "",
            f"**Sequential (1 tok/step)** (`{seq_stat}`): {seq_text}", "",
            f"**Block-K^2 (2 tok/step)** (`{block2_stat}`): {block2_text}", "",
        ]

    agg_seq = {k: sum(s[k] for s in seq_stats) / GENERATION_SAMPLES for k in ("distinct_1", "distinct_2", "immediate_repeat")}
    agg_block2 = {k: sum(s[k] for s in block2_stats) / GENERATION_SAMPLES for k in ("distinct_1", "distinct_2", "immediate_repeat")}
    print(f"Aggregate sequential: {agg_seq}")
    print(f"Aggregate block-K^2:  {agg_block2}")

    out = Path("experiments/records/EXP-20260723-k1-ha-skew-learned-noise-mse-ce-13m/block2_generation.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
