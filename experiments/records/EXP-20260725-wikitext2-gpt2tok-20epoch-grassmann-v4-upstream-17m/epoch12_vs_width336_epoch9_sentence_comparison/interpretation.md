# Grassmann epoch 12 versus latent epoch 9

## Scope

This is an unmatched-epoch qualitative comparison over eight fixed
WikiText-2 validation prompts. Grassmann used greedy AR; the width-336 latent
model used target-free prior-argmax block 4.

## Sentence quality

Grassmann has clearly better local grammar and phrase formation. Its
continuations usually resemble English sentences, but they repeatedly fall
into generic templates such as `the United States`, `the first season`, and
`the 2nd Battalion`, with occasional topic drift.

The latent model reuses prompt-domain vocabulary more strongly and avoids
whole-sample collapse, but grammatical continuity is substantially worse.
Frequent doubled function words (`the the`, `was was`), malformed BPE
fragments (`theth`, `Newth`, `Armys`), and short block-local constructions
remain visible.

## Aggregate signals

- Grassmann versus latent distinct-2: `0.452` versus `0.550`
- distinct-4: `0.635` versus `0.900`
- collapse rate: `0.25` versus `0.00`
- immediate repetition: `0.050` versus `0.131`
- prompt-vocabulary reuse: `0.363` versus `0.510`
- reference-vocabulary overlap: `0.471` versus `0.684`
- reference-token accuracy: `0.0293` versus `0.0332`

The small reference-accuracy difference is not meaningful with eight
samples. Diversity and overlap favor the latent output, while immediate
repetition and direct reading favor Grassmann's local fluency.

## Speed

For the registered eight prompts and 64-token continuations, Grassmann AR
took `0.902s`; latent block-4 took `0.334s`. In this shallow width-336 model,
block-4 was about `2.7x` faster. This is a wall-clock observation for one
batch regime, not a general throughput benchmark.

## Bottom line

Grassmann is the better sentence generator at these checkpoints because its
syntax is materially more coherent. The latent model demonstrates the
intended speed benefit and stronger topic/diversity retention, but has not
converted those advantages into fluent prose.
