# Step-10071 generation-quality interpretation

## Bottom line

At three epochs, only the H1 causally re-anchored sampler produces locally
language-shaped output.  It has learned byte/word morphology, punctuation,
short English collocations, and WikiText formatting.  It has not learned
reliable sentence-level syntax, factual continuation, or discourse coherence.

The H1 greedy decoder is substantially worse than the learned distribution:
it falls into a small set of repeated Wikipedia-like phrases.  The H16 block
path cannot yet generate language.  Greedy H16 is almost entirely spaces;
top-p H16 avoids the single-byte collapse but produces uncorrelated letter
streams rather than words and sentences.

## Why the earlier report was incomplete

The prior audit used eight greedy samples and one 128-byte aligned-accuracy
number.  That mixed three different facts:

1. exact reference agreement vanishes quickly after the first alternative
   byte, even for a valid continuation;
2. greedy decoding can expose mode collapse that is not representative of the
   whole probability distribution;
3. H16 can obtain an apparently high byte-match rate by predicting spaces.

This audit separates all three on 64 fixed validation prompts.

## Quantitative summary

| Continuation | Byte entropy | Space | Repeated 4-grams | Word type/token | Strict UTF-8 | Mean longest same-byte run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gold | 4.581 bits | 18.66% | 4.79% | 0.914 | 100% | 1.95 |
| H1 greedy | 4.029 bits | 22.57% | 53.38% | 0.478 | 100% | 2.05 |
| H1 top-p 0.9, T=0.8 | 4.278 bits | 19.79% | 9.11% | 0.806 | 100% | 1.80 |
| H16 greedy | 0.469 bits | 94.60% | 89.16% | 0.912* | 3.13% | 76.81 |
| H16 top-p 0.9, T=0.8 | 3.689 bits | 26.88% | 1.19% | 0.938* | 100% | 3.61 |

The starred word-diversity values are misleading.  Sparse accidental letter
runs in H16 are counted as distinct “words” by a regular expression even when
the text is not lexical English.  The preserved samples are required for that
interpretation.

H1 top-p is close to gold in coarse byte marginals: its space fraction differs
by only 1.12 percentage points, alphabetic fraction by 0.38 points, and every
sample is valid UTF-8.  Its repeated 4-gram rate is still almost twice gold and
its lexical diversity is lower.  More importantly, realistic marginals do not
produce semantic coherence by themselves.

## H1: what is genuinely working

The teacher-forced H1 NLL is 1.5191 (byte perplexity about 4.57) with 56.35%
accuracy.  On the 64 free-generation prompts, the first greedy byte is correct
59.38% of the time.  The model often handles a tokenization boundary and its
immediate grammar:

- prompt ending in `Richar` continues with `d`;
- prompt ending in `conti` continues with `nued`;
- prompt ending in `...war against t` continues with `he`;
- a prompt ending in `animal` continues with `s ,` under sampling.

This is real shallow language competence, not just spaces or a fixed output.
H1 top-p continuations contain recognizable words, punctuation, section
headers, and short grammatical fragments throughout 128 bytes.

What fails is composition beyond those fragments.  Representative sampled
continuations include:

```text
the group and choland , or support , and and could be decade ...
s , but did not constitution and and separation at the storm ...
he into a selected by conventional , and the riser itself ...
```

They are visibly English-shaped, but exhibit invalid selectional relations,
agreement errors, invented words, and abrupt topic changes.  Across the twelve
stored examples, none preserves the prompt's concrete people, place, event, or
argument for a complete sentence.  This is best described as local
morphology/collocation plus Wikipedia-style surface imitation, not coherent
sentence generation.

## H1 greedy: a distinct mode-collapse problem

Greedy output is not space collapse, but it repeatedly chooses a few generic
high-probability continuations such as `the southern`, `the second`,
`the compound`, and synthetic section headings.  Its repeated-4-gram rate is
53.38% versus 4.79% in gold, and repeated word-bigram rate is 35.76% versus
0.70%.  One sample ends in a 29-byte identical run, while others loop the same
noun phrases.

Top-p reduces repeated 4-grams to 9.11% and repeated word bigrams to 1.16%, so
the learned H1 distribution is materially better than its argmax trajectory.
It is still semantically poor; sampling fixes mode repetition, not reasoning
or long syntax.

## Exposure with generated context

Aligned byte accuracy is useful only for locating the loss of the literal
reference path:

| Region | H1 greedy | H1 top-p |
| --- | ---: | ---: |
| offsets 1-4 | 40.63% | 32.42% |
| offsets 5-8 | 11.33% | 10.94% |
| offsets 9-16 | 6.45% | 5.08% |
| offsets 17-32 | 8.30% | 10.74% |
| offsets 65-128 | 7.76% | 6.86% |

The mean exact reference-prefix length is 1.55 bytes for greedy and 1.27 for
top-p; the maximum is seven bytes.  This does not mean all later language is
wrong, since an alternative early word shifts every later byte.  It does show
that the checkpoint's strong teacher-forced H1 result only controls the first
few generated bytes before it must operate on its own changed context.

## H16: no sentence-generation competence yet

The all-space baseline matches 18.66% of the 128 reference bytes.  H16 greedy
matches 19.13%, only 0.46 percentage points more.  Its 94.60% space rate,
0.469-bit entropy, mean identical-byte run of 76.8, and 96.52% mean
cross-prompt positional mode rate establish literal collapse.

The collapse happens inside each 16-byte prediction block:

- block phase 1 is 54.49% spaces across successive blocks;
- phase 4 is 94.34% spaces;
- phase 7 is 99.41% spaces;
- phases 9 through 16 are exactly 100% spaces.

The very first H16 byte is identical in mechanism to H1 and obtains 59.38%
accuracy.  Across offsets 1-4 it still obtains 41.02%.  At offsets 9-16 its
18.55% match is exactly the all-space baseline for that region.  Thus the
apparently respectable H16 byte accuracy is an H1-like block entrance followed
by blank completion, not a sixteen-byte plan.

H16 top-p proves that its logits are not delta distributions on spaces: entropy
rises to 3.689 bits and the space rate falls to 26.88%.  Nevertheless the
actual outputs are strings such as:

```text
hoe dtsnlo sets i  t tah h hhe  pno  iiu ...
oopi sriifornd the cintstsot sarees ...
```

These are diverse byte sequences but not language.  Sampling each horizon's
marginal cannot recover the missing within-block joint correlation.  Therefore
neither H16 decoding mode has meaningful sentence quality at epoch 3.

## Exact quality statement

The defensible description at this checkpoint is:

- **H1 conditional:** good immediate byte, split-word, and short-collocation
  modeling.
- **H1 sampled continuation:** visibly language-like at the surface, but
  syntactically fragile and semantically unrelated after a few words.
- **H1 greedy continuation:** strong generic-phrase repetition.
- **H16 continuation:** no usable language generation; greedy is blank collapse
  and sampling is character soup.

This is stronger than a random or unigram byte model, but it is not yet a model
that writes coherent sentences.  The limitation is not captured by H1 NLL
alone and cannot be repaired at inference merely by switching greedy to
sampling for the H16 branch.

## Artifacts

- `metrics.tsv`: registered aggregate metrics for all modes and controls
- `position_buckets.tsv`: early/late free-running behavior
- `h16_phase.tsv`: within-block collapse by phase
- `samples.tsv`: reversible hexadecimal output and escaped readable text
