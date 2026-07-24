# Tokenized WikiText-103

This directory contains the exact BPE-8192 corpus used by the recorded
experiments. It was produced by `prepare_wikitext103_bpe.py` from
`Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, with seed-free
deterministic tokenizer training and a 55,000,000-token training cap.

`train.bin` is stored as `train.bin.gz` so the largest Git object remains
below GitHub's 100 MiB hard limit. `rotlm.dataset.token_memmap` expands it
atomically on first use and verifies the uncompressed byte count and SHA-256
from `manifest.tsv`. The resulting `train.bin` is ignored by Git.

| File | Purpose |
|---|---|
| `train.bin.gz` | gzip-compressed uint16 training token IDs |
| `validation.bin` | uint16 validation token IDs |
| `test.bin` | uint16 test token IDs; retained but not used during model selection |
| `tokenizer.json` | Hugging Face Tokenizers BPE-8192 model |
| `manifest.tsv` | uncompressed sizes, token counts, and SHA-256 checksums |

## Source and license

WikiText was introduced by Stephen Merity, Caiming Xiong, James Bradbury,
and Richard Socher in *Pointer Sentinel Mixture Models* (2016),
arXiv:1609.07843. The upstream
[Salesforce WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
is distributed under Creative Commons Attribution-ShareAlike terms. These
tokenized derivatives remain subject to the upstream dataset terms; the
software in this repository does not replace or narrow them.
