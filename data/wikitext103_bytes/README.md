# WikiText-103 UTF-8 byte stream

This directory is produced by `prepare_wikitext103_bytes.py` from
`Salesforce/wikitext`, configuration `wikitext-103-raw-v1`. Every nonempty
text row is encoded literally as UTF-8 after appending one newline. Byte value
`0..255` is the token ID; there are no tokenizer parameters or special-token
IDs.

The training stream is capped at 55,000,000 bytes to match the BPE producer's
number of stored training positions. Validation is complete. The test split
is deliberately not materialized for model selection.

Generated binary streams remain ignored by Git. `manifest.tsv` records their
dtype, size, token count, and SHA-256; `provenance.tsv` records the source,
dataset fingerprints, library version, and transformation. Re-run the
producer before training if the binary files are absent.

The upstream Salesforce WikiText dataset is distributed under Creative
Commons Attribution-ShareAlike terms. These derived byte streams remain
subject to those terms.
