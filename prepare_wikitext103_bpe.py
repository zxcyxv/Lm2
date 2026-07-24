"""Prepare a reproducible WikiText-103 raw BPE-8K binary corpus."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

ROOT = Path("data/wikitext103")
CACHE = Path("data/cache")
VOCAB = 8192
TRAIN_LIMIT = 55_000_000  # leaves room for exactly 50M source tokens plus windows


def nonempty_text(split):
    for text in split["text"]:
        if text:
            yield text + "\n"


def build_tokenizer(train):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=2,
                                  special_tokens=["<unk>", "<eos>"])
    tokenizer.train_from_iterator(nonempty_text(train), trainer=trainer,
                                  length=train.num_rows)
    tokenizer.save(str(ROOT / "tokenizer.json"))
    return tokenizer


def encode_split(tokenizer, split, path, limit=None):
    eos = tokenizer.token_to_id("<eos>"); chunks=[]; count=0; batch=[]
    def flush(values):
        nonlocal count
        if not values: return False
        for encoding in tokenizer.encode_batch(values):
            ids = encoding.ids + [eos]
            if limit is not None and count + len(ids) > limit:
                ids = ids[:limit-count]
            if ids: chunks.append(np.asarray(ids, dtype=np.uint16)); count += len(ids)
            if limit is not None and count >= limit: return True
        return False
    for text in nonempty_text(split):
        batch.append(text)
        if len(batch) == 1024:
            if flush(batch): break
            batch=[]
    else: flush(batch)
    array=np.concatenate(chunks); array.tofile(path)
    return len(array)


def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(8<<20),b""): digest.update(block)
    return digest.hexdigest()


def main():
    ROOT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    dataset=load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",cache_dir=str(CACHE))
    tokenizer=build_tokenizer(dataset["train"])
    records=[]
    for split,limit in (("train",TRAIN_LIMIT),("validation",None),("test",None)):
        path=ROOT/f"{split}.bin"; count=encode_split(tokenizer,dataset[split],path,limit)
        records.append((split,str(path),count,path.stat().st_size,sha256(path)))
        print(records[-1],flush=True)
    tok=ROOT/"tokenizer.json"
    with (ROOT/"manifest.tsv").open("w",newline="") as handle:
        writer=csv.writer(handle,delimiter="\t");writer.writerow(("split","path","tokens","bytes","sha256"));writer.writerows(records)
        writer.writerow(("tokenizer",str(tok),tokenizer.get_vocab_size(),tok.stat().st_size,sha256(tok)))


if __name__=="__main__":main()
