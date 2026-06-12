"""Calibration data loading utilities."""

import os
import random
import sys
from itertools import islice
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm


def get_wikitext2(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    cache_path = f"calib_wikitext2_{nsamples}_{seqlen}_{seed}_v{tokenizer.vocab_size}.pt"
    if cache_dir:
        cache_path = os.path.join(cache_dir, cache_path)
    if os.path.exists(cache_path):
        print("Loading calib from file...", file=sys.stderr)
        return torch.load(cache_path, weights_only=True)

    print("Loading wikitext2 dataset...", file=sys.stderr)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(traindata["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]

    random.seed(seed)
    calib = []
    for _ in tqdm(range(nsamples), desc="Sampling calibration data"):
        i = random.randint(0, enc.shape[0] - seqlen - 1)
        calib.append(enc[i : i + seqlen].unsqueeze(0))

    calib = torch.cat(calib, dim=0)
    torch.save(calib, cache_path)
    return calib


def get_c4(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    cache_path = f"calib_c4_{nsamples}_{seqlen}_{seed}_v{tokenizer.vocab_size}.pt"
    if cache_dir:
        cache_path = os.path.join(cache_dir, cache_path)
    if os.path.exists(cache_path):
        print("Loading calib from file...", file=sys.stderr)
        return torch.load(cache_path, weights_only=True)

    print("Loading C4 dataset...", file=sys.stderr)
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True)
    traindata = list(islice(traindata, 356445))

    random.seed(seed)
    calib = []
    for _ in tqdm(range(nsamples), desc="Sampling calibration data"):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        calib.append(trainenc.input_ids[:, i : i + seqlen])

    calib = torch.cat(calib, dim=0)
    torch.save(calib, cache_path)
    return calib


def get_calibration_data(
    dataset: str,
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    if dataset == "wikitext2":
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, cache_dir)
    elif dataset == "c4":
        return get_c4(nsamples, seed, seqlen, tokenizer, cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'c4' or 'wikitext2'.")
