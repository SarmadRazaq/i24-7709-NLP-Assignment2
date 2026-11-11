# -*- coding: utf-8 -*-
"""Shared paths, corpus loading, vocabulary and plotting helpers for Assignment 2."""
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
EMB_DIR = ROOT / "embeddings"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
RES_DIR = ROOT / "outputs" / "results"
for _d in (EMB_DIR, MODEL_DIR, DATA_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PAD, UNK = "<PAD>", "<UNK>"
VOCAB_SIZE = 10_000          # most frequent tokens kept (Part 1.1 requirement)
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
_ART_RE = re.compile(r"^###\s*Article\s+(\d+)\s*###\s*$")
# punctuation (Urdu + Latin) stripped when reading the *unprocessed* raw.txt
_RAW_PUNCT = re.compile(r"[۔،؟؛!٪\"'“”‘’()\[\]{}:;,.?!\-–—…/\\|*]+")


def load_articles(path=CORPUS / "cleaned.txt", raw=False):
    """Return list of articles; each article = list of sentences; sentence = list of tokens.
    For raw.txt only punctuation is split off (no normalisation / stemming), so it
    acts as the 'unprocessed' ablation baseline."""
    arts, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = _ART_RE.match(line)
            if m:
                cur = []
                arts.append(cur)
                continue
            if not line or cur is None:
                continue
            if raw:
                line = _RAW_PUNCT.sub(" ", line)
            toks = line.split()
            if toks:
                cur.append(toks)
    return arts


def load_metadata():
    with open(CORPUS / "Metadata.json", encoding="utf-8") as f:
        return json.load(f)


def flatten(arts):
    return [s for a in arts for s in a]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def build_vocab(sentences, size=VOCAB_SIZE, specials=(PAD, UNK)):
    cnt = Counter(t for s in sentences for t in s)
    words = [w for w, _ in cnt.most_common(size)]
    itos = list(specials) + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos, cnt


def encode(tokens, stoi):
    u = stoi[UNK]
    return [stoi.get(t, u) for t in tokens]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plotting Urdu text (matplotlib does not shape Arabic script / RTL by itself)
# ---------------------------------------------------------------------------
def ur(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:  # pragma: no cover - fallback if libs missing
        return text


def setup_matplotlib():
    import matplotlib
    matplotlib.rcParams["font.family"] = ["Tahoma", "Arial", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
