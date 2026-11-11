# -*- coding: utf-8 -*-
"""Skip-gram Word2Vec with negative sampling, implemented from scratch in PyTorch.

* separate centre (V) and context (U) embedding matrices, both |V| x d
* noise distribution P(w) ∝ f(w)^(3/4), K noise samples per positive pair
* loss  L = -log σ(u_o·v_c) - Σ_k log σ(-u_k·v_c)
* final embeddings = ½ (V + U)
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import encode


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.V = nn.Embedding(vocab_size, dim)   # centre-word vectors
        self.U = nn.Embedding(vocab_size, dim)   # context-word vectors
        bound = 0.5 / dim
        nn.init.uniform_(self.V.weight, -bound, bound)
        nn.init.uniform_(self.U.weight, -bound, bound)

    def forward(self, centre, context, negatives):
        v_c = self.V(centre)                              # (B, d)
        u_o = self.U(context)                             # (B, d)
        u_k = self.U(negatives)                           # (B, K, d)
        pos = F.logsigmoid((u_o * v_c).sum(-1))           # (B,)
        neg = F.logsigmoid(-torch.bmm(u_k, v_c.unsqueeze(-1)).squeeze(-1)).sum(-1)  # (B,)
        return -(pos + neg).mean()

    def embeddings(self):
        return (0.5 * (self.V.weight + self.U.weight)).detach().cpu().numpy()


def make_pairs(sentences, stoi, window=5, skip_ids=(0,), keep_prob=None, rng=None):
    """All (centre, context) pairs within a symmetric window of size `window`.
    If keep_prob is given, frequent tokens are first sub-sampled (Mikolov et al. 2013):
    token w is kept with probability min(1, sqrt(t / f(w)) + t / f(w))."""
    cs, os_ = [], []
    for s in sentences:
        ids = np.array(encode(s, stoi), dtype=np.int64)
        if keep_prob is not None:
            ids = ids[rng.random(len(ids)) < keep_prob[ids]]
        n = len(ids)
        for off in range(1, window + 1):
            if n <= off:
                break
            a, b = ids[:-off], ids[off:]
            cs += [a, b]
            os_ += [b, a]
    c, o = np.concatenate(cs), np.concatenate(os_)
    keep = ~np.isin(c, skip_ids) & ~np.isin(o, skip_ids)
    return c[keep], o[keep]


def noise_distribution(sentences, stoi, power=0.75):
    counts = np.zeros(len(stoi))
    for s in sentences:
        np.add.at(counts, encode(s, stoi), 1)
    counts[0] = 0  # PAD
    p = counts ** power
    return p / p.sum(), counts


def train_skipgram(sentences, stoi, dim=100, window=5, K=10, epochs=5, batch_size=1024,
                   lr=1e-3, log_every=200, seed=42, device="cpu", verbose=True, subsample=1e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    noise, counts = noise_distribution(sentences, stoi)
    keep_prob = None
    if subsample:
        f = counts / counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            keep_prob = np.minimum(1.0, np.sqrt(subsample / f) + subsample / f)
        keep_prob[~np.isfinite(keep_prob)] = 1.0
    noise_t = torch.tensor(noise, dtype=torch.float)
    model = SkipGramNS(len(stoi), dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log, running, t0 = [], 0.0, time.time()
    step, total_pairs = 0, 0
    for ep in range(epochs):
        centre, context = make_pairs(sentences, stoi, window, keep_prob=keep_prob, rng=rng)
        n = len(centre)
        total_pairs += n
        steps_per_epoch = (n + batch_size - 1) // batch_size
        perm = rng.permutation(n)
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            c = torch.from_numpy(centre[idx]).to(device)
            o = torch.from_numpy(context[idx]).to(device)
            neg = torch.multinomial(noise_t, len(idx) * K, replacement=True).view(len(idx), K).to(device)
            loss = model(c, o, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            step += 1
            if step % log_every == 0:
                log.append({"step": step, "epoch": ep + (b + 1) / steps_per_epoch, "loss": running / log_every})
                running = 0.0
        if verbose:
            last = log[-1]["loss"] if log else float("nan")
            print(f"  epoch {ep + 1}/{epochs}  loss={last:.4f}  pairs={n:,}  {time.time() - t0:.0f}s")
    return model, log, {"pairs_per_epoch": int(total_pairs / epochs), "steps_per_epoch": steps_per_epoch,
                        "subsample_t": subsample,
                        "train_seconds": round(time.time() - t0, 1)}
