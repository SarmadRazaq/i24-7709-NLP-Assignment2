# -*- coding: utf-8 -*-
"""Part 3: Transformer encoder topic classifier (+ BiLSTM classifier baseline for 8.2)."""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from .common import (EMB_DIR, FIG_DIR, MODEL_DIR, RES_DIR, SEED, encode, load_articles, load_json,
                     load_metadata, save_json, set_seed, setup_matplotlib, ur)
from .topics import CATEGORIES, assign_topics
from .transformer import TransformerClassifier

MAX_LEN = 256
EPOCHS = 20
BATCH = 16
WARMUP = 50


def build_data(seed=SEED):
    arts = load_articles()
    meta = load_metadata()
    labels, _ = assign_topics(arts, meta)
    stoi = load_json(EMB_DIR / "word2idx.json")
    X = np.zeros((len(arts), MAX_LEN), dtype=np.int64)       # 0 = <PAD>
    for i, a in enumerate(arts):
        ids = encode([t for s in a for t in s], stoi)[:MAX_LEN]  # truncate to 256 tokens
        X[i, :len(ids)] = ids
    y = np.array(labels)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    dist = {name: {CATEGORIES[c]: int((y[part] == c).sum()) for c in range(len(CATEGORIES))}
            for name, part in [("train", tr), ("val", va), ("test", te)]}
    return X, y, (tr, va, te), dist, stoi, arts, meta


def cosine_warmup(opt, warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


class BiLSTMClassifier(nn.Module):
    """Baseline for 8.2: 2-layer BiLSTM (Word2Vec C3 init, fine-tuned) + max-pool + linear."""

    def __init__(self, emb):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(emb.shape[1], 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.out = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, len(CATEGORIES)))

    def forward(self, ids):
        mask = (ids != 0).unsqueeze(-1)
        h, _ = self.lstm(self.emb(ids))
        h = h.masked_fill(~mask, -1e4).max(1).values
        return self.out(h)


def run_epochs(model, X, y, split, opt, sched=None, epochs=EPOCHS, seed=SEED, verbose=True, tag=""):
    tr, va, _ = split
    rng = np.random.default_rng(seed)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    hist, best, best_state = [], -1, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(tr)
        tot, correct = 0.0, 0
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            logits = model(Xt[b])
            loss = F.cross_entropy(logits, yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
            tot += loss.item() * len(b)
            correct += (logits.argmax(1) == yt[b]).sum().item()
        secs = time.time() - t0
        model.eval()
        with torch.no_grad():
            vlog = model(Xt[va])
            vloss = F.cross_entropy(vlog, yt[va]).item()
            vacc = (vlog.argmax(1) == yt[va]).float().mean().item()
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "train_acc": correct / len(tr),
                     "val_loss": vloss, "val_acc": vacc, "sec": round(secs, 2)})
        if vacc > best:
            best, best_state = vacc, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  {tag} ep{ep:02d} train_loss={tot / len(tr):.3f} acc={correct / len(tr):.3f} "
                  f"val_loss={vloss:.3f} val_acc={vacc:.3f} ({secs:.1f}s)")
    model.load_state_dict(best_state)
    return hist


def evaluate(model, X, y, idx):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[idx])).argmax(1).numpy()
    return pred, {"accuracy": accuracy_score(y[idx], pred),
                  "macro_f1": f1_score(y[idx], pred, average="macro", labels=list(range(len(CATEGORIES))), zero_division=0)}


def plot_curves(hist, path, title):
    import matplotlib.pyplot as plt
    setup_matplotlib()
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    a.plot(ep, [h["train_loss"] for h in hist], "o-", ms=3, label="train")
    a.plot(ep, [h["val_loss"] for h in hist], "s-", ms=3, label="val")
    a.set_title("loss")
    b.plot(ep, [h["train_acc"] for h in hist], "o-", ms=3, label="train")
    b.plot(ep, [h["val_acc"] for h in hist], "s-", ms=3, label="val")
    b.set_title("accuracy")
    for ax in (a, b):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cm(cm, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES, cbar=False, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title("Transformer topic classifier - test confusion matrix")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_attention(model, X, art_ids, itos, labels, path, heads=(0, 1), n_tok=30):
    """Final-layer attention for the first n_tok positions ([CLS] + tokens)."""
    import matplotlib.pyplot as plt
    setup_matplotlib()
    model.eval()
    fig, axes = plt.subplots(len(art_ids), len(heads) + 1, figsize=(6.2 * (len(heads) + 1), 5.6 * len(art_ids)))
    summary = []
    for r, i in enumerate(art_ids):
        ids = torch.tensor(X[i:i + 1])
        with torch.no_grad():
            logits, attns = model(ids, return_attn=True)
        A = attns[-1][0]                                          # (h, T, T) final layer
        n = min(n_tok, int((ids != 0).sum()) + 1)
        toks = ["[CLS]"] + [itos[t] for t in X[i][:n - 1]]
        lab = [ur(t) for t in toks]
        for c, h in enumerate(heads):
            ax = axes[r, c]
            ax.imshow(A[h, :n, :n].numpy(), cmap="viridis", aspect="auto")
            ax.set_xticks(range(n))
            ax.set_xticklabels(lab, rotation=90, fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels(lab, fontsize=7)
            ax.set_title(f"article {i + 1} ({CATEGORIES[labels[i]]}) - head {h + 1}", fontsize=9)
        # [CLS] attention over the full sequence, averaged over heads -> top tokens
        full = int((ids != 0).sum()) + 1
        cls_att = A[:, 0, 1:full].mean(0).numpy()
        top = np.argsort(-cls_att)[:12]
        ax = axes[r, -1]
        ax.barh(range(len(top))[::-1], cls_att[top], color="tab:orange")
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels([ur(itos[X[i][j]]) for j in top], fontsize=9)
        ax.set_title("[CLS] attention (mean of 4 heads), top-12 tokens", fontsize=9)
        summary.append({"article": int(i + 1), "label": CATEGORIES[labels[i]],
                        "cls_top_tokens": [(itos[X[i][j]], float(cls_att[j])) for j in top],
                        "per_head_cls_top5": {f"head{h + 1}": [itos[X[i][j]] for j in np.argsort(-A[h, 0, 1:full].numpy())[:5]]
                                              for h in range(A.shape[0])}})
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return summary


def run(verbose=True):
    X, y, split, dist, stoi, arts, meta = build_data()
    itos = [None] * len(stoi)
    for w, i in stoi.items():
        itos[i] = w
    tr, va, te = split
    steps = EPOCHS * math.ceil(len(tr) / BATCH)

    # ------------------ Transformer ------------------
    set_seed(SEED)
    model = TransformerClassifier(len(stoi), n_classes=len(CATEGORIES), d_model=128, n_heads=4, d_k=32,
                                  d_ff=512, n_layers=4, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    sched = cosine_warmup(opt, WARMUP, steps)
    t0 = time.time()
    hist_t = run_epochs(model, X, y, split, opt, sched, tag="TRF", verbose=verbose)
    t_trf = time.time() - t0
    pred, test_t = evaluate(model, X, y, te)
    cm = confusion_matrix(y[te], pred, labels=list(range(len(CATEGORIES))))
    torch.save({"state_dict": model.state_dict(), "categories": CATEGORIES,
                "config": {"d_model": 128, "heads": 4, "d_k": 32, "d_ff": 512, "layers": 4, "max_len": MAX_LEN}},
               MODEL_DIR / "transformer_cls.pt")
    plot_curves(hist_t, FIG_DIR / "transformer_curves.png", "Transformer: loss and accuracy per epoch")
    plot_cm(cm, FIG_DIR / "transformer_confusion.png")
    correct = [int(te[k]) for k in range(len(te)) if pred[k] == y[te[k]]]
    # pick 3 correctly classified articles from different classes where possible
    chosen, seen = [], set()
    for i in correct:
        if y[i] not in seen:
            chosen.append(i)
            seen.add(y[i])
        if len(chosen) == 3:
            break
    for i in correct:
        if len(chosen) == 3:
            break
        if i not in chosen:
            chosen.append(i)
    attn = plot_attention(model, X, chosen, itos, y, FIG_DIR / "transformer_attention.png")

    # ------------------ BiLSTM baseline ------------------
    set_seed(SEED)
    E = np.load(EMB_DIR / "embeddings_w2v.npy")
    lstm = BiLSTMClassifier(E)
    opt2 = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    hist_l = run_epochs(lstm, X, y, split, opt2, None, tag="LSTM", verbose=verbose)
    t_lstm = time.time() - t0
    pred_l, test_l = evaluate(lstm, X, y, te)
    plot_curves(hist_l, FIG_DIR / "bilstm_cls_curves.png", "BiLSTM classifier: loss and accuracy per epoch")

    def conv_epoch(h):
        best = max(r["val_acc"] for r in h)
        return int(next(r["epoch"] for r in h if r["val_acc"] >= best))

    res = {"class_distribution": dist, "n_params_transformer": sum(p.numel() for p in model.parameters()),
           "n_params_bilstm": sum(p.numel() for p in lstm.parameters()),
           "transformer": {"test": test_t, "confusion_matrix": cm.tolist(), "history": hist_t,
                           "train_seconds": round(t_trf, 1), "sec_per_epoch": round(t_trf / EPOCHS, 2),
                           "best_val_epoch": conv_epoch(hist_t), "attention_articles": attn},
           "bilstm": {"test": test_l, "history": hist_l, "train_seconds": round(t_lstm, 1),
                      "sec_per_epoch": round(t_lstm / EPOCHS, 2), "best_val_epoch": conv_epoch(hist_l),
                      "confusion_matrix": confusion_matrix(y[te], pred_l, labels=list(range(len(CATEGORIES)))).tolist()},
           "attention_article_titles": {int(i + 1): meta[i]["title"] for i in chosen}}
    save_json(res, RES_DIR / "part3_results.json")
    if verbose:
        print("Transformer test", test_t, "BiLSTM test", test_l)
    return res


if __name__ == "__main__":
    run()
