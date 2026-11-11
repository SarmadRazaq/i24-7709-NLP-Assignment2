# -*- coding: utf-8 -*-
"""Part 2 - Sections 4 & 5: BiLSTM(-CRF) POS tagging and NER, evaluation and ablations."""
import copy
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .annotate import read_conll
from .common import DATA_DIR, EMB_DIR, FIG_DIR, MODEL_DIR, RES_DIR, load_json, save_json, set_seed, setup_matplotlib, ur
from .conlleval import evaluate as conll_eval, report as conll_report, write_conlleval_file
from .lexicon import POS_TAGS
from .seqmodels import BiLSTMTagger

NER_TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
MAX_EPOCHS = 60
PATIENCE = 5
MIN_EPOCHS = 15   # warm-up: NER models predict all-O (F1=0) for the first epochs, so early
                  # stopping is only allowed after 15 epochs (patience still 5 on val F1)
BATCH = 16
ABLATION_SEEDS = (42, 7, 123)


# ---------------------------------------------------------------------------
def load_split(task, split):
    return read_conll(DATA_DIR / f"{task}_{split}.conll")


def batches(data, stoi, tag2id, batch_size, shuffle, rng=None):
    idx = np.arange(len(data))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = [data[i] for i in idx[s:s + batch_size]]
        L = max(len(t) for t, _ in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        y = torch.zeros(len(chunk), L, dtype=torch.long)
        m = torch.zeros(len(chunk), L, dtype=torch.bool)
        for b, (toks, tags) in enumerate(chunk):
            n = len(toks)
            x[b, :n] = torch.tensor([stoi.get(t, 1) for t in toks])
            y[b, :n] = torch.tensor([tag2id[t] for t in tags])
            m[b, :n] = True
        yield x, m.sum(1), y, m


def predict(model, data, stoi, tag2id, id2tag):
    model.eval()
    preds = []
    for x, lens, y, m in batches(data, stoi, tag2id, 64, False):
        preds += [[id2tag[i] for i in p] for p in model.predict(x, lens, m)]
    return preds


def eval_loss(model, data, stoi, tag2id):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, lens, y, m in batches(data, stoi, tag2id, 64, False):
            tot += model.loss(x, lens, y, m).item() * x.size(0)
            n += x.size(0)
    return tot / n


def score(task, gold, pred):
    if task == "pos":
        g = [t for s in gold for t in s]
        p = [t for s in pred for t in s]
        labels = sorted(set(g) | set(p))
        return {"accuracy": accuracy_score(g, p), "macro_f1": f1_score(g, p, labels=labels, average="macro", zero_division=0)}
    r = conll_eval(gold, pred)
    return {"f1": r["overall"]["f1"], "precision": r["overall"]["precision"], "recall": r["overall"]["recall"], "detail": r}


def main_metric(task, s):
    return s["macro_f1"] if task == "pos" else s["f1"]


def train_model(task, cfg, seed=42, verbose=False):
    """cfg keys: freeze, pretrained(bool), bidirectional, dropout, crf."""
    set_seed(seed)
    tags = POS_TAGS if task == "pos" else NER_TAGS
    tag2id = {t: i for i, t in enumerate(tags)}
    id2tag = {i: t for t, i in tag2id.items()}
    stoi = load_json(EMB_DIR / "word2idx.json")
    E = np.load(EMB_DIR / "embeddings_w2v.npy") if cfg.get("pretrained", True) else None
    train, val = load_split(task, "train"), load_split(task, "val")
    model = BiLSTMTagger(len(stoi), len(tags), emb_dim=100, hidden=128, layers=2,
                         bidirectional=cfg.get("bidirectional", True), dropout=cfg.get("dropout", 0.5),
                         pretrained=E, freeze=cfg.get("freeze", False), use_crf=cfg.get("crf", False))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    hist, best, best_state, bad, t0 = [], -1, None, 0, time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for x, lens, y, m in batches(train, stoi, tag2id, BATCH, True, rng):
            loss = model.loss(x, lens, y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        vl = eval_loss(model, val, stoi, tag2id)
        vs = score(task, [t for _, t in val], predict(model, val, stoi, tag2id, id2tag))
        vf = main_metric(task, vs)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vl, "val_f1": vf})
        if verbose:
            print(f"   ep{ep:02d} train={tot / n:.4f} val={vl:.4f} valF1={vf:.4f}")
        if vf > best:
            best, best_state, bad = vf, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE and ep >= MIN_EPOCHS:   # early stopping on validation F1
                break
    model.load_state_dict(best_state)
    secs = time.time() - t0
    return model, {"history": hist, "best_val_f1": best, "epochs_run": len(hist),
                   "best_epoch": int(np.argmax([h["val_f1"] for h in hist]) + 1),
                   "seconds": round(secs, 1), "sec_per_epoch": round(secs / len(hist), 2)}, (stoi, tag2id, id2tag)


def test_eval(task, model, maps):
    stoi, tag2id, id2tag = maps
    test = load_split(task, "test")
    gold = [t for _, t in test]
    pred = predict(model, test, stoi, tag2id, id2tag)
    return score(task, gold, pred), test, gold, pred


# ---------------------------------------------------------------------------
def plot_history(hists, path, title):
    import matplotlib.pyplot as plt
    setup_matplotlib()
    fig, axes = plt.subplots(1, len(hists), figsize=(5.2 * len(hists), 3.8), squeeze=False)
    for ax, (name, h) in zip(axes[0], hists.items()):
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_loss"] for r in h], "o-", ms=3, label="train loss")
        ax.plot(ep, [r["val_loss"] for r in h], "s-", ms=3, label="val loss")
        ax2 = ax.twinx()
        ax2.plot(ep, [r["val_f1"] for r in h], "g--", lw=1, label="val F1")
        ax2.set_ylabel("val F1", color="g")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(loc="upper center", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(cm, labels, path, title):
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def confused_pairs(test, gold, pred, labels, k=3, n_examples=2):
    g = [t for s in gold for t in s]
    p = [t for s in pred for t in s]
    cm = confusion_matrix(g, p, labels=labels)
    off = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if cm[i, j] + cm[j, i] > 0:
                off.append((int(cm[i, j] + cm[j, i]), labels[i], labels[j], int(cm[i, j]), int(cm[j, i])))
    off.sort(reverse=True)
    out = []
    for tot, a, b, ab, ba in off[:k]:
        ex = []
        for (toks, _), gs, ps in zip(test, gold, pred):
            hits = [i for i, (x, y) in enumerate(zip(gs, ps)) if {x, y} == {a, b} and x != y]
            if hits:
                ex.append({"sentence": " ".join(toks),
                           "errors": [f"{toks[i]} (gold={gs[i]}, pred={ps[i]})" for i in hits]})
            if len(ex) >= n_examples:
                break
        out.append({"pair": f"{a} <-> {b}", "count": tot, f"{a}->{b}": ab, f"{b}->{a}": ba, "examples": ex})
    return cm, out


def ner_errors(test, gold, pred, n=5):
    from .conlleval import chunks
    fps, fns = [], []
    for (toks, _), gs, ps in zip(test, gold, pred):
        gc, pc = chunks(gs), chunks(ps)
        sent = " ".join(toks)
        for typ, s, e in sorted(pc - gc, key=lambda c: c[1]):
            gold_here = [c for c in gc if c[1] < e and s < c[2]]
            fps.append({"entity": " ".join(toks[s:e]), "pred_type": typ,
                        "gold": [(" ".join(toks[c[1]:c[2]]), c[0]) for c in gold_here] or "O",
                        "sentence": sent})
        for typ, s, e in sorted(gc - pc, key=lambda c: c[1]):
            pred_here = [c for c in pc if c[1] < e and s < c[2]]
            fns.append({"entity": " ".join(toks[s:e]), "gold_type": typ,
                        "pred": [(" ".join(toks[c[1]:c[2]]), c[0]) for c in pred_here] or "O",
                        "sentence": sent})
    return fps[:n], fns[:n], len(fps), len(fns)


# ---------------------------------------------------------------------------
BASE = {"pretrained": True, "freeze": False, "bidirectional": True, "dropout": 0.5}
ABLATIONS = {"A1 unidirectional LSTM": {"bidirectional": False},
             "A2 no dropout": {"dropout": 0.0},
             "A3 random embedding init": {"pretrained": False}}


def run(verbose=True):
    res = {}
    for task in ["pos", "ner"]:
        crf = task == "ner"
        R = {}
        # ---- frozen vs fine-tuned (C3 Word2Vec initialisation) ----
        models, hists = {}, {}
        for mode, freeze in [("frozen", True), ("fine-tuned", False)]:
            m, info, maps = train_model(task, {**BASE, "freeze": freeze, "crf": crf})
            ts, test, gold, pred = test_eval(task, m, maps)
            models[mode] = (m, maps, test, gold, pred)
            hists[f"{task.upper()} {mode}"] = info["history"]
            R[mode] = {"val_f1": info["best_val_f1"], "test": {k: v for k, v in ts.items() if k != "detail"},
                       "epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
                       "sec_per_epoch": info["sec_per_epoch"], "history": info["history"]}
            if task == "ner":
                R[mode]["conlleval"] = ts["detail"]
            if verbose:
                print(task, mode, "valF1=%.4f" % info["best_val_f1"], {k: round(v, 4) for k, v in ts.items() if k != "detail"})
        best_mode = max(R, key=lambda k: R[k]["val_f1"])
        R["best_mode"] = best_mode
        m, maps, test, gold, pred = models[best_mode]
        torch.save({"state_dict": m.state_dict(), "tags": POS_TAGS if task == "pos" else NER_TAGS,
                    "config": {**BASE, "freeze": best_mode == "frozen", "crf": crf, "hidden": 128, "layers": 2}},
                   MODEL_DIR / f"bilstm_{task}.pt")
        plot_history(hists, FIG_DIR / f"bilstm_{task}_loss.png", f"BiLSTM {task.upper()}: loss per epoch")

        if task == "pos":
            cm, pairs = confused_pairs(test, gold, pred, POS_TAGS)
            plot_confusion(cm, POS_TAGS, FIG_DIR / "pos_confusion.png", f"POS confusion matrix (test, {best_mode})")
            R["confusion_matrix"] = cm.tolist()
            R["confused_pairs"] = pairs
            R["per_tag_test_counts"] = dict(Counter(t for s in gold for t in s))
        else:
            write_conlleval_file([t for t, _ in test], gold, pred, RES_DIR / "ner_test_conlleval.txt")
            R["conlleval_report"] = conll_report(conll_eval(gold, pred))
            fps, fns, nfp, nfn = ner_errors(test, gold, pred)
            R["false_positives"], R["false_negatives"] = fps, fns
            R["n_fp"], R["n_fn"] = nfp, nfn

        # ---- ablations (each vs the fine-tuned full model), 3 seeds ----
        abl = dict(ABLATIONS)
        if task == "ner":
            abl["A4 softmax instead of CRF"] = {"crf": False}
        abl_res = {}
        for name, change in {"Full model (fine-tuned)": {}, **abl}.items():
            scores, vals, eps, secs = [], [], [], []
            for sd in ABLATION_SEEDS:
                mm, info, mp = train_model(task, {**BASE, "crf": crf, **change}, seed=sd)
                ts, *_ = test_eval(task, mm, mp)
                scores.append(main_metric(task, ts))
                vals.append(info["best_val_f1"])
                eps.append(info["epochs_run"])
                secs.append(info["sec_per_epoch"])
                if task == "pos":
                    scores[-1] = (ts["accuracy"], ts["macro_f1"])
            if task == "pos":
                acc = [s[0] for s in scores]
                f1 = [s[1] for s in scores]
                abl_res[name] = {"test_acc_mean": float(np.mean(acc)), "test_acc_std": float(np.std(acc)),
                                 "test_f1_mean": float(np.mean(f1)), "test_f1_std": float(np.std(f1)),
                                 "val_f1_mean": float(np.mean(vals)), "epochs_mean": float(np.mean(eps))}
            else:
                abl_res[name] = {"test_f1_mean": float(np.mean(scores)), "test_f1_std": float(np.std(scores)),
                                 "val_f1_mean": float(np.mean(vals)), "epochs_mean": float(np.mean(eps))}
            if verbose:
                print("  ablation", task, name, {k: round(v, 4) for k, v in abl_res[name].items()})
        R["ablations"] = abl_res
        res[task] = R
    save_json(res, RES_DIR / "part2_results.json")
    return res


if __name__ == "__main__":
    run()
