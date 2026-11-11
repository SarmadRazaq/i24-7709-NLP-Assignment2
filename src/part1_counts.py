# -*- coding: utf-8 -*-
"""Part 1 - Section 1: TF-IDF and PPMI weighted representations."""
import numpy as np

from .common import (EMB_DIR, FIG_DIR, RES_DIR, UNK, PAD, build_vocab, encode, flatten,
                     load_articles, save_json, load_json, setup_matplotlib, ur)
from .topics import CATEGORIES, assign_topics
from . import lexicon as lx
from . import urdu_nlp as un

PPMI_VOCAB = 5000   # rows/cols of the saved PPMI matrix (see README: GitHub 100 MB file limit)
WINDOW = 5


# ---------------------------------------------------------------------------
# helpers shared by all of Part 1
# ---------------------------------------------------------------------------
def get_word2idx(arts=None, rebuild=False):
    path = EMB_DIR / "word2idx.json"
    if path.exists() and not rebuild:
        return load_json(path)
    arts = arts or load_articles()
    stoi, _, _ = build_vocab(flatten(arts))
    save_json(stoi, path)
    return stoi


def to_vocab_form(word, stoi):
    """Map a surface query word (e.g. عدالت) to the form used in cleaned.txt (عدال)."""
    w = lx.norm(word)
    for cand in (w, un.process_token(w), un.stem(w), un.lemmatize(w)):
        if cand in stoi:
            return cand
    return un.process_token(w)


def cosine_neighbours(M, stoi, itos, word, k=5, exclude=(PAD, UNK)):
    if word not in stoi or stoi[word] >= len(M):
        return []
    X = M.astype(np.float32)
    v = X[stoi[word]]
    norms = np.linalg.norm(X, axis=1) * (np.linalg.norm(v) + 1e-9) + 1e-9
    sims = X @ v / norms
    sims[stoi[word]] = -np.inf
    for e in exclude:
        if e in stoi and stoi[e] < len(sims):
            sims[stoi[e]] = -np.inf
    top = np.argsort(-sims)[:k]
    return [(itos[i], float(sims[i])) for i in top]


# ---------------------------------------------------------------------------
# 1.1 TF-IDF
# ---------------------------------------------------------------------------
def tfidf_matrix(arts, stoi):
    """Documents x vocabulary.  TF = count / doc length, IDF = log(N / (1 + df))."""
    N, V = len(arts), len(stoi)
    tf = np.zeros((N, V), dtype=np.float64)
    for d, a in enumerate(arts):
        ids = np.array(encode([t for s in a for t in s], stoi))
        np.add.at(tf[d], ids, 1.0)
        tf[d] /= max(len(ids), 1)
    df = (tf > 0).sum(0)
    idf = np.log(N / (1.0 + df))
    W = (tf * idf).astype(np.float32)
    W[:, stoi[PAD]] = 0.0
    return W, df, idf


def top_discriminative(W, labels, itos, df, k=10, min_df=3):
    """Score = mean TF-IDF inside the category - mean TF-IDF outside it."""
    labels = np.asarray(labels)
    out = {}
    for c, name in enumerate(CATEGORIES):
        inside, outside = W[labels == c].mean(0), W[labels != c].mean(0)
        score = inside - outside
        score[(df < min_df)] = -np.inf
        score[:2] = -np.inf  # PAD / UNK
        idx = np.argsort(-score)[:k]
        out[name] = [(itos[i], round(float(score[i]), 5)) for i in idx]
    return out


# ---------------------------------------------------------------------------
# 1.2 PPMI
# ---------------------------------------------------------------------------
def cooccurrence(sentences, stoi, V=PPMI_VOCAB, window=WINDOW):
    pairs = []
    for s in sentences:
        ids = np.array(encode(s, stoi), dtype=np.int64)
        n = len(ids)
        for off in range(1, window + 1):
            if n <= off:
                break
            a, b = ids[:-off], ids[off:]
            keep = (a < V) & (b < V)
            a, b = a[keep], b[keep]
            pairs.append(a * V + b)
            pairs.append(b * V + a)
    C = np.bincount(np.concatenate(pairs), minlength=V * V).astype(np.float64).reshape(V, V)
    C[0, :] = C[:, 0] = 0  # PAD never co-occurs
    return C


def ppmi(C):
    total = C.sum()
    pw = C.sum(1) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2((C / total) / np.outer(pw, pw))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)


# ---------------------------------------------------------------------------
# t-SNE of the 200 most frequent tokens, coloured by semantic category
# ---------------------------------------------------------------------------
TIME_WORDS = {un.process_token(lx.norm(w)) for w in
              "سال دن وقت بار بعد پہلے روز ماہ برس رات صبح منٹ گھنٹہ ہفتہ آج اب دور دوران سنہ".split()}
GEO = {t for e, typ in lx.GAZETTEER.items() if typ == "LOC" for t in e}
SEM_ORDER = ["Function word", "Verb / auxiliary", "Time & number", "Geography",
             "Sports", "Politics & state", "World & economy", "Society & media", "Other"]


def semantic_category(w):
    from .topics import KW
    if w in lx.LEXICON["POST"] | lx.LEXICON["PRON"] | lx.LEXICON["CONJ"] | lx.LEXICON["PRT"] | lx.LEXICON["DET"]:
        return "Function word"
    if w in lx.LEXICON["VERB"]:
        return "Verb / auxiliary"
    if w == "<NUM>" or w in lx.LEXICON["NUM"] or w in TIME_WORDS:
        return "Time & number"
    if w in GEO:
        return "Geography"
    if w in KW["Sports"]:
        return "Sports"
    if w in KW["Politics"]:
        return "Politics & state"
    if w in KW["International"] or w in KW["Economy"]:
        return "World & economy"
    if w in KW["Health & Society"] or w in {"فلم", "میڈیا", "سوشل", "خبر", "صارف", "اردو", "بی", "سی"}:
        return "Society & media"
    return "Other"


def tsne_plot(P, itos, n=200, path=FIG_DIR / "ppmi_tsne.png", title="t-SNE of PPMI vectors"):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    setup_matplotlib()
    ids = list(range(2, 2 + n))  # 2.. = most frequent real tokens
    X = P[ids].astype(np.float32)
    Y = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca",
             random_state=42).fit_transform(X)
    cats = [semantic_category(itos[i]) for i in ids]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(13, 10))
    for ci, c in enumerate(SEM_ORDER):
        m = np.array([x == c for x in cats])
        if m.any():
            ax.scatter(Y[m, 0], Y[m, 1], s=28, color=cmap(ci), label=f"{c} ({m.sum()})")
    for (x, y), i in zip(Y, ids):
        ax.annotate(ur(itos[i]), (x, y), fontsize=7, alpha=0.8,
                    xytext=(2, 2), textcoords="offset points")
    ax.legend(loc="best", fontsize=9, title="Semantic category")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, dict(zip([itos[i] for i in ids], cats))


QUERY_WORDS_PPMI = ["پاکستان", "حکومت", "عدالت", "فوج", "صحت", "تعلیم", "کرکٹ", "فلم",
                    "ایران", "پولیس", "میچ", "وزیر"]


def run(verbose=True):
    arts = load_articles()
    sents = flatten(arts)
    stoi = get_word2idx(arts, rebuild=True)
    itos = [None] * len(stoi)
    for w, i in stoi.items():
        itos[i] = w
    labels, _ = assign_topics(arts)

    # 1.1 ---------------------------------------------------------------
    W, df, idf = tfidf_matrix(arts, stoi)
    np.save(EMB_DIR / "tfidf_matrix.npy", W)
    top = top_discriminative(W, labels, itos, df)

    # 1.2 ---------------------------------------------------------------
    C = cooccurrence(sents, stoi)
    P = ppmi(C).astype(np.float16)
    np.save(EMB_DIR / "ppmi_matrix.npy", P)
    fig, cats = tsne_plot(P, itos)
    nn = {}
    for q in QUERY_WORDS_PPMI:
        w = to_vocab_form(q, stoi)
        nn[q] = {"form": w, "neighbours": cosine_neighbours(P, stoi, itos, w, k=5)}

    res = {"tfidf_shape": list(W.shape), "ppmi_shape": list(P.shape),
           "n_docs": len(arts), "vocab_size": len(stoi),
           "nonzero_ppmi_fraction": float((P > 0).mean()),
           "top10_tfidf": top, "ppmi_neighbours": nn,
           "tsne_category_counts": {c: list(cats.values()).count(c) for c in SEM_ORDER}}
    save_json(res, RES_DIR / "part1_counts.json")
    if verbose:
        print("TF-IDF", W.shape, "PPMI", P.shape)
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(run(), ensure_ascii=False, indent=1)[:4000])
