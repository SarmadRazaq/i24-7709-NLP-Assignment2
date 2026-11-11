# -*- coding: utf-8 -*-
"""Part 1 - Section 2: Skip-gram Word2Vec training + evaluation and the four-condition
comparison (C1 PPMI, C2 SG raw.txt, C3 SG cleaned.txt, C4 SG cleaned d=200)."""
import numpy as np
import torch

from .common import (CORPUS, EMB_DIR, FIG_DIR, RES_DIR, MODEL_DIR, PAD, UNK, build_vocab,
                     flatten, load_articles, load_json, save_json, setup_matplotlib)
from .part1_counts import PPMI_VOCAB, cosine_neighbours, get_word2idx, to_vocab_form
from .word2vec import train_skipgram
from . import lexicon as lx

EPOCHS = 15
MIN_COUNT = 10   # neighbour / analogy candidates must occur >= 10 times (Mikolov-style restriction)
BATCH = 1024
torch.set_num_threads(max(1, torch.get_num_threads()))

# The eight assignment query words (transliteration -> Urdu)
QUERY_8 = {"Pakistan": "پاکستان", "Hukumat": "حکومت", "Adalat": "عدالت", "Maeeshat": "معیشت",
           "Fauj": "فوج", "Sehat": "صحت", "Taleem": "تعلیم", "Aabadi": "آبادی"}

# 10 analogy tests a : b :: c : d   (surface forms; mapped to the vocabulary form)
ANALOGIES = [
    ("پاکستان", "لاہور", "انڈیا", "دہلی"),           # country : major city
    ("کرکٹ", "رنز", "فٹبال", "گول"),                 # sport : scoring unit
    ("فلم", "اداکار", "کرکٹ", "کھلاڑی"),             # domain : performer
    ("لیونل", "میسی", "کرسٹیانو", "رونالڈو"),        # first name : surname
    ("ٹیسٹ", "کرکٹ", "ورلڈ", "کپ"),                  # collocation (Test cricket : World Cup)
    ("مرد", "عورت", "لڑکا", "لڑکی"),                 # male : female
    ("بیٹی", "بیٹا", "بہن", "بھائی"),                # female : male
    ("انڈیا", "مودی", "پاکستان", "شریف"),            # country : head of government
    ("ایران", "تہران", "افغانستان", "کابل"),          # country : capital
    ("شوہر", "بیوی", "باپ", "ماں"),                  # male : female
]
# additional analogies that were also tried (reported in the notebook for transparency)
ANALOGY_POOL_EXTRA = [
    ("پاکستان", "اسلام", "انڈیا", "ہندو"), ("کرکٹ", "بلے", "فٹبال", "گیند"), ("بیٹا", "بیٹی", "بھائی", "بہن"),
    ("عدالت", "جج", "ہسپتال", "ڈاکٹر"), ("پاکستان", "روپے", "امریکہ", "ڈالر"), ("لاہور", "پنجاب", "کراچی", "سندھ"),
    ("پشاور", "خیبر", "کراچی", "سندھ"), ("عمران", "خان", "نواز", "شریف"), ("بابر", "اعظم", "شاہین", "آفریدی"),
    ("ایران", "ایرانی", "سعودی", "عرب"), ("پاکستان", "پنجاب", "انڈیا", "گجرات"), ("ڈالر", "امریکہ", "روپے", "پاکستان"),
    ("ماں", "باپ", "بہن", "بھائی"), ("پاکستان", "پاکستان", "انڈیا", "انڈیا"),
]

# 20 manually labelled related-word pairs for MRR (query, gold related word)
MRR_PAIRS = [("عدالت", "جج"), ("کرکٹ", "ٹیم"), ("فلم", "اداکار"), ("ہسپتال", "ڈاکٹر"),
             ("پولیس", "گرفتار"), ("تعلیم", "طلبہ"), ("رنز", "وکٹ"), ("فٹبال", "گول"),
             ("ڈالر", "روپے"), ("تیل", "قیمت"), ("لاہور", "کراچی"), ("شوہر", "بیوی"),
             ("ایران", "اسرائیل"), ("فوج", "جنگ"), ("حکومت", "وزیر"), ("صحت", "بیماری"),
             ("گانے", "موسیقی"), ("سعودی", "عرب"), ("ماں", "باپ"), ("میچ", "کھلاڑی")]
QUERY_5 = ["پاکستان", "حکومت", "عدالت", "کرکٹ", "فلم"]


# ---------------------------------------------------------------------------
class Space:
    """An embedding space + its vocabulary (handles cleaned vs raw word forms)."""

    def __init__(self, name, M, stoi, raw=False, counts=None, min_count=MIN_COUNT):
        self.name, self.M, self.stoi, self.raw = name, M.astype(np.float32), stoi, raw
        self.itos = [None] * len(stoi)
        for w, i in stoi.items():
            self.itos[i] = w
        self.Mn = self.M / (np.linalg.norm(self.M, axis=1, keepdims=True) + 1e-9)
        self.n = len(self.M)
        # candidate mask: excludes PAD/UNK and words rarer than min_count
        self.cand = np.ones(self.n, dtype=bool)
        self.cand[:2] = False
        if counts is not None:
            self.cand &= np.array([counts.get(self.itos[i], 0) >= min_count for i in range(self.n)])

    def form(self, w):
        if self.raw:
            w = lx.norm(w)
            return w if w in self.stoi else w
        return to_vocab_form(w, self.stoi)

    def has(self, w):
        f = self.form(w)
        return f in self.stoi and self.stoi[f] < self.n

    def nn(self, w, k=10):
        if not self.has(w):
            return []
        i = self.stoi[self.form(w)]
        sims = self.Mn @ self.Mn[i]
        sims[~self.cand] = -np.inf
        sims[i] = -np.inf
        top = np.argsort(-sims)[:k]
        return [(self.itos[j], float(sims[j])) for j in top]

    def rank_of(self, q, g):
        """Rank (1-based) of gold word g among all words by cosine to q (q excluded)."""
        if not (self.has(q) and self.has(g)):
            return None
        qi, gi = self.stoi[self.form(q)], self.stoi[self.form(g)]
        sims = self.Mn @ self.Mn[qi]
        sims[~self.cand] = -np.inf
        sims[qi] = -np.inf
        return int((sims > sims[gi]).sum()) + 1

    def analogy(self, a, b, c, k=3):
        if not all(self.has(x) for x in (a, b, c)):
            return []
        ia, ib, ic = (self.stoi[self.form(x)] for x in (a, b, c))
        v = self.Mn[ib] - self.Mn[ia] + self.Mn[ic]
        sims = self.Mn @ (v / (np.linalg.norm(v) + 1e-9))
        sims[~self.cand] = -np.inf
        sims[[ia, ib, ic]] = -np.inf
        top = np.argsort(-sims)[:k]
        return [(self.itos[i], float(sims[i])) for i in top]


def mrr(space, pairs=MRR_PAIRS):
    rr, detail = [], []
    for q, g in pairs:
        r = space.rank_of(q, g)
        rr.append(0.0 if r is None else 1.0 / r)
        detail.append({"query": q, "gold": g, "rank": r})
    return float(np.mean(rr)), detail


def evaluate_analogies(space, tests=ANALOGIES):
    out, correct = [], 0
    for a, b, c, d in tests:
        top = space.analogy(a, b, c)
        gold = space.form(d)
        ok = gold in [w for w, _ in top]
        correct += ok
        out.append({"test": f"{a} : {b} :: {c} : ?", "gold": d, "gold_form": gold,
                    "top3": top, "correct": bool(ok)})
    return out, correct


def plot_loss(logs, path=FIG_DIR / "w2v_loss.png"):
    import matplotlib.pyplot as plt
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, log in logs.items():
        ax.plot([r["epoch"] for r in log], [r["loss"] for r in log], label=name, lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean BCE loss (per 500 steps)")
    ax.set_title("Skip-gram negative-sampling training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run(epochs=EPOCHS, verbose=True):
    clean = flatten(load_articles())
    raw = flatten(load_articles(CORPUS / "raw.txt", raw=True))
    stoi = get_word2idx()
    stoi_raw, _, cnt_raw = build_vocab(raw)
    _, _, cnt_clean = build_vocab(clean)
    save_json(stoi_raw, EMB_DIR / "word2idx_raw.json")

    logs, info, spaces = {}, {}, {}
    # C3: Skip-gram on cleaned.txt (main model)
    for cid, sents, voc, d in [("C3", clean, stoi, 100), ("C2", raw, stoi_raw, 100), ("C4", clean, stoi, 200)]:
        if verbose:
            print(f"Training {cid} (d={d})")
        model, log, inf = train_skipgram(sents, voc, dim=d, epochs=epochs, batch_size=BATCH,
                                         lr=1e-3, K=10, window=5, verbose=verbose)
        E = model.embeddings()
        logs[cid], info[cid] = log, inf
        fname = {"C3": "embeddings_w2v.npy", "C2": "embeddings_w2v_raw_C2.npy",
                 "C4": "embeddings_w2v_d200_C4.npy"}[cid]
        np.save(EMB_DIR / fname, E)
        spaces[cid] = Space(cid, E, voc, raw=(cid == "C2"), counts=cnt_raw if cid == "C2" else cnt_clean)
    P = np.load(EMB_DIR / "ppmi_matrix.npy")
    ppmi_stoi = {w: i for w, i in stoi.items() if i < PPMI_VOCAB}
    spaces["C1"] = Space("C1", P, ppmi_stoi, counts=cnt_clean)
    plot_loss({"C3 cleaned d=100": logs["C3"], "C2 raw d=100": logs["C2"], "C4 cleaned d=200": logs["C4"]})

    c3 = spaces["C3"]
    neighbours8 = {k: {"urdu": v, "form": c3.form(v), "top10": c3.nn(v, 10)} for k, v in QUERY_8.items()}
    analog, n_correct = evaluate_analogies(c3)
    extra, n_extra = evaluate_analogies(c3, ANALOGY_POOL_EXTRA)

    conditions = {}
    for cid in ["C1", "C2", "C3", "C4"]:
        sp = spaces[cid]
        m, det = mrr(sp)
        conditions[cid] = {"mrr": m, "mrr_detail": det,
                           "top5": {q: sp.nn(q, 5) for q in QUERY_5},
                           "analogies_correct": evaluate_analogies(sp)[1]}

    res = {"train_info": info, "loss_log": logs, "neighbours_C3": neighbours8,
           "analogies_C3": analog, "analogies_correct_C3": n_correct,
           "analogies_extra_C3": extra, "analogies_extra_correct_C3": n_extra, "conditions": conditions,
           "epochs": epochs, "batch_size": BATCH}
    save_json(res, RES_DIR / "part1_w2v.json")
    if verbose:
        print("analogies correct (C3):", n_correct, "/", len(ANALOGIES))
        print({c: round(v["mrr"], 4) for c, v in conditions.items()})
    return res


if __name__ == "__main__":
    run()
