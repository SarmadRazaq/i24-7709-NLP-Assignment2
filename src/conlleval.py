# -*- coding: utf-8 -*-
"""Python port of the CoNLL-2000 `conlleval` entity-level scorer (BIO / IOB2).

Chunk boundaries follow conlleval.pl exactly: a chunk starts at B-X, or at I-X when the
previous tag is O or of a different type; it ends before O, B-*, or a type change.
An entity counts as correct only if both its span and type match exactly.
"""
from collections import defaultdict


def _split(tag):
    if tag == "O" or tag is None:
        return "O", None
    p, t = tag.split("-", 1)
    return p, t


def chunks(tags):
    """Return set of (type, start, end_exclusive)."""
    out, start, typ = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        p, t = _split(tag)
        end_chunk = typ is not None and (p in ("O", "B") or t != typ)
        if end_chunk:
            out.append((typ, start, i))
            typ = None
        start_chunk = p == "B" or (p == "I" and typ is None)
        if start_chunk:
            start, typ = i, t
    return set(out)


def evaluate(gold_seqs, pred_seqs, types=("PER", "LOC", "ORG", "MISC")):
    tp, n_gold, n_pred = defaultdict(int), defaultdict(int), defaultdict(int)
    correct_tok = total_tok = 0
    for g, p in zip(gold_seqs, pred_seqs):
        gc, pc = chunks(g), chunks(p)
        for c in gc:
            n_gold[c[0]] += 1
        for c in pc:
            n_pred[c[0]] += 1
        for c in gc & pc:
            tp[c[0]] += 1
        correct_tok += sum(a == b for a, b in zip(g, p))
        total_tok += len(g)

    def prf(t, ng, npred):
        p = t / npred if npred else 0.0
        r = t / ng if ng else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f, "gold": ng, "pred": npred, "tp": t}

    res = {ty: prf(tp[ty], n_gold[ty], n_pred[ty]) for ty in types}
    res["overall"] = prf(sum(tp.values()), sum(n_gold.values()), sum(n_pred.values()))
    res["overall"]["token_accuracy"] = correct_tok / max(total_tok, 1)
    return res


def report(res):
    o = res["overall"]
    lines = [f"accuracy: {100 * o['token_accuracy']:6.2f}%; precision: {100 * o['precision']:6.2f}%; "
             f"recall: {100 * o['recall']:6.2f}%; FB1: {100 * o['f1']:6.2f}"]
    for k, v in res.items():
        if k == "overall":
            continue
        lines.append(f"{k:>17}: precision: {100 * v['precision']:6.2f}%; recall: {100 * v['recall']:6.2f}%; "
                     f"FB1: {100 * v['f1']:6.2f}  {v['pred']}")
    return "\n".join(lines)


def write_conlleval_file(tokens, gold, pred, path):
    """Write `token gold pred` lines (input format of conlleval.pl)."""
    with open(path, "w", encoding="utf-8") as f:
        for ts, gs, ps in zip(tokens, gold, pred):
            for t, g, p in zip(ts, gs, ps):
                f.write(f"{t} {g} {p}\n")
            f.write("\n")
