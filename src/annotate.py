# -*- coding: utf-8 -*-
"""Part 2 - Section 3: dataset preparation.

* sample 500 sentences from cleaned.txt (125 from each of 4 topic categories,
  so >= 100 from each of at least 3 categories)
* rule-based POS tagger: hand-crafted lexicon + contextual rules on top of the
  Assignment 1 stemmer/lemmatiser forms
* rule-based BIO NER: longest-match gazetteer + title / name / organisation rules
* 70/15/15 split stratified by topic, written in CoNLL format
"""
import random
from collections import Counter

from sklearn.model_selection import train_test_split

from .common import DATA_DIR, RES_DIR, SEED, load_articles, save_json
from .topics import CATEGORIES, assign_topics
from . import lexicon as lx
from . import urdu_nlp as un

N_SENT = 500
TOPICS_USED = ["Politics", "Sports", "International", "Health & Society"]
PER_TOPIC = N_SENT // len(TOPICS_USED)          # 125
BOILERPLATE = ("سب سے زیادہ پڑھ", "بی بی سی اردو کی خبر", "سبسکرائب", "کلک کری", "واٹس ایپ چینل")

L = lx.LEXICON
FUNCTION = L["POST"] | L["PRON"] | L["CONJ"] | L["PRT"] | L["DET"]
AUX = {lx.norm(w) for w in lx.AUX_VERBS} | {un.process_token(lx.norm(w)) for w in lx.AUX_VERBS}
COPULA = {un.process_token(w) for w in "ہے ہیں تھا تھی تھے ہوں ہو".split()}
PROGRESSIVE = {un.process_token(w) for w in
               "رہا رہی رہے سکتا سکتی سکتے سکا سکے چکا چکی چکے گیا گئی گئے جاتا جاتی جاتے جائے لگا لگے".split()}
LIGHT_VERBS = {un.process_token(w) for w in
               "کر کرنا کرنے کرتا کرتی کرتے کیا کیے کریں کرے ہو ہونا ہونے ہوتا ہوتی ہوتے ہوا ہوئی ہوئے دیا دی دیے لیا لی رکھا".split()}
INTENSIFIERS = {un.process_token(w) for w in "بہت زیادہ انتہائی کافی نہایت سب سے بے حد".split()}
PUNCT = set("۔،؟؛!٪:\"'()[]-")
DEMONSTR = {"یہ", "وہ", "اس", "ان", "اسی", "انہی", "انھی", "کوئ", "کوئی", "کسی", "کچھ", "ہر"}
CASE_POST = {"کے", "کی", "کا", "کو", "نے", "سے", "میں", "پر", "تک"}


# ---------------------------------------------------------------------------
# POS tagger
# ---------------------------------------------------------------------------
def lexical_tag(tok):
    if tok == un.NUM_TOKEN or tok.isdigit():
        return "NUM"
    if all(ch in PUNCT for ch in tok):
        return "PUNC"
    for tag in lx.LEXICON_ORDER:
        if tok in L[tag]:
            return tag
    return None


def is_urdu(tok):
    return any("؀" <= ch <= "ۿ" for ch in tok)


def pos_tag(tokens, ner_tags=None):
    n = len(tokens)
    tags = [lexical_tag(t) for t in tokens]
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < n else None
        prv = tokens[i - 1] if i > 0 else None
        # entities from the gazetteer are nouns (proper nouns -> NOUN in the universal set)
        if ner_tags is not None and ner_tags[i] != "O":
            tags[i] = "NOUN"
            continue
        # کی / کیا / کیے followed by an auxiliary are light verbs, not postpositions
        if t in {"کی", "کیا", "کیے", "کے"} and nxt in AUX and prv not in FUNCTION:
            tags[i] = "VERB"
            continue
        # sentence-initial میں = pronoun "I"
        # کیا: question word only sentence-initially, otherwise perfective of کرنا
        if t == "کیا":
            tags[i] = "PRON" if i == 0 else "VERB"
            continue
        if t == "میں" and i == 0:
            tags[i] = "PRON"
            continue
        # demonstratives: determiner before a content word, pronoun otherwise
        if t in DEMONSTR:
            nt = lexical_tag(nxt) if nxt else None
            tags[i] = "DET" if (nxt and nxt not in FUNCTION and nt not in ("VERB", "POST", "CONJ", "PRT")) else "PRON"
            continue
        if tags[i] is not None:
            # open-class words that are also listed as nouns, or adverbs, are nominal
            # when governed by a case postposition (شروع میں, بچے کی ...)
            if tags[i] in ("ADV", "VERB", "ADJ") and nxt in CASE_POST and                     (tags[i] == "ADV" or t in L["NOUN"]):
                tags[i] = "NOUN"
            continue
        # ---- unknown word: contextual rules ----
        if not is_urdu(t) or len(t) == 1:
            tags[i] = "UNK"
        elif prv in INTENSIFIERS:
            tags[i] = "ADJ"
        elif nxt in PROGRESSIVE or t.endswith("ایا") or t.endswith("وائ"):
            tags[i] = "VERB"
        elif i == n - 1 and prv not in L["POST"]:
            tags[i] = "VERB"            # Urdu is verb-final
        elif nxt in LIGHT_VERBS:
            tags[i] = "NOUN"            # conjunct-verb nominal (e.g. حاصل کر)
        elif nxt is not None and nxt in L["POST"]:
            tags[i] = "NOUN"
        elif nxt in COPULA and prv in L["POST"]:
            tags[i] = "ADJ"             # predicative: ... سے مختلف ہے
        else:
            tags[i] = "NOUN"            # default: open-class noun
    return tags


# ---------------------------------------------------------------------------
# NER tagger
# ---------------------------------------------------------------------------
NAME_SUFFIX = {un.process_token(w) for w in "خان سنگھ شاہ بیگ".split()}
GIVEN_NAMES = {un.process_token(w) for w in "محمد علی احمد عبد عبدال حسن حسین عمر".split()}


def ner_tag(tokens):
    n, tags, i = len(tokens), ["O"] * len(tokens), 0
    G = lx.GAZETTEER
    while i < n:
        # 1) longest gazetteer match
        match = None
        for ln in range(min(lx.MAX_ENT_LEN, n - i), 0, -1):
            typ = G.get(tuple(tokens[i:i + ln]))
            if typ:
                match = (ln, typ)
                break
        if match:
            ln, typ = match
            # organisation suffix rule: <LOC/MISC/unknown> + بورڈ / یونیورسٹی ... -> ORG
            if i + ln < n and tokens[i + ln] in lx.ORG_SUFFIX_SET and typ in ("LOC", "MISC"):
                ln, typ = ln + 1, "ORG"
            tags[i] = "B-" + typ
            for j in range(i + 1, i + ln):
                tags[j] = "I-" + typ
            i += ln
            continue
        t = tokens[i]
        # 2) title + unknown name(s) -> PER   (e.g. ڈاکٹر <name> <name>)
        if (t,) in lx.TITLE_SET and i + 1 < n and lexical_tag(tokens[i + 1]) is None \
                and is_urdu(tokens[i + 1]) and tokens[i + 1] not in lx.ORG_SUFFIX_SET:
            j = i + 1
            while j < n and j < i + 4 and lexical_tag(tokens[j]) is None and is_urdu(tokens[j]) \
                    and not any(tuple(tokens[j:j + k]) in G for k in (1, 2)):
                j += 1
            if j > i + 1:
                tags[i + 1] = "B-PER"
                for k in range(i + 2, j):
                    tags[k] = "I-PER"
                i = j
                continue
        # 3) given name + unknown, or unknown + name suffix (X خان)  -> PER
        if i + 1 < n and lexical_tag(tokens[i + 1]) is None and is_urdu(tokens[i + 1]) and (
                t in GIVEN_NAMES or (tokens[i + 1] in NAME_SUFFIX and lexical_tag(t) is None and is_urdu(t))):
            tags[i], tags[i + 1] = "B-PER", "I-PER"
            i += 2
            continue
        i += 1
    return tags


# ---------------------------------------------------------------------------
# sampling + split
# ---------------------------------------------------------------------------
def sample_sentences(seed=SEED):
    arts = load_articles()
    labels, _ = assign_topics(arts)
    rng = random.Random(seed)
    pool = {c: [] for c in TOPICS_USED}
    for a, lab in zip(arts, labels):
        c = CATEGORIES[lab]
        if c not in pool:
            continue
        for s in a:
            line = " ".join(s)
            if 6 <= len(s) <= 40 and not any(b in line for b in BOILERPLATE):
                pool[c].append(s)
    out = []
    for c in TOPICS_USED:
        for s in rng.sample(pool[c], PER_TOPIC):
            out.append({"tokens": s, "topic": c})
    rng.shuffle(out)
    return out


def write_conll(rows, key, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"# topic = {r['topic']}\n")
            for tok, tag in zip(r["tokens"], r[key]):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")


def read_conll(path):
    sents, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if not line:
                if toks:
                    sents.append((toks, tags))
                toks, tags = [], []
                continue
            w, t = line.split("\t")
            toks.append(w)
            tags.append(t)
    if toks:
        sents.append((toks, tags))
    return sents


def build_dataset(seed=SEED, verbose=True):
    rows = sample_sentences(seed)
    for r in rows:
        r["ner"] = ner_tag(r["tokens"])
        r["pos"] = pos_tag(r["tokens"], r["ner"])
    topics = [r["topic"] for r in rows]
    train, rest = train_test_split(rows, test_size=0.30, stratify=topics, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=[r["topic"] for r in rest], random_state=seed)
    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        write_conll(part, "pos", DATA_DIR / f"pos_{name}.conll")
        write_conll(part, "ner", DATA_DIR / f"ner_{name}.conll")

    dist = {"topic": {k: dict(Counter(r["topic"] for r in v)) for k, v in splits.items()},
            "pos": {k: dict(Counter(t for r in v for t in r["pos"])) for k, v in splits.items()},
            "ner": {k: dict(Counter(t for r in v for t in r["ner"])) for k, v in splits.items()},
            "ner_entities": {k: dict(Counter(t[2:] for r in v for t in r["ner"] if t.startswith("B-")))
                             for k, v in splits.items()},
            "sizes": {k: len(v) for k, v in splits.items()},
            "tokens": {k: sum(len(r["tokens"]) for r in v) for k, v in splits.items()},
            "lexicon_sizes": lx.lexicon_sizes(), "gazetteer_sizes": lx.GAZ_COUNTS}
    save_json(dist, RES_DIR / "part2_dataset.json")
    if verbose:
        print("sizes", dist["sizes"], "tokens", dist["tokens"])
        print("POS (all)", Counter(t for r in rows for t in r["pos"]).most_common())
        print("NER entities (all)", Counter(t[2:] for r in rows for t in r["ner"] if t.startswith("B-")))
    return splits, dist


if __name__ == "__main__":
    build_dataset()
