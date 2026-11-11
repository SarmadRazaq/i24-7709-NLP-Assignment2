# -*- coding: utf-8 -*-
"""Shared Urdu NLP module: normalization, tokenizer, stemmer, lemmatizer, n-gram LMs.
Used by both the notebook pipeline and the Streamlit app (app.py)."""
# -*- coding: utf-8 -*-
"""Custom Urdu preprocessing: normalization, tokenizer, stemmer, lemmatizer.
All rule-based, from scratch (no pretrained NLP libraries)."""
import re, unicodedata

# ---------------------------------------------------------------------------
# 1. DIACRITICS  (Arabic combining marks used as Urdu aeraab / diacritics)
# ---------------------------------------------------------------------------
URDU_DIACRITICS = "".join([
    "ؐ", "ؑ", "ؒ", "ؓ", "ؔ", "ؕ", "ؖ",
    "ؗ", "ؘ", "ؙ", "ؚ",
    "ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ",
    "ْ", "ٓ", "ٔ", "ٕ", "ٖ", "ٗ", "٘",
    "ٰ",                                   # superscript alef (khari zabar)
    "ۖ", "ۗ", "ۘ", "ۙ", "ۚ", "ۛ", "ۜ",
    "۟", "۠", "ۡ", "ۢ", "ۣ", "ۤ", "ۥ",
    "ۦ", "ۧ", "ۨ", "۪", "۫", "۬", "ۭ",
])
_diac_re = re.compile("[" + re.escape(URDU_DIACRITICS) + "]")

def remove_diacritics(text: str) -> str:
    """3.1  Remove Urdu diacritics so both forms map to one word (عَلِم -> علم)."""
    return _diac_re.sub("", text)

# ---------------------------------------------------------------------------
# 2. CHARACTER / UNICODE NORMALIZATION  (Arabic variants -> Urdu canonical)
# ---------------------------------------------------------------------------
CHAR_MAP = {
    "ي": "ی",  # Arabic yeh ي  -> Urdu yeh ی
    "ى": "ی",  # alef maksura ى -> ی
    # NOTE: bari yeh (ے, U+06D2) is a DISTINCT Urdu letter (کے، ہے، نے) and is
    # deliberately NOT mapped to choti yeh, to avoid corrupting common words.
    "ك": "ک",  # Arabic kaf ك  -> Urdu kaf ک
    "ګ": "گ",  # gaf variant -> گ
    "ة": "ہ",  # teh marbuta ة -> heh goal ہ
    "ۃ": "ہ",  # teh marbuta goal -> ہ
    "ه": "ہ",  # Arabic heh ه -> Urdu gol heh ہ
    "ھ": "ھ",  # heh doachashmee ھ keep
    "أ": "ا",  # alef with hamza above أ -> ا
    "إ": "ا",  # alef with hamza below إ -> ا
    "آ": "ا",  # alef madda آ -> ا  (optional; keeps sparsity low)
    "ؤ": "و",  # waw with hamza ؤ -> و ... handled carefully below
    "ئ": "ی",  # yeh with hamza ئ -> ی
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}
# We keep madda (آ) mapping optional; comment out if too aggressive.
del CHAR_MAP["آ"]  # keep آ distinct (common word-initial), safer
del CHAR_MAP["ؤ"]
del CHAR_MAP["ئ"]

# zero-width / formatting chars to drop (except we normalise ZWNJ to nothing)
ZERO_WIDTH = ["​", "‌", "‍", "‎", "‏",
              "﻿", "­", "⁠"]

def normalize_unicode(text: str) -> str:
    """3 / 3.5  NFC normalise, canonicalise Arabic->Urdu letters, strip
    zero-width & directional marks, convert Arabic/Urdu digits to ASCII."""
    text = unicodedata.normalize("NFC", text)
    for z in ZERO_WIDTH:
        text = text.replace(z, "")
    text = text.translate({ord(k): v for k, v in CHAR_MAP.items()})
    return text

# ---------------------------------------------------------------------------
# 3. NOISE + NON-URDU REMOVAL
# ---------------------------------------------------------------------------
URL_RE      = re.compile(r"https?://\S+|www\.\S+|\S+\.(?:com|org|net|pk|co|uk)\b\S*")
EMAIL_RE    = re.compile(r"\S+@\S+")
HTML_RE     = re.compile(r"<[^>]+>")
EMOJI_RE    = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF" "\U0000FE00-\U0000FE0F" "]+", flags=re.UNICODE)
LATIN_RE    = re.compile(r"[A-Za-z][A-Za-z0-9'’.\-]*")   # English / Roman Urdu tokens
HASHTAG_RE  = re.compile(r"[#@]\S+")

# Urdu letters block (Arabic + Urdu extensions) plus space & Urdu punctuation & digits
URDU_KEEP_RE = re.compile(
    r"[^؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"
    r"\s0-9۔،؟؛!٪ء\-]")

def remove_noise(text: str) -> str:
    """3.2  Remove URLs, emails, html, emojis, hashtags/handles."""
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    return text

def remove_non_urdu(text: str) -> str:
    """3.3  Remove Latin (English/Roman Urdu) words and any non-Urdu symbol."""
    text = LATIN_RE.sub(" ", text)
    text = URDU_KEEP_RE.sub(" ", text)   # drop leftover foreign symbols
    return text

# ---------------------------------------------------------------------------
# 4. SENTENCE SEGMENTATION  (3.4)
# ---------------------------------------------------------------------------
SENT_END = "۔؟!"
_sent_split_re = re.compile(r"([" + SENT_END + r"])")

def segment_sentences(text: str):
    """Split on Urdu sentence terminators, keeping meaningful sentences."""
    parts = _sent_split_re.split(text)
    sents, buf = [], ""
    for p in parts:
        if p in SENT_END:
            s = (buf + p).strip()
            if s:
                sents.append(s)
            buf = ""
        else:
            buf += p
    if buf.strip():
        sents.append(buf.strip())
    return [s for s in sents if len(s.split()) >= 2]

# ---------------------------------------------------------------------------
# 5. WHITESPACE NORMALIZATION  (3.5)
# ---------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r" *([۔،؟!؛]) *", r"\1 ", text)
    return text.strip()

# ---------------------------------------------------------------------------
# FULL NORMALIZATION PIPELINE
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_diacritics(text)
    text = remove_noise(text)
    text = remove_non_urdu(text)
    text = normalize_whitespace(text)
    return text

# ---------------------------------------------------------------------------
# 6. CUSTOM URDU TOKENIZER  (3.6)
# ---------------------------------------------------------------------------
NUM_TOKEN = "<NUM>"
_num_re = re.compile(r"\d+([.,]\d+)*")
# postpositions that should always be their own token
POSTPOSITIONS = {"کے", "کا", "کی", "کو", "نے", "سے", "میں", "پر", "تک",
                 "پہ", "والے", "والا", "والی"}
PUNCT = set("۔،؟؛!٪ء-")

def tokenize(text: str):
    """Word/punctuation tokenizer. Numbers -> <NUM>; punctuation split off;
    postpositions kept as separate tokens (they already are, but we guard
    against them being glued to words by earlier noise)."""
    text = _num_re.sub(" " + NUM_TOKEN + " ", text)
    # pad Urdu punctuation so it becomes standalone tokens
    text = re.sub(r"([۔،؟؛!٪])", r" \1 ", text)
    raw = text.split()
    tokens = []
    for tok in raw:
        if tok == NUM_TOKEN or tok in PUNCT:
            tokens.append(tok)
            continue
        # strip leading/trailing punctuation still attached
        core = tok.strip("".join(PUNCT))
        if core:
            tokens.append(core)
    return tokens

# ---------------------------------------------------------------------------
# 7. CUSTOM URDU STEMMER  (3.6) - longest-suffix stripping
# ---------------------------------------------------------------------------
# ordered longest-first; remove common inflectional / derivational suffixes
# ordered longest-first. We avoid the most destructive endings ("ان", bare "ا",
# bare "و") that tend to shred proper nouns (پاکستان) and valid roots (دنیا).
STEM_SUFFIXES = [
    "یات", "یوں", "یاں", "ئیں", "تیوں", "تیں",
    "یہ", "گی", "گا", "گے", "تا", "تے", "تی", "نا", "نی",
    "وں", "ات", "ین", "یں", "ہا",
    "ی", "ے", "ں",
]
def stem(word: str) -> str:
    """Longest-suffix stripping; keeps a minimum stem length of 3 so that
    short words and proper nouns are left intact."""
    if word == NUM_TOKEN or word in POSTPOSITIONS:
        return word
    for suf in STEM_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: len(word) - len(suf)]
    return word

# ---------------------------------------------------------------------------
# 8. CUSTOM URDU LEMMATIZER  (3.6) - plural & gender normalization
# ---------------------------------------------------------------------------
# explicit irregular map (root/singular form)
LEMMA_MAP = {
    "لڑکوں": "لڑکا", "لڑکے": "لڑکا", "لڑکیاں": "لڑکی", "لڑکیوں": "لڑکی",
    "کتابیں": "کتاب", "کتابوں": "کتاب",
    "عورتیں": "عورت", "عورتوں": "عورت",
    "مردوں": "مرد", "بچوں": "بچہ", "بچے": "بچہ", "بچی": "بچہ",
    "دنوں": "دن", "راتوں": "رات", "سالوں": "سال", "لوگوں": "لوگ",
    "ملکوں": "ملک", "ممالک": "ملک", "شہروں": "شہر", "گھروں": "گھر",
    "خواتین": "عورت", "حضرات": "حضرت", "علاقوں": "علاقہ", "علاقے": "علاقہ",
}
# regular plural / oblique / gender rules (applied after map miss)
def lemmatize(word: str) -> str:
    if word == NUM_TOKEN or word in POSTPOSITIONS:
        return word
    if word in LEMMA_MAP:
        return LEMMA_MAP[word]
    # regular plural / oblique markers -> singular base.
    # (Generic gender rule ے->ا is intentionally omitted: it corrupts verb
    #  forms like ہوئے; gender pairs are handled through LEMMA_MAP instead.)
    for suf, repl in [("یوں", "ی"), ("یاں", "ی"), ("ئیں", ""),
                       ("وں", ""), ("یں", ""), ("ات", "")]:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: len(word) - len(suf)] + repl
    return word

# combined token normaliser used to build cleaned.txt
def process_token(tok: str, do_stem=True, do_lemma=True) -> str:
    if tok == NUM_TOKEN:
        return tok
    t = tok
    if do_lemma:
        t = lemmatize(t)
    if do_stem:
        t = stem(t)
    return t

# ---------------------------------------------------------------------------
# END-TO-END: raw article body -> list[ list[token] ]  (sentences of tokens)
# ---------------------------------------------------------------------------
def process_article(body: str, do_stem=True, do_lemma=True):
    norm = normalize_text(body)
    sentences = segment_sentences(norm)
    out = []
    for s in sentences:
        toks = [process_token(t, do_stem, do_lemma) for t in tokenize(s)]
        toks = [t for t in toks if t and t not in PUNCT]
        if len(toks) >= 2:
            out.append(toks)
    return out


# -*- coding: utf-8 -*-
"""N-gram language models (unigram / bigram / trigram) built from scratch.
Add-k (Laplace when k=1) smoothing, stupid-style backoff for the trigram model,
sampling-based generation with BBC-style length/EOS constraints, and perplexity."""
import math, random
from collections import defaultdict, Counter

BOS = "<s>"      # sentence start
EOS = "</s>"     # sentence end  (EOS token)
UNK = "<UNK>"

class NGramLM:
    def __init__(self, k=0.05, unk_min_count=1):
        self.k = k                     # add-k smoothing constant
        self.unk_min_count = unk_min_count
        self.uni = Counter()
        self.bi = defaultdict(Counter)     # bi[w1][w2]
        self.tri = defaultdict(Counter)    # tri[(w1,w2)][w3]
        self.vocab = set()
        self.N = 0                         # total unigram tokens

    # ---------------- training ----------------
    def _build_vocab(self, sentences):
        c = Counter()
        for s in sentences:
            c.update(s)
        self.vocab = {w for w, n in c.items() if n >= self.unk_min_count}
        self.vocab |= {BOS, EOS, UNK}

    def _map(self, w):
        return w if w in self.vocab else UNK

    def fit(self, sentences):
        """sentences: list of list[token] (already preprocessed)."""
        self._build_vocab(sentences)
        for s in sentences:
            toks = [self._map(w) for w in s]
            padded = [BOS, BOS] + toks + [EOS]
            for w in padded:
                self.uni[w] += 1
            for w1, w2 in zip(padded, padded[1:]):
                self.bi[w1][w2] += 1
            for w1, w2, w3 in zip(padded, padded[1:], padded[2:]):
                self.tri[(w1, w2)][w3] += 1
        self.N = sum(self.uni.values())
        self.V = len(self.vocab)
        return self

    # ---------------- smoothed probabilities ----------------
    def p_unigram(self, w):
        w = self._map(w)
        return (self.uni[w] + self.k) / (self.N + self.k * self.V)

    def p_bigram(self, w2, w1):
        w1, w2 = self._map(w1), self._map(w2)
        num = self.bi[w1][w2] + self.k
        den = self.uni[w1] + self.k * self.V
        return num / den

    def p_trigram(self, w3, w1, w2):
        w1, w2, w3 = self._map(w1), self._map(w2), self._map(w3)
        num = self.tri[(w1, w2)][w3] + self.k
        den = sum(self.tri[(w1, w2)].values()) + self.k * self.V
        return num / den

    def p_backoff(self, w3, w1, w2, alpha=0.4):
        """Trigram -> Bigram -> Unigram *stupid backoff* (Brants et al. 2007).
        If the trigram was seen, use its relative frequency; else fall back to
        the bigram (scaled by alpha); else to the add-k smoothed unigram
        (scaled by alpha^2). The unigram term is always > 0, so no n-gram ever
        receives zero probability."""
        w1, w2, w3 = self._map(w1), self._map(w2), self._map(w3)
        tri_ctx = sum(self.tri[(w1, w2)].values())
        if tri_ctx > 0 and self.tri[(w1, w2)][w3] > 0:
            return self.tri[(w1, w2)][w3] / tri_ctx
        bi_ctx = sum(self.bi[w2].values())
        if bi_ctx > 0 and self.bi[w2][w3] > 0:
            return alpha * self.bi[w2][w3] / bi_ctx
        return alpha * alpha * self.p_unigram(w3)

    # ---------------- perplexity ----------------
    def perplexity(self, sentences, model="trigram"):
        log_sum, N = 0.0, 0
        for s in sentences:
            toks = [self._map(w) for w in s]
            padded = [BOS, BOS] + toks + [EOS]
            if model == "unigram":
                for w in padded[2:]:
                    log_sum += math.log(self.p_unigram(w)); N += 1
            elif model == "bigram":
                for w1, w2 in zip(padded[1:], padded[2:]):
                    log_sum += math.log(self.p_bigram(w2, w1)); N += 1
            else:  # trigram w/ backoff
                for w1, w2, w3 in zip(padded, padded[1:], padded[2:]):
                    log_sum += math.log(self.p_backoff(w3, w1, w2)); N += 1
        return math.exp(-log_sum / N) if N else float("inf")

    # ---------------- sampling helpers ----------------
    @staticmethod
    def _sample(counter, temperature=1.0, rng=random):
        items = [(w, c) for w, c in counter.items() if w not in (BOS, UNK)]
        if not items:
            return None
        words = [w for w, _ in items]
        weights = [c ** (1.0 / temperature) for _, c in items]
        return rng.choices(words, weights=weights, k=1)[0]

    def next_bigram(self, w1, temperature=1.0, rng=random):
        return self._sample(self.bi[self._map(w1)], temperature, rng)

    def next_trigram_backoff(self, w1, w2, temperature=1.0, rng=random):
        w1, w2 = self._map(w1), self._map(w2)
        cand = self.tri[(w1, w2)]
        if sum(c for w, c in cand.items() if w not in (BOS, UNK)) > 0:
            return self._sample(cand, temperature, rng)
        cand = self.bi[w2]                              # backoff -> bigram
        if sum(c for w, c in cand.items() if w not in (BOS, UNK)) > 0:
            return self._sample(cand, temperature, rng)
        return self._sample(self.uni, temperature, rng)  # backoff -> unigram


# ---------------------------------------------------------------------------
# ARTICLE GENERATION with constraints
# ---------------------------------------------------------------------------
def generate_article(model, seed_tokens, mode="trigram",
                     min_words=200, target_words=240, max_words=300,
                     min_sentences=5, temperature=0.8, seed=None):
    """Generate a BBC-style article. Stops after >= target words at a sentence
    boundary; hard stop at max_words (forced termination if no EOS)."""
    rng = random.Random(seed)
    out = list(seed_tokens)
    sentences_done = 0
    # generation history uses last two tokens
    hist = [BOS, BOS] + list(seed_tokens)
    while True:
        w1, w2 = hist[-2], hist[-1]
        if mode == "bigram":
            nxt = model.next_bigram(w2, temperature, rng)
        else:
            nxt = model.next_trigram_backoff(w1, w2, temperature, rng)
        if nxt is None:
            nxt = EOS
        if nxt == EOS:
            sentences_done += 1
            out.append("۔")
            hist = [BOS, BOS]        # restart context for next sentence
            wc = len([t for t in out if t not in ("۔", "،")])
            if wc >= target_words and sentences_done >= min_sentences:
                break
            if wc >= max_words:
                break
            continue
        out.append(nxt)
        hist.append(nxt)
        wc = len([t for t in out if t not in ("۔", "،")])
        if wc >= max_words:                      # forced termination
            out.append("۔"); sentences_done += 1
            break
    return out, sentences_done


def detokenize(tokens):
    """Join tokens into a readable RTL Urdu string."""
    text = ""
    for t in tokens:
        if t in ("۔", "،", "؟", "!", "؛"):
            text = text.rstrip() + t + " "
        else:
            text += t + " "
    return text.strip()
