# -*- coding: utf-8 -*-
"""Keyword-guided assignment of every article in Metadata.json to one of 5 topics.

Metadata.json has an empty `section` field for all 300 articles, so categories are
derived from the indicative keywords given in the assignment (expanded with Urdu
synonyms).  Score = 3 * (keyword hits in title) + (keyword hits in cleaned body),
normalised by keyword-list size.  Articles with zero hits are assigned to the
category whose TF-IDF centroid (of keyword-labelled articles) is nearest.
"""
from collections import Counter

import numpy as np

from .common import load_articles, load_metadata
from . import urdu_nlp as un

CATEGORIES = ["Politics", "Sports", "Economy", "International", "Health & Society"]

KEYWORDS = {
    "Politics": """انتخاب انتخابات الیکشن حکومت حکومتی وزیر وزیراعظم وزرا پارلیمان پارلیمنٹ اسمبلی
        سینیٹ سیاست سیاسی سیاستدان جماعت پارٹی تحریک انصاف مسلم لیگ پیپلز عمران نواز شریف بھٹو زرداری
        اپوزیشن ووٹ ووٹر آئین آئینی عدالت عدالتی سپریم ہائیکورٹ جج مقدمہ مقدمے کیس قانون قانونی
        صدر گورنر وزیراعلی کابینہ احتجاج مظاہرہ مظاہرین جیل گرفتار گرفتاری پولیس فوج فوجی جنرل
        اسٹیبلشمنٹ مریم بلاول وفاقی صوبائی حکام سرکاری""",
    "Sports": """کرکٹ میچ ٹیم کھلاڑی کھلاڑیوں سکور رنز وکٹ وکٹیں بیٹنگ بولنگ بولر بلے باز کپتان
        ٹیسٹ ون ڈے ٹی ٹوئنٹی سیریز ٹورنامنٹ ورلڈ کپ فائنل سیمی فٹبال فیفا گول ہاکی اولمپکس
        کھیل کھیلوں سٹیڈیم اننگز سنچری پی ایس ایل آئی سی سی بورڈ کوچ لیگ چیمپیئن گیمز تمغہ
        ایتھلیٹ میس رونالڈو بابر شاہین ریفری""",
    "Economy": """مہنگائی تجارت تجارتی بینک بینکوں جی ڈی پی بجٹ معیشت معاشی اقتصادی قرض قرضہ آئی ایم
        ایف ٹیکس روپے روپیہ ڈالر قیمت قیمتیں قیمتوں تیل پیٹرول گیس بجلی سرمایہ سرمایہ کاری
        برآمدات درآمدات منڈی مارکیٹ کمپنی کمپنیوں کاروبار صنعت سٹاک شرح سود سبسڈی تنخواہ
        ملازمت بے روزگاری اربوں ارب کروڑ لاکھ مالی خسارہ""",
    "International": """اقوام متحدہ معاہدہ معاہدے غیر ملکی خارجہ دوطرفہ تنازع تنازعہ جنگ امریکہ امریکی
        چین چینی انڈیا انڈین ایران ایرانی اسرائیل اسرائیلی سعودی عرب افغانستان افغان روس روسی
        یوکرین ترکی یمن غزہ فلسطین فلسطینی سفیر سفارتی سفارتخانہ ٹرمپ حملہ حملے حملوں ڈرون
        ڈرونز میزائل فضائی دہلی مودی عالمی بین الاقوام سرحد سرحدی نیٹو یورپ برطانیہ""",
    "Health & Society": """ہسپتال بیماری بیماریوں مرض مریض ویکسین سیلاب تعلیم تعلیمی سکول یونیورسٹی طلبا
        طالبہ ڈاکٹر علاج صحت دوا دوائی وائرس کینسر دل خون خوراک غذا جسم ذہنی خواتین عورت بچے بچوں
        ماں شادی خاندان معاشرہ سماجی فلم فلموں اداکار اداکارہ گلوکار گلوکارہ موسیقی گانا گانے
        ڈرامہ فنکار بالی ووڈ ہالی وڈ شو ثقافت مذہب مذہبی تشدد قتل حادثہ زلزلہ موسم""",
}


def _norm_set(text):
    out = set()
    for w in text.split():
        w = un.remove_diacritics(un.normalize_unicode(w))
        out.add(w)
        out.add(un.process_token(w))
    return out


KW = {c: _norm_set(t) for c, t in KEYWORDS.items()}


def _title_tokens(title):
    toks = un.tokenize(un.normalize_text(title))
    return [un.process_token(t) for t in toks] + toks


def assign_topics(arts=None, meta=None):
    """Returns list[int] (category index per article) and per-article score matrix."""
    arts = arts if arts is not None else load_articles()
    meta = meta if meta is not None else load_metadata()
    n = len(arts)
    scores = np.zeros((n, len(CATEGORIES)))
    for i in range(n):
        body = Counter(t for s in arts[i] for t in s)
        title = Counter(_title_tokens(meta[i]["title"]))
        for j, c in enumerate(CATEGORIES):
            hits = 3 * sum(title[w] for w in KW[c]) + sum(body[w] for w in KW[c])
            scores[i, j] = hits / (sum(body.values()) ** 0.5 + 1)
    labels = scores.argmax(1)
    zero = scores.sum(1) == 0
    if zero.any():  # centroid fallback using bag-of-words cosine
        from .common import build_vocab
        stoi, _, _ = build_vocab([t for a in arts for s in a for t in [s]])
        X = np.zeros((n, len(stoi)))
        for i, a in enumerate(arts):
            for s in a:
                for t in s:
                    X[i, stoi.get(t, 1)] += 1
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        cents = np.stack([X[(labels == j) & ~zero].mean(0) for j in range(len(CATEGORIES))])
        labels[zero] = (X[zero] @ cents.T).argmax(1)
    return labels.tolist(), scores


if __name__ == "__main__":
    labs, _ = assign_topics()
    print(Counter(CATEGORIES[l] for l in labs))
