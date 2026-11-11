# i24-7709-NLP-Assignment2 — Neural NLP Pipeline for BBC Urdu

DS5007 Assignment 2. This continues the Assignment 1 BBC Urdu pipeline. Everything is implemented from scratch in **PyTorch + NumPy**. It uses no pretrained models, no Gensim and no HuggingFace, and none of `nn.Transformer`, `nn.MultiheadAttention` or `nn.TransformerEncoder`.

**Repository:** https://github.com/SarmadRazaq/i24-7709-NLP-Assignment2

## Folder layout (mirrors the zip submission)

```
i24-7709-NLP-Assignment2/
├── i24-7709_Assignment2_DS-1A.ipynb   # main notebook, all cells executed
├── report.pdf                          # 2–3 page report (Times New Roman 12 pt, 1.5 spacing)
├── embeddings/
│   ├── tfidf_matrix.npy                # 300 docs × 10,002 vocab (PAD, UNK + 10k words)
│   ├── ppmi_matrix.npy                 # 5,000 × 5,000 PPMI (float16), see note below
│   ├── embeddings_w2v.npy              # C3 Skip-gram, ½(V+U), 10,002 × 100
│   ├── word2idx.json                   # vocabulary shared by all parts
│   └── (extra) embeddings_w2v_raw_C2.npy, embeddings_w2v_d200_C4.npy, word2idx_raw.json
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
├── data/
│   ├── pos_train.conll  pos_test.conll  (+ pos_val.conll)
│   └── ner_train.conll  ner_test.conll  (+ ner_val.conll)
├── corpus/                             # input files: cleaned.txt, raw.txt, Metadata.json
├── src/                                # all source code (imported by the notebook)
│   ├── urdu_nlp.py        Assignment 1 normaliser / tokenizer / stemmer / lemmatizer
│   ├── common.py          paths, corpus loading, vocabulary, Urdu plotting helpers
│   ├── topics.py          keyword-guided 5-way topic assignment for Metadata.json
│   ├── lexicon.py         POS lexicon (≥200 entries per open class) + NER gazetteer
│   ├── part1_counts.py    1.1 TF-IDF, 1.2 PPMI, t-SNE, cosine neighbours
│   ├── word2vec.py        Skip-gram with negative sampling (from scratch)
│   ├── part1_w2v.py       2.1 training, 2.2 neighbours / analogies / C1–C4 + MRR
│   ├── annotate.py        3 sentence sampling, rule-based POS + BIO NER, 70/15/15 split
│   ├── seqmodels.py       2-layer BiLSTM tagger + CRF (forward algorithm, Viterbi)
│   ├── conlleval.py       Python port of the CoNLL conlleval entity scorer
│   ├── part2_train.py     4–5 BiLSTM POS/NER training, evaluation, ablations A1–A4
│   ├── transformer.py     7 attention, multi-head, FFN, sinusoidal PE, Pre-LN block, classifier
│   └── part3_train.py     6–8 dataset, training, evaluation, attention maps, BiLSTM comparison
├── outputs/figures/ outputs/results/   # every plot and JSON result used in the notebook/report
└── report/make_report.py               # builds report.pdf from outputs/results/*.json
```

## Setup

```bash
git clone https://github.com/SarmadRazaq/i24-7709-NLP-Assignment2.git
cd i24-7709-NLP-Assignment2
python -m venv .venv && .venv\Scripts\activate      # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.x (CPU), NumPy 2.x and scikit-learn 1.8. A GPU is not needed. Seeds are fixed (`SEED = 42`).

## Reproducing each part

Run everything from the repository root. Each script writes its artefacts to `embeddings/`, `models/` and `data/`, and its results and plots to `outputs/`.

| Step | Command | Produces | CPU time |
|---|---|---|---|
| Part 1.1–1.2: TF-IDF, PPMI, t-SNE | `python -m src.part1_counts` | `tfidf_matrix.npy`, `ppmi_matrix.npy`, `word2idx.json`, `ppmi_tsne.png` | ~20 s |
| Part 1.2: Skip-gram C2/C3/C4 and evaluation | `python -m src.part1_w2v` | `embeddings_w2v*.npy`, `w2v_loss.png`, `part1_w2v.json` | ~12 min |
| Part 2.3: annotation and splits | `python -m src.annotate` | `data/*.conll`, `part2_dataset.json` | ~5 s |
| Part 2.4–2.5: BiLSTM POS/NER and ablations | `python -m src.part2_train` | `bilstm_pos.pt`, `bilstm_ner.pt`, figures, `part2_results.json` | ~25 min |
| Part 3: Transformer and BiLSTM comparison | `python -m src.part3_train` | `transformer_cls.pt`, figures, `part3_results.json` | ~4 min |
| Report | `python report/make_report.py` | `report.pdf` (needs MS Word for docx→pdf) | ~10 s |

Run them in order: Part 2 needs the C3 embeddings from Part 1, and Part 3 needs `word2idx.json`. Alternatively, open the notebook and use **Run All**:

```bash
jupyter nbconvert --to notebook --execute --inplace i24-7709_Assignment2_DS-1A.ipynb --ExecutePreprocessor.timeout=-1
```

## Design notes and deviations

* **Topic labels.** The `section` field in `Metadata.json` is empty for all 300 articles. Each article is therefore assigned to one of the 5 required categories by scoring the indicative keywords, expanded with Urdu synonyms, in the title (weight 3) and the body (`src/topics.py`). The corpus has very few economy stories, so Economy is a small class (12 articles). Film and entertainment stories are counted under *Health & Society*.
* **PPMI matrix size.** The full 10k × 10k PPMI matrix is 400 MB, which exceeds GitHub's 100 MB file limit. `ppmi_matrix.npy` therefore covers the 5,000 most frequent vocabulary entries, stored as float16 (48 MB). TF-IDF and Word2Vec use the full 10,000-word vocabulary.
* **POS tag set.** The 11 names in the handout plus **PRT** (particle: بھی، ہی، تو، نہیں) make up the 12-tag Universal POS tag set. Auxiliaries are tagged VERB. `cleaned.txt` has no punctuation, so PUNC is not observed.
* **Annotation.** POS and NER labels are *silver* labels from the rule-based tagger and gazetteer (`src/annotate.py`). The BiLSTM models learn to reproduce and generalise these rules.
* **Part 8.2 comparison.** The BiLSTM taggers are sequence labellers, so for a like-for-like comparison a 2-layer BiLSTM *classifier* (C3 init, max-pooling) is trained on the same Part 3 split.

## Headline results

| Part | Result |
|---|---|
| 1 Skip-gram C3 | 8/10 analogies correct; MRR: C1 PPMI 0.431, C2 raw 0.237, C3 cleaned 0.229, C4 d=200 0.263 |
| 2 POS (BiLSTM, fine-tuned) | test accuracy 94.3%, macro-F1 0.921 (frozen embeddings: 80.5% / 0.683) |
| 2 NER (BiLSTM-CRF, fine-tuned) | conlleval entity F1 0.634 (LOC 0.87, ORG 0.67, MISC 0.50, PER 0.24) |
| 2 Ablations (NER F1) | unidirectional 0.536, no dropout 0.643, random init 0.581, softmax instead of CRF 0.620 |
| 3 Transformer | test accuracy 64.4%, macro-F1 0.508 (BiLSTM classifier: 66.7% / 0.504) |

The full notebook run takes about 45 minutes on a laptop CPU.
