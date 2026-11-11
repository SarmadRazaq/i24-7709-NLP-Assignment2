# -*- coding: utf-8 -*-
"""Build report.pdf (Times New Roman 12 pt, 1.5 spacing) from outputs/results/*.json.
Requires python-docx and docx2pdf (MS Word) for the PDF conversion."""
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
REPO = "https://github.com/SarmadRazaq/i24-7709-NLP-Assignment2"


def J(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


p1, w, d2, r2, r3 = J("part1_counts.json"), J("part1_w2v.json"), J("part2_dataset.json"), J("part2_results.json"), J("part3_results.json")

doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = Cm(2.0)
sec.left_margin = sec.right_margin = Cm(2.2)
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.element.rPr.rFonts.set(qn("w:cs"), "Times New Roman")
st.paragraph_format.line_spacing = 1.5
st.paragraph_format.space_after = Pt(2)


def para(text, bold_lead=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    return p


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)


def table(rows, header, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].add_run(h).bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.font.name = "Times New Roman"
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


f = lambda x: f"{100 * x:.1f}"

# ---------------------------------------------------------------- title
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("DS5007 Assignment 2: Neural NLP Pipeline for BBC Urdu")
r.bold = True
r.font.size = Pt(14)
s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.add_run(f"i24-7709 (DS-1A)   |   Repository: {REPO}")

# ---------------------------------------------------------------- overview
heading("1. Overview")
para("This assignment extends the Assignment 1 pipeline (300 BBC Urdu articles, 460k tokens) in three stages. "
     "Part 1 builds count-based (TF-IDF, PPMI) and neural (Skip-gram) word embeddings. Part 2 uses them in BiLSTM "
     "POS and NER taggers, the NER tagger with a CRF layer. Part 3 trains a Transformer encoder for 5-class topic classification. "
     "Everything is implemented from scratch in PyTorch and NumPy, and no pretrained models or built-in Transformer "
     "modules are used. Metadata.json has no section labels, so articles were assigned to the five required categories "
     "by keyword scoring (Politics 48, Sports 94, Economy 12, International 55, Health & Society 91). POS and NER "
     "gold labels are silver annotations produced by a rule-based tagger. It uses a hand-crafted lexicon "
     f"(NOUN {d2['lexicon_sizes']['NOUN']}, VERB {d2['lexicon_sizes']['VERB']}, ADJ {d2['lexicon_sizes']['ADJ']}, "
     f"ADV {d2['lexicon_sizes']['ADV']} entries) and a gazetteer ({d2['gazetteer_sizes']['PER_PK']} Pakistani persons, "
     f"{d2['gazetteer_sizes']['LOC']} locations, {d2['gazetteer_sizes']['ORG']} organisations). The lexicon is built on "
     "the Assignment 1 stemmer.")

# ---------------------------------------------------------------- part 1
heading("2. Part 1 Results: Word Embeddings")
top = p1["top10_tfidf"]
para(f"TF-IDF ({p1['tfidf_shape'][0]} × {p1['tfidf_shape'][1]}) gives clearly topical discriminative words, e.g. "
     "Sports: cricket, match, runs, player, wicket; Politics: police, court, assembly, minister. PPMI (window 5) "
     "neighbours are strong for collocated words (court → court/supreme/judge/hearing) but noisy for general nouns. "
     "In the t-SNE map of the 200 most frequent words, function words, auxiliaries, numbers and sports terms form "
     "separate clusters. The PPMI matrix is stored for the 5,000 most frequent words because of GitHub's file-size limit. "
     f"Skip-gram (V and U matrices, K = 10, unigram^0.75 noise, Adam 1e-3, batch 1024, {w['epochs']} epochs, subsampling 1e-4) "
     f"reduced the loss from about 3.3 to {w['loss_log']['C3'][-1]['loss']:.2f}. C3 solves {w['analogies_correct_C3']}/10 analogies "
     "(e.g. Pakistan:Lahore :: India:Delhi; Lionel:Messi :: Cristiano:Ronaldo; man:woman :: boy:girl). Capital-city and some kinship analogies fail.")
cond = w["conditions"]
names = {"C1": "C1 PPMI", "C2": "C2 SG raw", "C3": "C3 SG cleaned", "C4": "C4 SG d=200"}
table([[names[c], f"{cond[c]['mrr']:.3f}", f"{cond[c]['analogies_correct']}/10"] for c in ["C1", "C2", "C3", "C4"]],
      ["Condition", "MRR (20 pairs)", "Analogies"])
para("PPMI scores the highest MRR because the gold pairs are mostly direct collocates, which PPMI measures by "
     "construction. C3 gives the most coherent neighbours and the best analogy performance, so it is used in Part 2. Raw "
     "text (C2) splits counts across inflected surface forms. Doubling d (C4) gives only a small MRR gain and fewer "
     "correct analogies: 460k tokens are too few to exploit 200 dimensions. Two preprocessing artefacts are visible. The "
     "repeated BBC footer pulls 'Pakistan' toward boilerplate words, and the stemmer merges 'aabadi' (population) with "
     "'-abad' city names.")

# ---------------------------------------------------------------- part 2
heading("3. Part 2 Results: POS Tagging and NER")
pos, ner = r2["pos"], r2["ner"]
para(f"500 sentences (125 per topic from 4 topics) were split 70/15/15 stratified by topic "
     f"({d2['tokens']['train']}/{d2['tokens']['val']}/{d2['tokens']['test']} tokens). The 2-layer BiLSTM (128 units per "
     "direction, dropout 0.5) was initialised with C3 embeddings and trained with Adam (1e-3, wd 1e-4) and early stopping on "
     "validation F1. NER uses a CRF with Viterbi decoding.")
rows = []
for task, m1, m2 in [("POS", "accuracy", "macro_f1"), ("NER", "precision", "f1")]:
    R = r2[task.lower()]
    for mode in ["frozen", "fine-tuned"]:
        tt = R[mode]["test"]
        rows.append([task, mode, f(R[mode]["val_f1"]), f(tt[m1]), f(tt[m2])])
table(rows, ["Task", "Embeddings", "Val F1", "Test acc (POS) / P (NER)", "Test macro-F1 / entity F1"])
cp = pos["confused_pairs"]
cl = ner[ner["best_mode"]]["conlleval"]
para(f"POS reaches {f(pos[pos['best_mode']]['test']['accuracy'])}% token accuracy. The most confused pairs are "
     f"{cp[0]['pair']} ({cp[0]['count']}), {cp[1]['pair']} ({cp[1]['count']}) and {cp[2]['pair']} ({cp[2]['count']}). "
     "Stemming removes the inflections that separate adjectives from nouns, and the rule-based labels for unknown "
     "content words are themselves noisy. With conlleval, NER scores "
     f"LOC F1 {f(cl['LOC']['f1'])}, PER {f(cl['PER']['f1'])}, ORG {f(cl['ORG']['f1'])}, MISC {f(cl['MISC']['f1'])} "
     f"(overall P {f(cl['overall']['precision'])}, R {f(cl['overall']['recall'])}, F1 {f(cl['overall']['f1'])}). "
     "Locations are learned well because they recur, while person names are rare and mostly unseen. Several false "
     "positives are in fact better than the silver labels (e.g. a full three-token name where the rules marked two tokens).")
ab = [["POS", k, f"{f(v['test_f1_mean'])} ± {f(v['test_f1_std'])}"] for k, v in pos["ablations"].items()] + \
     [["NER", k, f"{f(v['test_f1_mean'])} ± {f(v['test_f1_std'])}"] for k, v in ner["ablations"].items()]
table(ab, ["Task", "Setting (3 seeds)", "Test macro-F1 / entity F1"])
pa, na = pos["ablations"], ner["ablations"]
full_p, full_n = pa["Full model (fine-tuned)"]["test_f1_mean"], na["Full model (fine-tuned)"]["test_f1_mean"]
para(f"Backward context matters most for NER (A1: {f(na['A1 unidirectional LSTM']['test_f1_mean'] - full_n)} F1 points), "
     "because Urdu postpositions and titles that mark an entity occur on both sides of it. Pre-trained embeddings help "
     f"both tasks (A3: POS {f(pa['A3 random embedding init']['test_f1_mean'] - full_p)}, NER "
     f"{f(na['A3 random embedding init']['test_f1_mean'] - full_n)}). Removing dropout (A2) has no significant effect "
     "with early stopping on this small dataset. The CRF (A4) gives a small, consistent gain over softmax "
     f"({f(full_n - na['A4 softmax instead of CRF']['test_f1_mean'])} points) with lower variance, because it prevents "
     "invalid transitions such as O → I-PER.")

# ---------------------------------------------------------------- part 3
heading("4. Part 3 Results: Transformer Topic Classifier")
T, B = r3["transformer"], r3["bilstm"]
para(f"The Transformer (4 Pre-LN blocks, 4 heads, d_model 128, d_ff 512, sinusoidal PE, [CLS] + MLP; "
     f"{r3['n_params_transformer']:,} parameters) was trained with AdamW (5e-4, wd 0.01) and a cosine schedule with 50 warm-up "
     f"steps for 20 epochs on 210 articles truncated to 256 tokens. It reaches test accuracy {f(T['test']['accuracy'])}% "
     f"and macro-F1 {f(T['test']['macro_f1'])}. The macro-F1 is pulled down by the tiny Economy class (2 test articles). "
     f"A BiLSTM classifier baseline on the same split gives accuracy {f(B['test']['accuracy'])}% and macro-F1 "
     f"{f(B['test']['macro_f1'])}. The Transformer reached its best validation accuracy at epoch {T['best_val_epoch']}, "
     f"the BiLSTM at epoch {B['best_val_epoch']}. On CPU, one epoch took {T['sec_per_epoch']} s for the Transformer and "
     f"{B['sec_per_epoch']} s for the BiLSTM. Final-layer attention heatmaps show [CLS] concentrating on topical content "
     "words (e.g. cricket/test/player for a sports article), and heads specialise: some attend to a few keywords, others "
     "spread almost uniformly. With only 200–300 articles both models overfit: training accuracy approaches 97% while "
     "validation loss rises after epochs 7–13. The 2.2-point accuracy gap equals one test article. The BiLSTM with pre-trained embeddings is the more appropriate "
     "choice at this scale. It has a stronger inductive bias and starts from distributional knowledge, while the "
     "Transformer has to learn its embeddings and attention patterns from scratch.")

# ---------------------------------------------------------------- conclusion
heading("5. Conclusion")
para("On a small, heavily preprocessed Urdu corpus, simple methods remain competitive. PPMI is a strong baseline for "
     "direct relatedness, while Skip-gram embeddings capture more transferable semantics and clearly help downstream "
     "tagging, especially when fine-tuned. Bidirectional context and pre-trained initialisation were the most important "
     "BiLSTM components, and the CRF gave a smaller but consistent gain for NER. The from-scratch Transformer works and "
     "produces interpretable attention, but with roughly 200 training documents it does not beat a recurrent "
     "baseline with pre-trained embeddings. The main limitations are the silver (rule-based) labels, keyword-derived "
     "topic labels with a very small Economy class, and stemming that over-merges words. Gold annotation and more data "
     "would be the most valuable next steps. All code, data splits, models and results are available at " + REPO + ".")

out_docx = ROOT / "report" / "report.docx"
doc.save(out_docx)
if "--no-pdf" not in sys.argv:
    from docx2pdf import convert
    convert(str(out_docx), str(ROOT / "report.pdf"))
    print("wrote", ROOT / "report.pdf")
