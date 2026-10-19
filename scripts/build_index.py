"""Extract the AIMA textbook into a TF-IDF retrieval index (run once).
Usage: python scripts/build_index.py path/to/textbook.pdf
"""
import re, sys, json, subprocess, joblib, pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF

pdf = sys.argv[1]
out = pathlib.Path(__file__).resolve().parent.parent / "backend" / "data"
pages = [p.get_text() for p in fitz.open(pdf)]

chunks, buf, start = [], "", None
for i, pg in enumerate(pages, 1):
    if i < 30 or i > 1060:          # skip front matter, bibliography, index
        continue
    pg = re.sub(r"[ \t]+", " ", pg)
    pg = re.sub(r"\n\s*\n+", "\n", pg).strip()
    if len(pg) < 200:
        continue
    if start is None:
        start = i
    buf += " " + pg.replace("\n", " ")
    if len(buf) >= 1400:
        chunks.append({"page": f"{start}-{i}" if start != i else str(i), "text": buf.strip()})
        buf, start = "", None
if buf.strip():
    chunks.append({"page": str(start), "text": buf.strip()})

vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5,
                      sublinear_tf=True, max_features=120000)
X = vec.fit_transform([c["text"] for c in chunks])
joblib.dump({"vec": vec, "X": X}, out / "textbook_index.joblib", compress=3)
json.dump(chunks, open(out / "textbook_chunks.json", "w"))
print(f"{len(chunks)} chunks, matrix {X.shape}")
