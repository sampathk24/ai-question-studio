"""Read an uploaded file (PDF, PPTX, DOCX, TXT/MD or an image) into a common 'document' structure.

doc = {id, name, kind, pages, lines[], units[{ref,title,text}], words, ocr{engine,confidence,pages}|None, notes[]}
  lines -> clean text lines, used to find numbered questions (question-paper mode)
  units -> slides / pages / sections, used to write new questions (study-material mode)
No extra libraries are needed for PPTX/DOCX: they are zip files of XML and are read directly.
"""
import io, re, uuid, zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict

import engine, ocr, paper

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED = {".pdf", ".pptx", ".docx", ".txt", ".md"} | IMAGE_EXT
LEGACY = {".doc": "Word 97-2003 (.doc)", ".ppt": "PowerPoint 97-2003 (.ppt)", ".xls": "Excel (.xls)", ".rtf": "RTF"}
MAX_CHARS, MAX_UNITS, OCR_PAGES = 400_000, 250, 15
STORE = OrderedDict()                                   # recent documents, kept in memory for follow-up questions


def put(doc):
    doc["id"] = uuid.uuid4().hex[:10]
    STORE[doc["id"]] = doc
    while len(STORE) > 12:
        STORE.popitem(last=False)
    return doc


def get(doc_id):
    return STORE.get(doc_id)


def _clean(s):
    return re.sub(r"[ \t\u00a0]+", " ", s or "").strip()


# ----------------------------------------------------------------- unit builders
def units_from_lines(lines, prefix="Section", size=900):
    units, buf, n = [], [], 0
    for ln in lines:
        buf.append(ln)
        n += len(ln)
        if n >= size:
            units.append({"ref": f"{prefix} {len(units) + 1}", "title": "", "text": "\n".join(buf)})
            buf, n = [], 0
    if buf:
        units.append({"ref": f"{prefix} {len(units) + 1}", "title": "", "text": "\n".join(buf)})
    return units


# ----------------------------------------------------------------- PPTX
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"


def _paras(el):
    return [_clean("".join(t.text or "" for t in p.iter(_A + "t"))) for p in el.iter(_A + "p")]


def read_pptx(data):
    z = zipfile.ZipFile(io.BytesIO(data))
    num = lambda n: int(re.search(r"(\d+)\.xml$", n).group(1))
    slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=num)
    units, lines = [], []
    for i, name in enumerate(slides, 1):
        root, title, body = ET.fromstring(z.read(name)), None, []
        for sp in root.iter(_P + "sp"):
            ph = sp.find(f"{_P}nvSpPr/{_P}nvPr/{_P}ph")
            ps = [p for p in _paras(sp) if p]
            if ph is not None and ph.get("type") in ("title", "ctrTitle") and title is None:
                title = " ".join(ps)
            else:
                body += ps
        for tr in root.iter(_A + "tr"):                                 # tables
            cells = [_clean("".join(t.text or "" for t in tc.iter(_A + "t"))) for tc in tr.iter(_A + "tc")]
            if any(cells):
                body.append(" | ".join(cells))
        notes = []
        rel = f"ppt/slides/_rels/slide{num(name)}.xml.rels"
        if rel in z.namelist():
            m = re.search(r'Target="[^"]*?(notesSlide\d+\.xml)"', z.read(rel).decode("utf8", "ignore"))
            if m and f"ppt/notesSlides/{m.group(1)}" in z.namelist():
                nroot = ET.fromstring(z.read(f"ppt/notesSlides/{m.group(1)}"))
                for sp in nroot.iter(_P + "sp"):
                    ph = sp.find(f"{_P}nvSpPr/{_P}nvPr/{_P}ph")
                    if ph is not None and ph.get("type") == "body":
                        notes += [p for p in _paras(sp) if p]
        text = "\n".join(body + notes)
        if title or text:
            units.append({"ref": f"Slide {i}", "title": title or "", "text": text})
            lines += ([title] if title else []) + body
    return units, lines, len(slides)


# ----------------------------------------------------------------- DOCX
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx(data):
    z = zipfile.ZipFile(io.BytesIO(data))
    body = ET.fromstring(z.read("word/document.xml")).find(_W + "body")
    units, lines, cur = [], [], None
    for ch in body:
        if ch.tag == _W + "p":
            text = _clean("".join(t.text or "" if t.tag == _W + "t" else " " for t in ch.iter() if t.tag in (_W + "t", _W + "tab", _W + "br")))
            if not text:
                continue
            st = ch.find(f"{_W}pPr/{_W}pStyle")
            style = (st.get(_W + "val") if st is not None else "") or ""
            if re.match(r"(?i)heading|title", style) and len(text) < 120:
                cur = {"ref": text, "title": text, "text": ""}
                units.append(cur)
                lines.append(text)
                continue
            lines.append(text)
        elif ch.tag == _W + "tbl":
            rows = []
            for tr in ch.iter(_W + "tr"):
                cells = [_clean(" ".join("".join(t.text or "" for t in p.iter(_W + "t")) for p in tc.iter(_W + "p"))) for tc in tr.iter(_W + "tc")]
                if any(cells):
                    rows.append(" | ".join(cells))
            lines += rows
            text = "\n".join(rows)
        else:
            continue
        if cur is None:
            cur = {"ref": "Section 1", "title": "", "text": ""}
            units.append(cur)
        cur["text"] += ("\n" if cur["text"] else "") + text
    if len(units) <= 1 and sum(len(u["text"]) for u in units) > 1800:     # no headings: split into chunks
        units = units_from_lines(lines)
    return units, lines, max(1, len(lines) // 40)


# ----------------------------------------------------------------- ingest
def ingest(data, filename):
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext in LEGACY:
        raise ValueError(f"{LEGACY[ext]} files can't be read directly. Open it and use Save As .docx / .pptx / .pdf, then upload again.")
    if ext not in SUPPORTED:
        raise ValueError("Unsupported file type. Please upload a PDF, PowerPoint (.pptx), Word (.docx), TXT or an image (PNG/JPG).")
    notes, ocr_info, pages = [], None, 1
    try:
        if ext == ".pptx":
            units, lines, pages = read_pptx(data)
            kind = "PowerPoint"
        elif ext == ".docx":
            units, lines, pages = read_docx(data)
            kind = "Word document"
        elif ext in (".txt", ".md"):
            lines = [l.strip() for l in data.decode("utf-8", "ignore").splitlines() if l.strip()]
            units, kind = units_from_lines(lines), "Text file"
        elif ext in IMAGE_EXT:
            kind = "Image"
            texts, conf, eng = ocr.ocr_images([ocr.load_image(data)])
            lines = paper.clean_lines(texts)
            units, ocr_info = units_from_lines(lines, "Image"), {"engine": eng, "confidence": round(conf, 2), "pages": 1}
        else:
            kind = "PDF"
            pgs, pages = paper.extract_pages(data)
            lines = paper.clean_lines(pgs)
            chars = sum(len(l) for l in lines)
            if chars < 40 * max(1, len(pgs)):                           # almost no text layer -> scanned PDF
                imgs = ocr.render_pdf_pages(data, OCR_PAGES)
                if not imgs:
                    raise ValueError("This PDF has no readable text and no pages I can scan.")
                texts, conf, eng = ocr.ocr_images(imgs)
                pgs, lines = texts, paper.clean_lines(texts)
                ocr_info = {"engine": eng, "confidence": round(conf, 2), "pages": len(imgs)}
                kind = "Scanned PDF"
                if pages > OCR_PAGES:
                    notes.append(f"Only the first {OCR_PAGES} pages were scanned.")
            units = [{"ref": f"Page {i}", "title": next((l for l in p.splitlines() if 3 < len(l.strip()) < 80), "").strip(), "text": "\n".join(l.strip() for l in p.splitlines() if l.strip())}
                     for i, p in enumerate(pgs, 1) if len(p.strip()) > 40]
    except (zipfile.BadZipFile, ET.ParseError, KeyError):
        raise ValueError("That file looks damaged or isn't a valid " + ext.upper().strip(".") + " file.")
    lines = [l for l in lines if l.strip()]
    units = [u for u in units if len(u["text"].strip()) + len(u["title"]) >= 25][:MAX_UNITS]
    if sum(len(l) for l in lines) < 40:
        raise ValueError("I couldn't find any readable text in this file.")
    if ocr_info and ocr_info["confidence"] < 0.7:
        notes.append(f"The scan quality is low (OCR confidence {round(ocr_info['confidence'] * 100)}%), so some words may be misread. A clearer or straighter scan gives better results.")
    words = sum(len(u["text"].split()) for u in units)
    return put({"name": filename, "kind": kind, "pages": pages, "lines": lines, "units": units, "words": words,
                "ocr": ocr_info, "notes": notes, "raw": "\n".join(lines)[:MAX_CHARS]})


def guess_kind(doc):
    """'paper' if the text looks like a numbered list of questions, else 'material'."""
    items = paper.split_numbered(doc["lines"])
    if len(items) < 3:
        return "material"
    def first_text(it):
        return " ".join((it["stem"] or [it["subs"][0]["lines"][0] if it["subs"] and it["subs"][0]["lines"] else ""])[:1])
    qlike = 0
    for it in items:
        t = _clean(first_text(it))
        fw = re.sub(r"[^a-z]", "", t.split()[0].lower()) if t.split() else ""
        if fw in engine.ALLOWED_START or fw in engine.BANNED_START or t.endswith("?") or it["subs"]:
            qlike += 1
    avg_len = sum(len(" ".join(it["stem"]).split()) for it in items) / len(items)
    return "paper" if qlike / len(items) >= 0.6 and avg_len < 60 else "material"
