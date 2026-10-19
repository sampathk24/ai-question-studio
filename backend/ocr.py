"""OCR for scanned question papers and photos.
Engines, first available wins:  RapidOCR (pip only)  ->  Tesseract  ->  Gemini vision (needs an API key).
"""
import io, os, re, shutil, subprocess, tempfile
import numpy as np

import llm

_rapid = None
WIN_TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _tesseract_ok():
    try:
        import pytesseract
    except ImportError:
        return False
    if shutil.which("tesseract"):
        return True
    if os.path.exists(WIN_TESS):
        pytesseract.pytesseract.tesseract_cmd = WIN_TESS
        return True
    return False


def _rapid_ok():
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def engine_name():
    if _rapid_ok():
        return "RapidOCR"
    if _tesseract_ok():
        return "Tesseract"
    if llm._key("GEMINI_API_KEY"):
        return "Gemini vision"
    return None


INSTALL_HELP = ("No OCR engine is available. Install one of these and restart the server:\n"
                "• Easiest: pip install rapidocr-onnxruntime   (pure pip, works on Windows)\n"
                "• Or install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) and run: pip install pytesseract\n"
                "• Or add a GEMINI_API_KEY to your .env file (uses Gemini to read the image)")


# ----------------------------------------------------------------- images
def load_image(data):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")     # fixes sideways phone photos
    w, h = img.size
    if max(w, h) > 2600:
        s = 2600 / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    elif w < 1100:                                                                 # small scans read better when enlarged
        img = img.resize((w * 2, h * 2))
    return img


def render_pdf_pages(data, max_pages=15):
    """Render PDF pages to images: PyMuPDF, else pdftoppm, else the images embedded in the PDF (pypdf)."""
    from PIL import Image
    try:
        try:
            import pymupdf as fz
        except ImportError:
            import fitz as fz
        doc = fz.open(stream=data, filetype="pdf")
        return [Image.open(io.BytesIO(doc[i].get_pixmap(dpi=200).tobytes("png"))).convert("RGB") for i in range(min(doc.page_count, max_pages))]
    except ImportError:
        pass
    if shutil.which("pdftoppm"):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "in.pdf")
            open(p, "wb").write(data)
            subprocess.run(["pdftoppm", "-r", "200", "-l", str(max_pages), "-png", p, os.path.join(d, "pg")], check=True, timeout=120)
            return [Image.open(os.path.join(d, f)).convert("RGB") for f in sorted(os.listdir(d)) if f.startswith("pg")]
    from pypdf import PdfReader
    out = []
    for pg in PdfReader(io.BytesIO(data)).pages[:max_pages]:
        for im in pg.images[:1]:
            out.append(Image.open(io.BytesIO(im.data)).convert("RGB"))
    return out


# ----------------------------------------------------------------- engines
def _group_boxes(result):
    """RapidOCR returns one box per text fragment; merge fragments on the same visual line."""
    items = []
    for box, text, score in result:
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        items.append((sum(ys) / 4, min(xs), max(ys) - min(ys), text, float(score)))
    items.sort()
    lines, cur, cur_y, cur_h = [], [], None, None
    for y, x, h, t, s in items:
        if cur and abs(y - cur_y) < 0.6 * max(cur_h, h):
            cur.append((x, t, s))
        else:
            if cur:
                lines.append(cur)
            cur, cur_y, cur_h = [(x, t, s)], y, h
    if cur:
        lines.append(cur)
    return [" ".join(t for _, t, _ in sorted(l)) for l in lines], [s for l in lines for _, _, s in l]


def _run_rapid(img):
    global _rapid
    from rapidocr_onnxruntime import RapidOCR
    _rapid = _rapid or RapidOCR()
    res, _ = _rapid(np.array(img)[:, :, ::-1])
    if not res:
        return [], 0.0
    lines, scores = _group_boxes(res)
    return lines, sum(scores) / len(scores)


def _run_tesseract(img):
    import pytesseract
    from PIL import ImageOps
    g = ImageOps.autocontrast(img.convert("L"))
    d = pytesseract.image_to_data(g, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT)
    rows, confs = {}, []
    for i, t in enumerate(d["text"]):
        if t.strip():
            rows.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(t)
            if float(d["conf"][i]) >= 0:
                confs.append(float(d["conf"][i]) / 100)
    return [" ".join(v) for _, v in sorted(rows.items())], (sum(confs) / len(confs) if confs else 0.0)


def _run_gemini(img):
    buf = io.BytesIO(); img.save(buf, "PNG")
    txt = llm.vision_text("Transcribe ALL text in this image exactly as written, one line per visual line. Keep question numbers "
                          "(1., 2., (a), (b)) and marks. Do not add commentary.", buf.getvalue())
    if txt is None:
        raise RuntimeError("Gemini could not read the image. " + (llm.last_error or ""))
    return [l.strip() for l in txt.splitlines() if l.strip()], 0.85


def ocr_images(images):
    """-> (list of page texts, average confidence 0-1, engine name)."""
    eng = engine_name()
    if not eng:
        raise ValueError(INSTALL_HELP)
    run = {"RapidOCR": _run_rapid, "Tesseract": _run_tesseract, "Gemini vision": _run_gemini}[eng]
    pages, confs = [], []
    for im in images:
        lines, c = run(im)
        pages.append("\n".join(lines))
        confs.append(c)
    return pages, (sum(confs) / len(confs) if confs else 0.0), eng
