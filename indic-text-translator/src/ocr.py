"""OCR fallback recogniser (Tesseract) + translation. Used by the demo; the classical
template matcher in match.py remains the 'our method' column of the accuracy table."""
from __future__ import annotations

import os
import sys
import unicodedata

import cv2
import numpy as np
import pytesseract

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA = os.path.join(ROOT, "tessdata")
os.environ["TESSDATA_PREFIX"] = TESSDATA
LANG = {"devanagari": "hin", "telugu": "tel", "kannada": "kan"}
GT_LANG = {"devanagari": "hi", "telugu": "te", "kannada": "kn"}


BLOCKS = {"devanagari": (0x0900, 0x097F), "telugu": (0x0C00, 0x0C7F), "kannada": (0x0C80, 0x0CFF)}


def _page_and_lines(img_bgr):
    s = preprocess(img_bgr)
    binary = s["text"]
    ink = binary > 0 if binary.mean() < 127 else binary == 0
    page = np.where(ink, 0, 255).astype(np.uint8)
    crops = []
    for (a, b) in s["lines"]:
        crop = page[max(0, a - 8):b + 8]
        crops.append(cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255))
    return crops


def _read(crops, lang):
    """-> (list of line texts, mean word confidence)."""
    out, confs = [], []
    for c in crops:
        d = pytesseract.image_to_data(c, lang=lang, config="--psm 7",
                                      output_type=pytesseract.Output.DICT)
        words = [(w, float(cf)) for w, cf in zip(d["text"], d["conf"]) if w.strip() and float(cf) >= 0]
        out.append(unicodedata.normalize("NFC", " ".join(w for w, _ in words)))
        confs += [cf for _, cf in words]
    return out, (float(np.mean(confs)) if confs else 0.0)


def detect_script(crops, min_share=0.7, min_conf=50.0):
    """Probe with all languages at once, then vote by Unicode block.
    Returns (script or None, share per script, mean confidence)."""
    lines, conf = _read(crops, "hin+tel+kan+eng")
    chars = [ch for l in lines for ch in l if ch.isalpha()]
    if not chars:
        return None, {}, conf
    share = {k: sum(lo <= ord(ch) <= hi for ch in chars) / len(chars) for k, (lo, hi) in BLOCKS.items()}
    best = max(share, key=share.get)
    if share[best] >= min_share and conf >= min_conf:
        return best, share, conf
    return None, share, conf


def recognise(img_bgr, script=None):
    """Auto-detect the script (unless given) and read it with that script's model.
    Returns dict(script, lines, share, conf); script is None when not one of the 3."""
    crops = _page_and_lines(img_bgr)
    if script is None:
        script, share, conf = detect_script(crops)
        if script is None:
            return dict(script=None, lines=[], share=share, conf=conf)
    else:
        share, conf = {}, 0.0
    lines, conf = _read(crops, LANG[script])
    return dict(script=script, lines=lines, share=share, conf=conf)


def ocr_lines(img_bgr, script):
    return recognise(img_bgr, script)["lines"]


def translate(text, script):
    """Google first (2 tries); fall back to MyMemory if Google rate-limits/blocks this network."""
    import time
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    src = GT_LANG[script]
    for attempt in range(2):
        try:
            out = GoogleTranslator(source=src, target="en").translate(text)
            if out:
                return out
        except Exception:
            time.sleep(1.5)
    for _ in range(2):
        try:
            out = MyMemoryTranslator(source=src + "-IN", target="en-GB").translate(text)
            if out and out.strip():
                return out
        except Exception:
            pass
        time.sleep(1)
    return "(translation service unavailable)"
