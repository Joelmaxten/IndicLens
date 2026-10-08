"""Evaluate recognition: character error rate vs ground truth (like the reference's
Table I, but at text level).

Rows: script. Columns: template font vs page font (same / different), clean / camera.
CER = total edit distance / total ground-truth characters (lower is better).
"""
from __future__ import annotations

import glob
import os
import sys
import unicodedata

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.library import load
from src.match import decode_line, edit_distance, recognize_unit
from src.preprocess import preprocess
from src.segment import segment_page

SCRIPTS = ["devanagari", "telugu", "kannada"]
FONTS = ["sans", "serif"]
_libs = {}
_pages = {}


def get_lib(script, font):
    k = (script, font)
    if k not in _libs:
        _libs[k] = load(f"cache/lib_{script}_{font}.npz")
    return _libs[k]


def get_page(script, font, kind):
    k = (script, font, kind)
    if k not in _pages:
        ext = "png" if kind == "clean" else "jpg"
        s = preprocess(cv2.imread(f"data/{script}_{font}_{kind}.{ext}"))
        _pages[k] = segment_page(s["text"], s["lines"])
    return _pages[k]


def truth(script, font):
    with open(f"data/{script}_{font}.txt", encoding="utf-8") as fh:
        return [unicodedata.normalize("NFC", l.strip()) for l in fh if l.strip()]


def read_page(script, page_font, kind, libs, method="xor", lam=0.0):
    out = []
    for L in get_page(script, page_font, kind):
        labels = [recognize_unit(libs, u.mask, method, lam)[0] for u in L["units"]]
        out.append(decode_line(L["units"], labels, L["script"] == "devanagari"))
    return out


def cer(script, page_font, kind, libs, method="xor", lam=0.0):
    gt = truth(script, page_font)
    hyp = read_page(script, page_font, kind, libs, method, lam)
    hyp += [""] * (len(gt) - len(hyp))
    d = sum(edit_distance(h, g) for h, g in zip(hyp, gt))
    return d / sum(len(g) for g in gt), hyp, gt


def table(method="xor", lam=0.0):
    rows = []
    for sc in SCRIPTS:
        r = {}
        for kind in ("clean", "camera"):
            for mode in ("same", "different", "both"):
                vals = []
                for pf in FONTS:
                    tf = pf if mode == "same" else [f for f in FONTS if f != pf][0] if mode == "different" else None
                    libs = [get_lib(sc, f) for f in FONTS] if mode == "both" else [get_lib(sc, tf)]
                    vals.append(cer(sc, pf, kind, libs, method, lam)[0])
                r[(kind, mode)] = float(np.mean(vals))
        rows.append((sc, r))
    return rows


def print_table(method="xor", lam=0.0):
    print(f"\nCharacter error rate (%), method={method}, aspect penalty lam={lam}")
    print(f"{'script':11s} | {'clean same':>10s} {'clean diff':>10s} {'clean both':>10s} | "
          f"{'cam same':>8s} {'cam diff':>8s} {'cam both':>8s}")
    for sc, r in table(method, lam):
        print(f"{sc:11s} | {100*r[('clean','same')]:10.1f} {100*r[('clean','different')]:10.1f} "
              f"{100*r[('clean','both')]:10.1f} | {100*r[('camera','same')]:8.1f} "
              f"{100*r[('camera','different')]:8.1f} {100*r[('camera','both')]:8.1f}")


if __name__ == "__main__":
    for m in ("xor", "chamfer"):
        print_table(m, 0.0)
