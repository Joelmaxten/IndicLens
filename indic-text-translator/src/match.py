"""Matching + decoding (milestone 4).

Scorers (all compare a unit resized to SIZE x SIZE with every template):
  xor      fraction of mismatching pixels (the reference's "XOR + resize")
  chamfer  distance-weighted mismatch: every ink pixel is penalised by its
           distance to the other image's nearest ink pixel (the reference's
           weighted distance score with penalty power p = 1)
An optional aspect-ratio penalty lam * |log(aspect_unit / aspect_template)|
is added to either score.
"""
from __future__ import annotations

import unicodedata

import numpy as np

from src.library import SIZE, Lib, chamfer_scores, to_bits


def scores(lib: Lib, mask: np.ndarray, method: str = "xor", lam: float = 0.0) -> np.ndarray:
    u = to_bits(mask).reshape(-1).astype(np.float32)
    if method == "xor":
        s = (lib.tsum + u.sum() - 2.0 * (lib.bits @ u)) / (SIZE * SIZE)
    elif method == "chamfer":
        s = chamfer_scores(lib, mask)
    else:
        raise ValueError(method)
    if lam:
        asp = mask.shape[1] / max(1, mask.shape[0])
        s = s + lam * np.abs(np.log(asp / np.maximum(lib.aspect, 1e-3)))
    return s


def recognize_unit(libs: list[Lib], mask: np.ndarray, method="xor", lam=0.0):
    """Best (label, score) over one or several libraries."""
    best = (None, 1e9)
    for lib in libs:
        s = scores(lib, mask, method, lam)
        j = int(np.argmin(s))
        if s[j] < best[1]:
            best = (lib.labels[j], float(s[j]))
    return best


def decode_line(units, labels: list[str], devanagari: bool) -> str:
    """Join unit labels into text. Inserts a space between words; for Devanagari
    moves the short-i sign (drawn BEFORE its letter) after the letter it belongs to."""
    toks = []
    for u, lab in zip(units, labels):
        toks.append((u.word, lab))
    if devanagari:
        out, i = [], 0
        while i < len(toks):
            w, lab = toks[i]
            if lab == "ि":
                j = i + 1
                # skip half-letters (labels ending in virama): ि sits after the whole cluster
                while j < len(toks) and toks[j][0] == w and toks[j][1].endswith("्"):
                    j += 1
                if j < len(toks) and toks[j][0] == w:
                    out.extend(toks[i + 1:j + 1])
                    out.append((w, "ि"))
                    i = j + 1
                    continue
            out.append((w, lab))
            i += 1
        toks = out
    text, prev = "", None
    for w, lab in toks:
        if prev is not None and w != prev:
            text += " "
        text += lab
        prev = w
    return unicodedata.normalize("NFC", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
