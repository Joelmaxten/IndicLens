"""Unit segmentation of a text line (milestone 3).

A "unit" is a connected blob of ink, optionally merged with marks that sit
directly above/below it (x-overlap grouping, like the reference's merging of
Thai vowels above/below a consonant).

Script-specific step: Devanagari letters in a word are joined by a horizontal
headline (shirorekha). We detect it with the row projection and delete that
band, which splits the word into letters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Unit:
    x0: int
    y0: int
    x1: int          # exclusive
    y1: int          # exclusive
    mask: np.ndarray = field(repr=False)   # uint8 crop, ink = 255
    word: int = 0    # word index inside the line

    @property
    def w(self): return self.x1 - self.x0
    @property
    def h(self): return self.y1 - self.y0


# --------------------------------------------------------------- headline
# Measured on the test pages: Devanagari lines score >= 0.93, Telugu/Kannada <= 0.85.
HEADLINE_RATIO = 0.89

def headline_band(line: np.ndarray) -> tuple[int, int] | None:
    """Rows [r0, r1) of the shirorekha if the line has one, else None.

    The headline is a row where a large fraction of all ink columns are on.
    """
    ink = line > 0
    cols = int(ink.any(axis=0).sum())
    if cols == 0:
        return None
    proj = ink.sum(axis=1).astype(float)
    peak = proj.max()
    if peak < HEADLINE_RATIO * cols:  # no long horizontal bar -> not Devanagari
        return None
    r = int(np.argmax(proj))
    lo = hi = r
    while lo > 0 and proj[lo - 1] >= 0.5 * peak:
        lo -= 1
    while hi < len(proj) - 1 and proj[hi + 1] >= 0.5 * peak:
        hi += 1
    return lo, hi + 1


def is_devanagari_line(line: np.ndarray) -> bool:
    return headline_band(line) is not None


def remove_headline(line: np.ndarray) -> tuple[np.ndarray, tuple[int, int] | None]:
    band = headline_band(line)
    out = line.copy()
    if band is not None:
        out[band[0]:band[1], :] = 0
    return out, band


# ------------------------------------------------------------------ units
def _components(binary: np.ndarray, min_area: int = 6) -> list[Unit]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    units = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        m = ((lab[y:y + h, x:x + w] == i) * 255).astype(np.uint8)
        units.append(Unit(x, y, x + w, y + h, m))
    units.sort(key=lambda u: (u.x0, u.y0))
    return units


def _xoverlap(a: Unit, b: Unit) -> float:
    ov = min(a.x1, b.x1) - max(a.x0, b.x0)
    return ov / max(1, min(a.w, b.w))


def _merge(g: list[Unit]) -> Unit:
    x0, y0 = min(v.x0 for v in g), min(v.y0 for v in g)
    x1, y1 = max(v.x1 for v in g), max(v.y1 for v in g)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for v in g:
        m[v.y0 - y0:v.y1 - y0, v.x0 - x0:v.x1 - x0] |= v.mask
    return Unit(x0, y0, x1, y1, m)


def group_units(units: list[Unit], line_shape: tuple[int, int], thresh: float = 0.5,
                small_h: float = 0.45, small_gap: float = 0.35) -> list[Unit]:
    """Merge marks with the letter they belong to.

    Pass 1: blobs that stack vertically (x-overlap > thresh of the narrower).
    Pass 2: tiny marks (e.g. a Kannada/Telugu subscript or a dot that lost its
    connection to the base after thresholding) join the nearest unit they overlap
    in x, if they sit close above/below it.
    """
    groups: list[list[Unit]] = []
    for u in units:
        placed = False
        for g in groups:
            gx0, gx1 = min(v.x0 for v in g), max(v.x1 for v in g)
            ov = min(u.x1, gx1) - max(u.x0, gx0)
            if ov / max(1, min(u.w, gx1 - gx0)) > thresh:
                g.append(u); placed = True; break
        if not placed:
            groups.append([u])
    merged = sorted((_merge(g) for g in groups), key=lambda u: u.x0)
    if len(merged) < 2:
        return merged
    body = float(np.percentile([u.h for u in merged], 75))
    mw = float(np.median([u.w for u in merged]))
    keep, marks = [], []
    for u in merged:
        (marks if (u.h < small_h * body and u.w < 0.6 * mw) else keep).append(u)
    if not keep:
        return merged
    groups2 = [[u] for u in keep]
    for m in marks:
        best, best_ov = None, 0
        for g in groups2:
            b = g[0]
            ov = min(m.x1, b.x1) - max(m.x0, b.x0)
            vgap = max(m.y0 - b.y1, b.y0 - m.y1, 0)
            if ov > best_ov and vgap < small_gap * body:
                best, best_ov = g, ov
        if best is not None:
            best.append(m)
        else:
            groups2.append([m])          # isolated mark stays its own unit
    out = sorted((_merge(g) for g in groups2), key=lambda u: u.x0)
    return out


def assign_words(units: list[Unit], gap_factor: float = 0.45) -> int:
    """Set unit.word using gaps in x. A gap wider than gap_factor * body height
    starts a new word. Returns the number of words."""
    if not units:
        return 0
    body = float(np.median([u.h for u in units]))
    w = 0
    units[0].word = 0
    for prev, cur in zip(units, units[1:]):
        if cur.x0 - prev.x1 > gap_factor * body:
            w += 1
        cur.word = w
    return w + 1


def segment_line(line: np.ndarray, group: bool = True, devanagari: bool | None = None):
    """Segment one binary text line (ink = 255).

    `devanagari` forces/skips headline removal (decided per page by a vote in
    segment_page); None means decide from this line alone.
    Returns (units, script_hint, headline_band).
    """
    if devanagari is None:
        devanagari = is_devanagari_line(line)
    work, band = remove_headline(line) if devanagari else (line, None)
    units = _components(work)
    if group:
        units = group_units(units, line.shape)
    assign_words(units)
    return units, ("devanagari" if devanagari else "dravidian"), band


def page_is_devanagari(text_img: np.ndarray, lines: list[tuple[int, int]]) -> bool:
    votes = []
    for (y0, y1) in lines:
        crop = text_img[y0:y1, :]
        xs = np.where((crop > 0).any(axis=0))[0]
        if xs.size:
            votes.append(is_devanagari_line(crop[:, xs[0]:xs[-1] + 1]))
    return bool(votes) and sum(votes) > len(votes) / 2


def segment_page(text_img: np.ndarray, lines: list[tuple[int, int]], group: bool = True):
    """Segment every line of a preprocessed page. Returns a list of dicts."""
    deva = page_is_devanagari(text_img, lines)
    out = []
    for (y0, y1) in lines:
        crop = text_img[y0:y1, :]
        xs = np.where((crop > 0).any(axis=0))[0]
        if xs.size == 0:
            continue
        x0, x1 = xs[0], xs[-1] + 1
        crop = crop[:, x0:x1]
        units, hint, band = segment_line(crop, group=group, devanagari=deva)
        out.append({"y": (y0, y1), "x": (x0, x1), "units": units, "script": hint, "band": band,
                    "line": crop})
    return out
