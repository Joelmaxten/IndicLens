"""Template library (milestone 4).

For one script + one font we render many candidate clusters (letters, letter +
vowel sign, conjuncts ...), cut each one into units with *exactly the same
segmentation code used on test pages*, and label every unit with the Unicode
text it came from.

Labelling
  * cluster renders as 1 unit  -> that unit is the whole cluster text.
  * cluster renders as k > 1   -> units are first matched against the library
    built so far; the one unmatched unit gets "cluster text minus the labels of
    the matched units" (e.g. the unit left over from "का" after removing the
    letter "क" is the sign "ा"). Clusters with more than one unknown unit are
    skipped (counted in the stats).

Templates are stored as SIZE x SIZE binary images, like the reference's
"resize then compare" approach.
"""
from __future__ import annotations

import os
import sys
import time
import unicodedata
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.segment import segment_line

SIZE = 32
FONT_PX = 44
FONT_DIR = "fonts"
CACHE = "cache"


def rng(a, b):
    return [chr(c) for c in range(a, b + 1)]


SCRIPTS = {
    "devanagari": dict(
        deva=True,
        consonants=rng(0x0915, 0x0939),
        vowels=list("अआइईउऊऋएऐओऔ"),
        matras=list("ािीुूृेैोौ"),
        nasals=list("ंँः"),
        virama="्",
    ),
    "telugu": dict(
        deva=False,
        consonants=rng(0x0C15, 0x0C39),
        vowels=list("అఆఇఈఉఊఋఎఏఐఒఓఔ"),
        matras=list("ాిీుూృౄెేైొోౌ"),
        nasals=list("ంః"),
        virama="్",
    ),
    "kannada": dict(
        deva=False,
        consonants=rng(0x0C95, 0x0CB9),
        vowels=list("ಅಆಇಈಉಊಋಎಏಐಒಓಔ"),
        matras=list("ಾಿೀುೂೃೄೆೇೈೊೋೌ"),
        nasals=list("ಂಃ"),
        virama="್",
    ),
}


def font_path(script, font):
    return f"{FONT_DIR}/noto-{font}-{script}-{script}.ttf"


def render_cluster(text: str, font_obj) -> np.ndarray | None:
    """Render text, return binary crop (ink = 255) or None if empty."""
    W, H = 70 * (len(text) + 1) + 40, 150
    im = Image.new("L", (W, H), 255)
    ImageDraw.Draw(im).text((30, 40), text, font=font_obj, fill=0,
                            layout_engine=ImageFont.Layout.RAQM)
    a = np.array(im) < 128
    ys, xs = np.where(a)
    if ys.size == 0:
        return None
    return (a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8)


def to_bits(mask: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Resize a binary unit mask to size x size (area interpolation, re-binarised)."""
    r = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    return r > 100


def candidates(cfg) -> list[tuple[int, str]]:
    """(pass, text) pairs in the order they must be built."""
    C, V, M, N, vi = cfg["consonants"], cfg["vowels"], cfg["matras"], cfg["nasals"], cfg["virama"]
    out = [(1, x) for x in V + C]
    out += [(2, c + m) for c in C for m in M + N]
    out += [(3, c + vi) for c in C]
    out += [(4, c1 + vi + c2) for c1 in C for c2 in C]
    out += [(5, c + m + n) for c in C for m in M for n in N]
    out += [(6, c1 + vi + c2 + m) for c1 in C for c2 in C for m in M + N]
    return out


class Lib:
    """Template set with incremental storage (also used by match.py).

    bits: (n, SIZE*SIZE) float32 0/1 ; dt: (n, SIZE*SIZE) float32 distance to the
    template's nearest ink pixel. Both live in growable buffers.
    """
    def __init__(self, cap: int = 1024):
        self._bits = np.zeros((cap, SIZE * SIZE), np.float32)
        self._dt = np.zeros((cap, SIZE * SIZE), np.float32)
        self._aspect = np.zeros(cap, np.float32)
        self._tsum = np.zeros(cap, np.float32)
        self.labels: list[str] = []
        self.basic: list[int] = []        # indices of short-label templates (<= 2 chars)

    @property
    def n(self): return len(self.labels)
    @property
    def bits(self): return self._bits[:self.n]
    @property
    def dt(self): return self._dt[:self.n]
    @property
    def aspect(self): return self._aspect[:self.n]
    @property
    def tsum(self): return self._tsum[:self.n]

    def add(self, mask, label):
        n = self.n
        if n == len(self._bits):
            self._bits = np.vstack([self._bits, np.zeros_like(self._bits)])
            self._dt = np.vstack([self._dt, np.zeros_like(self._dt)])
            self._aspect = np.concatenate([self._aspect, np.zeros_like(self._aspect)])
            self._tsum = np.concatenate([self._tsum, np.zeros_like(self._tsum)])
        b = to_bits(mask).reshape(-1)
        self._bits[n] = b
        self._dt[n] = distance_maps(b.reshape(1, -1))[0]
        self._aspect[n] = mask.shape[1] / mask.shape[0]
        self._tsum[n] = b.sum()
        self.labels.append(label)
        if len(label) <= 2:
            self.basic.append(n)


def distance_maps(bits: np.ndarray) -> np.ndarray:
    """(N, SIZE*SIZE) float32: distance of every pixel to the nearest ink pixel."""
    out = np.empty(bits.shape, np.float32)
    for i, b in enumerate(bits):
        img = np.where(b.reshape(SIZE, SIZE) > 0, 0, 255).astype(np.uint8)   # ink -> 0 for DT
        out[i] = cv2.distanceTransform(img, cv2.DIST_L2, 3).reshape(-1)
    return out


def chamfer_scores(lib: Lib, mask: np.ndarray, subset=None) -> np.ndarray:
    """Symmetric normalised chamfer distance of one unit against the templates
    (all of them, or only those in `subset`, a list of indices)."""
    u = to_bits(mask).reshape(-1).astype(np.float32)
    idx = np.arange(lib.n) if subset is None else np.asarray(subset)
    if u.sum() == 0 or idx.size == 0:
        return np.full(idx.size, 1e9, np.float32)
    dtu = distance_maps(u.reshape(1, -1))[0]
    s1 = lib.dt[idx] @ u / max(1.0, u.sum())                       # unit ink -> template
    s2 = (lib.bits[idx] @ dtu) / np.maximum(1.0, lib.tsum[idx])    # template ink -> unit
    return s1 + s2


def build(script: str, font: str, theta: float = 1.2, verbose=True) -> dict:
    cfg = SCRIPTS[script]
    fobj = ImageFont.truetype(font_path(script, font), FONT_PX, layout_engine=ImageFont.Layout.RAQM)
    lib = Lib()
    stats = dict(clusters=0, single=0, multi_ok=0, skipped=0, empty=0)
    texts_seen = set()
    cl = candidates(cfg)
    pending_dt_refresh = 0
    for pas, text in cl:
        text = unicodedata.normalize("NFC", text)
        if text in texts_seen:
            continue
        texts_seen.add(text)
        stats["clusters"] += 1
        line = render_cluster(text, fobj)
        if line is None:
            stats["empty"] += 1
            continue
        units, _, _ = segment_line(line, group=True, devanagari=cfg["deva"])
        if not units:
            stats["empty"] += 1
            continue
        if len(units) == 1:
            lib.add(units[0].mask, text)
            stats["single"] += 1
            continue
        # multi-unit cluster: label units by nearest known template, subtract
        rest = text
        unknown = []
        for u in units:
            if lib.n:
                sc = chamfer_scores(lib, u.mask, lib.basic)
                j = lib.basic[int(np.argmin(sc))]
                if sc.min() < theta and lib.labels[j] in rest:
                    rest = rest.replace(lib.labels[j], "", 1)
                    continue
            unknown.append(u)
        if len(unknown) == 1 and rest:
            lib.add(unknown[0].mask, unicodedata.normalize("NFC", rest))
            stats["multi_ok"] += 1
        elif len(unknown) == 0:
            stats["multi_ok"] += 1
        else:
            stats["skipped"] += 1
    return dict(lib=lib, stats=stats)


def save(lib: Lib, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, bits=np.packbits(lib.bits > 0.5, axis=1), labels=np.array(lib.labels, dtype=object),
                        aspect=lib.aspect, size=SIZE)


def load(path: str) -> Lib:
    z = np.load(path, allow_pickle=True)
    n = len(z["labels"])
    lib = Lib(cap=max(1024, n))
    bits = np.unpackbits(z["bits"], axis=1)[:, :SIZE * SIZE].astype(np.float32)
    lib._bits[:n] = bits
    lib._dt[:n] = distance_maps(bits)
    lib._aspect[:n] = z["aspect"]
    lib._tsum[:n] = bits.sum(1)
    lib.basic = [i for i, l in enumerate(z["labels"]) if len(str(l)) <= 2]
    lib.labels = [str(x) for x in z["labels"]]
    return lib


def _job(args):
    script, font = args
    t = time.time()
    r = build(script, font)
    save(r["lib"], f"{CACHE}/lib_{script}_{font}.npz")
    return script, font, r["lib"].n, r["stats"], time.time() - t


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    jobs = [(s, f) for s in SCRIPTS for f in ("sans", "serif")]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] == sys.argv[1]]
    with Pool(2) as p:
        for script, font, n, stats, dt in p.imap_unordered(_job, jobs):
            print(f"{script:11s} {font:5s} templates={n:6d} {stats} {dt:6.0f}s", flush=True)
