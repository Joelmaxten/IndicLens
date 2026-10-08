"""Classical preprocessing pipeline (follows the Thai EE368 reference).

Stages
  1. page detection        - crop the paper from a dark background (extension
                             over the reference, which struggled with this)
  2. rank filter + locally adaptive thresholding, AND of both binary images
  3. noise removal         - connected-component area filtering
  4. deskew                - Hough transform, drop short lines, modal angle
  5. text-box detection    - cumulative projection 10%/90% points
  6. line segmentation     - row projection, small marks merged into a line

All functions take/return plain NumPy arrays. `preprocess()` runs everything
and returns a dict of the intermediate images so they can be shown in the
report and the demo app.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_sauvola


# ----------------------------------------------------------------- stage 1
def find_page(gray: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    """Return (gray with everything outside the page painted white, found, page_mask).

    page_mask is 255 inside the (slightly shrunken) page. If the image is already
    a clean page (no dark surround) it is returned as is with a full mask.
    """
    full = np.full(gray.shape, 255, np.uint8)
    h, w = gray.shape
    small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    t, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Only treat it as "page on dark background" if the border is mostly dark.
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if (border == 0).mean() < 0.6:
        return gray, False, full
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray, False, full
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * mask.size:
        return gray, False, full
    page = np.zeros_like(mask)
    cv2.drawContours(page, [c], -1, 255, -1)
    # shrink a little so the paper edge / shadow does not become "text"
    k = max(3, int(0.015 * min(page.shape)))
    page = cv2.erode(page, np.ones((k, k), np.uint8))
    page = cv2.resize(page, (w, h), interpolation=cv2.INTER_NEAREST)
    # Fill the outside with the *surrounding paper brightness* (normalised
    # Gaussian blur of the paper pixels) instead of flat white: a flat fill
    # next to shadowed paper creates a fake edge that thresholding mistakes for ink.
    m = (page > 0).astype(np.float32)
    sig = 0.05 * min(h, w)
    num = cv2.GaussianBlur(gray.astype(np.float32) * m, (0, 0), sig)
    den = cv2.GaussianBlur(m, (0, 0), sig)
    paper = float(np.median(gray[page > 0]))
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), paper)
    fill = np.where(den > 0.05, fill, paper)
    out = np.where(page > 0, gray, np.clip(fill, 0, 255)).astype(np.uint8)
    return out, True, page


# ----------------------------------------------------------------- stage 2
def binarize(gray: np.ndarray, window: int | None = None, k: float = 0.8,
             rank: int = 1, rank_size: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-filter + locally adaptive thresholding.

    Returns (rank_filtered_gray, binary, binary_filtered). Binary images have
    text = 255. The final result is the AND of both, as in the reference.
    """
    h, w = gray.shape
    if window is None:
        window = int(max(25, min(h, w) / 20)) | 1
    # rank filter on grey levels: low rank ~ erosion -> thickens dark text
    filt = ndi.rank_filter(gray, rank=rank, size=rank_size)
    b1 = (gray < threshold_sauvola(gray, window_size=window, k=k)).astype(np.uint8) * 255
    b2 = (filt < threshold_sauvola(filt, window_size=window, k=k)).astype(np.uint8) * 255
    return filt, b1, b2


# ----------------------------------------------------------------- stage 3
def remove_noise(binary: np.ndarray, min_area: int = 8,
                 max_frac: float = 0.2) -> np.ndarray:
    """Drop specks and giant blobs by connected-component area.

    The reference used mean +/- 1 std of the area. For Indic scripts that is too
    aggressive: Devanagari words are single large blobs (headline joins the
    letters) while dots / vowel signs are legitimately tiny. So we only remove
    obvious specks and blobs covering a large part of the page.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, bool)
    h, w = binary.shape
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a < min_area:
            continue
        if a > max_frac * h * w or (bw > 0.9 * w and bh > 0.5 * h):
            continue
        keep[i] = True
    return (keep[lab].astype(np.uint8)) * 255


# ----------------------------------------------------------------- stage 4
def _skew_profile(binary: np.ndarray, span: float = 15.0, step: float = 0.25) -> float:
    """Fallback skew estimate: angle that maximises the sharpness of row sums."""
    h, w = binary.shape
    f = 400.0 / max(h, w)
    small = cv2.resize(binary, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    best, best_a = -1.0, 0.0
    for a in np.arange(-span, span + step, step):
        r = rotate(small, a)
        v = float((r.astype(np.float32).sum(axis=1) ** 2).sum())
        if v > best:
            best, best_a = v, a
    return float(best_a)           # + = text slopes down-right (Hough convention)


def estimate_skew(binary: np.ndarray) -> float:
    """Skew angle (degrees; + = text slopes down-right) of the text lines.

    Coarse estimate from a projection-profile search, then refined with the
    Hough transform as in the reference: close the text horizontally so each
    line becomes a long blob, take its edges, run the probabilistic Hough
    transform, keep lines near the coarse angle, drop those shorter than half
    the longest, and return the modal angle.
    """
    coarse = _skew_profile(binary)
    h, w = binary.shape
    kx = max(15, w // 15)
    lines_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    edges = cv2.Canny(lines_img, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40,
                           minLineLength=w // 8, maxLineGap=15)
    if segs is None:
        return coarse
    segs = np.asarray(segs, dtype=float).reshape(-1, 4)  # OpenCV 4: (N,1,4), 5: (N,4)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    ang = ((ang + 90) % 180) - 90                     # fold into [-90, 90)
    near = np.abs(ang - coarse) < 2.0                 # reject outlier directions
    ang, length = ang[near], length[near]
    if ang.size == 0:
        return coarse
    ang = ang[length >= 0.5 * length.max()]           # drop short lines (reference trick)
    hist, edges_ = np.histogram(ang, bins=np.arange(-45, 45.01, 0.5))
    m = np.argmax(hist)
    sel = ang[(ang >= edges_[m] - 0.5) & (ang <= edges_[m + 1] + 0.5)]
    return float(np.median(sel)) if sel.size else coarse


def rotate(img: np.ndarray, angle: float, border: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border)


# ----------------------------------------------------------------- stage 5
def _cum_bounds(profile: np.ndarray, lo: float = 0.10, hi: float = 0.90) -> tuple[int, int]:
    """Estimate the 0%..100% extent of a profile from its 10%/90% cumulative points."""
    c = np.cumsum(profile.astype(float))
    if c[-1] == 0:
        return 0, len(profile) - 1
    c /= c[-1]
    x = np.arange(len(c))
    p_lo, p_hi = np.interp(lo, c, x), np.interp(hi, c, x)
    # linear extrapolation to 0 % and 100 %
    slope = (p_hi - p_lo) / (hi - lo)
    a = int(max(0, np.floor(p_lo - lo * slope)))
    b = int(min(len(profile) - 1, np.ceil(p_hi + (1 - hi) * slope)))
    return a, b


def find_text_box(binary: np.ndarray, pad: int = 6, method: str = "extent") -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the text region.

    method="extent": first/last row and column that contain ink (the binary
        image is already page-masked and denoised, so this is safe).
    method="cumulative": the reference's 10%/90% cumulative-projection estimate.
        It assumes justified text; on ragged-right text it clips the ends of
        the longest lines (we saw this on Telugu/Kannada), so it is not the default.
    """
    h, w = binary.shape
    if method == "cumulative":
        x0, x1 = _cum_bounds((binary > 0).sum(axis=0))
        y0, y1 = _cum_bounds((binary > 0).sum(axis=1))
    else:
        xs = np.where((binary > 0).any(axis=0))[0]
        ys = np.where((binary > 0).any(axis=1))[0]
        if xs.size == 0:
            return 0, 0, w, h
        x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


# ----------------------------------------------------------------- stage 6
def segment_lines(binary: np.ndarray, rel_thresh: float = 0.02) -> list[tuple[int, int]]:
    """Return [(y0, y1), ...] row ranges of text lines.

    Row projection finds runs of ink. Runs much thinner than a typical line
    (vowel marks above / below a line, as in the reference's Thai case) are
    merged into their nearest neighbouring run; isolated thin runs are noise.
    """
    prof = (binary > 0).sum(axis=1).astype(float)
    on = prof > rel_thresh * prof.max()
    runs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append([start, y]); start = None
    if start is not None:
        runs.append([start, len(on)])
    if len(runs) <= 1:
        return [tuple(r) for r in runs]
    # "typical" line height: a high percentile, because thin mark-runs are
    # numerous and drag the median down.
    typical = float(np.percentile([r[1] - r[0] for r in runs], 80))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if (r[1] - r[0]) >= 0.5 * typical:
                continue
            up = r[0] - runs[i - 1][1] if i > 0 else 1e9
            dn = runs[i + 1][0] - r[1] if i + 1 < len(runs) else 1e9
            gap, j = (up, i - 1) if up <= dn else (dn, i + 1)
            if gap < 0.5 * typical:                      # attach to nearest neighbour
                runs[j] = [min(runs[j][0], r[0]), max(runs[j][1], r[1])]
            # either merged or an isolated sliver (noise): drop it
            runs.pop(i)
            changed = True
            break
    return [tuple(r) for r in runs]


# --------------------------------------------------------------------- all
def preprocess(img: np.ndarray, is_camera: bool | None = None) -> dict:
    """Run the whole pipeline. `img` may be BGR or grayscale uint8."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    stages: dict = {"gray": gray}

    page, found, page_mask = find_page(gray)
    stages["page"] = page
    stages["page_found"] = found

    # Unsharp mask: camera blur otherwise makes neighbouring Telugu/Kannada letters
    # touch after thresholding (tuned on the test pages: sigma 1.5, k = 0.8).
    sharp = cv2.addWeighted(page, 2.0, cv2.GaussianBlur(page, (0, 0), 1.5), -1.0, 0)
    stages["sharpened"] = sharp
    filt, b1, b2 = binarize(sharp)
    stages["rank_filtered"] = filt
    binary = cv2.bitwise_and(cv2.bitwise_and(b1, b2), page_mask)  # nothing outside the paper
    stages["binary"] = binary

    clean = remove_noise(binary)
    stages["denoised"] = clean

    angle = estimate_skew(clean)
    stages["angle"] = angle
    # rotate the clean binary (nearest, so it stays binary) and the page
    stages["deskewed"] = (rotate(clean, angle) > 127).astype(np.uint8) * 255
    stages["page_deskewed"] = rotate(page, angle, border=255)

    x0, y0, x1, y1 = find_text_box(stages["deskewed"])
    stages["text_box"] = (x0, y0, x1, y1)
    textimg = stages["deskewed"][y0:y1, x0:x1]
    stages["text"] = textimg

    stages["lines"] = segment_lines(textimg)
    return stages
