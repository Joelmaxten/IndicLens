# IndicLens - project context for Claude Code

Lab project (IPCV): photo of PRINTED Hindi / Telugu / Kannada text -> English translation.
Modelled on the Thai reference project in `../IPCV Project Reference` (Stanford EE368, Asavareongchai & Giarta):
fully classical image processing, NO OCR library, NO machine learning. OCR is only a fallback if template matching proves too hard.

## Rules
- The user does ALL git operations (add/commit/push). Never run git commit or git push; give the commands instead.
- Do not modify `../IPCV Project Reference` (read-only reference material, not in the repo).
- The user prefers step-by-step guidance: explain what you are doing and why, check in at each milestone.
- Stack: Python 3.11+, OpenCV, NumPy/SciPy, scikit-image, Pillow (Raqm), Noto fonts, deep-translator (Google), Streamlit demo, matplotlib, pytest.

## Layout
- `src/make_data.py`      generate clean + simulated-camera test pages (3 scripts x sans/serif)
- `src/preprocess.py`     page detection, unsharp + Sauvola threshold, noise removal, deskew (profile + Hough), text box, line segmentation
- `src/segment.py`        Devanagari headline removal, connected-component units, mark grouping, word gaps
- `src/library.py`        template library builder (renders letter/matra/conjunct clusters, segments them with the SAME code, labels units)
- `src/match.py`          XOR and chamfer (distance-weighted) matchers, decoder (word spaces, Devanagari short-i reorder), edit distance
- `src/evaluate.py`       character-error-rate table (same/different font, clean/camera)
- `data/`, `fonts/`, `results/`, `cache/` (cache = built template libraries, not committed)

## Status
- M1 test data: done. M2 preprocessing: done (6 lines/page on all 12 images, skew within ~0.3 deg).
- M3 segmentation: done (camera vs clean unit counts match on 30/36 lines).
- M4 template library + matchers + evaluation: code written but NOT yet run end to end.
  Next: run `python src/library.py` (should take ~1-2 min per script/font; a bug that made it quadratic was fixed by
  not adding duplicate exemplars), then `python src/evaluate.py`, inspect errors, tune (aspect penalty lam, SIZE, size features).
- M5 SVD/PCA + SIFT/RANSAC matchers + accuracy table. M6 translation (deep-translator) + Streamlit demo. M7 report/poster.

## Lessons learned (keep in mind)
- The reference's 10%/90% text-box rule clips ragged-right lines -> we use true ink extent.
- Camera blur merges Telugu/Kannada letters -> unsharp mask + stricter Sauvola (k=0.8) fixed it (tuned on these pages; re-tune on real photos).
- Devanagari headline (shirorekha) must be removed to separate letters; headline ratio threshold 0.89 (Devanagari >= 0.93, others <= 0.85).
- The reference's "rank filter AND original" step is a no-op here; kept for fidelity.
- Library labelling: a cluster that renders as one unit is labelled with its text; multi-unit clusters are labelled by matching
  known basic templates and subtracting (e.g. "का" -> unit "ा"). Unit search is restricted to basic templates (label <= 2 chars) for speed.
- Ground truth is text-level (data/*.txt), so evaluation uses edit distance, not per-unit labels.
- Test data is simulated; add real printed-and-photographed pages later.
