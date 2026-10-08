# IndicLens

Classical image-processing recognition and translation of Hindi, Telugu and Kannada text from photos.

Photo of printed Hindi / Telugu / Kannada text -> English translation, using classical
image processing only (after the Thai EE368 reference project).

Stack: Python, OpenCV, NumPy/SciPy, scikit-image, Pillow (Raqm) + Noto fonts,
deep-translator (translation), Streamlit (demo).

## Status
- [x] M1  fonts + test data  (`python src/make_data.py`)
- [x] M2  preprocessing pipeline (`python src/run_preprocess.py`)
- [x] M3  unit segmentation (`python src/run_segment.py`)
- [ ] M4  template database + XOR matcher
- [ ] M5  SVD/PCA + SIFT matchers, accuracy table
- [ ] M6  translation + Streamlit demo
- [ ] M7  report / poster
