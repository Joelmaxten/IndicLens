"""IndicLens demo: photo of printed Hindi / Telugu / Kannada text -> English.
Run:  streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st

from src.ocr import GT_LANG, recognise, translate
from src.preprocess import preprocess

st.set_page_config(page_title="IndicLens", layout="wide")
st.title("IndicLens")
st.caption("Printed Hindi / Telugu / Kannada photo -> English. "
           "Classical preprocessing (page detection, unsharp + Sauvola, deskew, line segmentation) "
           "then Tesseract recognition and Google translation.")

up = st.file_uploader("Photo of printed text", type=["png", "jpg", "jpeg"])
sample = st.selectbox("...or a bundled sample", ["-"] + [
    f"{s}_{f}_{k}" for s in GT_LANG for f in ("sans", "serif") for k in ("clean", "camera")])

img = None
if up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
elif sample != "-":
    import os
    p = f"data/{sample}." + ("png" if sample.endswith("clean") else "jpg")
    img = cv2.imread(p)

if img is not None:
    with st.spinner("Detecting language and reading..."):
        s = preprocess(img)
        res = recognise(img)            # script auto-detected
        script, lines = res["script"], res["lines"]
    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    c2.image(s["binary"], caption="Sharpened + Sauvola threshold", use_container_width=True)
    c3.image(s["text"], caption=f"Deskewed text region ({s['angle']:.2f} deg), {len(s['lines'])} lines",
             use_container_width=True)
    if script is None:
        st.error("Language not recognised. IndicLens supports printed Hindi, Telugu and Kannada only.")
    else:
        st.info(f"Detected language: **{script.title()}**")
        with st.spinner("Translating..."):
            text = "\n".join(l for l in lines if l)
            english = "\n".join(translate(l, script) for l in lines if l)
        st.subheader("Recognised text")
        st.code(text, language=None)
        st.subheader("English translation")
        st.success(english)
