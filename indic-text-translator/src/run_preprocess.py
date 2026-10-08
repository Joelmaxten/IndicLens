"""Run preprocessing on every test image, report results, save stage figures."""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess

TRUE_ANGLE = {  # rotation applied in make_data.make_camera (cv2 sign, CCW +)
    "devanagari_sans": -4, "devanagari_serif": 3,
    "telugu_sans": -3, "telugu_serif": 4,
    "kannada_sans": -2, "kannada_serif": 5,
}

def tile(imgs, h=420):
    out = []
    for im in imgs:
        if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        out.append(cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)))
    return cv2.hconcat(out)

def main():
    os.makedirs("results/preprocess", exist_ok=True)
    print(f"{'image':34s} {'page':5s} {'angle':>7s} {'expect':>7s} {'lines':>5s}")
    for p in sorted(glob.glob("data/*_clean.png") + glob.glob("data/*_camera.jpg")):
        name = os.path.basename(p)
        base = name.rsplit("_", 1)[0]
        cam = name.endswith(".jpg")
        s = preprocess(cv2.imread(p))
        exp = -TRUE_ANGLE[base] if cam else 0
        print(f"{name:34s} {str(s['page_found']):5s} {s['angle']:7.2f} {exp:7.1f} {len(s['lines']):5d}")
        vis = s["text"].copy(); vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for y0, y1 in s["lines"]:
            cv2.rectangle(vis, (0, y0), (vis.shape[1] - 1, y1), (0, 0, 255), 2)
        cv2.imwrite(f"results/preprocess/{name.rsplit('.',1)[0]}_stages.jpg",
                    tile([s["gray"], s["page"], s["binary"], s["denoised"], s["deskewed"]], 360))
        cv2.imwrite(f"results/preprocess/{name.rsplit('.',1)[0]}_lines.png", 255 - vis if False else vis)

if __name__ == "__main__":
    main()
