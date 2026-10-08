"""Segment every test page; compare camera vs clean unit counts per line."""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess
from src.segment import segment_page

def run(path, group=True):
    s = preprocess(cv2.imread(path))
    return segment_page(s["text"], s["lines"], group=group)

def draw(lines, path):
    rows = []
    for L in lines:
        vis = cv2.cvtColor(L["line"], cv2.COLOR_GRAY2BGR)
        if L["band"]:
            cv2.rectangle(vis, (0, L["band"][0]), (vis.shape[1] - 1, L["band"][1]), (255, 160, 0), 1)
        palette = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 160, 255)]
        for u in L["units"]:
            cv2.rectangle(vis, (u.x0, u.y0), (u.x1 - 1, u.y1 - 1), palette[u.word % 4], 1)
        rows.append(cv2.copyMakeBorder(vis, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for r in rows]
    cv2.imwrite(path, cv2.vconcat(rows))

def main():
    os.makedirs("results/segment", exist_ok=True)
    print(f"{'page':22s} {'hint':11s} {'units/line (clean)':34s} {'units/line (camera)':34s} match")
    for base in ["devanagari_sans", "devanagari_serif", "telugu_sans", "telugu_serif", "kannada_sans", "kannada_serif"]:
        c = run(f"data/{base}_clean.png"); k = run(f"data/{base}_camera.jpg")
        cn = [len(L["units"]) for L in c]; kn = [len(L["units"]) for L in k]
        hints = {L["script"] for L in c + k}
        print(f"{base:22s} {','.join(sorted(hints)):11s} {str(cn):34s} {str(kn):34s} {'OK' if cn == kn else 'DIFF'}")
        draw(c, f"results/segment/{base}_clean.png"); draw(k, f"results/segment/{base}_camera.png")

if __name__ == "__main__":
    main()
