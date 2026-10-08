"""Generate clean and simulated-camera test documents for Hindi, Telugu, Kannada.

Mirrors the reference project's data: a clean "screenshot" page per script and
font, plus a "camera" version (rotated, blurred, unevenly lit, on a dark
background). Ground-truth text is written next to each image.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = "data"
FONT_DIR = "fonts"
PAGE = (1240, 1000)  # W, H of the clean page
FONT_SIZE = 44
LINE_GAP = 40
MARGIN = (110, 90)

TEXT = {
    "devanagari": [
        "भारत एक विशाल देश है",
        "यहाँ अनेक भाषाएँ बोली जाती हैं",
        "विद्यार्थी स्कूल में पढ़ते हैं",
        "मेरा नाम जोएल है",
        "आज मौसम बहुत अच्छा है",
        "हमें सत्य बोलना चाहिए",
    ],
    "telugu": [
        "తెలుగు భాష చాలా అందమైనది",
        "మా ఊరు చాలా పెద్దది",
        "విద్యార్థులు పాఠశాలకు వెళ్తారు",
        "నా పేరు జోయెల్",
        "ఈ రోజు వాతావరణం బాగుంది",
        "మనం నిజం చెప్పాలి",
    ],
    "kannada": [
        "ಕನ್ನಡ ಭಾಷೆ ತುಂಬಾ ಸುಂದರವಾಗಿದೆ",
        "ನಮ್ಮ ಊರು ತುಂಬಾ ದೊಡ್ಡದು",
        "ವಿದ್ಯಾರ್ಥಿಗಳು ಶಾಲೆಗೆ ಹೋಗುತ್ತಾರೆ",
        "ನನ್ನ ಹೆಸರು ಜೋಯೆಲ್",
        "ಇಂದು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ",
        "ನಾವು ಸತ್ಯ ಹೇಳಬೇಕು",
    ],
}
FONTS = ["sans", "serif"]


def render_clean(script, font):
    path = f"{FONT_DIR}/noto-{font}-{script}-{script}.ttf"
    f = ImageFont.truetype(path, FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    im = Image.new("L", PAGE, 255)
    d = ImageDraw.Draw(im)
    y = MARGIN[1]
    for line in TEXT[script]:
        d.text((MARGIN[0], y), line, font=f, fill=0, layout_engine=ImageFont.Layout.RAQM)
        y += FONT_SIZE + LINE_GAP + 30
    return np.array(im)


def make_camera(clean, seed, angle):
    """Simulate a phone photo: dark table, rotated page, uneven light, blur, noise."""
    rng = np.random.default_rng(seed)
    h, w = clean.shape
    pad = 160
    canvas = np.full((h + 2 * pad, w + 2 * pad), 45, np.uint8)  # dark table
    canvas[pad:pad + h, pad:pad + w] = clean
    H, W = canvas.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img = cv2.warpAffine(canvas, M, (W, H), borderValue=45, flags=cv2.INTER_LINEAR)
    # uneven lighting: linear gradient + a soft shadow blob
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    light = 0.95 - 0.35 * xx + 0.1 * yy
    cx, cy = rng.uniform(0.3, 0.8) * W, rng.uniform(0.3, 0.8) * H
    shadow = np.exp(-(((xx * W - cx) ** 2 + (yy * H - cy) ** 2) / (2 * (0.25 * W) ** 2)))
    light = light - 0.25 * shadow
    img = np.clip(img.astype(np.float32) * light, 0, 255)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = img + rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # downscale like a phone image, then JPEG-compress
    img = cv2.resize(img, (int(W * 0.8), int(H * 0.8)), interpolation=cv2.INTER_AREA)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    for i, script in enumerate(TEXT):
        for j, font in enumerate(FONTS):
            name = f"{script}_{font}"
            clean = render_clean(script, font)
            cv2.imwrite(f"{OUT}/{name}_clean.png", clean)
            cam = make_camera(clean, seed=10 * i + j, angle=(-4, 3)[j] + i)
            cv2.imwrite(f"{OUT}/{name}_camera.jpg", cam, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with open(f"{OUT}/{name}.txt", "w", encoding="utf-8") as fh:
                fh.write("\n".join(TEXT[script]) + "\n")
            print("wrote", name)


if __name__ == "__main__":
    main()
