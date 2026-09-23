"""Compose mockup renders: inset real front-camera frame + label; 2x2 sheet."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(sys.argv[1] if len(sys.argv) > 1 else "mockups")
REF = D / "ref/front_2131.png"
NAMES = {"A": "A · FSD Dark (EEVEE)", "B": "B · Light Studio (EEVEE)", "C": "C · Cinematic (Cycles GPU)"}

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def label(im, text, sz=34):
    d = ImageDraw.Draw(im, "RGBA")
    f = font(sz)
    w = d.textlength(text, font=f)
    d.rounded_rectangle((24, im.height - sz - 44, 24 + w + 32, im.height - 20), 12, fill=(0, 0, 0, 150))
    d.text((40, im.height - sz - 34), text, font=f, fill=(255, 255, 255, 255))

ref = Image.open(REF).convert("RGB")
tiles = [ref.resize((1920, 1440 * 1080 // 1440)).crop((0, 0, 1920, 1080))]
r2 = ref.copy(); r2 = r2.resize((1440, 1080)); canvas = Image.new("RGB", (1920, 1080), (12, 12, 14)); canvas.paste(r2, (240, 0))
label(canvas, "Input · scene1 front camera, frame 2131")
tiles = [canvas]
for s in "ABC":
    p = D / f"style_{s}.png"
    if not p.exists():
        print("missing", p); continue
    im = Image.open(p).convert("RGB")
    inset = ref.resize((440, 330))
    ImageDraw.Draw(im).rectangle((28, 28, 28 + 444, 28 + 334), outline=(255, 255, 255), width=2)
    im.paste(inset, (30, 30))
    label(im, NAMES[s])
    im.save(D / f"mockup_{s}.png")
    tiles.append(im)
sheet = Image.new("RGB", (1920 * 2, 1080 * 2), (0, 0, 0))
for i, t in enumerate(tiles[:4]):
    sheet.paste(t, ((i % 2) * 1920, (i // 2) * 1080))
sheet.resize((1920, 1080)).save(D / "mockup_sheet.png")
print("ok", [p.name for p in sorted(D.glob("mockup_*.png"))])
