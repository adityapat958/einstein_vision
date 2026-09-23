"""Compose mockup renders with the real camera frame.

    python3 mockup_compose.py single RENDER REF "label" OUT     # render + inset of real frame
    python3 mockup_compose.py pair   RENDER REF "label" OUT     # real frame | render, side by side
    python3 mockup_compose.py sheet  OUT IMG [IMG ...]          # 2-col grid, 1920 wide
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


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


def single(render, ref, text, out):
    im = Image.open(render).convert("RGB")
    r = Image.open(ref).convert("RGB")
    iw = im.width * 23 // 100
    inset = r.resize((iw, iw * r.height // r.width))
    ImageDraw.Draw(im).rectangle((28, 28, 32 + inset.width, 32 + inset.height), outline=(255, 255, 255), width=2)
    im.paste(inset, (30, 30))
    label(im, text)
    im.save(out)


def pair(render, ref, text, out):
    im = Image.open(render).convert("RGB")
    r = Image.open(ref).convert("RGB")
    h = im.height
    r = r.resize((r.width * h // r.height, h))
    c = Image.new("RGB", (r.width + im.width, h))
    c.paste(r, (0, 0))
    c.paste(im, (r.width, 0))
    label(c, text)
    c.save(out)


def sheet(out, imgs):
    ims = [Image.open(p).convert("RGB") for p in imgs]
    w = 960
    ims = [i.resize((w, i.height * w // i.width)) for i in ims]
    rows = [ims[k:k + 2] for k in range(0, len(ims), 2)]
    H = sum(max(i.height for i in r) for r in rows)
    s = Image.new("RGB", (2 * w, H))
    y = 0
    for r in rows:
        for j, i in enumerate(r):
            s.paste(i, (j * w, y))
        y += max(i.height for i in r)
    s.save(out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "single":
        single(*sys.argv[2:6])
    elif mode == "pair":
        pair(*sys.argv[2:6])
    elif mode == "sheet":
        sheet(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
