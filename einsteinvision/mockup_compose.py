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


def inset(render, objs_json, oid, out, scale=3):
    """Paste a zoomed crop of vehicle `oid` (projected bbox from sequence_render --stills) top-right."""
    import json
    im = Image.open(render).convert("RGB")
    rows = [r for r in json.load(open(objs_json)) if r["oid"] == str(oid)]
    if rows:
        x1, y1, x2, y2 = rows[0]["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw = max(x2 - x1, (y2 - y1) * 1.6, 40) * 0.75
        box = (int(cx - hw), int(cy - hw / 1.6), int(cx + hw), int(cy + hw / 1.6))
        crop = im.crop(box)
        tw = im.width * 36 // 100
        crop = crop.resize((tw, tw * crop.height // max(crop.width, 1)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle(box, outline=(0, 255, 255), width=2)
        X = im.width - crop.width - 20
        im.paste(crop, (X, 20))
        d.rectangle((X - 2, 18, X + crop.width + 1, 21 + crop.height), outline=(0, 255, 255), width=3)
        st = rows[0]["state"]
        txt = f"t{oid}: {'PARKED' if st['parked'] else 'moving'} brake={st['brake']} ind={st['indicator']}"
        d.text((X + 8, 26), txt, font=font(22), fill=(255, 255, 0))
    else:
        ImageDraw.Draw(im).text((im.width - 520, 26), f"t{oid}: not placed in render", font=font(24), fill=(255, 80, 80))
    im.save(out)


def vstack(out, imgs, w=1920):
    ims = [Image.open(p).convert("RGB") for p in imgs]
    ims = [i.resize((w, i.height * w // i.width)) for i in ims]
    s = Image.new("RGB", (w, sum(i.height for i in ims)))
    y = 0
    for i in ims:
        s.paste(i, (0, y))
        y += i.height
    s.save(out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "single":
        single(*sys.argv[2:6])
    elif mode == "pair":
        pair(*sys.argv[2:6])
    elif mode == "sheet":
        sheet(sys.argv[2], sys.argv[3:])
    elif mode == "inset":
        inset(*sys.argv[2:6])
    elif mode == "vstack":
        vstack(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
