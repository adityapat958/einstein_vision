"""Portfolio demo reel: N short clips (render full-frame + real dashcam inset + caption),
crossfaded, one H.264 mp4 for the web + poster JPEG.

    python3 einsteinvision/reel.py --work renders/reel/work --out renders/reel/einsteinvision_demo.mp4 \
        --clip 1:840:"Highway · day" --clip 12:810:"Night highway" ...

Each --clip is scene:start_frame:caption; the render frames must already exist in
<work>/sS_fSTART/frame_XXXXX.png (sequence_render.py output) for start .. start+len+fade.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080


def font(sz, bold=True):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_CAP, F_SUB, F_TAG = font(40), font(24, False), font(22)


def rounded(im, r):
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, im.width - 1, im.height - 1), r, fill=255)
    return m


def compose(render, cam, caption, sub, alpha_txt=1.0):
    im = render.convert("RGB").resize((W, H), Image.LANCZOS)
    # dashcam inset, top-left, rounded, soft shadow, "CAMERA" tag
    iw = 500
    ih = iw * cam.height // cam.width
    c = cam.resize((iw, ih), Image.LANCZOS)
    x0, y0 = 40, 40
    sh = Image.new("RGBA", (iw + 60, ih + 60), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((30, 30, 30 + iw, 30 + ih), 18, fill=(0, 0, 0, 140))
    sh = sh.filter(ImageFilter.GaussianBlur(12))
    im.paste(sh, (x0 - 30, y0 - 22), sh)
    im.paste(c, (x0, y0), rounded(c, 16))
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle((x0, y0, x0 + iw, y0 + ih), 16, outline=(255, 255, 255, 190), width=2)
    tag = "CAMERA INPUT"
    tw = d.textlength(tag, font=F_TAG)
    d.rounded_rectangle((x0 + 12, y0 + 12, x0 + 32 + tw, y0 + 48), 8, fill=(0, 0, 0, 150))
    d.text((x0 + 22, y0 + 16), tag, font=F_TAG, fill=(255, 255, 255, 235))
    # caption, bottom-left
    a = int(255 * alpha_txt)
    if a > 0:
        cw = max(d.textlength(caption, font=F_CAP), d.textlength(sub, font=F_SUB))
        d.rounded_rectangle((40, H - 150, 40 + cw + 56, H - 40), 16, fill=(0, 0, 0, int(0.55 * a)))
        d.rectangle((40, H - 150, 46, H - 40), fill=(232, 33, 39, a))   # accent bar
        d.text((68, H - 140), caption, font=F_CAP, fill=(255, 255, 255, a))
        d.text((68, H - 86), sub, font=F_SUB, fill=(215, 215, 215, a))
    return im


def cam_frames(scene, start, n):
    v = sorted(Path(f"P3Data/Sequences/scene{scene}/Undist").glob("*-front_undistort.mp4"))[0]
    cap = cv2.VideoCapture(str(v))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", action="append", required=True, help='scene:start:caption[:subtitle]')
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fade", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=36.0)
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()
    work, fps = Path(a.work), a.fps
    n, nf = round(a.seconds * fps), round(a.fade * fps)
    seq = work / "_reel"
    seq.mkdir(parents=True, exist_ok=True)
    for p in seq.glob("*.jpg"):
        p.unlink()
    clips = []
    for spec in a.clip:
        parts = spec.split(":")
        s, st, cap = int(parts[0]), int(parts[1]), parts[2]
        sub = parts[3] if len(parts) > 3 else "Reconstructed from a single front camera · Blender Cycles"
        clips.append((s, st, cap, sub))

    def clip_frames(i):
        s, st, cap, sub = clips[i]
        m = n + (nf if i < len(clips) - 1 else 0)          # extra frames overlap the next clip's fade-in
        cams = cam_frames(s, st, m)
        rdir = work / f"s{s}_f{st}"
        for j in range(m):
            rp = rdir / f"frame_{st + j:05d}.png"
            if not rp.exists():
                rp = rdir / f"frame_{st + j:05d}.jpg"
            # caption fades in over the first 0.4 s, out over the last 0.3 s of the clip's own span
            t = j / fps
            at = min(1.0, t / 0.4, max(0.0, (a.seconds - t) / 0.3))
            yield compose(Image.open(rp), cams[min(j, len(cams) - 1)], cap, sub, at)

    k = 0
    carry = []                                           # previous clip's overlap frames (fade source)
    for i in range(len(clips)):
        tail = []
        for j, im in enumerate(clip_frames(i)):
            if j < len(carry):                           # crossfade: previous tail → this head
                im = Image.blend(carry[j], im, (j + 1) / (len(carry) + 1))
            if j < n:
                im.save(seq / f"{k:05d}.jpg", quality=94)
                k += 1
            else:
                tail.append(im)                          # overlap region → faded into the next clip
        carry = tail
        print(f"[reel] clip {i + 1}/{len(clips)} scene{clips[i][0]} f{clips[i][1]} → {k} frames", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(seq / "%05d.jpg"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                    "-profile:v", "high", "-movflags", "+faststart", "-an", str(out)], check=True)
    Image.open(seq / f"{min(k - 1, n // 2):05d}.jpg").save(out.with_suffix(".jpg"), quality=90)   # poster
    print(f"[reel] wrote {out} ({k / fps:.1f}s, {out.stat().st_size / 1e6:.1f} MB) + poster {out.with_suffix('.jpg')}")


if __name__ == "__main__":
    main()
