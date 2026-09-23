"""Road structure inference from lane_bev.py output (pure python, no bpy).

Given one frame of road_model.json (b, c, tracked lines) build the full
road: ego carriageway lane lines (detected + synthesised at the lane width),
median type (divided highway barrier vs double-yellow paint vs none) and
the oncoming carriageway.

All lateral positions are offsets `a` in x(y) = a + b·y + c·y²  (ego frame,
x right, y forward, metres) — every line/edge/barrier follows the same curve.
"""
from __future__ import annotations

import math

DEFAULT_W = 3.6
MAX_STALE = 20


def load(path):
    import json
    d = json.load(open(path))
    return {f["frame_index"]: f for f in d["frames"]}


def lane_width(offs):
    gaps = sorted(b - a for a, b in zip(offs, offs[1:]) if 2.8 <= b - a <= 4.6)
    return gaps[len(gaps) // 2] if gaps else DEFAULT_W


def derive(fr: dict | None, n_oncoming: int = 3) -> dict:
    """→ dict(b, c, W, lines=[dict(a,color,style,double,synth)], left, right,
              median=('barrier'|'double_yellow'|'none'), barrier_a, oncoming_lines)"""
    b = fr["b"] if fr else 0.0
    c = fr["c"] if fr else 0.0
    det = [L for L in (fr or {}).get("lines", []) if L.get("stale", 0) <= MAX_STALE]
    det.sort(key=lambda L: L["a"])
    # merge near-duplicates (< 0.7 m): keep yellow over white, solid over dashed
    lines = []
    for L in det:
        if lines and L["a"] - lines[-1]["a"] < 0.7:
            keep = max((lines[-1], L), key=lambda q: (q["color"] == "yellow", q["style"] == "solid"))
            lines[-1] = keep
        else:
            lines.append(dict(L))
    for L in lines:                                   # yellow right of ego: curb paint / glare
        if L["color"] == "yellow" and L["a"] > 0.3:
            L["color"], L["double"] = "white", False
    # crosswalk: ≥3 white stripes < 1.4 m apart (continental zebra seen in BEV) → drop, flag
    crosswalk = None
    i = 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1]["a"] - lines[j]["a"] < 1.4:
            j += 1
        if j - i >= 3:                                # ≥4 stripes
            crosswalk = (lines[i]["a"], lines[j]["a"])
            del lines[i:j + 1]
        else:
            i += 1
    W = lane_width([L["a"] for L in lines])

    if crosswalk is not None and not any(-2.6 < L["a"] < 2.6 for L in lines):
        # intersection: markings hidden by the crosswalk → ego-centred template
        lines = [dict(a=k * W - W / 2, color="white", style="dashed", double=False, synth=True)
                 for k in (0, 1)] + [L for L in lines if abs(L["a"]) >= 2.6]
    # ego lane must be bounded on both sides
    left = [L for L in lines if L["a"] < 0]
    right = [L for L in lines if L["a"] > 0]
    if not left:
        a0 = (right[0]["a"] - W) if right else -W / 2
        lines.append(dict(a=a0, color="white", style="dashed", double=False, synth=True))
    if not right:
        a0 = (left[-1]["a"] + W) if left else W / 2
        lines.append(dict(a=a0, color="white", style="dashed", double=False, synth=True))
    lines.sort(key=lambda L: L["a"])

    # yellow = left edge of our carriageway: drop anything detected left of it
    lines.sort(key=lambda L: L["a"])
    ys = [i for i, L in enumerate(lines) if L["color"] == "yellow"]
    median = "none"
    if ys:
        yi = ys[-1]                                   # innermost yellow on our left
        beyond = lines[:yi]
        lines = lines[yi:]
        Y = lines[0]
        Y["style"] = "solid"                           # US: left-edge yellow is always solid
        median = "double_yellow" if Y.get("double") else "barrier"
    else:
        beyond = []

    # fill gaps that are ≈ k·W with synthetic dashed lines
    filled = [lines[0]]
    for L in lines[1:]:
        gap = L["a"] - filled[-1]["a"]
        k = round(gap / W)
        if k >= 2 and abs(gap - k * W) < 0.6 * k:
            for j in range(1, k):
                filled.append(dict(a=filled[-1]["a"] + gap / k, color="white", style="dashed",
                                   double=False, synth=True))
        filled.append(L)
    lines = filled
    # right edge: if rightmost is dashed, the carriageway continues one more lane
    if lines[-1]["style"] == "dashed":
        lines.append(dict(a=lines[-1]["a"] + W, color="white", style="solid", double=False, synth=True))
    # left edge without yellow: close it with a solid white one lane further out
    if median == "none" and lines[0]["style"] == "dashed":
        lines.insert(0, dict(a=lines[0]["a"] - W, color="white", style="solid", double=False, synth=True))

    left_a, right_a = lines[0]["a"], lines[-1]["a"]
    barrier_a, onc = None, []
    if median == "barrier":
        barrier_a = left_a - 0.9
        o0 = barrier_a - 0.9
        onc = [dict(a=o0, color="yellow", style="solid", double=False, synth=True)]
        onc += [dict(a=o0 - k * W, color="white", style="dashed" if k < n_oncoming else "solid",
                     double=False, synth=True) for k in range(1, n_oncoming + 1)]
    elif median == "double_yellow":
        onc = [dict(a=left_a - k * W, color="white", style="dashed" if k < n_oncoming else "solid",
                    double=False, synth=True) for k in range(1, n_oncoming + 1)]
    return dict(b=b, c=c, W=W, lines=lines, left=left_a, right=right_a, median=median,
                barrier_a=barrier_a, oncoming_lines=onc, beyond_detected=len(beyond),
                crosswalk=crosswalk)


def x_at(road, a, y):
    return a + road["b"] * y + road["c"] * y * y


def heading_at(road, y):
    """Road tangent as yaw (CCW from +Y) at distance y."""
    return -math.atan(road["b"] + 2 * road["c"] * y)


def lane_centers(road):
    """Offsets `a` of all lane centres (ego carriageway + oncoming)."""
    L = [l["a"] for l in road["lines"]]
    cs = [(p + q) / 2 for p, q in zip(L, L[1:])]
    O = sorted([l["a"] for l in road["oncoming_lines"]])
    cs += [(p + q) / 2 for p, q in zip(O, O[1:])]
    return cs


def offset_of(road, x, y):
    """Inverse of x_at: lateral offset a of point (x, y)."""
    return x - road["b"] * y - road["c"] * y * y


def is_oncoming(road, x, y):
    a = offset_of(road, x, y)
    if road["median"] == "barrier":
        return a < road["barrier_a"]
    if road["median"] == "double_yellow":
        return a < road["left"]
    return False
