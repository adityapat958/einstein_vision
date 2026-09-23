"""Geometry diagnostics for mockup_render layouts (runs in Blender, no render).

For a frame: normalised asset dims, pairwise footprint overlap (clipping),
and reprojection of each placed vehicle's 3D box into the real front camera
vs. its detection bbox (scale / placement error).

    blender -b --factory-startup --python einsteinvision/mockup_diag.py -- \
        --json phase2_output/scene1/detections.json --frames 2131 1530 1200
"""
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mockup_render as mr          # noqa: E402
from mathutils import Vector        # noqa: E402

FY, CY, CAM_H = mr.FY, mr.CY, mr.CAM_H
PITCH = math.atan((mr.CY - mr.V_HORIZON) / mr.FY)     # + = down


def tpl_dims(assets, rel):
    col = mr.asset_template(assets, rel)
    pts = [o.matrix_world @ Vector(c) for o in col.objects for c in o.bound_box]
    mn = [min(p[i] for p in pts) for i in range(3)]
    mx = [max(p[i] for p in pts) for i in range(3)]
    return [mx[i] - mn[i] for i in range(3)], mn, mx


def footprint(p, dims):
    w, l = dims[0], dims[1]
    c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
    return [(p["x"] + dx * c - dy * s, p["y"] + dx * s + dy * c)
            for dx, dy in ((-w/2, -l/2), (w/2, -l/2), (w/2, l/2), (-w/2, l/2))]


def _sat_overlap(a, b):
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * x + ny * y for x, y in a]; pb = [nx * x + ny * y for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return False
    return True


def project(x, y, z):
    """world (x right, y fwd, z up; camera at (0,0,CAM_H) pitched down) → pixel."""
    zc = y * math.cos(PITCH) + (CAM_H - z) * math.sin(PITCH)
    yc = -y * math.sin(PITCH) + (CAM_H - z) * math.cos(PITCH)   # image-down
    if zc <= 0.1:
        return None
    return mr.CX + mr.FX * x / zc, CY + FY * yc / zc


def reproj_box(p, dims):
    fp = footprint(p, dims)
    uv = [project(x, y, z) for x, y in fp for z in (0.0, dims[2])]
    uv = [q for q in uv if q]
    return [min(q[0] for q in uv), min(q[1] for q in uv), max(q[0] for q in uv), max(q[1] for q in uv)]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--assets", default="P3Data/Assets")
    ap.add_argument("--frames", type=int, nargs="+", default=[2131])
    a = ap.parse_args(argv)
    assets = Path(a.assets)
    mr.reset_scene()
    print("=== normalised asset dims (w, l, h) m ===")
    dims = {}
    for rel in mr.ASSET_SPEC:
        d, mn, mx = tpl_dims(assets, rel)
        dims[rel] = d
        print(f"  {rel:36s} {d[0]:6.2f} {d[1]:6.2f} {d[2]:6.2f}   zmin={mn[2]:+.2f}")
    data = json.loads(Path(a.json).read_text())
    frames = data["frames"]
    for f in a.frames:
        fi = next(i for i, fr in enumerate(frames) if fr["frame_index"] == f)
        placed = mr.layout_objects(frames, fi, float(data.get("fps", 30)))
        ratios = []
        # keep det bbox for each placed object (layout order == kept order)
        print(f"\n=== frame {f}: {len(placed)} objects ===")
        for p in placed:
            rb = reproj_box(p, dims[p["asset"]])
            db = p.get("bbox")
            msg = ""
            if db:
                rw, dw = rb[2] - rb[0], db[2] - db[0]
                rh, dh = rb[3] - rb[1], db[3] - db[1]
                ratios += [rw / dw, rh / dh]
                msg = f"w_ratio={rw/dw:4.2f} h_ratio={rh/dh:4.2f} du={(rb[0]+rb[2]-db[0]-db[2])/2:+6.0f}px"
            print(f"  {p['cls']:6s} {str(p['sub']):9s} x={p['x']:6.1f} y={p['y']:5.1f} {msg}")
        hits = 0
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                A, B = placed[i], placed[j]
                if _sat_overlap(footprint(A, dims[A["asset"]]), footprint(B, dims[B["asset"]])):
                    hits += 1
                    print(f"  CLIP: {A['sub']}@({A['x']:.1f},{A['y']:.1f}) × {B['sub']}@({B['x']:.1f},{B['y']:.1f})")
        ego = dict(x=0.0, y=0.0, yaw=0.0)
        for P in placed:
            if _sat_overlap(footprint(ego, dims["Vehicles/SedanAndHatchback.blend"]), footprint(P, dims[P["asset"]])):
                hits += 1; print(f"  CLIP with EGO: {P['sub']}@({P['x']:.1f},{P['y']:.1f})")
        ratios.sort()
        print(f"  clipping pairs: {hits}   size ratio median={ratios[len(ratios)//2]:.2f} "
              f"range=[{ratios[0]:.2f},{ratios[-1]:.2f}]")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
