"""Which end of each normalised vehicle asset is the front? (geometry heuristic)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import mockup_render as mr
from mathutils import Vector
mr.reset_scene()
for rel in [k for k in mr.ASSET_SPEC if k.startswith("Vehicles/")]:
    col = mr.asset_template(Path("P3Data/Assets"), rel)
    pts = [o.matrix_world @ v.co for o in col.objects for v in o.data.vertices]
    ys = [p.y for p in pts]; zs = [p.z for p in pts]
    y0, y1, h = min(ys), max(ys), max(zs)
    L = y1 - y0
    # height profile along length: max z in 10 slices
    prof = []
    for i in range(10):
        a, b = y0 + i * L / 10, y0 + (i + 1) * L / 10
        zz = [p.z for p in pts if a <= p.y < b]
        prof.append(round(max(zz) / h, 2) if zz else 0)
    high = [p.y for p in pts if p.z > 0.8 * h]
    hy = (sum(high) / len(high) - y0) / L if high else 0.5
    print(f"{rel:34s} profile(-Y→+Y)={prof}  tall-part at {hy:.2f} of length")
