"""Download CC0 Poly Haven assets (models as .blend, textures as maps) at 1k.

    python3 einsteinvision/fetch_polyhaven.py --out P3Data/ExtraAssets
"""
import argparse, json, os, sys, urllib.request
from pathlib import Path

MODELS = ["street_lamp_01", "street_lamp_02", "concrete_road_barrier", "fire_hydrant",
          "metal_trash_can", "modular_electricity_poles", "utility_box_01", "water_manhole_cover",
          "shrub_01", "shrub_02", "shrub_04", "tree_small_02", "jacaranda_tree", "planter_box_01"]
TEXTURES = ["asphalt_02", "clean_asphalt", "concrete_floor_02", "brick_crosswalk"]
TEXTURES_2K = ["aerial_grass_rock"]                     # roadside ground (seen at grazing angles)
HDRIS = ["kloofendal_48d_partly_cloudy_puresky", "kloofendal_overcast_puresky", "overcast_soil_puresky"]
UA = {"User-Agent": "einstein-vision/1.0"}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


def dl(url, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
    dst.write_bytes(data)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="P3Data/ExtraAssets")
    ap.add_argument("--res", default="1k")
    a = ap.parse_args()
    out = Path(a.out)
    total = 0
    for mid in MODELS:
        try:
            f = get(f"https://api.polyhaven.com/files/{mid}")
            b = f["blend"][a.res]["blend"]
            total += dl(b["url"], out / mid / Path(b["url"]).name)
            for rel, inc in b.get("include", {}).items():
                total += dl(inc["url"], out / mid / rel)
            print(f"[ph] model {mid} ok")
        except Exception as e:
            print(f"[ph] model {mid} FAILED {e}")
    for tid in TEXTURES:
        try:
            f = get(f"https://api.polyhaven.com/files/{tid}")
            for m in ("Diffuse", "nor_gl", "Rough", "Displacement", "arm"):
                if m in f and a.res in f[m]:
                    fm = f[m][a.res].get("jpg") or f[m][a.res].get("png")
                    total += dl(fm["url"], out / "textures" / tid / Path(fm["url"]).name)
            print(f"[ph] texture {tid} ok")
        except Exception as e:
            print(f"[ph] texture {tid} FAILED {e}")
    for tid in TEXTURES_2K:
        try:
            f = get(f"https://api.polyhaven.com/files/{tid}")
            for m in ("Diffuse", "nor_gl", "Rough"):
                total += dl(f[m]["2k"]["jpg"]["url"], out / "textures" / tid / Path(f[m]["2k"]["jpg"]["url"]).name)
            print(f"[ph] texture {tid} (2k) ok")
        except Exception as e:
            print(f"[ph] texture {tid} FAILED {e}")
    for hid in HDRIS:                                   # sky-only HDRIs: our own ground plane stays visible
        try:
            u = get(f"https://api.polyhaven.com/files/{hid}")["hdri"]["2k"]["hdr"]["url"]
            total += dl(u, out / "hdri" / f"{hid}.hdr")
            print(f"[ph] hdri {hid} ok")
        except Exception as e:
            print(f"[ph] hdri {hid} FAILED {e}")
    (out / "LICENSE.txt").write_text("All assets from https://polyhaven.com — CC0 (public domain).\n")
    print(f"[ph] downloaded {total/1e6:.1f} MB into {out}")


if __name__ == "__main__":
    main()
