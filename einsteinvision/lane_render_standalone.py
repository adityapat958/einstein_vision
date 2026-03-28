"""Standalone Blender script: render lane data from lane_report_style.json.

Usage (lane-only, no background .blend file needed):
    blender --background \
        --python einsteinvision/lane_render_standalone.py -- \
        --lane-json cv_p3/lane_out/lane_report_style.json

Usage (render to image sequence):
    blender --background \
        --python einsteinvision/lane_render_standalone.py -- \
        --lane-json cv_p3/lane_out/lane_report_style.json \
        --output /tmp/lane_render/frame_####.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import bpy  # type: ignore
except ImportError:
    bpy = None

# Import directly from blender_render.py to avoid pulling in the full
# einsteinvision package (__init__.py → fusion → perception → yaml),
# none of which is available in Blender's embedded Python.
sys.path.insert(0, str(Path(__file__).parent))
from blender_render import LaneRenderConfig, LaneRenderer  # noqa: E402


def _setup_scene() -> None:
    """Clear the default Blender scene and add a camera + sun light."""
    # Remove everything in the default scene (Cube, Camera, Light)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Camera: sit 3 m up and 8 m behind the vehicle origin, look forward+down
    # Lane coords map to Blender as: vehicle-X→X, vehicle-Z→Y, vehicle-Y→Z
    cam_data = bpy.data.cameras.new("LaneCam")
    cam_data.lens = 28  # wide angle to capture more of the road
    cam_obj = bpy.data.objects.new("LaneCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -8.0, 3.0)   # behind & above origin
    cam_obj.rotation_euler = (1.1, 0.0, 0.0)  # ~63° tilt — looking forward
    bpy.context.scene.camera = cam_obj

    # Sun light for visibility
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.6, 0.2, 0.8)

    # Black background
    bpy.context.scene.world.color = (0.02, 0.02, 0.02)

    # Render settings: 1280×720, PNG
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.image_settings.file_format = "PNG"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EinsteinVision Lane-Only Blender Renderer")
    parser.add_argument("--lane-json", required=True, help="Path to lane_report_style.json")
    parser.add_argument("--output", default=None, help="Output path/pattern (e.g. /tmp/render/frame_####.png)")
    parser.add_argument("--bevel-depth", type=float, default=0.05, help="Lane tube radius in metres")
    parser.add_argument("--min-score", type=float, default=0.0, help="Min detection confidence to render")
    args = parser.parse_args(argv)

    if bpy is not None:
        _setup_scene()

    config = LaneRenderConfig(
        bevel_depth=args.bevel_depth,
        min_score=args.min_score,
    )
    renderer = LaneRenderer(config=config)
    renderer.render_lanes_from_json(args.lane_json)

    if args.output and bpy is not None:
        bpy.context.scene.render.filepath = args.output
        bpy.ops.render.render(animation=True, write_still=True)


if __name__ == "__main__":
    # Blender inserts its own args before '--'; strip them
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
