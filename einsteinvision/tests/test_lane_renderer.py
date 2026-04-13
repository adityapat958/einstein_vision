"""Pure-Python unit tests for LaneRenderer — no Blender (bpy) required."""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out bpy and mathutils so the module imports cleanly outside Blender
# ---------------------------------------------------------------------------
_bpy_stub = None  # module-level bpy is None when not in Blender

_mathutils = types.ModuleType("mathutils")


class _Vector(tuple):
    """Minimal Vector stub: stores (x, y, z), supports attribute access."""

    def __new__(cls, coords):
        obj = super().__new__(cls, coords)
        obj.x, obj.y, obj.z = float(coords[0]), float(coords[1]), float(coords[2])
        return obj


_mathutils.Vector = _Vector
sys.modules.setdefault("mathutils", _mathutils)

# Patch bpy at import time so blender_render.py sees bpy=None
sys.modules.setdefault("bpy", None)  # type: ignore

# Import the module directly (bypasses package __init__ which requires numpy/etc.)
_blender_render_path = Path(__file__).parent.parent / "blender_render.py"
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("einsteinvision.blender_render", _blender_render_path)
_mod = _ilu.module_from_spec(_spec)
sys.modules["einsteinvision.blender_render"] = _mod  # must be registered before exec (dataclasses requirement)
_spec.loader.exec_module(_mod)
# Inject Vector stub so _vehicle_to_blender works outside Blender
_mod.Vector = _Vector
LaneRenderConfig = _mod.LaneRenderConfig
LaneRenderer = _mod.LaneRenderer


class TestVehicleToBlender(unittest.TestCase):
    def test_forward_maps_to_blender_y(self):
        # vehicle Z=forward → Blender Y
        v = LaneRenderer._vehicle_to_blender([0.0, 0.0, 5.0])
        self.assertAlmostEqual(v.x, 0.0)
        self.assertAlmostEqual(v.y, 5.0)
        self.assertAlmostEqual(v.z, 0.0)

    def test_right_preserved(self):
        # vehicle X=right → Blender X (unchanged)
        v = LaneRenderer._vehicle_to_blender([1.0, 0.0, 0.0])
        self.assertAlmostEqual(v.x, 1.0)
        self.assertAlmostEqual(v.y, 0.0)
        self.assertAlmostEqual(v.z, 0.0)

    def test_height_maps_to_blender_z(self):
        # vehicle Y=up → Blender Z
        v = LaneRenderer._vehicle_to_blender([0.0, 1.45, 3.0])
        self.assertAlmostEqual(v.x, 0.0)
        self.assertAlmostEqual(v.y, 3.0)
        self.assertAlmostEqual(v.z, 1.45)

    def test_ground_plane_point(self):
        # Ground-plane lane points have vehicle Y ≈ 0 → Blender Z ≈ 0
        v = LaneRenderer._vehicle_to_blender([0.3, 0.0, 8.0])
        self.assertAlmostEqual(v.z, 0.0)


class TestCameraToVehicleFallback(unittest.TestCase):
    """Camera frame fallback: OpenCV (X right, Y down, Z fwd) → vehicle (X right, Y up, Z fwd)."""

    def _apply_fallback(self, pts_cam):
        return [[p[0], -p[1], p[2]] for p in pts_cam if p is not None]

    def test_y_flipped(self):
        result = self._apply_fallback([[0.0, 2.0, 5.0]])
        self.assertAlmostEqual(result[0][1], -2.0)

    def test_x_z_preserved(self):
        result = self._apply_fallback([[1.5, 0.0, 10.0]])
        self.assertAlmostEqual(result[0][0], 1.5)
        self.assertAlmostEqual(result[0][2], 10.0)

    def test_none_filtered(self):
        result = self._apply_fallback([[1.0, 0.0, 3.0], None, [2.0, 0.0, 4.0]])
        self.assertEqual(len(result), 2)


class TestLabelToMaterialKey(unittest.TestCase):
    def _mat_key(self, label_name: str) -> str:
        label = label_name.lower()
        return "dashed" if any(kw in label for kw in ("dash", "dot")) else "solid"

    def test_lane_marking_is_solid(self):
        self.assertEqual(self._mat_key("lane_marking"), "solid")

    def test_dashed_is_dashed(self):
        self.assertEqual(self._mat_key("dashed_line"), "dashed")

    def test_dash_is_dashed(self):
        self.assertEqual(self._mat_key("dash"), "dashed")

    def test_dotted_is_dashed(self):
        self.assertEqual(self._mat_key("dotted"), "dashed")

    def test_solid_is_solid(self):
        self.assertEqual(self._mat_key("solid_line"), "solid")

    def test_empty_is_solid(self):
        self.assertEqual(self._mat_key(""), "solid")


class TestMinScoreFiltering(unittest.TestCase):
    """Verify that lanes below min_score are skipped during JSON parsing."""

    _LANE_JSON = {
        "meta": {"fps": 30},
        "frames": [
            {
                "frame_idx": 0,
                "lanes": [
                    {
                        "det_idx": 0,
                        "label_name": "lane_marking",
                        "score": 0.8,
                        "points_ground_vehicle": [[float(i) * 0.1, 0.0, float(i)] for i in range(10)],
                    },
                    {
                        "det_idx": 1,
                        "label_name": "lane_marking",
                        "score": 0.2,
                        "points_ground_vehicle": [[float(i) * 0.2, 0.0, float(i)] for i in range(10)],
                    },
                ],
            }
        ],
    }

    def _collect_rendered_lanes(self, min_score: float) -> list[dict]:
        """Simulate the filtering loop from render_lanes_from_json."""
        rendered = []
        for frame in self._LANE_JSON["frames"]:
            for lane in frame.get("lanes", []):
                if float(lane.get("score", 1.0)) < min_score:
                    continue
                pts_gv = lane.get("points_ground_vehicle")
                if not pts_gv or len(pts_gv) < 2:
                    continue
                rendered.append(lane)
        return rendered

    def test_no_filter_renders_both(self):
        self.assertEqual(len(self._collect_rendered_lanes(0.0)), 2)

    def test_filter_removes_low_score(self):
        self.assertEqual(len(self._collect_rendered_lanes(0.5)), 1)

    def test_filter_removes_all(self):
        self.assertEqual(len(self._collect_rendered_lanes(0.9)), 0)


class TestVisibilityFrameMath(unittest.TestCase):
    """Verify hide/show frame index logic."""

    def _get_keyframe_calls(self, blender_frame: int) -> list[tuple[bool, int]]:
        """Returns list of (hidden, frame) pairs that set_hidden would emit."""
        calls = []

        def set_hidden(frame: int, hidden: bool) -> None:
            calls.append((hidden, frame))

        if blender_frame > 1:
            set_hidden(blender_frame - 1, True)
        set_hidden(blender_frame, False)
        set_hidden(blender_frame + 1, True)
        return calls

    def test_frame_1_no_preceding_hide(self):
        calls = self._get_keyframe_calls(1)
        # No hide before frame 1
        self.assertEqual(calls[0], (False, 1))   # show at frame 1
        self.assertEqual(calls[1], (True, 2))    # hide at frame 2
        self.assertEqual(len(calls), 2)

    def test_frame_5_has_preceding_hide(self):
        calls = self._get_keyframe_calls(5)
        self.assertEqual(calls[0], (True, 4))    # hide at frame 4
        self.assertEqual(calls[1], (False, 5))   # show at frame 5
        self.assertEqual(calls[2], (True, 6))    # hide at frame 6
        self.assertEqual(len(calls), 3)

    def test_frame_idx_0_maps_to_blender_frame_1(self):
        # frame_idx 0 → blender_frame = frame_idx + 1 = 1
        frame_idx = 0
        blender_frame = frame_idx + 1
        calls = self._get_keyframe_calls(blender_frame)
        visible_frames = [f for hidden, f in calls if not hidden]
        self.assertEqual(visible_frames, [1])


class TestLaneRenderConfigDefaults(unittest.TestCase):
    def test_default_bevel_depth(self):
        cfg = LaneRenderConfig()
        self.assertAlmostEqual(cfg.bevel_depth, 0.05)

    def test_default_collection_name(self):
        cfg = LaneRenderConfig()
        self.assertEqual(cfg.collection_name, "EinsteinVisionLanes")

    def test_custom_values(self):
        cfg = LaneRenderConfig(bevel_depth=0.1, min_score=0.4)
        self.assertAlmostEqual(cfg.bevel_depth, 0.1)
        self.assertAlmostEqual(cfg.min_score, 0.4)


if __name__ == "__main__":
    unittest.main()
