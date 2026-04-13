#!/bin/bash
# render_phase2_local.sh
#
# Run Phase 2 Blender rendering locally (Blender did not work on Turing cluster).
# Run this AFTER phase2_pipeline.py has completed on the cluster and you have
# synced the JSON outputs back to your local machine.
#
# Prerequisites:
#   - Blender installed locally (adjust BLENDER path below)
#   - ffmpeg installed (brew install ffmpeg  or  apt install ffmpeg)
#   - phase2_output/sceneN/detections.json   (from cluster run)
#   - cv_p3/lane_out/sceneN/lane_report_style.json  (from cluster run)
#
# Usage:
#   bash render_phase2_local.sh             # all 13 scenes
#   bash render_phase2_local.sh 1           # scene 1 only
#   bash render_phase2_local.sh 1 3         # scenes 1 and 3

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
# Adjust BLENDER to your local Blender binary
if [[ "$OSTYPE" == "darwin"* ]]; then
    BLENDER_DEFAULT="/Applications/Blender.app/Contents/MacOS/blender"
else
    BLENDER_DEFAULT="blender"
fi
BLENDER="${BLENDER_OVERRIDE:-$BLENDER_DEFAULT}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/einsteinvision/blender_render.py"
ASSETS="$REPO/cv_p3/P3Data/Assets"
LANE_OUT="$REPO/cv_p3/lane_out"
DET_OUT="$REPO/phase2_output"
RENDER_OUT="$REPO/cv_p3/blender_renders"
VIDEO_OUT="$REPO/cv_p3/blender_renders"

LANE_BEVEL=0.05
LANE_MIN_SCORE=0.1
RENDER_FPS=30

# ── Scene list ────────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    SCENES=("$@")
else
    SCENES=(1 2 3 4 5 6 7 8 9 10 11 12 13)
fi

echo "========================================"
echo " EinsteinVision Phase 2 — Local Render"
echo " Scenes: ${SCENES[*]}"
echo " Blender: $BLENDER"
echo "========================================"

for SCENE_NUM in "${SCENES[@]}"; do
    SCENE="scene${SCENE_NUM}"
    LANE_JSON="$LANE_OUT/$SCENE/lane_report_style.json"
    DET_JSON="$DET_OUT/$SCENE/detections.json"
    FRAME_DIR="$RENDER_OUT/$SCENE"
    OUTPUT_MP4="$VIDEO_OUT/${SCENE}.mp4"

    echo ""
    echo "── Scene $SCENE_NUM ──────────────────────────────"

    # Validate inputs
    if [[ ! -f "$DET_JSON" ]]; then
        echo "  [WARN] Detections JSON not found: $DET_JSON (will render lanes only)"
    fi

    mkdir -p "$FRAME_DIR"

    # Build Python script args (after the -- separator)
    SCRIPT_ARGS=(
        --assets-dir "$ASSETS"
        --lane-bevel-depth "$LANE_BEVEL"
        --lane-min-score "$LANE_MIN_SCORE"
        --output "$FRAME_DIR/frame_####.png"
    )
    if [[ -f "$LANE_JSON" ]]; then
        SCRIPT_ARGS+=(--lane-json "$LANE_JSON")
    else
        echo "  [INFO] No lane JSON found — rendering objects only"
    fi
    if [[ -f "$DET_JSON" ]]; then
        SCRIPT_ARGS+=(--json "$DET_JSON")
    fi

    echo "  Running Blender..."
    "$BLENDER" --background --python "$SCRIPT" -- "${SCRIPT_ARGS[@]}"

    # Count frames rendered
    N_FRAMES=$(ls "$FRAME_DIR"/frame_*.png 2>/dev/null | wc -l | tr -d ' ')
    echo "  Rendered $N_FRAMES PNG frames → $FRAME_DIR"

    if [[ $N_FRAMES -eq 0 ]]; then
        echo "  [WARN] No frames rendered for $SCENE — skipping encode."
        continue
    fi

    # Encode to MP4 with ffmpeg
    echo "  Encoding to MP4..."
    ffmpeg -y \
        -r "$RENDER_FPS" \
        -i "$FRAME_DIR/frame_%04d.png" \
        -c:v libx264 \
        -preset slow \
        -crf 18 \
        -pix_fmt yuv420p \
        "$OUTPUT_MP4" \
        -loglevel warning

    echo "  Saved: $OUTPUT_MP4"
done

echo ""
echo "========================================"
echo " All done."
echo "========================================"
