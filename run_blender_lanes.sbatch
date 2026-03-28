#!/bin/bash

#SBATCH --mail-user=apatwardhan@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J EV_blender_lanes
#SBATCH -o blender_lanes_%j.out
#SBATCH -e blender_lanes_%j.err
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -p short
#SBATCH -t 04:00:00

# Load X11 stack for headless rendering:
#   xvfb         — virtual framebuffer (fake X display)
#   libxxf86vm   — Blender links against this X11 extension
#   libxfont2    — Xvfb needs this for font rendering
module load xvfb/focal
module load libxxf86vm/1.1.5/o24nj2h
module load libxfont2/2.0.7/tggv7jy

# Spack modules don't always add to LD_LIBRARY_PATH — add manually
XFONT2_LIB=/cm/shared/spack/opt/spack/linux-x86_64/libxfont2-2.0.7-tggv7jyfvnnqrkggbzhp6xuclfwfiexm/lib
XXF86VM_LIB=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/libxxf86vm-1.1.5-o24nj2he2v2kxnmovhk57d6ix2stwpkq/lib
export LD_LIBRARY_PATH="${XFONT2_LIB}:${XXF86VM_LIB}:${LD_LIBRARY_PATH}"

BLENDER_DIR="$HOME/blender-3.3.21-linux-x64"
BLENDER_BIN="$BLENDER_DIR/blender"
REPO=/home/apatwardhan/repos/cv/p3
SCRIPT=$REPO/einsteinvision/lane_render_standalone.py
LANE_OUT=$REPO/cv_p3/lane_out
RENDER_OUT=$REPO/cv_p3/blender_renders

if [ ! -x "$BLENDER_BIN" ]; then
    echo "ERROR: Blender not found at $BLENDER_BIN — run: bash $REPO/install_blender.sh"
    exit 1
fi

# Start persistent virtual display
Xvfb :99 -screen 0 1280x720x24 &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start — check .err log"
    exit 1
fi
echo "Xvfb running on DISPLAY=$DISPLAY (PID=$XVFB_PID)"
echo "Using: $BLENDER_BIN"
$BLENDER_BIN --version

for i in {1..13}; do
    SCENE="scene${i}"
    LANE_JSON="$LANE_OUT/$SCENE/lane_report_style.json"

    if [ ! -f "$LANE_JSON" ] && [ -f "$LANE_OUT/lane_report_style.json" ] && [ "$SCENE" = "scene1" ]; then
        LANE_JSON="$LANE_OUT/lane_report_style.json"
        echo "[FALLBACK] $SCENE: using $LANE_JSON"
    fi

    if [ ! -f "$LANE_JSON" ]; then
        echo "[SKIP] $SCENE: no lane JSON at $LANE_JSON"
        continue
    fi

    OUTPUT_DIR="$RENDER_OUT/$SCENE"
    mkdir -p "$OUTPUT_DIR"

    echo "[START] $SCENE ($(date))"

    $BLENDER_BIN --background \
        --python "$SCRIPT" -- \
        --lane-json "$LANE_JSON" \
        --output "$OUTPUT_DIR/frame_####.png" \
        --bevel-depth 0.05 \
        --min-score 0.3

    if [ $? -eq 0 ]; then
        echo "[DONE]  $SCENE → $OUTPUT_DIR ($(date))"
    else
        echo "[ERROR] $SCENE failed — see blender_lanes_${SLURM_JOB_ID}.err"
    fi
done

kill $XVFB_PID 2>/dev/null || true
echo "Blender lane renders complete: $(date)"
echo "Frames written to $RENDER_OUT/scene{1..13}/"
