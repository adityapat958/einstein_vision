#!/bin/bash
# sync_from_cluster.sh
#
# Pulls all files needed for local Blender rendering from the Turing cluster.
#
# Usage:
#   bash sync_from_cluster.sh              # sync everything
#   bash sync_from_cluster.sh --det-only   # only detections.json (fast)
#   bash sync_from_cluster.sh --lanes-only # only lane JSONs
#
# Prerequisites:
#   - SSH alias "turing" configured in ~/.ssh/config
#     (or set CLUSTER env var: CLUSTER=user@turing.wpi.edu bash sync_from_cluster.sh)

CLUSTER="${CLUSTER:-turing}"
REMOTE_REPO="~/repos/cv/p3"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="all"
if [[ "${1:-}" == "--det-only" ]];   then MODE="det"; fi
if [[ "${1:-}" == "--lanes-only" ]]; then MODE="lanes"; fi

echo "========================================"
echo " Syncing from $CLUSTER:$REMOTE_REPO"
echo " Mode: $MODE"
echo " Local: $LOCAL_REPO"
echo "========================================"

# ── Phase 2 detections (89 MB total) ─────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "det" ]]; then
    echo ""
    echo "── Syncing phase2_output/ (detections.json for all 13 scenes)..."
    rsync -av --progress \
        --include="*/" \
        --include="detections.json" \
        --exclude="*" \
        "$CLUSTER:$REMOTE_REPO/phase2_output/" \
        "$LOCAL_REPO/phase2_output/"
fi

# ── Lane JSONs (small, needed for lane rendering) ─────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "lanes" ]]; then
    echo ""
    echo "── Syncing lane_out/ (lane_report_style.json for all 13 scenes)..."
    rsync -av --progress \
        --include="*/" \
        --include="lane_report_style.json" \
        --exclude="*" \
        "$CLUSTER:$REMOTE_REPO/cv_p3/lane_out/" \
        "$LOCAL_REPO/cv_p3/lane_out/"
fi

echo ""
echo "========================================"
echo " Sync complete."
echo " Run: bash slurm/render_phase2_local.sh 1   (test scene 1)"
echo "      bash slurm/render_phase2_local.sh      (all 13 scenes)"
echo "========================================"
