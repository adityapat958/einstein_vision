#!/bin/bash
# Submit Blender lane render job (CPU — reads JSONs, renders PNG sequences)
# Run AFTER submit_lane_export.sh completes (or pass --dependency).
# Usage: bash submit_blender_lanes.sh [lane_export_job_id]

set -e

SBATCH_FILE="/home/apatwardhan/repos/cv/p3/run_blender_lanes.sbatch"

DEPENDENCY=""
if [ -n "$1" ]; then
    DEPENDENCY="--dependency=afterok:$1"
    echo "Will run after job $1 completes."
fi

JOB_ID=$(sbatch $DEPENDENCY "$SBATCH_FILE" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo "Blender lane render job submitted: $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -j $JOB_ID"
echo "  tail -f blender_lanes_${JOB_ID}.out"
echo "  python3 /home/apatwardhan/repos/cv/p3/monitor_job.py $JOB_ID --follow"
echo ""
echo "Cancel: scancel $JOB_ID"
