#!/bin/bash
# Submit lane export job (GPU — generates lane_report_style.json per scene)
# Usage: bash submit_lane_export.sh

set -e

SBATCH_FILE="/home/apatwardhan/repos/cv/p3/run_lane_export.sbatch"

JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo "Lane export job submitted: $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -j $JOB_ID"
echo "  tail -f lane_export_${JOB_ID}.out"
echo "  python3 /home/apatwardhan/repos/cv/p3/monitor_job.py $JOB_ID --follow"
echo ""
echo "Cancel: scancel $JOB_ID"
