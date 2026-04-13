#!/bin/bash
"""
Quick submission script for RCNN lane detection job.
Usage: bash submit_job.sh
"""

set -e

SBATCH_FILE="/home/apatwardhan/repos/cv/p3/run_lane_detection.sbatch"

if [ ! -f "$SBATCH_FILE" ]; then
    echo "ERROR: SBATCH script not found at $SBATCH_FILE"
    exit 1
fi

echo "Submitting lane detection job to SLURM..."
echo ""

# Submit job and capture job ID
JOB_OUTPUT=$(sbatch "$SBATCH_FILE")
JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')

echo "✓ Job submitted successfully!"
echo "  Job ID: $JOB_ID"
echo "  Command: sbatch $SBATCH_FILE"
echo ""
echo "Monitor with one of these commands:"
echo "  python3 /home/apatwardhan/repos/cv/p3/monitor_job.py $JOB_ID                    # Check status"
echo "  python3 /home/apatwardhan/repos/cv/p3/monitor_job.py $JOB_ID --follow          # Real-time monitoring"
echo "  python3 /home/apatwardhan/repos/cv/p3/monitor_job.py $JOB_ID --output --tail 50  # Last 50 lines"
echo "  python3 /home/apatwardhan/repos/cv/p3/monitor_job.py $JOB_ID --results         # View results summary"
echo ""
echo "Check job queue:"
echo "  squeue -j $JOB_ID"
echo ""
echo "Job ID saved. You can check it anytime with:"
echo "  python3 /home/apatwardhan/repos/cv/p3/monitor_job.py $JOB_ID"
