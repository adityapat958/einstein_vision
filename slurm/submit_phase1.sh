#!/bin/bash
# Submit Phase 1 job to SLURM
# Usage: ./submit_phase1.sh

echo "Submitting EinsteinVision Phase 1 job to SLURM..."

# Check if files exist
if [ ! -f "cv_p3/Code/phase1_pipeline.py" ]; then
    echo "ERROR: phase1_pipeline.py not found!"
    exit 1
fi

# Submit job
JOB_ID=$(sbatch run_phase1.sbatch | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo ""
echo "Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f einsteinvision_phase1_${JOB_ID}.out"
echo ""
echo "Cancel with:"
echo "  scancel $JOB_ID"
