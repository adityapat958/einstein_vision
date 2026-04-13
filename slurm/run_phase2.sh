#!/bin/bash
#SBATCH --mail-user=apatwardhan@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J phase2_pipeline
#SBATCH -o logs/phase2_%j.out
#SBATCH -e logs/phase2_%j.err
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100
#SBATCH -p long
#SBATCH -t 24:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ev

export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/home/apatwardhan/repos/cv/p3
CALIB_MAT=$REPO/cv_p3/calibration.mat
WEIGHTS=$REPO/cv_p3/ext_models/yolo11n.pt

cd $REPO

echo "[Phase 2] Starting pipeline at $(date)"
echo "[Phase 2] Node: $SLURMD_NODENAME"
echo "[Phase 2] GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python3 cv_p3/Code/phase2_pipeline.py \
    --scene all \
    --weights "$WEIGHTS" \
    --calib-file "$CALIB_MAT" \
    --depth-model-size small \
    --device cuda \
    --camera-height 1.45 \
    --camera-pitch-deg 5.0 \
    --no-pose

echo "[Phase 2] Done at $(date)"
