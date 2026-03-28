#!/bin/bash

#SBATCH --mail-user=apatwardhan@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J EV_lane_export
#SBATCH -o lane_export_%j.out
#SBATCH -e lane_export_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -C RTX6000B|A100|V100
#SBATCH -p long
#SBATCH -t 12:00:00

conda init
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/apatwardhan/repos/cv/p3
RCNN_DIR=$REPO/cv_p3/RCNN_lane_detection
WEIGHTS=$RCNN_DIR/outputs/training/road_line/model_15.pth
DATA_DIR=$REPO/cv_p3/P3Data/Sequences
OUT_BASE=$REPO/cv_p3/lane_out

CLASS_NAMES="__background__ divider-line dotted-line double-line random-line road-sign-line solid-line"
LANE_IDS="1 2 3 4 5"   # exclude road-sign-line(5) if desired; use "1 2 3 4 5 6" for all

cd $RCNN_DIR

for i in {1..13}; do
    SCENE="scene${i}"
    INPUT=$(find $DATA_DIR/$SCENE/Undist -name "*front_undistort.mp4" 2>/dev/null | head -1)

    if [ -z "$INPUT" ]; then
        echo "[SKIP] $SCENE: no undistorted video found"
        continue
    fi

    OUTPUT_DIR="$OUT_BASE/$SCENE"
    mkdir -p "$OUTPUT_DIR"

    echo "[START] $SCENE → $OUTPUT_DIR  ($(date))"

    python3 lane_export.py \
        --weights "$WEIGHTS" \
        --input "$INPUT" \
        --output-dir "$OUTPUT_DIR" \
        --threshold 0.30 \
        --mask-threshold 0.35 \
        --num-classes 7 \
        --class-names $CLASS_NAMES \
        --lane-class-ids 1 2 3 4 5 6

    if [ $? -eq 0 ]; then
        echo "[DONE]  $SCENE ($(date))"
    else
        echo "[ERROR] $SCENE failed — see lane_export_${SLURM_JOB_ID}.err"
    fi
done

echo "Lane export complete: $(date)"
echo "JSONs written to $OUT_BASE/scene{1..13}/lane_report_style.json"
