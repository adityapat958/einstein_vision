#!/bin/bash

#SBATCH --mail-user=apatwardhan@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J RCNN_lane_detection_p3
#SBATCH -o logs/rcnn_lane_detection_%j.out
#SBATCH -e logs/rcnn_lane_detection_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100|H100|H200|A30
#SBATCH -p long
#SBATCH -t 24:00:00

# conda init 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl 

# Set working directory
cd /home/apatwardhan/repos/cv/p3/cv_p3

# Create output directory for results
OUTPUT_DIR="lane_detection_results"
mkdir -p $OUTPUT_DIR

# Log file for tracking all runs
LOG_FILE="${OUTPUT_DIR}/processing_log.txt"
echo "Lane Detection Processing Started: $(date)" > $LOG_FILE
echo "Threshold: 0.3" >> $LOG_FILE
echo "Processing all 13 sequences..." >> $LOG_FILE
echo "-----------------------------------" >> $LOG_FILE

# Run RCNN lane detection on all 13 sequences
for i in {1..13}; do
    SCENE="scene${i}"
    INPUT_VIDEO=$(ls /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/${SCENE}/Undist/*front_undistort.mp4 2>/dev/null | head -1)

    if [ -z "$INPUT_VIDEO" ]; then
        echo "$(date): ERROR - Video not found for ${SCENE}" | tee -a $LOG_FILE
        continue
    fi
    
    echo "$(date): Processing $SCENE..." | tee -a $LOG_FILE
    
    # Run inference with lower threshold (0.3)
    python3 RCNN_lane_detection/inference_video.py \
        -i "$INPUT_VIDEO" \
        -t 0.3 \
        -w RCNN_lane_detection/outputs/training/road_line/model_15.pth \
        2>"${OUTPUT_DIR}/${SCENE}_inference.err" | tee -a "${OUTPUT_DIR}/${SCENE}_inference.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "$(date): Completed $SCENE successfully" | tee -a $LOG_FILE
        # Copy output to results folder with scene name
        INFERENCE_OUTPUT="RCNN_lane_detection/outputs/inference/video_undistorted.mp4"
        if [ -f "$INFERENCE_OUTPUT" ]; then
            cp "$INFERENCE_OUTPUT" "${OUTPUT_DIR}/lane_detection_${SCENE}.mp4"
            echo "$(date): Saved output to ${OUTPUT_DIR}/lane_detection_${SCENE}.mp4" | tee -a $LOG_FILE
        fi
    else
        echo "$(date): ERROR processing $SCENE - see ${OUTPUT_DIR}/${SCENE}_inference.log" | tee -a $LOG_FILE
    fi
    echo "-----------------------------------" >> $LOG_FILE
done

echo "Lane Detection Processing Completed: $(date)" | tee -a $LOG_FILE
echo "All results saved to: $OUTPUT_DIR" | tee -a $LOG_FILE
