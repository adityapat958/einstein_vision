# RCNN Lane Detection - Phase 1 Job Submission Guide

## Quick Start

### 1. Submit the job to Turing cluster:
```bash
cd /home/apatwardhan/repos/cv/p3
bash submit_job.sh
```

This will:
- Submit the SBATCH script to SLURM
- Print the Job ID
- Show you how to monitor it

### 2. Monitor your job:

**Check current status:**
```bash
python3 monitor_job.py <JOB_ID>
```

**Real-time monitoring (updates every 10s):**
```bash
python3 monitor_job.py <JOB_ID> --follow
```

**View last N lines of output:**
```bash
python3 monitor_job.py <JOB_ID> --output --tail 50
```

**View error logs:**
```bash
python3 monitor_job.py <JOB_ID> --error --tail 50
```

**View processing summary and generated outputs:**
```bash
python3 monitor_job.py <JOB_ID> --results
```

### 3. Check SLURM queue directly:
```bash
squeue -j <JOB_ID>                    # Check your job
squeue -u apatwardhan                 # All your jobs
scancel <JOB_ID>                      # Cancel a job
```

## What the Script Does

The `run_lane_detection.sbatch` script:

1. **Allocates resources on Turing:**
   - 1 node, 4 CPU cores
   - 32 GB RAM
   - 1 GPU (RTX6000B, A100, or V100)
   - Long partition (up to 24 hours)

2. **Processes all 13 sequences:**
   - Reads from: `cv_p3/P3Data/Sequences/scene{1-13}/video_undistorted.mp4`
   - Runs RCNN lane detection with **threshold 0.3** (lower than the previous 0.9)
   - Generates detections for each frame

3. **Saves results to:**
   - `cv_p3/lane_detection_results/`
   - Individual videos: `lane_detection_scene{1-13}.mp4`
   - Processing log: `processing_log.txt`
   - Individual scene logs: `scene{i}_inference.log`

## Configuration

To modify the script, edit `run_lane_detection.sbatch`:

| Parameter | Current | Notes |
|-----------|---------|-------|
| Threshold | 0.3 | Lower = more detections. Change `-t 0.3` to your value |
| Batch size | N/A | Inference runs per-frame |
| GPU time limit | 24h | Change `-t 24:00:00` if needed |
| GPU type | Any | RTX6000B, A100, or V100 |

## Common Commands

```bash
# Save output for later review
python3 monitor_job.py <JOB_ID> --output --tail 100 > job_output.txt

# Cancel the job
scancel <JOB_ID>

# Watch for completion (stop with Ctrl+C)
while squeue -j <JOB_ID> > /dev/null 2>&1; do 
  echo "Still running..."; sleep 60; 
done && echo "Job complete!"

# View all output files
ls -lh cv_p3/lane_detection_results/
```

## Troubleshooting

**Job not found:**
- Wait a few seconds for it to appear on the queue
- Check you used the correct Job ID
- Verify on the SLURM web portal: https://hpc.wpi.edu/slurm/

**Job failed:**
1. Check error output: `python3 monitor_job.py <JOB_ID> --error`
2. Check last lines of output: `python3 monitor_job.py <JOB_ID> --output --tail 100`
3. Common issues:
   - Missing model checkpoint at `RCNN_lane_detection/out/checkpoint.pth`
   - Missing input videos in `P3Data/Sequences/`
   - Out of GPU memory (reduces batch size in code if needed)

**Job taking too long:**
- Current threshold 0.3 should be manageable
- Each sequence is independent; if one scene fails, others continue
- Can adjust timeout in script: `#SBATCH -t HH:MM:SS`

## Output Format

Results in `cv_p3/lane_detection_results/`:
```
lane_detection_results/
├── processing_log.txt           # Master log file
├── lane_detection_scene1.mp4    # Rendered lane detection (scene 1)
├── lane_detection_scene2.mp4    # Rendered lane detection (scene 2)
├── ...
├── lane_detection_scene13.mp4   # Rendered lane detection (scene 13)
├── scene1_inference.log         # Individual logs for debugging
├── scene2_inference.log
└── ...
```

## Next Steps for Phase 1

1. After job completes and outputs are generated:
   - Review the lane detection quality
   - Adjust threshold if needed and re-submit
   
2. For rendering in Blender:
   - Use `cv_p3/RCNN_lane_detection/blender_import_lanes.py`
   - Import the lane detection JSON outputs
   
3. Prepare Phase 1 submission:
   - Organize rendered images/videos
   - Create `References.md` listing all packages used
   - Zip as `Group<NUM>_p3ph1.zip`

---
**Job ID Format:** Remember to save your Job ID so you can monitor it later!
