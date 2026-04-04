#!/bin/bash

#SBATCH --mail-user=apatwardhan@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J rcnn_lane_detection
#SBATCH -o logs/rcnn_lane_detection_%j.out
#SBATCH -e logs/rcnn_lane_detection_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100|H100|H200|A30
#SBATCH -p long
#SBATCH -t 48:00:00

conda init
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/apatwardhan/repos/cv/p3/cv_p3/RCNN_lane_detection

python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene2/Undist/2023-03-03_10-31-11-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene3/Undist/2023-02-14_11-49-54-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene4/Undist/2023-02-14_11-51-54-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene5/Undist/2023-02-14_11-56-56-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene6/Undist/2023-03-03_15-31-56-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene7/Undist/2023-03-03_11-21-43-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene8/Undist/2023-03-03_11-40-47-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene9/Undist/2023-03-04_17-20-36-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene10/Undist/2023-03-06_19-48-30-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene11/Undist/2023-03-11_17-19-53-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene12/Undist/2023-03-13_06-00-16-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
python3 inference_video.py -i /home/apatwardhan/repos/cv/p3/cv_p3/P3Data/Sequences/scene13/Undist/2023-03-03_06-59-50-front_undistort.mp4 -t 0.5 -w outputs/training/road_line/model_15.pth
