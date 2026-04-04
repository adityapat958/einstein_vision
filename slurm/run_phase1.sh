#!/bin/bash

#SBATCH --mail-user=apatwardhan@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J EinsteinVision_Phase1
#SBATCH -o logs/einsteinvision_phase1_%j.out
#SBATCH -e logs/einsteinvision_phase1_%j.err
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100
#SBATCH -p long
#SBATCH -t 24:00:00

conda init
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/apatwardhan/repos/cv/p3/cv_p3

python3 Code/phase1_pipeline.py --scene all --output-dir phase1_output --device 0
