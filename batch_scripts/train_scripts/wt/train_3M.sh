#!/bin/bash

#SBATCH --job-name=my_project       # Name of the job
#SBATCH --time=48:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL            # Send email on job start, end, and fail

conda init
source ~/.bashrc

conda activate ES_env_311

python train_gpt2_v1_wt.py --config ./config/gpt2_v1_wt/3M.yaml