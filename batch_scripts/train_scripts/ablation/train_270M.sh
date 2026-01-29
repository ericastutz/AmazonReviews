#!/bin/bash

#SBATCH --job-name=my_project       # Name of the job
#SBATCH --time=24:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL            # Send email on job start, end, and fail

conda init
source ~/.bashrc

conda activate ES_env_311

python train_cat_ablation.py  --config ./config/cat_ablation/270M/270M_0.05.yaml
python train_cat_ablation.py  --config ./config/cat_ablation/270M/270M_0.1.yaml
python train_cat_ablation.py  --config ./config/cat_ablation/270M/270M_0.15.yaml
python train_cat_ablation.py  --config ./config/cat_ablation/270M/270M_0.3.yaml
python train_cat_ablation.py  --config ./config/cat_ablation/270M/270M_0.5.yaml
python train_cat_ablation.py  --config ./config/cat_ablation/270M/270M_0.8.yaml
python train_cat_ablation.py  --config ./config/cat_ablation/270M/270M_1.0.yaml

