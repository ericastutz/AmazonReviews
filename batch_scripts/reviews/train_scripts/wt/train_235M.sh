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

python train_gpt2_v1_wt.py --config ./config/reviews/gpt2_v1_wt/235M.yaml
python train_cat_v1_wt.py  --config ./config/reviews/cat_v1_wt/235M/235M_0.05.yaml
python train_cat_v1_wt.py  --config ./config/reviews/cat_v1_wt/235M/235M_0.1.yaml
python train_cat_v1_wt.py  --config ./config/reviews/cat_v1_wt/235M/235M_0.15.yaml
python train_cat_v1_wt.py  --config ./config/reviews/cat_v1_wt/235M/235M_0.3.yaml
python train_cat_v1_wt.py  --config ./config/reviews/cat_v1_wt/235M/235M_0.5.yaml
python train_cat_v1_wt.py  --config ./config/reviews/cat_v1_wt/235M/235M_0.8.yaml
python train_cat_v1_wt.py  --config ./config/reviews/cat_v1_wt/235M/235M_1.0.yaml
