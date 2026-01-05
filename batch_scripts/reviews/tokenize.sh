#!/bin/bash

#SBATCH --job-name=my_project       # Name of the job
#SBATCH --time=48:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512G
#SBATCH --mail-type=ALL 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

python -u /home/els95/project_pi_ajl89/els95/targetGuidedTransformer/data/reviews/download_reviews.py
python -u /home/els95/project_pi_ajl89/els95/targetGuidedTransformer/data/reviews/train_tokenizer.py
python -u /home/els95/project_pi_ajl89/els95/targetGuidedTransformer/data/reviews/compile_reviews.py # encode
python -u /home/els95/project_pi_ajl89/els95/targetGuidedTransformer/data/reviews/text_decode.py # verification