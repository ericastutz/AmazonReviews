#!/bin/bash

#SBATCH --job-name=my_project       # Name of the job
#SBATCH --time=01:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL            # Send email on job start, end, and fail

conda init
source ~/.bashrc

conda activate ES_env_311

python eval_full_attr.py \
   --model_name 'cat_ablation' \
   --ckpt_path ./out/cat_ablation/270M/270M_0.05/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_ablation/270M/270M_0.05/partial_eval_results \
   --total_sequences 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --summary_only True \
   --device 'auto' 


