#!/bin/bash

#SBATCH --job-name=my_project       # Name of the job
#SBATCH --time=00:30:00             # Maximum runtime (HH:MM:SS)
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL            # Send email on job start, end, and fail

conda init
source ~/.bashrc

conda activate ES_env_311

python eval_full_standard.py \
   --model_name 'gpt2_v1_wt' \
   --ckpt_path ./out/gpt2_v1_wt/3M/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'standard_inference' \
   --data_prefix 'val' \
   --out_dir ./out/gpt2_v1_wt/3M/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 512 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto'  


