#!/bin/bash

#SBATCH --job-name=my_project       # Name of the job
#SBATCH --time=05:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL            # Send email on job start, end, and fail

conda init
source ~/.bashrc

conda activate ES_env_311

python eval_full_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/957M/957M_0.05/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/957M/957M_0.05/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 64 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto' 

python eval_full_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/957M/957M_0.1/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/957M/957M_0.1/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 64 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto' 

python eval_full_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/957M/957M_0.15/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/957M/957M_0.15/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 64 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto' 

python eval_full_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/957M/957M_0.3/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/957M/957M_0.3/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 64 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto' 

python eval_full_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/957M/957M_0.5/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/957M/957M_0.5/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 64 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto' 

python eval_full_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/957M/957M_0.8/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/957M/957M_0.8/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 64 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto' 

python eval_full_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/957M/957M_1.0/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/957M/957M_1.0/full_eval_results \
   --total_seqs 1000000 \
   --batch_size 64 \
   --max_seq_len 512 \
   --summary_only \
   --device 'auto' 

