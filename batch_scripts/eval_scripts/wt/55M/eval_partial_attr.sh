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

python eval_partial_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/55M/55M_0.05/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/55M/55M_0.05/partial_eval_results \
   --total_seqs 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --seed_lengths 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 \
   --summary_only \
   --device 'auto' 

python eval_partial_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/55M/55M_0.1/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/55M/55M_0.1/partial_eval_results \
   --total_seqs 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --seed_lengths 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 \
   --summary_only \
   --device 'auto' 

python eval_partial_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/55M/55M_0.15/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/55M/55M_0.15/partial_eval_results \
   --total_seqs 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --seed_lengths 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 \
   --summary_only \
   --device 'auto' 

python eval_partial_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/55M/55M_0.3/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/55M/55M_0.3/partial_eval_results \
   --total_seqs 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --seed_lengths 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 \
   --summary_only \
   --device 'auto' 

python eval_partial_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/55M/55M_0.5/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/55M/55M_0.5/partial_eval_results \
   --total_seqs 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --seed_lengths 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 \
   --summary_only \
   --device 'auto' 

python eval_partial_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/55M/55M_0.8/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/55M/55M_0.8/partial_eval_results \
   --total_seqs 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --seed_lengths 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 \
   --summary_only \
   --device 'auto' 

python eval_partial_attr.py \
   --model_name 'cat_v1_wt' \
   --ckpt_path ./out/cat_v1_wt/55M/55M_1.0/ckpt.pt \
   --data_dir ./data/v_32768 \
   --dl_name 'cat_inference' \
   --data_prefix 'val' \
   --out_dir ./out/cat_v1_wt/55M/55M_1.0/partial_eval_results \
   --total_seqs 1000000 \
   --batch_size 256 \
   --max_seq_len 512 \
   --seed_lengths 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 \
   --summary_only \
   --device 'auto' 