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

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"

python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1/7M/ckpt.pt \
    --config_path ./config/gpt2_v1/7M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1/7M/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1/9M/ckpt.pt \
    --config_path ./config/gpt2_v1/9M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1/9M/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1/20M/ckpt.pt \
    --config_path ./config/gpt2_v1/20M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1/20M/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1/72M/ckpt.pt \
    --config_path ./config/gpt2_v1/72M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1/72M/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1/270M/ckpt.pt \
    --config_path ./config/gpt2_v1/270M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1/270M/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''
python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1/1B/ckpt.pt \
    --config_path ./config/gpt2_v1/1B.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1/1B/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''
python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1_wt/3M/ckpt.pt \
    --config_path ./config/gpt2_v1_wt/3M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1_wt/3M/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''
python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1_wt/5M/ckpt.pt \
    --config_path ./config/gpt2_v1_wt/5M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1_wt/5M/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''
python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1_wt/12M/ckpt.pt \
    --config_path ./config/gpt2_v1_wt/12M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1_wt/12M/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1_wt/55M/ckpt.pt \
    --config_path ./config/gpt2_v1_wt/55M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1_wt/55M/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1_wt/235M/ckpt.pt \
    --config_path ./config/gpt2_v1_wt/235M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1_wt/235M/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''
python ./eval/eval_loss_standard.py \
    --ckpt_path ./out/gpt2_v1_wt/957M/ckpt.pt \
    --config_path ./config/gpt2_v1_wt/957M.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/gpt2_v1_wt/957M/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''