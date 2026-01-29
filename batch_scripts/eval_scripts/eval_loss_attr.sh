#!/bin/bash

#SBATCH --job-name=my_project       # Name of the job
#SBATCH --time=00:10:00             # Maximum runtime (HH:MM:SS)
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
'''
#------7M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/7M/7M_0.05/ckpt.pt \
    --config_path ./config/cat_v1/7M/7M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/7M/7M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/7M/7M_0.1/ckpt.pt \
    --config_path ./config/cat_v1/7M/7M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/7M/7M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/7M/7M_0.15/ckpt.pt \
    --config_path ./config/cat_v1/7M/7M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/7M/7M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/7M/7M_0.3/ckpt.pt \
    --config_path ./config/cat_v1/7M/7M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/7M/7M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/7M/7M_0.5/ckpt.pt \
    --config_path ./config/cat_v1/7M/7M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/7M/7M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/7M/7M_0.8/ckpt.pt \
    --config_path ./config/cat_v1/7M/7M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/7M/7M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/7M/7M_1.0/ckpt.pt \
    --config_path ./config/cat_v1/7M/7M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/7M/7M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------9M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/9M/9M_0.05/ckpt.pt \
    --config_path ./config/cat_v1/9M/9M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/9M/9M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/9M/9M_0.1/ckpt.pt \
    --config_path ./config/cat_v1/9M/9M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/9M/9M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/9M/9M_0.15/ckpt.pt \
    --config_path ./config/cat_v1/9M/9M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/9M/9M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/9M/9M_0.3/ckpt.pt \
    --config_path ./config/cat_v1/9M/9M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/9M/9M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/9M/9M_0.5/ckpt.pt \
    --config_path ./config/cat_v1/9M/9M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/9M/9M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/9M/9M_0.8/ckpt.pt \
    --config_path ./config/cat_v1/9M/9M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/9M/9M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/9M/9M_1.0/ckpt.pt \
    --config_path ./config/cat_v1/9M/9M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/9M/9M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------20M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/20M/20M_0.05/ckpt.pt \
    --config_path ./config/cat_v1/20M/20M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/20M/20M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/20M/20M_0.1/ckpt.pt \
    --config_path ./config/cat_v1/20M/20M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/20M/20M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/20M/20M_0.15/ckpt.pt \
    --config_path ./config/cat_v1/20M/20M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/20M/20M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/20M/20M_0.3/ckpt.pt \
    --config_path ./config/cat_v1/20M/20M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/20M/20M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/20M/20M_0.5/ckpt.pt \
    --config_path ./config/cat_v1/20M/20M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/20M/20M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/20M/20M_0.8/ckpt.pt \
    --config_path ./config/cat_v1/20M/20M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/20M/20M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/20M/20M_1.0/ckpt.pt \
    --config_path ./config/cat_v1/20M/20M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/20M/20M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------72M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/72M/72M_0.05/ckpt.pt \
    --config_path ./config/cat_v1/72M/72M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/72M/72M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/72M/72M_0.1/ckpt.pt \
    --config_path ./config/cat_v1/72M/72M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/72M/72M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/72M/72M_0.15/ckpt.pt \
    --config_path ./config/cat_v1/72M/72M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/72M/72M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/72M/72M_0.3/ckpt.pt \
    --config_path ./config/cat_v1/72M/72M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/72M/72M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/72M/72M_0.5/ckpt.pt \
    --config_path ./config/cat_v1/72M/72M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/72M/72M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/72M/72M_0.8/ckpt.pt \
    --config_path ./config/cat_v1/72M/72M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/72M/72M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/72M/72M_1.0/ckpt.pt \
    --config_path ./config/cat_v1/72M/72M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/72M/72M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------270M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/270M/270M_0.05/ckpt.pt \
    --config_path ./config/cat_v1/270M/270M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/270M/270M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/270M/270M_0.1/ckpt.pt \
    --config_path ./config/cat_v1/270M/270M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/270M/270M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/270M/270M_0.15/ckpt.pt \
    --config_path ./config/cat_v1/270M/270M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/270M/270M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/270M/270M_0.3/ckpt.pt \
    --config_path ./config/cat_v1/270M/270M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/270M/270M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/270M/270M_0.5/ckpt.pt \
    --config_path ./config/cat_v1/270M/270M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/270M/270M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/270M/270M_0.8/ckpt.pt \
    --config_path ./config/cat_v1/270M/270M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/270M/270M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1/270M/270M_1.0/ckpt.pt \
    --config_path ./config/cat_v1/270M/270M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1/270M/270M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------3M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/3M/3M_0.05/ckpt.pt \
    --config_path ./config/cat_v1_wt/3M/3M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/3M/3M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/3M/3M_0.1/ckpt.pt \
    --config_path ./config/cat_v1_wt/3M/3M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/3M/3M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/3M/3M_0.15/ckpt.pt \
    --config_path ./config/cat_v1_wt/3M/3M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/3M/3M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/3M/3M_0.3/ckpt.pt \
    --config_path ./config/cat_v1_wt/3M/3M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/3M/3M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/3M/3M_0.5/ckpt.pt \
    --config_path ./config/cat_v1_wt/3M/3M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/3M/3M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/3M/3M_0.8/ckpt.pt \
    --config_path ./config/cat_v1_wt/3M/3M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/3M/3M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/3M/3M_1.0/ckpt.pt \
    --config_path ./config/cat_v1_wt/3M/3M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/3M/3M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------5M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/5M/5M_0.05/ckpt.pt \
    --config_path ./config/cat_v1_wt/5M/5M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/5M/5M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/5M/5M_0.1/ckpt.pt \
    --config_path ./config/cat_v1_wt/5M/5M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/5M/5M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/5M/5M_0.15/ckpt.pt \
    --config_path ./config/cat_v1_wt/5M/5M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/5M/5M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/5M/5M_0.3/ckpt.pt \
    --config_path ./config/cat_v1_wt/5M/5M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/5M/5M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/5M/5M_0.5/ckpt.pt \
    --config_path ./config/cat_v1_wt/5M/5M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/5M/5M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/5M/5M_0.8/ckpt.pt \
    --config_path ./config/cat_v1_wt/5M/5M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/5M/5M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/5M/5M_1.0/ckpt.pt \
    --config_path ./config/cat_v1_wt/5M/5M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/5M/5M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------12M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/12M/12M_0.05/ckpt.pt \
    --config_path ./config/cat_v1_wt/12M/12M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/12M/12M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/12M/12M_0.1/ckpt.pt \
    --config_path ./config/cat_v1_wt/12M/12M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/12M/12M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/12M/12M_0.15/ckpt.pt \
    --config_path ./config/cat_v1_wt/12M/12M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/12M/12M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/12M/12M_0.3/ckpt.pt \
    --config_path ./config/cat_v1_wt/12M/12M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/12M/12M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/12M/12M_0.5/ckpt.pt \
    --config_path ./config/cat_v1_wt/12M/12M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/12M/12M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/12M/12M_0.8/ckpt.pt \
    --config_path ./config/cat_v1_wt/12M/12M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/12M/12M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/12M/12M_1.0/ckpt.pt \
    --config_path ./config/cat_v1_wt/12M/12M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/12M/12M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------55M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/55M/55M_0.05/ckpt.pt \
    --config_path ./config/cat_v1_wt/55M/55M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/55M/55M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/55M/55M_0.1/ckpt.pt \
    --config_path ./config/cat_v1_wt/55M/55M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/55M/55M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/55M/55M_0.15/ckpt.pt \
    --config_path ./config/cat_v1_wt/55M/55M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/55M/55M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/55M/55M_0.3/ckpt.pt \
    --config_path ./config/cat_v1_wt/55M/55M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/55M/55M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/55M/55M_0.5/ckpt.pt \
    --config_path ./config/cat_v1_wt/55M/55M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/55M/55M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/55M/55M_0.8/ckpt.pt \
    --config_path ./config/cat_v1_wt/55M/55M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/55M/55M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/55M/55M_1.0/ckpt.pt \
    --config_path ./config/cat_v1_wt/55M/55M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/55M/55M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------235M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/235M/235M_0.05/ckpt.pt \
    --config_path ./config/cat_v1_wt/235M/235M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/235M/235M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/235M/235M_0.1/ckpt.pt \
    --config_path ./config/cat_v1_wt/235M/235M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/235M/235M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/235M/235M_0.15/ckpt.pt \
    --config_path ./config/cat_v1_wt/235M/235M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/235M/235M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/235M/235M_0.3/ckpt.pt \
    --config_path ./config/cat_v1_wt/235M/235M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/235M/235M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/235M/235M_0.5/ckpt.pt \
    --config_path ./config/cat_v1_wt/235M/235M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/235M/235M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/235M/235M_0.8/ckpt.pt \
    --config_path ./config/cat_v1_wt/235M/235M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/235M/235M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/235M/235M_1.0/ckpt.pt \
    --config_path ./config/cat_v1_wt/235M/235M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/235M/235M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto

#------957M-------#
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/957M/957M_0.05/ckpt.pt \
    --config_path ./config/cat_v1_wt/957M/957M_0.05.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/957M/957M_0.05/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/957M/957M_0.1/ckpt.pt \
    --config_path ./config/cat_v1_wt/957M/957M_0.1.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/957M/957M_0.1/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/957M/957M_0.15/ckpt.pt \
    --config_path ./config/cat_v1_wt/957M/957M_0.15.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/957M/957M_0.15/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/957M/957M_0.3/ckpt.pt \
    --config_path ./config/cat_v1_wt/957M/957M_0.3.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/957M/957M_0.3/loss_eval_results \
    --num_iters 2000 \
    --device auto
'''
python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/957M/957M_0.5/ckpt.pt \
    --config_path ./config/cat_v1_wt/957M/957M_0.5.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/957M/957M_0.5/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/957M/957M_0.8/ckpt.pt \
    --config_path ./config/cat_v1_wt/957M/957M_0.8.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/957M/957M_0.8/loss_eval_results \
    --num_iters 2000 \
    --device auto

python ./eval/eval_loss_attr.py \
    --ckpt_path ./out/cat_v1_wt/957M/957M_1.0/ckpt.pt \
    --config_path ./config/cat_v1_wt/957M/957M_1.0.yaml \
    --data_dir ./data/v_32768 \
    --out_dir ./out/cat_v1_wt/957M/957M_1.0/loss_eval_results \
    --num_iters 2000 \
    --device auto
