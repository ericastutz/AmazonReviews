import argparse
import torch
import pickle
import os
import yaml
import pandas as pd
from terrarium.models import get_model, load_model
from terrarium.dataloaders import get_dataloader

# example:
# python eval/eval_loss_standard.py \
#   --model ./out/reviews/55M_W0_1/ckpt.pt \
#   --data_dir ./data/reviews_dynamic \
#   --out_dir ./out/reviews/55M_W0_1/loss_eval_results \
#   --num_iters 2000 \
#   --device auto \

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', required=True, help='Path to model checkpoint (.pt)')
parser.add_argument('--config_path', required=True, help='Path to model config (.yaml)')
parser.add_argument('--data_dir', required=True, help='Path to data directory (must contain tok.pkl)')
parser.add_argument('--out_dir', required=True, help='Directory to write evaluation outputs')
parser.add_argument('--num_iters', type=int, default=2000)
parser.add_argument('--device', default='cpu')

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

print(f'Evaluating model: {args.ckpt_path} Data directory: {args.data_dir} Output directory: {args.out_dir}')

# ----- set up task -------- #
ckpt_path       = args.ckpt_path
config_path     = args.config_path
data_dir        = args.data_dir
out_dir         = args.out_dir
num_iters       = args.num_iters
device          = args.device

# priority 'cuda' > 'mps' > 'cpu'
if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

print(f'Using device: {device}')

# load config
with open(config_path, 'r') as file:
    config_yaml = yaml.safe_load(file)
model_cfg = config_yaml['model']
model_name = model_cfg['name']
dl_name = config_yaml['dataloader']
batch_size = model_cfg['batch_size']
max_seq_len = model_cfg['block_size']

# override eval_iters if provided
if args.num_iters is not None:
    model_cfg['eval_iters'] = args.num_iters
num_iters = model_cfg['eval_iters']

# always cpu first to prevent gpu memory leak on load
model = load_model(model_name,ckpt_path,device_override='cpu').to(device)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False) # always first to cpu

# with cuda and not on windows, compile the model for speed
if device == 'cuda' and os.name != 'nt':
    model = torch.compile(model)

# load data
dl_train = get_dataloader(dl_name,batch_size,max_seq_len,f'{data_dir}/train_token_ids.bin',device)
dl_val = get_dataloader(dl_name,batch_size,max_seq_len,f'{data_dir}/val_token_ids.bin',device)

# run model for num_iters
est_loss = model.estimate_loss_light(dl_train, dl_val)

if est_loss:
    results = {
        "val_total_loss": float(est_loss["val"]["total"]),
        "val_total_loss_var": float(est_loss["val"].get("total_var", float("nan"))),
    }
    df = pd.DataFrame([results])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loss_eval.csv")
    df.to_csv(csv_path, index=False)
else:
    print("failed model run")
