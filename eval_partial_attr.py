import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import re
from torch.nn.utils.rnn import pad_sequence
from terrarium.models import get_model, load_model
from terrarium.dataloaders import get_dataloader

# example:
# python eval_partial_attr.py \
#   --model_name 'cat_v1' \
#   --ckpt_path ./out/cat_v1/55M_W0_1/ckpt.pt \
#   --data_dir ./data/v_32768 \
#   --dl_name 'cat_inference' \
#   --data_prefix 'val' \
#   --out_dir ./out/reviews/55M_W0_1/partial_eval_results \
#   --total_seqs 128 \
#   --batch_size 64 \
#   --max_seq_len 512 \
#   --seed_lengths 5 10 15 20 \
#   --summary_only True \
#   --device 'auto' 

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="Model name")
parser.add_argument("--ckpt_path", required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--data_dir", required=True, help="Path to data directory (must contain tok.pkl)")
parser.add_argument("--dl_name", required=True, help="Dataloader name")
parser.add_argument("--data_prefix", type=str, default="val")
parser.add_argument("--out_dir", required=True, help="Directory to write evaluation outputs")
parser.add_argument("--total_seqs", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_seq_len", type=int, default=513)
parser.add_argument("--seed_lengths", nargs="+", type=int, default=[5,10,15,20])
parser.add_argument("--summary_only", action="store_true", help="If set, only output summary stats.")
parser.add_argument("--device", default="cpu")


args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

print(f"Evaluating model: {args.ckpt_path} Data directory: {args.data_dir} Output directory: {args.out_dir}")

# ----- set up task -------- #
model_name      = args.model_name
ckpt_path       = args.ckpt_path
data_dir        = args.data_dir
dl_name         = args.dl_name
data_prefix     = args.data_prefix
out_dir         = args.out_dir
total_seqs      = args.total_seqs
batch_size      = args.batch_size
max_seq_len     = args.max_seq_len
seed_lengths    = args.seed_lengths
summary_only    = args.summary_only
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

# always cpu first to prevent gpu memory leak on load
model = load_model(model_name,ckpt_path,device_override='cpu').to(device)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False) # always first to cpu

# we do not need gradients for this
model.eval()

# with cuda and not on windows, compile the model for speed
if device == 'cuda' and os.name != 'nt':
    model = torch.compile(model)

with open(f"{data_dir}/tok.pkl",'rb') as f:
    tok = pickle.load(f)
sos_idx = tok.encode('<|sos|>')
dl = get_dataloader(dl_name,batch_size,max_seq_len,f"{data_dir}/{data_prefix}_token_ids.bin",f"{data_dir}/{data_prefix}_attr_ids.bin",sos_idx,device)

# check if dataset has enough sequences
if len(dl.split_locs)-1 < total_seqs:
    total_seqs = len(dl.split_locs)-1
    print(f'Warning: requested total_seqs reduced to {total_seqs} due to data size.')
n_batches = total_seqs // batch_size

# create the holder for results
results = {
    s: {
        'true': [],
        'sampled_attr': [],
        'top1_attr': [],
        'cross_entropy_loss_attr': []
    }
    for s in seed_lengths
}

# --- run the inference over batches --- #
for b in tqdm(range(n_batches)):
    x,y,attr = dl.next_batch()
    attrs = torch.tensor([attri[0].item() for attri in attr]).to(device)

    # pad and assemble x and y
    x_pad = torch.zeros((len(x), max_seq_len), dtype=torch.long)
    y_pad = torch.zeros((len(y), max_seq_len), dtype=torch.long)
    seq_lengths = torch.zeros((len(x),), dtype=torch.long)
    for i, (xi, yi) in enumerate(zip(x, y)):
        L = len(xi)
        seq_lengths[i] = L
        x_pad[i, :L] = xi
        y_pad[i, :L] = yi
        
    x_pad,y_pad,seq_lengths = x_pad.to(device),y_pad.to(device),seq_lengths.to(device)
    
    # forward pass
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        _,attr_logits,_,_ = model.forward(x_pad,targets=y_pad) # (B,T,A)
    B,T,A = attr_logits.shape
    cond_prob_obs = torch.softmax(attr_logits,dim=-1)

    n_minus_1_idx = (seq_lengths - 2).clamp_min(0)  # B,
    sampled_attr = torch.multinomial(cond_prob_obs.view(-1,A),num_samples=1).view(B,T) # B,T
    top1_attr = torch.argmax(cond_prob_obs,dim=-1) # B,T
    nll_attr = -torch.log(cond_prob_obs.clamp_min(1e-9)) # B,T

    for s in seed_lengths:
        rows_to_use = (seq_lengths > s).nonzero(as_tuple=True)[0]
        if rows_to_use.numel() == 0:
            continue
        results[s]['true'].append(attrs[rows_to_use].tolist())
        results[s]['sampled_attr'].append(sampled_attr[rows_to_use, s-2].tolist())
        results[s]['top1_attr'].append(top1_attr[rows_to_use, s-2].tolist())
        results[s]['cross_entropy_loss_attr'].append(nll_attr[rows_to_use, s-2, attrs[rows_to_use]].tolist())


# ---- data post processing ---- #
for s in seed_lengths:
    results[s]['true'] = [item for sublist in results[s]['true'] for item in sublist]
    results[s]['sampled_attr'] = [item for sublist in results[s]['sampled_attr'] for item in sublist]
    results[s]['top1_attr'] = [item for sublist in results[s]['top1_attr'] for item in sublist]
    results[s]['cross_entropy_loss_attr'] = [item for sublist in results[s]['cross_entropy_loss_attr'] for item in sublist]


# now to df for csv output, making seed length a col so can contatenate all into one file
all_dfs = []
for s in seed_lengths:
    df = pd.DataFrame({
        'seed_length': [s]*len(results[s]['true']),
        'true': results[s]['true'],
        'sampled_attr': results[s]['sampled_attr'],
        'top1_attr': results[s]['top1_attr'],
        'cross_entropy_loss_attr': results[s]['cross_entropy_loss_attr']
    })
    df['seed_length'] = df['seed_length'].astype(int)
    df['true'] = df['true'].astype(int)
    all_dfs.append(df)

# concatenate all dataframes
final_df = pd.concat(all_dfs, ignore_index=True)

if not summary_only:
    final_df.to_csv(os.path.join(out_dir,f'online_classification_results_{total_seqs}.csv'), index=False)

# overall summary by seed length
df_overall_summary = final_df.groupby('seed_length', as_index=False).agg(
    n_attrs=('true', 'count'),
    sampled_accuracy=('sampled_attr', lambda x: np.mean(x == final_df.loc[x.index, 'true'])),
    top1_accuracy=('top1_attr', lambda x: np.mean(x == final_df.loc[x.index, 'true'])),
    cross_entropy_mean=('cross_entropy_loss_attr', 'mean')
)

# summary by seed and true attr
df_stratified_summary = (
    final_df
    .groupby(['seed_length', 'true'], as_index=False)
    .agg(
        n_attrs=('true', 'count'),
        sampled_accuracy=('sampled_attr', lambda x: np.mean(x == final_df.loc[x.index, 'true'])),
        top1_accuracy=('top1_attr', lambda x: np.mean(x == final_df.loc[x.index, 'true'])),
        cross_entropy_mean=('cross_entropy_loss_attr', 'mean')
    )
    .reset_index()  # make seed_length, true into columns again
)

df_overall_summary.to_csv(os.path.join(out_dir,f'online_classification_overall_summary_{total_seqs}.csv'),index=False)
df_stratified_summary.to_csv(os.path.join(out_dir,f'online_classification_stratified_summary_{total_seqs}.csv'),index=False)