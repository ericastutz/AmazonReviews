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
# python eval_full_standard.py \
#   --model_name 'gpt2_v1' \
#   --ckpt_path ./out/gpt2_v1/55M/ckpt.pt \
#   --data_dir ./data/v_32768 \
#   --dl_name 'cat_inference' \
#   --data_prefix 'val' \
#   --out_dir ./out/55M/full_eval_results \
#   --total_seqs 128 \
#   --batch_size 64 \
#   --max_seq_len 512 \
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
dl = get_dataloader(dl_name,batch_size,max_seq_len,f"{data_dir}/{data_prefix}_token_ids.bin",sos_idx,device)

# check if dataset has enough sequences
if len(dl.split_locs)-1 < total_seqs:
    total_seqs = len(dl.split_locs)-1
    print(f'Warning: requested total_seqs reduced to {total_seqs} due to data size.')
n_batches = total_seqs // batch_size

# create the holder for results
results = {
        'true_token': [],
        'sampled_token': [],
        'top1_token': [],
        'cross_entropy_loss_token': []
}

# --- run the inference over batches --- #
for b in tqdm(range(n_batches)):
    x,y = dl.next_batch()
    
    # assemble x and y from the return sequences t.
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
        logits,_ = model.forward(x_pad,targets=y_pad) # (B,T,V)
    B,T,V = logits.shape
    rows_to_use = torch.arange(B, device=device)
    last_idx = (seq_lengths - 1).clamp_min(0)  # B,
    last_logits = logits[rows_to_use, last_idx, :]
    last_token_probs  = torch.softmax(last_logits, dim=-1)

    true_token = y_pad[rows_to_use, last_idx]
    sampled_token = torch.multinomial(last_token_probs, num_samples=1).squeeze(1)
    top1_token = torch.argmax(last_token_probs, dim=-1)
    nll_token = -torch.log(torch.gather(last_token_probs, 1, true_token.unsqueeze(1)).clamp_min(1e-9)).squeeze(1)

    results['true_token'].append(true_token.tolist())
    results['sampled_token'].append(sampled_token.tolist())
    results['top1_token'].append(top1_token.tolist())
    results['cross_entropy_loss_token'].append(nll_token.tolist())


# ---- data post processing ---- #
results['true_token'] = [item for sublist in results['true_token'] for item in sublist]
results['sampled_token'] = [item for sublist in results['sampled_token'] for item in sublist]
results['top1_token'] = [item for sublist in results['top1_token'] for item in sublist]
results['cross_entropy_loss_token'] = [item for sublist in results['cross_entropy_loss_token'] for item in sublist]

# now to df for csv output
final_df = pd.DataFrame({
    'true_token': results['true_token'],
    'sampled_token': results['sampled_token'],
    'top1_token': results['top1_token'],
    'cross_entropy_loss_token': results['cross_entropy_loss_token']
})

if not summary_only:
    final_df.to_csv(os.path.join(out_dir,f'online_classification_results_{total_seqs}.csv'), index=False)
    
# summary by true token
df_stratified_summary = (
    final_df
    .groupby('true_token')
    .agg(
        n=('true_token', 'count'),

        sampled_accuracy_token=('sampled_token',
            lambda x: np.mean(x.values == final_df.loc[x.index, 'true_token'].values)
        ),
        top1_accuracy_token=('top1_token',
            lambda x: np.mean(x.values == final_df.loc[x.index, 'true_token'].values)
        ),
        cross_entropy_mean_token=('cross_entropy_loss_token', 'mean'),
    )
    .reset_index()
)

df_stratified_summary.to_csv(os.path.join(out_dir,f'online_classification_stratified_summary_{total_seqs}.csv'),index=False)