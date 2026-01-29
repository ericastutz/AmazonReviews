import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
from terrarium.models import get_model, load_model
from terrarium.dataloaders import get_dataloader

# example:
# python eval_full_gpt2.py \
#   --model_name 'gpt2_v1' \
#   --ckpt_path ./out/gpt2_v1/55M/ckpt.pt \
#   --data_dir ./data/v_32768 \
#   --dl_name 'standard_inference' \
#   --data_prefix 'val' \
#   --out_dir ./out/reviews/55M/full_eval_results \
#   --total_sequences 128 \
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
parser.add_argument("--total_sequences", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_seq_len", type=int, default=513)
parser.add_argument("--summary_only",type=bool,default=True,help="If true, only output summary statistics")
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
total_sequences = args.total_sequences
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
dl = get_dataloader(dl_name,batch_size,max_seq_len,f"{data_dir}/{data_prefix}_token_ids.bin",device)

# check if dataset has enough sequences
if len(dl.split_locs)-1 < total_sequences:
    total_sequences = len(dl.split_locs)-1
    print(f'Warning: requested total_sequences reduced to {total_sequences} due to data size.')
n_batches = total_sequences // batch_size

# create the holder for results
results = {
        'true_token': [],
        'sampled_token': [],
        'top1_token': [],
        'cross_entropy_loss': []
}

# --- run the inference over batches --- #
for b in tqdm(range(n_batches)):
    x,y = dl.next_batch()
    tokens = torch.tensor([seq[-1].item() for seq in y], dtype=torch.long, device=device)

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
        logits,_ = model.forward(x_pad,targets=y_pad) # (B,T,V) and (B,T,O)
    B,T,V = logits.shape
    token_probs = torch.softmax(logits,dim=-1)

    # for the observed next token, get cond prob, sampled outcome, top1 outcome, nll
    sampled_token = torch.multinomial(token_probs.view(-1,V),num_samples=1).view(B,T) # B,T
    top1_token = torch.argmax(token_probs,dim=-1) # B,T
    nll = -torch.log(torch.gather(token_probs,2,y_pad.unsqueeze(-1)).clamp_min(1e-9)).squeeze(-1) # B,T

    rows = torch.arange(B, device=device)
    last_idx = (seq_lengths - 1).clamp_min(0)  # B,
    
    last_sampled = sampled_token[rows, last_idx] # (B,)
    last_top1 = top1_token[rows, last_idx] # (B,)
    last_ce = nll[rows, last_idx] # (B,)

    results['true_token'].append(tokens.tolist())
    results['sampled_token'].append(last_sampled.tolist())
    results['top1_token'].append(last_top1.tolist())
    results['cross_entropy_loss'].append(last_ce.tolist())

# ---- data post processing ---- #
results['true_token'] = [item for sublist in results['true_token'] for item in sublist]
results['sampled_token'] = [item for sublist in results['sampled_token'] for item in sublist]
results['top1_token'] = [item for sublist in results['top1_token'] for item in sublist]
results['cross_entropy_loss'] = [item for sublist in results['cross_entropy_loss'] for item in sublist]

# now to df for csv output
final_df = pd.DataFrame({
    'true_token': results['true_token'],
    'sampled_token': results['sampled_token'],
    'top1_token': results['top1_token'],
    'cross_entropy_loss': results['cross_entropy_loss']
})

if not summary_only:
    final_df.to_csv(os.path.join(out_dir,'online_classification_results.csv'), index=False)

# overall summary by seed length
df_overall_summary = final_df.groupby('seed_length').agg(
    n_attrs=('true_attr', 'count'),
    sampled_accuracy=('sampled_outcome', lambda x: np.mean(x == final_df.loc[x.index, 'true_attr'])),
    top1_accuracy=('top1_outcome', lambda x: np.mean(x == final_df.loc[x.index, 'true_attr'])),
    cross_entropy_mean=('cross_entropy_loss', 'mean')
)

# summary by true token
df_stratified_summary = (
    final_df
    .groupby(['true_token'])
    .agg(
        n_tokens=('true_token', 'count'),
        sampled_accuracy=('sampled_token', lambda x: np.mean(x == final_df.loc[x.index, 'true_token'])),
        top1_accuracy=('top1_token', lambda x: np.mean(x == final_df.loc[x.index, 'true_token'])),
        cross_entropy_mean=('cross_entropy_loss', 'mean')
    )
    .reset_index()  # make seed_length, true_token into columns again
)

df_stratified_summary.to_csv(os.path.join(out_dir,'online_classification_stratified_summary.csv'),index=False)