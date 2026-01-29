import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
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
#   --total_sequences 128 \
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
parser.add_argument("--total_sequences", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_seq_len", type=int, default=513)
parser.add_argument("--seed_lengths", nargs="+", type=int, default=[5,10,15,20])
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
dl = get_dataloader(dl_name,batch_size,max_seq_len,f"{data_dir}/{data_prefix}_token_ids.bin",f"{data_dir}/{data_prefix}_attr_ids.bin",device)

# check if dataset has enough sequences
if len(dl.split_locs)-1 < total_sequences:
    total_sequences = len(dl.split_locs)-1
    print(f'Warning: requested total_sequences reduced to {total_sequences} due to data size.')
n_batches = total_sequences // batch_size

# create the holder for results
results = {
    s: {
        'true_attr': [],
        'sampled_outcome': [],
        'top1_outcome': [],
        'cross_entropy_loss': []
    }
    for s in seed_lengths
}

# --- run the inference over batches --- #
for b in tqdm(range(n_batches)):
    x,y,attr = dl.next_batch()
    attrs = torch.tensor([attri[0].item() for attri in attr])

    # assemble x and y from the return sequences t.
    x_pad = torch.zeros((len(x), max_seq_len), dtype=torch.long)
    y_pad = torch.zeros((len(y), max_seq_len), dtype=torch.long)
    seq_lengths = np.zeros((len(x),),dtype=np.int32)
    for i, (xi, yi) in enumerate(zip(x, y)):
        L = len(xi)
        seq_lengths[i] = L
        x_pad[i, :L] = xi
        y_pad[i, :L] = yi
        
    x_pad,y_pad = x_pad.to(device),y_pad.to(device)
    
    # forward pass
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits,outcome_logits,_,_ = model.forward(x_pad,targets=y_pad) # (B,T,V) and (B,T,O)
    B,T,O = outcome_logits.shape
    token_probs = torch.softmax(logits,dim=-1)
    cond_prob_obs = torch.softmax(outcome_logits,dim=-1)

    # for the observed next token, get cond prob, sampled outcome, top1 outcome, nll
    sampled_outcome = torch.multinomial(cond_prob_obs.view(-1,O),num_samples=1).view(B,T) # B,T
    top1_outcome = torch.argmax(cond_prob_obs,dim=-1) # B,T
    nll = -torch.log(torch.clamp(cond_prob_obs,1e-9,1.0)) # B,T

    for s in seed_lengths:
        rows_to_use = np.where(seq_lengths > s)[0]
        if len(rows_to_use) == 0:
            continue
        results[s]['true_attr'].append(attrs[rows_to_use].tolist())
        results[s]['sampled_outcome'].append(sampled_outcome[rows_to_use, s-2].tolist())
        results[s]['top1_outcome'].append(top1_outcome[rows_to_use, s-2].tolist())
        results[s]['cross_entropy_loss'].append(nll[rows_to_use, s-2, attrs[rows_to_use]].tolist())


# ---- data post processing ---- #
for s in seed_lengths:
    results[s]['true_attr'] = [item for sublist in results[s]['true_attr'] for item in sublist]
    results[s]['sampled_outcome'] = [item for sublist in results[s]['sampled_outcome'] for item in sublist]
    results[s]['top1_outcome'] = [item for sublist in results[s]['top1_outcome'] for item in sublist]
    results[s]['cross_entropy_loss'] = [item for sublist in results[s]['cross_entropy_loss'] for item in sublist]

# now to df for csv output, making seed length a col so can contatenate all into one file
all_dfs = []
for s in seed_lengths:
    df = pd.DataFrame({
        'seed_length': [s]*len(results[s]['true_attr']),
        'true_attr': results[s]['true_attr'],
        'sampled_outcome': results[s]['sampled_outcome'],
        'top1_outcome': results[s]['top1_outcome'],
        'cross_entropy_loss': results[s]['cross_entropy_loss']
    })
    all_dfs.append(df)

# concatenate all dataframes
final_df = pd.concat(all_dfs, ignore_index=True)

if not summary_only:
    final_df.to_csv(os.path.join(out_dir,'online_classification_results.csv'), index=False)

# overall summary by seed length
df_overall_summary = final_df.groupby('seed_length').agg(
    n_attrs=('true_attr', 'count'),
    sampled_accuracy=('sampled_outcome', lambda x: np.mean(x == final_df.loc[x.index, 'true_attr'])),
    top1_accuracy=('top1_outcome', lambda x: np.mean(x == final_df.loc[x.index, 'true_attr'])),
    cross_entropy_mean=('cross_entropy_loss', 'mean')
)

# summary by seed and true attr
df_stratified_summary = (
    final_df
    .groupby(['seed_length', 'true_attr'])
    .agg(
        n_attrs=('true_attr', 'count'),
        sampled_accuracy=('sampled_outcome', lambda x: np.mean(x == final_df.loc[x.index, 'true_attr'])),
        top1_accuracy=('top1_outcome', lambda x: np.mean(x == final_df.loc[x.index, 'true_attr'])),
        cross_entropy_mean=('cross_entropy_loss', 'mean')
    )
    .reset_index()  # make seed_length, true_attr into columns again
)

df_overall_summary.to_csv(os.path.join(out_dir,'online_classification_overall_summary.csv'),index=False)
df_stratified_summary.to_csv(os.path.join(out_dir,'online_classification_stratified_summary.csv'),index=False)