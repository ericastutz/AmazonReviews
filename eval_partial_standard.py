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
# python eval_partial_standard.py \
#   --model_name 'gpt2_v1' \
#   --ckpt_path ./out/gpt2_v1/55M/ckpt.pt \
#   --data_dir ./data/v_32768 \
#   --dl_name 'standard_inference' \
#   --data_prefix 'val' \
#   --out_dir ./out/55M/partial_eval_results \
#   --total_seqs 128 \
#   --batch_size 64 \
#   --max_seq_len 512 \
#   --num_samples 512 \
#   --targets 261 262 263 264 265 266
#   --seed_lengths 5 10 15 20 \
#   --search_id 257
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
parser.add_argument("--search_id", nargs="*", type=int, default=None,help="Optional list of token ids to search for. If omitted, defaults to --targets.")
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
num_samples     = args.num_samples
targets         = args.targets
seed_lengths    = args.seed_lengths
search_id       = args.search_id
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
targets = torch.tensor(targets, device=device, dtype=torch.long)
K = targets.numel()
if args.search_id is None or len(args.search_id) == 0:
    search_id = targets
else:
    search_id = torch.tensor(args.search_id, device=device, dtype=torch.long)
dl = get_dataloader(dl_name,batch_size,max_seq_len,f"{data_dir}/{data_prefix}_token_ids.bin",sos_idx,device)

# check if dataset has enough sequences
if len(dl.split_locs)-1 < total_seqs:
    total_seqs = len(dl.split_locs)-1
    print(f'Warning: requested total_seqs reduced to {total_seqs} due to data size.')
n_batches = total_seqs // batch_size

# create the holder for results
results = {
    s: {
        "mc": [
            {
                "true_token": [],
                "gen_token": []
            }
            for _ in range(num_samples)
        ],
        "counts": [],
        "sampled_token": [],
        "top1_token": [],
        "cross_entropy_loss_token": []

    }
    for s in seed_lengths
}

# --- run the inference over batches --- #
for b in tqdm(range(total_seqs)):
    x,y = dl.mc_next_batch(B_override=num_samples)
    x,y = torch.stack(x, dim=0), torch.stack(y, dim=0)
    seq_lengths = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

    B,T = x.shape
    true_token = y[:, -1]

    for s in seed_lengths:
        if s-1 <= 0 or s-1 >= T:
            continue
        x_s = x[:,:s-1]
        y_s = y[:,:s-1]

        output = model._generate(x_s,max_seq_len-x_s.shape[1]) 
        T_out = output.size(1)

        # Find first occurrence of any token in search_id
        mask = torch.isin(output, search_id)
        found = mask.any(dim=1)
        first_occ = mask.float().argmax(dim=1)
        first_occ = first_occ + 1
        valid = found & (first_occ < T_out)
        gather_pos = first_occ.clamp(min=0, max=T_out-1)
        gen_token = output.gather(1, gather_pos.unsqueeze(1)).squeeze(1)
        gen_token[~valid] = -1 

        for n in range(x_s.size(0)):
            results[s]["mc"][n]["true_token"].append(true_token.tolist()[n])
            results[s]["mc"][n]["gen_token"].append(gen_token.tolist()[n])
    
        # perform bin counts
        valid_tokens = gen_token[valid]
        if valid_tokens.numel() > 0:
            counts = (valid_tokens.unsqueeze(1) == targets.unsqueeze(0)).sum(dim=0)  # (K,)
        else:
            counts = torch.zeros(K, device=x.device, dtype=torch.long)

        total = counts.sum()
        if total > 0:
            token_probs = counts.float() / total
        sampled_token = int(targets[torch.multinomial(token_probs, num_samples=1).item()].item())
        top1_token = int(targets[int(token_probs.argmax().item())].item())

        true_token_s = (true_token.unsqueeze(1) == targets.unsqueeze(0)).float().argmax(dim=1)   
        nll_token = -torch.log(torch.gather(token_probs.unsqueeze(0).expand(true_token.size(0), -1), 1, true_token_s.unsqueeze(1)).clamp_min(1e-9)).squeeze(1)

        results[s]["counts"].append(counts.detach().cpu().tolist())
        results[s]["sampled_token"].append(sampled_token)
        results[s]["top1_token"].append(top1_token)
        results[s]["cross_entropy_loss_token"].append(nll_token.detach().cpu().numpy())


# ---- data post processing ---- #
rows = []
for s, d in results.items():
    for n, rec in enumerate(d["mc"]):
        L = len(rec["gen_token"])
        for t in range(L):
            rows.append({
                "seed_len": s,
                "sample_id": n,
                "sample": t,
                "true_token": rec["true_token"][t],
                "gen_token": rec["gen_token"][t],
            })

final_df = pd.DataFrame(rows)

if not summary_only:
        final_df.to_csv(os.path.join(out_dir,f'online_classification_results_{total_seqs}.csv'), index=False)

flat = {
    "seed_len": [],
    "sample": [],
    "sampled_token": [],
    "top1_token": [],
    "cross_entropy_loss_token": []
}

# If you want counts as columns too:
targets_list = targets.detach().cpu().tolist()
for tok in targets_list:
    flat[f"count_{tok}"] = []

for s, d in results.items():
    nb = len(d["top1_token"])
    for i in range(nb):
        flat["seed_len"].append(s)
        flat["sample"].append(i)
        flat["sampled_token"].append(d["sampled_token"][i])
        flat["top1_token"].append(d["top1_token"][i])

        nll_list = d["cross_entropy_loss_token"][i]
        flat["cross_entropy_loss_token"].append(float(np.nanmean(nll_list)))

        counts = d["counts"][i]
        for j, tok in enumerate(targets_list):
            flat[f"count_{tok}"].append(counts[j])

df_stratified_summary.to_csv(os.path.join(out_dir,f'online_classification_stratified_summary_{total_seqs}.csv'),index=False)