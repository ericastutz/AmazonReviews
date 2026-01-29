import os, pickle, yaml, time, math, argparse
import torch
import numpy as np
from dataclasses import dataclass, asdict
from terrarium.models import get_model, load_model
from terrarium.dataloaders import get_dataloader
from utils.logger import Logger
import matplotlib.pyplot as plt
from IPython.display import display

@dataclass
class Config:
    name: str = "cat_v1"
    device: str = "cpu"
    vocab_size: int = 18
    block_size: int = 45
    n_heads: int = 2
    n_layer: int = 2
    n_embd: int = 24
    dropout: int = 0
    bias: int = 0
    total_batch_size: int = 450
    batch_size: int = 9
    max_iters: int = 1000
    lr_decay_iters: int = 950
    warmup_iters: int = 50
    eval_iters: int = 100
    eval_interval: int = 100
    clip_grad: bool = True
    learning_rate: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.05
    eps: float = 1e-8
    attr_size: int = 6
    attr_weight: float = 1.0


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_model(model_cfg, model_name, ckpt_path, out_dir, init_from, device):
    
    token_losses = []
    attr_losses = []

    if init_from == 'new':
        model = get_model(model_name, model_cfg).to(device)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_val_loss = float('inf')
        iter_global = 0

        sd = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

        _, _ = model.load_state_dict(sd, strict=False)

        # freeze trained model params
        for p in model.parameters():
            p.requires_grad = False

        # unfreeze attr params
        for p in model.attr_layer.parameters():
            p.requires_grad = True
        for p in model.attr_ln.parameters():
            p.requires_grad = True
        for p in model.attr_head.parameters():
            p.requires_grad = True
            
        optimizer = model.configure_optimizer()

    else:  # 'resume'
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')

        model = load_model(model_name,ckpt_path,device_override='cpu').to(device)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False) # always first to cpu
        
        # freeze trained model params
        for p in model.parameters():
            p.requires_grad = False

        # unfreeze attr params
        for p in model.attr_layer.parameters():
            p.requires_grad = True
        for p in model.attr_ln.parameters():
            p.requires_grad = True
        for p in model.attr_head.parameters():
            p.requires_grad = True
        
        optimizer = model.configure_optimizer()
        optimizer.load_state_dict(checkpoint['optimizer'])

        iter_global = checkpoint['iter_global']
        best_val_loss = checkpoint['best_val_loss']
        token_losses = checkpoint.get("token_losses", [])
        attr_losses = checkpoint.get("attr_losses", [])
        iters = checkpoint.get("iters", [])
    
    model = torch.compile(model)
    return model, optimizer, iter_global, best_val_loss, token_losses, attr_losses

def get_lr(it,config):
    # linear warm up followed by cos decay down to minimum (0.1*lr)
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return (config.learning_rate * 0.1)
    # actual cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return (config.learning_rate * 0.1) + coeff * (config.learning_rate - (config.learning_rate * 0.1))

def train(model,optimizer,tok,dl_train,dl_val,iter_global=0,best_val_loss=1000,token_losses=None,attr_losses=None,out_dir=None,stats=True,logging=True,generate_text=True):    

    logger = None
    if out_dir is not None:
        logger = Logger(out_dir,['iter','train_token_loss','train_val_loss','train_attr_loss','val_attr_loss','dt','tok_sec','epoch'])

    grad_accum_steps = model.config.total_batch_size // (model.config.batch_size*model.config.block_size)
    n_ne,n_e = model.get_num_params()
    if stats:
        print("\n----------------------\nTraining Run:\n----------------------")
        print(f"model size {model.get_model_size():.2f}MB")
        print(f"total parameters {(n_ne+n_e)/1e6:.2f}M | embedding {n_e/1e6:.2f}M | non-embedding {n_ne/1e6:.2f}M")
        print(f"Using device: {next(model.parameters()).device}")
        print(f"total batch size (tokens) {model.config.total_batch_size} | grad accum steps {grad_accum_steps}.")
        print(f"train total tokens {dl_train.total_tokens} | batches per epoch {dl_train.total_tokens // (dl_train.B*dl_train.T*grad_accum_steps)}")
        print(f"val total tokens {dl_val.total_tokens} | batches per epoch {dl_val.total_tokens // (dl_val.B*dl_val.T*model.config.eval_iters)}")

    torch.set_float32_matmul_precision('high')
    t0 = time.time()

    # plotting
    plt.ioff()
    fig_token, ax_token = plt.subplots()
    fig_attr, ax_attr = plt.subplots()

    while iter_global <= model.config.max_iters:

        t1 = time.time()
        dt = max(t1 - t0, 1e-9)
        t0 = t1
        tokens_processed = model.config.batch_size * model.config.block_size * grad_accum_steps
        tokens_per_sec = tokens_processed / dt

        if iter_global % model.config.eval_interval == 0 and iter_global > 0:
            est_loss = model.estimate_loss_light(dl_train, dl_val)
            print(f"iter {iter_global}: loss {est_loss['train']['token']:.4f} | val loss {est_loss['val']['token']:.4f} | dt {dt*1000:.2f}ms | tok/sec {tokens_per_sec:.2f} | train epoch {dl_train.epoch}",flush=True)

            if est_loss['val']['total'] < best_val_loss and out_dir is not None:
                best_val_loss = est_loss['val']['total'].item()
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': asdict(model.config),
                    'iter_global': iter_global,
                    'best_val_loss': best_val_loss,
                    'tokenizer':tok,
                    'token_losses': token_losses,
                    'attr_losses': attr_losses
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            if logging and logger is not None:
                logger.log(iter_global,
                       est_loss['train']['token'].item(),
                       est_loss['val']['token'].item(),
                       est_loss['train']['attr'].item(),
                       est_loss['val']['attr'].item(),
                       dt*1000,tokens_per_sec,dl_train.epoch)
            
            if generate_text:
                seed_id = tok.encode('<|sos|>')
                _,text = model.sample(tok,seed_idx=seed_id,max=100)
                print(text)
        
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        attr_loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x,y,attr = dl_train.next_batch()
            with torch.autocast(device_type=model.config.device, dtype=torch.bfloat16):
                _, _, attr_loss, loss = model(x,y,attr)
            loss = loss / grad_accum_steps
            attr_loss = attr_loss / grad_accum_steps
            loss_accum += loss.detach()
            attr_loss_accum += attr_loss.detach()
            loss.backward()

        lr = get_lr(iter_global,model.config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        iter_global += 1
        
        token_loss = loss_accum.item() - attr_loss_accum.item() * model.config.attr_weight
        attr_loss = attr_loss_accum.item()
        token_losses.append(token_loss)
        attr_losses.append(attr_loss)
        
        if out_dir is not None and iter_global % 100 == 0:
            # plot next-token loss
            ax_token.clear()
            ax_token.plot(range(len(token_losses)), token_losses)
            ax_token.set_xlabel("Iteration")
            ax_token.set_ylabel("Next-token Loss")
            fig_token.tight_layout()
            fig_token.savefig(os.path.join(out_dir, "token_loss.png"))

            # plot attr loss
            ax_attr.clear()
            ax_attr.plot(range(len(attr_losses)), attr_losses)
            ax_attr.set_xlabel("Iteration")
            ax_attr.set_ylabel("Conditional Attribute Loss")
            fig_attr.tight_layout()
            fig_attr.savefig(os.path.join(out_dir, "attr_loss.png"))

        print(f"iter {iter_global} | loss {loss_accum:.4f} | token loss {token_loss:.4f} | attr_loss {attr_loss:.4f} | dt {dt*1000:.2f}ms | tok/sec {tokens_per_sec:.2f} | train epoch {dl_train.epoch}",flush=True)

    est_loss = model.estimate_loss(dl_train, dl_val)
    print(f"iter {iter_global} | train loss {loss_accum:.4f} | dt {dt*1000:.2f}ms | tok/sec {tokens_per_sec:.2f} | train epoch {dl_train.epoch}",flush=True)

if __name__ == "__main__":
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    parser.add_argument('-ckpt', '--ckpt', required=True, help='Path to trained standard checkpoint (.pt)')
    args = parser.parse_args()
    ckpt = args.ckpt
    config_yaml = load_config(args.config)
    data_dir = os.path.normpath(config_yaml['data_dir'])
    out_dir = os.path.normpath(config_yaml['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    model_cfg = config_yaml['model']
    model_name = model_cfg['name']
    batch_size = model_cfg['batch_size']
    max_seq_len = model_cfg['block_size']
    device = model_cfg['device']
    dl_name = config_yaml['dataloader']

    model, optimizer, iter_global, best_val_loss, token_losses, attr_losses = setup_model(model_cfg,model_name,ckpt,out_dir,config_yaml['init_from'],device)

    train_token_file = os.path.join(data_dir,'train_token_ids.bin')
    train_attr_file = os.path.join(data_dir,'train_attr_ids.bin')
    val_token_file = os.path.join(data_dir,'val_token_ids.bin')
    val_attr_file = os.path.join(data_dir,'val_attr_ids.bin')

    dl_train = get_dataloader(dl_name,batch_size,max_seq_len,train_token_file,train_attr_file,device)
    dl_val = get_dataloader(dl_name,batch_size,max_seq_len,val_token_file,val_attr_file,device)

    with open(os.path.join(data_dir,'tok.pkl'),'rb') as f:
        tok = pickle.load(f)
    
    # ---- sanity check: what is the token loss immediately after loading ckpt? ----
    model.eval()
    with torch.no_grad():
        est0 = model.estimate_loss_light(dl_train, dl_val)

    print("\n=== Loaded checkpoint losses (before any training) ===")
    print(f"train token: {est0['train']['token']:.4f} | val token: {est0['val']['token']:.4f}")
    print(f"train attr : {est0['train']['attr']:.4f} | val attr : {est0['val']['attr']:.4f}")
    print(f"train total: {est0['train']['total']:.4f} | val total: {est0['val']['total']:.4f}\n")


    train(model,optimizer,tok,dl_train,dl_val,
          logging=config_yaml['log'],
          out_dir=out_dir,
          iter_global=iter_global,
          token_losses=token_losses,
          attr_losses=attr_losses,
          best_val_loss=best_val_loss,
          generate_text=True)
    
