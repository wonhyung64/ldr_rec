#%%
"""Rebuttal-only copy of seq_rec.py for BSARec (vanilla: uniform negatives, cosine/tau scoring).

The training procedure is unchanged. Only the bookkeeping differs, so the existing
checkpoints can never be loaded or overwritten:
  - checkpoints always go to ./weights_rebuttal/<dataset>/ (--weights_path is ignored)
  - file names carry the BSARec configuration:
        rebuttal_vanilla_cos_bsarec_c{c}_a{alpha}_e{epoch}_seed{seed}.pt
  - no wandb, no in-training evaluation (see rebuttal_bsarec_eval.py)
"""
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import optim

from module.utils import parse_args, set_seed, set_device
from module.dataset import UserItemTime
from module.model import score_pair
from module.bsarec import BSARec

SAVE_ROOT = "./weights_rebuttal"


def get_epoch(path):
    match = re.search(r"_e(\d+)_", path.name)
    return int(match.group(1)) if match else -1


#%%
args = parse_args()
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{SAVE_ROOT}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)
tag = f"rebuttal_vanilla_cos_bsarec_c{args.c}_a{args.alpha}"
print(f"[{tag}] seed={args.seed} -> {args.save_path}")

#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)
dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)

#%%
model = BSARec(
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    embedding_k=args.recdim,
    device=args.device,
    tau=args.tau,
    depth=args.depth,
    max_seq_len=args.max_seq_len,
    n_heads=args.n_heads,
    dropout=args.dropout,
    c=args.c,
    alpha=args.alpha,
    ).to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

#%%
epoch = 0

save_dir = Path(args.save_path)
matched_files = sorted(save_dir.glob(f"{tag}_e???_seed{args.seed}.pt"))
if len(matched_files) > 0:
    recent_file = max(matched_files, key=get_epoch)
    checkpoint = torch.load(recent_file, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"MODEL LOADED! {recent_file}")


while epoch < args.epochs:
    epoch += 1
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(hot_idxs)
    epoch_user_loss = 0.0

    for idx in range(batch_num):
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]
        anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        pos_score = score_pair(model, pos_item, anchor_hist_items, anchor_user)
        neg_score = score_pair(model, neg_item, anchor_hist_items, anchor_user)

        user_loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).sum()
        optimizer.zero_grad()
        user_loss.backward()
        optimizer.step()

        epoch_user_loss += user_loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] ldr: {epoch_user_loss / batch_num:.4f}", flush=True)

    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_user_loss,
            "args": vars(args),
        }, f"{args.save_path}/{tag}_e{epoch}_seed{args.seed}.pt")

    if epoch % args.pair_reset_interval == 0:
        print("Reset uniform negative users")
        dataset.get_pair_item_uniform(k=args.contrast_size-1)

print(f"[{tag}] seed={args.seed} done (epoch {epoch})")
