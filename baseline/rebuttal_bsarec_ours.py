#%%
"""Rebuttal-only copy of debiased_seq_rec_weighted_fixed.py for BSARec (Ours, Hawkes negatives).

The training procedure is unchanged (ablation=shared, dr-anchor=user, timefix). Only
the bookkeeping differs, so the existing checkpoints can never be loaded or overwritten:
  - checkpoints always go to ./weights_rebuttal/<dataset>/ (--weights_path is ignored)
  - file names carry the BSARec configuration:
        rebuttal_ours_bsarec_c{c}_a{alpha}_lambda{lambda1}_e{epoch}_seed{seed}.pt
  - no wandb, no end-of-training evaluation (see rebuttal_bsarec_eval.py)
"""
import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from module.utils import parse_args, set_seed, set_device
from module.dataset import UserItemTime
from module.model import score_pair
from module.bsarec import BSARec
from module.debias import build_unshared_debias_model
from module.sampler import make_prior_snapshot, sample_epoch_negatives

SAVE_ROOT = "./weights_rebuttal"


def get_epoch(path):
    match = re.search(r"_e(\d+)_", path.name)
    return int(match.group(1)) if match else -1


#%%
args = parse_args()
assert args.dr_anchor == "user", "the submitted Ours runs use --dr-anchor=user"
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{SAVE_ROOT}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)
tag = f"rebuttal_ours_bsarec_c{args.c}_a{args.alpha}_lambda{args.lambda1}"
print(f"[{tag}] seed={args.seed} -> {args.save_path}")

#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)
dataset.item_time_array = dataset.item_time_array / 86400.0

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

#%%
debiased_class = build_unshared_debias_model(BSARec)   # == --ablation=shared
model = debiased_class(
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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.decay,
)

#%%
dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)

snapshot = make_prior_snapshot(model)
hot_negs = sample_epoch_negatives(
    snapshot=snapshot,
    train_events=dataset.train_hot_events,
    num_items=dataset.m_item,
    num_negatives=args.contrast_size-1,
)

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
    epoch_item_loss = 0.0

    for idx in range(batch_num):
        user_loss = torch.zeros(1).to(args.device)
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]

        """USER"""
        hot_anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        hot_neg_item = torch.tensor(hot_negs[hot_sample_idx], dtype=torch.long, device=args.device)
        pos_score = score_pair(model, hot_pos_item, anchor_hist_items, hot_anchor_user)
        neg_score = score_pair(model, hot_neg_item, anchor_hist_items, hot_anchor_user)
        user_loss += -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean() * args.lambda1

        epoch_user_loss += user_loss.item()

        """ITEM"""
        cold_sample_idx = cold_idxs[cold_mini_batch*idx : (idx + 1)*cold_mini_batch]
        cold_pos_item = torch.tensor(dataset.cold_pos_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        pos_item = torch.cat([cold_pos_item, hot_pos_item], dim=0)

        hot_neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        cold_neg_item = torch.tensor(dataset.cold_neg_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.cat([cold_neg_item, hot_neg_item], dim=0)

        hot_pos_time = torch.Tensor(dataset.hot_event_time_list[hot_sample_idx]).reshape(-1, 1, 1).to(args.device)
        cold_pos_time = torch.Tensor(dataset.cold_event_time_list[cold_sample_idx]).reshape(-1, 1, 1).to(args.device)
        pos_time = torch.cat([cold_pos_time, hot_pos_time], dim=0)

        hot_pos_time_all = torch.Tensor(dataset.hot_pos_time_all[hot_sample_idx]).to(args.device)
        cold_pos_time_all = torch.Tensor(dataset.cold_pos_time_all[cold_sample_idx]).to(args.device)
        pos_time_all = torch.cat([cold_pos_time_all, hot_pos_time_all], dim=0)

        hot_neg_time_all = torch.Tensor(dataset.hot_neg_time_all[hot_sample_idx]).to(args.device)
        cold_neg_time_all = torch.Tensor(dataset.cold_neg_time_all[cold_sample_idx]).to(args.device)
        neg_time_all = torch.cat([cold_neg_time_all, hot_neg_time_all], dim=0)

        batch_items = torch.concat([pos_item.unsqueeze(-1), neg_item], -1).reshape(pos_item.shape[0], -1)
        batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

        logits = model.prior(batch_items, pos_time, batch_time_all)
        log_logits = torch.log(logits + 1e-9)
        item_loss = -nn.functional.log_softmax(log_logits, dim=-1)[:, 0].mean() * (1-args.lambda1)
        epoch_item_loss += item_loss.item()

        dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)

        total_loss = item_loss + user_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] user: {epoch_user_loss / batch_num:.4f} / item: {epoch_item_loss / batch_num:.4f}", flush=True)

    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_user_loss,
            "args": vars(args),
        }, f"{args.save_path}/{tag}_e{epoch}_seed{args.seed}.pt")

    if epoch % args.pair_reset_interval == 0:
        print("Reset negative users")
        snapshot = make_prior_snapshot(model)
        hot_negs = sample_epoch_negatives(
            snapshot=snapshot,
            train_events=dataset.train_hot_events,
            num_items=dataset.m_item,
            num_negatives=args.contrast_size-1,
        )

print(f"[{tag}] seed={args.seed} done (epoch {epoch})")
