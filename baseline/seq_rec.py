#%%
import os
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn.functional as F

from torch import optim
from datetime import datetime

from module.utils import parse_args, set_seed, set_device
from module.procedure import computeTopNAccuracy
from module.dataset import UserItemTime
from module.model import build_model, score_pair, score_all


#%%
"""
Residual-only density-ratio baselines trained with logistic loss.

Key design choices:
- No prior module.
- Uniform negative sampling distribution q(u) over users.
- Keep the training/evaluation skeleton close to the user's current code.
- Make model replacement easy by sharing the same interface:
    residual_score(item_idx, hist_item_idx, user_idx=None)
    score_all_items(hist_item_idx, user_idx=None)
"""

args = parse_args()
args.model_name = "grurec"
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)


wandb_login = False
file_dir = inspect.getfile(inspect.currentframe())
file_name = file_dir.split("/")[-1]
if file_name.endswith(".py"):
    try:
        wandb_login = wandb.login(key=open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
    except Exception:
        wandb_login = False

if wandb_login:
    expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
    args.expt_name = f"{file_name.split('.')[-2]}_{args.model_name}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec2", config=vars(args))
    wandb.run.name = args.expt_name



#%%
dataset = UserItemTime(args)
dataset.build_user_histories(max_seq_len=args.max_seq_len)
dataset.get_pair_item_uniform(k=args.contrast_size-1)
hot_ratio = dataset.hotDataSize / dataset.trainDataSize

#%%
mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

model = build_model(args, dataset, mini_batch)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


#%%
best_valid_score = 0.0
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0
cnt = 1
for epoch in range(1, args.epochs + 1):
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(hot_idxs)
    epoch_user_loss = 0.0

    for idx in range(batch_num):
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]
        anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        pos_score = score_pair(model, pos_item, anchor_hist_items, anchor_user)
        neg_score = score_pair(model, neg_item, anchor_hist_items, anchor_user)

        user_loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).sum()
        optimizer.zero_grad()
        user_loss.backward()
        optimizer.step()

        epoch_user_loss += user_loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] ldr: {epoch_user_loss / batch_num:.4f}")

    if epoch % args.pair_reset_interval == 0:
        print("Reset uniform negative users")
        dataset.get_pair_item_uniform(k=args.contrast_size-1)

    if epoch % args.evaluate_interval == 0:
        pred_list = []
        gt_list = []

        model.eval()
        for (user, item), pos_time_val in dataset.valid_user_item_time.items():
            hist_item_np = dataset.get_histories_for_users_at_times([user], [pos_time_val], max_seq_len=args.max_seq_len)
            hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
            user_t = torch.tensor([user], dtype=torch.long, device=args.device)

            with torch.no_grad():
                pred = score_all(model, hist_item_t, user_t).squeeze(0).cpu()

            exclude_items = list(dataset._allPos[user])
            pred[exclude_items] = -9999
            _, pred_k = torch.topk(pred, k=max(args.topks))
            pred_list.append(pred_k.cpu())
            gt_list.append([item])

        valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

        if wandb_login:
            wandb_var.log({
                "train_ldr": epoch_user_loss / batch_num,
            })
            wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
            wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
            wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
            wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

        current_valid_score = valid_results[1][0]
        if current_valid_score - best_valid_score <= 0.0:
            cnt += 1
        else:
            best_valid_score = current_valid_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            cnt = 1

        if cnt == 5:
            break

pred_list = []
gt_list = []

best_model = build_model(args, dataset, mini_batch)
best_model.load_state_dict(best_state)
best_model.eval()

for (user, item), pos_time_val in dataset.test_user_item_time.items():
    hist_item_np = dataset.get_histories_for_users_at_times([user], [pos_time_val], max_seq_len=args.max_seq_len)
    hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
    user_t = torch.tensor([user], dtype=torch.long, device=args.device)

    with torch.no_grad():
        pred = score_all(best_model, hist_item_t, user_t).squeeze(0).cpu()

    exclude_items = list(dataset._allPos[user])
    pred[exclude_items] = -9999
    _, pred_k = torch.topk(pred, k=max(args.topks))
    pred_list.append(pred_k.cpu())
    gt_list.append([item])

test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

if wandb_login:
    wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
    wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
    wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
    wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))
    wandb_var.log({"best_valid_score": best_valid_score})
    wandb_var.log({"best_epoch": best_epoch})
    wandb_var.finish()

# %%
