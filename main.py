#%%
import os
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, Sequence, Tuple


from module.utils import *
from module.eval import *
from module.sampling import *
from module.dataset import *
from module.model import *


EPS = 1e-12
WANDB = False


def compute_joint_batch_loss(
    model: HawkesMFDebias,
    train_events: Sequence[Tuple[int, int, float]],
    batch_idx: np.ndarray,
    residual_negatives: np.ndarray,
    item_time_padded: np.ndarray,
    num_items: int,
    prior_num_negatives: int,
    prior_weight: float,
    residual_weight: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pos_items_np = np.asarray([train_events[i][1] for i in batch_idx], dtype=np.int64)
    users_np = np.asarray([train_events[i][0] for i in batch_idx], dtype=np.int64)
    query_times_np = np.asarray([train_events[i][2] for i in batch_idx], dtype=np.float32)

    # Prior loss: sampled-softmax surrogate with uniform negatives.
    prior_neg_items_np = prior_sampling(pos_items_np, num_items, prior_num_negatives)
    prior_candidates_np = np.concatenate([pos_items_np[:, None], prior_neg_items_np], axis=1)
    prior_histories_np = item_time_padded[prior_candidates_np]

    prior_candidates_t = torch.tensor(prior_candidates_np, dtype=torch.long, device=device)
    query_times_t = torch.tensor(query_times_np, dtype=torch.float32, device=device)
    prior_histories_t = torch.tensor(prior_histories_np, dtype=torch.float32, device=device)

    prior_logits = model.prior_logits_for_candidates(prior_candidates_t, query_times_t, prior_histories_t)
    log_prior_logits = torch.log(prior_logits + EPS)
    prior_loss = -F.log_softmax(log_prior_logits, dim=1)[:, 0].mean()

    # Residual loss: logistic NCE with frozen-prior negatives pre-built for the epoch.
    users_t = torch.tensor(users_np, dtype=torch.long, device=device)
    pos_items_t = torch.tensor(pos_items_np, dtype=torch.long, device=device)
    residual_neg_items_t = torch.tensor(residual_negatives[batch_idx], dtype=torch.long, device=device)

    pos_scores = model.residual_scores(users_t, pos_items_t)
    user_vec = model.user_embedding(users_t)
    neg_item_vec = model.item_embedding(residual_neg_items_t)
    neg_scores = (user_vec.unsqueeze(1) * neg_item_vec).sum(dim=-1)
    residual_loss = -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()

    total_loss = prior_weight * prior_loss + residual_weight * residual_loss
    stats = {
        "prior_loss": float(prior_loss.detach().item()),
        "residual_loss": float(residual_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }
    return total_loss, stats


#%%
args = parse_args()
set_seed(args.seed)
args.device = torch.device(set_device())

file_dir = inspect.getfile(inspect.currentframe())
file_name = file_dir.split("/")[-1]
if file_name.split(".")[-1] == "py":
    try:
        wandb_login = wandb.login(key=open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
    except:
        pass

if wandb_login:
    expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
    args.expt_name = f"{file_name.split(".")[-2]}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec2", config=vars(args))
    wandb.run.name = args.expt_name


data = InteractionData(args.data_path, args.dataset)
model = HawkesMFDebias(
    num_users=data.num_users,
    num_items=data.num_items,
    embedding_dim=args.embedding_dim,
    prior_hidden_dim=args.prior_hidden_dim,
    prior_depth=args.prior_depth,
    tau=args.tau,
    prior_vector_norm=args.prior_vector_norm,
    residual_vector_norm=args.residual_vector_norm,
).to(args.device)

optimizer = torch.optim.Adam([
    {
        "params": list(model.base_net.parameters()) + list(model.excitation_net.parameters()) + [model.log_beta],
        "lr": args.prior_lr,
        "weight_decay": args.prior_decay,
    },
    {
        "params": list(model.user_embedding.parameters()),
        "lr": args.residual_lr,
        "weight_decay": args.residual_decay,
    },
    {
        "params": list(model.item_embedding.parameters()),
        "lr": args.shared_lr,
        "weight_decay": args.shared_decay,
    },
    ])


#%%
best_state = copy.deepcopy(model.state_dict())
best_valid = -float("inf")
patience = 0


for epoch in range(1, args.epochs + 1):
    # Freeze current prior snapshot once per epoch and build residual negatives from it.
    snapshot = make_prior_snapshot(model, args.device)
    epoch_negatives = residual_sampling(
        snapshot=snapshot,
        train_events=data.train_events,
        num_items=data.num_items,
        num_negatives=args.residual_num_negatives,
    )

    model.train()
    indices = np.random.permutation(len(data.train_events))
    sum_prior, sum_residual, sum_total = 0.0, 0.0, 0.0
    num_batches = 0

    for start in range(0, len(indices), args.batch_size):
        batch_idx = indices[start : start + args.batch_size]
        optimizer.zero_grad()
        total_loss, stats = compute_joint_batch_loss(
            model=model,
            train_events=data.train_events,
            batch_idx=batch_idx,
            residual_negatives=epoch_negatives,
            item_time_padded=data.train_item_time_padded,
            num_items=data.num_items,
            prior_num_negatives=args.prior_num_negatives,
            prior_weight=args.prior_weight,
            residual_weight=args.residual_weight,
            device=args.device,
        )
        total_loss.backward()
        optimizer.step()

        sum_prior += stats["prior_loss"]
        sum_residual += stats["residual_loss"]
        sum_total += stats["total_loss"]
        num_batches += 1

    prior_loss = sum_prior / max(num_batches, 1)
    residual_loss = sum_residual / max(num_batches, 1)
    joint_loss = sum_total / max(num_batches, 1)
    print(
        f"[Epoch {epoch:03d}] total_loss={joint_loss:.6f} "
        f"prior_loss={prior_loss:.6f} residual_loss={residual_loss:.6f} "
        f"beta={model.current_beta().item():.6f}"
    )

    if epoch % args.eval_every != 0:
        continue

    valid_metrics = evaluate_split(
        model=model,
        events=data.valid_events,
        train_pos_items=data.all_train_pos_items,
        item_time_padded=data.train_item_time_padded,
        pad_value=data.train_time_pad_value,
        topks=args.topks,
        ranking_mode=args.ranking_mode,
        device=args.device,
        item_chunk_size=args.eval_item_chunk_size,
    )
    valid_recall = valid_metrics["recall"][0]
    print(
        f"  valid recall@{args.topks[0]}={valid_recall:.6f} "
        f"ndcg@{args.topks[0]}={valid_metrics['ndcg'][0]:.6f}"
    )

    if valid_recall > best_valid:
        best_valid = valid_recall
        best_state = copy.deepcopy(model.state_dict())
        patience = 0
        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(best_state, args.save_path)
    else:
        patience += 1
        if patience >= args.patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_state)
test_metrics = evaluate_split(
    model=model,
    events=data.test_events,
    train_pos_items=data.all_train_pos_items,
    item_time_padded=data.train_item_time_padded,
    pad_value=data.train_time_pad_value,
    topks=args.topks,
    ranking_mode=args.ranking_mode,
    device=args.device,
    item_chunk_size=args.eval_item_chunk_size,
)
print("\n[Test metrics]")
for metric_name, values in test_metrics.items():
    joined = ", ".join(f"{metric_name}@{k}={v:.6f}" for k, v in zip(args.topks, values))
    print(joined)

# %%
