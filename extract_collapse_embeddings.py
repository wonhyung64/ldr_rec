#%%
"""
Extract analysis-time embeddings (h_u, z_v~) for the geometric-collapse study.

Implements the protocol described for the three comparison conditions:
  1. Vanilla            : uniform negative sampling backbone
  2. Vanilla + Post-hoc L2 Norm : same checkpoint as Vanilla, normalized only at analysis time
  3. Ours                : Hawkes-debiased negative sampling backbone

For "Ours", we deliberately read the *recommender's own* item embedding
(`model.item_embedding`), not the Hawkes prior model's item embedding
(`model.p_item_embedding` under the "shared"-ablation / unshared-debias wrapper).
This is the embedding that enters the density-ratio score
s(x_u(t), v) = rho(h_u(t), z_v~), before it is recombined with the
population item prior log pi(v|t) into the final ranking score r_eta.

User context embeddings h_u(t) are taken once per user, from the history
strictly before that user's test-split interaction time (h_u = h_u(t_test)),
matching how the backbone is evaluated at test time.
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))

from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import (
    build_debias_model,
    build_unshared_debias_model,
    build_linear_debias_model,
    build_unshared_linear_debias_model,
)

ABLATION_BUILDERS = {
    "none": build_debias_model,
    "shared": build_unshared_debias_model,
    "linear": build_linear_debias_model,
    "both": build_unshared_linear_debias_model,
}


def set_device(device="none"):
    if device == "none":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def build_model(model_name, num_users, num_items, recdim, device, tau, depth,
                 max_seq_len, n_heads, dropout, time_span, debiased, ablation,
                 alpha=0.5, c=3):
    model_class = MODEL_REGISTRY[model_name]
    if debiased:
        model_class = ABLATION_BUILDERS[ablation](model_class)
    extra_kwargs = {"alpha": alpha, "c": c} if model_name == "bsarec" else {}
    if model_name == "tisasrec":
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_k=recdim,
            device=device,
            tau=tau,
            depth=depth,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            dropout=dropout,
            time_span=time_span,
        ).to(device)

    else:
        model = model_class(
            num_users=num_users,
            num_items=num_items,
            embedding_k=recdim,
            device=device,
            tau=tau,
            depth=depth,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            dropout=dropout,
            **extra_kwargs,
        ).to(device)
    return model


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    result = model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {path} (epoch={ckpt.get('epoch')}): {result}")
    return model


@torch.no_grad()
def extract_user_context_embeddings(model, dataset, user_item_time, max_seq_len, device):
    """h_u = h_u(t_test): one context embedding per user from the history
    strictly before that user's held-out interaction time. Also returns the
    ground-truth positive item for that event, used later as an empirical
    proxy for the cone-membership angle alpha in Theorem 1."""
    model.eval()
    user_ids, pos_items, embeds = [], [], []
    for (user, item), t in user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [t]), max_seq_len)
        hist_item = torch.tensor(hist_item_np, dtype=torch.long, device=device)
        hist_time = torch.tensor(hist_time_np, dtype=torch.long, device=device) * 24 * 60 * 60
        h = model.encode_user(hist_item, hist_time)  # [1, D]
        user_ids.append(user)
        pos_items.append(item)
        embeds.append(h.squeeze(0).detach().cpu().numpy())
    return (
        np.array(user_ids, dtype=np.int64),
        np.array(pos_items, dtype=np.int64),
        np.stack(embeds, axis=0),
    )


def extract_user_embeddings(model, dataset, user_item_time, max_seq_len, device):
    """h_u = h_u(t_test): one context embedding per user from the history
    strictly before that user's held-out interaction time. Also returns the
    ground-truth positive item for that event, used later as an empirical
    proxy for the cone-membership angle alpha in Theorem 1."""
    model.eval()
    user_ids, pos_items, embeds = [], [], []
    for (user, item), t in user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [t]), max_seq_len)
        hist_item = torch.tensor(hist_item_np, dtype=torch.long, device=device)
        hist_time = torch.tensor(hist_time_np, dtype=torch.long, device=device) * 24 * 60 * 60
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)
        h = model.encode_user(hist_item, user_t)  # [1, D]
        user_ids.append(user)
        pos_items.append(item)
        embeds.append(h.squeeze(0).detach().cpu().numpy())
    return (
        np.array(user_ids, dtype=np.int64),
        np.array(pos_items, dtype=np.int64),
        np.stack(embeds, axis=0),
    )



@torch.no_grad()
def extract_item_embeddings(model, num_items):
    """z_v~: recommender's own item embedding used for scoring (not the
    Hawkes prior model's item embedding)."""
    return model.item_embedding.weight[:num_items].detach().cpu().numpy()


def item_popularity(dataset):
    counts = np.zeros(dataset.m_item, dtype=np.int64)
    idx, c = np.unique(dataset.trainItem, return_counts=True)
    counts[idx] = c
    return counts


def l2_normalize(x, eps=1e-8):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="micro_video")
    p.add_argument("--model-name", type=str, default="tisasrec")
    p.add_argument("--condition", type=str, default="ours") #"vanilla"
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Override checkpoint path; default resolves from the standard naming convention.")
    p.add_argument("--ablation", type=str, default="shared", help="Only used when --condition=ours.")
    p.add_argument("--lambda1", type=float, default=0.7, help="Only used to resolve the default --condition=ours checkpoint name.")
    p.add_argument("--epoch", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--recdim", type=int, default=128)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--max-seq-len", type=int, default=50)
    p.add_argument("--n-heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.5, help="BSARec only: frequency-mixing weight.")
    p.add_argument("--c", type=int, default=3, help="BSARec only: low-pass filter cutoff.")
    p.add_argument("--weights-path", type=str, default="./weights")
    p.add_argument("--out-dir", type=str, default="./collapse_analysis/embeddings")
    p.add_argument("--device", type=str, default="none")
    try:
        return p.parse_args()
    except:
        return p.parse_args([])
    


# %%
args = parse_args()
args.device = set_device(args.device)
time_span = 2048 if args.dataset == "ml-1m" else 512

# args.model_name = "mf"
# args.lambda1 = 0.1
# args.condition = "vanilla"

save_dir = f"{args.weights_path}/{args.dataset}"
if args.checkpoint is not None:
    ckpt_path = args.checkpoint
elif args.condition == "ours":
    ckpt_path = f"{save_dir}/_{args.model_name}_lambda{args.lambda1}_e{args.epoch}_seed{args.seed}_ablation{args.ablation}.pt"
else:
    ckpt_path = f"{save_dir}/backbone_{args.model_name}_e{args.epoch}_seed{args.seed}.pt"

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint not found for condition='{args.condition}': {ckpt_path}\n"
        f"(Vanilla must be trained with baseline/seq_rec_tisasrec.py first.)"
    )

dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)

model = build_model(
    model_name=args.model_name,
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    recdim=args.recdim,
    device=args.device,
    tau=args.tau,
    depth=args.depth,
    max_seq_len=args.max_seq_len,
    n_heads=args.n_heads,
    dropout=args.dropout,
    time_span=time_span,
    debiased=(args.condition == "ours"),
    ablation=args.ablation,
    alpha=args.alpha,
    c=args.c,
)
load_checkpoint(model, ckpt_path, args.device)

if args.model_name in ("mf", "ncf"):
    user_ids, pos_items, h_u = extract_user_embeddings(
        model, dataset, dataset.test_user_item_time, args.max_seq_len, args.device
    )
else:
    user_ids, pos_items, h_u = extract_user_context_embeddings(
        model, dataset, dataset.test_user_item_time, args.max_seq_len, args.device
    )

z_v = extract_item_embeddings(model, dataset.m_item)
pop = item_popularity(dataset)

os.makedirs(args.out_dir, exist_ok=True)
out_path = f"{args.out_dir}/{args.condition}_{args.model_name}_{args.dataset}_seed{args.seed}.npz"
np.savez(
    out_path,
    user_ids=user_ids,
    pos_items=pos_items,
    h_u=h_u,
    z_v=z_v,
    item_popularity=pop,
    condition=args.condition,
)
print(f"Saved: {out_path}")
print(f"  h_u: {h_u.shape}, mean norm={np.linalg.norm(h_u, axis=-1).mean():.4f}")
print(f"  z_v: {z_v.shape}, mean norm={np.linalg.norm(z_v, axis=-1).mean():.4f}")