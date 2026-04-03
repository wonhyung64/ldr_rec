import os
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from datetime import datetime
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from modules.utils import parse_args, set_seed, set_device
from modules.procedure import computeTopNAccuracy


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


class UserItemTime(Dataset):
    def __init__(self, args):
        path = f"{args.data_path}/{args.dataset}"
        time_file = path + '/interaction_time_dict.npy'
        train_file = path + '/training_dict.npy'
        valid_file = path + '/validation_dict.npy'
        test_file = path + '/testing_dict.npy'

        self.time_dict = np.load(time_file, allow_pickle=True).item()
        self.train_dict = np.load(train_file, allow_pickle=True).item()
        self.valid_dict = np.load(valid_file, allow_pickle=True).item()
        self.test_dict = np.load(test_file, allow_pickle=True).item()

        self.trainUniqueUsers, self.trainUser, self.trainItem, self.trainDataSize = self.load_set(self.train_dict)
        self.validUniqueUsers, self.validUser, self.validItem, self.validDataSize = self.load_set(self.valid_dict)
        self.testUniqueUsers, self.testUser, self.testItem, self.testDataSize = self.load_set(self.test_dict)

        self.m_item = max(self.trainItem.max(), self.validItem.max(), self.testItem.max()) + 1
        self.n_user = max(self.trainUser.max(), self.validUser.max(), self.testUser.max()) + 1
        self.UserItemNet = csr_matrix(
            (np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )

        self._allPos = self.getUserPosItems(list(range(self.n_user)), self.UserItemNet)
        self._allPosUsers = self.getItemPosUsers(list(range(self.m_item)), self.UserItemNet)

        self.train_user_item_time = self.set_to_pair(self.train_dict, self.time_dict)
        self.valid_user_item_time = self.set_to_pair(self.valid_dict, self.time_dict)
        self.test_user_item_time = self.set_to_pair(self.test_dict, self.time_dict)

        self.user_interactions = self.build_user_interactions(self.time_dict)
        self.build_event_lists()

    def load_set(self, set_dict):
        UniqueUsers, Item, User = [], [], []
        dataSize = 0
        for uid in set_dict.keys():
            if len(set_dict[uid]) != 0:
                UniqueUsers.append(uid)
                User.extend([uid] * len(set_dict[uid]))
                Item.extend(set_dict[uid])
                dataSize += len(set_dict[uid])
        return np.array(UniqueUsers), np.array(User), np.array(Item), dataSize

    def set_to_pair(self, set_dict, time_dict):
        user_item_time = {}
        for user in set_dict:
            for item in set_dict[user]:
                time = time_dict[user][item] / 60 / 60 / 24
                user_item_time[(user, item)] = time
        return user_item_time

    def getUserPosItems(self, users, UserItemNet):
        posItems = []
        for user in users:
            posItems.append(UserItemNet[user, :].nonzero()[1])
        return posItems

    def getItemPosUsers(self, items, UserItemNet):
        posUsers = []
        for item in items:
            posUsers.append(UserItemNet[:, item].nonzero()[0])
        return posUsers

    def build_user_interactions(self, time_dict):
        user_interactions = {u: [] for u in range(self.n_user)}
        for u, item_time in time_dict.items():
            for it, t in item_time.items():
                t_day = float(t) / 60 / 60 / 24
                user_interactions[u].append((t_day, it))
        for u in user_interactions:
            user_interactions[u].sort(key=lambda x: x[0])
        return user_interactions

    def build_event_lists(self):
        self.train_events = [(u, v, float(t)) for (u, v), t in self.train_user_item_time.items()]
        self.valid_events = [(u, v, float(t)) for (u, v), t in self.valid_user_item_time.items()]
        self.test_events = [(u, v, float(t)) for (u, v), t in self.test_user_item_time.items()]

        self.train_events.sort(key=lambda x: x[2])
        self.valid_events.sort(key=lambda x: x[2])
        self.test_events.sort(key=lambda x: x[2])

        self.user_list = np.array([u for (u, v, t) in self.train_events], dtype=np.int64)
        self.item_list = np.array([v for (u, v, t) in self.train_events], dtype=np.int64)
        self.event_time_list = np.array([t for (u, v, t) in self.train_events], dtype=np.float32)
        self.trainDataSize = len(self.train_events)

    def build_user_histories(self, max_seq_len=50):
        self.max_seq_len = max_seq_len

        def _build_histories(events):
            hist_item_list = []
            for (u, v, t) in events:
                hist = [(tt, vv) for (tt, vv) in self.user_interactions[u] if tt < t]
                hist = hist[-max_seq_len:]
                h_items = [vv for (tt, vv) in hist]
                pad_len = max_seq_len - len(h_items)
                h_items = [self.m_item] * pad_len + h_items
                hist_item_list.append(h_items)
            return np.array(hist_item_list, dtype=np.int64)

        self.train_hist_item_list = _build_histories(self.train_events)
        self.valid_hist_item_list = _build_histories(self.valid_events)
        self.test_hist_item_list = _build_histories(self.test_events)
        self.hist_item_list = self.train_hist_item_list

    def get_histories_for_users_at_times(self, users, query_times, max_seq_len):
        hist_items_batch = []
        for u, t in zip(users, query_times):
            hist = [(tt, vv) for (tt, vv) in self.user_interactions[int(u)] if tt < float(t)]
            hist = hist[-max_seq_len:]
            h_items = [vv for (tt, vv) in hist]
            pad_len = max_seq_len - len(h_items)
            h_items = [self.m_item] * pad_len + h_items
            hist_items_batch.append(h_items)
        return np.array(hist_items_batch, dtype=np.int64)

    def get_pair_user_uniform(self, k=1):
        pos_user = self.user_list.astype(np.int64)
        N = len(pos_user)
        neg_user = np.random.randint(0, self.n_user - 1, size=(N, k), dtype=np.int64)
        neg_user += (neg_user >= pos_user[:, None])
        self.pos_user_list = pos_user
        self.neg_user_list = neg_user

    def __getitem__(self, idx):
        return (
            self.item_list[idx],
            self.pos_user_list[idx],
            self.neg_user_list[idx],
            self.hist_item_list[idx],
            self.event_time_list[idx],
        )

    def __len__(self):
        return self.trainDataSize


class ResidualBase(nn.Module):
    def __init__(self, num_users, num_items, embedding_k, device, tau=1.0, depth=1, max_seq_len=50, n_heads=2, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.dropout = dropout
        self.padding_item_id = self.num_items

        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_k, padding_idx=self.padding_item_id)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)

    def encode_user(self, hist_item_idx, user_idx=None):
        raise NotImplementedError

    def get_item_repr(self, item_idx):
        return self.item_embedding(item_idx)

    def residual_score(self, item_idx, hist_item_idx, user_idx=None):
        u = self.encode_user(hist_item_idx, user_idx=user_idx)
        v = self.get_item_repr(item_idx)
        h = torch.sum(u * v, dim=-1, keepdim=True)
        return h

    def score_all_items(self, hist_item_idx, user_idx=None):
        u = self.encode_user(hist_item_idx, user_idx=user_idx)
        v_all = self.get_item_repr(torch.arange(self.num_items, device=hist_item_idx.device))
        h = torch.matmul(u, v_all.T)
        return h


class MFResidual(ResidualBase):
    """Matrix Factorization baseline: static user/item embeddings."""
    def encode_user(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("MFResidual requires user_idx.")
        return self.user_embedding(user_idx)


class NCFResidual(ResidualBase):
    """Neural Collaborative Filtering baseline with static user/item embeddings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(8, self.embedding_k)
        layers = [nn.Linear(self.embedding_k * 2, hidden), nn.ReLU()]
        for _ in range(max(0, self.depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1, bias=False)]
        self.mlp = nn.Sequential(*layers)

    def encode_user(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        return self.user_embedding(user_idx)

    def residual_score(self, item_idx, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        u = self.user_embedding(user_idx)
        v = self.get_item_repr(item_idx)
        x = torch.cat([u, v], dim=-1)
        h = self.mlp(x)
        return h

    def score_all_items(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        u = self.user_embedding(user_idx)
        v_all = self.get_item_repr(torch.arange(self.num_items, device=u.device)).unsqueeze(0).expand(u.size(0), -1, -1)
        u_expand = u.unsqueeze(1).expand(-1, self.num_items, -1)
        h = self.mlp(torch.cat([u_expand, v_all], dim=-1)).squeeze(-1)
        return h


class GRURecResidual(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_gru = nn.GRU(
            input_size=self.embedding_k,
            hidden_size=self.embedding_k,
            num_layers=1,
            batch_first=True,
        )

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_emb = self.item_embedding(hist_item_idx)
        _, h_n = self.user_gru(hist_emb)
        u = h_n[-1]
        hist_mask = (hist_item_idx != self.padding_item_id)
        no_hist = (hist_mask.sum(dim=-1) == 0)
        if no_hist.any():
            fallback = self.user_embedding(user_idx) if user_idx is not None else torch.zeros_like(u)
            u = torch.where(no_hist.unsqueeze(-1), fallback, u)
        return u


class MLP4RecResidual(ResidualBase):
    """History MLP baseline over flattened sequence embeddings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_dim = self.max_seq_len * self.embedding_k
        hidden = max(32, self.embedding_k * 2)
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(self.dropout)]
        for _ in range(max(0, self.depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(self.dropout)]
        layers += [nn.Linear(hidden, self.embedding_k)]
        self.mlp = nn.Sequential(*layers)

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_emb = self.item_embedding(hist_item_idx)
        flat = hist_emb.reshape(hist_emb.size(0), -1)
        u = self.mlp(flat)
        hist_mask = (hist_item_idx != self.padding_item_id)
        no_hist = (hist_mask.sum(dim=-1) == 0)
        if no_hist.any():
            fallback = self.user_embedding(user_idx) if user_idx is not None else torch.zeros_like(u)
            u = torch.where(no_hist.unsqueeze(-1), fallback, u)
        return u


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pos_emb(pos)


class SASRecResidual(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_k,
            nhead=self.n_heads,
            dim_feedforward=max(64, self.embedding_k * 4),
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, self.depth))
        self.out_proj = nn.Linear(self.embedding_k, self.embedding_k)

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_mask = (hist_item_idx != self.padding_item_id)
        x = self.item_embedding(hist_item_idx)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=~hist_mask)

        lengths = hist_mask.sum(dim=-1)
        last_idx = torch.clamp(lengths - 1, min=0)
        u = x[torch.arange(x.size(0), device=x.device), last_idx]
        u = self.out_proj(u)

        no_hist = (lengths == 0)
        if no_hist.any():
            fallback = self.user_embedding(user_idx) if user_idx is not None else torch.zeros_like(u)
            u = torch.where(no_hist.unsqueeze(-1), fallback, u)
        return u


class BERT4RecResidual(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_k,
            nhead=self.n_heads,
            dim_feedforward=max(64, self.embedding_k * 4),
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, self.depth))
        self.pooler = nn.Linear(self.embedding_k, self.embedding_k)

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_mask = (hist_item_idx != self.padding_item_id)
        x = self.item_embedding(hist_item_idx)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=~hist_mask)

        mask_f = hist_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        u = (x * mask_f).sum(dim=1) / denom
        u = self.pooler(u)

        no_hist = (hist_mask.sum(dim=-1) == 0)
        if no_hist.any():
            fallback = self.user_embedding(user_idx) if user_idx is not None else torch.zeros_like(u)
            u = torch.where(no_hist.unsqueeze(-1), fallback, u)
        return u


MODEL_REGISTRY = {
    "mf": MFResidual,
    "ncf": NCFResidual,
    "grurec": GRURecResidual,
    "sasrec": SASRecResidual,
    "mlp4rec": MLP4RecResidual,
    "bert4rec": BERT4RecResidual,
}


def build_model(args, dataset, mini_batch):
    model_name = getattr(args, "model_name", "grurec").lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(
        num_users=dataset.n_user,
        num_items=dataset.m_item,
        embedding_k=args.recdim,
        device=args.device,
        tau=getattr(args, "tau", 1.0),
        depth=getattr(args, "depth", 1),
        max_seq_len=getattr(args, "max_seq_len", 50),
        n_heads=getattr(args, "n_heads", 2),
        dropout=getattr(args, "dropout", 0.1),
    ).to(args.device)


def score_pair(model, item_idx, hist_item_idx, user_idx):
    return model.residual_score(item_idx=item_idx, hist_item_idx=hist_item_idx, user_idx=user_idx)


def score_all(model, hist_item_idx, user_idx):
    return model.score_all_items(hist_item_idx=hist_item_idx, user_idx=user_idx)


args = parse_args()
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)

if not hasattr(args, "model_name"):
    args.model_name = "grurec"
if not hasattr(args, "max_seq_len"):
    args.max_seq_len = 50
if not hasattr(args, "n_heads"):
    args.n_heads = 2
if not hasattr(args, "dropout"):
    args.dropout = 0.1

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


dataset = UserItemTime(args)
dataset.build_user_histories(max_seq_len=args.max_seq_len)
dataset.get_pair_user_uniform(k=1)

mini_batch = args.batch_size // max(1, getattr(args, "contrast_size", 1))
if mini_batch <= 0:
    mini_batch = args.batch_size
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)

model = build_model(args, dataset, mini_batch)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

best_valid_score = 0.0
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0
cnt = 1

for epoch in range(1, args.epochs + 1):
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(all_idxs)
    epoch_user_loss = 0.0

    for idx in range(batch_num):
        sample_idx = all_idxs[mini_batch * idx: (idx + 1) * mini_batch]

        pos_user = torch.tensor(dataset.pos_user_list[sample_idx], dtype=torch.long, device=args.device)
        neg_user = torch.tensor(dataset.neg_user_list[sample_idx, 0], dtype=torch.long, device=args.device)
        anchor_item = torch.tensor(dataset.item_list[sample_idx], dtype=torch.long, device=args.device)

        pos_hist_items = torch.tensor(dataset.hist_item_list[sample_idx], dtype=torch.long, device=args.device)
        neg_hist_items_np = dataset.get_histories_for_users_at_times(
            dataset.neg_user_list[sample_idx, 0],
            dataset.event_time_list[sample_idx],
            max_seq_len=args.max_seq_len,
        )
        neg_hist_items = torch.tensor(neg_hist_items_np, dtype=torch.long, device=args.device)

        pos_score = score_pair(model, anchor_item, pos_hist_items, pos_user)
        neg_score = score_pair(model, anchor_item, neg_hist_items, neg_user)

        user_loss = -(F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean())

        optimizer.zero_grad()
        user_loss.backward()
        optimizer.step()

        epoch_user_loss += user_loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] ldr: {epoch_user_loss / batch_num:.4f}")

    if epoch % args.pair_reset_interval == 0:
        print("Reset uniform negative users")
        dataset.get_pair_user_uniform(k=1)

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
