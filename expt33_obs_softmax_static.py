import os
import copy
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from datetime import datetime
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from modules.utils import parse_args, set_seed, set_device
from modules.procedure import computeTopNAccuracy


"""DIRECT OBSERVED CHOICE MODEL WITH STATIC USER-ITEM EMBEDDINGS
p(v|u,t,H_t) \propto exp(f_theta(u,v,t,H_t))
trained by uniform sampled softmax.
"""


class JointRecObsSoftmaxStatic(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_k: int,
        device,
        tau: float = 0.5,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.tau = tau
        self.device = device

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

    def score_items(self, user_idx, item_idx):
        """
        user_idx:
            [B]
        item_idx:
            [B]   -> return [B]
            [B,K] -> return [B,K]
        """
        u = F.normalize(self.user_embedding(user_idx), dim=-1)  # [B, D]

        if item_idx.dim() == 1:
            v = F.normalize(self.item_embedding(item_idx), dim=-1)  # [B, D]
            return torch.sum(u * v, dim=-1) / self.tau
        elif item_idx.dim() == 2:
            v = F.normalize(self.item_embedding(item_idx), dim=-1)  # [B, K, D]
            return torch.einsum("bd,bkd->bk", u, v) / self.tau
        else:
            raise ValueError("item_idx must be [B] or [B, K]")

    def score_all_items(self, user_idx):
        """
        user_idx: [B]
        return: [B, I]
        """
        u = F.normalize(self.user_embedding(user_idx), dim=-1)      # [B, D]
        v_all = F.normalize(self.item_embedding.weight, dim=-1)     # [I, D]
        return torch.matmul(u, v_all.T) / self.tau                  # [B, I]


class UserItemTime(Dataset):
    def __init__(self, args):
        path = f"{args.data_path}/{args.dataset}"
        time_file = path + '/interaction_time_dict.npy'
        train_file = path + '/training_dict.npy'
        valid_file = path + '/validation_dict.npy'
        test_file = path + '/testing_dict.npy'

        self.time_dict = np.load(time_file, allow_pickle=True).item()  # {user:{item:timestamp},}
        self.train_dict = np.load(train_file, allow_pickle=True).item()  # {user:[item,],}
        self.valid_dict = np.load(valid_file, allow_pickle=True).item()
        self.test_dict = np.load(test_file, allow_pickle=True).item()

        self.trainUniqueUsers, self.trainUser, self.trainItem, self.trainDataSize = self.load_set(self.train_dict)
        self.validUniqueUsers, self.validUser, self.validItem, self.validDataSize = self.load_set(self.valid_dict)
        self.testUniqueUsers, self.testUser, self.testItem, self.testDataSize = self.load_set(self.test_dict)

        self.m_item = max(self.trainItem.max(), self.validItem.max(), self.testItem.max()) + 1
        self.n_user = max(self.trainUser.max(), self.validUser.max(), self.testUser.max()) + 1
        self.UserItemNet = csr_matrix(
            (np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )

        self._allPos = self.getUserPosItems(list(range(self.n_user)), self.UserItemNet)
        self._allPosUsers = self.getItemPosUsers(list(range(self.m_item)), self.UserItemNet)

        self.train_user_item_time = self.set_to_pair(self.train_dict, self.time_dict)
        self.valid_user_item_time = self.set_to_pair(self.valid_dict, self.time_dict)
        self.test_user_item_time = self.set_to_pair(self.test_dict, self.time_dict)

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
        UniqueUsers = np.array(UniqueUsers)
        User = np.array(User)
        Item = np.array(Item)
        return UniqueUsers, User, Item, dataSize

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

    def build_event_lists(self):
        self.train_events = []
        for (u, v), t in self.train_user_item_time.items():
            self.train_events.append((u, v, float(t)))
        self.train_events.sort(key=lambda x: x[2])

        self.valid_events = []
        for (u, v), t in self.valid_user_item_time.items():
            self.valid_events.append((u, v, float(t)))
        self.valid_events.sort(key=lambda x: x[2])

        self.test_events = []
        for (u, v), t in self.test_user_item_time.items():
            self.test_events.append((u, v, float(t)))
        self.test_events.sort(key=lambda x: x[2])

        self.user_list = np.array([u for (u, v, t) in self.train_events], dtype=np.int64)
        self.item_list = np.array([v for (u, v, t) in self.train_events], dtype=np.int64)
        self.event_time_list = np.array([t for (u, v, t) in self.train_events], dtype=np.float32)
        self.trainDataSize = len(self.train_events)

    def __len__(self):
        return self.trainDataSize


def sample_uniform_neg_items(pos_items_np, num_items, neg_size):
    """
    pos_items_np: [B] numpy array
    return: [B, neg_size] numpy array
    uniform over {0, ..., num_items-1} \ {pos_item}
    """
    neg_items = np.random.randint(0, num_items - 1, size=(len(pos_items_np), neg_size))
    neg_items += (neg_items >= pos_items_np[:, None])
    return neg_items.astype(np.int64)


# %%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"expt33_obs_softmax_static_{expt_num}"
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)

wandb_login = False
try:
    wandb_login = wandb.login(key=open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
except:
    pass
if wandb_login:
    configs = vars(args)
    wandb_var = wandb.init(project="ldr_rec2", config=configs)
    wandb.run.name = args.expt_name


# %%
dataset = UserItemTime(args)

mini_batch = args.batch_size
neg_size = args.contrast_size - 1
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)

# %%
model = JointRecObsSoftmaxStatic(
    dataset.n_user,
    dataset.m_item,
    args.recdim,
    args.device,
    args.tau,
).to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


# %%
best_valid_score = 0.
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0
cnt = 1

for epoch in range(1, args.epochs + 1):
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(all_idxs)
    epoch_train_nll = 0.

    for idx in range(batch_num):
        sample_idx = all_idxs[mini_batch * idx: (idx + 1) * mini_batch]

        pos_users_np = dataset.user_list[sample_idx]   # [B]
        pos_items_np = dataset.item_list[sample_idx]   # [B]

        neg_items_np = sample_uniform_neg_items(
            pos_items_np=pos_items_np,
            num_items=dataset.m_item,
            neg_size=neg_size,
        )  # [B, M]

        pos_users = torch.tensor(pos_users_np, dtype=torch.long).to(args.device)
        pos_items = torch.tensor(pos_items_np, dtype=torch.long).to(args.device)
        neg_items = torch.tensor(neg_items_np, dtype=torch.long).to(args.device)

        pos_scores = model.score_items(pos_users, pos_items)   # [B]
        neg_scores = model.score_items(pos_users, neg_items)   # [B, M]

        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, 1+M]
        loss = -F.log_softmax(logits, dim=1)[:, 0].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_nll += loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] sampled_softmax_nll: {epoch_train_nll / batch_num:.4f}")

    if epoch % args.evaluate_interval == 0:
        item_nll_list = []
        pred_list = []
        gt_list = []

        model.eval()

        for i, ((user, item), pos_time_val) in enumerate(dataset.valid_user_item_time.items()):
            user_idx = torch.tensor([user], dtype=torch.long).to(args.device)

            with torch.no_grad():
                item_scores = model.score_all_items(user_idx).squeeze(0)   # [I]
                item_log_prob = item_scores - torch.logsumexp(item_scores, dim=0)

            item_nll = -item_log_prob[item].item()
            item_nll_list.append(item_nll)

            pred = item_scores.cpu().clone()
            exclude_items = list(dataset._allPos[user])
            pred[exclude_items] = -9999
            _, pred_k = torch.topk(pred, k=max(args.topks))
            pred_list.append(pred_k.cpu())
            gt_list.append([item])

        valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

        if wandb_login:
            wandb_var.log({
                "train_sampled_softmax_nll": epoch_train_nll / batch_num,
                "valid_item_nll": np.mean(item_nll_list),
            })
            wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
            wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
            wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
            wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

        current_valid_score = valid_results[1][0]

        if current_valid_score - best_valid_score <= 0.:
            cnt += 1
        else:
            best_valid_score = current_valid_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            cnt = 1

        if cnt == 5:
            break


# %%
pred_list = []
gt_list = []
item_nll_list = []

best_model = JointRecObsSoftmaxStatic(
    dataset.n_user,
    dataset.m_item,
    args.recdim,
    args.device,
    args.tau,
).to(args.device)

best_model.load_state_dict(best_state)
best_model.eval()

for i, ((user, item), pos_time_val) in enumerate(dataset.test_user_item_time.items()):
    user_idx = torch.tensor([user], dtype=torch.long).to(args.device)

    with torch.no_grad():
        item_scores = best_model.score_all_items(user_idx).squeeze(0)
        item_log_prob = item_scores - torch.logsumexp(item_scores, dim=0)

    item_nll = -item_log_prob[item].item()
    item_nll_list.append(item_nll)

    pred = item_scores.cpu().clone()
    exclude_items = list(dataset._allPos[user])
    pred[exclude_items] = -9999
    _, pred_k = torch.topk(pred, k=max(args.topks))
    pred_list.append(pred_k.cpu())
    gt_list.append([item])

test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

if wandb_login:
    wandb_var.log({
        "test_item_nll": np.mean(item_nll_list),
        "best_valid_score": best_valid_score,
        "best_epoch": best_epoch,
    })
    wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
    wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
    wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
    wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))
    wandb_var.finish()
