#%%
import os
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


from modules.utils import parse_args, set_seed, set_device
from modules.dataset import UserItemTime
from modules.procedure import evaluate, computeTopNAccuracy

"""EXPERIMENT FOR THE SIMPLEST p(u,v|t,H_t) WITH HAWKES PROCESS"""

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)
        return out, user_embed, item_embed


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"user_mf_{expt_num}"
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True) 


wandb_login = False
try:
    wandb_login = wandb.login(key = open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
except:
    pass
if wandb_login:
    configs = vars(args)
    wandb_var = wandb.init(project="ldr_rec", config=configs)
    wandb.run.name = args.expt_name


#%%
dataset = UserItemTime(args)
if args.neg_sampling == "uniform":
	dataset.get_pair_user_event_uniform(args.contrast_size-1)
elif args.neg_sampling == "hardmix":
	dataset.get_pair_user_event_hardmix(args.contrast_size-1, hard_ratio=0.5)

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)

#%%
model_user = MF(dataset.n_user, dataset.m_item, 4)
model_user = model_user.to(args.device)
optimizer_user = optim.Adam(model_user.parameters(), lr=1e-3, weight_decay=args.decay)


#%%
for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model_user.train()

	np.random.shuffle(all_idxs)
	epoch_total_loss = 0.
	for idx in range(batch_num):
		sample_idx = all_idxs[mini_batch*idx : (idx+1)*mini_batch]

		"""USER"""
		anchor_item = torch.tensor(dataset.item_list[sample_idx])
		pos_user = torch.tensor(dataset.pos_user_list[sample_idx]).unsqueeze(-1).to(args.device)
		neg_users = torch.tensor(dataset.neg_user_list[sample_idx]).to(args.device)
		batch_items = anchor_item.repeat_interleave(args.contrast_size).to(args.device)
		batch_users = torch.concat([pos_user, neg_users], -1).reshape([args.batch_size])
		batch_x = torch.stack([batch_users, batch_items], -1)

		batch_scores, _, __ = model_user(batch_x)
		batch_scores = batch_scores.reshape([mini_batch, args.contrast_size])

		user_loss = -F.log_softmax(batch_scores, dim=-1)[:, 0].mean()

		total_loss = user_loss
		total_loss.backward()
		optimizer_user.step()
		optimizer_user.zero_grad()

		epoch_total_loss += total_loss.item()

	print(f"[Epoch {epoch:>4d} Train Loss] total: {epoch_total_loss/batch_num:.4f}")


	if epoch % 10 == 0:
		print("Reset negative pairs")
		if args.neg_sampling == "uniform":
			dataset.get_pair_user_event_uniform(args.contrast_size-1)
		elif args.neg_sampling == "hardmix":
			dataset.get_pair_user_event_hardmix(args.contrast_size-1, hard_ratio=0.5)

	if epoch % args.evaluate_interval == 0:
		pos_scores = []
		lse_scores = []
		u_norms = []
		v_norms = []

		model_user.eval()
		with torch.no_grad():
			U = model_user.user_embedding.weight
			V = model_user.item_embedding.weight
			user_score = torch.matmul(model_user.user_embedding.weight, model_user.item_embedding.weight.T)

		u_norms.append(U.norm(dim=1).mean().item())
		v_norms.append(V.norm(dim=1).mean().item())

		pred_list = []
		gt_list = []
		nll_user_all_list = []

		for i, ((user, item), pos_time) in enumerate((dataset.valid_user_item_time).items()):
			pos_score = user_score[user,item]
			lse_score = torch.logsumexp(user_score[:, v], dim=0)
			full_nll = -(pos_score - lse).item()

			pos_scores.append(pos_score.item())
			lse_scores.append(lse_score.item())
			nll_user_all_list.append(full_nll)

			pred = (user_score[user,:] - torch.logsumexp(user_score, dim=0))
			exclude_items = list(dataset._allPos[user])
			pred[exclude_items] = -(9999)
			_, pred_k = torch.topk(pred, k=max(args.topks))
			pred_list.append(pred_k.cpu())
			gt_list.append([item])

		valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

		print(f"[Epoch {epoch:>4d} Valid full NLL] total: {np.mean(nll_user_all_list):.4f}")

		if wandb_login:

			wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
			wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
			wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
			wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

			wandb_var.log({"train_user_nll_partial": epoch_total_loss / batch_num})
			wandb_var.log({"valid_user_nll_all": np.mean(nll_user_all_list)})

			wandb.log({
				"valid_pos_score_mean": float(np.mean(pos_scores)),
				"valid_logsumexp_mean": float(np.mean(lse_scores)),
				"valid_nll_mean": float(np.mean(lse_scores) - np.mean(pos_scores)),
				"user_emb_norm_mean": float(np.mean(u_norms)),
				"item_emb_norm_mean": float(np.mean(v_norms)),
			})

wandb_var.finish()