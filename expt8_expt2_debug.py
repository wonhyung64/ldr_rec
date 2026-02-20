#%%
import os
import wandb
import torch
import numpy as np
import torch.nn as nn
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
dataset.get_pair_item_bpr(args.contrast_size-1)

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

		epoch_total_loss += total_loss

	print(f"[Epoch {epoch:>4d} Train Loss] total: {epoch_total_loss.item()/batch_num:.4f}")

	if epoch % 10 == 0:
		print("Reset negative pairs")
		dataset.get_pair_item_bpr(args.contrast_size-1)

	if epoch % args.evaluate_interval == 0:
		model_user.eval()
		with torch.no_grad():
			user_score = torch.matmul(model_user.user_embedding.weight, model_user.item_embedding.weight.T).exp()

		sampled_idx = np.random.choice(len(user_score), args.contrast_size-1)

		pred_list = []
		gt_list = []
		nll_user_all_list = []
		nll_user_partial_list = []
		for i, ((user, item), pos_time) in enumerate((dataset.valid_user_item_time).items()):
			pos_score = user_score[user,item]
			full_nll = -torch.log(pos_score/user_score[:,item].sum()).item()
			partial_nll = -torch.log(pos_score/(user_score[sampled_idx, item].sum()+pos_score)).item()
			nll_user_all_list.append(full_nll)
			nll_user_partial_list.append(partial_nll)

			pred = (user_score.log()[user,:] - torch.log(user_score.sum(0)))
			exclude_items = list(dataset._allPos[user])
			pred[exclude_items] = -(9999)
			_, pred_k = torch.topk(pred, k=max(args.topks))
			pred_list.append(pred_k.cpu())
			gt_list.append([item])

		valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

		print(f"[Epoch {epoch:>4d} Valid full NLL] total: {np.mean(nll_user_all_list):.4f}")
		print(f"[Epoch {epoch:>4d} Valid part NLL] total: {np.mean(nll_user_partial_list):.4f}")

		if wandb_login:

			wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
			wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
			wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
			wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

			wandb_var.log({"train_item_nll_partial": epoch_total_loss.item() / batch_num})
			wandb_var.log({"valid_user_nll_partial": np.mean(nll_user_partial_list)})
			wandb_var.log({"valid_user_nll_all": np.mean(nll_user_all_list)})

wandb_var.finish()