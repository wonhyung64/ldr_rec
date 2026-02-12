#%%
import os
import copy
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

class JointRec(nn.Module):
	"""Model for Joint Probability P(u,v,t) = P(u|t,v)P(t|v)p(v)"""
	def __init__(self, num_users:int, num_items:int, embedding_k:int, mini_batch, device, depth:int=0):
		super(JointRec, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_k = embedding_k
		self.depth = depth
		self.mini_batch = mini_batch
		self.base_fn = nn.Embedding(num_items, 1)
		self.amplitude_fn = nn.Embedding(num_items, 1)
		self.intensity_decay = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(device)
		self.soft = nn.Softplus()
		self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
		self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)


	def forward(self, batch_items, pos_time, batch_time_all):
		"""intensity"""
		base = self.soft(self.base_fn(batch_items)).reshape(self.mini_batch, -1)
		amplitude = self.soft(self.amplitude_fn(batch_items)).reshape(self.mini_batch, -1)
		batch_time_mask = batch_time_all < pos_time
		batch_time_delta = pos_time - batch_time_all
		intensity_decay = self.soft(self.intensity_decay)
		time_intensity = torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask

		"""score"""

		return base + (time_intensity.sum(-1) * amplitude)


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
args.expt_name = f"item_user_earlynll_{expt_num}"
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
dataset.get_pair_user_bpr(args.contrast_size-1)

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)
all_item_idxs = np.arange(dataset.m_item)
all_user_idxs = np.arange(dataset.n_user)

#%%
model = JointRec(dataset.n_user, dataset.m_item, args.recdim, mini_batch, args.device)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.decay)

model_user = MF(dataset.n_user, dataset.m_item, 4)
model_user = model_user.to(args.device)
optimizer_user = optim.Adam(model_user.parameters(), lr=1e-3, weight_decay=args.decay)


#%%
best_user = 999.
best_item = 999.
best_user_model = copy.copy(model_user)
best_item_model = copy.copy(model)
user_train = 1
item_train = 1
item_cnt = 1
user_cnt = 1

for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
	model_user.train()
	if epoch % 10 == 0:
		print("Reset negative pairs")
		dataset.get_pair_item_bpr(args.contrast_size-1)
		dataset.get_pair_user_bpr(args.contrast_size-1)

	np.random.shuffle(all_idxs)
	epoch_item_loss = 0.
	epoch_user_loss = 0.

	
	for idx in range(batch_num):
		sample_idx = all_idxs[mini_batch*idx : (idx+1)*mini_batch]

		""""ITEM"""
		if item_train:
			pos_item = torch.tensor(dataset.pos_item_list[sample_idx]).unsqueeze(-1).to(args.device)
			neg_items = torch.tensor(dataset.neg_item_list[sample_idx]).to(args.device)
			pos_time = torch.tensor(dataset.pos_time_list[sample_idx]).reshape(-1, 1, 1).to(args.device)
			pos_time_all = torch.tensor(dataset.pos_time_all[sample_idx]).to(args.device)
			neg_time_all = torch.tensor(dataset.neg_time_all[sample_idx]).to(args.device)
			batch_items = torch.concat([pos_item, neg_items], -1).reshape([args.batch_size])
			batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

			logits = model(batch_items, pos_time, batch_time_all)
			item_loss = -torch.log(logits[:,0]/logits.sum(-1)).mean()

			item_loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			epoch_item_loss += item_loss

		"""USER"""
		if user_train:
			anchor_item = torch.tensor(dataset.item_list[sample_idx])
			pos_user = torch.tensor(dataset.pos_user_list[sample_idx]).unsqueeze(-1).to(args.device)
			neg_users = torch.tensor(dataset.neg_user_list[sample_idx]).to(args.device)
			batch_items = anchor_item.repeat_interleave(args.contrast_size).to(args.device)
			batch_users = torch.concat([pos_user, neg_users], -1).reshape([args.batch_size])
			batch_x = torch.stack([batch_users, batch_items], -1)

			batch_scores, _, __ = model_user(batch_x)
			batch_scores = batch_scores.reshape([mini_batch, args.contrast_size])
			user_loss = -torch.log(batch_scores[:,0].exp() / batch_scores.exp().sum(-1)).mean()

			user_loss.backward()
			optimizer_user.step()
			optimizer_user.zero_grad()
			epoch_user_loss += user_loss

	print(f"[Epoch {epoch:>4d} Train Loss] user: {epoch_user_loss.item()/batch_num:.4f} / item: {epoch_item_loss.item()/batch_num:.4f}")

	if (epoch % args.evaluate_interval == 0) & ((user_train) | (item_train)):
		model.eval()
		model_user.eval()

		with torch.no_grad():
			user_score = torch.matmul(model_user.user_embedding.weight, model_user.item_embedding.weight.T)
		user_score = user_score.exp()

		sampled_idx = np.random.choice(len(user_score), args.contrast_size-1)

		base_all = []
		amplitude_all = []
		for idx in range(dataset.m_item//args.batch_size + 1):
			item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
			item_idx = torch.Tensor(item_idx).int().to(args.device)
			with torch.no_grad():
				base = model.soft(model.base_fn(item_idx))
				amplitude = model.soft(model.amplitude_fn(item_idx))
			base_all.append(base)
			amplitude_all.append(amplitude)

		pred_list = []
		gt_list = []
		nll_all_list = []
		nll_user_all_list = []
		for i, ((user, item), pos_time) in enumerate((dataset.valid_user_item_time).items()):
			logits_all = []
			pos_time = torch.Tensor([pos_time]).to(args.device)
			for idx in range(dataset.m_item//args.batch_size + 1):
				item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
				batch_time_all = torch.Tensor(dataset.item_time_array[item_idx]).to(args.device)
				batch_time_mask = batch_time_all < pos_time
				batch_time_delta = pos_time - batch_time_all
				item_idx = torch.Tensor(item_idx).int().to(args.device)
				with torch.no_grad():
					intensity_decay = model.soft(model.intensity_decay)
					time_intensity = (torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask).sum(-1, keepdim=True)
					logits = (base_all[idx] + amplitude_all[idx] * time_intensity).flatten().cpu()
				logits_all.append(logits)
			logits_all = torch.concat(logits_all)
			nll_all_list.append(-torch.log(logits_all[item] / logits_all.sum()))

			pos_score = user_score[user,item]
			full_nll = -torch.log(pos_score/user_score[:,item].sum()).item()
			nll_user_all_list.append(full_nll)

		item_nll = torch.stack(nll_all_list).mean().item()
		user_nll = np.mean(nll_user_all_list)

		if wandb_login:
			wandb_var.log({"valid_item_nll_all": item_nll})
			wandb_var.log({"valid_user_nll_all": user_nll})

		if item_nll - best_item > 0.:
			item_cnt += 1
		else:
			best_item_model = copy.copy(model)
			best_item = item_nll
			item_cnt = 1
		if item_cnt >= 5:
			item_train = 0
			break

		if user_nll - best_user > 0.:
			user_cnt += 1
		else:
			best_user_model = copy.copy(model_user)
			best_user = user_nll
			user_cnt = 1
		if user_cnt >= 5:
			user_train = 0
			break

		if (user_cnt == 1) | (item_cnt == 1):

			best_user_model.eval()
			best_item_model.eval()

			with torch.no_grad():
				user_score = torch.matmul(best_user_model.user_embedding.weight, best_user_model.item_embedding.weight.T)
			user_score = user_score.exp()

			base_all = []
			amplitude_all = []
			for idx in range(dataset.m_item//args.batch_size + 1):
				item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
				item_idx = torch.Tensor(item_idx).int().to(args.device)
				with torch.no_grad():
					base = best_item_model.soft(best_item_model.base_fn(item_idx))
					amplitude = best_item_model.soft(best_item_model.amplitude_fn(item_idx))
				base_all.append(base)
				amplitude_all.append(amplitude)

			pred_list = []
			gt_list = []
			for i, ((user, item), pos_time) in enumerate((dataset.valid_user_item_time).items()):
				logits_all = []
				pos_time = torch.Tensor([pos_time]).to(args.device)
				for idx in range(dataset.m_item//args.batch_size + 1):
					item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
					batch_time_all = torch.Tensor(dataset.item_time_array[item_idx]).to(args.device)
					batch_time_mask = batch_time_all < pos_time
					batch_time_delta = pos_time - batch_time_all
					item_idx = torch.Tensor(item_idx).int().to(args.device)
					with torch.no_grad():
						intensity_decay = best_item_model.soft(best_item_model.intensity_decay)
						time_intensity = (torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask).sum(-1, keepdim=True)
						logits = (base_all[idx] + amplitude_all[idx] * time_intensity).flatten().cpu()
					logits_all.append(logits)
				logits_all = torch.concat(logits_all)

				pred = user_score[user,:].log().cpu() - torch.log(user_score.sum(0)).cpu() + logits_all.log()
				exclude_items = list(dataset._allPos[user])
				pred[exclude_items] = -(9999)
				_, pred_k = torch.topk(pred.squeeze(-1), k=max(args.topks))
				pred_list.append(pred_k.cpu())
				gt_list.append([item])

				valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

				if wandb_login:

					wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
					wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
					wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
					wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))


wandb_var.finish()
