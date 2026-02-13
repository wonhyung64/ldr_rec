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

		base_fn = [nn.Linear(embedding_k, 1),]
		self.base_fn = nn.Sequential(*base_fn)

		amplitude_fn = [nn.Linear(embedding_k, 1)]
		self.amplitude_fn = nn.Sequential(*amplitude_fn)

		# self.intensity_decay = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(device)
		self.intensity_decay = nn.Parameter(torch.randn(1))
		self.soft = nn.Softplus()
		self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
		self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)


	def hawkes(self, batch_items, pos_time, batch_time_all):
		item_embed = self.item_embedding(batch_items)
		base = self.soft(self.base_fn(item_embed)).reshape(self.mini_batch, -1)
		amplitude = self.soft(self.amplitude_fn(item_embed)).reshape(self.mini_batch, -1)
		batch_time_mask = batch_time_all < pos_time
		batch_time_delta = pos_time - batch_time_all
		intensity_decay = self.soft(self.intensity_decay)
		time_intensity = torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask
		return base + (time_intensity.sum(-1) * amplitude)

	def mf(self, x):
		user_idx = x[:,0]
		item_idx = x[:,1]
		user_embed = self.user_embedding(user_idx)
		item_embed = self.item_embedding(item_idx)
		out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)
		return out, user_embed, item_embed


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"item_user_share_{expt_num}"
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
model.parameters
model.intensity_decay


#%%
best_recall = 999.
best_item_model = copy.copy(model)
cnt = 1

for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
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
		pos_item = torch.tensor(dataset.pos_item_list[sample_idx]).unsqueeze(-1).to(args.device)
		neg_items = torch.tensor(dataset.neg_item_list[sample_idx]).to(args.device)
		pos_time = torch.tensor(dataset.pos_time_list[sample_idx]).reshape(-1, 1, 1).to(args.device)
		pos_time_all = torch.tensor(dataset.pos_time_all[sample_idx]).to(args.device)
		neg_time_all = torch.tensor(dataset.neg_time_all[sample_idx]).to(args.device)
		batch_items = torch.concat([pos_item, neg_items], -1).reshape([args.batch_size])
		batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

		logits = model.hawkes(batch_items, pos_time, batch_time_all)
		log_logits = torch.log(logits + 1e-9)
		item_loss = -nn.functional.log_softmax(log_logits, dim=-1)[:, 0].mean()

		"""USER"""
		anchor_item = torch.tensor(dataset.item_list[sample_idx])
		pos_user = torch.tensor(dataset.pos_user_list[sample_idx]).unsqueeze(-1).to(args.device)
		neg_users = torch.tensor(dataset.neg_user_list[sample_idx]).to(args.device)
		batch_items = anchor_item.repeat_interleave(args.contrast_size).to(args.device)
		batch_users = torch.concat([pos_user, neg_users], -1).reshape([args.batch_size])
		batch_x = torch.stack([batch_users, batch_items], -1)

		batch_scores, _, __ = model.mf(batch_x)
		batch_scores = batch_scores.reshape([mini_batch, args.contrast_size])
		user_loss = -torch.nn.functional.log_softmax(batch_scores, dim=-1)[:, 0].mean()

		assert torch.isfinite(batch_scores).all()
		assert torch.isfinite(logits).all()

		epoch_item_loss += item_loss
		epoch_user_loss += user_loss
		(user_loss+item_loss).backward()
		optimizer.step()
		optimizer.zero_grad()

	print(f"[Epoch {epoch:>4d} Train Loss] user: {epoch_user_loss.item()/batch_num:.4f} / item: {epoch_item_loss.item()/batch_num:.4f}")

	if epoch % args.evaluate_interval == 0:
		model.eval()

		with torch.no_grad():
			user_logits = torch.matmul(model.user_embedding.weight, model.item_embedding.weight.T)


		sampled_idx = np.random.choice(len(user_logits), args.contrast_size-1)

		base_all = []
		amplitude_all = []
		for idx in range(dataset.m_item//args.batch_size + 1):
			item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
			item_idx = torch.Tensor(item_idx).int().to(args.device)
			with torch.no_grad():
				base = model.soft(model.base_fn(model.item_embedding(item_idx)))
				amplitude = model.soft(model.amplitude_fn(model.item_embedding(item_idx)))
			base_all.append(base)
			amplitude_all.append(amplitude)

		pred_list = []
		gt_list = []
		nll_all_list = []
		nll_partial_list = []
		nll_user_all_list = []
		nll_user_partial_list = []
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
			log_lambda_all = torch.log(logits_all + 1e-9)
			nll_all_list.append(-(log_lambda_all[item] - torch.logsumexp(log_lambda_all, dim=0)))

			log_lambda_pos = log_lambda_all[item]
			log_lambda_neg = log_lambda_all[sampled_idx]
			den = torch.logsumexp(torch.cat([log_lambda_pos.view(1), log_lambda_neg]), dim=0)
			nll_partial_list.append(-(log_lambda_pos - den))

			pos_logit = user_logits[user, item]
			full_nll = -(pos_logit - torch.logsumexp(user_logits[:, item], dim=0)).item()
			nll_user_all_list.append(full_nll)

			neg_logits = user_logits[sampled_idx, item]
			den = torch.logsumexp(torch.cat([pos_logit.view(1), neg_logits]), dim=0)
			partial_nll = -(pos_logit - den).item()
			nll_user_partial_list.append(partial_nll)

			log_p_u_given_v = user_logits[user, :] - torch.logsumexp(user_logits, dim=0)
			pred = log_p_u_given_v.cpu() + log_lambda_all.cpu()
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

			wandb_var.log({"train_item_nll_partial": epoch_item_loss.item() / batch_num})
			wandb_var.log({"valid_item_nll_partial": torch.stack(nll_partial_list).mean().item()})
			wandb_var.log({"valid_item_nll_all": torch.stack(nll_all_list).mean().item()})

			wandb_var.log({"train_user_nll_partial": epoch_user_loss.item() / batch_num})
			wandb_var.log({"valid_user_nll_partial": np.mean(nll_user_partial_list)})
			wandb_var.log({"valid_user_nll_all": np.mean(nll_user_all_list)})

wandb_var.finish()
