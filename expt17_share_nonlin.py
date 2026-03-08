#%%
import os
import copy
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

class JointRec(nn.Module):
	"""Model for Joint Probability P(u,v,t) = P(u|t,v)P(t|v)p(v)"""
	def __init__(self, num_users:int, num_items:int, embedding_k:int, mini_batch, device, depth:int=0, tau:float=0.5):
		super(JointRec, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_k = embedding_k
		self.depth = depth
		self.mini_batch = mini_batch
		self.soft = nn.Softplus()
		self.tau = tau

		self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
		self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

		base_fn = [nn.Linear(embedding_k, embedding_k//2), nn.Softplus()]
		for _ in range(depth):
			base_fn.append(nn.Linear(embedding_k//2, embedding_k//2))
			base_fn.append(nn.Softplus())
		base_fn.append(nn.Linear(embedding_k//2, 1, bias=False))
		self.base_fn = nn.Sequential(*base_fn)

		amplitude_fn = [nn.Linear(embedding_k, embedding_k//2), nn.Softplus()]
		for _ in range(depth):
			amplitude_fn.append(nn.Linear(embedding_k//2, embedding_k//2))
			amplitude_fn.append(nn.Softplus())
		amplitude_fn.append(nn.Linear(embedding_k//2, 1, bias=False))
		self.amplitude_fn = nn.Sequential(*amplitude_fn)

		decay_fn = [nn.Linear(embedding_k, embedding_k//2), nn.Softplus()]
		for _ in range(depth):
			decay_fn.append(nn.Linear(embedding_k//2, embedding_k//2))
			decay_fn.append(nn.Softplus())
		decay_fn.append(nn.Linear(embedding_k//2, 1, bias=False))
		self.decay_fn = nn.Sequential(*decay_fn)

	def popularity(self, batch_items, pos_time, batch_time_all):
		item_embed = F.normalize(self.item_embedding(batch_items), dim=-1)
		base = self.soft(self.base_fn(item_embed)).reshape(self.mini_batch, -1)
		amplitude = self.soft(self.amplitude_fn(item_embed)).reshape(self.mini_batch, -1)
		batch_time_mask = batch_time_all < pos_time
		batch_time_delta = (pos_time - batch_time_all).clamp(0.0)
		intensity_decay = self.soft(self.decay_fn(item_embed)).reshape(self.mini_batch, -1)
		time_intensity = torch.exp(-intensity_decay.unsqueeze(-1) * batch_time_delta) * batch_time_mask
		return base + (time_intensity.sum(-1) * amplitude), time_intensity, base, amplitude

	def interaction(self, x):
		user_idx = x[:,0]
		item_idx = x[:,1]
		user_embed = F.normalize(self.user_embedding(user_idx), dim=-1)
		item_embed = F.normalize(self.item_embedding(item_idx), dim=-1)
		out = torch.sum(user_embed.mul(item_embed), -1, keepdim=True) / self.tau
		return out, user_embed, item_embed


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"expt17_share_nonlin_{expt_num}"
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
dataset.get_pair_item_event_uniform(args.contrast_size-1)
if args.neg_sampling == "uniform":
	dataset.get_pair_user_event_uniform(args.contrast_size-1)
elif args.neg_sampling == "hardmix":
	dataset.get_pair_user_event_hardmix(args.contrast_size-1, hard_ratio=0.5)

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)
all_item_idxs = np.arange(dataset.m_item)
all_user_idxs = np.arange(dataset.n_user)

#%%
model = JointRec(dataset.n_user, dataset.m_item, args.recdim, mini_batch, args.device, args.depth, args.tau)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

#%%
best_joint_nll = 999.
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0
cnt = 1

for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
	np.random.shuffle(all_idxs)
	epoch_item_loss = 0.
	epoch_user_loss = 0.
	epoch_time_intensity = 0.

	for idx in range(batch_num):
		sample_idx = all_idxs[mini_batch*idx : (idx+1)*mini_batch]

		""""ITEM"""
		pos_item = torch.tensor(dataset.pos_item_list[sample_idx]).unsqueeze(-1).to(args.device)
		neg_items = torch.tensor(dataset.neg_item_list[sample_idx]).to(args.device)
		pos_time = torch.Tensor(dataset.pos_time_list[sample_idx]).reshape(-1, 1, 1).to(args.device)
		pos_time_all = torch.Tensor(dataset.pos_time_all[sample_idx]).to(args.device)
		neg_time_all = torch.Tensor(dataset.neg_time_all[sample_idx]).to(args.device)

		batch_items = torch.concat([pos_item, neg_items], -1).reshape([args.batch_size])
		batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

		logits, time_intensity, base, amplitude = model.popularity(batch_items, pos_time, batch_time_all)
		log_logits = torch.log(logits + 1e-9)
		item_loss = -nn.functional.log_softmax(log_logits, dim=-1)[:, 0].mean()
		batch_intensity = (time_intensity[:,0,:].sum(-1) / (time_intensity[:,0,:] != 0).sum(-1).clamp(1)).mean()

		"""USER"""
		anchor_item = torch.tensor(dataset.item_list[sample_idx])
		pos_user = torch.tensor(dataset.pos_user_list[sample_idx]).unsqueeze(-1).to(args.device)
		neg_users = torch.tensor(dataset.neg_user_list[sample_idx]).to(args.device)
		batch_items = anchor_item.repeat_interleave(args.contrast_size).to(args.device)
		batch_users = torch.concat([pos_user, neg_users], -1).reshape([args.batch_size])
		batch_x = torch.stack([batch_users, batch_items], -1)

		batch_scores, _, __ = model.interaction(batch_x)
		batch_scores = batch_scores.reshape([mini_batch, args.contrast_size])
		user_loss = -F.log_softmax(batch_scores, dim=-1)[:, 0].mean()

		total_loss = user_loss * args.lambda1 + item_loss * (1-args.lambda1)
		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		epoch_item_loss += item_loss.item()
		epoch_user_loss += user_loss.item()
		epoch_time_intensity += batch_intensity.item()

	print(f"[Epoch {epoch:>4d} Train Loss] user: {epoch_user_loss/batch_num:.4f} / item: {epoch_item_loss/batch_num:.4f}")

	if epoch % args.pair_reset_interval == 0:
		print("Reset negative pairs")
		dataset.get_pair_item_event_uniform(args.contrast_size-1)
		if args.neg_sampling == "uniform":
			dataset.get_pair_user_event_uniform(args.contrast_size-1)
		elif args.neg_sampling == "hardmix":
			dataset.get_pair_user_event_hardmix(args.contrast_size-1, hard_ratio=0.5)

	if epoch % args.evaluate_interval == 0:
		user_pos_score_list = []
		user_lse_score_list = []
		user_nll_list = []
		item_nll_list = []
		joint_nll_list = []
		pred_list = []
		gt_list = []

		model.eval()

		with torch.no_grad():
			user_embed = F.normalize(model.user_embedding.weight, dim=-1)
			item_embed = F.normalize(model.item_embedding.weight, dim=-1)
			user_score = torch.matmul(user_embed, item_embed.T) / model.tau

		item_decay_all, item_base_all, item_amplitude_all = [], [], []
		for idx in range(dataset.m_item//args.batch_size + 1):
			item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
			sub_item_embed = item_embed[item_idx]
			with torch.no_grad():
				intensity_decay = model.soft(model.decay_fn(sub_item_embed))
				base = model.soft(model.base_fn(sub_item_embed))
				amplitude = model.soft(model.amplitude_fn(sub_item_embed))
			item_decay_all.append(intensity_decay)
			item_base_all.append(base)
			item_amplitude_all.append(amplitude)

		for i, ((user, item), pos_time) in enumerate((dataset.valid_user_item_time).items()):

			item_logits_list = []
			pos_time = torch.Tensor([pos_time]).to(args.device)
			for idx in range(dataset.m_item//args.batch_size + 1):
				item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
				batch_time_all = torch.Tensor(dataset.item_time_array[item_idx]).to(args.device)
				batch_time_mask = batch_time_all < pos_time
				batch_time_delta = (pos_time - batch_time_all).clamp(min=0.0)
				item_idx = torch.Tensor(item_idx).int().to(args.device)
				time_intensity = (torch.exp(-item_decay_all[idx] * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
				logits = (item_base_all[idx] + item_amplitude_all[idx] * time_intensity).flatten()
				item_logits_list.append(logits)
			item_logits = torch.concat(item_logits_list)
			item_log_prob = torch.log(item_logits / item_logits.sum())

			user_lse_score = torch.logsumexp(user_score, dim=0)
			user_log_prob = user_score - user_lse_score.unsqueeze(0)

			item_nll = -item_log_prob[item].item()
			user_nll = -user_log_prob[user,item].item()
			joint_nll = item_nll + user_nll

			user_pos_score_list.append(user_score[user,item].item())
			user_lse_score_list.append(user_lse_score[item].item())
			item_nll_list.append(item_nll)
			user_nll_list.append(user_nll)
			joint_nll_list.append(joint_nll)

			pred = (user_log_prob[user] + item_log_prob).cpu()
			exclude_items = list(dataset._allPos[user])
			pred[exclude_items] = -(9999)
			_, pred_k = torch.topk(pred, k=max(args.topks))
			pred_list.append(pred_k.cpu())
			gt_list.append([item])

		valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

		if wandb_login:
			wandb_var.log({
				"valid_item_nll": np.mean(item_nll_list),
				"valid_user_nll": np.mean(user_nll_list),
				"valid_joint_nll": np.mean(joint_nll_list),
				"valid_pos_score_mean": float(np.mean(user_pos_score_list)),
				"valid_logsumexp_mean": float(np.mean(user_lse_score_list)),
				"train_item_nll": epoch_item_loss/batch_num,
				"train_user_nll": epoch_user_loss/batch_num,
				"train_time_intensity": epoch_time_intensity/batch_num,
				"user_emb_norm_mean": user_embed.mean().item(),
				"item_emb_norm_mean": item_embed.mean().item(),
				})

			wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
			wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
			wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
			wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

		if np.mean(joint_nll_list) - best_joint_nll > 0.:
			cnt += 1
		else:
			best_model = copy.copy(model)
			best_state = copy.deepcopy(model.state_dict())
			best_epoch = epoch
			cnt = 1

		if cnt == 5:
			break


#%%
pred_list = []
gt_list = []
user_nll_list = []
item_nll_list = []
joint_nll_list = []

best_model = JointRec(dataset.n_user, dataset.m_item, args.recdim, mini_batch, args.device, args.depth, args.tau)
best_model.load_state_dict(best_state)
best_model.eval()

with torch.no_grad():
	user_embed = F.normalize(best_model.user_embedding.weight, dim=-1)
	item_embed = F.normalize(best_model.item_embedding.weight, dim=-1)
	user_score = torch.matmul(user_embed, item_embed.T) / best_model.tau

item_decay_all, item_base_all, item_amplitude_all = [], [], []
for idx in range(dataset.m_item//args.batch_size + 1):
	item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
	sub_item_embed = item_embed[item_idx]
	with torch.no_grad():
		intensity_decay = model.soft(model.decay_fn(sub_item_embed))
		base = best_model.soft(best_model.base_fn(sub_item_embed))
		amplitude = best_model.soft(best_model.amplitude_fn(sub_item_embed))
	item_decay_all.append(intensity_decay)
	item_base_all.append(base)
	item_amplitude_all.append(amplitude)


for i, ((user, item), pos_time) in enumerate((dataset.test_user_item_time).items()):

	item_logits_list = []
	pos_time = torch.Tensor([pos_time]).to(args.device)
	for idx in range(dataset.m_item//args.batch_size + 1):
		item_idx = all_item_idxs[idx*args.batch_size: (idx+1)*args.batch_size]
		batch_time_all = torch.Tensor(dataset.item_time_array[item_idx]).to(args.device)
		batch_time_mask = batch_time_all < pos_time
		batch_time_delta = (pos_time - batch_time_all).clamp(min=0.0)
		item_idx = torch.Tensor(item_idx).int().to(args.device)
		time_intensity = (torch.exp(-item_decay_all[idx] * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
		logits = (item_base_all[idx] + item_amplitude_all[idx] * time_intensity).flatten()
		item_logits_list.append(logits)
	item_logits = torch.concat(item_logits_list)
	item_log_prob = torch.log(item_logits / item_logits.sum())

	user_lse_score = torch.logsumexp(user_score, dim=0)
	user_log_prob = user_score - user_lse_score.unsqueeze(0)

	item_nll = -item_log_prob[item].item()
	user_nll = -user_log_prob[user,item].item()
	joint_nll = item_nll + user_nll

	item_nll_list.append(item_nll)
	user_nll_list.append(user_nll)
	joint_nll_list.append(joint_nll)
	
	pred = (user_log_prob[user] + item_log_prob).cpu()
	exclude_items = list(dataset._allPos[user])
	valid_items = dataset.getUserValidItems(torch.tensor([user]), dataset.valid_dict)
	exclude_items.extend(valid_items)
	pred[exclude_items] = -(9999)

	_, pred_k = torch.topk(pred, k=max(args.topks))
	pred_list.append(pred_k.cpu())
	gt_list.append([item])

test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

if wandb_login:
	wandb_var.log({
		"test_item_nll": np.mean(item_nll_list),
		"test_user_nll": np.mean(user_nll_list),
		"test_joint_nll": np.mean(joint_nll_list),
		})

	wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
	wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
	wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
	wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))

	wandb_var.log({"best_joint_nll": best_joint_nll})
	wandb_var.log({"best_epoch": best_epoch})

	wandb_var.finish()
