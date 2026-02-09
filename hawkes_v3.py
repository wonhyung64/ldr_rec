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


class JointRec(nn.Module):
	"""Model for Joint Probability P(u,v,t) = P(u|t,v)P(t|v)p(v)"""
	def __init__(self, num_users:int, num_items:int, embedding_k:int, device, depth:int=0):
		super(JointRec, self).__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_k = embedding_k
		self.depth = depth

		self.base_fn = nn.Embedding(num_items, 1)
		self.amplitude_fn = nn.Embedding(num_items, 1)
		self.intensity_decay = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(args.device) # beta
		self.soft = nn.Softplus()


	def forward(self, batch_items, batch_time, batch_time_all):
		batch_time = batch_time
		batch_time_mask = batch_time_all < batch_time
		batch_time_delta = batch_time - batch_time_all
		base = self.soft(self.base_fn(batch_items))
		amplitude = self.soft(self.amplitude_fn(batch_items))
		intensity_decay = self.soft(self.intensity_decay)
		amplitude_sum = (amplitude * torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask).sum(-1)
		return base.squeeze(-1) + amplitude_sum


#%%
		base = soft(base_fn(batch_items.cpu()))
		amplitude = soft(amplitude_fn(batch_items.cpu()))
		intensity_decay = soft(intensity_decay)
		amplitude_sum = (amplitude.cuda() * torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask).sum(-1)



		"""HAWKES PROCESS"""
		item_embed = self.item_embedding(batch_items)

		intensity, compensator = self.hawkes_process(item_embed, batch_time, batch_time_all)
		return intensity, compensator

		base = soft(base_fn(batch_items.cpu()))
		amplitude = soft(amplitude_fn(batch_items.cpu()))
		intensity_decay = soft(intensity_decay)
		amplitude_sum = (amplitude.cuda() * torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask).sum(-1)

		base_fn = nn.Embedding(num_items, 1)
		amplitude_fn = nn.Embedding(num_items, 1)
		intensity_decay = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(args.device) # beta
		soft = nn.Softplus()
		intensity_all = self.soft(base) + (soft(amplitude) * torch.exp(-soft(intensity_decay) * batch_time_delta * batch_time_mask) * batch_time_mask).sum(-1)


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"joint_rec_{expt_num}"
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
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


#%%
model = JointRec(dataset.n_user, dataset.m_item, args.recdim, args.device)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.decay)


#%%
mini_batch = args.batch_size // args.neg_size
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)
dataset.get_pair_item_bpr(args.neg_size-1)
dataset.get_pair_user_bpr(args.neg_size-1)



#%%


for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
	if epoch % 10 == 0:
		dataset.get_pair_item_bpr(args.neg_size-1)
		dataset.get_pair_user_bpr(args.neg_size-1)

	np.random.shuffle(all_idxs)
	epoch_total_loss = 0.

    for idx in range(batch_num):
        sample_idx = all_idxs[mini_batch*idx : (idx+1)*mini_batch]

		pos_item = torch.tensor(np.array(dataset.pos_item_list)[sample_idx]).unsqueeze(-1)
		neg_items = torch.tensor(np.array(dataset.neg_item_list)[sample_idx])
		pos_time = torch.tensor(np.array(dataset.user_time_list)[sample_idx])
		pos_time_all = torch.tensor(np.array(dataset.user_time_all)[sample_idx])

		torch.concat([pos_item, neg_items], -1).flatten()
		pos_item = (X[0]).to(args.device)

		batch_time = (X[3]).to(args.device)/60/60/24
		batch_time_all = (X[4]).to(args.device)/60/60/24
		
		neg_item = 
		
		for i in range(0, item_batch_num+1):
			batch_items = all_item[i*item_batch : (i+1)*item_batch]
			batch_items = torch.tensor(batch_item).to(args.device)
			model(batch_item, batch_time, batch_time_all)

			# base_fn = nn.Embedding(dataset.m_item, 1).to(args.device)
			# amplitude_fn = nn.Embedding(dataset.m_item, 1).to(args.device)
			# intensity_decay = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(args.device) # beta
			# soft = nn.Softplus()

			# batch_time_mask = batch_time_all < batch_time
			# batch_time_delta = batch_time - batch_time_all
			# base = soft(base_fn(batch_items))
			# amplitude = soft(amplitude_fn(batch_items))
			# intensity_decay = soft(intensity_decay)
			# amplitude_sum = (amplitude * torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask).sum(-1)
			# base.squeeze(-1) + amplitude_sum


		tmp[-10:]	
			item_idx = 
			model = 

		batch_time_mask = batch_time_all < batch_time
		batch_time_delta = batch_time_all - batch_time
		batch_time_target = batch_time_all == batch_time
		
		
		sample_idx = all_idxs[idx*args.batch_size : (idx+1)*args.batch_size]
		item_idx = all_idxs[sample_idx]
		batch_time_all = torch.tensor(dataset.item_time_array[item_idx]).to(args.device)
		batch_time_max = torch.where(batch_time_all == 9999999999, -1000, batch_time_all).max(-1).values.unsqueeze(-1)
		batch_items = torch.tensor(item_idx[:1000]).to(args.device)
		batch_time_all = batch_time_all[:1000,:] / 60/60/24
		batch_time_max = batch_time_max[:1000,:] / 60/60/24

		batch_items = X[0].to(args.device)
		batch_pos = X[1].to(args.device)
		batch_neg = X[2].to(args.device)
		batch_stage = X[3].to(args.device)



		intensity, compensator = model.hawkes_process(batch_items, batch_time_max, batch_time_all)

		total_loss = (-intensity + compensator).mean()

		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		epoch_total_loss += total_loss

		if np.isnan(epoch_total_loss.detach().cpu().numpy()):
			break

	if np.isnan(epoch_total_loss.detach().cpu().numpy()):
		break

	print(f"[Epoch {epoch:>4d} Train Loss] total: {epoch_total_loss.item():.4f}")

while True:
	intensity, compensator = model.hawkes_process(batch_items, batch_time_max, batch_time_all)

	total_loss = (-intensity + compensator).mean()

	total_loss.backward()
	optimizer.step()
	optimizer.zero_grad()

	print(f"[Epoch {epoch:>4d} Train Loss] total: {total_loss.item():.4f}")

	"""tmp"""

	model.eval()

	item_embed = model.item_embedding(batch_items)
	batch_time_mask = batch_time_all < batch_time_max
	batch_time_target = batch_time_all == batch_time_max
	batch_time_delta = batch_time_max - (batch_time_all)

	"""compute"""
	base = model.base_fn(item_embed)
	amplitude = model.amplitude_fn(item_embed)
	intensity_all = base + (amplitude * torch.exp(-model.soft(model.intensity_decay) * batch_time_delta * batch_time_mask) * batch_time_mask).cumsum(-1)
	intensity_all = torch.concat([base, intensity_all[:,:-1]], -1)
	"""denorminator"""
	intensities_neg = intensity_all < 0 
	restart_time = batch_time_all + ((torch.where(intensities_neg, base - intensity_all, base) / (base)).log()) / model.soft(model.intensity_decay)
	restart_time = torch.concat([torch.zeros_like(restart_time[:,0:1]), restart_time[:, :-1]], -1)
	dump_time = torch.concat([torch.zeros_like(restart_time[:,0:1]), batch_time_all[:, :-1]], -1)
	valid_compensator = ((batch_time_all > restart_time) & (batch_time_all <= batch_time_max)).int()
	integral_term1 = (base * (batch_time_all - restart_time) * valid_compensator)
	integral_term2 = ((intensity_all - base) / model.soft(model.intensity_decay) * valid_compensator)
	integral_exp_term1 = torch.exp(-model.soft(model.intensity_decay) * (restart_time - dump_time))
	integral_exp_term2 = torch.exp(-model.soft(model.intensity_decay) * (batch_time_all - dump_time))
	integral_exp_term = ((integral_exp_term1 - integral_exp_term2) * valid_compensator)
	compensator = integral_term1 + integral_term2 * integral_exp_term

	u_all = (1 - (-compensator[batch_time_target]).exp()).detach().cpu().numpy()
	ks = stats.kstest(u_all, "uniform") if u_all.size > 0 else None
	print(f"KS : {ks}")



#%%

best_epoch, best_criteria, cnt = 0, 0, 0
for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
	train_loader.dataset.get_pair_bpr()
	epoch_hawkes_loss, epoch_utv_loss, epoch_total_loss = 0., 0., 0.

	for X in train_loader:
		batch_items = X[0].to(args.device)
		batch_pos = X[1].to(args.device)
		batch_neg = X[2].to(args.device)
		batch_time = X[3].to(args.device).unsqueeze(-1) /60/60/24
		batch_time_all = X[4].to(args.device) / 60/60/24

		intensity, compensator = model(batch_items, batch_pos, batch_time, batch_time_all)
		# total_loss = -torch.log((intensity + 1e-9) / (compensator + 1e-9)).mean()
		total_loss = (-intensity + compensator).mean()

		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		# likelihood = (intensity + 1e-9) / (compensator + 1e-9)
		# invalid = likelihood < 0
		# intensity[invalid]
		# compensator[invalid]
		# batch_items[invalid]
		# batch_pos[invalid]

		epoch_total_loss += total_loss

		if np.isnan(epoch_total_loss.detach().cpu().numpy()):
			break

	if np.isnan(epoch_total_loss.detach().cpu().numpy()):
		break

	print(f"[Epoch {epoch:>4d} Train Loss] total: {epoch_total_loss.item():.4f}")

#%%

	if epoch % 5 == 0:


		true_list, pred_list = evaluate(args, "valid", dataset, model, item_pop)

		valid_auc = []
		for i in range(len(true_list)):
			y_true = np.isin(pred_list[i], true_list[i]).astype(int)
			y_score = -np.arange(len(pred_list[i]))
			if y_true.sum():
				valid_auc.append(roc_auc_score(y_true, y_score))
			else:
				valid_auc.append(0)
		valid_auc = np.mean(valid_auc)
		valid_results = computeTopNAccuracy(true_list, pred_list, args.topks)
		print(valid_results)

		if wandb_login:
			wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
			wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
			wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
			wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))
			wandb_var.log({f"valid_auc": valid_auc})

		# if valid_results[1][0] > best_criteria:
		if valid_auc > best_criteria:
			best_epoch = epoch
			# best_crieteria = valid_results[1][0]
			best_crieteria = valid_auc
			true_list, pred_list = evaluate(args, "test", dataset, model, item_pop)
			test_results = computeTopNAccuracy(true_list, pred_list, args.topks)
			torch.save(model.state_dict(), f"{args.save_path}/{args.expt_name}.pth")
			print(test_results)

			if wandb_login:
				wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
				wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
				wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
				wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))
				wandb_var.log({"best_epoch": best_epoch, "best_criteria": best_criteria})

		# early stopping
		# if epoch > 100:
		# 	if valid_results[1][0] - best_recall < 1e-4:
		# 		cnt += 1
		# 	else:
		# 		cnt = 1
		# 	if cnt >= 20:
		# 		break

# %%
