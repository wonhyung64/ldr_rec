#%%
import os
import wandb
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader

from modules.utils import parse_args, set_seed, set_device
from modules.dataset import DisenData
from modules.procedure import evaluate, computeTopNAccuracy
from modules.model import JointRec


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
dataset = DisenData(args)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

item_pop = []
for i in range(dataset.m_item):
	item_pop.append(dataset.item_inter[i])
item_pop = torch.Tensor(item_pop)
item_pop = item_pop.to(args.device)
train_pop = item_pop[:, :args.period]


#%%
model = JointRec(dataset.n_users, dataset.m_items, dataset.period, args.recdim, train_pop.sum().int().item(), args.device)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

#%%
best_epoch, best_criteria, cnt = 0, 0, 0
for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
	train_loader.dataset.get_pair_bpr_item()
	epoch_hawkes_loss, epoch_utv_loss, epoch_total_loss = 0., 0., 0.

	for X in train_loader:
		batch_items = X[0].to(args.device)
		batch_pos = X[1].to(args.device)
		batch_neg = X[2].to(args.device)
		batch_stage = X[3].to(args.device)


		pos_x = torch.stack([batch_pos, batch_items, batch_stage], -1)
		neg_x = torch.stack([batch_neg, batch_items, batch_stage], -1)
		v = pos_x[:, 1].cpu().numpy()
		t = pos_x[:, 2].cpu().numpy()
		mask = torch.tensor(np.arange(args.period)[None, :] <= t[:, None]).unsqueeze(-1).to(args.device)
		sub_pop = train_pop[v].unsqueeze(-1)

		pos_log_p_utv, log_p_tv, _ = model(pos_x, sub_pop, mask)
		neg_log_p_utv, _, __ = model(neg_x, sub_pop, mask)

		pred = torch.cat([pos_log_p_utv, neg_log_p_utv])
		true = torch.cat([torch.ones_like(pos_log_p_utv), torch.zeros_like(neg_log_p_utv)])

		utv_loss = nn.BCEWithLogitsLoss()(pred, true)
		hawkes_loss = -log_p_tv.mean()

		total_loss = utv_loss + hawkes_loss * args.lambda1
				
		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		epoch_hawkes_loss += hawkes_loss
		epoch_utv_loss += utv_loss
		epoch_total_loss += total_loss
	print(f"[Epoch {epoch:>4d} Train Loss] p_utv: {epoch_utv_loss.item():.4f} / p_tv: {epoch_hawkes_loss.item():.4f} / total: {epoch_total_loss.item():.4f}")


	if epoch % 5 == 0:

		from sklearn.metrics import roc_auc_score

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
