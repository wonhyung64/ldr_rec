#%%
import os
import wandb
import torch
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader

from dataset import DisenData
from model import MF_TPAB, PopPredictor
from utils import parse_args, set_seed, set_device
from procedure import computeTopNAccuracy, evaluate
from loss_function import score_fn, bpr_loss_fn, reg_loss_fn, bootstrap_loss_fn


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"mf_tpab_{expt_num}"
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True) 


wandb_login = False
try:
    wandb_login = wandb.login(key = open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
except:
    pass
if wandb_login:
    configs = vars(args)
    wandb_var = wandb.init(project="pop_shift", config=configs)
    wandb.run.name = args.expt_name



#%%
dataset = DisenData(args)
train_loader = DataLoader(dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)

item_pop = []
for i in range(dataset.m_item):
	item_pop.append(dataset.item_inter[i])
item_pop = torch.Tensor(item_pop)
item_pop = item_pop.to(args.device)

#%%
model = MF_TPAB(args.recdim, dataset.n_users, dataset.m_items, dataset.num_item_pop, dataset.pthres)
model = model.to(args.device)
predictor = PopPredictor(args)
predictor = predictor.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


#%%
best_epoch, best_recall, cnt = 0, 0, 0
for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
	train_loader.dataset.get_pair_bpr()
	epoch_bpr_loss, epoch_reg_loss, epoch_bootstrap_loss, epoch_total_loss = 0., 0., 0., 0.

	for X in train_loader:
		batch_users = X[0].to(args.device)
		batch_pos = X[1].to(args.device)
		batch_neg = X[2].to(args.device)
		batch_stage = X[3].to(args.device)
		batch_pos_inter = torch.stack(X[4])
		batch_pos_inter = batch_pos_inter.to(args.device)
		batch_neg_inter = torch.stack(X[5])
		batch_neg_inter = batch_neg_inter.to(args.device)

		"""embeddings"""
		users_emb, pos_emb, neg_emb, users_pop_emb, pos_pop_emb, neg_pop_emb \
			= model(batch_users, batch_pos, batch_neg, batch_pos_inter, batch_neg_inter, batch_stage, args.period)

		"""scores"""
		pos_scores = score_fn(users_emb, pos_emb) + score_fn(users_pop_emb, pos_pop_emb)
		neg_scores = score_fn(users_emb, neg_emb) + score_fn(users_pop_emb, neg_pop_emb)
		
		"""loss"""
		bpr_loss = bpr_loss_fn(pos_scores, neg_scores) 
		reg_loss = reg_loss_fn([users_emb, pos_emb, neg_emb, users_pop_emb, pos_pop_emb, neg_pop_emb]) / len(batch_users)
		bootstrap_loss = bootstrap_loss_fn(users_emb, pos_emb, neg_emb, users_pop_emb, pos_pop_emb, neg_pop_emb, args.batch_size) * args.lambda1
		total_loss = bpr_loss + reg_loss + bootstrap_loss
				
		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		epoch_bpr_loss += bpr_loss
		epoch_reg_loss += reg_loss
		epoch_bootstrap_loss += bootstrap_loss
		epoch_total_loss += total_loss
	print(f"[Epoch {epoch:>4d} Train Loss] bpr: {epoch_bpr_loss.item():.4f} / reg: {epoch_reg_loss.item():.4f} / bootstrap: {epoch_bootstrap_loss.item():.4f} / total: {epoch_total_loss.item():.4f}")

	if epoch % 5 == 0:
		true_list, pred_list = evaluate(args, "valid", dataset, item_pop, model, predictor)
		valid_results = computeTopNAccuracy(true_list, pred_list, args.topks)

		if wandb_login:
			wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
			wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
			wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
			wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

		if valid_results[1][0] > best_recall:
			best_epoch = epoch
			best_recall = valid_results[1][0]
			true_list, pred_list = evaluate(args, "test", dataset, item_pop, model, predictor)
			test_results = computeTopNAccuracy(true_list, pred_list, args.topks)
			torch.save(model.state_dict(), f"{args.save_path}/{args.expt_name}.pth")

			if wandb_login:
				wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
				wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
				wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
				wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))
				wandb_var.log({"best_epoch": best_epoch, "best_recall": best_recall})

		# early stopping
		if epoch > 100:
			if valid_results[1][0] - best_recall < 1e-4:
				cnt += 1
			else:
				cnt = 1
			if cnt >= 20:
				break

# %%
