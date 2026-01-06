#%%
import os
import sys
import wandb
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from modules.utils import parse_args, set_seed, set_device
from modules.dataset import DisenData
from modules.procedure import evaluate, computeTopNAccuracy
from modules.model import JointRec


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"ablation_pv_{expt_num}"
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True) 
args.data_path = "../data"



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
testDict = dataset.test_dict
stage_idx_all = torch.tensor([i for i in range(args.period+2)]).to(args.device)
popularity = item_pop[:, :args.period+2]
log_p_v_all = torch.log(popularity.sum(dim=1) / popularity.sum()).unsqueeze(-1)

true_list, pred_list = [], []
for u in range(dataset.n_users):
	"""True Rating"""
	if len(testDict[u]) == 0:
		continue
	true_list.append(testDict[u])
	pred = log_p_v_all.squeeze(-1)


	"""Filtering item history indices"""
	exclude_items = list(dataset._allPos[u])
	valid_items = dataset.getUserValidItems(torch.tensor([u])) # exclude validation items
	exclude_items.extend(valid_items)
	pred[exclude_items] = -(9999)
	_, pred_k = torch.topk(pred.squeeze(-1), k=max(args.topks))
	pred_list.append(pred_k.cpu())
	

test_results = computeTopNAccuracy(true_list, pred_list, args.topks)

if wandb_login:
	wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
	wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
	wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
	wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))

# %%
