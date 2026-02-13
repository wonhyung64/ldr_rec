#%%
import os
import wandb
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from datetime import datetime

from modules.utils import parse_args, set_seed, set_device
from modules.dataset import UserItemTime

"""EXPERIMENT FOR THE SIMPLEST p(v|t,H_t) WITH HAWKES PROCESS"""

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
		self.intensity_decay = nn.Parameter(torch.randn(1))
		self.soft = nn.Softplus()


	def forward(self, batch_items, pos_time, batch_time_all):
		base = self.soft(self.base_fn(batch_items)).reshape(self.mini_batch, -1)
		amplitude = self.soft(self.amplitude_fn(batch_items)).reshape(self.mini_batch, -1)
		batch_time_mask = batch_time_all < pos_time
		batch_time_delta = pos_time - batch_time_all
		intensity_decay = self.soft(self.intensity_decay)
		time_intensity = torch.exp(-intensity_decay * batch_time_delta * batch_time_mask) * batch_time_mask
		return base + (time_intensity.sum(-1) * amplitude)


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"item_{expt_num}"
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

#%%
model = JointRec(dataset.n_user, dataset.m_item, args.recdim, mini_batch, args.device)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.decay)


#%%
for epoch in range(1, args.epochs+1):
	torch.cuda.empty_cache()
	model.train()
	if epoch % 10 == 0:
		print("Reset negative pairs")
		dataset.get_pair_item_bpr(args.contrast_size-1)
		dataset.get_pair_user_bpr(args.contrast_size-1)

	np.random.shuffle(all_idxs)
	epoch_total_loss = 0.
	for idx in range(batch_num):
		""""DATA"""
		sample_idx = all_idxs[mini_batch*idx : (idx+1)*mini_batch]
		pos_item = torch.tensor(dataset.pos_item_list[sample_idx]).unsqueeze(-1).to(args.device)
		neg_items = torch.tensor(dataset.neg_item_list[sample_idx]).to(args.device)
		pos_time = torch.tensor(dataset.pos_time_list[sample_idx]).reshape(-1, 1, 1).to(args.device)
		pos_time_all = torch.tensor(dataset.pos_time_all[sample_idx]).to(args.device)
		neg_time_all = torch.tensor(dataset.neg_time_all[sample_idx]).to(args.device)
	
		batch_items = torch.concat([pos_item, neg_items], -1).reshape([args.batch_size])
		batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

		"""MODEL"""
		logits = model(batch_items, pos_time, batch_time_all)
		total_loss = -torch.log(logits[:,0]/logits.sum(-1)).mean()

		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		epoch_total_loss += total_loss

	print(f"[Epoch {epoch:>4d} Train Loss] total: {epoch_total_loss.item()/batch_num:.4f}")


	if epoch % args.evaluate_interval == 0:
		model.eval()

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


		nll_all_list = []
		nll_partial_list = []
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
			logits_partial = logits_all[np.random.choice(len(logits_all), args.contrast_size-1)]
			nll_all_list.append(-torch.log(logits_all[item] / logits_all.sum()))
			nll_partial_list.append(-torch.log(logits_all[item] / (logits_partial.sum()+logits_all[item])))
			
		print(f"[Epoch {epoch:>4d} Valid NLL] total: {torch.stack(nll_all_list).mean().item():.4f}")
		wandb_var.log({"valid_nll_all": torch.stack(nll_all_list).mean().item()})
		wandb_var.log({"train_nll_partial": epoch_total_loss.item() / batch_num})
		wandb_var.log({"valid_nll_partial": torch.stack(nll_partial_list).mean().item()})

wandb_var.finish()