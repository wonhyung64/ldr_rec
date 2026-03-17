#%%
import os
import math
import copy
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from datetime import datetime
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from modules.utils import parse_args, set_seed, set_device
from modules.procedure import computeTopNAccuracy

"""EXPERIMENT FOR THE SIMPLEST p(u,v|t,H_t) WITH HAWKES PROCESS"""

class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int, max_bucket: int = 128):
        super().__init__()
        self.max_bucket = max_bucket
        self.emb = nn.Embedding(max_bucket + 1, d_model)

    def bucketize(self, delta_t: torch.Tensor):
        # delta_t >= 0
        # log bucketization
        bucket = torch.floor(torch.log1p(delta_t)).long()
        bucket = torch.clamp(bucket, min=0, max=self.max_bucket)
        return bucket

    def forward(self, delta_t: torch.Tensor):
        bucket = self.bucketize(delta_t)
        return self.emb(bucket)


class JointRecTransformer(nn.Module):
	def __init__(self, num_users: int, num_items: int, embedding_k: int, mini_batch: int, device, depth: int = 0, tau: float = 0.5, max_seq_len: int = 50, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,):
		super().__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_k = embedding_k
		self.depth = depth
		self.mini_batch = mini_batch
		self.soft = nn.Softplus()
		self.tau = tau
		self.max_seq_len = max_seq_len
		self.device = device
		self.padding_item_id = self.num_items
		self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_k, padding_idx=self.padding_item_id)

		base_fn = [nn.Linear(embedding_k, embedding_k // 2), nn.Softplus()]
		for _ in range(depth):
			base_fn += [nn.Linear(embedding_k // 2, embedding_k // 2), nn.Softplus()]
		base_fn += [nn.Linear(embedding_k // 2, 1, bias=False)]
		self.base_fn = nn.Sequential(*base_fn)
		
		amplitude_fn = [nn.Linear(embedding_k, embedding_k // 2), nn.Softplus()]
		for _ in range(depth):
			amplitude_fn += [nn.Linear(embedding_k//2, embedding_k//2), nn.Softplus()]
		amplitude_fn += [nn.Linear(embedding_k // 2, 1, bias=False)]
		self.amplitude_fn = nn.Sequential(*amplitude_fn)
		self.intensity_decay = nn.Parameter(torch.randn(1))

        # -------- residual / Transformer side --------
		self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
		self.pos_embedding = nn.Embedding(self.max_seq_len, self.embedding_k)
		self.time_embedding = TimeEmbedding(self.embedding_k, max_bucket=128)

		enc_layer = nn.TransformerEncoderLayer(
			d_model=self.embedding_k,
			nhead=n_heads,
			dim_feedforward=self.embedding_k * 4,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
        )
		self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

		self.user_proj = nn.Linear(self.embedding_k, self.embedding_k, bias=False)
		self.item_proj = nn.Linear(self.embedding_k, self.embedding_k, bias=False)

		self.layernorm = nn.LayerNorm(self.embedding_k)
		self.dropout = nn.Dropout(dropout)

    # ---------- prior ----------
	def popularity(self, batch_items, pos_time, batch_time_all):
		item_embed = F.normalize(self.item_embedding(batch_items), dim=-1)
		base = self.soft(self.base_fn(item_embed)).reshape(self.mini_batch, -1)
		amplitude = self.soft(self.amplitude_fn(item_embed)).reshape(self.mini_batch, -1)

		batch_time_mask = batch_time_all < pos_time
		batch_time_delta = (pos_time - batch_time_all).clamp(0.0)
		intensity_decay = self.soft(self.intensity_decay)
		time_intensity = torch.exp(-intensity_decay * batch_time_delta) * batch_time_mask

		return base + (time_intensity.sum(-1) * amplitude), time_intensity, base, amplitude

	def encode_user_history(self, user_idx, hist_items, hist_times, query_time):
		"""
		user_idx:    [B]
		hist_items:  [B, L]   padded item ids
		hist_times:  [B, L]   padded timestamps
		query_time:  [B]
		"""
		B, L = hist_items.shape
		device = hist_items.device

		# padding mask for history tokens
		hist_pad_mask = (hist_times <= 0)  # [B, L]

		# history token embeddings
		item_emb = self.item_embedding(hist_items)  # [B, L, D]

		pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
		pos_emb = self.pos_embedding(torch.clamp(pos_ids, max=self.max_seq_len - 1))

		delta_t = (query_time.unsqueeze(1) - hist_times).clamp(min=0.0)
		time_emb = self.time_embedding(delta_t)

		hist_x = item_emb + pos_emb + time_emb  # [B, L, D]

		# prepend a user token so that every sequence has at least one valid token
		user_tok = self.user_embedding(user_idx).unsqueeze(1)  # [B, 1, D]

		x = torch.cat([user_tok, hist_x], dim=1)  # [B, 1+L, D]
		x = self.layernorm(x)
		x = self.dropout(x)

		# prepend False for user token: it is never masked
		user_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
		pad_mask = torch.cat([user_mask, hist_pad_mask], dim=1)  # [B, 1+L]

		h = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, 1+L, D]

		# use the user token output as the final user state
		h_user = h[:, 0, :]  # [B, D]
		q_u = self.user_proj(h_user)  # [B, D]

		return q_u

	def residual_score(self, user_idx, item_idx, hist_items, hist_times, query_time):
		q_u = self.encode_user_history(user_idx, hist_items, hist_times, query_time)
		k_v = self.item_proj(self.item_embedding(item_idx))
		return torch.sum(q_u * k_v, dim=-1, keepdim=True), q_u, k_v

	def score_all_items(self, user_idx, hist_items, hist_times, query_time):
		q_u = self.encode_user_history(user_idx, hist_items, hist_times, query_time)   # [B, D]
		k_all = self.item_proj(self.item_embedding.weight)                              # [I, D]
		return torch.matmul(q_u, k_all.T) / self.tau


class UserItemTime(Dataset):
	def __init__(self, args):
		path = f"{args.data_path}/{args.dataset}"
		time_file = path + '/interaction_time_dict.npy'
		train_file = path + '/training_dict.npy'
		valid_file = path + '/validation_dict.npy'
		test_file = path + '/testing_dict.npy'

		self.time_dict = np.load(time_file, allow_pickle=True).item() # {user:{item:timestamp},}
		self.train_dict = np.load(train_file, allow_pickle=True).item() # {user:[item,],}
		self.valid_dict = np.load(valid_file, allow_pickle=True).item()
		self.test_dict = np.load(test_file, allow_pickle=True).item()

		self.trainUniqueUsers, self.trainUser, self.trainItem, self.trainDataSize = self.load_set(self.train_dict)
		self.validUniqueUsers, self.validUser, self.validItem, self.validDataSize = self.load_set(self.valid_dict)
		self.testUniqueUsers, self.testUser, self.testItem, self.testDataSize = self.load_set(self.test_dict)

		self.m_item = max(self.trainItem.max(), self.validItem.max(), self.testItem.max()) + 1
		self.n_user = max(self.trainUser.max(), self.validUser.max(), self.testUser.max()) + 1
		self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
		# all (user, item) matrix of train, if interacted 1.

		# pre-calculate
		self._allPos = self.getUserPosItems(list(range(self.n_user)), self.UserItemNet)
		self._allPosUsers = self.getItemPosUsers(list(range(self.m_item)), self.UserItemNet)

		self.train_user_item_time = self.set_to_pair(self.train_dict, self.time_dict)
		self.valid_user_item_time = self.set_to_pair(self.valid_dict, self.time_dict)
		self.test_user_item_time = self.set_to_pair(self.test_dict, self.time_dict)
		self.item_time_array = self.time_dict_to_array(self.time_dict)
		self.train_item_time_array = self.train_time_dict_to_array(self.time_dict, self.train_dict)

		# ---- new: per-user sorted interaction history (days) ----
		self.user_interactions = self.build_user_interactions(self.time_dict)

		# ---- new: train/valid/test event lists for sequence models ----
		self.build_event_lists()

	def load_set(self, set_dict):
		UniqueUsers, Item, User = [], [], []
		dataSize = 0
		for uid in set_dict.keys():
			if len(set_dict[uid]) != 0:
				UniqueUsers.append(uid)
				User.extend([uid] * len(set_dict[uid]))
				Item.extend(set_dict[uid])
				dataSize += len(set_dict[uid])
		UniqueUsers = np.array(UniqueUsers)
		User = np.array(User)  # [interact1_user, interact2_user, interact3_user, ...]
		Item = np.array(Item)  # [interact1_item, interact2_item, interact3_item, ...]
		return UniqueUsers, User, Item, dataSize

	def set_to_pair(self, set_dict, time_dict):
		user_item_time = {}
		for user in set_dict:
			for item in set_dict[user]:
				time = time_dict[user][item] / 60 / 60 / 24
				user_item_time[(user, item)] = time
		return user_item_time

	def time_dict_to_array(self, time_dict):
		max_pos, item_time_dict, item_time_array = 0, {}, []
		for _, user_dict in time_dict.items():
			for item_idx, times in user_dict.items():
				try:
					assert item_time_dict[item_idx]
				except:
					item_time_dict[item_idx] = []
				item_time_dict[item_idx].append(times)
		max_time = 0.
		for i in range(max(item_time_dict.keys()) + 1):
			try:
				times = np.array(item_time_dict[i]) / 60 / 60 / 24
				max_time = max(np.max(times), max_time)
			except:
				times = np.array([])
			max_pos = max(max_pos, len(times))
			times.sort()
			item_time_array.append(times)
		self.max_pos = max_pos
		self.max_time = max_time + 1
		for i in range(max(item_time_dict.keys()) + 1):
			item_time_array[i] = np.pad(item_time_array[i], (0, max_pos - len(item_time_array[i])), "constant", constant_values=self.max_time)
		item_time_array = np.stack(item_time_array, 0)
		return item_time_array

	def train_time_dict_to_array(self, time_dict, train_dict):
		item_time_dict = {i: [] for i in range(self.m_item)}

		for u, items in train_dict.items():
			if len(items) == 0:
				continue
			u_time = time_dict.get(u, None)
			if u_time is None:
				continue

			for it in items:
				if it not in u_time:
					continue

				t = u_time[it]
				if isinstance(t, (list, tuple, np.ndarray)):
					item_time_dict[it].extend(list(t))
				else:
					item_time_dict[it].append(t)

		item_time_list = []

		for i in range(self.m_item):
			times = np.array(item_time_dict[i], dtype=np.float64)
			if times.size > 0:
				times = times / 60 / 60 / 24
				times.sort()
			item_time_list.append(times)

		for i in range(self.m_item):
			times = item_time_list[i]
			if len(times) < self.max_pos:
				item_time_list[i] = np.pad(
					times,
					(0, self.max_pos - len(times)),
					mode="constant",
					constant_values=self.max_time
				)

		item_time_array = np.stack(item_time_list, axis=0) if self.max_pos > 0 else np.full((self.m_item, 0), self.max_time)
		return item_time_array

	def getUserPosItems(self, users, UserItemNet):
		posItems = []
		for user in users:
			posItems.append(UserItemNet[user, :].nonzero()[1])
		return posItems

	def getItemPosUsers(self, items, UserItemNet):
		posUsers = []
		for item in items:
			posUsers.append(UserItemNet[:, item].nonzero()[0])
		return posUsers

	def getUserValidItems(self, users, valid_dict):
		validItems = []
		for user in users:
			if user in valid_dict:
				validItems.append(valid_dict[user])
		return validItems

	# ============================================================
	# New utilities for Transformer residual learning
	# ============================================================

	def build_user_interactions(self, time_dict):
		"""
		Build sorted interaction list per user in day scale.

		Returns:
			user_interactions[u] = [(time_in_days, item), ...] sorted by time
		"""
		user_interactions = {u: [] for u in range(self.n_user)}
		for u, item_time in time_dict.items():
			for it, t in item_time.items():
				t_day = float(t) / 60 / 60 / 24
				user_interactions[u].append((t_day, it))
		for u in user_interactions:
			user_interactions[u].sort(key=lambda x: x[0])
		return user_interactions

	def build_event_lists(self):
		"""
		Build flattened event lists for train/valid/test.
		Each event is (user, item, time_in_days).
		"""
		self.train_events = []
		for (u, v), t in self.train_user_item_time.items():
			self.train_events.append((u, v, float(t)))
		self.train_events.sort(key=lambda x: x[2])

		self.valid_events = []
		for (u, v), t in self.valid_user_item_time.items():
			self.valid_events.append((u, v, float(t)))
		self.valid_events.sort(key=lambda x: x[2])

		self.test_events = []
		for (u, v), t in self.test_user_item_time.items():
			self.test_events.append((u, v, float(t)))
		self.test_events.sort(key=lambda x: x[2])

		# Default arrays used by training loop
		self.user_list = np.array([u for (u, v, t) in self.train_events], dtype=np.int64)
		self.item_list = np.array([v for (u, v, t) in self.train_events], dtype=np.int64)
		self.event_time_list = np.array([t for (u, v, t) in self.train_events], dtype=np.float32)
		self.trainDataSize = len(self.train_events)

	def build_user_histories(self, max_seq_len=50):
		"""
		For each event in train/valid/test, build padded user history sequence
		using interactions strictly before the event time.

		Saves:
			train_hist_item_list, train_hist_time_list
			valid_hist_item_list, valid_hist_time_list
			test_hist_item_list,  test_hist_time_list

		Padding:
			item id = 0
			time = 0.0
		"""
		self.max_seq_len = max_seq_len

		def _build_histories(events):
			hist_item_list = []
			hist_time_list = []

			for (u, v, t) in events:
				hist = [(tt, vv) for (tt, vv) in self.user_interactions[u] if tt < t]
				hist = hist[-max_seq_len:]

				h_items = [vv for (tt, vv) in hist]
				h_times = [tt for (tt, vv) in hist]

				pad_len = max_seq_len - len(h_items)
				h_items = [0] * pad_len + h_items
				h_times = [0.0] * pad_len + h_times

				hist_item_list.append(h_items)
				hist_time_list.append(h_times)

			return np.array(hist_item_list, dtype=np.int64), np.array(hist_time_list, dtype=np.float32)

		self.train_hist_item_list, self.train_hist_time_list = _build_histories(self.train_events)
		self.valid_hist_item_list, self.valid_hist_time_list = _build_histories(self.valid_events)
		self.test_hist_item_list, self.test_hist_time_list = _build_histories(self.test_events)

		# default aliases used during training
		self.hist_item_list = self.train_hist_item_list
		self.hist_time_list = self.train_hist_time_list

	def get_pair_user_event_timebucket(self, k=1, bucket_size=1.0):
		"""
		Negative user sampler for user-side NCE.

		Sample negative users approximately from p(u|t,H_t) using
		uniform sampling over active users within the same time bucket.

		Args:
			k: number of negative users per event (for your experiment, use k=1)
			bucket_size: time bucket in day units.
				e.g. 1.0 -> same day bucket
		"""
		# Build active-user pool by time bucket using TRAIN events only
		bucket_to_users = {}
		for (u, v, t) in self.train_events:
			b = int(t // bucket_size)
			if b not in bucket_to_users:
				bucket_to_users[b] = set()
			bucket_to_users[b].add(u)

		all_users = np.arange(self.n_user, dtype=np.int64)

		pos_user_list = []
		neg_user_list = []

		for idx in range(self.trainDataSize):
			pos_u = int(self.user_list[idx])
			t = float(self.event_time_list[idx])
			b = int(t // bucket_size)

			candidate_users = list(bucket_to_users.get(b, []))
			candidate_users = [u for u in candidate_users if u != pos_u]

			# fallback if the bucket has no other active users
			if len(candidate_users) == 0:
				candidate_users = [u for u in all_users if u != pos_u]

			replace = len(candidate_users) < k
			neg_us = np.random.choice(candidate_users, size=k, replace=replace)

			pos_user_list.append(pos_u)
			neg_user_list.append(neg_us.tolist())

		self.pos_user_list = np.array(pos_user_list, dtype=np.int64)
		self.neg_user_list = np.array(neg_user_list, dtype=np.int64)

	def prepare_user_timebucket_sampler(self, bucket_size=1.0):
		"""
		Precompute bucket -> active users array and event -> bucket mapping.
		Call once unless bucket_size changes.
		"""
		self.user_bucket_size = float(bucket_size)

		# active users per bucket from TRAIN events
		bucket_to_users = {}
		for (u, v, t) in self.train_events:
			b = int(t // self.user_bucket_size)
			if b not in bucket_to_users:
				bucket_to_users[b] = set()
			bucket_to_users[b].add(u)

		# store as numpy arrays for fast indexing
		self.bucket_to_users_np = {
			b: np.array(sorted(list(users)), dtype=np.int64)
			for b, users in bucket_to_users.items()
		}

		# bucket id for each train event
		self.train_event_bucket = np.array(
			[int(t // self.user_bucket_size) for t in self.event_time_list],
			dtype=np.int64
		)

		# event indices grouped by bucket
		self.bucket_to_event_idx = {}
		for idx, b in enumerate(self.train_event_bucket):
			if b not in self.bucket_to_event_idx:
				self.bucket_to_event_idx[b] = []
			self.bucket_to_event_idx[b].append(idx)
		self.bucket_to_event_idx = {
			b: np.array(idxs, dtype=np.int64)
			for b, idxs in self.bucket_to_event_idx.items()
		}

		# global users for fallback
		self.all_users_np = np.arange(self.n_user, dtype=np.int64)

	def get_pair_user_event_timebucket_fast(self, k=1):
		"""
		Fast negative sampler for K=1 using precomputed time buckets.
		Requires prepare_user_timebucket_sampler(...) to be called first.
		"""
		assert k == 1, "This fast sampler is specialized for K=1."

		if not hasattr(self, "bucket_to_users_np"):
			raise RuntimeError("Call prepare_user_timebucket_sampler(bucket_size=...) first.")

		pos_user = self.user_list  # [N]
		N = len(pos_user)

		neg_user = np.empty((N, 1), dtype=np.int64)

		for b, event_idx in self.bucket_to_event_idx.items():
			users_arr = self.bucket_to_users_np[b]   # active users in this bucket
			pos_u_b = pos_user[event_idx]            # positive users for events in this bucket

			if len(users_arr) >= 2:
				# vectorized rejection sampling to avoid sampling the positive user
				rand_idx = np.random.randint(0, len(users_arr), size=len(event_idx))
				sampled = users_arr[rand_idx]

				mask = (sampled == pos_u_b)
				while mask.any():
					rand_idx_resample = np.random.randint(0, len(users_arr), size=mask.sum())
					sampled[mask] = users_arr[rand_idx_resample]
					mask = (sampled == pos_u_b)

				neg_user[event_idx, 0] = sampled

			else:
				# fallback: if only one active user in bucket and it is the positive user,
				# sample from global users excluding pos_u
				u = pos_u_b
				r = np.random.randint(0, self.n_user - 1, size=len(event_idx))
				r += (r >= u)
				neg_user[event_idx, 0] = r

		self.pos_user_list = pos_user.astype(np.int64)
		self.neg_user_list = neg_user.astype(np.int64)


	def get_pair_item_event_uniform(self, neg_size, sample_num=None):
		# 0) event index가 없으면 1회 구축
		if not hasattr(self, "train_event_items"):
			times = self.train_item_time_array
			mask = (times < self.max_time)

			self.train_event_items = np.repeat(np.arange(self.m_item), mask.sum(axis=1))
			self.train_event_times = times[mask]
			self.trainEventSize = len(self.train_event_items)

		if sample_num is None:
			sample_num = self.trainEventSize

		# 1) positive event sample
		ev_idx = np.random.randint(0, self.trainEventSize, sample_num)
		pos_item = self.train_event_items[ev_idx]
		pos_time = self.train_event_times[ev_idx]

		# 2) uniform negative excluding pos_item
		neg_item = np.random.randint(0, self.m_item - 1, size=(sample_num, neg_size))
		neg_item += (neg_item >= pos_item[:, None])

		# 3) histories
		self.pos_item_list = pos_item
		self.neg_item_list = neg_item
		self.pos_time_list = pos_time
		self.pos_time_all = self.item_time_array[pos_item]
		self.neg_time_all = self.item_time_array[neg_item]

	def __getitem__(self, idx):
		return (
			self.item_list[idx],
			self.pos_user_list[idx],
			self.neg_user_list[idx],
			self.hist_item_list[idx],
			self.hist_time_list[idx],
			self.event_time_list[idx],
		)

	def __len__(self):
		return self.trainDataSize


#%%
args = parse_args()
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.seed)
args.device = set_device()
args.expt_name = f"expt21_trans_nce_{expt_num}"
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True) 

wandb_login = False
try:
    wandb_login = wandb.login(key = open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
except:
    pass
if wandb_login:
    configs = vars(args)
    wandb_var = wandb.init(project="ldr_rec2", config=configs)
    wandb.run.name = args.expt_name


#%%
dataset = UserItemTime(args)
dataset.build_user_histories(max_seq_len=args.max_seq_len)
dataset.prepare_user_timebucket_sampler(bucket_size=args.user_bucket_size)
dataset.get_pair_user_event_timebucket_fast(k=1)
dataset.get_pair_item_event_uniform(args.contrast_size-1)

valid_hist_items_t = torch.from_numpy(dataset.valid_hist_item_list).long().to(args.device)
valid_hist_times_t = torch.from_numpy(dataset.valid_hist_time_list).float().to(args.device)

test_hist_items_t = torch.from_numpy(dataset.test_hist_item_list).long().to(args.device)
test_hist_times_t = torch.from_numpy(dataset.test_hist_time_list).float().to(args.device)

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)
all_item_idxs = np.arange(dataset.m_item)
all_user_idxs = np.arange(dataset.n_user)

#%%
model = JointRecTransformer(
    dataset.n_user,
    dataset.m_item,
    args.recdim,
    mini_batch,
    args.device,
    args.depth,
    args.tau,
    max_seq_len=args.max_seq_len,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    dropout=args.dropout,
).to(args.device)

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
		pos_user = torch.tensor(dataset.pos_user_list[sample_idx], dtype=torch.long).to(args.device)   # [B]
		neg_user = torch.tensor(np.array(dataset.neg_user_list, dtype=np.int64)[sample_idx, 0], dtype=torch.long).to(args.device)  # [B]
		anchor_item = torch.tensor(dataset.item_list[sample_idx], dtype=torch.long).to(args.device)    # [B]
		query_time = torch.tensor(dataset.pos_time_list[sample_idx], dtype=torch.float32).to(args.device)  # [B]

		hist_items = torch.tensor(np.array(dataset.hist_item_list)[sample_idx], dtype=torch.long).to(args.device)       # [B, L]
		hist_times = torch.tensor(np.array(dataset.hist_time_list)[sample_idx], dtype=torch.float32).to(args.device)    # [B, L]

		pos_score, _, _ = model.residual_score(
			pos_user, anchor_item, hist_items, hist_times, query_time
		)

		neg_score, _, _ = model.residual_score(
			neg_user, anchor_item, hist_items, hist_times, query_time
        )

		user_loss = -(F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean())

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
		dataset.get_pair_user_event_timebucket_fast(k=1)

	if epoch % args.evaluate_interval == 0:
		user_nll_list = []
		item_nll_list = []
		joint_nll_list = []
		pred_list = []
		gt_list = []

		model.eval()
		intensity_decay = model.soft(model.intensity_decay)

		# precompute item tower
		with torch.no_grad():
			all_item_keys = model.item_proj(model.item_embedding.weight)
			all_item_emb = F.normalize(model.item_embedding.weight, dim=-1)

		item_base_all, item_amplitude_all = [], []
		for idx2 in range(dataset.m_item // args.batch_size + 1):
			item_idx = all_item_idxs[idx2*args.batch_size : (idx2+1)*args.batch_size]
			if len(item_idx) == 0:
				continue
			sub_item_embed = all_item_emb[item_idx]
			with torch.no_grad():
				base = model.soft(model.base_fn(sub_item_embed))
				amplitude = model.soft(model.amplitude_fn(sub_item_embed))
			item_base_all.append(base)
			item_amplitude_all.append(amplitude)

		for i, ((user, item), pos_time_val) in enumerate(dataset.valid_user_item_time.items()):
			pos_time_t = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)

			# ----- prior log prob over items -----
			item_logits_list = []
			for idx2 in range(dataset.m_item // args.batch_size + 1):
				item_idx = all_item_idxs[idx2 * args.batch_size: (idx2 + 1) * args.batch_size]
				if len(item_idx) == 0:
					continue
				batch_time_all = torch.tensor(dataset.item_time_array[item_idx], dtype=torch.float32).to(args.device)
				batch_time_mask = batch_time_all < pos_time_t
				batch_time_delta = (pos_time_t - batch_time_all).clamp(min=0.0)

				with torch.no_grad():
					time_intensity = (torch.exp(-intensity_decay * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
					logits = (item_base_all[idx2] + item_amplitude_all[idx2] * time_intensity).flatten()
				item_logits_list.append(logits)

			item_logits = torch.concat(item_logits_list)
			item_log_prob = torch.log(item_logits + 1e-12) - torch.log(item_logits.sum() + 1e-12)

			# ----- residual scores over items -----
			hist_items = valid_hist_items_t[i:i+1]
			hist_times = valid_hist_times_t[i:i+1]
			user_idx = torch.tensor([user], dtype=torch.long).to(args.device)
			query_time = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)

			with torch.no_grad():
				q_u = model.encode_user_history(user_idx, hist_items, hist_times, query_time)   # [1, D]
				residual_scores = torch.matmul(q_u, all_item_keys[:-1,:].T).squeeze(0) / model.tau     # [I]
				residual_log_prob = residual_scores - torch.logsumexp(residual_scores, dim=0)

			# final score = prior + residual
			pred = (item_log_prob.cpu() + residual_scores.cpu()).cpu()

			item_nll = -item_log_prob[item].item()
			user_nll = -residual_log_prob[item].item()
			joint_nll = -(item_log_prob[item].item() + residual_scores[item].item())

			item_nll_list.append(item_nll)
			user_nll_list.append(user_nll)
			joint_nll_list.append(joint_nll)

			exclude_items = list(dataset._allPos[user])
			pred[exclude_items] = -9999
			_, pred_k = torch.topk(pred, k=max(args.topks))
			pred_list.append(pred_k.cpu())
			gt_list.append([item])

		valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

		if wandb_login:
			wandb_var.log({
				"valid_item_nll": np.mean(item_nll_list),
				"valid_user_nll": np.mean(user_nll_list),
				"valid_joint_nll": np.mean(joint_nll_list),
				"train_item_nll": epoch_item_loss/batch_num,
				"train_user_nll": epoch_user_loss/batch_num,
				"train_time_intensity": epoch_time_intensity/batch_num,
				"intendety_decay": model.soft(model.intensity_decay).item(),
				})

			wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
			wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
			wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
			wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

		if np.mean(joint_nll_list) - best_joint_nll > 0.:
			cnt += 1
		else:
			best_joint_nll = np.mean(joint_nll_list)
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

best_model = JointRecTransformer(
    dataset.n_user,
    dataset.m_item,
    args.recdim,
    mini_batch,
    args.device,
    args.depth,
    args.tau,
    max_seq_len=args.max_seq_len,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    dropout=args.dropout,
).to(args.device)
best_model.load_state_dict(best_state)

best_model.eval()
intensity_decay = best_model.soft(best_model.intensity_decay)


# precompute item tower
with torch.no_grad():
	all_item_keys = best_model.item_proj(best_model.item_embedding.weight)
	all_item_emb = F.normalize(best_model.item_embedding.weight, dim=-1)

item_base_all, item_amplitude_all = [], []
for idx2 in range(dataset.m_item // args.batch_size + 1):
	item_idx = all_item_idxs[idx2*args.batch_size : (idx2+1)*args.batch_size]
	if len(item_idx) == 0:
		continue
	sub_item_embed = all_item_emb[item_idx]
	with torch.no_grad():
		base = best_model.soft(best_model.base_fn(sub_item_embed))
		amplitude = best_model.soft(best_model.amplitude_fn(sub_item_embed))
	item_base_all.append(base)
	item_amplitude_all.append(amplitude)

for i, ((user, item), pos_time_val) in enumerate(dataset.valid_user_item_time.items()):
	pos_time_t = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)

	# ----- prior log prob over items -----
	item_logits_list = []
	for idx2 in range(dataset.m_item // args.batch_size + 1):
		item_idx = all_item_idxs[idx2 * args.batch_size: (idx2 + 1) * args.batch_size]
		if len(item_idx) == 0:
			continue
		batch_time_all = torch.tensor(dataset.item_time_array[item_idx], dtype=torch.float32).to(args.device)
		batch_time_mask = batch_time_all < pos_time_t
		batch_time_delta = (pos_time_t - batch_time_all).clamp(min=0.0)

		with torch.no_grad():
			time_intensity = (torch.exp(-intensity_decay * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
			logits = (item_base_all[idx2] + item_amplitude_all[idx2] * time_intensity).flatten()
		item_logits_list.append(logits)

	item_logits = torch.concat(item_logits_list)
	item_log_prob = torch.log(item_logits + 1e-12) - torch.log(item_logits.sum() + 1e-12)

	# ----- residual scores over items -----
	hist_items = test_hist_items_t[i:i+1]
	hist_times = test_hist_times_t[i:i+1]
	user_idx = torch.tensor([user], dtype=torch.long).to(args.device)
	query_time = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)

	with torch.no_grad():
		q_u = best_model.encode_user_history(user_idx, hist_items, hist_times, query_time)
		residual_scores = torch.matmul(q_u, all_item_keys[:-1,:].T).squeeze(0) / best_model.tau
		residual_log_prob = residual_scores - torch.logsumexp(residual_scores, dim=0)

	# final score = prior + residual
	pred = (item_log_prob.cpu() + residual_scores.cpu()).cpu()

	item_nll = -item_log_prob[item].item()
	user_nll = -residual_log_prob[item].item()
	joint_nll = -(item_log_prob[item].item() + residual_scores[item].item())

	item_nll_list.append(item_nll)
	user_nll_list.append(user_nll)
	joint_nll_list.append(joint_nll)

	exclude_items = list(dataset._allPos[user])
	pred[exclude_items] = -9999
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


