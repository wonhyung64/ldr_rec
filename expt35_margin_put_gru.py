#%%
import os
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from datetime import datetime
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from modules.utils import parse_args, set_seed, set_device
from modules.procedure import computeTopNAccuracy


"""EXPERIMENT FOR THE SIMPLEST p(u,v|t,H_t) WITH HAWKES PROCESS

Updated for density-ratio training:
- negative user u^- is sampled from empirical p(u|t) obtained by marginalizing
  observed p(u,v|t) over items within a time bucket, instead of uniform over
  active users in the bucket.
"""
class JointRecStatic(nn.Module):
	def __init__(
		self,
		num_users: int,
		num_items: int,
		embedding_k: int,
		mini_batch: int,
		device,
		depth: int = 0,
		tau: float = 0.5,
	):
		super().__init__()
		self.num_users = num_users
		self.num_items = num_items
		self.embedding_k = embedding_k
		self.depth = depth
		self.mini_batch = mini_batch
		self.soft = nn.Softplus()
		self.tau = tau
		self.device = device

		# prior side
		self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

		base_fn = [nn.Linear(embedding_k, embedding_k // 2), nn.Softplus()]
		for _ in range(depth):
			base_fn += [nn.Linear(embedding_k // 2, embedding_k // 2), nn.Softplus()]
		base_fn += [nn.Linear(embedding_k // 2, 1, bias=False)]
		self.base_fn = nn.Sequential(*base_fn)

		amplitude_fn = [nn.Linear(embedding_k, embedding_k // 2), nn.Softplus()]
		for _ in range(depth):
			amplitude_fn += [nn.Linear(embedding_k // 2, embedding_k // 2), nn.Softplus()]
		amplitude_fn += [nn.Linear(embedding_k // 2, 1, bias=False)]
		self.amplitude_fn = nn.Sequential(*amplitude_fn)

		self.intensity_decay = nn.Parameter(torch.randn(1))

		# residual side: static user embedding -> GRU(history)
		self.padding_item_id = self.num_items
		self.item_resid_embedding = nn.Embedding(
			self.num_items + 1,   # +1 for padding item
			self.embedding_k,
			padding_idx=self.padding_item_id,
		)
		self.user_gru = nn.GRU(
			input_size=self.embedding_k,
			hidden_size=self.embedding_k,
			num_layers=1,
			batch_first=True,
		)

	def popularity(self, batch_items, pos_time, batch_time_all):
		item_embed = F.normalize(self.item_embedding(batch_items), dim=-1)
		base = self.soft(self.base_fn(item_embed)).reshape(self.mini_batch, -1)
		amplitude = self.soft(self.amplitude_fn(item_embed)).reshape(self.mini_batch, -1)

		batch_time_mask = batch_time_all < pos_time
		batch_time_delta = (pos_time - batch_time_all).clamp(0.0)
		intensity_decay = self.soft(self.intensity_decay)
		time_intensity = torch.exp(-intensity_decay * batch_time_delta) * batch_time_mask

		return base + (time_intensity.sum(-1) * amplitude), time_intensity, base, amplitude

	def encode_user_history(self, hist_item_idx):
		"""
		hist_item_idx: [B, L]
		return: normalized user state [B, D]
		"""
		hist_emb = self.item_resid_embedding(hist_item_idx)   # [B, L, D]
		_, h_n = self.user_gru(hist_emb)                      # [1, B, D]
		u = h_n[-1]                                           # [B, D]

		# all-padding history -> zero vector
		hist_mask = (hist_item_idx != self.padding_item_id)
		no_hist = (hist_mask.sum(dim=-1) == 0)
		if no_hist.any():
			u = torch.where(no_hist.unsqueeze(-1), torch.zeros_like(u), u)

		u = F.normalize(u, dim=-1)
		return u

	def residual_score(self, item_idx, hist_item_idx):
		u = self.encode_user_history(hist_item_idx)                  # [B, D]
		v = F.normalize(self.item_resid_embedding(item_idx), dim=-1) # [B, D]
		score = torch.sum(u * v, dim=-1, keepdim=True) / self.tau
		return score, u, v

	def score_all_items(self, hist_item_idx):
		u = self.encode_user_history(hist_item_idx)                          # [B, D]
		v_all = F.normalize(self.item_resid_embedding.weight[:self.num_items], dim=-1)  # [I, D]
		return torch.matmul(u, v_all.T) / self.tau                           # [B, I]


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
				h_items = [self.m_item] * pad_len + h_items
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

	def get_histories_for_users_at_times(self, users, query_times, max_seq_len):
		hist_items_batch = []
		hist_times_batch = []

		for u, t in zip(users, query_times):
			hist = [(tt, vv) for (tt, vv) in self.user_interactions[int(u)] if tt < float(t)]
			hist = hist[-max_seq_len:]

			h_items = [vv for (tt, vv) in hist]
			h_times = [tt for (tt, vv) in hist]

			pad_len = max_seq_len - len(h_items)
			h_items = [self.m_item] * pad_len + h_items   # padding_item_id
			h_times = [0.0] * pad_len + h_times

			hist_items_batch.append(h_items)
			hist_times_batch.append(h_times)

		return (
			np.array(hist_items_batch, dtype=np.int64),
			np.array(hist_times_batch, dtype=np.float32),
		)

	def get_pair_user_event_timebucket(self, k=1, bucket_size=1.0):
		"""
		Negative user sampler for user-side NCE.

		Sample negative users from the empirical marginal p(u|t), estimated from
		observed train events by counting users within the same time bucket:

			p_hat(u | bucket b) propto #{(u, v, t) in D_train : floor(t / bucket_size) = b}

		This corresponds to marginalizing the observed p(u, v | t) over items.

		Args:
			k: number of negative users per event.
			bucket_size: time bucket in day units.
		"""
		self.prepare_user_timebucket_sampler(bucket_size=bucket_size)
		self.get_pair_user_event_timebucket_fast(k=k)

	def prepare_user_timebucket_sampler(self, bucket_size=1.0):
		"""
		Precompute empirical p(u|t) by time bucket from TRAIN events.

		For each time bucket b, we estimate
			p_hat(u | b) propto count_train(u, b),
		where count_train(u, b) is the number of observed interactions of user u
		in bucket b. This is the empirical marginal of p(u, v | t) over v.

		Implementation note:
			We store per-bucket active users, probabilities, and cumulative
			distributions (CDFs) so that sampling can be done with vectorized
			searchsorted instead of repeated np.random.choice calls.
		"""
		self.user_bucket_size = float(bucket_size)

		train_event_bucket = np.floor(self.event_time_list / self.user_bucket_size).astype(np.int64)
		self.train_event_bucket = train_event_bucket

		global_user_counts = np.bincount(self.user_list, minlength=self.n_user).astype(np.float64)
		if global_user_counts.sum() == 0:
			global_user_counts[:] = 1.0

		self.all_users_np = np.arange(self.n_user, dtype=np.int64)
		self.all_user_probs_np = (global_user_counts / global_user_counts.sum()).astype(np.float64)
		self.all_user_cdf_np = np.cumsum(self.all_user_probs_np)
		self.all_user_cdf_np[-1] = 1.0

		self.bucket_to_users_np = {}
		self.bucket_to_user_probs_np = {}
		self.bucket_to_user_cdf_np = {}
		self.bucket_to_event_idx = {}

		unique_buckets, inverse = np.unique(train_event_bucket, return_inverse=True)
		for local_b, b in enumerate(unique_buckets):
			event_idx = np.flatnonzero(inverse == local_b).astype(np.int64)
			users_in_bucket = self.user_list[event_idx]
			counts = np.bincount(users_in_bucket, minlength=self.n_user).astype(np.float64)
			active = np.flatnonzero(counts > 0).astype(np.int64)
			probs = counts[active]
			probs /= probs.sum()
			cdf = np.cumsum(probs)
			cdf[-1] = 1.0

			b = int(b)
			self.bucket_to_users_np[b] = active
			self.bucket_to_user_probs_np[b] = probs
			self.bucket_to_user_cdf_np[b] = cdf
			self.bucket_to_event_idx[b] = event_idx

	def sample_users_excluding_positive(self, users_arr, probs_arr, pos_u, size=1, cdf_arr=None):
		"""
		Sample users from a categorical distribution while excluding the positive user.
		The probabilities are renormalized after removing pos_u.

		This path is kept for compatibility/debugging. For training-time fast
		sampling, use `_sample_one_excluding_vectorized` via
		`get_pair_user_event_timebucket_fast`.
		"""
		mask = (users_arr != pos_u)
		filtered_users = users_arr[mask]
		filtered_probs = probs_arr[mask]

		if filtered_users.size == 0:
			filtered_users = self.all_users_np[self.all_users_np != pos_u]
			filtered_probs = self.all_user_probs_np[self.all_users_np != pos_u]

		filtered_probs = filtered_probs.astype(np.float64)
		prob_sum = filtered_probs.sum()
		if prob_sum <= 0:
			filtered_probs = np.ones_like(filtered_probs, dtype=np.float64) / len(filtered_probs)
		else:
			filtered_probs = filtered_probs / prob_sum

		replace = len(filtered_users) < size
		return np.random.choice(filtered_users, size=size, replace=replace, p=filtered_probs)

	def _sample_one_excluding_vectorized(self, users_arr, cdf_arr, pos_u_arr):
		"""
		Vectorized sampler for k=1.

		First sample from the original categorical with searchsorted on the CDF,
		then only resample the collided positions (sample == pos_u). In practice
		this is much faster than calling np.random.choice for every event.
		"""
		if users_arr.size == 0:
			raise RuntimeError("users_arr must be non-empty.")

		pos_u_arr = np.asarray(pos_u_arr, dtype=np.int64)
		out = users_arr[np.searchsorted(cdf_arr, np.random.random(size=pos_u_arr.shape[0]), side='left')]
		mask = (out == pos_u_arr)

		if users_arr.size == 1:
			out = self.all_users_np[np.searchsorted(self.all_user_cdf_np, np.random.random(size=pos_u_arr.shape[0]), side='left')]
			mask = (out == pos_u_arr)

		while mask.any():
			resampled = users_arr[np.searchsorted(cdf_arr, np.random.random(size=mask.sum()), side='left')]
			out[mask] = resampled
			mask = (out == pos_u_arr)

		return out.astype(np.int64)

	def get_pair_user_event_timebucket_fast(self, k=1):
		"""
		Fast negative sampler using precomputed empirical p(u|t) per time bucket.
		Requires prepare_user_timebucket_sampler(...) to be called first.
		"""
		if not hasattr(self, "bucket_to_users_np") or not hasattr(self, "bucket_to_user_cdf_np"):
			raise RuntimeError("Call prepare_user_timebucket_sampler(bucket_size=...) first.")

		pos_user = self.user_list.astype(np.int64)
		N = len(pos_user)

		if k == 1:
			neg_user = np.empty((N, 1), dtype=np.int64)
			for b, event_idx in self.bucket_to_event_idx.items():
				users_arr = self.bucket_to_users_np.get(b, self.all_users_np)
				cdf_arr = self.bucket_to_user_cdf_np.get(b, self.all_user_cdf_np)
				pos_u_b = pos_user[event_idx]

				if users_arr.size == 1:
					neg_user[event_idx, 0] = self._sample_one_excluding_vectorized(
						self.all_users_np,
						self.all_user_cdf_np,
						pos_u_b,
					)
				else:
					neg_user[event_idx, 0] = self._sample_one_excluding_vectorized(
						users_arr,
						cdf_arr,
						pos_u_b,
					)
		else:
			neg_user = np.empty((N, k), dtype=np.int64)
			for j in range(k):
				for b, event_idx in self.bucket_to_event_idx.items():
					users_arr = self.bucket_to_users_np.get(b, self.all_users_np)
					cdf_arr = self.bucket_to_user_cdf_np.get(b, self.all_user_cdf_np)
					pos_u_b = pos_user[event_idx]
					if users_arr.size == 1:
						neg_user[event_idx, j] = self._sample_one_excluding_vectorized(
							self.all_users_np,
							self.all_user_cdf_np,
							pos_u_b,
						)
					else:
						neg_user[event_idx, j] = self._sample_one_excluding_vectorized(
							users_arr,
							cdf_arr,
							pos_u_b,
						)

		self.pos_user_list = pos_user
		self.neg_user_list = neg_user

	def prepare_item_event_sampler(self):
		"""
		Prepare fast prior-side positive event source from TRAIN events.

		Instead of reconstructing (item, time) events from the padded
		item_time_array and materializing large pos/neg history tensors at every
		reset, we directly reuse the flattened TRAIN event arrays:
			(item_list, event_time_list).
		This makes pair reset much cheaper and avoids storing huge
		neg_time_all tensors in RAM.
		"""
		self.train_event_items = self.item_list.astype(np.int64)
		self.train_event_times = self.event_time_list.astype(np.float32)
		self.trainEventSize = len(self.train_event_items)

	def get_pair_item_event_uniform(self, neg_size, sample_num=None):
		"""
		Fast prior-side sampler.

		Positive events are sampled from observed TRAIN (item, time) events.
		Negatives are sampled uniformly over items excluding the positive item.

		Only indices/items/times are materialized here; expensive time-history
		gathers are deferred to the mini-batch step.
		"""
		if not hasattr(self, "train_event_items"):
			self.prepare_item_event_sampler()

		if sample_num is None:
			sample_num = self.trainEventSize

		# 1) positive event sample from observed TRAIN events
		ev_idx = np.random.randint(0, self.trainEventSize, sample_num, dtype=np.int64)
		pos_item = self.train_event_items[ev_idx]
		pos_time = self.train_event_times[ev_idx]

		# 2) uniform negative excluding pos_item
		neg_item = np.random.randint(0, self.m_item - 1, size=(sample_num, neg_size), dtype=np.int64)
		neg_item += (neg_item >= pos_item[:, None])

		# 3) store only lightweight arrays; gather histories lazily per batch
		self.pos_item_list = pos_item
		self.neg_item_list = neg_item
		self.pos_time_list = pos_time

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
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True) 

file_dir = inspect.getfile(inspect.currentframe())
file_name = file_dir.split("/")[-1]
if file_name.split(".")[-1] == "py":
    try:
        wandb_login = wandb.login(key=open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
    except:
        pass

if wandb_login:
    expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
    args.expt_name = f"{file_name.split('.')[-2]}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec2", config=vars(args))
    wandb.run.name = args.expt_name

#%%
dataset = UserItemTime(args)

if not hasattr(args, "max_seq_len"):
	args.max_seq_len = 50

dataset.build_user_histories(max_seq_len=args.max_seq_len)
dataset.prepare_user_timebucket_sampler(bucket_size=args.user_bucket_size)
dataset.prepare_item_event_sampler()
dataset.get_pair_user_event_timebucket_fast(k=1)
dataset.get_pair_item_event_uniform(args.contrast_size-1)

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch
all_idxs = np.arange(dataset.trainDataSize)
all_item_idxs = np.arange(dataset.m_item)
all_user_idxs = np.arange(dataset.n_user)

#%%
model = JointRecStatic(
    dataset.n_user,
    dataset.m_item,
    args.recdim,
    mini_batch,
    args.device,
    args.depth,
    args.tau,
).to(args.device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


#%%
best_valid_score = 0.
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

	# debug logs
	epoch_hist_emb_norm = 0.
	epoch_pos_score_mean = 0.
	epoch_neg_score_mean = 0.
	epoch_pos_score_std = 0.
	epoch_neg_score_std = 0.

	for idx in range(batch_num):
		sample_idx = all_idxs[mini_batch*idx : (idx+1)*mini_batch]

		""""ITEM"""
		pos_item = torch.tensor(dataset.pos_item_list[sample_idx]).unsqueeze(-1).to(args.device)
		neg_items = torch.tensor(dataset.neg_item_list[sample_idx]).to(args.device)
		pos_time = torch.Tensor(dataset.pos_time_list[sample_idx]).reshape(-1, 1, 1).to(args.device)
		pos_time_all = torch.tensor(dataset.item_time_array[dataset.pos_item_list[sample_idx]], dtype=torch.float32).to(args.device)
		neg_time_all = torch.tensor(dataset.item_time_array[dataset.neg_item_list[sample_idx]], dtype=torch.float32).to(args.device)

		batch_items = torch.concat([pos_item, neg_items], -1).reshape([args.batch_size])
		batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

		logits, time_intensity, base, amplitude = model.popularity(batch_items, pos_time, batch_time_all)
		log_logits = torch.log(logits + 1e-9)
		item_loss = -nn.functional.log_softmax(log_logits, dim=-1)[:, 0].mean()
		batch_intensity = (time_intensity[:,0,:].sum(-1) / (time_intensity[:,0,:] != 0).sum(-1).clamp(1)).mean()

		"""USER"""
		neg_user = torch.tensor(dataset.neg_user_list[sample_idx, 0], dtype=torch.long).to(args.device)
		anchor_item = torch.tensor(dataset.item_list[sample_idx], dtype=torch.long).to(args.device)

		# positive user history: precomputed
		pos_hist_items = torch.tensor(
			dataset.hist_item_list[sample_idx],
			dtype=torch.long
		).to(args.device)

		# negative user history at the same event time
		neg_hist_items_np, _ = dataset.get_histories_for_users_at_times(
			dataset.neg_user_list[sample_idx, 0],
			dataset.event_time_list[sample_idx],
			max_seq_len=args.max_seq_len,
		)
		neg_hist_items = torch.tensor(neg_hist_items_np, dtype=torch.long).to(args.device)

		pos_score, _, _ = model.residual_score(anchor_item, pos_hist_items)
		neg_score, _, _ = model.residual_score(anchor_item, neg_hist_items)

		with torch.no_grad():
			pos_hist_mask = (pos_hist_items != model.padding_item_id).float()   # [B, L]
			pos_hist_emb = model.item_resid_embedding(pos_hist_items)           # [B, L, D]

			denom = pos_hist_mask.sum().clamp(min=1.0)

			batch_hist_emb_norm = (
				(pos_hist_emb.norm(dim=-1) * pos_hist_mask).sum() / denom
			)

			batch_pos_score_mean = pos_score.mean()
			batch_neg_score_mean = neg_score.mean()
			batch_pos_score_std = pos_score.std(unbiased=False)
			batch_neg_score_std = neg_score.std(unbiased=False)

		user_loss = -(F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean())

		total_loss = user_loss * args.lambda1 + item_loss * (1-args.lambda1)
		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		epoch_item_loss += item_loss.item()
		epoch_user_loss += user_loss.item()
		epoch_time_intensity += batch_intensity.item()

		epoch_hist_emb_norm += batch_hist_emb_norm.item()
		epoch_pos_score_mean += batch_pos_score_mean.item()
		epoch_neg_score_mean += batch_neg_score_mean.item()
		epoch_pos_score_std += batch_pos_score_std.item()
		epoch_neg_score_std += batch_neg_score_std.item()

	print(f"[Epoch {epoch:>4d} Train Loss] user: {epoch_user_loss/batch_num:.4f} / item: {epoch_item_loss/batch_num:.4f}")

	if epoch % args.pair_reset_interval == 0:
		print("Reset negative pairs")
		dataset.get_pair_item_event_uniform(args.contrast_size-1)
		dataset.get_pair_user_event_timebucket_fast(k=1)

	if epoch % args.evaluate_interval == 0:
		item_nll_list = []
		pred_list = []
		gt_list = []

		model.eval()
		intensity_decay = model.soft(model.intensity_decay)

		# precompute item tower
		with torch.no_grad():
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
			hist_item_np, _ = dataset.get_histories_for_users_at_times(
				[user],
				[pos_time_val],
				max_seq_len=args.max_seq_len,
			)
			hist_item_t = torch.tensor(hist_item_np, dtype=torch.long).to(args.device)

			with torch.no_grad():
				residual_scores = model.score_all_items(hist_item_t).squeeze(0)   # [I]

			# final score = prior + residual
			pred = (item_log_prob.cpu() + residual_scores.cpu()).cpu()

			item_nll = -item_log_prob[item].item()
			item_nll_list.append(item_nll)

			exclude_items = list(dataset._allPos[user])
			pred[exclude_items] = -9999
			_, pred_k = torch.topk(pred, k=max(args.topks))
			pred_list.append(pred_k.cpu())
			gt_list.append([item])

		valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

		if wandb_login:

			wandb_var.log({
				"valid_item_nll": np.mean(item_nll_list),
				"train_item_nll": epoch_item_loss/batch_num,
				"train_ldr": epoch_user_loss/batch_num,
				"train_time_intensity": epoch_time_intensity/batch_num,
				"intendety_decay": model.soft(model.intensity_decay).item(),

				# debug
				"debug_hist_emb_norm": epoch_hist_emb_norm / batch_num,
				"debug_pos_score_mean": epoch_pos_score_mean / batch_num,
				"debug_neg_score_mean": epoch_neg_score_mean / batch_num,
				"debug_pos_score_std": epoch_pos_score_std / batch_num,
				"debug_neg_score_std": epoch_neg_score_std / batch_num,
			})

			wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
			wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
			wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
			wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

		current_valid_score = valid_results[1][0]

		if current_valid_score - best_valid_score <= 0.:
			cnt += 1
		else:
			best_valid_score = current_valid_score
			best_state = copy.deepcopy(model.state_dict())
			best_epoch = epoch
			cnt = 1

		if cnt == 5:
			break


#%%
pred_list = []
gt_list = []
item_nll_list = []

best_model = JointRecStatic(
    dataset.n_user,
    dataset.m_item,
    args.recdim,
    mini_batch,
    args.device,
    args.depth,
    args.tau,
).to(args.device)

best_model.load_state_dict(best_state)

best_model.eval()
intensity_decay = best_model.soft(best_model.intensity_decay)


# precompute item tower
with torch.no_grad():
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

for i, ((user, item), pos_time_val) in enumerate(dataset.test_user_item_time.items()):
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
	hist_item_np, _ = dataset.get_histories_for_users_at_times(
		[user],
		[pos_time_val],
		max_seq_len=args.max_seq_len,
	)
	hist_item_t = torch.tensor(hist_item_np, dtype=torch.long).to(args.device)

	with torch.no_grad():
		residual_scores = best_model.score_all_items(hist_item_t).squeeze(0)

	# final score = prior + residual
	pred = (item_log_prob.cpu() + residual_scores.cpu()).cpu()

	item_nll = -item_log_prob[item].item()
	item_nll_list.append(item_nll)

	exclude_items = list(dataset._allPos[user])
	pred[exclude_items] = -9999
	_, pred_k = torch.topk(pred, k=max(args.topks))
	pred_list.append(pred_k.cpu())
	gt_list.append([item])

test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

if wandb_login:
	wandb_var.log({
		"test_item_nll": np.mean(item_nll_list),
		})

	wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
	wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
	wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
	wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))

	wandb_var.log({"best_valid_score": best_valid_score})
	wandb_var.log({"best_epoch": best_epoch})

	wandb_var.finish()


