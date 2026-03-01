import math
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


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
				time = time_dict[user][item] /60/60/24
				user_item_time[(user,item)] = time
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
		for i in range(max(item_time_dict.keys())+1):
			try:
				times = np.array(item_time_dict[i]) /60/60/24
				max_time = max(np.max(times), max_time)
			except:
				times = np.array([])
			max_pos = max(max_pos, len(times))
			times.sort()
			item_time_array.append(times)
		self.max_pos = max_pos
		self.max_time = max_time+1
		for i in range(max(item_time_dict.keys())+1):
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
				# time_dict[u][it]가 없으면 skip (데이터 정합성 문제 대비)
				if it not in u_time:
					continue

				t = u_time[it]
				# t가 scalar일 수도, list/array일 수도 있으니 둘 다 처리
				if isinstance(t, (list, tuple, np.ndarray)):
					item_time_dict[it].extend(list(t))
				else:
					item_time_dict[it].append(t)

		# 3) convert to days and compute max_pos / max_time based on TRAIN only
		item_time_list = []

		for i in range(self.m_item):
			times = np.array(item_time_dict[i], dtype=np.float64)
			if times.size > 0:
				times = times / 60 / 60 / 24  # seconds -> days
				times.sort()
			item_time_list.append(times)


		# 모든 아이템이 max_pos 길이를 갖도록 padding
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
			posItems.append(UserItemNet[user,:].nonzero()[1])
		return posItems

	def getItemPosUsers(self, items, UserItemNet):
		posUsers = []
		for item in items:
			posUsers.append(UserItemNet[:,item].nonzero()[0])
		return posUsers

	def getUserValidItems(self, users, valid_dict):
		validItems = []
		for user in users:
			if user in valid_dict:
				validItems.append(valid_dict[user])
		return validItems

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

	def get_pair_user_event_uniform(self, neg_size: int, sample_num: int = None):
		if sample_num is None:
			sample_num = self.trainDataSize  # number of train interactions (events)
			# sample_num = dataset.trainDataSize  # number of train interactions (events)

		# 1) sample positive events uniformly
		ev_idx = np.random.randint(0, self.trainDataSize, size=sample_num)
		# ev_idx = np.random.randint(0, dataset.trainDataSize, size=sample_num)
		items = self.trainItem[ev_idx]
		# items = dataset.trainItem[ev_idx]
		pos_users = self.trainUser[ev_idx]
		# pos_users = dataset.trainUser[ev_idx]

		# 2) negative users: uniform, but must NOT be any of the positive users for that item
		# neg_size = args.contrast_size-1
		neg_users = np.empty((sample_num, neg_size), dtype=np.int64)
		for i in range(sample_num):
			v = items[i]
			pos_set = self._allPosUsers[v]
			# pos_set = dataset._allPosUsers[v]

			# rejection sampling (simple & correct)
			while True:
				cand = np.random.randint(0, self.n_user, size=neg_size)
				# cand = np.random.randint(0, dataset.n_user, size=neg_size)
				if np.isin(cand, pos_set).any():
					continue
				neg_users[i] = cand
				break

		# 3) times (optional; keeps your existing interface)
		pos_user_time_list = np.array(
			[self.train_user_item_time[(u, v)] for u, v in zip(pos_users, items)],
			# [dataset.train_user_item_time[(u, v)] for u, v in zip(pos_users, items)],
			dtype=np.float64
		)

		# 4) item histories WITHOUT leakage
		#    IMPORTANT: use train_item_time_array (train-only)
		pos_user_time_all = self.train_item_time_array[items]
		# pos_user_time_all = dataset.train_item_time_array[items]

		# 5) store to dataset fields (same naming style as your get_pair_item_bpr)
		self.item_list = items.astype(np.int64)
		self.pos_user_list = pos_users.astype(np.int64)
		self.neg_user_list = neg_users
		self.pos_user_time_list = pos_user_time_list
		self.pos_user_time_all = pos_user_time_all


	def __getitem__(self, idx):
		return self.item_list[idx], self.pos_user_list[idx], self.neg_user_list[idx], self.item_time_list[idx], self.item_time_all[idx]
	
	def __len__(self):
		return self.trainDataSize
