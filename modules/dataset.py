import math
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset



class DisenData(Dataset):
	def __init__(self, args):
		path = f"{args.data_path}/{args.dataset}"
		# self.split = args.a_split
		# self.folds = args.a_fold
		self.period = args.period
		self.n_pop_group = args.n_pop_group
		self.Graph = None
		self.n_user = 0
		self.m_item = 0
		self.time_type = args.time_type

		time_file = path + '/interaction_time_dict.npy'
		train_file = path + '/training_dict.npy'
		valid_file = path + '/validation_dict.npy'
		test_file = path + '/testing_dict.npy'

		self.time_dict = np.load(time_file, allow_pickle=True).item()
		# { user1 : { item1 : timestamp }, { item2 : timestamp } }

		self.train_dict = np.load(train_file, allow_pickle=True).item()
		# { user1 : [ item1, item2 ] }
		trainUniqueUsers, trainItem, trainUser = [], [], []
		self.traindataSize = 0
		for uid in self.train_dict.keys():
			if len(self.train_dict[uid]) != 0:
				trainUniqueUsers.append(uid)
				trainUser.extend([uid] * len(self.train_dict[uid]))
				trainItem.extend(self.train_dict[uid])
				self.m_item = max(self.m_item, max(self.train_dict[uid]))
				self.n_user = max(self.n_user, uid)
				self.traindataSize += len(self.train_dict[uid])
		self.trainUniqueUsers = np.array(trainUniqueUsers)
		self.trainUser = np.array(trainUser)  # [interact1_user, interact2_user, interact3_user, ...]
		self.trainItem = np.array(trainItem)  # [interact1_item, interact2_item, interact3_item, ...]

		self.valid_dict = np.load(valid_file, allow_pickle=True).item()
		validUniqueUsers, validItem, validUser = [], [], []
		self.validDataSize = 0
		for uid in self.valid_dict.keys():
			if len(self.valid_dict[uid]) != 0:
				validUniqueUsers.append(uid)
				validUser.extend([uid] * len(self.valid_dict[uid]))
				validItem.extend(self.valid_dict[uid])
				self.m_item = max(self.m_item, max(self.valid_dict[uid]))
				self.n_user = max(self.n_user, uid)
				self.validDataSize += len(self.valid_dict[uid])
		self.validUniqueUsers = np.array(validUniqueUsers)
		self.validUser = np.array(validUser)
		self.validItem = np.array(validItem)

		self.test_dict = np.load(test_file, allow_pickle=True).item()
		testUniqueUsers, testItem, testUser = [], [], []
		self.testDataSize = 0
		for uid in self.test_dict.keys():
			if len(self.test_dict[uid]) != 0:
				testUniqueUsers.append(uid)
				testUser.extend([uid] * len(self.test_dict[uid]))
				testItem.extend(self.test_dict[uid])
				self.m_item = max(self.m_item, max(self.test_dict[uid]))
				self.n_user = max(self.n_user, uid)
				self.testDataSize += len(self.test_dict[uid])
		self.testUniqueUsers = np.array(testUniqueUsers)
		self.testUser = np.array(testUser)
		self.testItem = np.array(testItem)

		self.m_item += 1
		self.n_user += 1

		self.UserItemNet = csr_matrix((np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
										shape=(self.n_user, self.m_item))  # all (user, item) matrix of train, if interacted 1.
		# pre-calculate
		self._allPos = self.getUserPosItems(list(range(self.n_user)))
		self._allPosUsers = self.getItemPosUsers(list(range(self.m_items)))
		if self.time_type == "discrete":
			self.pair_stage, self.item_inter = self.get_item_inter(
				self.train_dict, self.valid_dict, self.test_dict, self.time_dict, self.period)
		elif self.time_type == "continuous":
			self.pair_stage, self.item_inter = self.get_item_inter(
				self.train_dict, self.valid_dict, self.test_dict, self.time_dict, 0)
		# pair_stage: stage of interaction
		# item_inter: interaction num per user

		pop_item = []
		for _, item_inter_ in self.item_inter.items():
			pop_item.extend(item_inter_)
		sorted_pop_item = list(set(pop_item))
		sorted_pop_item = sorted(sorted_pop_item)
		# sorted_pop_item : scales of item popularity for each user-stage

		self.num_item_pop = self.n_pop_group
		pmax = max(sorted_pop_item)
		self.pthres = [] # popularity category
		for i in range(self.num_item_pop):
			self.pthres.append(pmax** ((1/self.num_item_pop) * (i+1)))

	def get_item_inter(self, train_dict, valid_dict, test_dict, time_dict, period):
		pair_stage = {}
		item_inter = {} # number of interactions of per user in 8 train stage, 1 valid stage, 1 test stage
		# stage_num = period
		time_stage = self.get_stage(train_dict, valid_dict, test_dict, time_dict, period)
		stage_all_inter = [0] * (period + 1 * 2) # number of interactions of all users in 8 train stage, 1 valid stage, 1 test stage

		# get the inter_cnt per stage of data
		# item_inter: {item_0:{stage_0:cnt_0, stage_1:cnt_1, stage_2:cnt_2....}, item_1:{stage_0:cnt_0, stage_1:cnt_1, stage_2:cnt_2....}}
		for user in train_dict:
			for item in train_dict[user]:
				time = time_dict[user][item]
				if period:
					pair_stage[(user,item)] = time_stage[time]
				else:
					pair_stage[(user,item)] = time
				stage_all_inter[int(time_stage[time])] += 1
				if item not in item_inter:
					item_inter[item] = [0] * (period + 1 * 2)
				item_inter[item][time_stage[time]] += 1

		for user in valid_dict:
			for item in valid_dict[user]:
				time = time_dict[user][item]
				if period:
					pair_stage[(user,item)] = time_stage[time]
				else:
					pair_stage[(user,item)] = time
				stage_all_inter[int(time_stage[time])] += 1
				if item not in item_inter:
					item_inter[item] = [0] * (period + 1 * 2)
				item_inter[item][time_stage[time]] += 1

		for user in test_dict:
			for item in test_dict[user]:
				time = time_dict[user][item]
				if period:
					pair_stage[(user,item)] = time_stage[time]
				else:
					pair_stage[(user,item)] = time
				stage_all_inter[int(time_stage[time])] += 1
				if item not in item_inter:
					item_inter[item] = [0] * (period + 1 * 2)
				item_inter[item][time_stage[time]] += 1

		return pair_stage, item_inter

	def get_stage(self, train_dict, valid_dict, test_dict, time_dict, period):
		time_list = []
		stage_num = period # separate continuous time as discreate time stage.
		time_stage = {}

		# get train data
		for user in train_dict:
			for item in train_dict[user]:
				time_list.append(time_dict[user][item])
		time_list = sorted(time_list)
		# assign the stage of train data
		time_duration = time_list[-1] - time_list[0] # 11days?
		size = math.ceil(time_duration / stage_num) + 1
		for time in time_list:
			time_stage[time] = (time - time_list[0]) // size # time stage 0~7

		# get valid data
		time_list = []
		for user in valid_dict:
			for item in valid_dict[user]:
				time_list.append(time_dict[user][item])
		time_list = sorted(time_list)
		# assign the stage of valid data
		time_duration = time_list[-1] - time_list[0]
		size = math.ceil(time_duration / 1) + 1
		for time in time_list:
			time_stage[time] = (time - time_list[0]) // size + stage_num

		# get test data
		time_list = []
		for user in test_dict:
			for item in test_dict[user]:
				time_list.append(time_dict[user][item])
		time_list = sorted(time_list)
		# assign the stage of valid data
		time_duration = time_list[-1] - time_list[0]
		size = math.ceil(time_duration / 1) + 1
		for time in time_list:
			time_stage[time] = (time - time_list[0]) // size + stage_num + 1

		return time_stage

	def getUserPosItems(self, users):
		posItems = []
		for user in users:
			posItems.append(self.UserItemNet[user].nonzero()[1])
		return posItems

	def getItemPosUsers(self, items):
		posUsers = []
		for item in items:
			posUsers.append(self.UserItemNet[:,item].nonzero()[0])
		return posUsers

	def getUserValidItems(self, users):
		validItems = []
		for user in users:
			if user in self.valid_dict:
				validItems.append(self.valid_dict[user])
		return validItems
	
	def get_pair_bpr(self):
		"""
		"""
		user_num = self.traindataSize
		# print(user_num)
		users = np.random.randint(0, self.n_users, user_num)

		self.user = []
		self.posItem = []
		self.negItem = []
		a = 0
		cnt = 0
		# for i, user in enumerate(users):
		while True:
			for user in users:
				posForUser = self._allPos[user]
				# the users have no positive items -> unseen items
				if len(posForUser) == 0:
					a += 1
					continue
				cnt += 1
				posindex = np.random.randint(0, len(posForUser))
				positem = posForUser[posindex]
				while True:
					negitem = np.random.randint(0, self.m_items)
					if negitem in posForUser:
						continue
					else:
						break
				self.user.append(user)
				self.posItem.append(positem)
				self.negItem.append(negitem)
				if cnt == user_num:
					break
			if cnt == user_num:
				break


	def get_pair_bpr_item(self):
		sample_num = self.traindataSize
		items = np.random.randint(0, self.m_item, sample_num)

		self.item = []
		self.posUser = []
		self.negUser = []

		cnt = 0
		for item in items:
			posForItem = self._allPosUsers[item]
			if len(posForItem) == 0:
				continue
			cnt += 1
			posindex = np.random.randint(0, len(posForItem))
			posuser = posForItem[posindex]

			while True:
				neguser = np.random.randint(0, self.n_user)
				if neguser in posForItem:
					continue
				else:
					break

			self.item.append(item)
			self.posUser.append(posuser)
			self.negUser.append(neguser)
			if cnt == sample_num:
				break

	@property
	def n_users(self):
		return self.n_user
	
	@property
	def m_items(self):
		return self.m_item
	
	@property
	def trainDataSize(self):
		return self.traindataSize
	
	@property
	def trainDict(self):
		return self.train_dict

	@property
	def validDict(self):
		return self.valid_dict
	
	@property
	def pairStage(self):
		return self.pair_stage
	
	@property
	def itemInter(self):
		return self.item_inter

	@property
	def testDict(self):
		return self.test_dict

	@property
	def allPos(self):
		return self._allPos

	# def __getitem__(self, idx):
	# 	return self.user[idx], self.posItem[idx], self.negItem[idx], self.pair_stage[(self.user[idx], self.posItem[idx])], self.item_inter[self.posItem[idx]], self.item_inter[self.negItem[idx]]
	def __getitem__(self, idx):
		return self.item[idx], self.posUser[idx], self.negUser[idx], self.pair_stage[(self.posUser[idx], self.item[idx])]#, self.item_inter[self.posItem[idx]], self.item_inter[self.negItem[idx]]
	
	def __len__(self):
		return self.traindataSize


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
				time = time_dict[user][item]
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
		for i in range(max(item_time_dict.keys())+1):
			try:
				times = np.array(item_time_dict[i])
			except:
				times = np.array([])
			max_pos = max(max_pos, len(times))
			times.sort()
			item_time_array.append(times)
		for i in range(max(item_time_dict.keys())+1):
			item_time_array[i] = np.pad(item_time_array[i], (0, max_pos - len(item_time_array[i])), "constant", constant_values=9999999999)
		item_time_array = np.stack(item_time_array, 0)
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

	def get_pair_user_bpr(self, neg_size):
		sample_num = self.trainDataSize
		users = np.random.randint(0, self.n_user, sample_num)

		self.user_list, self.pos_item_list, self.neg_item_list, self.pos_time_list, self.pos_time_all, self.neg_time_all = [], [], [], [], [], []
		cnt = 0
		for user in users:
			pos_items = self._allPos[user]
			if len(pos_items) == 0:
				continue
			cnt += 1
			idx = np.random.randint(0, len(pos_items))
			pos_item = pos_items[idx]
			while True:
				neg_item = np.random.randint(0, self.m_item, size=neg_size)
				if np.isin(neg_item, pos_item).any():
					continue
				break
			self.user_list.append(user)
			self.pos_item_list.append(pos_item)
			self.neg_item_list.append(neg_item)
			self.pos_time_list.append(self.train_user_item_time[(user, pos_item)])
			self.pos_time_all.append(self.item_time_array[pos_item])
			self.neg_time_all.append(self.item_time_array[neg_item])
			if cnt == sample_num:
				break
		self.user_list = np.array(self.user_list)
		self.pos_item_list = np.array(self.pos_item_list)
		self.neg_item_list = np.array(self.neg_item_list)
		self.pos_time_list = np.array(self.pos_time_list)
		self.pos_time_all = np.array(self.pos_time_all)
		self.neg_time_all = np.array(self.neg_time_all)

	def get_pair_item_bpr(self, neg_size):
		sample_num = self.trainDataSize
		items = np.random.randint(0, self.m_item, sample_num)
		self.item_list, self.pos_user_list, self.neg_user_list, self.item_time_list, self.item_time_all = [], [], [], [], []
		cnt = 0
		for item in items:
			pos_users = self._allPosUsers[item]
			if len(pos_users) == 0:
				continue
			cnt += 1
			idx = np.random.randint(0, len(pos_users))
			pos_user = pos_users[idx]
			while True:
				neg_user = np.random.randint(0, self.n_user, size=neg_size)
				if np.isin(neg_user, pos_users).any():
					continue
				break
			self.item_list.append(item)
			self.pos_user_list.append(pos_user)
			self.neg_user_list.append(neg_user)
			self.item_time_list.append(self.train_user_item_time[(pos_user, item)])
			self.item_time_all.append(self.item_time_array[item])
			if cnt == sample_num:
				break


	def __getitem__(self, idx):
		return self.item_list[idx], self.pos_user_list[idx], self.neg_user_list[idx], self.item_time_list[idx], self.item_time_all[idx]
	
	def __len__(self):
		return self.trainDataSize
