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

        self.time_dict = np.load(time_file, allow_pickle=True).item()
        self.train_dict = np.load(train_file, allow_pickle=True).item()
        self.valid_dict = np.load(valid_file, allow_pickle=True).item()
        self.test_dict = np.load(test_file, allow_pickle=True).item()

        self.trainUniqueUsers, self.trainUser, self.trainItem, self.trainDataSize = self.load_set(self.train_dict)
        self.validUniqueUsers, self.validUser, self.validItem, self.validDataSize = self.load_set(self.valid_dict)
        self.testUniqueUsers, self.testUser, self.testItem, self.testDataSize = self.load_set(self.test_dict)

        self.m_item = max(self.trainItem.max(), self.validItem.max(), self.testItem.max()) + 1
        self.n_user = max(self.trainUser.max(), self.validUser.max(), self.testUser.max()) + 1
        self.UserItemNet = csr_matrix(
            (np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )

        self._allPos = self.getUserPosItems(list(range(self.n_user)), self.UserItemNet)
        # self._allPosUsers = self.getItemPosUsers(list(range(self.m_item)), self.UserItemNet)

        self.train_user_item_time = self.set_to_pair(self.train_dict, self.time_dict)
        self.valid_user_item_time = self.set_to_pair(self.valid_dict, self.time_dict)
        self.test_user_item_time = self.set_to_pair(self.test_dict, self.time_dict)

        self.user_interactions = self.build_user_interactions(self.time_dict)
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
        return np.array(UniqueUsers), np.array(User), np.array(Item), dataSize

    def set_to_pair(self, set_dict, time_dict):
        user_item_time = {}
        for user in set_dict:
            for item in set_dict[user]:
                # time = time_dict[user][item] / 60 / 60 / 24
                time = time_dict[user][item]
                user_item_time[(user, item)] = time
        return user_item_time

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

    def build_user_interactions(self, time_dict):
        user_interactions = {u: [] for u in range(self.n_user)}
        for u, item_time in time_dict.items():
            for it, t in item_time.items():
                t_day = float(t)
                # t_day = float(t) / 60 / 60 / 24
                user_interactions[u].append((t_day, it))
        for u in user_interactions:
            user_interactions[u].sort(key=lambda x: x[0])
        return user_interactions

    def build_event_lists(self):
        self.train_events, self.train_hot_events, self.train_cold_events = [], [], []
        for (u,v), t in self.train_user_item_time.items():
                self.train_events.append((u, v, float(t)))
        self.train_events.sort(key=lambda x: (x[0], x[2]))

        u_start = -1
        t_start = 0
        for i, (u, v, t) in enumerate(self.train_events):
            if u == 1150:
                break
            if (u != u_start):
                self.train_cold_events.append((u, v, float(t)))
                u_start = u
                t_start = t 
            elif (u == u_start) & (t == t_start):
                self.train_cold_events.append((u, v, float(t)))
                t_start = t 
            elif (u == u_start) & (t != t_start):
                self.train_hot_events.append((u, v, float(t)))
                t_start = t 
            else:
                raise ValueError("Invalid Sample")

        self.valid_events = [(u, v, float(t)) for (u, v), t in self.valid_user_item_time.items()]
        self.test_events = [(u, v, float(t)) for (u, v), t in self.test_user_item_time.items()]

        self.train_hot_events.sort(key=lambda x: x[2])
        self.train_cold_events.sort(key=lambda x: x[2])
        self.valid_events.sort(key=lambda x: x[2])
        self.test_events.sort(key=lambda x: x[2])

        self.hot_user_list = np.array([u for (u, v, t) in self.train_hot_events], dtype=np.int64)
        self.hot_item_list = np.array([v for (u, v, t) in self.train_hot_events], dtype=np.int64)
        self.hot_event_time_list = np.array([t for (u, v, t) in self.train_hot_events], dtype=np.float32)
        self.hotDataSize = len(self.train_hot_events)

        self.cold_user_list = np.array([u for (u, v, t) in self.train_cold_events], dtype=np.int64)
        self.cold_item_list = np.array([v for (u, v, t) in self.train_cold_events], dtype=np.int64)
        self.cold_event_time_list = np.array([t for (u, v, t) in self.train_cold_events], dtype=np.float32)
        self.coldDataSize = len(self.train_cold_events)

        self.trainDataSize = self.hotDataSize + self.coldDataSize

    def build_user_histories(self, max_seq_len=50):
        self.max_seq_len = max_seq_len

        def _build_histories(events):
            hist_item_list, hist_time_list = [], []
            for (u, v, t) in events:
                hist = [(tt, vv) for (tt, vv) in self.user_interactions[u] if tt < t]
                hist = hist[-max_seq_len:]
                h_items, h_times = [], []
                for (tt, vv) in hist:
                    h_items.append(vv)
                    h_times.append(tt)
                pad_len = max_seq_len - len(h_items)
                h_items = [self.m_item] * pad_len + h_items
                h_times = [h_times[0]] * pad_len + h_times
                hist_item_list.append(h_items)
                hist_time_list.append(h_times)
            return np.array(hist_item_list, dtype=np.int64), np.array(hist_time_list)

        self.train_hist_item_list, self.train_hist_time_list = _build_histories(self.train_hot_events)
        self.valid_hist_item_list, self.valid_hist_time_list = _build_histories(self.valid_events)
        self.test_hist_item_list, self.test_hist_time_list = _build_histories(self.test_events)

    def get_histories_for_users_at_times(self, users, query_times, max_seq_len):
        hist_items_batch = []
        for u, t in zip(users, query_times):
            hist = [(tt, vv) for (tt, vv) in self.user_interactions[int(u)] if tt < float(t)]
            hist = hist[-max_seq_len:]
            h_items = [vv for (tt, vv) in hist]
            pad_len = max_seq_len - len(h_items)
            h_items = [self.m_item] * pad_len + h_items
            hist_items_batch.append(h_items)
        return np.array(hist_items_batch, dtype=np.int64)

    def get_pair_user_uniform(self, k=1):
        pos_user = self.user_list.astype(np.int64)
        N = len(pos_user)
        neg_user = np.random.randint(0, self.n_user - 1, size=(N, k), dtype=np.int64)
        neg_user += (neg_user >= pos_user[:, None])
        self.pos_user_list = pos_user
        self.neg_user_list = neg_user

    def get_pair_item_uniform(self, k=1):
        hot_pos_item = self.hot_item_list.astype(np.int64)
        hot_N = len(hot_pos_item)
        hot_neg_item = np.random.randint(0, self.m_item - 1, size=(hot_N, k), dtype=np.int64)
        hot_neg_item += (hot_neg_item >= hot_pos_item[:, None])
        self.hot_pos_item_list = hot_pos_item
        self.hot_neg_item_list = hot_neg_item

        cold_pos_item = self.cold_item_list.astype(np.int64)
        cold_N = len(cold_pos_item)
        cold_neg_item = np.random.randint(0, self.m_item - 1, size=(cold_N, k), dtype=np.int64)
        cold_neg_item += (cold_neg_item >= cold_pos_item[:, None])
        self.cold_pos_item_list = cold_pos_item
        self.cold_neg_item_list = cold_neg_item
