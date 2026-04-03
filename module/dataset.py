import os
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, List, Sequence, Tuple


class InteractionData:
    def __init__(self, data_path: str, dataset: str) -> None:
        path = os.path.join(data_path, dataset)
        self.time_dict = np.load(os.path.join(path, "interaction_time_dict.npy"), allow_pickle=True).item()
        self.train_dict = np.load(os.path.join(path, "training_dict.npy"), allow_pickle=True).item()
        self.valid_dict = np.load(os.path.join(path, "validation_dict.npy"), allow_pickle=True).item()
        self.test_dict = np.load(os.path.join(path, "testing_dict.npy"), allow_pickle=True).item()

        self.train_users, self.train_items = self._flatten_dict(self.train_dict)
        self.valid_users, self.valid_items = self._flatten_dict(self.valid_dict)
        self.test_users, self.test_items = self._flatten_dict(self.test_dict)

        self.num_users = int(max(self.train_users.max(initial=0), self.valid_users.max(initial=0), self.test_users.max(initial=0)) + 1)
        self.num_items = int(max(self.train_items.max(initial=0), self.valid_items.max(initial=0), self.test_items.max(initial=0)) + 1)

        self.user_item_net = csr_matrix(
            (np.ones(len(self.train_users)), (self.train_users, self.train_items)),
            shape=(self.num_users, self.num_items),
        )
        self.all_train_pos_items = self._get_user_pos_items()

        self.train_events = self._build_events(self.train_dict)
        self.valid_events = self._build_events(self.valid_dict)
        self.test_events = self._build_events(self.test_dict)

        self.train_item_time_padded, self.train_time_pad_value = self._build_train_item_time_array()

    def _flatten_dict(self, split_dict: Dict[int, Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        users: List[int] = []
        items: List[int] = []
        for u, item_list in split_dict.items():
            for v in item_list:
                users.append(int(u))
                items.append(int(v))
        if not users:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)

    def _build_events(self, split_dict: Dict[int, Sequence[int]]) -> List[Tuple[int, int, float]]:
        events: List[Tuple[int, int, float]] = []
        for u, item_list in split_dict.items():
            for v in item_list:
                t_raw = self.time_dict[u][v]
                t_day = float(t_raw) / 60.0 / 60.0 / 24.0
                events.append((int(u), int(v), t_day))
        events.sort(key=lambda x: x[2])
        return events

    def _get_user_pos_items(self) -> List[np.ndarray]:
        return [self.user_item_net[u, :].nonzero()[1] for u in range(self.num_users)]

    def _build_train_item_time_array(self) -> Tuple[np.ndarray, float]:
        item_times: List[List[float]] = [[] for _ in range(self.num_items)]
        for _, v, t in self.train_events:
            item_times[v].append(float(t))
        max_len = max((len(x) for x in item_times), default=0)
        max_time = max((max(x) for x in item_times if x), default=0.0) + 1.0
        padded: List[np.ndarray] = []
        for times in item_times:
            arr = np.asarray(sorted(times), dtype=np.float32)
            if max_len > len(arr):
                arr = np.pad(arr, (0, max_len - len(arr)), mode="constant", constant_values=max_time)
            padded.append(arr)
        if max_len == 0:
            return np.full((self.num_items, 0), max_time, dtype=np.float32), max_time
        return np.stack(padded, axis=0), max_time
