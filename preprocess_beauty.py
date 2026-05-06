#%%
import numpy as np
import pandas as pd


#%%
file_dir = "./data"
file = f"{file_dir}/beauty/All_Beauty.jsonl"
raw_df = pd.read_json(file, lines=True)


#%%
df = raw_df[["user_id", "parent_asin", "rating", "timestamp"]]
df.columns = ["user_id", "item_id", "rating", "timestamp"]
df["user_id"], user_map = pd.factorize(df["user_id"])
df["item_id"], item_map = pd.factorize(df["item_id"])
df["timestamp"] = df["timestamp"].view("int64") // 10**9

if df["user_id"].min() == 1:
    df["user_id"] = df["user_id"] - 1
if df["item_id"].min() == 1:
    df["item_id"] = df["item_id"] - 1

print(df["rating"].value_counts())

#%%
df = df.loc[df["rating"] >= 4, ["user_id", "item_id", "timestamp"]].reset_index(drop=True)
# df = df[df["item_id"].isin(df["item_id"].value_counts()[(df["item_id"].value_counts() >= 5)].index.tolist())]

# 1. 현재 df에 실제 등장하는 old item id만 추출
used_old_item_ids = sorted(df["item_id"].unique())

# 2. old item id -> new item id
old2new_item = {old_id: new_id for new_id, old_id in enumerate(used_old_item_ids)}

# 3. new item id -> old item id
new2old_item = {new_id: old_id for old_id, new_id in old2new_item.items()}

# 4. new item id -> ASIN
new2asin = {new_id: item_map[old_id] for new_id, old_id in new2old_item.items()}

# 5. ASIN -> new item id
asin2new = {asin: new_id for new_id, asin in new2asin.items()}

# 6. df의 item_id를 연속된 id로 교체
df["item_id"] = df["item_id"].map(old2new_item)

df["item_id"].min(), df["item_id"].max(), df["item_id"].nunique()


#%%
train_time_upper = df["timestamp"].quantile(0.8)
valid_time_upper = df["timestamp"].quantile(0.9)
train_idx = df[df["timestamp"] <= train_time_upper].index
valid_idx = df[(df["timestamp"] > train_time_upper) & (df["timestamp"] <= valid_time_upper)].index
test_idx = df[df["timestamp"] > valid_time_upper].index


# %%
from tqdm import tqdm

_train_dict, _valid_dict, _test_dict, _time_dict = {}, {}, {}, {}
df = df.sort_values(["user_id", "timestamp"])

for u in tqdm(df["user_id"].unique().tolist()):
    u_df = df[df["user_id"] == u][["item_id", "timestamp"]]
    train_time_bool = u_df["timestamp"] <= train_time_upper
    valid_time_bool = (u_df["timestamp"] > train_time_upper) & (u_df["timestamp"] <= valid_time_upper)
    test_time_bool = u_df["timestamp"] > valid_time_upper

    _train_dict[u] = u_df[train_time_bool]["item_id"].tolist()
    _valid_dict[u] = u_df[valid_time_bool]["item_id"].tolist()
    _test_dict[u] = u_df[test_time_bool]["item_id"].tolist()
    _time_dict[u] = dict(zip(u_df["item_id"], u_df["timestamp"]))


#%%
train_dict, valid_dict, test_dict, time_dict = {}, {}, {}, {}
train_user_list = list(_train_dict.keys())
train_user_list.sort()

user_idx = 0
exclude_num = 0
for u in train_user_list:
    v = _train_dict[u]
    if len(v) < 5:
        exclude_num += 1
    else:
        train_dict[user_idx] = _train_dict[u]
        valid_dict[user_idx] = _valid_dict[u]
        test_dict[user_idx] = _test_dict[u]
        time_dict[user_idx] = _time_dict[u]
        user_idx += 1


len(train_dict)
#%%
np.save(f"{file_dir}/beauty/training_dict.npy", train_dict, allow_pickle=True)
np.save(f"{file_dir}/beauty/testing_dict.npy", test_dict, allow_pickle=True)
np.save(f"{file_dir}/beauty/validation_dict.npy", valid_dict, allow_pickle=True)
np.save(f"{file_dir}/beauty/interaction_time_dict.npy", time_dict, allow_pickle=True)


# %%
user_set, item_set = [], []
train_num, valid_num, test_num = 0, 0, 0

for k,v in train_dict.items():
    user_set.append(k)
    item_set.extend(v)
    train_num += len(v)

for k,v in valid_dict.items():
    user_set.append(k)
    item_set.extend(v)
    valid_num += len(v)

for k,v in test_dict.items():
    user_set.append(k)
    item_set.extend(v)
    test_num += len(v)

print(f"user num: {len(set(user_set))} / item num: {len(set(item_set))}")
print(f"train num: {train_num} / valid num: {valid_num} / test num: {test_num}")
# %%
train_dict = np.load(f"{file_dir}/beauty/training_dict.npy", allow_pickle=True).item()
test_dict = np.load(f"{file_dir}/beauty/testing_dict.npy", allow_pickle=True).item()

train_items = []
for k,v in train_dict.items():
    train_items.extend(v)

test_items = []
for k,v in test_dict.items():
    test_items.extend(v)

# %%
train = list(set(train_items))

train.sort()


test = list(set(test_items))
test.sort()

train

tmp = [i in train for i in test]
sum(tmp) / len(tmp)