#%%
import numpy as np
import pandas as pd



#%%
file_dir = "./data"
# df = pd.read_csv(f"{file_dir}/ml-100k/u.data", sep="\t", header=None)
df = pd.read_csv(f"{file_dir}/ml-1m/ratings.dat", sep="::", header=None)
df.columns = ["user_id", "item_id", "rating", "timestamp"]


if df["user_id"].min() == 1:
    df["user_id"] = df["user_id"] - 1
if df["item_id"].min() == 1:
    df["item_id"] = df["item_id"] - 1

print(df["rating"].value_counts())

#%%
df = df.loc[df["rating"] >= 4, ["user_id", "item_id", "timestamp"]].reset_index(drop=True)
# df = df.sort_values("timestamp")


#%%
df["user_id"].value_counts()


#%%


#%%
print()
print(df.head())


total_sample_num = len(df)


#%%



#%%
train_time_upper = df["timestamp"].quantile(0.8)
valid_time_upper = df["timestamp"].quantile(0.9)
train_idx = df[df["timestamp"] <= train_time_upper].index
valid_idx = df[(df["timestamp"] > train_time_upper) & (df["timestamp"] <= valid_time_upper)].index
test_idx = df[df["timestamp"] > valid_time_upper].index


#%%


# %%
_train_dict, _valid_dict, _test_dict, _time_dict = {}, {}, {}, {}
df = df.sort_values(["user_id", "timestamp"])

for u in df["user_id"].unique().tolist():
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
        _time_dict[0][3185]
    else:
        train_dict[user_idx] = _train_dict[u]
        valid_dict[user_idx] = _valid_dict[u]
        test_dict[user_idx] = _test_dict[u]
        time_dict[user_idx] = _time_dict[u]
        user_idx += 1


#%%
np.save(f"{file_dir}/ml-1m/training_dict.npy", train_dict, allow_pickle=True)
np.save(f"{file_dir}/ml-1m/testing_dict.npy", test_dict, allow_pickle=True)
np.save(f"{file_dir}/ml-1m/validation_dict.npy", valid_dict, allow_pickle=True)
np.save(f"{file_dir}/ml-1m/interaction_time_dict.npy", time_dict, allow_pickle=True)


# %%
