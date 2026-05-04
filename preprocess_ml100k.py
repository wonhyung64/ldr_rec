#%%
import numpy as np
import pandas as pd



#%%
file_dir = "./data"
df = pd.read_csv(f"{file_dir}/ml-100k/u.data", sep="\t", header=None)
df.columns = ["user_id", "item_id", "rating", "timestamp"]


if df["user_id"].min() == 1:
    df["user_id"] = df["user_id"] - 1
if df["item_id"].min() == 1:
    df["item_id"] = df["item_id"] - 1

print(df["rating"].value_counts())

df = df.loc[df["rating"] >= 4, ["user_id", "item_id", "timestamp"]].reset_index(drop=True)
df = df.sort_values("timestamp")


print()
print(df.head())


total_sample_num = len(df)

train_time_upper = df["timestamp"].quantile(0.8)
valid_time_upper = df["timestamp"].quantile(0.9)
train_idx = df[df["timestamp"] <= train_time_upper].index
valid_idx = df[(df["timestamp"] > train_time_upper) & (df["timestamp"] <= valid_time_upper)].index
test_idx = df[df["timestamp"] > valid_time_upper].index

# %%

meta = pd.read_csv(f"{file_dir}/ml-100k/u.genre", sep="\t", header=None)
meta = pd.read_csv(f"{file_dir}/ml-100k/u.item", sep="\t", header=None, )
df.columns = [["user_id", "item_id", "rating", "timestamp"]]
print(df.head())

pd.read_csv(f"{file_dir}/ml-100k/u.info", sep="\t", header=None, )
pd.read_csv(f"{file_dir}/ml-100k/u.genre", sep="\t", header=None, )
pd.read_csv(f"{file_dir}/ml-100k/u.user", sep="\t", header=None, )
pd.read_csv(f"{file_dir}/ml-100k/u.occupation", sep="\t", header=None, )

#%%
meta = pd.read_csv(
    f"{file_dir}/ml-100k/u.item",
    sep="|",                  # u.item은 보통 | 구분자로 읽습니다
    header=None,
    encoding="ISO-8859-1",    # 또는 latin-1
    engine="python"
)
meta.columns = [
    "item_id", "movie_title", "release_date", "video_release_date",
    "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
    "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir"," Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
    "Thriller", "War", "Western",
]
print(meta.head())


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
    
np.save(f"{file_dir}/ml-100k/training_dict.npy", _train_dict, allow_pickle=True)
np.save(f"{file_dir}/ml-100k/testing_dict.npy", _test_dict, allow_pickle=True)
np.save(f"{file_dir}/ml-100k/validation_dict.npy", _valid_dict, allow_pickle=True)
np.save(f"{file_dir}/ml-100k/interaction_time_dict.npy", _time_dict, allow_pickle=True)



# %%
