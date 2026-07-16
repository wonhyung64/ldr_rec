#%%
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


# %%
path1 = "/Users/wonhyung64/Github/ldr_rec/collapse_analysis/embeddings/ours_mf_micro_video_seed1.npz"
path2 = "/Users/wonhyung64/Github/ldr_rec/collapse_analysis/embeddings/vanilla_mf_micro_video_seed1.npz"


#%%
contents1 = np.load(path1)
contents2 = np.load(path2)

assert np.all(contents1["user_ids"] == contents2["user_ids"])
user_ids = np.load(path1)["user_ids"]

assert np.all(contents1["pos_items"] == contents2["pos_items"])
pos_items = np.load(path1)["pos_items"]

assert np.all(contents1["item_popularity"] == contents2["item_popularity"])
item_popularity = np.load(path1)["item_popularity"]

ours_user_emb = contents1["h_u"]
vanilla_user_emb = contents2["h_u"]

ours_item_emb = contents1["z_v"]
vanilla_item_emb = contents2["z_v"]


# %%
ours_user_unit = nn.functional.normalize(torch.tensor(ours_user_emb), dim=-1, eps=1e-8)
ours_item_unit = nn.functional.normalize(torch.tensor(ours_item_emb), dim=-1, eps=1e-8)
vanilla_user_unit = nn.functional.normalize(torch.tensor(vanilla_user_emb), dim=-1, eps=1e-8)
vanilla_item_unit = nn.functional.normalize(torch.tensor(vanilla_item_emb), dim=-1, eps=1e-8)


# %%
best_item_idx = item_popularity.argmax()


#%%
ours_cossim = torch.matmul(ours_item_unit[best_item_idx], ours_user_unit.T)
vanilla_cossim = torch.matmul(vanilla_item_unit[best_item_idx], vanilla_user_unit.T)


#%%
plt.hist(ours_cossim, color="blue", bins=50)
plt.hist(vanilla_cossim, color="orange", bins=50)

