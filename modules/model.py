#%%
import torch
import numpy as np
import torch.nn as nn


class JointRec(nn.Module):
    """
    Model for Joint Probability P(u,v,t) = P(u|t,v)P(t|v)p(v)
    """
    def __init__(self, num_users:int, num_items:int, num_stages:int, embedding_k:int, all_popularity:int, device, depth:int=0):
        super(JointRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_stages = num_stages
        self.embedding_k = embedding_k
        self.depth = depth
        self.all_popularity = all_popularity
        self.stage_idx_all = torch.tensor([s for s in range(self.num_stages)]).to(device)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.stage_embedding = nn.Sequential(*[
	        nn.Linear(3, self.embedding_k//2, bias=False),
            nn.ReLU(),
            nn.Linear(self.embedding_k//2, self.embedding_k, bias=False),
            nn.ReLU(),
	        ])

        self.intensity_decay = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(device)
        self.user_conformity = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(device)

        """HAWKES PROCESS"""
        item_intensity = [nn.Linear(self.embedding_k, self.embedding_k//2), nn.ReLU()]
        for _ in range(depth):
            item_intensity.append(nn.Linear(self.embedding_k//2, self.embedding_k//2))
            item_intensity.append(nn.ReLU())
        item_intensity.append(nn.Linear(self.embedding_k//2, 1, bias=False))
        item_intensity.append(nn.Softplus())
        self.item_intensity = nn.Sequential(*item_intensity)

        item_stage_amplitude = [nn.Linear(self.embedding_k*2, self.embedding_k), nn.Tanh()]
        for _ in range(depth):
            item_stage_amplitude.append(nn.Linear(self.embedding_k, self.embedding_k))
            item_stage_amplitude.append(nn.Tanh())
        item_stage_amplitude.append(nn.Linear(self.embedding_k, 1, bias=False))
        item_stage_amplitude.append(nn.Tanh())
        self.item_stage_amplitude = nn.Sequential(*item_stage_amplitude)

        """USER ITEM STAGE INTERACTION"""
        layers_y = [nn.Linear(self.embedding_k*2, self.embedding_k), nn.ReLU()]
        for _ in range(depth):
            layers_y.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_y.append(nn.ReLU())
        layers_y.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.y = nn.Sequential(*layers_y)

    def forward(self, sub_x, sub_pop, mask):
        user_idx = sub_x[:,0]
        item_idx = sub_x[:,1]
        stage_idx = sub_x[:,2]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        log_p_tv = self.log_hawkes(item_embed, stage_idx, self.stage_idx_all, sub_pop, mask)
        log_p_utv = self.y(z_embed) + log_p_tv.detach() * self.user_conformity
        return log_p_utv, log_p_tv

    def log_hawkes(self, item_embed, stage_idx, stage_idx_all, sub_pop, mask):
        B, D = item_embed.shape
        item_expand  = item_embed.unsqueeze(1).expand(B, S, D)
        item_stage_embed = torch.cat([item_expand, stage_expand], dim=-1) 
        amplitude = nn.Softplus()(self.item_stage_amplitude(item_stage_embed))
        intensity = self.item_intensity(item_embed)
        density = self.hawkes_density(sub_pop, mask, intensity, amplitude, stage_idx_all)
        partition = self.hawkes_partition(stage_idx_all, sub_pop, intensity, amplitude)
        return torch.log(density / partition)

    def stage_feature(self, stage):
        return torch.stack([stage, stage**2, torch.log1p(stage)], dim=-1)

    def hawkes_density(self, sub_pop, mask, intensity, amplitude, stage_idx_all):
        return intensity + (sub_pop * mask * amplitude * self.exp_kernel(mask.sum(1)-1, stage_idx_all)).sum(1)

    def exp_kernel(self, last_stage_idx, stage_idx_all):
        return (nn.Softplus()(self.intensity_decay) * torch.exp(-nn.Softplus()(self.intensity_decay) * (last_stage_idx - stage_idx_all))).unsqueeze(-1)

    def hawkes_partition(self, stage_idx_all, sub_pop, intensity, amplitude):
        return intensity * self.num_stages + (sub_pop * amplitude * self.closed_kernel(stage_idx_all)).sum(1)

    def closed_kernel(self, stage_idx_all):
        return (1 - torch.exp(-nn.Softplus()(self.intensity_decay) * (self.num_stages - stage_idx_all))).reshape(1,-1,1)


#%%
if __name__ == "__main__":
    """OPTIONS"""
    num_users = 2
    num_items = 3
    num_stages = 2
    embedding_k = 64
    depth = 0
    device = "cuda"
    batch_size = 2


    """DATA-LOADER"""
    uvt_interaction = np.array([[[1,0], [0,0], [1,1]], [[1,1], [0,1], [0,1]]])
    train_pop = torch.tensor(uvt_interaction.sum(axis=0)).to(device)
    all_x = np.argwhere(uvt_interaction == 1)

    sub_x = all_x[np.random.choice(all_x.shape[0], size=batch_size, replace=False)]
    sub_x = torch.tensor(sub_x).to(device)
    v = sub_x[:, 1].cpu().numpy()
    t = sub_x[:, 2].cpu().numpy()
    mask = torch.tensor(np.arange(num_stages)[None, :] < t[:, None]).unsqueeze(-1).to(device)
    sub_pop = train_pop[v].unsqueeze(-1)

    """IMPLEMENTATION"""
    model = JointRec(num_users, num_items, num_stages, embedding_k, uvt_interaction.sum(), "cuda")
    model = model.to("cuda")
    print(model(sub_x.to("cuda"), sub_pop.to("cuda"), mask.to("cuda")))


#%%
# device = "cuda"
user_embed_fn = nn.Embedding(num_users, embedding_k)
item_embed_fn = nn.Embedding(num_items, embedding_k)
stage_embed_fn = nn.Sequential(*[
	nn.Linear(3, embedding_k//2, bias=False),
    nn.ReLU(),
    nn.Linear(embedding_k//2, embedding_k, bias=False),
    nn.ReLU(),
	])
stage_idx_all = torch.tensor([s for s in range(num_stages)]).to(device)

beta = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(device)
alpha = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(device)
user_conformity = torch.autograd.Variable(torch.randn(1), requires_grad=True).to(device)

"""HAWKES PROCESS"""
item_intensity = [nn.Linear(embedding_k, embedding_k//2), nn.ReLU()]
for _ in range(depth):
    item_intensity.append(nn.Linear(embedding_k//2, embedding_k//2))
    item_intensity.append(nn.ReLU())
item_intensity.append(nn.Linear(embedding_k//2, 1, bias=False))
item_intensity.append(nn.Softplus())
item_intensity_fn = nn.Sequential(*item_intensity)

"""USER ITEM STAGE INTERACTION"""
layers_y = [nn.Linear(embedding_k*2, embedding_k), nn.ReLU()]
for _ in range(depth):
    layers_y.append(nn.Linear(embedding_k, embedding_k))
    layers_y.append(nn.ReLU())
layers_y.append(nn.Linear(embedding_k, 1, bias=False))
y_fn = nn.Sequential(*layers_y)

#%%
item_embed_fn = item_embed_fn.to(device)
item_intensity_fn = item_intensity_fn.to(device)
user_embed_fn = user_embed_fn.to(device)
#%%

"""# def update_item():"""
all_item_embed = []
all_item_intensity = []
for idx in range(0, num_items, batch_size):
    idx_list = list(range(num_items))[idx : idx+batch_size]
    item_embed = item_embed_fn(torch.tensor(idx_list).to(device))
    all_item_embed.append(item_embed)
    all_item_intensity.append(item_intensity_fn(item_embed))
all_item_embed = torch.concat(all_item_embed, 0)
all_item_intensity = torch.concat(all_item_intensity, 0)


"""'''forward'''"""
user_idx = sub_x[:,0]
item_idx = sub_x[:,1]
stage_idx = sub_x[:,2]


"""hawkes density"""
item_intensity = all_item_intensity[item_idx]
kernel = (alpha * torch.exp(-beta * (stage_idx.unsqueeze(-1) - stage_idx_all))).unsqueeze(-1)
density = nn.ReLU()(item_intensity + (sub_pop * kernel * mask).sum(1))

"""hawkes partition"""



"""density ratio"""
user_embed = user_embed_fn(user_idx)
z_embed = torch.cat([user_embed, all_item_embed[item_idx]], axis=1)
# log_p_tv = self.log_hawkes(item_embed, stage_idx, self.stage_idx_all, sub_pop, mask)
#         log_p_utv = self.y(z_embed) + log_p_tv.detach() * self.user_conformity


# def exp_kernel(self, last_stage_idx, stage_idx_all):
all_item_intensity = item_intensity(item_embedding.weight)
sub_pop

mask

B, D = item_embed.shape
item_expand  = item_embed.unsqueeze(1).expand(B, S, D)
item_stage_embed = torch.cat([item_expand, stage_expand], dim=-1) 
amplitude = nn.Softplus()(self.item_stage_amplitude(item_stage_embed))
density = self.hawkes_density(sub_pop, mask, intensity, amplitude, stage_idx_all)
partition = self.hawkes_partition(stage_idx_all, sub_pop, intensity, amplitude)
return torch.log(density / partition)
#
# %%
