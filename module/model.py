import torch
import torch.nn as nn


class MF(nn.Module):
    """
    Matrix Factorization
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)
        return out, user_embed, item_embed


class NCF(nn.Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int, depth:int=0):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.depth = depth
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        layers_y1 = [nn.Linear(self.embedding_k*2, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_y1.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_y1.append(nn.ReLU())
        layers_y1.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.y1 = nn.Sequential(*layers_y1)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        out = self.y1(z_embed)
        return out, user_embed, item_embed


class LinearCF(nn.Module):
    """
    Linear Collaborative Filtering
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int):
        super(LinearCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = nn.Linear(self.embedding_k*2, 1, bias=True)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        out = self.linear_1(z_embed)
        return out, user_embed, item_embed


class LDR(nn.Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int, depth:int=0):
        super(LDR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.depth = depth
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

        layers_joint_out = [nn.Linear(self.embedding_k*2, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_joint_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_joint_out.append(nn.ReLU())
        layers_joint_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.joint_out = nn.Sequential(*layers_joint_out)

        layers_user_out = [nn.Linear(self.embedding_k, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_user_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_user_out.append(nn.ReLU())
        layers_user_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.user_out = nn.Sequential(*layers_user_out)

        layers_item_out = [nn.Linear(self.embedding_k, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_item_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_item_out.append(nn.ReLU())
        layers_item_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.item_out = nn.Sequential(*layers_item_out)


    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)

        z_embed = torch.cat([user_embed, item_embed], axis=1)
        joint_out = self.joint_out(z_embed)
        user_out = self.user_out(user_embed)
        item_out = self.item_out(item_embed)

        return joint_out+user_out+item_out, joint_out+user_out, item_out


class LDRMF(nn.Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int, depth:int=0):
        super(LDRMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.depth = depth
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

        layers_user_out = [nn.Linear(self.embedding_k, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_user_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_user_out.append(nn.ReLU())
        layers_user_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.user_out = nn.Sequential(*layers_user_out)

        layers_item_out = [nn.Linear(self.embedding_k, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_item_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_item_out.append(nn.ReLU())
        layers_item_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.item_out = nn.Sequential(*layers_item_out)


    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)

        joint_out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)
        user_out = self.user_out(user_embed)
        item_out = self.item_out(item_embed)

        return joint_out+user_out+item_out, joint_out+user_out, item_out


class SharedNCF(nn.Module):
    """
    Neural Collaborative Filtering with Embedding Sharing
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int, depth:int=0):
        super(SharedNCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.depth = depth
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.ctr = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )
        layers_y1 = [nn.Linear(self.embedding_k*2, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_y1.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_y1.append(nn.ReLU())
        layers_y1.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.y1 = nn.Sequential(*layers_y1)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        ctr = self.ctr(z_embed)
        cvr = self.y1(z_embed)
        ctcvr = torch.mul(nn.Sigmoid()(ctr), nn.Sigmoid()(cvr))
        return cvr, ctr, ctcvr


class SharedMF(nn.Module):
    """
    Matrix Factorization with Embedding Sharing
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int):
        super(SharedMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.ctr = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        ctr = self.ctr(z_embed)
        cvr = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)
        ctcvr = torch.mul(nn.Sigmoid()(ctr), nn.Sigmoid()(cvr))
        return cvr, ctr, ctcvr

class LDRDouble(nn.Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int, depth:int=0):
        super(LDRDouble, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.depth = depth
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

        layers_joint_out = [nn.Linear(self.embedding_k*2, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_joint_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_joint_out.append(nn.ReLU())
        layers_joint_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.joint_out = nn.Sequential(*layers_joint_out)

        layers_user_out = [nn.Linear(self.embedding_k, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_user_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_user_out.append(nn.ReLU())
        layers_user_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.user_out = nn.Sequential(*layers_user_out)

        layers_item_out = [nn.Linear(self.embedding_k, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_item_out.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_item_out.append(nn.ReLU())
        layers_item_out.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.item_out = nn.Sequential(*layers_item_out)


    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)

        z_embed = torch.cat([user_embed, item_embed], axis=1)
        joint_out = self.joint_out(z_embed)
        user_out = self.user_out(user_embed)
        item_out = self.item_out(item_embed)

        return joint_out+user_out+item_out, user_out, item_out
