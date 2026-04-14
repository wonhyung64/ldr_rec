import torch
import torch.nn as nn


class ResidualBase(nn.Module):
    def __init__(self, num_users, num_items, embedding_k, device, tau=1.0, depth=1, max_seq_len=50, n_heads=2, dropout=0.1, norm_first = True):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.dropout = dropout
        self.padding_item_id = self.num_items
        self.norm_first = norm_first

        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_k, padding_idx=self.padding_item_id)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)

    def encode_user(self, hist_item_idx, user_idx=None):
        raise NotImplementedError

    def get_item_repr(self, item_idx):
        return self.item_embedding(item_idx)

    def residual_score(self, item_idx, hist_item_idx, user_idx=None):
        u = self.encode_user(hist_item_idx, user_idx=user_idx)
        mini_batch, recdim = u.shape
        v = self.get_item_repr(item_idx).reshape(mini_batch, -1, recdim)
        # u = F.normalize(u, dim=-1, eps=1e-8)
        # v = F.normalize(v, dim=-1, eps=1e-8)
        h = torch.sum(u.unsqueeze(1) * v, dim=-1)
        return h

    def score_all_items(self, hist_item_idx, user_idx=None):
        u = self.encode_user(hist_item_idx, user_idx=user_idx)
        v_all = self.get_item_repr(torch.arange(self.num_items, device=hist_item_idx.device))
        h = torch.matmul(u, v_all.T)
        return h
