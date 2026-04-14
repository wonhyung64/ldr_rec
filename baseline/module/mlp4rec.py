import torch
from .base import ResidualBase


class MLP4Rec(ResidualBase):
    """History MLP baseline over flattened sequence embeddings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_dim = self.max_seq_len * self.embedding_k
        hidden = max(32, self.embedding_k * 2)
        layers = [torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
        for _ in range(max(0, self.depth - 1)):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
        layers += [torch.nn.Linear(hidden, self.embedding_k)]
        self.mlp = torch.nn.Sequential(*layers)

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_emb = self.item_embedding(hist_item_idx)
        flat = hist_emb.reshape(hist_emb.size(0), -1)
        u = self.mlp(flat)
        hist_mask = (hist_item_idx != self.padding_item_id)
        no_hist = (hist_mask.sum(dim=-1) == 0)
        if no_hist.any():
            fallback = self.user_embedding(user_idx) if user_idx is not None else torch.zeros_like(u)
            u = torch.where(no_hist.unsqueeze(-1), fallback, u)
        return u

