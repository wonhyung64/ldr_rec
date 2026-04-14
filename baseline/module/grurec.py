import torch
from .base import ResidualBase


class GRURec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_gru = torch.nn.GRU(
            input_size=self.embedding_k,
            hidden_size=self.embedding_k,
            num_layers=1,
            batch_first=True,
        )

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_emb = self.item_embedding(hist_item_idx)
        _, h_n = self.user_gru(hist_emb)
        u = h_n[-1]
        hist_mask = (hist_item_idx != self.padding_item_id)
        no_hist = (hist_mask.sum(dim=-1) == 0)
        if no_hist.any():
            fallback = self.user_embedding(user_idx) if user_idx is not None else torch.zeros_like(u)
            u = torch.where(no_hist.unsqueeze(-1), fallback, u)
        return u
