import torch
from .base import ResidualBase


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model, padding_item_id):
        super().__init__()
        self.padding_item_id = padding_item_id
        self.pos_emb = torch.nn.Embedding(max_len+1, d_model)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(1, L+1, device=x.device).unsqueeze(0).expand(B, -1)
        hist_mask = x != self.padding_item_id
        pos = hist_mask * pos
        return self.pos_emb(pos)


class BERT4Rec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k, self.padding_item_id)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embedding_k,
            nhead=self.n_heads,
            dim_feedforward=max(64, self.embedding_k * 4),
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=max(1, self.depth))
        self.pooler = torch.nn.Linear(self.embedding_k, self.embedding_k)

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_mask = (hist_item_idx != self.padding_item_id)
        x = self.item_embedding(hist_item_idx)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=~hist_mask)

        mask_f = hist_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        u = (x * mask_f).sum(dim=1) / denom
        u = self.pooler(u)

        no_hist = (hist_mask.sum(dim=-1) == 0)
        if no_hist.any():
            fallback = self.user_embedding(user_idx) if user_idx is not None else torch.zeros_like(u)
            u = torch.where(no_hist.unsqueeze(-1), fallback, u)
        return u

