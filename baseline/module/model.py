import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MF(ResidualBase):
    """Matrix Factorization baseline: static user/item embeddings."""
    def encode_user(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("MFResidual requires user_idx.")
        return self.user_embedding(user_idx)


class NCF(ResidualBase):
    """Neural Collaborative Filtering baseline with static user/item embeddings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(8, self.embedding_k)
        layers = [nn.Linear(self.embedding_k * 2, hidden), nn.ReLU()]
        for _ in range(max(0, self.depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1, bias=False)]
        self.mlp = nn.Sequential(*layers)

    def encode_user(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        return self.user_embedding(user_idx)

    def residual_score(self, item_idx, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        u = self.user_embedding(user_idx)
        v = self.get_item_repr(item_idx)
        x = torch.cat([u, v], dim=-1)
        h = self.mlp(x)
        return h

    def score_all_items(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        u = self.user_embedding(user_idx)
        v_all = self.get_item_repr(torch.arange(self.num_items, device=u.device)).unsqueeze(0).expand(u.size(0), -1, -1)
        u_expand = u.unsqueeze(1).expand(-1, self.num_items, -1)
        h = self.mlp(torch.cat([u_expand, v_all], dim=-1)).squeeze(-1)
        return h


class GRURec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_gru = nn.GRU(
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


class MLP4Rec(ResidualBase):
    """History MLP baseline over flattened sequence embeddings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_dim = self.max_seq_len * self.embedding_k
        hidden = max(32, self.embedding_k * 2)
        layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(self.dropout)]
        for _ in range(max(0, self.depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(self.dropout)]
        layers += [nn.Linear(hidden, self.embedding_k)]
        self.mlp = nn.Sequential(*layers)

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


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, padding_item_id):
        super().__init__()
        self.padding_item_id = padding_item_id
        self.pos_emb = nn.Embedding(max_len+1, d_model)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(1, L+1, device=x.device).unsqueeze(0).expand(B, -1)
        hist_mask = x != self.padding_item_id
        pos = hist_mask * pos
        return self.pos_emb(pos)


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs


class SASRec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k, self.padding_item_id)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.embedding_k, eps=1e-8)

        for _ in range(max(1, self.depth)):
            new_attn_layernorm = nn.LayerNorm(self.embedding_k, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  nn.MultiheadAttention(self.embedding_k, self.n_heads, self.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.embedding_k, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.embedding_k, self.dropout)
            self.forward_layers.append(new_fwd_layer)

    def encode_user(self, hist_item_idx, user_idx=None):
        seqs = self.item_embedding(hist_item_idx)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        seqs = seqs + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        u = self.last_layernorm(seqs)[:,-1,:]

        return u



class BERT4Rec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_k,
            nhead=self.n_heads,
            dim_feedforward=max(64, self.embedding_k * 4),
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, self.depth))
        self.pooler = nn.Linear(self.embedding_k, self.embedding_k)

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


MODEL_REGISTRY = {
    "mf": MF,
    "ncf": NCF,
    "grurec": GRURec,
    "sasrec": SASRec,
    "mlp4rec": MLP4Rec,
    "bert4rec": BERT4Rec,
}


def build_model(args, dataset, mini_batch):
    model_name = getattr(args, "model_name", "grurec").lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(
        num_users=dataset.n_user,
        num_items=dataset.m_item,
        embedding_k=args.recdim,
        device=args.device,
        tau=getattr(args, "tau", 1.0),
        depth=getattr(args, "depth", 1),
        max_seq_len=getattr(args, "max_seq_len", 50),
        n_heads=getattr(args, "n_heads", 2),
        dropout=getattr(args, "dropout", 0.1),
        
    ).to(args.device)


def score_pair(model, item_idx, hist_item_idx, user_idx):
    return model.residual_score(item_idx=item_idx, hist_item_idx=hist_item_idx, user_idx=user_idx)


def score_all(model, hist_item_idx, user_idx):
    return model.score_all_items(hist_item_idx=hist_item_idx, user_idx=user_idx)

