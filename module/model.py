import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class HawkesMFDebias(nn.Module):
    """
    Shared-item model:
      - Hawkes prior: mu_phi(z_v) + alpha_phi(z_v) * h_v(t)
      - Residual model: MF score w_u^T z_v
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        prior_hidden_dim: int,
        prior_depth: int,
        tau: float,
        prior_vector_norm: bool,
        residual_vector_norm: bool,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.softplus = nn.Softplus()

        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.residual_vector_norm = L2Normalize(dim=-1) if residual_vector_norm else nn.Identity()
        self.prior_vector_norm = L2Normalize(dim=-1) if prior_vector_norm else nn.Identity()

        self.base_net = self._build_mlp(embedding_dim, prior_hidden_dim, prior_depth)
        self.excitation_net = self._build_mlp(embedding_dim, prior_hidden_dim, prior_depth)
        self.log_beta = nn.Parameter(torch.zeros(()))
        self.log_beta = nn.Parameter(torch.zeros(()))

        self.reset_parameters()
        self.tau = tau

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Softplus())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1, bias=False))
        return nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        for module in list(self.base_net) + list(self.excitation_net):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def current_beta(self) -> torch.Tensor:
        return self.softplus(self.log_beta) + 1e-6

    def prior_parameters_from_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.prior_vector_norm(self.item_embedding.weight)
        mu = self.softplus(self.base_net(z)).squeeze(-1) + 1e-8
        alpha = self.softplus(self.excitation_net(z)).squeeze(-1) + 1e-8
        beta = self.current_beta()
        return mu, alpha, beta

    def prior_logits_for_candidates(
        self,
        candidate_items: torch.Tensor,
        query_times: torch.Tensor,
        candidate_histories: torch.Tensor,
    ) -> torch.Tensor:
        """
        candidate_items: [B, C]
        query_times: [B]
        candidate_histories: [B, C, L], train-item times padded with a large pad value
        return: intensity logits [B, C]
        """
        item_vec = self.prior_vector_norm(self.item_embedding(candidate_items))
        mu = self.softplus(self.base_net(item_vec)).squeeze(-1) + 1e-8
        alpha = self.softplus(self.excitation_net(item_vec)).squeeze(-1) + 1e-8
        beta = self.current_beta()

        query = query_times.view(-1, 1, 1)
        mask = candidate_histories < query
        delta = (query - candidate_histories).clamp(min=0.0)
        h = (torch.exp(-beta * delta) * mask).sum(dim=-1)
        return mu + alpha * h

    def residual_scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = self.residual_vector_norm(self.user_embedding(users))
        item_vec = self.residual_vector_norm(self.item_embedding(items))
        return torch.sum(user_vec * item_vec, dim=-1)

    def residual_scores_all_items(self, users: torch.Tensor) -> torch.Tensor:
        user_vec = self.residual_vector_norm(self.user_embedding(users))
        item_vec = self.residual_vector_norm(self.item_embedding.weight)
        return user_vec @ item_vec.t()


class L2Normalize(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
