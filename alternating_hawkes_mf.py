import argparse
import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FenwickTree:
    """Fenwick tree for positive weights, used by the excitation sampler."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.tree = np.zeros(self.size + 1, dtype=np.float64)

    def add(self, index: int, delta: float) -> None:
        i = int(index) + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def total(self) -> float:
        return float(self.prefix_sum(self.size - 1))

    def prefix_sum(self, index: int) -> float:
        i = int(index) + 1
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def sample(self, mass: float) -> int:
        if mass < 0.0 or mass >= self.total() + 1e-9:
            raise ValueError("mass must be in [0, total)")
        idx = 0
        bit_mask = 1 << (self.size.bit_length() - 1)
        running = 0.0
        while bit_mask != 0:
            nxt = idx + bit_mask
            if nxt <= self.size and running + self.tree[nxt] <= mass:
                idx = nxt
                running += self.tree[nxt]
            bit_mask >>= 1
        return min(idx, self.size - 1)


class BaseSampler:
    """CDF sampler for the frozen base distribution p_base(v) ∝ mu_v."""

    def __init__(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        self.cdf = np.cumsum(weights / total)
        self.cdf[-1] = 1.0

    def sample(self) -> int:
        u = np.random.rand()
        return int(np.searchsorted(self.cdf, u, side="right"))


@dataclass
class PriorSnapshot:
    mu: np.ndarray
    alpha: np.ndarray
    beta: float
    base_mass: float
    base_sampler: BaseSampler


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
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.softplus = nn.Softplus()

        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        self.base_net = self._build_mlp(embedding_dim, prior_hidden_dim, prior_depth)
        self.excitation_net = self._build_mlp(embedding_dim, prior_hidden_dim, prior_depth)
        self.log_beta = nn.Parameter(torch.zeros(()))

        self.reset_parameters()

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
        z = self.item_embedding.weight
        mu = self.softplus(self.base_net(z)).squeeze(-1) + 1e-8
        alpha = self.softplus(self.excitation_net(z)).squeeze(-1) + 1e-8
        beta = self.current_beta()
        return mu, alpha, beta

    def residual_scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(users)
        item_vec = self.item_embedding(items)
        return torch.sum(user_vec * item_vec, dim=-1)

    def residual_scores_all_items(self, users: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(users)
        item_vec = self.item_embedding.weight
        return user_vec @ item_vec.t()


class InteractionData:
    def __init__(self, data_path: str, dataset: str) -> None:
        path = os.path.join(data_path, dataset)
        self.time_dict = np.load(os.path.join(path, "interaction_time_dict.npy"), allow_pickle=True).item()
        self.train_dict = np.load(os.path.join(path, "training_dict.npy"), allow_pickle=True).item()
        self.valid_dict = np.load(os.path.join(path, "validation_dict.npy"), allow_pickle=True).item()
        self.test_dict = np.load(os.path.join(path, "testing_dict.npy"), allow_pickle=True).item()

        self.train_users, self.train_items = self._flatten_dict(self.train_dict)
        self.valid_users, self.valid_items = self._flatten_dict(self.valid_dict)
        self.test_users, self.test_items = self._flatten_dict(self.test_dict)

        self.num_users = int(max(self.train_users.max(initial=0), self.valid_users.max(initial=0), self.test_users.max(initial=0)) + 1)
        self.num_items = int(max(self.train_items.max(initial=0), self.valid_items.max(initial=0), self.test_items.max(initial=0)) + 1)

        self.user_item_net = csr_matrix(
            (np.ones(len(self.train_users)), (self.train_users, self.train_items)),
            shape=(self.num_users, self.num_items),
        )
        self.all_train_pos_items = self._get_user_pos_items()

        self.train_events = self._build_events(self.train_dict)
        self.valid_events = self._build_events(self.valid_dict)
        self.test_events = self._build_events(self.test_dict)

        # Train-only item histories are used for prior fitting and for evaluation without leakage.
        self.train_item_time_padded, self.train_time_pad_value = self._build_train_item_time_array()

    def _flatten_dict(self, split_dict: Dict[int, Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        users: List[int] = []
        items: List[int] = []
        for u, item_list in split_dict.items():
            for v in item_list:
                users.append(int(u))
                items.append(int(v))
        if not users:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)

    def _build_events(self, split_dict: Dict[int, Sequence[int]]) -> List[Tuple[int, int, float]]:
        events: List[Tuple[int, int, float]] = []
        for u, item_list in split_dict.items():
            for v in item_list:
                t_raw = self.time_dict[u][v]
                t_day = float(t_raw) / 60.0 / 60.0 / 24.0
                events.append((int(u), int(v), t_day))
        events.sort(key=lambda x: x[2])
        return events

    def _get_user_pos_items(self) -> List[np.ndarray]:
        return [self.user_item_net[u, :].nonzero()[1] for u in range(self.num_users)]

    def _build_train_item_time_array(self) -> Tuple[np.ndarray, float]:
        item_times: List[List[float]] = [[] for _ in range(self.num_items)]
        for _, v, t in self.train_events:
            item_times[v].append(float(t))
        max_len = max((len(x) for x in item_times), default=0)
        max_time = max((max(x) for x in item_times if x), default=0.0) + 1.0
        padded: List[np.ndarray] = []
        for times in item_times:
            arr = np.asarray(sorted(times), dtype=np.float32)
            if max_len > len(arr):
                arr = np.pad(arr, (0, max_len - len(arr)), mode="constant", constant_values=max_time)
            padded.append(arr)
        if max_len == 0:
            return np.full((self.num_items, 0), max_time, dtype=np.float32), max_time
        return np.stack(padded, axis=0), max_time


@torch.no_grad()
def make_prior_snapshot(model: HawkesMFDebias, device: torch.device) -> PriorSnapshot:
    model.eval()
    mu_t, alpha_t, beta_t = model.prior_parameters_from_embeddings()
    mu = mu_t.detach().cpu().numpy().astype(np.float64)
    alpha = alpha_t.detach().cpu().numpy().astype(np.float64)
    beta = float(beta_t.detach().cpu().item())
    base_mass = float(mu.sum())
    return PriorSnapshot(
        mu=mu,
        alpha=alpha,
        beta=beta,
        base_mass=base_mass,
        base_sampler=BaseSampler(mu),
    )


def exact_prior_loss(model: HawkesMFDebias, train_events: Sequence[Tuple[int, int, float]], device: torch.device) -> torch.Tensor:
    mu, alpha, beta = model.prior_parameters_from_embeddings()
    base_mass = mu.sum()
    if len(train_events) == 0:
        return base_mass * 0.0

    x_state: Dict[int, torch.Tensor] = {}
    last_t: Dict[int, float] = {}
    total_excitation = torch.zeros((), device=device)
    prev_time = float(train_events[0][2])
    loss = torch.zeros((), device=device)

    for idx, (_, item, t) in enumerate(train_events):
        t = float(t)
        if idx == 0:
            delta_global = 0.0
        else:
            delta_global = t - prev_time
        if delta_global > 0.0:
            total_excitation = total_excitation * torch.exp(-beta * delta_global)

        if item in x_state:
            delta_item = t - last_t[item]
            h = x_state[item] * torch.exp(-beta * delta_item)
        else:
            h = torch.zeros((), device=device)

        lambda_pos = mu[item] + alpha[item] * h
        lambda_total = base_mass + total_excitation
        loss = loss - torch.log(lambda_pos + EPS) + torch.log(lambda_total + EPS)

        x_state[item] = h + 1.0
        last_t[item] = t
        total_excitation = total_excitation + alpha[item]
        prev_time = t

    return loss / max(len(train_events), 1)


def sample_epoch_negatives(
    snapshot: PriorSnapshot,
    train_events: Sequence[Tuple[int, int, float]],
    num_items: int,
    num_negatives: int,
) -> np.ndarray:
    """
    Pre-sample negatives once per epoch from the frozen prior snapshot.
    The sampler uses the exact base/excitation mixture decomposition.
    """
    negatives = np.empty((len(train_events), num_negatives), dtype=np.int64)
    tree = FenwickTree(num_items)
    c_weights = np.zeros(num_items, dtype=np.float64)
    total_c = 0.0
    g = 1.0
    prev_time = float(train_events[0][2]) if train_events else 0.0

    def sample_one_excluding(pos_item: int, rho: float) -> int:
        for _ in range(20):
            if np.random.rand() < rho or total_c <= 0.0:
                candidate = snapshot.base_sampler.sample()
            else:
                mass = np.random.rand() * max(tree.total(), 1e-12)
                candidate = tree.sample(mass)
            if candidate != pos_item:
                return int(candidate)
        # Fallback: uniform rejection if the prior mass is too concentrated.
        candidate = np.random.randint(0, num_items - 1)
        candidate += candidate >= pos_item
        return int(candidate)

    for idx, (_, item, t) in enumerate(train_events):
        t = float(t)
        delta = max(t - prev_time, 0.0)
        if delta > 0.0:
            g *= math.exp(-snapshot.beta * delta)
        excitation_mass = g * total_c
        rho = snapshot.base_mass / (snapshot.base_mass + excitation_mass + EPS)

        for j in range(num_negatives):
            negatives[idx, j] = sample_one_excluding(item, rho)

        delta_c = snapshot.alpha[item] / max(g, EPS)
        c_weights[item] += delta_c
        total_c += delta_c
        tree.add(item, delta_c)
        prev_time = t

    return negatives


def residual_epoch(
    model: HawkesMFDebias,
    train_events: Sequence[Tuple[int, int, float]],
    negatives: np.ndarray,
    optimizer_residual: torch.optim.Optimizer,
    optimizer_shared: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> float:
    model.train()
    num_events = len(train_events)
    indices = np.random.permutation(num_events)
    total_loss = 0.0
    num_batches = 0

    for start in range(0, num_events, batch_size):
        batch_idx = indices[start : start + batch_size]
        users = torch.tensor([train_events[i][0] for i in batch_idx], dtype=torch.long, device=device)
        pos_items = torch.tensor([train_events[i][1] for i in batch_idx], dtype=torch.long, device=device)
        neg_items = torch.tensor(negatives[batch_idx], dtype=torch.long, device=device)

        optimizer_residual.zero_grad()
        optimizer_shared.zero_grad()

        pos_scores = model.residual_scores(users, pos_items)
        user_vec = model.user_embedding(users)
        neg_item_vec = model.item_embedding(neg_items)
        neg_scores = (user_vec.unsqueeze(1) * neg_item_vec).sum(dim=-1)

        loss = -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()
        loss.backward()
        optimizer_residual.step()
        optimizer_shared.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def prior_log_prob_all_items(
    model: HawkesMFDebias,
    query_time: float,
    item_time_padded: np.ndarray,
    pad_value: float,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    mu_all, alpha_all, beta = model.prior_parameters_from_embeddings()
    logits_list: List[torch.Tensor] = []
    num_items = mu_all.shape[0]
    query_time_t = torch.tensor(float(query_time), dtype=torch.float32, device=device)

    for start in range(0, num_items, chunk_size):
        end = min(start + chunk_size, num_items)
        times = torch.tensor(item_time_padded[start:end], dtype=torch.float32, device=device)
        mask = times < min(query_time, pad_value)
        delta = (query_time_t - times).clamp(min=0.0)
        h = (torch.exp(-beta * delta) * mask).sum(dim=-1)
        logits = mu_all[start:end] + alpha_all[start:end] * h
        logits_list.append(logits)

    logits = torch.cat(logits_list, dim=0)
    return torch.log(logits + EPS) - torch.log(logits.sum() + EPS)


@torch.no_grad()
def evaluate_split(
    model: HawkesMFDebias,
    events: Sequence[Tuple[int, int, float]],
    train_pos_items: Sequence[np.ndarray],
    item_time_padded: np.ndarray,
    pad_value: float,
    topks: Sequence[int],
    ranking_mode: str,
    device: torch.device,
    item_chunk_size: int,
) -> Dict[str, List[float]]:
    model.eval()
    gt_items: List[List[int]] = []
    pred_items: List[np.ndarray] = []

    for user, item, query_time in events:
        user_t = torch.tensor([user], dtype=torch.long, device=device)
        residual_scores = model.residual_scores_all_items(user_t).squeeze(0).detach().cpu()
        if ranking_mode == "biased":
            prior_log_prob = prior_log_prob_all_items(
                model,
                query_time=query_time,
                item_time_padded=item_time_padded,
                pad_value=pad_value,
                chunk_size=item_chunk_size,
                device=device,
            ).detach().cpu()
            scores = prior_log_prob + residual_scores
        elif ranking_mode == "unbiased":
            scores = residual_scores
        else:
            raise ValueError(f"Unknown ranking_mode: {ranking_mode}")

        exclude = train_pos_items[user]
        if len(exclude) > 0:
            scores[exclude] = -1e30

        topk = max(topks)
        _, indices = torch.topk(scores, k=topk)
        pred_items.append(indices.cpu().numpy())
        gt_items.append([item])

    return compute_topn_metrics(gt_items, pred_items, topks)


def compute_topn_metrics(
    ground_truth: Sequence[Sequence[int]],
    predictions: Sequence[np.ndarray],
    topks: Sequence[int],
) -> Dict[str, List[float]]:
    precision_list: List[float] = []
    recall_list: List[float] = []
    ndcg_list: List[float] = []
    mrr_list: List[float] = []

    for k in topks:
        precision = 0.0
        recall = 0.0
        ndcg = 0.0
        mrr = 0.0

        for gt, pred in zip(ground_truth, predictions):
            gt_set = set(gt)
            pred_k = list(pred[:k])
            hits = [1 if x in gt_set else 0 for x in pred_k]
            hit_count = sum(hits)

            precision += hit_count / k
            recall += hit_count / max(len(gt_set), 1)

            dcg = 0.0
            rr = 0.0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    dcg += 1.0 / math.log2(rank + 1)
                    if rr == 0.0:
                        rr = 1.0 / rank
            ideal_hits = min(len(gt_set), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg += dcg / max(idcg, EPS)
            mrr += rr

        n = max(len(ground_truth), 1)
        precision_list.append(precision / n)
        recall_list.append(recall / n)
        ndcg_list.append(ndcg / n)
        mrr_list.append(mrr / n)

    return {
        "precision": precision_list,
        "recall": recall_list,
        "ndcg": ndcg_list,
        "mrr": mrr_list,
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data = InteractionData(args.data_path, args.dataset)
    model = HawkesMFDebias(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
        prior_hidden_dim=args.prior_hidden_dim,
        prior_depth=args.prior_depth,
    ).to(device)

    optimizer_prior = torch.optim.Adam(
        list(model.base_net.parameters()) + list(model.excitation_net.parameters()) + [model.log_beta],
        lr=args.prior_lr,
        weight_decay=args.weight_decay,
    )
    optimizer_residual = torch.optim.Adam(
        model.user_embedding.parameters(),
        lr=args.residual_lr,
        weight_decay=args.weight_decay,
    )
    optimizer_shared = torch.optim.Adam(
        model.item_embedding.parameters(),
        lr=args.shared_lr,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_valid = -float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        # Step 0-2: freeze snapshot -> compute frozen prior -> build epoch negatives.
        snapshot = make_prior_snapshot(model, device)
        epoch_negatives = sample_epoch_negatives(
            snapshot=snapshot,
            train_events=data.train_events,
            num_items=data.num_items,
            num_negatives=args.num_negatives,
        )

        # Step 3: prior update with exact full likelihood scan.
        optimizer_prior.zero_grad()
        optimizer_shared.zero_grad()
        prior_loss = exact_prior_loss(model, data.train_events, device)
        prior_loss.backward()
        optimizer_prior.step()
        optimizer_shared.step()

        # Step 4: residual update using the pre-built negatives.
        residual_loss = residual_epoch(
            model=model,
            train_events=data.train_events,
            negatives=epoch_negatives,
            optimizer_residual=optimizer_residual,
            optimizer_shared=optimizer_shared,
            batch_size=args.batch_size,
            device=device,
        )

        print(
            f"[Epoch {epoch:03d}] prior_loss={prior_loss.item():.6f} "
            f"residual_loss={residual_loss:.6f} beta={model.current_beta().item():.6f}"
        )

        if epoch % args.eval_every != 0:
            continue

        valid_metrics = evaluate_split(
            model=model,
            events=data.valid_events,
            train_pos_items=data.all_train_pos_items,
            item_time_padded=data.train_item_time_padded,
            pad_value=data.train_time_pad_value,
            topks=args.topks,
            ranking_mode=args.ranking_mode,
            device=device,
            item_chunk_size=args.eval_item_chunk_size,
        )
        valid_recall = valid_metrics["recall"][0]
        print(
            f"  valid recall@{args.topks[0]}={valid_recall:.6f} "
            f"ndcg@{args.topks[0]}={valid_metrics['ndcg'][0]:.6f}"
        )

        if valid_recall > best_valid:
            best_valid = valid_recall
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                torch.save(best_state, args.save_path)
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_split(
        model=model,
        events=data.test_events,
        train_pos_items=data.all_train_pos_items,
        item_time_padded=data.train_item_time_padded,
        pad_value=data.train_time_pad_value,
        topks=args.topks,
        ranking_mode=args.ranking_mode,
        device=device,
        item_chunk_size=args.eval_item_chunk_size,
    )
    print("\n[Test metrics]")
    for metric_name, values in test_metrics.items():
        joined = ", ".join(f"{metric_name}@{k}={v:.6f}" for k, v in zip(args.topks, values))
        print(joined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alternating Hawkes-prior + MF-residual recommender"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--prior_hidden_dim", type=int, default=32)
    parser.add_argument("--prior_depth", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_negatives", type=int, default=5)
    parser.add_argument("--prior_lr", type=float, default=1e-3)
    parser.add_argument("--residual_lr", type=float, default=1e-3)
    parser.add_argument("--shared_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ranking_mode", type=str, choices=["biased", "unbiased"], default="biased")
    parser.add_argument("--topks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--eval_item_chunk_size", type=int, default=4096)
    parser.add_argument("--save_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
