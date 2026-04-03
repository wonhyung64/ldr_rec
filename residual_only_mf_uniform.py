import argparse
import copy
import math
import os
import random
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


class ResidualOnlyMF(nn.Module):
    """
    Residual-only MF baseline.

    This keeps the same residual parameterization as the debiased model,
    but removes the Hawkes prior entirely:
        r(u, v) = w_u^T z_v
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int, tau:float=0.1) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.tau = tau
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def residual_scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = F.normalize(self.user_embedding(users), dim=-1)
        item_vec = F.normalize(self.item_embedding(items), dim=-1)
        return torch.sum(user_vec * item_vec, dim=-1) / self.tau

    def residual_scores_all_items(self, users: torch.Tensor) -> torch.Tensor:
        user_vec = F.normalize(self.user_embedding(users), dim=-1)
        item_vec = F.normalize(self.item_embedding.weight, dim=-1)
        return user_vec @ item_vec.t() / self.tau


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


def sample_uniform_negatives_excluding(
    positive_items: np.ndarray,
    num_items: int,
    num_negatives: int,
) -> np.ndarray:
    positive_items = np.asarray(positive_items, dtype=np.int64)
    negatives = np.random.randint(0, num_items - 1, size=(len(positive_items), num_negatives), dtype=np.int64)
    negatives += negatives >= positive_items[:, None]
    return negatives


def residual_epoch_uniform(
    model: ResidualOnlyMF,
    train_events: Sequence[Tuple[int, int, float]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_items: int,
    num_negatives: int,
    device: torch.device,
) -> float:
    """
    Residual-only training with uniform negatives.
    This mirrors the previous residual logistic objective, but removes the Hawkes prior.
    """
    model.train()
    num_events = len(train_events)
    indices = np.random.permutation(num_events)
    total_loss = 0.0
    num_batches = 0

    for start in range(0, num_events, batch_size):
        batch_idx = indices[start : start + batch_size]
        users_np = np.asarray([train_events[i][0] for i in batch_idx], dtype=np.int64)
        pos_items_np = np.asarray([train_events[i][1] for i in batch_idx], dtype=np.int64)
        neg_items_np = sample_uniform_negatives_excluding(pos_items_np, num_items, num_negatives)

        users = torch.tensor(users_np, dtype=torch.long, device=device)
        pos_items = torch.tensor(pos_items_np, dtype=torch.long, device=device)
        neg_items = torch.tensor(neg_items_np, dtype=torch.long, device=device)

        optimizer.zero_grad()

        pos_scores = model.residual_scores(users, pos_items)
        user_vec = model.user_embedding(users)
        neg_item_vec = model.item_embedding(neg_items)
        neg_scores = (user_vec.unsqueeze(1) * neg_item_vec).sum(dim=-1)

        loss = -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_split(
    model: ResidualOnlyMF,
    events: Sequence[Tuple[int, int, float]],
    train_pos_items: Sequence[np.ndarray],
    topks: Sequence[int],
    device: torch.device,
) -> Dict[str, List[float]]:
    model.eval()
    gt_items: List[List[int]] = []
    pred_items: List[np.ndarray] = []

    for user, item, _ in events:
        user_t = torch.tensor([user], dtype=torch.long, device=device)
        scores = model.residual_scores_all_items(user_t).squeeze(0).detach().cpu()

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
    model = ResidualOnlyMF(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=args.embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.user_embedding.parameters()),
                "lr": args.residual_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": list(model.item_embedding.parameters()),
                "lr": args.shared_lr,
                "weight_decay": args.weight_decay,
            },
        ]
    )

    best_state = copy.deepcopy(model.state_dict())
    best_valid = -float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        residual_loss = residual_epoch_uniform(
            model=model,
            train_events=data.train_events,
            optimizer=optimizer,
            batch_size=args.batch_size,
            num_items=data.num_items,
            num_negatives=args.num_negatives,
            device=device,
        )

        print(f"[Epoch {epoch:03d}] residual_loss={residual_loss:.6f}")

        if epoch % args.eval_every != 0:
            continue

        valid_metrics = evaluate_split(
            model=model,
            events=data.valid_events,
            train_pos_items=data.all_train_pos_items,
            topks=args.topks,
            device=device,
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
        topks=args.topks,
        device=device,
    )
    print("\n[Test metrics]")
    for metric_name, values in test_metrics.items():
        joined = ", ".join(f"{metric_name}@{k}={v:.6f}" for k, v in zip(args.topks, values))
        print(joined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual-only MF baseline with uniform negative sampling"
    )
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="micro_video")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_negatives", type=int, default=15)
    parser.add_argument("--residual_lr", type=float, default=1e-3)
    parser.add_argument("--shared_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--topks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--save_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
