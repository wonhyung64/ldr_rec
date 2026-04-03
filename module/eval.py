import math
import torch
import numpy as np
from typing import Dict, List, Sequence, Tuple
from .model import HawkesMFDebias

EPS = 1e-12

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