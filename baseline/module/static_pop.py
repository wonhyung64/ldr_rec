import os
import json
import numpy as np
import torch

from module.procedure import computeTopNAccuracy


# same calibration grid used to tune eta for the Hawkes model
ETA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


def static_power_prior(user_item_net, pop_power):
    """pi_a(v) ∝ n_v^a over training interactions.

    a=1 is the original static prior (Table 3), a=0 is uniform over seen items.
    Under Prop. 1 the learned score targets log p0 - a*log n_v, i.e. a 1/pop^a correction.
    Items unseen in training keep zero mass, as in the original static prior.
    """
    counts = np.asarray(user_item_net.sum(0)).ravel().astype(np.float64)
    prior = np.zeros_like(counts)
    seen = counts > 0
    prior[seen] = counts[seen] ** pop_power
    return prior / prior.sum()


def ckpt_tag(pop_power):
    # pop_power=1 keeps the legacy name so existing Table 3 checkpoints are reused
    return "static_prior" if pop_power == 1.0 else f"static_prior_pp{pop_power}"


def evaluate_eta_grid(score_fn, user_item_time, all_pos, item_log_prob, topks, eta_grid=ETA_GRID):
    """Full-ranking eval of r = eta*log pi + (1-eta)*s for every eta, one forward pass per user.

    eta=0.5 is ranking-equivalent to the legacy static-prior score (log pi + s).
    """
    pred_lists = {eta: [] for eta in eta_grid}
    gt_list = []
    for (user, item), pos_time_val in user_item_time.items():
        resid = score_fn(user, pos_time_val)
        exclude_items = list(all_pos[user])
        for eta in eta_grid:
            pred = item_log_prob * eta + resid * (1 - eta)
            pred[exclude_items] = -9999
            _, pred_k = torch.topk(pred, k=max(topks))
            pred_lists[eta].append(pred_k.tolist())
        gt_list.append([item])
    return {eta: computeTopNAccuracy(gt_list, pred_lists[eta], topks) for eta in eta_grid}


def log_eta_results(wandb_var, split, epoch, results, topks):
    for eta, res in results.items():
        for name, values in zip(["precision", "recall", "ndcg", "mrr"], res):
            wandb_var.log({f"{split}_{name}_{k}_{epoch}_eta{eta}": v for k, v in zip(topks, values)})
            if eta == 0.5:
                # legacy keys (Table 3 protocol)
                wandb_var.log({f"{split}_{name}_{k}_{epoch}": v for k, v in zip(topks, values)})


def dump_results(args, epoch, valid_results, test_results, out_dir="./results/static_pop"):
    os.makedirs(out_dir, exist_ok=True)

    def to_dict(results):
        return {
            str(eta): {
                f"{name}@{k}": float(v)
                for name, values in zip(["precision", "recall", "ndcg", "mrr"], res)
                for k, v in zip(args.topks, values)
            }
            for eta, res in results.items()
        }

    record = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "pop_power": args.pop_power,
        "seed": args.seed,
        "lambda1": args.lambda1,
        "tau": args.tau,
        "epoch": epoch,
        "valid": to_dict(valid_results),
        "test": to_dict(test_results),
    }
    path = f"{out_dir}/{args.dataset}_{args.model_name}_pp{args.pop_power}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved results to {path}")
