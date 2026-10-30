#%%
"""Rebuttal analysis (CW75: how much of the gain is the debiased score alone?):
head/tail Recall@10 of the debiased score s (eta=0) vs. the calibrated score and the vanilla backbone.

No popularity-free ground truth exists, so the debiased component is judged on tail items
(Figure 3 protocol: under each popularity definition the top 20% of items are head, the rest tail;
definitions = overall / recent_3d / recent_7d, files from split_test_head_tail.py).

Everything is an eval-only re-scoring of the Table 1 checkpoints (eta only enters at inference):
  - Ours    : r = eta*log pi + (1-eta)*s, eta in ETA_GRID (eta=0 -> debiased score alone)
  - Vanilla : dot-product backbone score (uniform negatives)
Per checkpoint it stores, for every eta,
  - Recall/NDCG@{10,20} on all / head / tail test events (each popularity definition)
  - tail share of the top-10 lists (how much exposure tail items get)
  - Spearman correlation between the score over all items and log item popularity, averaged over
    test events (how much popularity is left in the score; Corollary 1 predicts it is high for
    uniform negatives and small for s)
Checkpoints, model construction and the protocol are shared with rebuttal_eta_std.py.

Usage (from the repo root):
  python rebuttal_eta0_tail.py --dataset ml-1m              # evaluate all cells of one dataset
  python rebuttal_eta0_tail.py --summarize                  # tables from saved json
  python rebuttal_eta0_tail.py --dataset ml-1m --models sasrec --seeds 1 --limit 500 \
      --out-dir results/rebuttal_eta0_tail_smoke            # smoke test
"""
import os
import sys
import glob
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from scipy.stats import rankdata, ttest_ind

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "baseline"))
os.chdir(REPO_ROOT)

from rebuttal_eta_std import (DATASETS, MODELS, ETA_GRID, OURS, build_model, ckpt_path, scores)
from rebuttal_bsarec_eval import per_user_bsarec_attention, hawkes_logprob
from split_test_head_tail import head_tail_items

POP_DEFS = ["overall", "recent_3d", "recent_7d"]
TOPKS = [10, 20]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-1m")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--seeds", default="1,2,3,4")
    p.add_argument("--kinds", default="ours,vanilla")
    p.add_argument("--weights-dir", default="./weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="results/rebuttal_eta0_tail")
    p.add_argument("--limit", type=int, default=0, help="subsample test events (smoke test)")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--pop-def", default="recent_3d", help="popularity definition for the main summary table")
    return p.parse_args()


#%% data
def build_test_events(ds, limit, rng):
    items = list(ds.test_user_item_time.items())
    if limit and len(items) > limit:
        items = [items[i] for i in sorted(rng.choice(len(items), size=limit, replace=False))]
    users = np.array([u for (u, v), t in items], dtype=np.int64)
    hist, hist_time = ds.build_histories([(u, 0, t) for (u, v), t in items], 50)
    return {
        "keys": [k for k, t in items],
        "users": torch.tensor(users),
        "gt": torch.tensor([v for (u, v), t in items]),
        "times": torch.tensor([t for k, t in items], dtype=torch.float32),
        "hist": torch.tensor(hist),
        # TiSASRec input, as in the training scripts: long(days) * 86400
        "hist_time": torch.tensor(hist_time, dtype=torch.long) * 24 * 60 * 60,
        "excl": [list(ds._allPos[u]) for u in users],
    }


def popularity_info(ds, keys, data_path, dataset):
    """Per definition: tail-item mask, Spearman rank of log popularity over items, head/tail event indices."""
    key_to_idx = {k: i for i, k in enumerate(keys)}
    info = {}
    for pop_def in POP_DEFS:
        pop = np.load(f"{data_path}/{dataset}/item_popularity_{pop_def}.npy", allow_pickle=True).item()
        head, _ = head_tail_items(pop)
        is_tail = torch.ones(ds.m_item, dtype=torch.bool)
        is_tail[list(head)] = False
        counts = np.zeros(ds.m_item)
        for i, c in pop.items():
            counts[i] = c
        pop_rank = torch.tensor(rankdata(counts), dtype=torch.float32)  # average ranks for ties
        parts = {}
        for part in ["head", "tail"]:
            d = getattr(ds, f"test_{part}_{pop_def}_dict")
            parts[part] = [key_to_idx[k] for k in ds.set_to_pair(d, ds.time_dict, ds.time_unit) if k in key_to_idx]
        info[pop_def] = {"is_tail": is_tail, "pop_rank": pop_rank, **parts}
    return info


#%% evaluation
def spearman_rows(x, ref_rank):
    """Spearman correlation of each row of x (B x M) with a fixed item ranking (M,)."""
    rx = x.argsort(dim=-1).argsort(dim=-1).float()
    rx = rx - rx.mean(-1, keepdim=True)
    ry = ref_rank - ref_rank.mean()
    return (rx @ ry) / (rx.norm(dim=-1) * ry.norm() + 1e-12)


@torch.no_grad()
def evaluate(model, name, kind, ev, pinfo, item_times, device, bs, etas, m_item):
    feat_all = ev["hist_time"] if name == "tisasrec" else ev["users"]
    v_all = model.get_item_repr(torch.arange(model.num_items, device=device))
    ctx = per_user_bsarec_attention if name == "bsarec" else nullcontext

    # batched scoring must reproduce the scripts' one-event-per-forward scoring
    n_chk = min(8, len(ev["gt"]))
    with ctx():
        got = scores(model, name, kind, ev["hist"][:n_chk].to(device), feat_all[:n_chk].to(device), v_all)
    ref = torch.cat([scores(model, name, kind, ev["hist"][i:i+1].to(device), feat_all[i:i+1].to(device), v_all)
                     for i in range(n_chk)])
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4), f"{name}: batched scoring != per-event scoring"

    pop_rank = {d: pinfo[d]["pop_rank"].to(device) for d in POP_DEFS}
    topk = {e: [] for e in etas}
    rho = {e: {d: [] for d in POP_DEFS} for e in etas}
    with ctx():
        for i in range(0, len(ev["gt"]), bs):
            j = min(i + bs, len(ev["gt"]))
            s = scores(model, name, kind, ev["hist"][i:j].to(device), feat_all[i:j].to(device), v_all)
            lp = hawkes_logprob(model, ev["times"][i:j], item_times, device).to(device) if kind == "ours" else None
            rows = torch.cat([torch.full((len(x),), r, dtype=torch.long) for r, x in enumerate(ev["excl"][i:j])])
            cols = torch.cat([torch.as_tensor(x, dtype=torch.long) for x in ev["excl"][i:j]])
            for e in etas:
                pred = s.clone() if lp is None else e * lp + (1 - e) * s
                for d in POP_DEFS:  # popularity left in the score, over all real items
                    rho[e][d].append(spearman_rows(pred[:, :m_item], pop_rank[d]).cpu())
                pred[rows.to(device), cols.to(device)] = -9999
                topk[e].append(torch.topk(pred, k=max(TOPKS), dim=-1).indices.cpu())

    out = {}
    gt = ev["gt"]
    for e in etas:
        tk = torch.cat(topk[e])
        rank_of_gt = (tk == gt[:, None]).float()  # one ground-truth item per event
        hit = {k: rank_of_gt[:, :k].sum(-1) for k in TOPKS}
        dcg = {k: (rank_of_gt[:, :k] / torch.log2(torch.arange(2, k + 2).float())).sum(-1) for k in TOPKS}

        def metrics(idx):
            idx = torch.as_tensor(idx, dtype=torch.long)
            m = {"n": len(idx)}
            for k in TOPKS:
                m[f"recall@{k}"] = float(hit[k][idx].mean()) if len(idx) else None
                m[f"ndcg@{k}"] = float(dcg[k][idx].mean()) if len(idx) else None
            return m

        r = {"all": metrics(range(len(gt)))}
        for d in POP_DEFS:
            r[f"head_{d}"] = metrics(pinfo[d]["head"])
            r[f"tail_{d}"] = metrics(pinfo[d]["tail"])
            r[f"tail_share@10_{d}"] = float(pinfo[d]["is_tail"][tk[:, :10]].float().mean())
            r[f"spearman_pop_{d}"] = float(torch.cat(rho[e][d]).mean())
        out[str(e)] = r
    return out


#%% summary
def summarize(out_dir, pop_def):
    runs = {}
    for f in glob.glob(os.path.join(out_dir, "*.json")):
        r = json.load(open(f))
        runs.setdefault((r["dataset"], r["model"], r["kind"]), {})[r["seed"]] = r

    def get(rs, eta, key, metric=None):
        vals = [r["test"][str(eta)][key] for r in rs]
        return [v[metric] for v in vals] if metric else vals

    def ms(vals, nd=4):
        if not vals:
            return "-"
        sd = f"±{np.std(vals, ddof=1):.{nd}f}" if len(vals) > 1 else ""
        return f"{np.mean(vals):.{nd}f}{sd}"

    def pval(a, b):  # one-sided Welch, H1: mean(a) > mean(b)
        if len(a) < 2 or len(b) < 2:
            return "-"
        return f"{ttest_ind(a, b, equal_var=False, alternative='greater').pvalue:.3f}"

    cells = [(ds, m) for ds in DATASETS for m in MODELS
             if (ds, m, "ours") in runs or (ds, m, "vanilla") in runs]
    print(f"# eta=0 (debiased score alone) vs. vanilla / calibrated, test, mean±std over seeds (sample std)")
    print(f"Head/tail split: {pop_def} popularity, top 20% items = head. "
          f"eta_paper = Table 1 setting. p = one-sided Welch, Ours eta=0 > Vanilla.\n")

    for metric in ["recall@10", "ndcg@10"]:
        print(f"## Tail {metric} ({pop_def})")
        print("| dataset | model | seeds | Vanilla | Ours eta=0 | Ours eta_paper | eta_paper | p (eta=0 > Van) |")
        print("|---|---|---|---|---|---|---|---|")
        for ds, m in cells:
            o = [runs[(ds, m, "ours")][s] for s in sorted(runs.get((ds, m, "ours"), {}))]
            v = [runs[(ds, m, "vanilla")][s] for s in sorted(runs.get((ds, m, "vanilla"), {}))]
            eta_p = OURS[(ds, m)][1]
            key = f"tail_{pop_def}"
            a0 = get(o, 0.0, key, metric) if o else []
            ap = get(o, eta_p, key, metric) if o else []
            av = get(v, 0.0, key, metric) if v else []
            print(f"| {ds} | {m} | {len(o)}/{len(v)} | {ms(av)} | {ms(a0)} | {ms(ap)} | {eta_p} | {pval(a0, av)} |")
        print()

    print(f"## Overall / head Recall@10 ({pop_def}) — where the eta=0 trade-off lands")
    print("| dataset | model | All: Van | All: eta=0 | All: eta_paper | Head: Van | Head: eta=0 | Head: eta_paper |")
    print("|---|---|---|---|---|---|---|---|")
    for ds, m in cells:
        o = [runs[(ds, m, "ours")][s] for s in sorted(runs.get((ds, m, "ours"), {}))]
        v = [runs[(ds, m, "vanilla")][s] for s in sorted(runs.get((ds, m, "vanilla"), {}))]
        eta_p = OURS[(ds, m)][1]
        row = []
        for key in ["all", f"head_{pop_def}"]:
            row += [ms(get(v, 0.0, key, "recall@10")) if v else "-",
                    ms(get(o, 0.0, key, "recall@10")) if o else "-",
                    ms(get(o, eta_p, key, "recall@10")) if o else "-"]
        print(f"| {ds} | {m} | " + " | ".join(row) + " |")
    print()

    print(f"## Popularity left in the score ({pop_def}): Spearman(score, item popularity) / tail share of top-10")
    print("| dataset | model | rho: Van | rho: eta=0 | rho: eta_paper | tail@10: Van | tail@10: eta=0 | tail@10: eta_paper |")
    print("|---|---|---|---|---|---|---|---|")
    for ds, m in cells:
        o = [runs[(ds, m, "ours")][s] for s in sorted(runs.get((ds, m, "ours"), {}))]
        v = [runs[(ds, m, "vanilla")][s] for s in sorted(runs.get((ds, m, "vanilla"), {}))]
        eta_p = OURS[(ds, m)][1]
        row = []
        for key in [f"spearman_pop_{pop_def}", f"tail_share@10_{pop_def}"]:
            row += [ms(get(v, 0.0, key), 3) if v else "-",
                    ms(get(o, 0.0, key), 3) if o else "-",
                    ms(get(o, eta_p, key), 3) if o else "-"]
        print(f"| {ds} | {m} | " + " | ".join(row) + " |")
    print()

    print(f"## Tail Recall@10 ({pop_def}) of Ours along the eta grid (eta=0: debiased score only)")
    print("| dataset | model | " + " | ".join(f"eta={e}" for e in ETA_GRID) + " | Vanilla |")
    print("|---|---|" + "---|" * (len(ETA_GRID) + 1))
    for ds, m in cells:
        o = [runs[(ds, m, "ours")][s] for s in sorted(runs.get((ds, m, "ours"), {}))]
        v = [runs[(ds, m, "vanilla")][s] for s in sorted(runs.get((ds, m, "vanilla"), {}))]
        key = f"tail_{pop_def}"
        row = [ms(get(o, e, key, "recall@10")) if o else "-" for e in ETA_GRID]
        print(f"| {ds} | {m} | " + " | ".join(row) + f" | {ms(get(v, 0.0, key, 'recall@10')) if v else '-'} |")


#%% main
def main():
    args = parse()
    if args.summarize:
        summarize(args.out_dir, args.pop_def)
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    jobs = []
    for name in args.models.split(","):
        for kind in args.kinds.split(","):
            for seed in map(int, args.seeds.split(",")):
                path = ckpt_path(args.weights_dir, args.dataset, name, kind, seed)
                out = os.path.join(args.out_dir, f"{args.dataset}_{name}_{kind}_seed{seed}.json")
                if not os.path.exists(path):
                    print(f"[missing] {path}")
                elif os.path.exists(out) and not args.overwrite:
                    print(f"[cached] {out}")
                else:
                    jobs.append((name, kind, seed, path, out))
    if not jobs:
        return

    from module.dataset import UserItemTime
    os.makedirs(args.out_dir, exist_ok=True)
    ds = UserItemTime("./data", args.dataset, "d", 50, 50)
    ev = build_test_events(ds, args.limit, np.random.default_rng(0))
    pinfo = popularity_info(ds, ev["keys"], "./data", args.dataset)
    item_times = torch.tensor(ds.item_time_array / 86400.0, dtype=torch.float32, device=device)
    print(f"device={device}  test={len(ev['gt'])}  " + "  ".join(
        f"{d}: head={len(pinfo[d]['head'])} tail={len(pinfo[d]['tail'])}" for d in POP_DEFS), flush=True)

    for name, kind, seed, path, out in jobs:
        print(f"[eval] {args.dataset} {name} {kind} seed{seed}: {path}", flush=True)
        model = build_model(name, kind, args.dataset, ds, path, device)
        etas = ETA_GRID if kind == "ours" else [0.0]
        res = {"dataset": args.dataset, "model": name, "kind": kind, "seed": seed, "checkpoint": path,
               "limit": args.limit, "test": evaluate(model, name, kind, ev, pinfo, item_times, device,
                                                     args.bs, etas, ds.m_item)}
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        show = [0.0, OURS[(args.dataset, name)][1]] if kind == "ours" else [0.0]
        print("   " + "  ".join(f"eta={e}: all R@10={res['test'][str(e)]['all']['recall@10']:.4f} "
                                f"tail_3d R@10={res['test'][str(e)]['tail_recent_3d']['recall@10']:.4f} "
                                f"rho_3d={res['test'][str(e)]['spearman_pop_recent_3d']:.3f}" for e in show), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
