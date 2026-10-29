#%%
"""Rebuttal analysis (VShq error bars, CW75 eta=0): seed-to-seed std of test Recall@10
for Ours at eta=0 vs. the tuned eta, next to the vanilla backbone.

eta (--alpha1) only enters at inference (r = eta*log pi + (1-eta)*s), so every number here is
an eval-only re-scoring of the Table 1 checkpoints; nothing is retrained.
  - Ours    : weights/{ds}/_{model}_lambda{gamma}_e500_seed{k}_ablationshared_timefix.pt
              (the *_weighted_fixed.py runs of Table 1), cosine/tau score, eta grid
  - Vanilla : weights/{ds}/_backbone_{model}_e500_seed{k}.pt (Table 1 backbone), dot-product score
Protocol matches the training scripts: history before t, train items excluded, full ranking.

Usage (from the repo root):
  python rebuttal_eta_std.py --dataset ml-1m              # evaluate all cells of one dataset
  python rebuttal_eta_std.py --summarize                  # std table from saved json
  python rebuttal_eta_std.py --dataset ml-1m --models sasrec --seeds 1 --limit 300 \
      --out-dir results/rebuttal_eta_std_smoke            # smoke test
"""
import os
import re
import sys
import glob
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "baseline"))
os.chdir(REPO_ROOT)

from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model
from module.procedure import computeTopNAccuracy
from rebuttal_bsarec_eval import per_user_bsarec_attention, hawkes_logprob
from rebuttal_static_alpha_table import HAWKES, HAWKES_STD, VANILLA, VANILLA_STD

DATASETS = ["micro_video", "ml-1m", "kuairand"]
MODELS = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
ETA_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
TOPKS = [10, 20]

# Table 1 "+(Ours)" runs (seed-4 commands of the *_weighted_fixed.py report scripts):
# (dataset, model) -> (gamma = --lambda1, eta used in the paper = --alpha1 picked on seed-1 validation)
OURS = {
    ("micro_video", "mf"): (0.9, 0.3), ("ml-1m", "mf"): (0.9, 0.7), ("kuairand", "mf"): (0.1, 0.5),
    ("micro_video", "grurec"): (0.5, 0.3), ("ml-1m", "grurec"): (0.9, 0.5), ("kuairand", "grurec"): (0.3, 0.7),
    ("micro_video", "sasrec"): (0.7, 0.3), ("ml-1m", "sasrec"): (0.5, 0.5), ("kuairand", "sasrec"): (0.9, 0.5),
    ("micro_video", "tisasrec"): (0.3, 0.3), ("ml-1m", "tisasrec"): (0.9, 0.5), ("kuairand", "tisasrec"): (0.7, 0.5),
    ("micro_video", "fearec"): (0.1, 0.3), ("ml-1m", "fearec"): (0.1, 0.5), ("kuairand", "fearec"): (0.3, 0.5),
    ("micro_video", "bsarec"): (0.5, 0.3), ("ml-1m", "bsarec"): (0.7, 0.7), ("kuairand", "bsarec"): (0.9, 0.5),
}
# BSARec (c, alpha) per dataset; Ours and vanilla differ only on ML-1M
BSAREC_CFG = {
    "ours": {"micro_video": (1, 0.7), "ml-1m": (1, 0.7), "kuairand": (1, 0.9)},
    "vanilla": {"micro_video": (1, 0.7), "ml-1m": (5, 0.9), "kuairand": (1, 0.9)},
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-1m")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--seeds", default="1,2,3,4")
    p.add_argument("--kinds", default="ours,vanilla")
    p.add_argument("--weights-dir", default="./weights")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="results/rebuttal_eta_std")
    p.add_argument("--limit", type=int, default=0, help="subsample events (smoke test)")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--sample-std", action="store_true", help="ddof=1 instead of population std")
    return p.parse_args()


#%% data
def build_events(ds, pairs, seq_len, limit, rng):
    items = list(pairs.items())
    if limit and len(items) > limit:
        items = [items[i] for i in sorted(rng.choice(len(items), size=limit, replace=False))]
    users = np.array([u for (u, v), t in items], dtype=np.int64)
    hist, hist_time = ds.build_histories([(u, 0, t) for (u, v), t in items], seq_len)
    return {
        "users": torch.tensor(users),
        "gts": [[v] for (u, v), t in items],
        "times": torch.tensor([t for k, t in items], dtype=torch.float32),
        "hist": torch.tensor(hist),
        # TiSASRec input, as in the training scripts: long(days) * 86400
        "hist_time": torch.tensor(hist_time, dtype=torch.long) * 24 * 60 * 60,
        "excl": [list(ds._allPos[u]) for u in users],
    }


#%% model
def build_model(name, kind, dataset, ds, cfg_path, device):
    cls = MODEL_REGISTRY[name]
    if kind == "ours":
        cls = build_unshared_debias_model(cls)
    kw = dict(num_users=ds.n_user, num_items=ds.m_item, embedding_k=128, device=device,
              tau=0.1 if kind == "ours" else 0.5, depth=0, max_seq_len=50, n_heads=1, dropout=0.2)
    if name == "bsarec":
        kw["c"], kw["alpha"] = BSAREC_CFG[kind][dataset]
    if name == "tisasrec":
        kw["time_span"] = 2048 if dataset == "ml-1m" else 512
        if kind == "vanilla":  # seq_rec_tisasrec.py passes depth=args.tau
            kw["depth"] = kw["tau"]
    model = cls(**kw).to(device)
    model.load_state_dict(torch.load(cfg_path, map_location=device)["model_state_dict"])
    return model.eval()


@torch.no_grad()
def scores(model, name, kind, hist, feat, v_all):
    """Backbone score s for a batch of events (cosine/tau for Ours, dot product for vanilla)."""
    if kind == "ours":
        return model.score_all_items(hist, feat)
    return model.encode_user(hist, feat) @ v_all.T


@torch.no_grad()
def evaluate(model, name, kind, ev, item_times, device, bs, etas):
    """Recall/NDCG@{10,20} for every eta, one pass over the events."""
    feat_all = ev["hist_time"] if name == "tisasrec" else ev["users"]
    v_all = model.get_item_repr(torch.arange(model.num_items, device=device))
    ctx = per_user_bsarec_attention if name == "bsarec" else nullcontext

    # batched scoring must reproduce the scripts' one-event-per-forward scoring
    n_chk = min(8, len(ev["gts"]))
    with ctx():
        got = scores(model, name, kind, ev["hist"][:n_chk].to(device), feat_all[:n_chk].to(device), v_all)
    ref = torch.cat([scores(model, name, kind, ev["hist"][i:i+1].to(device), feat_all[i:i+1].to(device), v_all)
                     for i in range(n_chk)])
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4), f"{name}: batched scoring != per-event scoring"

    preds = {e: [] for e in etas}
    with ctx():
        for i in range(0, len(ev["gts"]), bs):
            j = min(i + bs, len(ev["gts"]))
            s = scores(model, name, kind, ev["hist"][i:j].to(device), feat_all[i:j].to(device), v_all)
            lp = hawkes_logprob(model, ev["times"][i:j], item_times, device).to(device) if kind == "ours" else None
            rows = torch.cat([torch.full((len(x),), r, dtype=torch.long) for r, x in enumerate(ev["excl"][i:j])])
            cols = torch.cat([torch.as_tensor(x, dtype=torch.long) for x in ev["excl"][i:j]])
            for e in etas:
                pred = s.clone() if lp is None else e * lp + (1 - e) * s
                pred[rows.to(device), cols.to(device)] = -9999
                preds[e].append(torch.topk(pred, k=max(TOPKS), dim=-1).indices.cpu())
    out = {}
    for e in etas:
        _, recall, ndcg, _ = computeTopNAccuracy(ev["gts"], torch.cat(preds[e]).tolist(), TOPKS)
        out[str(e)] = {**{f"recall@{k}": float(r) for k, r in zip(TOPKS, recall)},
                       **{f"ndcg@{k}": float(n) for k, n in zip(TOPKS, ndcg)}}
    return out


def ckpt_path(weights_dir, dataset, name, kind, seed):
    if kind == "ours":
        gamma = OURS[(dataset, name)][0]
        return f"{weights_dir}/{dataset}/_{name}_lambda{gamma}_e500_seed{seed}_ablationshared_timefix.pt"
    return f"{weights_dir}/{dataset}/_backbone_{name}_e500_seed{seed}.pt"


#%% summary
def summarize(out_dir, ddof):
    runs = {}
    for f in glob.glob(os.path.join(out_dir, "*.json")):
        r = json.load(open(f))
        runs.setdefault((r["dataset"], r["model"], r["kind"]), {})[r["seed"]] = r

    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-", None
        sd = float(np.std(vals, ddof=ddof)) if len(vals) > 1 else None
        return (f"{np.mean(vals):.4f}" + (f"±{sd:.4f}" if sd is not None else "")), sd

    def cv(vals):  # std/mean, since the eta=0 mean is much lower than the tuned-eta mean
        return f"{np.std(vals, ddof=ddof) / np.mean(vals):.3f}" if len(vals) > 1 and np.mean(vals) > 0 else "-"

    def r10(r, eta):
        return r["test"][str(eta)]["recall@10"]

    cols = ["seeds", "Ours eta=0", "Ours eta_paper", "eta_paper", "Ours eta*(valid, per seed)",
            "std ratio (eta_paper/eta=0)", "CV eta=0 / eta_paper / vanilla", "Vanilla (re-eval)",
            "Paper Ours", "Paper Vanilla"]
    print(f"\n## Test Recall@10, mean±std over seeds ({'sample' if ddof else 'population'} std)")
    print("eta_paper = Table 1 setting (picked on seed-1 validation); eta* = per-seed validation argmax")
    print("| dataset | model | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for ds in DATASETS:
        for m in MODELS:
            ours = runs.get((ds, m, "ours"), {})
            van = runs.get((ds, m, "vanilla"), {})
            if not ours and not van:
                continue
            eta_p = OURS[(ds, m)][1]
            rs = [ours[s] for s in sorted(ours)]
            s0, sd0 = ms([r10(r, 0.0) for r in rs])
            sp, sdp = ms([r10(r, eta_p) for r in rs])
            sstar, _ = ms([r10(r, max(ETA_GRID, key=lambda e: r["valid"][str(e)]["recall@10"])) for r in rs])
            ratio = f"{sdp / sd0:.2f}" if sd0 and sdp is not None else "-"
            sv, _ = ms([r10(van[s], 0.0) for s in sorted(van)])
            cvs = " / ".join([cv([r10(r, 0.0) for r in rs]), cv([r10(r, eta_p) for r in rs]),
                              cv([r10(van[s], 0.0) for s in sorted(van)])])
            paper_o = f"{HAWKES[(ds, m)][0]:.4f}±{HAWKES_STD[(ds, m)][0]:.4f}"
            paper_v = (f"{VANILLA[(ds, m)][0]:.4f}±{VANILLA_STD[(ds, m)][0]:.4f}"
                       if (ds, m) in VANILLA_STD else "-")
            seeds = ",".join(map(str, sorted(set(ours) | set(van))))
            print(f"| {ds} | {m} | {seeds} | {s0} | {sp} | {eta_p} | {sstar} | {ratio} | {cvs} | {sv} | {paper_o} | {paper_v} |")


#%% main
def main():
    args = parse()
    if args.summarize:
        summarize(args.out_dir, 1 if args.sample_std else 0)
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

    os.makedirs(args.out_dir, exist_ok=True)
    ds = UserItemTime("./data", args.dataset, "d", 50, 50)
    rng = np.random.default_rng(0)
    evs = {"valid": build_events(ds, ds.valid_user_item_time, 50, args.limit, rng),
           "test": build_events(ds, ds.test_user_item_time, 50, args.limit, rng)}
    item_times = torch.tensor(ds.item_time_array / 86400.0, dtype=torch.float32, device=device)
    print(f"device={device}  valid={len(evs['valid']['gts'])}  test={len(evs['test']['gts'])}", flush=True)

    for name, kind, seed, path, out in jobs:
        print(f"[eval] {args.dataset} {name} {kind} seed{seed}: {path}", flush=True)
        model = build_model(name, kind, args.dataset, ds, path, device)
        etas = ETA_GRID if kind == "ours" else [0.0]
        res = {"dataset": args.dataset, "model": name, "kind": kind, "seed": seed, "checkpoint": path,
               "limit": args.limit}
        for split, ev in evs.items():
            res[split] = evaluate(model, name, kind, ev, item_times, device, args.bs, etas)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print("   test R@10 " + "  ".join(f"eta={e}:{res['test'][str(e)]['recall@10']:.4f}" for e in etas), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    summarize(args.out_dir, 1 if args.sample_std else 0)


if __name__ == "__main__":
    main()
