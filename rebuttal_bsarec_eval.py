#%%
"""Rebuttal analysis: BSARec on ML-1M under different backbone configurations.

For every checkpoint matched by a spec it computes, on the full valid/test splits
(same protocol as the training scripts: history before t, train items excluded):
  - Recall/NDCG@{10,20} on an eta grid, eta* selected by validation Recall@10
  - head / tail (overall popularity) Recall@10 at eta*
  - representation probes at eta*: last-item-only input, mean-user representation,
    history-order shuffle, share of variance explained by the last item, mean
    pairwise cosine of user representations

Usage (from the repo root):
  python rebuttal_bsarec_eval.py --with-reference      # evaluate default specs (+ SASRec)
  python rebuttal_bsarec_eval.py --summarize           # mean/std table from saved json
  python rebuttal_bsarec_eval.py --limit 500 --out-dir results/rebuttal_bsarec_smoke   # smoke test
"""
import os
import re
import sys
import glob
import json
import argparse
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "baseline"))
os.chdir(REPO_ROOT)

from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model
from module.procedure import computeTopNAccuracy
from module.bsarec import BSARecLayer


# label, model, kind, backbone cfg, tau, checkpoint glob ({ds} = dataset)
#   kind: ours = Hawkes negatives + cosine/tau, eta-calibrated ranking
#         dot  = vanilla backbone (Table 1), dot-product ranking
#         cos  = vanilla backbone trained with cosine/tau (Table 2 "Vanilla"), cosine ranking
DEFAULT_SPECS = [
    ("submitted_ours_c1_a0.7", "bsarec", "ours", {"c": 1, "alpha": 0.7}, 0.1,
     "./weights/{ds}/_bsarec_lambda0.7_e500_seed*_ablationshared_timefix.pt"),
    ("ours_c5_a0.9", "bsarec", "ours", {"c": 5, "alpha": 0.9}, 0.1,
     "./weights_rebuttal/bsarec_c5_a0.9/{ds}/_bsarec_lambda0.7_e500_seed*_ablationshared_timefix.pt"),
    ("ours_c3_a0.9", "bsarec", "ours", {"c": 3, "alpha": 0.9}, 0.1,
     "./weights_rebuttal/bsarec_c3_a0.9/{ds}/_bsarec_lambda0.7_e500_seed*_ablationshared_timefix.pt"),
    ("vanilla_dot_c5_a0.9", "bsarec", "dot", {"c": 5, "alpha": 0.9}, 0.1,
     "./weights/{ds}/_backbone_bsarec_e500_seed*.pt"),
    ("vanilla_cos_c5_a0.9", "bsarec", "cos", {"c": 5, "alpha": 0.9}, 0.1,
     "./weights_rebuttal/bsarec_vanilla_cos_c5_a0.9/{ds}/norm_backbone_bsarec_e500_seed*.pt"),
]
REFERENCE_SPECS = [
    ("ref_sasrec_ours", "sasrec", "ours", {}, 0.1,
     "./weights/{ds}/_sasrec_lambda0.5_e500_seed*_ablationshared_timefix.pt"),
    ("ref_sasrec_vanilla_dot", "sasrec", "dot", {}, 0.1,
     "./weights/{ds}/_backbone_sasrec_e500_seed*.pt"),
]
ETA_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
TOPKS = [10, 20]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-1m")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="results/rebuttal_bsarec")
    p.add_argument("--with-reference", action="store_true", help="also evaluate SASRec reference checkpoints")
    p.add_argument("--only", default="", help="comma-separated labels to evaluate")
    p.add_argument("--spec", action="append", default=[],
                   help="extra spec 'label|model|kind|c|alpha|tau|glob' (c/alpha may be '-')")
    p.add_argument("--limit", type=int, default=0, help="subsample events (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summarize", action="store_true")
    return p.parse_args()


#%% evaluation helpers
@contextmanager
def per_user_bsarec_attention():
    """Reproduce the per-user evaluation protocol of the training scripts (one user
    per forward pass) exactly, while still encoding many users per batch."""
    original = BSARecLayer.forward

    def forward(self, input_tensor):
        dsp = self.filter_layer(input_tensor)
        B, L, D = input_tensor.shape
        flat = input_tensor.reshape(1, B * L, D)
        gsp, _ = self.attention_layer(flat, flat, flat)
        return self.alpha * dsp + (1 - self.alpha) * gsp.reshape(B, L, D)

    BSARecLayer.forward = forward
    try:
        yield
    finally:
        BSARecLayer.forward = original


@torch.no_grad()
def encode(model, name, hist, users, device, per_user=True, bs=1024):
    out = []
    ctx = per_user_bsarec_attention() if (name == "bsarec" and per_user) else nullcontext()
    with ctx:
        for i in range(0, len(hist), bs):
            out.append(model.encode_user(hist[i:i+bs].to(device), users[i:i+bs].to(device)).cpu())
    return torch.cat(out)


def build_events(ds, pairs, seq_len, limit, rng):
    items = list(pairs.items())
    if limit and len(items) > limit:
        items = [items[i] for i in sorted(rng.choice(len(items), size=limit, replace=False))]
    users = np.array([u for (u, v), t in items], dtype=np.int64)
    hist, _ = ds.build_histories([(u, 0, t) for (u, v), t in items], seq_len)
    excl = [np.asarray(ds._allPos[u], dtype=np.int64) for u in users]
    lens = np.array([len(e) for e in excl], dtype=np.int64)
    return {
        "keys": [k for k, t in items],
        "users": torch.tensor(users),
        "gts": [[v] for (u, v), t in items],
        "times": torch.tensor([t for k, t in items], dtype=torch.float32),
        "hist": torch.tensor(hist),
        "excl_rows": np.repeat(np.arange(len(items)), lens),
        "excl_cols": np.concatenate(excl) if len(excl) else np.zeros(0, dtype=np.int64),
        "excl_offs": np.concatenate([[0], np.cumsum(lens)]),
    }


def subset(ev, idx):
    idx = np.asarray(idx, dtype=np.int64)
    excl = [ev["excl_cols"][ev["excl_offs"][i]:ev["excl_offs"][i+1]] for i in idx]
    lens = np.array([len(e) for e in excl], dtype=np.int64)
    return {
        "keys": [ev["keys"][i] for i in idx],
        "users": ev["users"][idx],
        "gts": [ev["gts"][i] for i in idx],
        "times": ev["times"][idx],
        "hist": ev["hist"][idx],
        "excl_rows": np.repeat(np.arange(len(idx)), lens),
        "excl_cols": np.concatenate(excl) if len(excl) else np.zeros(0, dtype=np.int64),
        "excl_offs": np.concatenate([[0], np.cumsum(lens)]),
    }


@torch.no_grad()
def hawkes_logprob(model, times, item_times, device, chunk=128):
    mu, alpha, beta = model.prior_parameters_from_embeddings()
    out = []
    for i in range(0, len(times), chunk):
        t = times[i:i+chunk].to(device).view(-1, 1, 1)
        h = (torch.exp(-beta * (t - item_times).clamp(min=0.0)) * (item_times < t)).sum(-1)
        lam = mu + alpha * h
        out.append((torch.log(lam + 1e-12) - torch.log(lam.sum(-1, keepdim=True) + 1e-12)).cpu())
    return torch.cat(out)


@torch.no_grad()
def rank(U, V, kind, tau, LP, ev, etas, device, chunk=1024):
    """Top-max(TOPKS) lists for r = eta*log pi + (1-eta)*s (s only for vanilla kinds)."""
    V = V.to(device)
    Vn = F.normalize(V, dim=-1)
    out = {e: [] for e in etas}
    for i in range(0, len(U), chunk):
        j = min(i + chunk, len(U))
        u = U[i:j].to(device)
        S = u @ V.T if kind == "dot" else F.normalize(u, dim=-1) @ Vn.T / tau
        lo, hi = ev["excl_offs"][i], ev["excl_offs"][j]
        rows = torch.as_tensor(ev["excl_rows"][lo:hi] - i, device=device)
        cols = torch.as_tensor(ev["excl_cols"][lo:hi], device=device)
        lp = LP[i:j].to(device) if LP is not None else None
        for e in etas:
            pred = S.clone() if lp is None else e * lp + (1 - e) * S
            pred[rows, cols] = -9999
            out[e].append(torch.topk(pred, k=max(TOPKS), dim=-1).indices.cpu())
    return {e: torch.cat(v).tolist() for e, v in out.items()}


def metrics(gts, preds):
    _, recall, ndcg, _ = computeTopNAccuracy(gts, preds, TOPKS)
    out = {f"recall@{k}": float(r) for k, r in zip(TOPKS, recall)}
    out.update({f"ndcg@{k}": float(n) for k, n in zip(TOPKS, ndcg)})
    return out


def mean_pairwise_cosine(U, n=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(U), generator=g)[:n]
    Un = F.normalize(U[idx].double(), dim=-1)
    s = Un.sum(0)
    return float(((s @ s) - len(Un)) / (len(Un) * (len(Un) - 1)))


def last_item_r2(U, last_items):
    """Share of (normalized) user-repr variance explained by the identity of the last item."""
    Un = F.normalize(U.double(), dim=-1)
    uniq, inv, cnt = torch.unique(last_items, return_inverse=True, return_counts=True)
    keep = cnt[inv] > 1
    Un, inv = Un[keep], inv[keep]
    means = torch.zeros(len(uniq), Un.shape[1], dtype=Un.dtype).index_add_(0, inv, Un) / cnt.unsqueeze(1).double()
    within = ((Un - means[inv]) ** 2).sum()
    total = ((Un - Un.mean(0)) ** 2).sum()
    return float(1 - within / total)


def shuffle_history(hist, pad_id, seed=0):
    """Permute every history except its last item; left padding stays in place."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.rand(hist.shape, generator=g)
    keys[hist == pad_id] = -1.0
    keys[:, -1] = 2.0
    return torch.gather(hist, 1, torch.argsort(keys, dim=1))


#%% per-checkpoint evaluation
def build_model(name, kind, cfg, tau, ds, device):
    cls = MODEL_REGISTRY[name]
    if kind == "ours":
        cls = build_unshared_debias_model(cls)
    return cls(num_users=ds.n_user, num_items=ds.m_item, embedding_k=128, device=device, tau=tau,
               depth=0, max_seq_len=50, n_heads=1, dropout=0.2, **cfg).to(device)


def evaluate_checkpoint(spec, path, ds, valid_ev, test_ev, item_times, device):
    label, name, kind, cfg, tau, _ = spec
    model = build_model(name, kind, cfg, tau, ds, device)
    model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    model.eval()
    V = model.item_embedding.weight[:ds.m_item].detach()
    etas = ETA_GRID if kind == "ours" else [0.0]

    if name == "bsarec":  # the batched per-user encoder must match one-user-per-pass exactly
        n_chk = min(16, len(test_ev["hist"]))
        ref = torch.cat([encode(model, name, test_ev["hist"][i:i+1], test_ev["users"][i:i+1], device, per_user=False)
                         for i in range(n_chk)])
        got = encode(model, name, test_ev["hist"][:n_chk], test_ev["users"][:n_chk], device)
        assert torch.allclose(ref, got, atol=1e-4), "per-user encoding mismatch"

    res = {"label": label, "checkpoint": path, "model": name, "kind": kind, "cfg": cfg, "tau": tau,
           "seed": int(re.search(r"seed(\d+)", path).group(1))}

    U = {}
    LP = {}
    for split, ev in [("valid", valid_ev), ("test", test_ev)]:
        U[split] = encode(model, name, ev["hist"], ev["users"], device)
        LP[split] = hawkes_logprob(model, ev["times"], item_times, device) if kind == "ours" else None
        preds = rank(U[split], V, kind, tau, LP[split], ev, etas, device)
        res[split] = {str(e): metrics(ev["gts"], preds[e]) for e in etas}

    eta_star = max(etas, key=lambda e: res["valid"][str(e)]["recall@10"])
    res["eta_star"] = eta_star
    res["test_at_eta_star"] = res["test"][str(eta_star)]

    # head / tail (overall popularity, as in Figure 3) at eta*
    key_to_idx = {k: i for i, k in enumerate(test_ev["keys"])}
    for part, d in [("head", ds.test_head_overall_dict), ("tail", ds.test_tail_overall_dict)]:
        idx = [key_to_idx[k] for k in ds.set_to_pair(d, ds.time_dict, ds.time_unit) if k in key_to_idx]
        sub = subset(test_ev, idx)
        p = rank(U["test"][idx], V, kind, tau, None if LP["test"] is None else LP["test"][idx], sub, [eta_star], device)
        res[f"test_{part}"] = dict(metrics(sub["gts"], p[eta_star]), n=len(idx))

    # representation probes on test at eta*
    hist = test_ev["hist"]
    pad = ds.m_item
    hist_last = torch.full_like(hist, pad)
    hist_last[:, -1] = hist[:, -1]
    U_last = encode(model, name, hist_last, test_ev["users"], device)
    U_shuf = encode(model, name, shuffle_history(hist, pad), test_ev["users"], device)
    U_mean = U["test"].mean(0, keepdim=True).expand_as(U["test"]).contiguous()
    probe = {}
    for tag, Ux in [("last_item_only", U_last), ("mean_user", U_mean)]:
        p = rank(Ux, V, kind, tau, LP["test"], test_ev, [eta_star], device)
        probe[f"recall@10_{tag}"] = metrics(test_ev["gts"], p[eta_star])["recall@10"]
    probe["shuffle_rel_change"] = float(((U["test"] - U_shuf).norm(dim=-1) / U["test"].norm(dim=-1).clamp(min=1e-12)).mean())
    probe["last_item_r2"] = last_item_r2(U["test"], hist[:, -1])
    probe["mean_pairwise_cosine"] = mean_pairwise_cosine(U["test"])
    if name == "bsarec":  # internal diagnostic: batched (training-time) vs per-user representations
        idx = torch.randperm(len(hist), generator=torch.Generator().manual_seed(0))[:4000]  # random mix, as in training batches
        U_b = encode(model, name, hist[idx], test_ev["users"][idx], device, per_user=False, bs=1000)
        probe["batched_vs_per_user_rel_change"] = float(((U["test"][idx] - U_b).norm(dim=-1) / U["test"][idx].norm(dim=-1)).mean())
    res["probe"] = probe
    return res


#%% summary
def summarize(out_dir, dataset):
    files = sorted(glob.glob(os.path.join(out_dir, dataset, "*.json")))
    groups = {}
    for f in files:
        r = json.load(open(f))
        groups.setdefault(r["label"], []).append(r)

    def ms(vals, fmt="{:.4f}"):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        return fmt.format(np.mean(vals)) + (("±" + fmt.format(np.std(vals))) if len(vals) > 1 else "")

    cols = ["seeds", "eta*", "R@10", "N@10", "R@10(eta=0)", "R@10(eta=1)", "head R@10", "tail R@10",
            "last-only", "mean-user", "shuffle", "R2(last)", "cos-sim"]
    print(f"\n[{dataset}] test metrics, eta* by validation Recall@10 (mean±std over seeds)")
    print("| label | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for label, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        row = [
            ",".join(str(r["seed"]) for r in rs),
            ",".join(str(r["eta_star"]) for r in rs),
            ms([r["test_at_eta_star"]["recall@10"] for r in rs]),
            ms([r["test_at_eta_star"]["ndcg@10"] for r in rs]),
            ms([r["test"].get("0.0", {}).get("recall@10") for r in rs]),
            ms([r["test"].get("1.0", {}).get("recall@10") for r in rs]),
            ms([r["test_head"]["recall@10"] for r in rs]),
            ms([r["test_tail"]["recall@10"] for r in rs]),
            ms([r["probe"]["recall@10_last_item_only"] for r in rs]),
            ms([r["probe"]["recall@10_mean_user"] for r in rs]),
            ms([r["probe"]["shuffle_rel_change"] for r in rs], "{:.3f}"),
            ms([r["probe"]["last_item_r2"] for r in rs], "{:.2f}"),
            ms([r["probe"]["mean_pairwise_cosine"] for r in rs], "{:.3f}"),
        ]
        print(f"| {label} | " + " | ".join(row) + " |")


#%% main
def main():
    args = parse()
    if args.summarize:
        summarize(args.out_dir, args.dataset)
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    specs = DEFAULT_SPECS + (REFERENCE_SPECS if args.with_reference else [])
    for s in args.spec:
        label, name, kind, c, alpha, tau, pattern = s.split("|")
        cfg = {} if c == "-" else {"c": int(c), "alpha": float(alpha)}
        specs.append((label, name, kind, cfg, float(tau), pattern))
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s[0] in keep]

    jobs = []
    for spec in specs:
        paths = sorted(glob.glob(spec[5].format(ds=args.dataset)))
        if not paths:
            print(f"[skip] {spec[0]}: no checkpoint for {spec[5].format(ds=args.dataset)}")
        jobs += [(spec, p) for p in paths]
    if not jobs:
        return

    os.makedirs(os.path.join(args.out_dir, args.dataset), exist_ok=True)
    ds = UserItemTime("./data", args.dataset, "d", 50, 50)
    rng = np.random.default_rng(0)
    valid_ev = build_events(ds, ds.valid_user_item_time, 50, args.limit, rng)
    test_ev = build_events(ds, ds.test_user_item_time, 50, args.limit, rng)
    item_times = torch.tensor(ds.item_time_array / 86400.0, dtype=torch.float32, device=device)
    print(f"device={device}  valid={len(valid_ev['gts'])}  test={len(test_ev['gts'])}")

    for spec, path in jobs:
        seed = re.search(r"seed(\d+)", path).group(1)
        out = os.path.join(args.out_dir, args.dataset, f"{spec[0]}_seed{seed}.json")
        if os.path.exists(out) and not args.overwrite:
            print(f"[cached] {out}")
            continue
        print(f"[eval] {spec[0]} seed{seed}: {path}", flush=True)
        res = evaluate_checkpoint(spec, path, ds, valid_ev, test_ev, item_times, device)
        res["limit"] = args.limit
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        t = res["test_at_eta_star"]
        print(f"   eta*={res['eta_star']}  test R@10={t['recall@10']:.4f} N@10={t['ndcg@10']:.4f}  "
              f"head={res['test_head']['recall@10']:.4f} tail={res['test_tail']['recall@10']:.4f}  probe={res['probe']}", flush=True)

    summarize(args.out_dir, args.dataset)


if __name__ == "__main__":
    main()
