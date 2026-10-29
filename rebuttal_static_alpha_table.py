"""Summarize the static-popularity exponent sweep (AAAI-27 rebuttal, reviewer VShq).

Reads ./results/static_pop/*.json written by baseline/*_static_prior.py --pop-power=a and,
for each (dataset, backbone), picks alpha (and eta) by validation Recall@10.

    python rebuttal_static_alpha_table.py [--results ./results/static_pop] [--population-std]

The last table reports mean±std over seeds with (alpha*, eta*) fixed from seed-1 validation
(seeds 2-4 come from report_rebuttal_static_alpha_seeds.sh).
"""
import json
import argparse
import statistics
from glob import glob
from collections import defaultdict


DATASETS = ["micro_video", "ml-1m", "kuairand"]
DATASET_NAMES = {"micro_video": "Micro-Video", "ml-1m": "MovieLens-1M", "kuairand": "KuaiRand"}
MODEL_NAMES = {"mf": "MF", "grurec": "GRU", "sasrec": "SASRec", "tisasrec": "TiSASRec", "fearec": "FEARec", "bsarec": "BSARec"}

# paper, test Recall@10 / NDCG@10, mean over 4 seeds
HAWKES = {  # Table 1 "+(Ours)" (= Table 3 Hawkes row)
    ("micro_video", "mf"): (0.0602, 0.0302), ("ml-1m", "mf"): (0.0378, 0.0177), ("kuairand", "mf"): (0.0688, 0.0329),
    ("micro_video", "grurec"): (0.0925, 0.0555), ("ml-1m", "grurec"): (0.0684, 0.0335), ("kuairand", "grurec"): (0.0535, 0.0268),
    ("micro_video", "sasrec"): (0.1629, 0.0807), ("ml-1m", "sasrec"): (0.1068, 0.0519), ("kuairand", "sasrec"): (0.0754, 0.0366),
    ("micro_video", "tisasrec"): (0.1750, 0.0871), ("ml-1m", "tisasrec"): (0.1108, 0.0536), ("kuairand", "tisasrec"): (0.0763, 0.0369),
    ("micro_video", "fearec"): (0.1760, 0.0896), ("ml-1m", "fearec"): (0.0852, 0.0408), ("kuairand", "fearec"): (0.0598, 0.0297),
    ("micro_video", "bsarec"): (0.1728, 0.0883), ("ml-1m", "bsarec"): (0.0335, 0.0160), ("kuairand", "bsarec"): (0.0638, 0.0308),
}
HAWKES_STD = {  # Table 1 "+(Ours)", std over the same 4 seeds
    ("micro_video", "mf"): (0.0014, 0.0005), ("ml-1m", "mf"): (0.0005, 0.0002), ("kuairand", "mf"): (0.0007, 0.0001),
    ("micro_video", "grurec"): (0.0070, 0.0054), ("ml-1m", "grurec"): (0.0010, 0.0006), ("kuairand", "grurec"): (0.0005, 0.0003),
    ("micro_video", "sasrec"): (0.0025, 0.0014), ("ml-1m", "sasrec"): (0.0022, 0.0012), ("kuairand", "sasrec"): (0.0003, 0.0002),
    ("micro_video", "tisasrec"): (0.0028, 0.0009), ("ml-1m", "tisasrec"): (0.0008, 0.0005), ("kuairand", "tisasrec"): (0.0009, 0.0002),
    ("micro_video", "fearec"): (0.0018, 0.0006), ("ml-1m", "fearec"): (0.0019, 0.0010), ("kuairand", "fearec"): (0.0008, 0.0005),
    ("micro_video", "bsarec"): (0.0022, 0.0010), ("ml-1m", "bsarec"): (0.0002, 0.0001), ("kuairand", "bsarec"): (0.0007, 0.0004),
}
STATIC_A1 = {  # Table 3 "Static popularity" Recall@10 (alpha=1, eta=0.5 protocol)
    ("micro_video", "mf"): 0.0598, ("ml-1m", "mf"): 0.0343, ("kuairand", "mf"): 0.0674,
    ("micro_video", "grurec"): 0.1114, ("ml-1m", "grurec"): 0.0641, ("kuairand", "grurec"): 0.0433,
    ("micro_video", "sasrec"): 0.1268, ("ml-1m", "sasrec"): 0.1024, ("kuairand", "sasrec"): 0.0689,
    ("micro_video", "tisasrec"): 0.1388, ("ml-1m", "tisasrec"): 0.1049, ("kuairand", "tisasrec"): 0.0712,
    ("micro_video", "fearec"): 0.1406, ("ml-1m", "fearec"): 0.0967, ("kuairand", "fearec"): 0.0686,
    ("micro_video", "bsarec"): 0.0299, ("ml-1m", "bsarec"): 0.0662, ("kuairand", "bsarec"): 0.0558,
}
STATIC_A1_STD = {  # Table 3 "Static popularity", Recall@10 std
    ("micro_video", "mf"): 0.0004, ("ml-1m", "mf"): 0.0004, ("kuairand", "mf"): 0.0005,
    ("micro_video", "grurec"): 0.0021, ("ml-1m", "grurec"): 0.0007, ("kuairand", "grurec"): 0.0005,
    ("micro_video", "sasrec"): 0.0015, ("ml-1m", "sasrec"): 0.0010, ("kuairand", "sasrec"): 0.0007,
    ("micro_video", "tisasrec"): 0.0017, ("ml-1m", "tisasrec"): 0.0001, ("kuairand", "tisasrec"): 0.0007,
    ("micro_video", "fearec"): 0.0008, ("ml-1m", "fearec"): 0.0016, ("kuairand", "fearec"): 0.0009,
    ("micro_video", "bsarec"): 0.0097, ("ml-1m", "bsarec"): 0.0154, ("kuairand", "bsarec"): 0.0134,
}
VANILLA = {  # Table 1 backbone (uniform negatives, i.e. alpha=0)
    ("micro_video", "mf"): (0.0121, 0.0061), ("ml-1m", "mf"): (0.0308, 0.0145), ("kuairand", "mf"): (0.0427, 0.0218),
    ("micro_video", "sasrec"): (0.1217, 0.0604), ("ml-1m", "sasrec"): (0.0805, 0.0348), ("kuairand", "sasrec"): (0.0515, 0.0220),
    ("micro_video", "tisasrec"): (0.1453, 0.0691), ("ml-1m", "tisasrec"): (0.1058, 0.0464), ("kuairand", "tisasrec"): (0.0590, 0.0248),
}
VANILLA_STD = {
    ("micro_video", "mf"): (0.0009, 0.0006), ("ml-1m", "mf"): (0.0006, 0.0004), ("kuairand", "mf"): (0.0006, 0.0004),
    ("micro_video", "sasrec"): (0.0385, 0.0120), ("ml-1m", "sasrec"): (0.0223, 0.0083), ("kuairand", "sasrec"): (0.0112, 0.0040),
    ("micro_video", "tisasrec"): (0.0049, 0.0026), ("ml-1m", "tisasrec"): (0.0215, 0.0126), ("kuairand", "tisasrec"): (0.0008, 0.0003),
}

KEY = "recall@10"
SELECT_SEED = 1  # alpha* and eta* are chosen on this seed's validation split, then kept fixed for the other seeds


def load(results_dir):
    runs = defaultdict(dict)
    for path in sorted(glob(f"{results_dir}/*.json")):
        r = json.load(open(path))
        runs[(r["dataset"], r["model_name"])][(r["pop_power"], r["seed"])] = r
    return runs


def best_eta(r, etas=None):
    etas = etas or list(r["valid"].keys())
    return max(etas, key=lambda e: r["valid"][e][KEY])


def select_alpha_eta(by_alpha):
    # joint (alpha, eta) selection on validation
    return max(((a, e) for a, r in by_alpha.items() for e in r["valid"]),
               key=lambda ae: by_alpha[ae[0]]["valid"][ae[1]][KEY])


def pct(a, b):
    return f"{100 * (a - b) / b:+.1f}%"


def pm(mean, std):
    return f"{mean:.4f}" if std is None else f"{mean:.4f}±{std:.4f}"


def cell_order(k):
    return (DATASETS.index(k[0]) if k[0] in DATASETS else 99, k[1])


def seed_summary(runs, population_std):
    # sample std reproduces the paper (Tab.3 SASRec/Micro-Video static: 0.1268±0.0015 from the same 4 checkpoints)
    sd = statistics.pstdev if population_std else statistics.stdev
    std = lambda v: sd(v) if len(v) > 1 else None
    print(f"\n## Mean ± std over seeds (alpha*, eta* fixed from seed-{SELECT_SEED} validation; "
          f"{'population' if population_std else 'sample'} std)\n")
    print("| Dataset | Backbone | alpha*, eta* | seeds | Vanilla R@10 (Tab.1) | Static a=1 R@10 (Tab.3) "
          "| Tuned static R@10 | Tuned static N@10 | Hawkes R@10 (Tab.1) | Hawkes N@10 (Tab.1) | Hawkes vs tuned |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (dataset, model) in sorted(runs, key=cell_order):
        cell = runs[(dataset, model)]
        by_alpha = {a: r for (a, seed), r in cell.items() if seed == SELECT_SEED}
        if not by_alpha:
            continue
        a_star, e_star = select_alpha_eta(by_alpha)
        picked = [cell[(a, seed)] for (a, seed) in sorted(cell, key=lambda k: k[1]) if a == a_star]
        rec = [r["test"][e_star][KEY] for r in picked]
        ndcg = [r["test"][e_star]["ndcg@10"] for r in picked]

        key = (dataset, model)
        hk, hk_sd = HAWKES.get(key), HAWKES_STD.get(key)
        van, van_sd = VANILLA.get(key), VANILLA_STD.get(key)
        cells = [
            DATASET_NAMES.get(dataset, dataset),
            MODEL_NAMES.get(model, model),
            f"{a_star:g}, {e_star}",
            ",".join(str(r["seed"]) for r in picked),
            pm(van[0], van_sd[0]) if van else "n/a",
            pm(STATIC_A1[key], STATIC_A1_STD[key]) if key in STATIC_A1 else "n/a",
            pm(statistics.mean(rec), std(rec)),
            pm(statistics.mean(ndcg), std(ndcg)),
            pm(hk[0], hk_sd[0]) if hk else "n/a",
            pm(hk[1], hk_sd[1]) if hk else "n/a",
            pct(hk[0], statistics.mean(rec)) if hk else "n/a",
        ]
        print("| " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="./results/static_pop")
    parser.add_argument("--population-std", action="store_true", help="ddof=0 std (np.std default) instead of the paper's sample std")
    args = parser.parse_args()

    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no results in {args.results}")

    print("## Per-alpha detail (validation-selected eta per alpha)\n")
    print("| Dataset | Backbone | seed | alpha | eta* | Valid R@10 | Test R@10 | Test N@10 | Test R@10 (eta=0.5) |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = []
    for (dataset, model) in sorted(runs, key=cell_order):
        by_seed = defaultdict(dict)
        for (a, seed), r in runs[(dataset, model)].items():
            by_seed[seed][a] = r
        for seed, by_alpha in sorted(by_seed.items()):
            for a, r in sorted(by_alpha.items()):
                e = best_eta(r)
                print(f"| {DATASET_NAMES.get(dataset, dataset)} | {MODEL_NAMES.get(model, model)} | {seed} | {a:g} | {e} "
                      f"| {r['valid'][e][KEY]:.4f} | {r['test'][e][KEY]:.4f} | {r['test'][e]['ndcg@10']:.4f} | {r['test']['0.5'][KEY]:.4f} |")

            a_star, e_star = select_alpha_eta(by_alpha)
            # Table 3 protocol: eta fixed to 0.5, select alpha only
            a_half = max(by_alpha, key=lambda a: by_alpha[a]["valid"]["0.5"][KEY])
            summary.append((dataset, model, seed, sorted(by_alpha), a_star, e_star, by_alpha[a_star]["test"][e_star],
                            a_half, by_alpha[a_half]["test"]["0.5"]))

    print("\n## Tuned static vs. Hawkes (test, alpha/eta chosen by validation Recall@10)\n")
    print("| Dataset | Backbone | alphas searched | Vanilla (a=0) R@10 | Static a=1 R@10 (Tab.3) "
          "| Tuned static [a*, eta=0.5] R@10 | Tuned static [a*, eta*] R@10 / N@10 | Hawkes R@10 / N@10 (Tab.1) | Hawkes vs tuned |")
    print("|---|---|---|---|---|---|---|---|---|")
    missing_a1 = []
    for dataset, model, seed, alphas, a_star, e_star, t_star, a_half, t_half in summary:
        hk = HAWKES.get((dataset, model))
        van = VANILLA.get((dataset, model))
        s1 = STATIC_A1.get((dataset, model))
        if 1.0 not in alphas:
            missing_a1.append(f"{dataset}/{model}")
        cells = [
            DATASET_NAMES.get(dataset, dataset),
            f"{MODEL_NAMES.get(model, model)} (seed {seed})",
            ", ".join(f"{a:g}" for a in alphas),
            f"{van[0]:.4f}" if van else "n/a",
            f"{s1:.4f}" if s1 else "n/a",
            f"{t_half[KEY]:.4f} (a={a_half:g})",
            f"{t_star[KEY]:.4f} / {t_star['ndcg@10']:.4f} (a={a_star:g}, eta={e_star})",
            f"{hk[0]:.4f} / {hk[1]:.4f}" if hk else "n/a",
            pct(hk[0], t_star[KEY]) if hk else "n/a",
        ]
        print("| " + " | ".join(cells) + " |")

    if missing_a1:
        print(f"\nNOTE: no alpha=1 JSON for {', '.join(missing_a1)} (eval-only re-scoring of the Table 3 checkpoint did not run "
              "or the checkpoint is gone), so alpha=1 is not part of the selection there; compare with the Tab.3 column.")
    print("\nRows above are per seed; Tab.1/Tab.3 references are 4-seed means.")

    seed_summary(runs, args.population_std)


if __name__ == "__main__":
    main()
