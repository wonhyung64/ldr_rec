"""3x3 grid bar chart of Recall@10 by item popularity (Head/Tail) for
Micro Video / MovieLens-1M / KuaiRand, across three popularity windows
(Overall / Recent 3d / Recent 5d), comparing 4 backbone models (ours) and
4 baselines.

Reads assets/head_tail.xlsx (sheets: ex2_micro_video, ex2_ml_1m, ex2_kuairand),
each row = one run (one seed) with columns head_overall/head_3d/head_5d/
tail_overall/tail_3d/tail_5d holding Recall@10. Bars show the mean across the
4 seeds per method; error bars show +/- 1 std across seeds.

Usage:
    python plot_head_tail.py
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter


def no_leading_zero(x, pos):
    """Format tick labels like .05 / .10 instead of 0.05 / 0.10 (0 stays 0)."""
    if x == 0:
        return "0"
    s = f"{x:.2f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)


def lighten(hex_color, amount=0.35):
    """Blend a hex color toward white by `amount` (0 = unchanged, 1 = white)."""
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.titleweight": "normal",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 15,
    "axes.edgecolor": "#4d4d4d",
    "axes.linewidth": 0.9,
})

XLSX_PATH = "assets/head_tail2.xlsx"
# head_tail2.xlsx has empty stub rows for KuaiRand cells whose rerun hasn't
# finished; fall back to their last-known values from the previous results
# file as a temporary stand-in.
FALLBACK_XLSX_PATH = "assets/head_tail.xlsx"
OUT_PATH = "assets/head_tail_barplot.png"

SHEET_TO_DATASET = {
    "ex2_micro_video": "Micro Video",
    "ex2_ml_1m": "MovieLens-1M",
    "ex2_kuairand": "KuaiRand",
}
DATASETS = ["Micro Video", "MovieLens-1M", "KuaiRand"]

PERIODS = [
    ("Overall", "overall"),
    ("Recent 3d", "3d"),
    ("Recent 7d", "5d"),  # xlsx column suffix is "5d" but represents a 7-day window
]

# display name -> possible run-name prefixes (with the trailing timestamp/seed
# suffix stripped), in legend/plot order. SASRec/TiSASRec/FEARec/BSARec were
# rerun with a fixed setup (the "_fixed_" prefix); KuaiRand hasn't finished
# rerunning all of them yet, so the older "_pop_" runs are kept as a fallback
# so those bars aren't empty (see PLACEHOLDER_CELLS below).
METHOD_ALIASES = {
    "SASRec": ["debiased_seq_rec_weighted_fixed_sasrec", "debiased_seq_rec_weighted_pop_sasrec"],
    "TiSASRec": [
        "debiased_seq_rec_tisasrec_weighted_fixed_tisasrec",
        "debiased_seq_rec_tisasrec_weighted_pop_tisasrec",
    ],
    "FEARec": ["debiased_seq_rec_weighted_fixed_fearec", "debiased_seq_rec_weighted_pop_fearec"],
    "BSARec": ["debiased_seq_rec_weighted_fixed_bsarec", "debiased_seq_rec_weighted_pop_bsarec"],
    "DCRec": ["dcrec_pop"],
    "PAUDRec": ["paud_rec_pop"],
    "PAAC": ["paac_pop"],
    "TPAB": ["tpab"],
}
METHODS = list(METHOD_ALIASES.keys())
_PREFIX_TO_METHOD = {p: m for m, prefixes in METHOD_ALIASES.items() for p in prefixes}
_FIXED_PREFIXES = {prefixes[0] for prefixes in METHOD_ALIASES.values()}

# SASRec/TiSASRec/FEARec/BSARec are sampled from the paper's eta line plot
# legend (teal, blue-gray, rose, mustard). DCRec/PAUDRec/PAAC/SAPID are new
# baselines not in that figure; they use softer hues from the same muted
# qualitative family (steel blue, plum, terracotta, sage) so the palette
# stays cohesive. Swap in exact hex codes once available.
_BASE_COLORS = {
    "SASRec": "#3AA69B",
    "TiSASRec": "#8C95A3",
    "FEARec": "#DB7396",
    "BSARec": "#E8A93D",
    "DCRec": "#4C72B0",
    "PAUDRec": "#9B7FB8",
    "PAAC": "#C0785A",
    "TPAB": "#94A84C",
}
COLORS = {method: lighten(hex_color, 0.35) for method, hex_color in _BASE_COLORS.items()}


def match_method(run_name):
    prefix = re.sub(r"_\d{6}_\d{6}_\d+$", "", str(run_name))
    return _PREFIX_TO_METHOD.get(prefix), prefix


VALUE_COLS = [f"{g}_{suffix}" for suffix in ["overall", "3d", "5d"] for g in ("head", "tail")]


def read_sheet(xlsx_path, sheet):
    df = pd.read_excel(pd.ExcelFile(xlsx_path), sheet_name=sheet, header=0)
    # some sheets label the recent-window columns "_7d" (the actual window
    # length) instead of "_5d"; normalize to "_5d" so both line up.
    df = df.rename(columns={"head_7d": "head_5d", "tail_7d": "tail_5d"})
    df = df.dropna(subset=["Name"]).copy()
    matched = df["Name"].apply(match_method)
    df["method"] = matched.apply(lambda t: t[0])
    df["prefix"] = matched.apply(lambda t: t[1])
    return df[df["method"].notna()].copy()


def load_data():
    frames = []
    for sheet, dataset in SHEET_TO_DATASET.items():
        df = read_sheet(XLSX_PATH, sheet)
        df["dataset"] = dataset
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    # cells whose rerun ("_fixed_") results aren't in yet are empty stub rows
    # (all value columns NaN) matched only via the older "_pop_" prefix.
    is_stale_prefix = ~full["prefix"].isin(_FIXED_PREFIXES)
    is_empty = full[VALUE_COLS].isna().all(axis=1)
    placeholders = sorted(set(zip(full.loc[is_stale_prefix & is_empty, "dataset"],
                                   full.loc[is_stale_prefix & is_empty, "method"])))

    stats = full.groupby(["dataset", "method"])[VALUE_COLS].agg(["mean", "std"])

    # backfill placeholder cells with their last-known values from the
    # previous results file, as a temporary stand-in.
    dataset_to_sheet = {v: k for k, v in SHEET_TO_DATASET.items()}
    for dataset, method in placeholders:
        fb = read_sheet(FALLBACK_XLSX_PATH, dataset_to_sheet[dataset])
        fb = fb[fb["method"] == method]
        stats.loc[(dataset, method), :] = fb[VALUE_COLS].agg(["mean", "std"]).T.stack().values

    return stats, placeholders


N_OURS = 4  # first N_OURS entries in METHODS (SASRec/TiSASRec/FEARec/BSARec) are ours


def subgroup_offsets(n_methods, n_ours, bar_w, subgroup_gap):
    """Bar centers with no gap within each subgroup (ours / baselines) and a
    visible gap between the two subgroups."""
    n_base = n_methods - n_ours
    block_ours = n_ours * bar_w
    block_base = n_base * bar_w
    total_width = block_ours + block_base + subgroup_gap
    start = -total_width / 2
    offsets = []
    for i in range(n_methods):
        if i < n_ours:
            left = start + i * bar_w
        else:
            left = start + block_ours + subgroup_gap + (i - n_ours) * bar_w
        offsets.append(left + bar_w / 2)
    return offsets


def plot(stats):
    n_rows, n_cols = len(PERIODS), len(DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.05 * n_rows), sharex=False)

    groups = ["Head", "Tail"]
    n_methods = len(METHODS)
    bar_w = 0.09
    offsets = subgroup_offsets(n_methods, N_OURS, bar_w, subgroup_gap=0.0)
    group_x = np.arange(len(groups))
    ours_lo, ours_hi = offsets[0] - bar_w / 2, offsets[N_OURS - 1] + bar_w / 2

    bars_for_legend = {}
    for r, (period_label, period_key) in enumerate(PERIODS):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#c2c2c2", alpha=0.9)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#888888")

            # shaded band behind "ours" bars (SASRec/TiSASRec/FEARec/BSARec) in
            # each Head/Tail cluster, to set them apart from the baselines.
            for gx in group_x:
                ax.axvspan(gx + ours_lo, gx + ours_hi, color="#ffe27a", alpha=0.35, zorder=0, lw=0)

            max_height = 0.0
            for m_idx, method in enumerate(METHODS):
                means, stds = [], []
                for group_key in ("head", "tail"):
                    col = f"{group_key}_{period_key}"
                    mean = stats.loc[(dataset, method), (col, "mean")]
                    std = stats.loc[(dataset, method), (col, "std")]
                    means.append(mean)
                    stds.append(std)
                offset = offsets[m_idx]
                ax.bar(
                    group_x + offset, means, width=bar_w,
                    yerr=stds, capsize=1.3, error_kw={"linewidth": 0.6, "ecolor": "#555555", "capthick": 0.6},
                    color=COLORS[method], edgecolor="black", linewidth=0.7, zorder=3,
                )
                max_height = max(max_height, max(m + s for m, s in zip(means, stds)))
                bars_for_legend[method] = plt.Rectangle(
                    (0, 0), 1, 1, facecolor=COLORS[method], edgecolor="black", linewidth=0.7,
                )

            ax.set_xlim(-0.55, 1.55)
            ax.set_ylim(0, max_height * 1.18)
            ax.set_xticks(group_x)
            ax.set_xticklabels(groups)
            ax.yaxis.set_major_formatter(FuncFormatter(no_leading_zero))

            for gx in group_x:
                ax.text(
                    gx + (ours_lo + ours_hi) / 2, max_height * 1.18 * 0.965, "Ours",
                    ha="center", va="top", fontsize=10, fontweight="bold", color="black", zorder=4,
                )

            if r == 0:
                ax.set_title(dataset, pad=10)
            if c == 0:
                ax.set_ylabel(period_label)
            else:
                ax.set_ylabel("")

    fig.legend(
        list(bars_for_legend.values()), list(bars_for_legend.keys()),
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=n_methods,
        frameon=False, handlelength=1.0, handleheight=1.0, columnspacing=1.1,
    )

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.05, top=0.86, hspace=0.26, wspace=0.16)
    return fig


def main():
    stats, placeholders = load_data()
    fig = plot(stats)
    fig.savefig(OUT_PATH, dpi=300)
    print(f"saved {OUT_PATH}")
    if placeholders:
        print("placeholder (old-run) cells used:")
        for dataset, method in placeholders:
            print(f"  {dataset} / {method}")


if __name__ == "__main__":
    main()
