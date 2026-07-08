#%%
"""Sensitivity analysis plots for alpha1 (paper notation: eta).

Reads assets/sensitivity_alpha1.xlsx and, for each dataset, draws one line
plot with one curve per model (Recall@10 vs. eta).
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

XLSX_PATH = "assets/sensitivity_alpha1.xlsx"
SHEET_NAME = "sen_gamma"
OUT_DIR = Path("assets/sensitivity_alpha1_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_TITLES = {
    "micro_video": "Micro Video",
    "ml-1m": "MovieLens-1M",
    "kuairand": "KuaiRand",
}

MODEL_LABELS = {
    "mf": "MF",
    "ncf": "NCF",
    "grurec": "GRU4Rec",
    "sasrec": "SASRec",
    "tisasrec": "TiSASRec",
    "fearec": "FEARec",
    "bsarec": "BSARec",
}
MODEL_ORDER = list(MODEL_LABELS.keys())

ALPHA1_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
# Palette matched to the paper's figure (navy header, steel/teal blues,
# pink and peach accent boxes, slate gray).
COLORS = [
    "#2E3F5C",  # navy (header bar)
    "#3E7CB1",  # steel blue
    "#5FA8D3",  # sky blue
    "#4FB0A5",  # teal
    "#8C97A8",  # slate gray
    "#D98CA0",  # rose pink (Ranking Score box)
    "#E8B172",  # peach/tan (Prior Negative Sampling box)
]


def load_data(xlsx_path=XLSX_PATH, sheet_name=SHEET_NAME):
    """Parse the raw sheet into a tidy DataFrame with columns:
    dataset, model, alpha1, recall_10."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    rows = []
    for _, r in raw.iterrows():
        dataset, model, alpha1, recall_10 = r[2], r[6], r[7], r[4]
        if dataset not in DATASET_TITLES or model not in MODEL_LABELS:
            continue
        rows.append((dataset, model, float(alpha1), float(recall_10)))

    df = pd.DataFrame(rows, columns=["dataset", "model", "alpha1", "recall_10"])
    # Average over seeds, in case of duplicates.
    df = df.groupby(["dataset", "model", "alpha1"], as_index=False)["recall_10"].mean()
    return df


def _draw_lines(ax, sub):
    lines = {}
    for i, model in enumerate(MODEL_ORDER):
        m = sub[sub["model"] == model].sort_values("alpha1")
        if m.empty:
            continue
        (line,) = ax.plot(
            m["alpha1"],
            m["recall_10"],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            linewidth=1.8,
            markersize=5.5,
            markeredgewidth=0,
            label=MODEL_LABELS[model],
        )
        lines[model] = line
    return lines


def plot_combined(df, out_path=None, figsize=(10, 3.1)):
    """One row of 3 compact subplots (one per dataset) sharing a single
    legend, sized for a paper's full text width."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    handles, labels_seen = None, None
    for ax, dataset in zip(axes, DATASET_TITLES):
        sub = df[df["dataset"] == dataset]
        lines = _draw_lines(ax, sub)
        if handles is None:
            handles = [lines[m] for m in MODEL_ORDER if m in lines]
            labels_seen = [MODEL_LABELS[m] for m in MODEL_ORDER if m in lines]

        ax.set_title(DATASET_TITLES[dataset], fontsize=12, pad=4)
        ax.set_xlabel(r"$\eta$", fontsize=12)
        ax.set_xticks(ALPHA1_VALUES)
        ax.tick_params(axis="both", labelsize=9)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Recall@10", fontsize=12)

    fig.legend(
        handles,
        labels_seen,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=7,
        fontsize=9.5,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is None:
        out_path = OUT_DIR / "alpha1_sensitivity_combined.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_dataset(df, dataset, out_path=None, figsize=(3.6, 3.1)):
    """Standalone compact single-dataset plot with a small in-panel legend."""
    sub = df[df["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=figsize)
    _draw_lines(ax, sub)

    ax.set_xlabel(r"$\eta$", fontsize=13)
    ax.set_ylabel("Recall@10", fontsize=13)
    ax.set_title(DATASET_TITLES[dataset], fontsize=13, pad=4)
    ax.set_xticks(ALPHA1_VALUES)
    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, ncol=2, loc="best", frameon=False, handletextpad=0.4)
    fig.tight_layout()

    if out_path is None:
        out_path = OUT_DIR / f"alpha1_sensitivity_{dataset}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


#%%
if __name__ == "__main__":
    data = load_data()
    for dataset in DATASET_TITLES:
        plot_dataset(data, dataset)
    plot_combined(data)
