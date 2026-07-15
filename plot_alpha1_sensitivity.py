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
ALPHA1_VALUES_WITH_ETA0 = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
ALPHA1_VALUES_FULL = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

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


def _draw_lines(ax, sub, alpha1_values, linewidth=3.0, markersize=10):
    # Plot at evenly-spaced categorical positions rather than the raw eta
    # values: the sweep is irregular (0, 0.1, 0.3, ..., 0.9, 1), so a linear
    # x-axis crowds tick labels together wherever two etas sit close (e.g.
    # 0/0.1 or 0.9/1). Equal spacing keeps every tick label legible.
    position = {v: i for i, v in enumerate(alpha1_values)}
    lines = {}
    for i, model in enumerate(MODEL_ORDER):
        m = sub[(sub["model"] == model) & (sub["alpha1"].isin(alpha1_values))].sort_values("alpha1")
        if m.empty:
            continue
        (line,) = ax.plot(
            [position[v] for v in m["alpha1"]],
            m["recall_10"],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            linewidth=linewidth,
            markersize=markersize,
            markeredgewidth=0,
            label=MODEL_LABELS[model],
        )
        lines[model] = line
    return lines


def _xtick_labels(alpha1_values):
    return [f"{v:g}" for v in alpha1_values]


def _set_xticks(ax, alpha1_values):
    ax.set_xticks(range(len(alpha1_values)))
    ax.set_xticklabels(_xtick_labels(alpha1_values))
    ax.set_xlim(-0.5, len(alpha1_values) - 0.5)


# NOTE: this figure is meant to be shrunk to a single (~3.3in) paper column,
# so fonts/lines are drawn much larger than they look on screen here.
def plot_combined(df, alpha1_values=ALPHA1_VALUES, out_path=None, figsize=(15, 4.9)):
    """One row of 3 compact subplots (one per dataset) sharing a single
    legend, sized for a paper's full text width."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.32)

    handles, labels_seen = None, None
    for ax, dataset in zip(axes, DATASET_TITLES):
        sub = df[df["dataset"] == dataset]
        lines = _draw_lines(ax, sub, alpha1_values)
        if handles is None:
            handles = [lines[m] for m in MODEL_ORDER if m in lines]
            labels_seen = [MODEL_LABELS[m] for m in MODEL_ORDER if m in lines]

        ax.set_title(DATASET_TITLES[dataset], fontsize=26, pad=10)
        ax.set_xlabel(r"$\eta$", fontsize=28)
        _set_xticks(ax, alpha1_values)
        ax.tick_params(axis="both", labelsize=20, length=6, width=1.2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(True, axis="y", linestyle=":", linewidth=1.1, alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(1.2)

    axes[0].set_ylabel("Recall@10", fontsize=26)

    fig.legend(
        handles,
        labels_seen,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=7,
        fontsize=22,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if out_path is None:
        out_path = OUT_DIR / "alpha1_sensitivity_combined.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_dataset(df, dataset, alpha1_values=ALPHA1_VALUES, out_path=None, figsize=(6.5, 5.6)):
    """Standalone compact single-dataset plot with a small in-panel legend."""
    sub = df[df["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=figsize)
    _draw_lines(ax, sub, alpha1_values, linewidth=2.6, markersize=9)

    ax.set_xlabel(r"$\eta$", fontsize=24)
    ax.set_ylabel("Recall@10", fontsize=22)
    ax.set_title(DATASET_TITLES[dataset], fontsize=24, pad=8)
    _set_xticks(ax, alpha1_values)
    ax.tick_params(axis="both", labelsize=18, length=6, width=1.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(True, axis="y", linestyle=":", linewidth=1.0, alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)
    ax.legend(fontsize=14, ncol=2, loc="best", frameon=False, handletextpad=0.4)
    fig.tight_layout()

    if out_path is None:
        out_path = OUT_DIR / f"alpha1_sensitivity_{dataset}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


#%%
if __name__ == "__main__":
    data = load_data()

    # Original sweep (eta in {0.1, ..., 0.9}), unchanged filenames.
    for dataset in DATASET_TITLES:
        plot_dataset(data, dataset, ALPHA1_VALUES)
    plot_combined(data, ALPHA1_VALUES)

    # New sweep including eta = 0 (resid-score-only ranking), saved under
    # distinct filenames so it can be compared against the original.
    for dataset in DATASET_TITLES:
        plot_dataset(
            data, dataset, ALPHA1_VALUES_WITH_ETA0,
            out_path=OUT_DIR / f"alpha1_sensitivity_{dataset}_with_eta0.png",
        )
    plot_combined(
        data, ALPHA1_VALUES_WITH_ETA0,
        out_path=OUT_DIR / "alpha1_sensitivity_combined_with_eta0.png",
    )

    # Full sweep including eta = 0 and eta = 1, saved under distinct
    # filenames so it can be compared against the other two versions.
    for dataset in DATASET_TITLES:
        plot_dataset(
            data, dataset, ALPHA1_VALUES_FULL,
            out_path=OUT_DIR / f"alpha1_sensitivity_{dataset}_full.png",
        )
    plot_combined(
        data, ALPHA1_VALUES_FULL,
        out_path=OUT_DIR / "alpha1_sensitivity_combined_full.png",
    )
