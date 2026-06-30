#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATASET = "ml-1m"
TRAIN_PATH = f"./data/{DATASET}/training_dict.npy"
TIME_PATH = f"./data/{DATASET}/interaction_time_dict.npy"
OUT_DIR = Path(f"./data/{DATASET}/prior_diagnostic_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_interactions(train_path=TRAIN_PATH, time_path=TIME_PATH):
    """
    Expected input format:
      training_dict.npy: {user_id: [item_id1, item_id2, ...]}
      interaction_time_dict.npy: {user_id: {item_id: timestamp}}

    Returns a DataFrame with columns: user, item, timestamp, datetime.
    """
    train_dict = np.load(train_path, allow_pickle=True).item()
    time_dict = np.load(time_path, allow_pickle=True).item()

    rows = []
    missing = 0
    duplicate_user_item = 0

    for u, item_list in train_dict.items():
        tmap = time_dict.get(u, {})
        if len(item_list) != len(set(item_list)):
            duplicate_user_item += 1
        for v in item_list:
            if v in tmap:
                rows.append((u, v, tmap[v]))
            else:
                missing += 1

    df = pd.DataFrame(rows, columns=["user", "item", "timestamp"])

    # Robustly handle second vs millisecond timestamps.
    if df["timestamp"].median() > 1e12:
        df["timestamp"] = df["timestamp"] / 1000.0

    df["timestamp"] = df["timestamp"].astype(float)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df):,} interactions.")
    print(f"Time range: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"# users: {df['user'].nunique():,}, # items: {df['item'].nunique():,}")
    print(f"Missing timestamps skipped: {missing:,}")
    if duplicate_user_item > 0:
        print(f"Warning: {duplicate_user_item:,} users have duplicated items in training_dict.")
        print("         interaction_time_dict is keyed by item, so duplicate timestamps may be collapsed.")

    return df


def add_time_bins(df, n_bins=50):
    """Add integer time bins and return bin edges/centers."""
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    edges = np.linspace(t_min, t_max, n_bins + 1)
    out = df.copy()
    out["time_bin"] = pd.cut(
        out["timestamp"],
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    out["time_bin"] = out["time_bin"].astype(int)
    centers = (edges[:-1] + edges[1:]) / 2.0
    centers_dt = pd.to_datetime(centers, unit="s")
    return out, edges, centers_dt


def plot_top_item_trajectories(df, n_bins=50, top_k=10, out_path=None):
    """
    Figure 1 option A: trajectories of empirical time-varying item shares.

    y-axis: p_hat_emp(v | time bin)
      = count(item v in bin) / count(all interactions in bin)
    """
    dfb, _, centers_dt = add_time_bins(df, n_bins=n_bins)
    top_items = dfb["item"].value_counts().head(top_k).index.tolist()

    counts = (
        dfb[dfb["item"].isin(top_items)]
        .groupby(["time_bin", "item"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(n_bins), columns=top_items, fill_value=0)
    )
    denom = dfb.groupby("time_bin").size().reindex(range(n_bins), fill_value=0)
    share = counts.div(denom.replace(0, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for item in top_items:
        ax.plot(centers_dt, share[item], marker="o", linewidth=1.5, markersize=3, label=str(item))

    ax.set_title("Temporal item-prior dynamics: top item popularity trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Empirical item share $\hat p(v \mid \mathrm{time\ bin})$")
    ax.legend(title="item", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if out_path is None:
        out_path = OUT_DIR / "fig1_top_item_trajectories.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return share


def plot_item_time_heatmap(df, n_bins=50, top_k=30, out_path=None):
    """
    Figure 1 option B: item-time heatmap.

    For visibility, rows are top items and values are row-wise z-scores of empirical item shares.
    This highlights temporal bursts even when absolute popularity levels differ across items.
    """
    dfb, _, centers_dt = add_time_bins(df, n_bins=n_bins)
    top_items = dfb["item"].value_counts().head(top_k).index.tolist()

    counts = (
        dfb[dfb["item"].isin(top_items)]
        .groupby(["time_bin", "item"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(n_bins), columns=top_items, fill_value=0)
    )
    denom = dfb.groupby("time_bin").size().reindex(range(n_bins), fill_value=0)
    share = counts.div(denom.replace(0, np.nan), axis=0).fillna(0.0)

    mat = share[top_items].T.to_numpy(dtype=float)
    row_mean = mat.mean(axis=1, keepdims=True)
    row_std = mat.std(axis=1, keepdims=True)
    mat_z = (mat - row_mean) / np.where(row_std == 0, 1, row_std)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(mat_z, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-wise z-score of item share")

    ax.set_title("Temporal item-prior dynamics: item-time heatmap")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Top items")
    ax.set_yticks(np.arange(len(top_items)))
    ax.set_yticklabels([str(x) for x in top_items], fontsize=7)

    tick_pos = np.linspace(0, n_bins - 1, min(8, n_bins)).astype(int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([centers_dt[i].strftime("%m-%d\n%H:%M") for i in tick_pos], fontsize=8)
    fig.tight_layout()

    if out_path is None:
        out_path = OUT_DIR / "fig1_item_time_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return mat_z


def event_triggered_response(
    df,
    max_lag_hours=48,
    bin_width_hours=1,
    min_item_events=10,
):
    """
    Estimate an event-triggered same-item response curve.

    For each trigger event (v, t), count future events of the same item v in
    [t + tau, t + tau + Delta]. The curve is normalized by an item-specific
    baseline rate, so y=1 means no post-event excitation relative to baseline.
    """
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    duration = ts_max - ts_min
    bin_width = bin_width_hours * 3600.0
    lag_starts = np.arange(0, max_lag_hours, bin_width_hours) * 3600.0

    observed_counts = np.zeros(len(lag_starts), dtype=float)
    baseline_counts = np.zeros(len(lag_starts), dtype=float)
    valid_triggers = np.zeros(len(lag_starts), dtype=float)

    num_items_used = 0

    for item, g in df.groupby("item", sort=False):
        times = np.sort(g["timestamp"].to_numpy(dtype=float))
        if len(times) < min_item_events:
            continue
        num_items_used += 1

        item_rate = len(times) / duration  # events per second

        for j, lag_start in enumerate(lag_starts):
            lag_end = lag_start + bin_width

            # Use only triggers for which the future window is observable.
            triggers = times[times + lag_end <= ts_max]
            if triggers.size == 0:
                continue

            # side='right' excludes the trigger event itself when lag_start=0.
            left = np.searchsorted(times, triggers + lag_start, side="right")
            right = np.searchsorted(times, triggers + lag_end, side="right")
            cnt = (right - left).sum()

            observed_counts[j] += cnt
            baseline_counts[j] += triggers.size * item_rate * bin_width
            valid_triggers[j] += triggers.size

    response = pd.DataFrame({
        "lag_hours": lag_starts / 3600.0,
        "raw_rate_per_hour": np.divide(
            observed_counts,
            valid_triggers * bin_width_hours,
            out=np.full_like(observed_counts, np.nan),
            where=valid_triggers > 0,
        ),
        "relative_to_baseline": np.divide(
            observed_counts,
            baseline_counts,
            out=np.full_like(observed_counts, np.nan),
            where=baseline_counts > 0,
        ),
        "num_valid_triggers": valid_triggers.astype(int),
    })

    print(f"Items used for event-triggered response: {num_items_used:,}")
    return response


def fit_exponential_decay(response):
    """
    Fit y = 1 + a * exp(-beta * lag) using a log-linear approximation.
    Only points with y > 1 are used.
    """
    x = response["lag_hours"].to_numpy(dtype=float)
    y = response["relative_to_baseline"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 1.05)

    if mask.sum() < 3:
        return None

    slope, intercept = np.polyfit(x[mask], np.log(y[mask] - 1.0), deg=1)
    beta = -slope
    a = np.exp(intercept)

    if beta <= 0:
        return None

    y_hat = 1.0 + a * np.exp(-beta * x)
    return a, beta, y_hat


def plot_event_triggered_response(
    df,
    max_lag_hours=48,
    bin_width_hours=1,
    min_item_events=10,
    out_path=None,
):
    """
    Figure 2: Hawkes self-excitation evidence.

    A decreasing curve above 1 shortly after the trigger event supports
    self-exciting item-level dynamics.
    """
    response = event_triggered_response(
        df,
        max_lag_hours=max_lag_hours,
        bin_width_hours=bin_width_hours,
        min_item_events=min_item_events,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        response["lag_hours"],
        response["relative_to_baseline"],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Empirical response",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Baseline")

    fit = fit_exponential_decay(response)
    if fit is not None:
        a, beta, y_hat = fit
        ax.plot(
            response["lag_hours"],
            y_hat,
            linestyle="--",
            linewidth=2,
            label=rf"Exponential fit: $1 + {a:.2f}e^{{-{beta:.3f}\tau}}$",
        )

    ax.set_title("Event-triggered same-item response")
    ax.set_xlabel("Lag after item event, $\tau$ (hours)")
    ax.set_ylabel("Same-item occurrence rate / item baseline rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if out_path is None:
        out_path = OUT_DIR / "fig2_event_triggered_response.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return response


#%%

DATASET = "ml-1m"
TRAIN_PATH = f"./data/{DATASET}/training_dict.npy"
TIME_PATH = f"./data/{DATASET}/interaction_time_dict.npy"
OUT_DIR = Path(f"./data/{DATASET}/prior_diagnostic_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_interactions()
z = plot_item_time_heatmap(df, n_bins=50, top_k=30)

# Figure 1: choose either trajectories or heatmap for the main paper.
plot_top_item_trajectories(df, n_bins=50, top_k=10)
# micro_video_heatmap = plot_item_time_heatmap(df, n_bins=50, top_k=30)
movie_lens_heatmap = plot_item_time_heatmap(df, n_bins=50, top_k=30)

# Figure 2: event-triggered response curve for Hawkes self-excitation evidence.
response = plot_event_triggered_response(
    df,
    max_lag_hours=48,
    bin_width_hours=1,
    min_item_events=10,
)

response.to_csv(OUT_DIR / "fig2_event_triggered_response_values.csv", index=False)
print(f"Saved: {OUT_DIR / 'fig2_event_triggered_response_values.csv'}")


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_temporal_heatmaps_one_column(
    heatmaps,
    dataset_names=("Micro Video", "MovieLens", "KuaiRand"),
    save_path="temporal_heatmaps_one_column.pdf",
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """
    heatmaps: list of 2D arrays
        Each array shape = [num_items, num_time_bins]
    """

    # 공통 color scale 사용
    if vmin is None:
        vmin = min(np.nanpercentile(h, 1) for h in heatmaps)
    if vmax is None:
        vmax = max(np.nanpercentile(h, 99) for h in heatmaps)

    # AAAI 한 column 폭에 맞춘 크기
    # 한 column은 보통 3.25 inch 정도
    fig = plt.figure(figsize=(3.35, 1.35), dpi=600)

    gs = GridSpec(
        1, 4,
        width_ratios=[1, 1, 1, 0.06],
        wspace=0.10,
        left=0.02,
        right=0.96,
        top=0.92,
        bottom=0.28,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    im = None
    for ax, mat, name, panel in zip(
        axes,
        heatmaps,
        dataset_names,
        ["(a)", "(b)", "(c)"]
    ):
        im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        # 내부 title 제거하고, panel label만 아래에 크게 표시
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # tick은 거의 제거해야 한 column에서 안 뭉개짐
        ax.set_xticks([])
        ax.set_yticks([])

        # 테두리도 제거하면 heatmap이 더 크게 보임
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.5, -0.22,
            f"{panel} {name}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )

    # 공통 colorbar 하나만 사용
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label("Normalized\nitem-prior", fontsize=5, labelpad=2)

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

#%%

heatmaps = [
    micro_video_heatmap,
    movielens_heatmap,
    kuairand_heatmap,
]

plot_temporal_heatmaps_one_column(
    heatmaps,
    dataset_names=("Micro Video", "MovieLens", "KuaiRand"),
    save_path="aaai'26/fig/temporal_heatmaps_one_column.pdf",
)
# %%


#%%

import numpy as np
import matplotlib.pyplot as plt


def plot_single_temporal_heatmap(
    mat,
    dataset_name,
    save_path,
    time_labels=None,
    item_labels=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    figsize=(3.3, 2.0),
    dpi=600,
):
    """
    mat: np.ndarray, shape = [num_items, num_time_bins]
        Row = item, Column = time bin.
    """

    mat = np.asarray(mat)

    if vmin is None:
        vmin = np.nanpercentile(mat, 1)
    if vmax is None:
        vmax = np.nanpercentile(mat, 99)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    # 제목은 너무 크면 subcaption과 겹치므로 작게
    ax.set_title(dataset_name, fontsize=8, pad=2)

    ax.set_xlabel("Time bin", fontsize=7)
    ax.set_ylabel("Item", fontsize=7)

    # x tick은 최대 4개만 표시
    num_time_bins = mat.shape[1]
    xticks = np.linspace(0, num_time_bins - 1, min(4, num_time_bins), dtype=int)
    ax.set_xticks(xticks)

    if time_labels is not None:
        ax.set_xticklabels([time_labels[i] for i in xticks], fontsize=5, rotation=0)
    else:
        ax.set_xticklabels(xticks, fontsize=5)

    # y tick도 최대 4개만 표시
    num_items = mat.shape[0]
    yticks = np.linspace(0, num_items - 1, min(4, num_items), dtype=int)
    ax.set_yticks(yticks)

    if item_labels is not None:
        ax.set_yticklabels([item_labels[i] for i in yticks], fontsize=5)
    else:
        ax.set_yticklabels(yticks, fontsize=5)

    ax.tick_params(axis="both", length=2, pad=1)

    # colorbar 작게
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar.set_label("Normalized item-prior", fontsize=6, labelpad=2)

    # 테두리 얇게
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    fig.tight_layout(pad=0.2)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
# %%
plot_single_temporal_heatmap(
    mat=micro_video_heatmap,
    dataset_name="Micro Video",
    save_path="aaai'26/fig/micro-video_heatmap_clean.pdf",
)

#%%

import numpy as np
import matplotlib.pyplot as plt


def plot_single_temporal_heatmap_minimal(
    mat,
    save_path,
    vmin=None,
    vmax=None,
    cmap="viridis",
    figsize=(3.3, 1.6),
    dpi=600,
):
    mat = np.asarray(mat)

    if vmin is None:
        vmin = np.nanpercentile(mat, 1)
    if vmax is None:
        vmax = np.nanpercentile(mat, 99)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    # 제목 제거
    ax.set_title("")

    # 축 라벨 제거
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 눈금과 눈금 숫자 모두 제거
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    # 테두리 제거
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 여백 최소화
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
# %%
plot_single_temporal_heatmap_minimal(
    mat=micro_video_heatmap,
    save_path="aaai'26/fig/micro-video_heatmap_minimal.pdf",
)

plot_single_temporal_heatmap_minimal(
    mat=movielens_heatmap,
    save_path="aaai'26/fig/ml-1m_heatmap_minimal.pdf",
)

plot_single_temporal_heatmap_minimal(
    mat=kuairand_heatmap,
    save_path="aaai'26/fig/kuairand_heatmap_minimal.pdf",
)


#%%


def plot_item_time_heatmap(df, n_bins=50, top_k=30, out_path=None, fontsize=15):
    """
    Figure 1 option B: item-time heatmap.

    For visibility, rows are top items and values are row-wise z-scores of empirical item shares.
    This highlights temporal bursts even when absolute popularity levels differ across items.
    """
    dfb, _, centers_dt = add_time_bins(df, n_bins=n_bins)
    top_items = dfb["item"].value_counts().head(top_k).index.tolist()
    
    counts = (
        dfb[dfb["item"].isin(top_items)]
        .groupby(["time_bin", "item"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(n_bins), columns=top_items, fill_value=0)
    )
    denom = dfb.groupby("time_bin").size().reindex(range(n_bins), fill_value=0)
    share = counts.div(denom.replace(0, np.nan), axis=0).fillna(0.0)

    mat = share[top_items].T.to_numpy(dtype=float)
    row_mean = mat.mean(axis=1, keepdims=True)
    row_std = mat.std(axis=1, keepdims=True)
    mat_z = (mat - row_mean) / np.where(row_std == 0, 1, row_std)

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(mat_z, aspect="auto", cmap="Blues")

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.01,        # heatmap과 colorbar 사이 간격
        fraction=0.025,  # colorbar 폭
        shrink=1.0,       # colorbar 높이
    )
    cbar.set_ticks([])
    cbar.set_label("Frequency", fontsize=fontsize-5)


    ax.set_title(f"{TITLE}", fontsize=fontsize)
    ax.set_xlabel("Time bins", fontsize=fontsize-5)
    ax.set_ylabel(f"Items", fontsize=fontsize-5)
    # ax.set_ylabel(f"            ", fontsize=fontsize)

    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(len(top_items)))

    # 숫자/라벨은 숨기고 눈금만 남김
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(
        axis="both",
        which="both",
        length=3,
        width=0.8,
        direction="out"
    )

    fig.tight_layout()

    if out_path is None:
        out_path = OUT_DIR / f"{DATASET}_heatmap"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return mat_z
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%%


DATASET, TITLE = "micro_video", "Micro Video"

# DATASET, TITLE = "ml-1m", "MovieLens-1M"
# DATASET, TITLE = "kuairand", "KuaiRand"
TRAIN_PATH = f"./data/{DATASET}/training_dict.npy"
TIME_PATH = f"./data/{DATASET}/interaction_time_dict.npy"
OUT_DIR = Path(f"../../Desktop")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_interactions(TRAIN_PATH, TIME_PATH)
z = plot_item_time_heatmap(df, n_bins=50, top_k=10, fontsize=30)

os.getcwd()
# %%
