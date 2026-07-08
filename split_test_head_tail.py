"""Split testing_dict.npy into head/tail subsets by item popularity.

Popularity definitions:
    (1) overall   - item frequency over ALL interactions in training_dict.npy,
                     validation_dict.npy and testing_dict.npy combined.
    (2) recent_Nd - item frequency over interactions that occurred in the last
                     N days before the test period starts (i.e. within
                     training_dict.npy + validation_dict.npy only, so no
                     information from the test period itself leaks into the
                     popularity used to bucket it). Two windows are reported:
                     N=3 and N=7 days.

For every definition, items are ranked by frequency (ties broken by item id)
and the top 20% of *items* are labelled "head", the remaining 80% "tail".
Items that never occur in train/valid/test are excluded from the ranking
universe (they cannot appear in the test split anyway).

interaction_time_dict.npy stores raw epoch timestamps whose unit (seconds vs.
milliseconds) differs by dataset (e.g. kuairand uses milliseconds). The unit
is auto-detected per dataset and everything is normalized to seconds before
the recency windows are applied.

Usage:
    python split_test_head_tail.py --data_path ./data --dataset micro_video ml-1m kuairand
"""
import os
import argparse
import datetime
import numpy as np
from collections import Counter


SECONDS_PER_DAY = 60 * 60 * 24
MS_THRESHOLD = 1e11  # any real unix-seconds epoch today is well below this


def load_dicts(path):
    train_dict = np.load(f"{path}/training_dict.npy", allow_pickle=True).item()
    valid_dict = np.load(f"{path}/validation_dict.npy", allow_pickle=True).item()
    test_dict = np.load(f"{path}/testing_dict.npy", allow_pickle=True).item()
    time_dict = np.load(f"{path}/interaction_time_dict.npy", allow_pickle=True).item()
    return train_dict, valid_dict, test_dict, time_dict


def normalize_time_dict(time_dict):
    """Detect whether timestamps are in seconds or milliseconds and
    normalize everything to seconds."""
    sample = None
    for user_times in time_dict.values():
        for t in user_times.values():
            sample = t
            break
        if sample is not None:
            break

    unit = "ms" if sample is not None and sample > MS_THRESHOLD else "s"
    if unit == "ms":
        time_dict = {u: {i: t / 1000.0 for i, t in items.items()} for u, items in time_dict.items()}
    return time_dict, unit


def iter_interactions(set_dict, time_dict):
    """Yield (user, item, timestamp_seconds) for every interaction in set_dict."""
    for user, items in set_dict.items():
        user_times = time_dict[user]
        for item in items:
            yield user, item, user_times[item]


def compute_popularity(interactions, item_universe):
    counts = Counter(item for _, item, _ in interactions)
    return Counter({item: counts.get(item, 0) for item in item_universe})


def head_tail_items(popularity, head_ratio=0.2):
    ranked = sorted(popularity.items(), key=lambda kv: (-kv[1], kv[0]))
    n_head = max(1, round(len(ranked) * head_ratio))
    head_items = {item for item, _ in ranked[:n_head]}
    tail_items = {item for item, _ in ranked[n_head:]}
    return head_items, tail_items


def filter_dict_by_items(set_dict, item_set):
    return {user: [item for item in items if item in item_set] for user, items in set_dict.items()}


def summarize(name, head_items, tail_items, head_test_dict, tail_test_dict):
    n_head_inter = sum(len(v) for v in head_test_dict.values())
    n_tail_inter = sum(len(v) for v in tail_test_dict.values())
    n_total = n_head_inter + n_tail_inter
    print(
        f"  [{name}] items head={len(head_items)} tail={len(tail_items)} | "
        f"test interactions head={n_head_inter} ({n_head_inter / n_total:.1%}) "
        f"tail={n_tail_inter} ({n_tail_inter / n_total:.1%})"
    )


def fmt(t):
    return datetime.datetime.utcfromtimestamp(t).isoformat()


def process_dataset(data_path, dataset, head_ratio, recent_windows_days):
    path = f"{data_path}/{dataset}"
    train_dict, valid_dict, test_dict, time_dict = load_dicts(path)
    time_dict, unit = normalize_time_dict(time_dict)

    train_inter = list(iter_interactions(train_dict, time_dict))
    valid_inter = list(iter_interactions(valid_dict, time_dict))
    test_inter = list(iter_interactions(test_dict, time_dict))
    all_inter = train_inter + valid_inter + test_inter

    item_universe = {item for _, item, _ in all_inter}
    test_start_time = min(t for _, _, t in test_inter)

    print(f"== {dataset} == (detected time unit: {unit})")
    print(f"  items: {len(item_universe)} | train/valid/test interactions: "
          f"{len(train_inter)}/{len(valid_inter)}/{len(test_inter)}")
    print(f"  train: {fmt(min(t for _, _, t in train_inter))} ~ {fmt(max(t for _, _, t in train_inter))}")
    print(f"  valid: {fmt(min(t for _, _, t in valid_inter))} ~ {fmt(max(t for _, _, t in valid_inter))}")
    print(f"  test : {fmt(test_start_time)} ~ {fmt(max(t for _, _, t in test_inter))}")

    definitions = {}

    # (1) overall popularity - all interactions across train/valid/test
    definitions["overall"] = compute_popularity(all_inter, item_universe)

    # (2) recent popularity - only pre-test (train+valid) interactions within
    # a window ending right before the test period starts, so no test-period
    # data is used to bucket the test set itself.
    pre_test_inter = train_inter + valid_inter
    for n_days in recent_windows_days:
        cutoff = test_start_time - n_days * SECONDS_PER_DAY
        recent_inter = [(u, i, t) for (u, i, t) in pre_test_inter if t >= cutoff]
        definitions[f"recent_{n_days}d"] = compute_popularity(recent_inter, item_universe)

    os.makedirs(path, exist_ok=True)
    for name, popularity in definitions.items():
        head_items, tail_items = head_tail_items(popularity, head_ratio)
        head_test_dict = filter_dict_by_items(test_dict, head_items)
        tail_test_dict = filter_dict_by_items(test_dict, tail_items)

        summarize(name, head_items, tail_items, head_test_dict, tail_test_dict)

        np.save(f"{path}/testing_dict_head_{name}.npy", head_test_dict)
        np.save(f"{path}/testing_dict_tail_{name}.npy", tail_test_dict)
        np.save(f"{path}/item_popularity_{name}.npy", dict(popularity))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, nargs="+", default=["micro_video", "ml-1m", "kuairand"])
    parser.add_argument("--head_ratio", type=float, default=0.2)
    parser.add_argument("--recent_windows_days", type=int, nargs="+", default=[3, 7])
    args = parser.parse_args()

    for dataset in args.dataset:
        process_dataset(args.data_path, dataset, args.head_ratio, args.recent_windows_days)


if __name__ == "__main__":
    main()
