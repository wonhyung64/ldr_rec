import torch
import random
import argparse
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hawkes-prior + Residual recommender"
    )
    parser.add_argument("--cred_path", type=str, default="./assets")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--device", type=str, default="none")
    parser.add_argument("--topks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--ranking_mode", type=str, choices=["biased", "unbiased"], default="biased")
    parser.add_argument("--eval_item_chunk_size", type=int, default=4096)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--patience", type=int, default=999)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--eval_every", type=int, default=600)

    """common options"""
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="micro_video")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--shared_lr", type=float, default=1e-3)
    parser.add_argument("--shared_decay", type=float, default=0.0)

    """residual options"""
    parser.add_argument("--residual_num_negatives", type=int, default=5)
    parser.add_argument("--residual_lr", type=float, default=1e-3)
    parser.add_argument("--residual_decay", type=float, default=0.0)
    parser.add_argument("--residual_weight", type=float, default=1.0)
    parser.add_argument("--residual_vector_norm", type=bool, default=True)
    parser.add_argument("--tau", type=float, default=1.0)

    """prior options"""
    parser.add_argument("--prior_hidden_dim", type=int, default=32)
    parser.add_argument("--prior_depth", type=int, default=1)
    parser.add_argument("--prior_num_negatives", type=int, default=20)
    parser.add_argument("--prior_lr", type=float, default=1e-3)
    parser.add_argument("--prior_decay", type=float, default=0.0)
    parser.add_argument("--prior_weight", type=float, default=1.0)
    parser.add_argument("--prior_vector_norm", type=bool, default=False)

    try:
        return parser.parse_args()
    except: 
        return parser.parse_args([])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(device="none"):
    if device == "none":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else: 
            device = "cpu"
    return device
