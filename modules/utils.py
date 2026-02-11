import torch
import random
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--batch_size', type=int,default=32384)
    parser.add_argument('--recdim', type=int,default=64,)
    parser.add_argument('--lr', type=float,default=0.001,)
    parser.add_argument('--decay', type=float,default=0.,)
    parser.add_argument('--lambda1', type=float,default=1.,)
    parser.add_argument('--data_path', type=str, default='./data',
                        help='the path to dataset')
    parser.add_argument('--cred_path', type=str, default='./assets',
                        help='the path to credential')
    parser.add_argument('--weights_path', type=str, default='./weights',
                        help='the path to credential')
    parser.add_argument('--dataset', type=str,default='micro_video',
                        help="available datasets: ['micro_video', 'kuai', 'amazon_book']")
    parser.add_argument('--topks', type=list, default=[10, 20, 50, 100])
    parser.add_argument('--epochs', type=int,default=600)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--period', type=int, default=8, help='number of stage')
    parser.add_argument('--n_pop_group', type=int, default=20, help='number of popularity groups')
    parser.add_argument('--ablation-model', type=str, default='pv') # [pv, puv, ptv, ptv_pv, puv_pv]
    parser.add_argument('--time-type', type=str, default='continuous') # [pv, puv, ptv, ptv_pv, puv_pv]
    parser.add_argument('--contrast-size', type=int, default=16)
    parser.add_argument('--evaluate-interval', type=int, default=20)

    try:
        return parser.parse_args()
    except:
        return parser.parse_args([])


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_device(device="none"):
    if device == "none":
        if torch.cuda.is_available():
            device = "cuda:1"
        elif torch.backends.mps.is_available():
            device = "mps"
        else: 
            device = "cpu"

    return device
