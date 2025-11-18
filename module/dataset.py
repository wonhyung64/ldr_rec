import os
import numpy as np
import pandas as pd


def load_data(data_dir:str, dataset_name:str):
    """
    Load dataset.
    """
    dataset_dir = os.path.join(data_dir, dataset_name)

    if dataset_name == "yahoo_r3":
        train_file = os.path.join(dataset_dir, "ydata-ymusic-rating-study-v1_0-train.txt")
        test_file = os.path.join(dataset_dir, "ydata-ymusic-rating-study-v1_0-test.txt")
        x_train = []
        with open(train_file, "r") as f:
            for line in f:
                x_train.append(line.strip().split())
        x_train = np.array(x_train).astype(int)
        x_test = []
        with open(test_file, "r") as f:
            for line in f:
                x_test.append(line.strip().split())
        x_test = np.array(x_test).astype(int)

    elif dataset_name == "coat":
        train_file = os.path.join(dataset_dir, "train.csv")
        test_file = os.path.join(dataset_dir, "test.csv")
        x_train = pd.read_csv(train_file).to_numpy()
        x_train = np.stack([x_train[:,0]+1, x_train[:,1]+1, x_train[:,2]], axis=-1)
        x_test = pd.read_csv(test_file).to_numpy()
        x_test = np.stack([x_test[:,0]+1, x_test[:,1]+1, x_test[:,2]], axis=-1)

    elif dataset_name == "KuaiRec":
        train_file = os.path.join(dataset_dir, "data/big_matrix.csv")
        test_file = os.path.join(dataset_dir, "data/small_matrix.csv")
        x_train = pd.read_csv(train_file)
        x_train["interaction"] = x_train["watch_ratio"].map(lambda x: 1 if x >= 2. else 0)
        x_train = x_train[["user_id", "video_id", "interaction"]].to_numpy()
        x_train = np.stack([x_train[:,0]+1, x_train[:,1]+1, x_train[:,2]], axis=-1)
        x_test = pd.read_csv(test_file)
        x_test["interaction"] = x_test["watch_ratio"].map(lambda x: 1 if x >= 2. else 0)
        x_test = x_test[["user_id", "video_id", "interaction"]].to_numpy()
        x_test = np.stack([x_test[:,0]+1, x_test[:,1]+1, x_test[:,2]], axis=-1)

    print(f"Loaded from {dataset_name} dataset")
    print("[train] num data:", x_train.shape[0])
    print("[test]  num data:", x_test.shape[0])
    print("user, item indices start from '1'.")

    return x_train, x_test


def binarize(y, thres:int=3):
    """
    Binarize relevance score.
    """
    y[y< thres] = 0
    y[y>=thres] = 1

    return y


def generate_total_sample(num_user:int, num_item:int):
    """
    Generate indices of all user-item pairs.
    """
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])

    return np.array(sample)
