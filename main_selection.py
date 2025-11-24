#%%
import os
import sys
import subprocess
import torch
import argparse
import numpy as np
import scipy.sparse as sps
from datetime import datetime
from sklearn.metrics import roc_auc_score
from module.model import NCF, MF, LDR, LDRMF, SharedMF, SharedNCF
from module.metric import ndcg_func, recall_func, ap_func
from module.dataset import binarize, load_data, generate_total_sample
from module.utils import set_device, set_seed
from sklearn.model_selection import KFold
try:
    import wandb
except: 
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


#%%
parser = argparse.ArgumentParser()
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--dataset-name", type=str, default="ml-1m")
parser.add_argument("--num-epochs", type=int, default=500)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--weights-dir", type=str, default="./weights")
parser.add_argument("--base-model", type=str, default="ncf")
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--lambda1", type=float, default=1e-4)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


#%%
expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.random_seed)
args.device = set_device()
args.expt_name = f"{args.base_model}_{expt_num}"


#%%
x_train_cv = np.load(f"{args.data_dir}/{args.dataset_name}/train.npy")
x_test = np.load(f"{args.data_dir}/{args.dataset_name}/test.npy")
x_train_cv, y_train_cv = x_train_cv[:,:-1]-1, x_train_cv[:,-1]
x_test, y_test = x_test[:, :-1]-1, x_test[:,-1]
y_train_cv = binarize(y_train_cv)
y_test = binarize(y_test)


kf = KFold(n_splits=4, shuffle=True, random_state=args.random_seed)
for cv_num, (train_idx, valid_idx) in enumerate(kf.split(x_train_cv)):

    wandb_login = False
    try:
        wandb_login = wandb.login(key = open(f"{args.data_dir}/wandb_key.txt", 'r').readline())
    except:
        pass
    if wandb_login:
        configs = vars(args)
        wandb_var = wandb.init(project="ldr_rec", config=configs)
        wandb.run.name = args.expt_name


    x_train = x_train_cv[train_idx]
    y_train = y_train_cv[train_idx]
    x_valid = x_train_cv[valid_idx]
    y_valid = y_train_cv[valid_idx]

    num_samples = len(x_train)
    total_batch = num_samples // args.batch_size

    num_users = np.max([
        x_train[:,0].max()-x_train[:,0].min()+1,
        x_valid[:,0].max()-x_valid[:,0].min()+1,
        x_test[:,0].max()-x_test[:,0].min()+1,
        ])

    num_items = np.max([
        x_train[:,1].max()-x_train[:,1].min()+1,
        x_valid[:,1].max()-x_valid[:,1].min()+1,
        x_test[:,1].max()-x_test[:,1].min()+1,
        ])
    print(f"# user: {num_users}, # item: {num_items}")

    obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0], x_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    obs = binarize(obs, 1.)
    x_all = generate_total_sample(num_users, num_items)


    x_valid_tensor = torch.LongTensor(x_valid).to(args.device)
    x_test_tensor = torch.LongTensor(x_test).to(args.device)


    if args.base_model == "ncf":
        model = NCF(num_users, num_items, args.embedding_k, depth=args.depth)
    elif args.base_model == "mf":
        model = MF(num_users, num_items, args.embedding_k)
    elif args.base_model == "ldr":
        model = LDR(num_users, num_items, args.embedding_k, depth=args.depth)
    elif args.base_model == "ldr_w":
        model = LDR(num_users, num_items, args.embedding_k, depth=args.depth)
    elif args.base_model == "ldrmf":
        model = LDRMF(num_users, num_items, args.embedding_k, depth=args.depth)
    elif args.base_model == "shared_ncf":
        model = SharedNCF(num_users, num_items, args.embedding_k, depth=args.depth)
    elif args.base_model == "shared_mf":
        model = SharedMF(num_users, num_items, args.embedding_k, depth=args.depth)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fcn = torch.nn.BCELoss(reduction="mean")


    best_valid_auc = 0.
    for epoch in range(1, args.num_epochs+1):
        sample_idxs = np.arange(x_train.shape[0])
        np.random.shuffle(sample_idxs)

        all_idxs = np.arange(x_all.shape[0])
        np.random.shuffle(all_idxs)


        model.train()
        epoch_rec_loss = 0.
        epoch_margin_loss = 0.
        epoch_total_loss = 0.
        for idx in range(total_batch):
            selected_idx = sample_idxs[args.batch_size*idx : args.batch_size*(idx+1)]
            sub_x = x_train[selected_idx]
            sub_x = torch.LongTensor(sub_x).to(args.device)
            sub_y = y_train[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(args.device)

            pred, _, __ = model(sub_x)
            rec_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y)
            epoch_rec_loss += rec_loss


            selected_idx = all_idxs[args.batch_size*idx : args.batch_size*(idx+1)]
            sub_x = x_all[selected_idx]
            sub_x = torch.LongTensor(sub_x).to(args.device)
            sub_t = obs[selected_idx]
            sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(args.device)

            _, pred_t, _ = model(sub_x)
            select_loss = loss_fcn(torch.nn.Sigmoid()(pred_t), sub_t)
            epoch_margin_loss += select_loss


            total_loss = rec_loss + args.lambda1*select_loss
            epoch_total_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")


        best_epoch = 0
        if epoch % args.evaluate_interval == 0:
            model.eval()

            # valdation
            if args.base_model == "ldr":
                _, pred, __ = model(x_valid_tensor)
            elif args.base_model == "ldr_w":
                pred, _, __ = model(x_valid_tensor)
            else:
                pred, _, __ = model(x_valid_tensor)
            pred = pred.flatten().cpu().detach().numpy()

            ndcg_dict: dict = {}
            recall_dict: dict = {}
            ap_dict: dict = {}
            auc_dict = {}

            valid_ndcg_res = ndcg_func(pred, x_valid, y_valid, args.top_k_list)
            valid_recall_res = recall_func(pred, x_valid, y_valid, args.top_k_list)
            valid_ap_res = ap_func(pred, x_valid, y_valid, args.top_k_list)
            valid_auc = roc_auc_score(y_valid, pred)

            for top_k in args.top_k_list:
                ndcg_dict[f"valid_ndcg_{top_k}"] = np.mean(valid_ndcg_res[f"ndcg_{top_k}"])
                recall_dict[f"valid_recall_{top_k}"] = np.mean(valid_recall_res[f"recall_{top_k}"])
                ap_dict[f"valid_ap_{top_k}"] = np.mean(valid_ap_res[f"ap_{top_k}"])
            auc_dict["valid_auc"] = valid_auc


            # test
            if args.base_model == "ldr":
                _, pred, __ = model(x_test_tensor)
            elif args.base_model == "ldr_w":
                pred, _, __ = model(x_test_tensor)
            else:
                pred, _, __ = model(x_test_tensor)
            pred = pred.flatten().cpu().detach().numpy()

            test_ndcg_res = ndcg_func(pred, x_test, y_test, args.top_k_list)
            test_recall_res = recall_func(pred, x_test, y_test, args.top_k_list)
            test_ap_res = ap_func(pred, x_test, y_test, args.top_k_list)
            test_auc = roc_auc_score(y_test, pred)

            for top_k in args.top_k_list:
                ndcg_dict[f"test_ndcg_{top_k}"] = np.mean(test_ndcg_res[f"ndcg_{top_k}"])
                recall_dict[f"test_recall_{top_k}"] = np.mean(test_recall_res[f"recall_{top_k}"])
                ap_dict[f"test_ap_{top_k}"] = np.mean(test_ap_res[f"ap_{top_k}"])
            auc_dict["test_auc"] = test_auc

            if valid_auc > best_valid_auc:
                best_epoch = epoch
                best_valid_auc = valid_auc
                auc_dict["best_valid_auc"] = auc_dict["valid_auc"]
                auc_dict["best_test_auc"] = auc_dict["test_auc"]
                for top_k in args.top_k_list:
                    ndcg_dict[f"best_test_ndcg_{top_k}"] = ndcg_dict[f"test_ndcg_{top_k}"]
                    recall_dict[f"best_test_recall_{top_k}"] = recall_dict[f"test_recall_{top_k}"]
                    ap_dict[f"best_test_ap_{top_k}"] = ap_dict[f"test_ap_{top_k}"]


            print(f"NDCG: {ndcg_dict}")
            print(f"Recall: {recall_dict}")
            print(f"AP: {ap_dict}")
            print(f"AUC: {auc_dict}")


            if wandb_login:
                wandb_var.log(ndcg_dict)
                wandb_var.log(recall_dict)
                wandb_var.log(ap_dict)
                wandb_var.log(auc_dict)
                wandb_var.log({"cv_num":cv_num, "best_epoch":best_epoch})


    save_dir = f"{args.weights_dir}/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True) 
    torch.save(model.state_dict(), f"{save_dir}/{args.expt_name}.pth")

    if wandb_login:
        wandb.finish()
