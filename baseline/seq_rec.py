#%%
import os
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn.functional as F

from torch import optim
from datetime import datetime

from module.utils import parse_args, set_seed, set_device
from module.procedure import computeTopNAccuracy
from module.dataset import UserItemTime
from module.model import build_model, score_pair, score_all


#%%
"""
Residual-only density-ratio baselines trained with logistic loss.

Key design choices:
- No prior module.
- Uniform negative sampling distribution q(u) over users.
- Keep the training/evaluation skeleton close to the user's current code.
- Make model replacement easy by sharing the same interface:
    residual_score(item_idx, hist_item_idx, user_idx=None)
    score_all_items(hist_item_idx, user_idx=None)
"""

args = parse_args()
args.model_name = "mlp4rec"
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)


wandb_login = False
file_dir = inspect.getfile(inspect.currentframe())
file_name = file_dir.split("/")[-1]
if file_name.endswith(".py"):
    try:
        wandb_login = wandb.login(key=open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
    except Exception:
        wandb_login = False

if wandb_login:
    expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
    args.expt_name = f"{file_name.split('.')[-2]}_{args.model_name}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec2", config=vars(args))
    wandb.run.name = args.expt_name



#%%
dataset = UserItemTime(args)
dataset.build_user_histories(max_seq_len=args.max_seq_len)
dataset.get_pair_item_uniform(k=args.contrast_size-1)
hot_ratio = dataset.hotDataSize / dataset.trainDataSize

#%%
mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

model = build_model(args, dataset, mini_batch)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)




#%%

# import torch
# from torch import nn
from functools import partial
# from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
# from recbole.model.loss import BPRLoss

class ContextSeqEmbAbstractLayer(nn.Module):
    """For Deep Interest Network and feature-rich sequential recommender systems, return features embedding matrices."""

    def __init__(self):
        super(ContextSeqEmbAbstractLayer, self).__init__()
        self.token_field_offsets = {}
        self.token_embedding_table = nn.ModuleDict()
        self.float_embedding_table = nn.ModuleDict()
        self.token_seq_embedding_table = nn.ModuleDict()

        self.token_field_names = None
        self.token_field_dims = None
        self.float_field_names = None
        self.float_field_dims = None
        self.token_seq_field_names = None
        self.token_seq_field_dims = None
        self.num_feature_field = None

    def get_fields_name_dim(self):
        """get user feature field and item feature field.

        """
        self.token_field_names = {type: [] for type in self.types}
        self.token_field_dims = {type: [] for type in self.types}
        self.float_field_names = {type: [] for type in self.types}
        self.float_field_dims = {type: [] for type in self.types}
        self.token_seq_field_names = {type: [] for type in self.types}
        self.token_seq_field_dims = {type: [] for type in self.types}
        self.num_feature_field = {type: 0 for type in self.types}

        for type in self.types:
            for field_name in self.field_names[type]:
                if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.token_field_names[type].append(field_name)
                    self.token_field_dims[type].append(self.dataset.num(field_name))
                elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.token_seq_field_names[type].append(field_name)
                    self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                else:
                    self.float_field_names[type].append(field_name)
                    self.float_field_dims[type].append(self.dataset.num(field_name))
                self.num_feature_field[type] += 1

    def get_embedding(self):
        """get embedding of all features.

        """
        for type in self.types:
            if len(self.token_field_dims[type]) > 0:
                self.token_field_offsets[type] = np.array((0, *np.cumsum(self.token_field_dims[type])[:-1]),
                                                          dtype=np.long)
                self.token_embedding_table[type] = FMEmbedding(
                    self.token_field_dims[type], self.token_field_offsets[type], self.embedding_size
                ).to(self.device)
            if len(self.float_field_dims[type]) > 0:
                self.float_embedding_table[type] = nn.Embedding(
                    np.sum(self.float_field_dims[type], dtype=np.int32), self.embedding_size
                ).to(self.device)
            if len(self.token_seq_field_dims) > 0:
                self.token_seq_embedding_table[type] = nn.ModuleList()
                for token_seq_field_dim in self.token_seq_field_dims[type]:
                    self.token_seq_embedding_table[type].append(
                        nn.Embedding(token_seq_field_dim, self.embedding_size).to(self.device)
                    )

    def embed_float_fields(self, float_fields, type, embed=True):
        """Get the embedding of float fields.
        In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
        when the type is user, [batch_size, max_item_length] should be recognised as [batch_size]

        Args:
            float_fields(torch.Tensor): [batch_size, max_item_length, num_float_field]
            type(str): user or item
            embed(bool): embed or not

        Returns:
            torch.Tensor: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]

        """
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[-1]
        # [batch_size, max_item_length, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, max_item_length, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table[type](index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(-1))

        return float_embedding

    def embed_token_fields(self, token_fields, type):
        """Get the embedding of token fields

        Args:
            token_fields(torch.Tensor): input, [batch_size, max_item_length, num_token_field]
            type(str): user or item

        Returns:
            torch.Tensor: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]

        """
        if token_fields is None:
            return None
        # [batch_size, max_item_length, num_token_field, embed_dim]
        if type == 'item':
            embedding_shape = token_fields.shape + (-1,)
            token_fields = token_fields.reshape(-1, token_fields.shape[-1])
            token_embedding = self.token_embedding_table[type](token_fields)
            token_embedding = token_embedding.view(embedding_shape)
        else:
            token_embedding = self.token_embedding_table[type](token_fields)
        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, type):
        """Get the embedding of token_seq fields.

        Args:
            token_seq_fields(torch.Tensor): input, [batch_size, max_item_length, seq_len]`
            type(str): user or item
            mode(str): mean/max/sum

        Returns:
            torch.Tensor: result [batch_size, max_item_length, num_token_seq_field, embed_dim]

        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[type][i]
            mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)  # [batch_size, max_item_length, 1]
            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, max_item_length, seq_len, embed_dim]
            mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
            if self.pooling_mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9
                result = torch.max(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
                result = result.values
            elif self.pooling_mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=-2)  # [batch_size, max_item_length, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, max_item_length, embed_dim]
                result = result.unsqueeze(-2)  # [batch_size, max_item_length, 1, embed_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=-2)  # [batch_size, max_item_length, num_token_seq_field, embed_dim]

    def embed_input_fields(self, user_idx, item_idx):
        """Get the embedding of user_idx and item_idx

        Args:
            user_idx(torch.Tensor): interaction['user_id']
            item_idx(torch.Tensor): interaction['item_id_list']

        Returns:
            dict: embedding of user feature and item feature

        """
        user_item_feat = {'user': self.user_feat, 'item': self.item_feat}
        user_item_idx = {'user': user_idx, 'item': item_idx}
        float_fields_embedding = {}
        token_fields_embedding = {}
        token_seq_fields_embedding = {}
        sparse_embedding = {}
        dense_embedding = {}

        for type in self.types:
            float_fields = []
            for field_name in self.float_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_fields.append(feature if len(feature.shape) == (2 + (type == 'item')) else feature.unsqueeze(-1))
            if len(float_fields) > 0:
                float_fields = torch.cat(float_fields, dim=-1)  # [batch_size, max_item_length, num_float_field]
            else:
                float_fields = None
            # [batch_size, max_item_length, num_float_field]
            # or [batch_size, max_item_length, num_float_field, embed_dim] or None
            float_fields_embedding[type] = self.embed_float_fields(float_fields, type)

            token_fields = []
            for field_name in self.token_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_fields.append(feature.unsqueeze(-1))
            if len(token_fields) > 0:
                token_fields = torch.cat(token_fields, dim=-1)  # [batch_size, max_item_length, num_token_field]
            else:
                token_fields = None
            # [batch_size, max_item_length, num_token_field, embed_dim] or None
            token_fields_embedding[type] = self.embed_token_fields(token_fields, type)

            token_seq_fields = []
            for field_name in self.token_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            token_seq_fields_embedding[type] = self.embed_token_seq_fields(token_seq_fields, type)

            if token_fields_embedding[type] is None:
                sparse_embedding[type] = token_seq_fields_embedding[type]
            else:
                if token_seq_fields_embedding[type] is None:
                    sparse_embedding[type] = token_fields_embedding[type]
                else:
                    sparse_embedding[type] = torch.cat([token_fields_embedding[type], token_seq_fields_embedding[type]],
                                                       dim=-2)
            dense_embedding[type] = float_fields_embedding[type]

        # sparse_embedding[type]
        # shape: [batch_size, max_item_length, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding[type]
        # shape: [batch_size, max_item_length, num_float_field]
        #     or [batch_size, max_item_length, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding

    def forward(self, user_idx, item_idx):
        return self.embed_input_fields(user_idx, item_idx)



class FeatureSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(self, dataset, embedding_size, selected_features, pooling_mode, device):
        super(FeatureSeqEmbLayer, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = None
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {'item': selected_features}

        self.types = ['item']
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ['mean', 'max', 'sum']
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class PreNormResidual(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x.clone())) + x
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = torch.nn.Linear):
    return torch.nn.Sequential(
        dense(dim, dim * expansion_factor),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        torch.nn.Dropout(dropout)
    )


class SequentialRecommender(torch.nn.Module):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config['device']

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

class MLP4Rec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(MLP4Rec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        expansion_factor = 4
        chan_first = partial(torch.nn.Conv1d, kernel_size = 1)
        chan_last = torch.nn.Linear
        self.num_feature_field = len(config['selected_features'])
        self.layerSize = self.num_feature_field + 1

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = torch.nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )

        self.sequenceMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.featureMixer = PreNormResidual(self.hidden_size, FeedForward(self.layerSize, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.layers = torch.nn.ModuleList([])
        for i in range(self.num_feature_field+1):
            self.layers.append(self.sequenceMixer)
            self.layers.append(self.channelMixer)
        self.LayerNorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = torch.nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        if sparse_embedding is not None:
            feature_embeddings = sparse_embedding
        if dense_embedding is not None:
            if sparse_embedding is not None:
                feature_embeddings = torch.cat((sparse_embedding,dense_embedding),2)
            else:
                feature_embeddings = dense_embedding
        item_emb = torch.unsqueeze(item_emb,2)
        item_emb = torch.cat((item_emb,feature_embeddings),2)
        mixer_outputs = torch.split(item_emb,[1]*(self.num_feature_field+1),2)
        mixer_outputs = torch.stack(mixer_outputs,0)
        mixer_outputs = torch.squeeze(mixer_outputs)
        for _ in range(self.n_layers):
            for x in range(self.num_feature_field+1):
                mixer_outputs[x] = self.layers[x*2](mixer_outputs[x])
                mixer_outputs[x] = self.layers[(x*2)+1](mixer_outputs[x])
            mixer_outputs = torch.movedim(mixer_outputs,0,2)
            batch_size = mixer_outputs.size()[0]
            mixer_outputs = torch.flatten(mixer_outputs,0,1)
            mixer_outputs = self.featureMixer(mixer_outputs)
            mixer_outputs = torch.reshape(mixer_outputs,(batch_size,self.max_seq_length,self.layerSize,self.hidden_size))
            mixer_outputs = torch.movedim(mixer_outputs,2,0)

        output = self.gather_indexes(mixer_outputs[0], item_seq_len - 1)
        output = self.LayerNorm(output)
        return output


#%%
best_valid_score = 0.0
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0
cnt = 1
for epoch in range(1, args.epochs + 1):
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(hot_idxs)
    epoch_user_loss = 0.0

    for idx in range(batch_num):
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]
        anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        pos_score = score_pair(model, pos_item, anchor_hist_items, anchor_user)
        neg_score = score_pair(model, neg_item, anchor_hist_items, anchor_user)

        user_loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).sum()
        optimizer.zero_grad()
        user_loss.backward()
        optimizer.step()

        epoch_user_loss += user_loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] ldr: {epoch_user_loss / batch_num:.4f}")

    if epoch % args.pair_reset_interval == 0:
        print("Reset uniform negative users")
        dataset.get_pair_item_uniform(k=args.contrast_size-1)

    if epoch % args.evaluate_interval == 0:
        pred_list = []
        gt_list = []

        model.eval()
        for (user, item), pos_time_val in dataset.valid_user_item_time.items():
            hist_item_np = dataset.get_histories_for_users_at_times([user], [pos_time_val], max_seq_len=args.max_seq_len)
            hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
            user_t = torch.tensor([user], dtype=torch.long, device=args.device)

            with torch.no_grad():
                pred = score_all(model, hist_item_t, user_t).squeeze(0).cpu()

            exclude_items = list(dataset._allPos[user])
            pred[exclude_items] = -9999
            _, pred_k = torch.topk(pred, k=max(args.topks))
            pred_list.append(pred_k.cpu())
            gt_list.append([item])

        valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

        if wandb_login:
            wandb_var.log({
                "train_ldr": epoch_user_loss / batch_num,
            })
            wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
            wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
            wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
            wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))

        current_valid_score = valid_results[1][0]
        if current_valid_score - best_valid_score <= 0.0:
            cnt += 1
        else:
            best_valid_score = current_valid_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            cnt = 1

        if cnt == 5:
            break

pred_list = []
gt_list = []

best_model = build_model(args, dataset, mini_batch)
best_model.load_state_dict(best_state)
best_model.eval()

for (user, item), pos_time_val in dataset.test_user_item_time.items():
    hist_item_np = dataset.get_histories_for_users_at_times([user], [pos_time_val], max_seq_len=args.max_seq_len)
    hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
    user_t = torch.tensor([user], dtype=torch.long, device=args.device)

    with torch.no_grad():
        pred = score_all(best_model, hist_item_t, user_t).squeeze(0).cpu()

    exclude_items = list(dataset._allPos[user])
    pred[exclude_items] = -9999
    _, pred_k = torch.topk(pred, k=max(args.topks))
    pred_list.append(pred_k.cpu())
    gt_list.append([item])

test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

if wandb_login:
    wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
    wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
    wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
    wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))
    wandb_var.log({"best_valid_score": best_valid_score})
    wandb_var.log({"best_epoch": best_epoch})
    wandb_var.finish()

# %%
