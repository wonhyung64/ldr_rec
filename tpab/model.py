import torch

from torch import nn
import torch.nn.functional as F


class MF_TPAB(nn.Module):
	def __init__(self, latent_dim, num_users, num_items, num_item_pop, pthres):
		super(MF_TPAB, self).__init__()
		self.latent_dim = latent_dim
		self.num_users  = num_users
		self.num_items  = num_items
		self.num_item_pop  = num_item_pop
		self.pthres = pthres
		self.__init_weight()
        
	def __init_weight(self):
		self.embedding_user = torch.nn.Embedding(
    		num_embeddings=self.num_users, embedding_dim=int(self.latent_dim))
		self.embedding_item = torch.nn.Embedding(
    		num_embeddings=self.num_items, embedding_dim=int(self.latent_dim))
		self.embedding_user_pop = torch.nn.Embedding(
    		num_embeddings=self.num_users, embedding_dim=int(self.latent_dim))
		self.embedding_item_pop = torch.nn.Embedding(
    		num_embeddings=self.num_item_pop, embedding_dim=int(self.latent_dim))
		nn.init.normal_(self.embedding_user.weight, std=0.1)
		nn.init.normal_(self.embedding_item.weight, std=0.1)
		nn.init.normal_(self.embedding_user_pop.weight, std=0.1)
		nn.init.normal_(self.embedding_item_pop.weight, std=0.1)

	def forward(self, batch_users, batch_pos, batch_neg, batch_pos_inter, batch_neg_inter, batch_stage, period):
		"""preference"""
		users_emb = self.embedding_user(batch_users.long())
		pos_emb   = self.embedding_item(batch_pos.long())
		neg_emb   = self.embedding_item(batch_neg.long())
		"""conformity"""
		users_pop_emb = self.embedding_user_pop(batch_users.long())
		pos_pred_pop = (batch_pos_inter.T * F.one_hot(batch_stage, num_classes=period+2)).sum(1)
		neg_pred_pop = (batch_neg_inter.T * F.one_hot(batch_stage, num_classes=period+2)).sum(1)
		pos_pop_emb = self.get_pop_embeddings(pos_pred_pop)
		neg_pop_emb = self.get_pop_embeddings(neg_pred_pop)
		return users_emb, pos_emb, neg_emb, users_pop_emb, pos_pop_emb, neg_pop_emb

    # Coarsening
	def get_pop_embeddings(self, pred_pop):
		num_pop = self.num_item_pop
		pthres = self.pthres
        # using popularity as categorical input
		reindexed_list = []
		for pop in pred_pop.tolist():
			for i in range(num_pop):
				if pop <= pthres[i]:
					pop_idx = i
					break
				# if pop is greater then pmax, then pop_idx is the last index
				elif i == num_pop-1:
					pop_idx = i
			reindexed_list.append(pop_idx)
		pop_emb = self.embedding_item_pop(torch.tensor(reindexed_list).to(pred_pop.device).long())
		return pop_emb

    # Inference
	def predict(self, users, predictor, stage, item_pop):
		users = users.long()
		pred_pop = predictor(stage, item_pop.T)
		users_emb = self.embedding_user(users)
		items_emb = self.embedding_item.weight
		users_pop_emb = self.embedding_user_pop(users)
		items_pop_emb = self.get_pop_embeddings(pred_pop)
		ratings = torch.matmul(users_emb, items_emb.t())
		ratings_pop = torch.matmul(users_pop_emb, items_pop_emb.t())
		ratings = ratings + ratings_pop
		return ratings


class PopPredictor(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.a = args.predict
		self.period = args.period

	def forward(self, stage, item):
		# item: N * 10
		# stage: N * 1
		item = item.T
		c_0 = item[:, 0].clone()
		c_1 = item[:, 1].clone()
		x_1 = c_0 - (c_1 - c_0) / self.a
		x_0 = x_1 - (c_0 - x_1) / self.a
		new_items = torch.cat([x_0.reshape(1, -1), x_1.reshape(1, -1), item.T]).T
		new_stages = stage + 2
		return (self.a * ((new_items * F.one_hot(new_stages-1, num_classes=self.period+4)).sum(1) 
        		- (new_items * F.one_hot(new_stages-2, num_classes=self.period+4)).sum(1))
        		+ (new_items * F.one_hot(new_stages-1, num_classes=self.period+4)).sum(1))
