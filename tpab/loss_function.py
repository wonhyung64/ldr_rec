import torch


def bpr_loss_fn(pos_scores, neg_scores):
	return torch.negative(torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10)).mean()


def reg_loss_fn(embs: list):
	return torch.tensor([emb.norm(2).pow(2)/2 for emb in embs]).sum()


def bootstrap_loss_fn(users_emb, pos_emb, neg_emb, users_pop_emb, pos_pop_emb, neg_pop_emb, batch_size):
	users_ori = torch.cat([users_pop_emb, users_emb], dim=1)
	random_order = torch.randperm(len(users_emb))
	pos_pop_new = pos_pop_emb[random_order]
	neg_pop_new = neg_pop_emb[random_order]
	pos_new = torch.cat([pos_pop_new, pos_emb], dim=1)
	neg_new = torch.cat([neg_pop_new, neg_emb], dim=1)

	pos_scores = score_fn(users_ori, pos_new)
	neg_scores = score_fn(users_ori, neg_new)
	boots_bpr_loss = bpr_loss_fn(pos_scores, neg_scores)

	return boots_bpr_loss


def score_fn(user_emb, item_emb):
	return torch.sum(user_emb * item_emb, dim=1)
