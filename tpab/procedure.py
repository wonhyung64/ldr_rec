import math
import torch
from utils import minibatch


def evaluate(args, flag, dataset, item_pop, model, predictor):
	if flag == "valid":
		testDict = dataset.valid_dict
		stage = torch.full((dataset.m_item, 1), args.period)
	elif flag == "test":
		testDict = dataset.test_dict
		stage = torch.full((dataset.m_item, 1), args.period+1)
	stage = torch.squeeze(stage).to(args.device)

	model = model.eval()
	true_list, pred_list = [], []
	for batch_users in minibatch(list(testDict.keys()), batch_size=args.batch_size):
		"""True Rating"""
		true = [testDict[u] for u in batch_users] # test positive item indices
		true_list.extend(true)

		"""Pred Rating"""
		allPos = dataset.getUserPosItems(batch_users) # train positive item indices
		batch_users_gpu = torch.Tensor(batch_users).long()
		batch_users_gpu = batch_users_gpu.to(args.device)
		with torch.no_grad():
			pred = model.predict(batch_users_gpu, predictor, stage, item_pop)

		"""Filtering item history indices"""
		exclude_index = []
		exclude_items = []
		for user_idx, items in enumerate(allPos):
			exclude_index.extend([user_idx] * len(items))
			exclude_items.extend(items)
		if flag == "test":
			valid_items = dataset.getUserValidItems(batch_users) # exclude validation items
			for user_idx, items in enumerate(valid_items):
				exclude_index.extend([user_idx] * len(items))
				exclude_items.extend(items)
		pred[exclude_index, exclude_items] = -(9999)
		_, pred_k = torch.topk(pred, k=max(args.topks))
		pred_list.extend(pred_k.cpu())

	return true_list, pred_list

        
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                cnt += 1

        precision.append(round(sumForPrecision / cnt, 4))
        recall.append(round(sumForRecall / cnt, 4))
        NDCG.append(round(sumForNdcg / cnt, 4))
        MRR.append(round(sumForMRR / cnt, 4))
        
    return precision, recall, NDCG, MRR

