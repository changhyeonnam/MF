import torch
import torch.nn as nn
import numpy as np

class MatrixFactorization(nn.Module):

    def __init__(self,
                 num_users:int,
                 num_items:int,
                 num_factors:int,
                 bias_uId:dict,
                 bias_mId:dict,
                 avg:int,
                 device,
                 confidence_score_dict) -> None:

        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.bias_uId = bias_uId
        self.bias_mId = bias_mId
        self.avg = avg
        self.device = device
        self.cs_dict = confidence_score_dict
        self.user_embedding = nn.Embedding(self.num_users,self.num_factors) # num_embeddings = batchsize * numuser embedding_dim = num_factors
        self.item_embedding = nn.Embedding(self.num_items,self.num_factors) # sparse = False
        torch.nn.init.ones_(self.user_embedding.weight)
        torch.nn.init.ones_(self.item_embedding.weight)

    def forward(self,users,items):
        result = torch.bmm(self.user_embedding(users),torch.transpose(self.item_embedding(items),1,2))
        bias_uId=np.array([])
        bias_mId=np.array([])
        for item in users:
            bias_uId=np.append(bias_uId,self.bias_uId[item.item()])
        bias_uId=bias_uId.reshape((-1,1,1))
        for item in items:
            bias_mId=np.append(bias_mId,self.bias_mId[item.item()])
        bias_mId=bias_mId.reshape((-1,1,1))
        bias = bias_uId+bias_mId+self.avg #broad casting
        bias = torch.FloatTensor(bias).to(self.device)
        result = result+bias

        confidence_score = np.array([])
        for item in users:
            confidence_score = np.append(confidence_score,self.cs_dict[item.item()])
        confidence_score=confidence_score.reshape((-1,1,1))
        confidence_score = torch.FloatTensor(confidence_score).to(self.device)
        result = torch.bmm(result,confidence_score)
        return result

    def __call__(self,*args):
        return self.forward(*args)

