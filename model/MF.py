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
                 device) -> None:

        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.bias_uId = bias_uId
        self.bias_mId = bias_mId
        self.avg = avg
        self.device = device
        self.user_embedding = nn.Embedding(self.num_users,self.num_factors) # num_embeddings = batchsize * numuser embedding_dim = num_factors
        self.item_embedding = nn.Embedding(self.num_items,self.num_factors) # sparse = False
        torch.nn.init.ones_(self.user_embedding.weight)
        torch.nn.init.ones_(self.item_embedding.weight)

    def forward(self,users,items):
        result = torch.bmm(self.user_embedding(users),torch.transpose(self.item_embedding(items),1,2))
        bias_uId_tensor=np.array([])
        bias_mId_tensor=np.array([])
        for item in users:
            bias_uId_tensor=np.append(bias_uId_tensor,self.bias_uId[item.item()])
        bias_uId_tensor=bias_uId_tensor.reshape((-1,1,1))
        for item in items:
            bias_mId_tensor=np.append(bias_mId_tensor,self.bias_mId[item.item()])
        bias_mId_tensor=bias_mId_tensor.reshape((-1,1,1))
        bias = bias_uId_tensor+bias_mId_tensor+self.avg #broad casting
        bias = torch.FloatTensor(bias).to(self.device)
        return result + bias

    def __call__(self,*args):
        return self.forward(*args)

