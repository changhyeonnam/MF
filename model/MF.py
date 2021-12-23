import torch
import torch.nn as nn

class MatrixFactorization(nn.Module): #

    def __init__(self,num_users:int,num_items:int,num_factors:int) -> None:
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors

        self.user_embedding = nn.Embedding(self.num_users,self.num_factors) # sparse = False
        self.item_embedding = nn.Embedding(self.num_items,self.num_factors) # sparse = False

    def forward(self,users,items):
        return torch.bmm(self.user_embedding(users),torch.transpose(self.item_embedding(items),1,2))

    def __call__(self,*args):
        return self.forward(*args)

    def predcit(self,users,items):
        return self.forward(users,items)