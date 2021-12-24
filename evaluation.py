<<<<<<< HEAD
import torch 
import torch.nn as nn

class RMSELoss():
    def __init__(self):
        super(RMSELoss,self).__init__()
    
    def forward(self,pred,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(pred,y))
        return loss
    def __call__(self,*args):
=======
import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,pred,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(pred, y))
        return loss

    def __call(self,*args):
>>>>>>> 8c5960360070bc4719cf06b6a3f2dacd8a8ee16a
        return self.forward(*args)
