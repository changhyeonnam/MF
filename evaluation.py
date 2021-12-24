import torch 
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,pred,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(pred, y))
        return loss

    def __call__(self,*args):
        return self.forward(*args)

class Test():
    def __init__(self,model:torch.nn.Module,
                 dataloader:torch.utils.data.dataloader,
                 criterion:torch.nn,):
        self.model = model
        self.dataloader = dataloader