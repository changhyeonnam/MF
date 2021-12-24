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
        self.criterion = criterion
    def test(self):
        model = self.model
        dataloader = self.dataloader
        criterion = self.criterion
        total_batch = len(dataloader)
        avg_cost = 0
        with torch.no_grad:
            for user,item,target in dataloader:
                pred = torch.flatten(model(user,item),start_dim=1)
                cost = criterion(pred,target)
                avg_cost = argmax(avg_cost,cost/total_batch)

        return avg_cost

