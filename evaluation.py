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
                 criterion:torch.nn,
                 device:torch.device,
                 print_cost=True):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.print_cost = print_cost

    def test(self):
        model = self.model
        dataloader = self.dataloader
        criterion = self.criterion
        total_batch = len(dataloader)
        avg_cost = 0
        device = self.device
        with torch.no_grad():
            for idx,(user,item, bias_user, bias_item, o_avg, c_score, target) in enumerate(dataloader):
                user,item,target = user.to(device),item.to(device),target.to(device)
                pred = torch.flatten(model(user, item,bias_user,bias_item,o_avg,c_score),start_dim=1).to(device)
                cost = criterion(pred,target)
                if self.print_cost:
                    print(f" cost for test dataset at batch#{idx} : {cost}")
                avg_cost+=cost
            if self.print_cost:
                print(f"average cost for test dataset at {avg_cost/total_batch}")
        return avg_cost/total_batch
