import torch
class Train():
    def __init__(self,model:torch.nn.Module
                 ,optimizer:torch.optim,
                 epochs:int,
                 dataloader:torch.utils.data.dataloader,
                 criterion:torch.nn,
                 device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def train(self):
        model = self.model
        optimizer = self.optimizer
        total_epochs = self.epochs
        dataloader = self.dataloader
        criterion = self.criterion
        total_batch = len(dataloader)
        loss = []
        for epochs in range(0,total_epochs):
            avg_cost = 0
            for idx,(user,item,target) in enumerate(dataloader):
                user, item, target = user.to(self.device), item.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                pred = torch.flatten(model(user, item),start_dim=1).to(self.device)

                cost = criterion(pred,target)
                cost.backward()
                optimizer.step()
                avg_cost += cost / total_batch
            print('Epoch:', '%04d' % (epochs + 1), 'cost =', '{:.9f}'.format(avg_cost))
            loss.append(avg_cost)
        print('Learning finished')
        return loss
