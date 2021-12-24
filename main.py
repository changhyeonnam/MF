import argparse
import torch
import torch.optim as optim
from utils import MovieLens
from torch.utils.data import DataLoader
from model.MF import MatrixFactorization
from train import Train
from evaluation import RMSELoss
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
print('device:',device)

parser = argparse.ArgumentParser(description="Matrix Factorization with movieLens")
parser.add_argument('-e','--epochs',default=1,type=int)
parser.add_argument('-b','--batch',default=32,type=int)
parser.add_argument('--lr','--learning_rate',default=1e-3,type=int)
args = parser.parse_args()

root_path = "dataset"
train_set = MovieLens(root=root_path,file_size='small',train=True,download=False)
test_set = MovieLens(root=root_path,file_size='small',train=False,download=False)
train_num_users, train_num_items = train_set.get_numberof_users_items()

dataloader = DataLoader(
    dataset= train_set,
    batch_size= args.batch,
    shuffle = True,
)
model=MatrixFactorization(num_users=train_num_users*args.batch,
                          num_items=train_num_items*args.batch,
                          num_factors=100).to(device)
optimizer = optim.Adam(model.parameters(),lr=args.lr)
criterion = RMSELoss()

if __name__=="__main__":
    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  epochs=args.epochs,
                  dataloader=dataloader,
                  device=device)
    epochs,costs= train.train()
    plt.plot(epochs,costs)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.show()


