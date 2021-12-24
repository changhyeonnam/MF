import argparse
import torch
import torch.optim as optim
from utils import MovieLens
from torch.utils.data import DataLoader
from model.MF import MatrixFactorization
from train import Train
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
print('device:',device)

parser = argparse.ArgumentParser(description="Matrix Factorization with movieLens")
parser.add_argument('-e','--epochs',default=1,type=int)
parser.add_argument('-b','--batch',default=32,type=int)
parser.add_argument('--lr','--learning_rate',default=1e-5,type=int)
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
model=MatrixFactorization(num_users=train_num_users,
                          num_items=train_num_items,
                          num_factors=100).to(device)
optimizer = optim.Adam(model.parameters(),lr=args.lr)
criterion = torch.nn.MSELoss().to(device)

if __name__=="__main__":
    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  epochs=args.epochs,
                  dataloader=dataloader,
                  device=device)
    train.train()


