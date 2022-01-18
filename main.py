import argparse
import torch
import torch.optim as optim
from utils import MovieLens
from utils import Download
from torch.utils.data import DataLoader
from model.MF import MatrixFactorization
from train import Train
from evaluation import RMSELoss
from evaluation import Test
import matplotlib.pyplot as plt
import os
import time


# print device info
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
print('device:',device)

# print gpu info
if device == 'cuda':
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


# parser setting
parser = argparse.ArgumentParser(description="Matrix Factorization with movieLens")
parser.add_argument('-e','--epoch',type=int,default=1,help="Number of epochs")
parser.add_argument('-b','--batch',type=int,default=32,help="Batch size")
parser.add_argument('-f','--factor',type=int,default=8,help='choose number of predictive factor')
parser.add_argument('-lr', '--lr', default=1e-3, type=float,help='learning rate for optimizer')
parser.add_argument('-fi', '--file_size', default='small', type=str)
parser.add_argument('-bi','--use_bias',default='False',type=str)
parser.add_argument('-cs','--use_confidence',default='False',type=str)
parser.add_argument('-down','--download',default='True',type=str)
args = parser.parse_args()

# can't parse boolean type. need to change string to boolean type
use_bias = True if args.use_bias=='True' else False
use_cs = True if args.use_confidence=='True' else False
use_downlaod = True if args.download=='True' else False

root_path = "dataset"

# load dataframe
dataset = Download(root=root_path,file_size=args.file_size,download=use_downlaod)
total_dataframe, train_dataframe, test_dataframe = dataset.load_data()

# make torch.utils.data.Data object
train_set = MovieLens(df=train_dataframe,total_df=total_dataframe)
test_set = MovieLens(df=test_dataframe,total_df=total_dataframe)

# get number of unique userID, unique  movieID
train_num_users, train_num_items = total_dataframe['userId'].max()+1, total_dataframe['movieId'].max()+1

# dataloader for train, test
dataloader_train = DataLoader(
    dataset= train_set,
    batch_size= args.batch,
    shuffle = True,
)
dataloader_test = DataLoader(
    dataset= test_set,
    batch_size=args.batch,
    shuffle=False,
)

# model for MF
model=MatrixFactorization(num_users=train_num_users,
                          num_items=train_num_items,
                          num_factors=args.factor,
                          device=device,
                          use_bias=use_bias,
                          use_confidence=use_cs,
                          )

# for parallel GPU
#if torch.cuda.device_count() >1:
#    print("Multi gpu", torch.cuda.device_count())
#    model = torch.nn.DataParallel(model)

model.to(device)
optimizer = optim.Adam(model.parameters(),lr=args.lr)
criterion = RMSELoss()

if __name__=="__main__":
    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  epochs=args.epoch,
                  dataloader=dataloader_train,
                  device=device,
                  print_cost=True)

    test = Test(model=model,
                criterion=criterion,
                dataloader=dataloader_test,
                device = device,
                print_cost=True)

    costs= train.train()
    plt.plot(range(0,args.epoch),costs)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    now = time.localtime()
    fig_file = f"loss_curve_epochs.png"
    if os.path.isfile(fig_file):
        os.remove(fig_file)
    plt.savefig(fig_file)
    test.test()
    


