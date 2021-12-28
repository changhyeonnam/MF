# code for finding best number of factor for matrix factorization
import argparse
import torch
import torch.optim as optim
from utils import MovieLens
from torch.utils.data import DataLoader
from model.MF import MatrixFactorization
from train import Train
from evaluation import RMSELoss
from evaluation import Test
import matplotlib.pyplot as plt
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

parser = argparse.ArgumentParser(description="Matrix Factorization with movieLens")
parser.add_argument('-e', '--epochs', default=1, type=int)
parser.add_argument('-f', '--factor', default=30, type=int)  # number of factor for MF
parser.add_argument('-b', '--batch', default=32, type=int)
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float)
parser.add_argument('-s', '--size', default='small', type=str)
parser.add_argument('-d', '--download', default=False, type=bool)
parser.add_argument('-bi', '--bias', default=True, type=bool)
parser.add_argument('-c', '--confidence', default=True, type=bool)

args = parser.parse_args()

root_path = "dataset"
train_set = MovieLens(root=root_path, file_size=args.size, train=True, download=args.download)
test_set = MovieLens(root=root_path, file_size=args.size, train=False, download=False)

train_num_users, train_num_items = train_set.get_numberof_users_items()

bias_uId, bias_mId, overall_avg = train_set.get_bias()
print("Bias loaded!")

user_count_dict, movie_count_dict = train_set.uId_dict, train_set.mId_dict
confidence_score = train_set.normalize(user_count_dict, target=1.0)
print("Confidence loaded!")

dataloader = DataLoader(
    dataset=train_set,
    batch_size=args.batch,
    shuffle=True,
)
dataloader_test = DataLoader(
    dataset=test_set,
    batch_size=args.batch,
    shuffle=False,
)





if __name__ == "__main__":
    result=[]
    for num_factor in range(1,args.factor+1):
        model = MatrixFactorization(num_users=train_num_users * args.batch,
                                    num_items=train_num_items * args.batch,
                                    num_factors=num_factor,
                                    bias_uId=bias_uId,
                                    bias_mId=bias_mId,
                                    avg=overall_avg,
                                    device=device,
                                    confidence_score_dict=confidence_score,
                                    bias_select=args.bias,
                                    confidence_select=args.confidence
                                    ).to(device)
        optimizer = optim.Adam(model.parameters(),lr=arg.lr)
        criterion = RMSELoss()
        train = Train(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      epochs=args.epochs,
                      dataloader=dataloader,
                      device=device,
                      print_cost=False)
        test = Test(model=model,
                    criterion=criterion,
                    dataloader=dataloader_test,
                    device=device,
                    print_cost=False)
        costs = train.train()
        cost_test = test.test()
        print(f"num_factor value : {num_factor}, test_average_rmse : {cost_test}")
        result.append(cost_test)

    plt.plot(range(1, args.factor + 1), result)
    plt.xlabel('number of factor')
    plt.ylabel('RMSE')
    fig_file = "loss_curve_factor.png"
    if os.path.isfile(fig_file):
        os.remove(fig_file)
    plt.savefig(fig_file)


