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

os.environ['KMP_DUPLICATE_LIB_OK']=''


device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
print('device:',device)

print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

parser = argparse.ArgumentParser(description="Matrix Factorization with movieLens")
parser.add_argument('-e', '--epochs', default=1, type=int)
parser.add_argument('-f', '--factor', default=30, type=int)  # number of factor for MF
parser.add_argument('-b', '--batch', default=32, type=int)
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float)
parser.add_argument('-s', '--size', default='small', type=str)
parser.add_argument('-bi','--bias',default='False',type=str)
parser.add_argument('-cs','--confidence',default='False',type=str)

args = parser.parse_args()
if args.bias == 'True':
    use_bias=True
else:
    use_bias=False

if args.confidence=='True':
    use_cs=True
else:
    use_cs=False



root_path = "dataset"
train_set = MovieLens(root=root_path,file_size=args.size,train=True,download=True)
test_set = MovieLens(root=root_path,file_size=args.size,train=False,download=False)

train_num_users, train_num_items = train_set.get_numberof_users_items()

bias_uId,bias_mId,overall_avg = train_set.get_bias()
print("Bias loaded!")

user_count_dict,movie_count_dict =  train_set.uId_dict,train_set.mId_dict
confidence_score = train_set.normalize(user_count_dict,target=1.0)
print("Confidence loaded!")

dataloader = DataLoader(
    dataset= train_set,
    batch_size= args.batch,
    shuffle = True,
)
dataloader_test = DataLoader(
    dataset= test_set,
    batch_size=args.batch,
    shuffle=False,
)

model=MatrixFactorization(num_users=train_num_users*args.batch,
                          num_items=train_num_items*args.batch,
                          num_factors=args.factor,
                          bias_uId=bias_uId,
                          bias_mId=bias_mId,
                          avg=overall_avg,
                          device=device,
                          confidence_score_dict=confidence_score,
                          bias_select=use_bias,
                          confidence_select=use_cs,
                          )
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
                  epochs=args.epochs,
                  dataloader=dataloader,
                  device=device,
                  print_cost=True)
    test = Test(model=model,
                criterion=criterion,
                dataloader=dataloader_test,
                device = device,
                print_cost=True)
    costs= train.train()
    plt.plot(range(0,args.epochs),costs)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    now = time.localtime()
    fig_file = f"loss_curve_epochs_{args.epochs}_batch_{args.batch}_size_{args.size}_lr_{args.lr}_factor_{args.factor}.png"
    if os.path.isfile(fig_file):
        os.remove(fig_file)
    plt.savefig(fig_file)
    test.test()
    


