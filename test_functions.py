import zipfile

from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import pandas  as pd
import numpy as np
from zipfile import ZipFile
import pathlib
import requests
import pandas as pd
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
def download_movielens(root:str) -> None:
    url = ("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
    req = requests.get(url, stream=True)
    print('Downloading MovieLens dataset')
    if not os.path.exists(root):
        os.makedirs(root)
    with open(os.path.join(root,'ml-latest-small.zip'),mode='wb') as fd:
        for chunk in req.iter_content(chunk_size=None):
            fd.write(chunk)
    with ZipFile(os.path.join(root, 'ml-latest-small.zip'), "r") as zip:
        # Extract files
        print("Extracting all the files now...")
        zip.extractall(path=root)
        print("Downloading Complete")

def read_ratings_csv(root:str) -> pd.DataFrame:
    print("Reading csvfile")
    zipfile  = os.path.join(root,'ml-latest-small.zip')
    if not os.path.isfile(zipfile):
        download_movielens(root)
    fname = os.path.join(root,'ml-latest-small','ratings.csv')
    df = pd.read_csv(fname,sep=',').drop(columns=['timestamp'])
    print("Reading Complete")
    return df


def _train_test_split(root) -> None:
    df = read_ratings_csv(root)
    print('Spliting Traingset & Testset')
    train, test = train_test_split(df, test_size=0.2)
    train_dataset_dir = os.path.join(root,'train-dataset-movieLens', 'dataset')
    train_label_dir = os.path.join(root,'train-dataset-movieLens', 'label')
    test_dataset_dir = os.path.join(root,'test-dataset-movieLens', 'dataset')
    test_label_dir = os.path.join(root,'test-dataset-movieLens', 'label')
    dir_list = [train_dataset_dir, train_label_dir, test_dataset_dir, test_label_dir]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
    train_dataset, train_label = train.iloc[:, :-1], train.iloc[:, [-1]]
    test_datset, test_label = test.iloc[:, :-1], test.iloc[:, [-1]]
    dataset = [train_dataset, train_label, test_datset, test_label]
    data_dir_dict = {}
    for i in range(0, 4):
        data_dir_dict[dir_list[i]] = dataset[i]
    for i,(dir, df) in enumerate(data_dir_dict.items()):
        if i % 2 == 0:
            df.to_csv(dir+'/dataset.csv')
        else:
            df.to_csv(dir+'/label.csv')
    print('Spliting Done')

def load_data(root,train):
    data_file = f"{'train' if train else 'test'}-dataset-movieLens/dataset/dataset.csv"
    data = pd.read_csv(os.path.join(root,data_file))

    label_file = f"{'train' if train else 'test'}-dataset-movieLens/label/label.csv"
    targets = pd.read_csv(os.path.join(root,label_file))
    return data, targets

A = torch.FloatTensor([[1,2],[3,4]])
B = torch.FloatTensor([[1],[2]])
# print(A.shape)
# print(B.shape)
# print(torch.matmul(A,B))

# loss = torch.nn.MSELoss()
# input = torch.randn(3,5,requires_grad=True)
# target = torch.randn(3,5)
# output = loss(input,target)
# output.backward()
# print(output)
# result =[]
# for i in range(0,10):
#     result.append(i)
#
# plt.plot(range(0,10),result)
# plt.show()