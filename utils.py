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
from sklearn.model_selection  import train_test_split

class MovieLens(Dataset):
    def __init__(self,
                 root:str,
                 file_size:str,
                 train:bool=False,
                 download:bool=False)->None:
        super(MovieLens, self).__init__()
        self.root = root
        self.file_dir = 'ml-latest-'+file_size
        if download:
            self._download_movielens()
            self._train_test_split()
        self.train = train
        self.data, self.target = self._load_data()

    def _load_data(self):
        data_file = f"{'train' if self.train else 'test'}-dataset-movieLens/dataset/dataset.csv"
        data = pd.read_csv(os.path.join(self.root, data_file))

        label_file = f"{'train' if self.train else 'test'}-dataset-movieLens/label/label.csv"
        targets = pd.read_csv(os.path.join(self.root, label_file))
        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        user = torch.LongTensor([self.data.userId.values[index]])
        item = torch.LongTensor([self.data.movieId.values[index]])
        target = torch.FloatTensor([self.target.rating.values[index]])
        return user,item,target

    def _download_movielens(self) -> None:
        file = self.file_dir+'.zip'
        print(file)
        url = ("http://files.grouplens.org/datasets/movielens/"+file)
        req = requests.get(url, stream=True)
        print('Downloading MovieLens dataset')
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        with open(os.path.join(self.root, file), mode='wb') as fd:
            for chunk in req.iter_content(chunk_size=None):
                fd.write(chunk)
        with ZipFile(os.path.join(self.root, file), "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=self.root)
            print("Downloading Complete!")

    def _read_ratings_csv(self) -> pd.DataFrame:
        file = self.file_dir+'.zip'
        print("Reading csvfile")
        zipfile = os.path.join(self.root,file)
        if not os.path.isfile(zipfile):
            self._download_movielens(self.root)
        fname = os.path.join(self.root, self.file_dir, 'ratings.csv')
        df = pd.read_csv(fname, sep=',').drop(columns=['timestamp'])
        print("Reading Complete!")
        return df

    def get_numberof_users_items(self) -> tuple:
        df = self.data
        return df["userId"].nunique(), df["movieId"].nunique()

    def _train_test_split(self) -> None:
        df = self._read_ratings_csv()
        print('Spliting Traingset & Testset')
        train, test = train_test_split(df, test_size=0.2) # should add stratify
        train_dataset_dir = os.path.join(self.root, 'train-dataset-movieLens', 'dataset')
        train_label_dir = os.path.join(self.root, 'train-dataset-movieLens', 'label')
        test_dataset_dir = os.path.join(self.root, 'test-dataset-movieLens', 'dataset')
        test_label_dir = os.path.join(self.root, 'test-dataset-movieLens', 'label')
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
        for i, (dir, df) in enumerate(data_dir_dict.items()):
            if i % 2 == 0:
                df.to_csv(dir + '/dataset.csv')
            else:
                df.to_csv(dir + '/label.csv')
        print('Spliting Complete!')




