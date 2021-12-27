import zipfile
from torch.utils.data import Dataset
import torch
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
                 download:bool=False,
                 )->None:
        super(MovieLens, self).__init__()
        self.root = root
        if file_size =='large':
            self.file_dir='ml-latest'
        else:
            self.file_dir = 'ml-latest-'+file_size
        if download:
            self._download_movielens()
            self.df = self._read_ratings_csv()
            self._train_test_split()
        if not download and train:
            self.df = self._read_ratings_csv()
        self.train = train
        if self.train:
            self.uId_dict,self.mId_dict = self.get_key_count()
        self.data, self.target = self._load_data()

    def _load_data(self):
        data_file = f"{'train' if self.train else 'test'}-dataset-movieLens/dataset/dataset.csv"
        data = pd.read_csv(os.path.join(self.root, data_file))

        label_file = f"{'train' if self.train else 'test'}-dataset-movieLens/label/label.csv"
        targets = pd.read_csv(os.path.join(self.root, label_file))
        print(f"loading {'train' if self.train else 'test'} file Complete!")
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
            self._download_movielens(self)
        fname = os.path.join(self.root, self.file_dir, 'ratings.csv')
        df = pd.read_csv(fname, sep=',').drop(columns=['timestamp'])
        print("Reading Complete!")
        return df

    def get_numberof_users_items(self) -> tuple:
        df = self.df
        return df["userId"].nunique(), df["movieId"].nunique()

    def get_key_count(self) ->(dict,dict):
        uId_dict = {}
        mId_dict = {}
        df = self.df
        for index, row in df.iterrows():
            try:
                count = uId_dict[row['userId']] + 1
            except KeyError:
                count = 1
            uId_dict[row['userId']] = count
            try:
                count2 = mId_dict[row['movieId']] + 1
            except KeyError:
                count2 = 1
            mId_dict[row['movieId']] = count2
        return uId_dict,mId_dict



    def normalize(self, d, target=1.0):
        avg = sum(d.values())/float(len(d))
        list_val=[]
        for val in d.values():
            list_val.append(val)
        std = np.std(list_val)
        return {key: (value-avg)/std for key, value in d.items()}


    def get_bias(self) -> (dict,dict,int):

        uId_dict, mId_dict = self.uId_dict, self.mId_dict
        rating_user_sum = {}
        rating_movie_sum = {}
        bias_user = {}
        bias_movie = {}
        overall_sum = 0
        df = self.df
        for index, row in df.iterrows():
            try:
                rating_user = rating_user_sum[row['userId']] + row['rating']
            except KeyError:
                rating_user = row['rating']
            rating_user_sum[row['userId']] = rating_user

            try:
                rating_movie = rating_movie_sum[row['movieId']] + row['rating']
            except KeyError:
                rating_movie = row['rating']

            rating_movie_sum[row['movieId']] = rating_movie
            overall_sum += row['rating']


        for key, value in uId_dict.items():
            bias_user[key] = rating_user_sum[key] / value  # value = number of rating by user[key]
        for key, value in mId_dict.items():
            bias_movie[key] = rating_movie_sum[key] / value # value = number of rating by movie[key]
        avg_rating = overall_sum / len(df)

        return bias_user,bias_movie,avg_rating

    def _train_test_split(self) -> None:
        df = self.df
        print('Spliting Traingset & Testset')
        train, test = train_test_split(df,test_size=0.2) # should add stratify
        train_dataset_dir = os.path.join(self.root, 'train-dataset-movieLens', 'dataset')
        train_label_dir = os.path.join(self.root, 'train-dataset-movieLens', 'label')
        test_dataset_dir = os.path.join(self.root, 'test-dataset-movieLens', 'dataset')
        test_label_dir = os.path.join(self.root, 'test-dataset-movieLens', 'label')
        dir_list = [train_dataset_dir, train_label_dir, test_dataset_dir, test_label_dir]
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
        train_dataset, train_label = train.iloc[:, :-1], train.iloc[:, [-1]]
        test_dataset, test_label = test.iloc[:, :-1], test.iloc[:, [-1]]
        dataset = [train_dataset, train_label, test_dataset, test_label]
        data_dir_dict = {}
        for i in range(0, 4):
            data_dir_dict[dir_list[i]] = dataset[i]
        for i, (dir, df) in enumerate(data_dir_dict.items()):
            if i % 2 == 0:
                df.to_csv(dir + '/dataset.csv')
            else:
                df.to_csv(dir + '/label.csv')
        print('Spliting Complete!')




