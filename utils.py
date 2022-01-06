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

class Download():
    def __init__(self,
                 root:str,
                 file_size:str='small',
                 download:bool=True,
                 )->None:
        self.root = root
        self.download = download
        self.file_dir = 'ml-latest' if file_size =='large' else 'ml-latest-'+file_size
        self.fname = os.path.join(self.root, self.file_dir, 'ratings.csv')
        if self.download or not os.path.isfile(self.fname) :
            self._download_movielens()
        self.df = self._read_ratings_csv()
        self._train_test_split()

    def _download_movielens(self) -> None:
        '''
        Download dataset from url, if there is no root dir, then mkdir root dir.
        After downloading, it wil be extracted
        :return: None
        '''
        file = self.file_dir+'.zip'
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
        '''
        at first, check if file exists. if it doesn't then call _download().
        it will read ratings.csv, and transform to dataframe.
        it will drop columns=['timestamp'].
        :return:
        '''
        print("Reading file")
        fname = os.path.join(self.root, self.file_dir, 'ratings.csv')
        if not os.path.isfile(fname):
            self._download_movielens()
        df = pd.read_csv(fname, sep=',').drop(columns=['timestamp'])
        print("Reading Complete!")
        return df

    def _train_test_split(self) -> None:
        '''
        this function is called when downloading dataset.
        split dataset in to train and test dataset.
        :return: None
        '''
        df = self.df
        print('Spliting Traingset & Testset')
        # Since MovieLens dataset is user-based dataset, I used Stratified k-fold.
        train, test,dummy_1,dummy_2 = train_test_split(df,df['userId'],test_size=0.2,stratify=df['userId'])
        train_dir = os.path.join(self.root, 'train-dataset-movieLens.csv')
        test_dir = os.path.join(self.root, 'test-dataset-movieLens.csv')
        train.to_csv(train_dir,sep=',',index = False)
        test.to_csv(test_dir,sep=',',index=False)
        print('Spliting Complete!')

    def load_data(self):
        '''
        load total dataframe,train dataframe,test dataframe
        :return: dataframe,dataframe,dataframe
        '''
        print("loading file...")
        train_data_fname = "train-dataset-movieLens.csv"
        test_data_fname = "test-dataset-movieLens.csv"
        total_dataset = pd.read_csv(self.fname,sep=',')
        train_dataset = pd.read_csv(os.path.join(self.root, train_data_fname),sep=',')
        test_dataset = pd.read_csv(os.path.join(self.root,test_data_fname),sep=',')
        print("loading file Complete!")
        return total_dataset, train_dataset, test_dataset



class MovieLens(Dataset):
    def __init__(self,
                 df:pd.DataFrame,
                 total_df:pd.DataFrame
                 )->None:
        super(MovieLens, self).__init__()
        self.df = df
        self.total_df=total_df
        self.user, self.item, self.target = self._data_label_split()
        self.uId_dict, self.mId_dict = self._get_user_count()
        self.bias_user, self.bias_item, self.average = self._get_bias()
        self.confidence_dict =self.uId_dict

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        user = torch.LongTensor([self.user.userId.values[index]])
        item = torch.LongTensor([self.item.movieId.values[index]])
        target = torch.FloatTensor([self.target.rating.values[index]])
        bias_user = torch.FloatTensor([self.bias_user[self.user.userId.values[index]]])
        bias_item = torch.FloatTensor([self.bias_item[self.item.movieId.values[index]]])
        o_avg = torch.FloatTensor([self.average])
        c_score =torch.FloatTensor([self.confidence_dict[self.user.userId.values[index]]])
        return user, item, bias_user, bias_item, o_avg, c_score, target

    def _get_user_count(self) ->(dict):
        '''
        function for generate confidence score
        uId_dict: each user's rating times [key:userId, value: userId's rating times]
        and then apply gaussian distribution to uId_dict.
        :return: user_dictionry, item_dictionary
        '''
        uId_dict = {}
        mId_dict = {}
        df =  self.total_df
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
        return uId_dict, mId_dict

    def normalize(self, dict, target=1.0):
        avg = sum(dict.values())/float(len(dict))
        list_val=[]
        for val in dict.values():
            list_val.append(val)
        std = np.std(list_val)
        return {key: (value-avg)/std for key, value in dict.items()}


    def _get_bias(self) -> (dict,dict,int):
        '''
        function for generate bias.
        bias_user : each user's rating average, dict type
        bias_item : each item's rating average, dict type
        avg_rating : overall average rating
        :return: (dict,dict,int)
        '''
        uId_dict, mId_dict = self.uId_dict, self.mId_dict
        rating_user_sum = {}
        rating_movie_sum = {}
        bias_user = {}
        bias_movie = {}
        overall_sum = 0
        df =  self.total_df
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

    def _data_label_split(self) ->(pd.DataFrame,pd.DataFrame, pd.DataFrame):
        '''
        this function divde in to(user,movie) and (rating)
        :return: pd.
        '''
        df = self.df
        user, item, target = df.iloc[:,[0]],df.iloc[:,[1]],df.iloc[:,[-1]]
        return user,item,target



