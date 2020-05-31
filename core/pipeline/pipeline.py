import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import censusdata
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from distutils.util import strtobool
import warnings
import pickle

class Pipeline:
    """ Generic utils for ml pipeline """

    def __init__(self):
        self.raw_data = None
        self.data = None
        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None

        self.model = {"trained_model": None, "predict_labels": None}
    
    # Getters and setters

    def get_test_targets(self):
        return self.test_targets

    def get_train_features(self):
        return self.train_features

    def set_train_features(self, df):
        self.train_features = df

    def set_test_features(self, df):
        self.test_features = df

    def get_test_features(self):
        return self.test_features

    def get_model(self):
        return self.model

    def get_raw_data(self):
        return self.raw_data

    def load_data(self, data):
        """ Load a processed dataframe into the pipeline"""
        self.data = data

    # Top-level Methdods

    def read_data(self, pkl_path=False, csv_path=False):
        """ 
        Reads pkl or csv file.
        pkl_path (str): pickle file path
        csv_path(str): csv file path
        """

        if pkl_path:
            self.raw_data = pickle.load(open(pkl_path, "rb" ))
        elif csv_path:
            self.raw_data = pd.read_csv(csv_path, sep=',', header = 0)

    def train_test_split(self, target_col, test_size=0.2, seed=42):
        """
        Splits dataset into train/test and feature/target sets.
        target_col (str): a single column header representing the label column
        test-size (float): the percent of the dataset to use for testing
        seed (float): random seed for the data shuffling
        """

        train,test = train_test_split(self.data,test_size=test_size, random_state=seed)
        self.test_targets = test[target_col]
        self.test_features = test.drop([target_col], axis=1)
        self.train_targets = train[target_col]
        self.train_features = train.drop([target_col], axis=1)

    def encode_target_bool(self, target_col):
        """ encode target column from boolean to binary """
        self.train_targets = self.train_targets.map(lambda x: int(x))
        self.test_targets = self.test_targets.map(lambda x: int(x))

    def impute(self, cols, method="mean", constant=None):
        """
        Impute an array of columns. Currently only supports mean imputation.
        cols (list): list of strings representing the columns to impute
        method (string): method of imputation, currently only supports mean imputation
        """

        if method == "mean":
            for col in cols:
                # impute training features
                col_mean = constant
                if not constant:
                    col_mean = self.train_features[col].mean()
                self.train_features[col].fillna(col_mean, inplace=True)

                # impute testing features with training mean
                self.test_features[col].fillna(col_mean, inplace=True)

    def normalize(self, cols,  scaler=None):
        """
        Normalizes the values in columns of a dataset.
        cols (list): list of column headers to normalize
        scaler (obj): non-default scaler object 
        """

        scaled_features = self.train_features.copy()
        col_names = cols
        features = scaled_features[col_names]
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        scaled_features[col_names] = features
        self.train_features = scaled_features

        # normalize continuous train targets
        #self.train_targets = scaler.transform(train_targets)

        # normalize test_features based on train_feature scaler
        scaled_features = self.test_features.copy()
        col_names = cols
        features = scaled_features[col_names]
        features = scaler.transform(features.values)
        scaled_features[col_names] = features
        self.test_features = scaled_features


    def onehot_encode(self, cols):
        """
        Conducts one-hot encoding of selected columns. Drops the original columns.
        cols (list): list of column headers to one-hot encode. 
        """
        
        train_features = self.train_features
        test_features = self.test_features

        train_onehot_cols = train_features[cols].astype(str)
        train_onehot_encoded = pd.get_dummies(train_onehot_cols)
        test_onehot_cols = test_features[cols].astype(str)
        test_onehot_encoded = pd.get_dummies(test_onehot_cols)

        zero_cols = np.setdiff1d(train_onehot_encoded.columns, test_onehot_encoded.columns)

        for zero_col in zero_cols:
            test_onehot_encoded[zero_col] = 0

        # append back the one-hot encoded features and drop the original columns
        train_features = train_features.drop(cols, axis=1)
        test_features = test_features.drop(cols, axis=1)

        train_features[train_onehot_encoded.columns] = train_onehot_encoded
        test_features[test_onehot_encoded.columns] = test_onehot_encoded 

        self.train_features = train_features
        self.test_features = test_features


    # Data summary methods

    def summarize_by_month(self, date_col, var):
        """
        Graphs variable frequency by month
        date_col (string): header of column representing the date
        var (string): header of column that we want to track by month
        """

        self.data['month'] = pd.to_datetime(self.data['Date']).dt.to_period('M')
        months = self.data['month'].sort_values()
        start_month = months.iloc[0]
        end_month = months.iloc[-1]
        index = pd.PeriodIndex(start=start_month, end=end_month)
        self.data.groupby('month')[var].count().reindex(index).plot.bar()


    def summarize_by_var_freq(self, col, xlabel=None, ylabel=None, title=None):
        """
        Graphs frequency of values in a column.
        col (string): header of the column to graph
        xlabel (string): xlabel of graph
        ylabel (string): ylabel of graph
        title (stribng): title of graph
        """

        plt_freq = self.data[col].value_counts().plot('bar')
        if xlabel:
            plt_freq.set_xlabel(xlabel)
        if ylabel:
            plt_freq.set_ylabel(ylabel)
        if title:
            plt_freq.set_title(title)

    # Training, prediction, evaluation methods

    def fit(self, clf):
        """ fits a classifier given clf object  """
        self.model["trained_model"] = clf.fit(self.train_features, self.train_targets)
        self.model["predict_labels"] = None # wipes predictions if new model is trained

    def predict(self):
        """ makes prediction on training data given that a classifier was already trained """
        self.model["predict_labels"] = self.model["trained_model"].predict(self.test_features)

    def eval_accuracy_score(self):
        """ evaluates accuracy of currently cached model """
        return accuracy_score(self.test_targets, self.model["predict_labels"])
        

