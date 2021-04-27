import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif as mi

from tensorflow.keras import Input


class Dataset():
    def __init__(self, path, train_size=0.7):
        self.path = path
        self.train_size = train_size
        self.data = self.get_data()
        self.extract_data()
        self.split_data()
        
        self.l1 = 'classification'
        self.l2 = 'reconstruction'

    def get_data(self):
        return pd.read_csv(self.path)
    
    def set_input(self):
        self.Input = Input(shape=self.features.shape[-2:])
    
    def extract_data(self):
        pid = self.data.columns[0]
        lbl = self.data.columns[1]
        self.features = np.expand_dims(self.data.drop([pid, lbl], axis=1), axis=1)
        self.labels = self.data[lbl]
        
    def split_data(self):
        np.random.seed(0)
        self.set_input()
        dataset = train_test_split(self.features, 
                                   self.labels, 
                                   stratify=self.labels, 
                                   train_size=self.train_size, 
                                   random_state=0)
        
        self.xtr, self.xte, self.ytr, self.yte = dataset
        self.weights = self.class_weights()
        
    def combine_data(self):
        x = np.concatenate((self.xtr, self.xte))
        y = np.concatenate((self.ytr, self.yte))
        return x, y
    
    def class_weights(self):
        neg, true = compute_class_weight(class_weight='balanced', classes=np.unique(self.ytr), y=self.ytr)
        return {0: neg, 1: true}  
    
    def feature_selection(self, percentile=10, mode='default'):
        np.random.seed(0)
        if mode == 'chi': score_func = chi2
        elif mode == 'mutual_info': score_func = mi
        else: print('{} not in {{chi, mutual_info}} so using default features'.format(mode)); return;
        
        x, y = self.combine_data()
        if len(x.shape) != 2: x = x.squeeze();
        self.features = SelectPercentile(score_func, percentile=percentile).fit_transform(x, y)
        self.features = np.expand_dims(self.features, axis=1)
        self.labels = y
        self.split_data()
    
    def chi(self, x, y):
        if len(x.shape) != 2: x = x.squeeze();
        return chi2(x, y)
    
    def mutual_info(self, x, y, n=3):
        if len(x.shape) != 2: x = x.squeeze();
        return mi(x, y, n_neighbors=n, random_state=0)    