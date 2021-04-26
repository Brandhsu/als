import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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
    
    def extract_data(self):
        pid = self.data.columns[0]
        lbl = self.data.columns[1]
        self.features = np.expand_dims(self.data.drop([pid, lbl], axis=1), axis=1)
        self.labels = self.data[lbl]
        self.Input = Input(shape=self.features.shape[-2:])
        
    def split_data(self):
        np.random.seed(0)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.features, 
                                                                                                self.labels, 
                                                                                                stratify=self.labels, 
                                                                                                train_size=self.train_size, 
                                                                                                random_state=0)
        self.weights = self.class_weights()
    
    def class_weights(self):
        neg, true = compute_class_weight(class_weight='balanced', classes=np.unique(self.train_labels), y=self.train_labels)
        return {0: neg, 1: true}  