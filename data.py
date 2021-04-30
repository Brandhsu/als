import pandas as pd 
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif as mi
from sklearn.preprocessing import normalize

from tensorflow.keras import Input


# --- Helpers
def invert_labels(col):
    binary = lambda x: 1 if x == 0 else 0 if x == 1 else nan
    return np.array([binary(c)for c in col.values])

def query_columns(df, query='_Classifier'):
    return [col for col in df.columns if query in col]
    
    
class Dataset():
    def __init__(self, path, train_size=0.7):
        self.path = path
        self.train_size = train_size
        self.organize_data()

    def get_data(self):
        return pd.read_csv(self.path)
    
    def organize_data(self):
        if type(self.path) == list:
            self.transcriptomics_data(self.path)
            self.losses = list(self.label_names.keys()) + ['reconstruction']
        else:     
            self.data = self.get_data()
            self.extract_data()
            self.losses = ['classification', 'reconstruction']

    def transcriptomics_data(self, data_paths):
        assert len(data_paths) == 3
        data_paths.sort()
        datasets = [pd.read_csv(path) for path in data_paths]
        df = datasets[1]
        datasets = [datasets[0], datasets[2]]
        for dataset in datasets:
            df = pd.merge(df, dataset[[dataset.columns[0], dataset.columns[1]]], on=dataset.columns[0], how='left')

        # --- Renaming classes
        df['Limb_Classifier'] = invert_labels(df['SiteOnset_Class'])
        df['High_Classifier'] = invert_labels(df['ALSFRS_Class_Median'])
        df = df.rename(columns={'SiteOnset_Class': 'Bulbar_Classifier', 
                           'ALSFRS_Class_Median': 'Median_Classifier'})

        df = df.fillna(0)
        self.label_names = {classifier: df[classifier] for classifier in query_columns(df)}
        self.labels = np.column_stack([v for k, v in self.label_names.items()])
        
        # Should refactor
        self.data = df
        pid = self.data.columns[0]
        lbls = [lbl for lbl in self.label_names.keys()]
        data = self.data.drop([pid], axis=1)
        data = data.drop(lbls, axis=1)
        self.features_ = np.expand_dims(data, axis=1)
        self.feature_names_ = data.columns.values
        self.pids = self.data[pid]
    
    def normalize(self):
        return np.expand_dims(normalize(self.squeeze(self.features_)), axis=1)
    
    def set_input(self):
        self.Input = Input(shape=self.features.shape[-2:])
        
    def set_data(self, dataset):
        self.xtr, self.xte, self.ytr, self.yte = dataset
    
    def extract_data(self):
        pid = self.data.columns[0]
        lbl = self.data.columns[1]
        data = self.data.drop([pid, lbl], axis=1)
        self.features_ = np.expand_dims(data, axis=1)
        self.feature_names_ = data.columns.values
        self.labels = self.data[lbl]
        self.pids = self.data[pid]
        
    def split_data(self):
        np.random.seed(0)
        self.set_input()
        dataset = train_test_split(self.features, 
                                   self.labels, 
                                   self.pids,
                                   stratify=self.labels, 
                                   train_size=self.train_size, 
                                   random_state=0)
        
        self.xtr, self.xte, self.ytr, self.yte, self.ptr, self.pte = dataset
        
    def combine_data(self):
        x = np.concatenate((self.xtr, self.xte))
        y = np.concatenate((self.ytr, self.yte))
        return x, y
    
    def squeeze(self, x):
        return x.squeeze() if len(x.shape) != 2 else x
    
    def class_weights(self, class_id=0):
        if len(self.ytr.shape) == 1:
            neg, true = compute_class_weight(class_weight='balanced', classes=np.unique(self.ytr), y=self.ytr)
        else:
            neg, true = compute_class_weight(class_weight='balanced', classes=np.unique(self.ytr[:, class_id]), y=self.ytr[:, class_id])
        return {0: neg, 1: true}  
    
    def feature_selection(self, norm=True, percentile=20, mode='no'):
        np.random.seed(0)
        
        self.features = self.squeeze(self.normalize()) if norm else self.squeeze(self.features_)
        print('Using {} feature selection and data normalization = {}'.format(mode, norm))
        
        if mode == 'chi': 
            self.features = SelectPercentile(chi2, percentile=percentile).fit_transform(self.features, self.labels)
        elif mode == 'mutual_info': 
            self.features = SelectPercentile(mi, percentile=percentile).fit_transform(self.features, self.labels)
            
        self.features = np.expand_dims(self.features, axis=1)
        self.split_data()
    
    def chi(self, x, y):
        return chi2(self.squeeze(x), y)
    
    def mutual_info(self, x, y, n=3):
        return mi(self.squeeze(x), y, n_neighbors=n, random_state=0) 
    
    def pca(self, n_components=None, verbose=False):
        pca = PCA(n_components).fit(self.squeeze(self.features))
        if verbose:
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
        self.pca_ = pca