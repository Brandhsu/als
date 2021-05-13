import pandas as pd 
import numpy as np
from numpy import nan
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import normalize

from tensorflow.keras import Input


# --- Helpers
def invert_labels(col):
    binary = lambda x: 1 if x == 0 else 0 if x == 1 else nan
    return np.array([binary(c)for c in col.values])

def query_columns(df, query='_Classifier'):
    return [col for col in df.columns if query in col]

def save(obj, path='./DESeq2.pkl'): 
    with open(path, 'wb') as f:
        state  = obj.__dict__.copy()
        del state['Input']
        pkl.dump(state, f, pkl.DEFAULT_PROTOCOL, fix_imports=False)
            
def load(path='./DESeq2.pkl'):
    with open(path, 'rb') as f:
        return pkl.load(f, fix_imports=False)
    
def pca(dataset, n_components=None, verbose=False):
    pca = PCA(n_components).fit(dataset.features)
    if verbose:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
    return pca

def non_als_indices(a, b):
    n = a.shape[-1]
    indices = []
    
    for i in range(n):
        if a[i] == b[i] == 0:
            indices.append(i)
            
    return indices

# Untested (used for TODO1)
def adjust_dataset(dataset, cindex=[1, 3]):
    assert(len(cindex) == 2)
    '''Supports cindex=[1,3] or [2, 4]'''
    
    # get non-als indices base on subtasks
    a = dataset.labels[:, cindex[0]]
    b = dataset.labels[:, cindex[1]]
    indices = non_als_indices(a, b)
    
    features = np.delete(dataset.features_, indices, axis=0)
    labels   = np.delete(dataset.labels_, indices, axis=0)
    pids     = np.delete(dataset.pids_, indices, axis=0)
    
    return features, labels, pids

    
class Dataset():
    def __init__(self, path):
        self.path = path
        self.rand_state = 0
        self.process_data()
        
    def get_data(self):
        return pd.read_csv(self.path)
    
    def set_input(self):
        self.Input = Input(shape=self.features.shape[-2:])
    
    def process_data(self):
        if type(self.path) == list:
            self.transcriptomics_data(self.path)
            self.extract_data(losses=len(list(self.label_dict.keys())))
            self.losses = list(self.label_dict.keys()) + ['reconstruction']
        else:     
            self.data = self.get_data()
            self.extract_data()
            self.losses = [self.data.columns[1], 'reconstruction']

    def transcriptomics_data(self, data_paths):
        assert len(data_paths) == 3
        
        # --- Create initial dataframe
        data_paths.sort()
        datasets = [pd.read_csv(path) for path in data_paths]
        df = datasets[1]
        datasets = [datasets[0], datasets[2]]
        for dataset in datasets:
            df = pd.merge(df, dataset[[dataset.columns[0], dataset.columns[1]]], on=dataset.columns[0], how='left')

        # --- Set labels for each task into a single dataframe
        df.insert(loc=2, column='High_Classifier',   value=invert_labels(df['ALSFRS_Class_Median']))
        df.insert(loc=2, column='Median_Classifier', value=df['ALSFRS_Class_Median'])
        df.insert(loc=2, column='Limb_Classifier',   value=invert_labels(df['SiteOnset_Class']))
        df.insert(loc=2, column='Bulbar_Classifier', value=df['SiteOnset_Class'])
        df = df.drop('SiteOnset_Class', axis=1)
        df = df.drop('ALSFRS_Class_Median', axis=1)
        df = df.fillna(0)
        
        # --- Store data
        self.data = df
        
        # --- Store labels
        self.label_dict = {classifier: df[classifier] for classifier in query_columns(df)}
        
    def extract_data(self, losses=1):
        pid = self.data.columns[0]
        lbl = self.data.columns[1] if losses == 1 else [lbl for lbl in self.label_dict.keys()]
        data = self.data.drop([pid], axis=1)
        data = data.drop([lbl], axis=1) if losses == 1 else data.drop(lbl, axis=1)
        self.features_ = np.expand_dims(data, axis=1)
        self.feature_names_ = data.columns.values
        self.labels_ = np.array(self.data[lbl]) if losses == 1 else np.column_stack([v for k, v in self.label_dict.items()])
        self.pids_ = np.array(self.data[pid])
    
    def cross_valid_split_data(self, train_index, test_index, start=0):
        self.start = start
        self.xtr, self.xte = self.features[train_index], self.features[test_index]
        self.ytr, self.yte = self.labels[train_index], self.labels[test_index]
        self.ptr, self.pte = self.pids_[train_index], self.pids_[test_index]
        self.weights = {i: self.set_class_weights(i) for i in range(self.start, len(self.losses)-1)}
        
    def split_data(self, train_size=0.7, start=0):
        np.random.seed(self.rand_state)
        self.train_size = train_size
        self.start = start
        self.set_input()        
        
        dataset = train_test_split(self.features, 
                                   self.labels, 
                                   self.pids_,
                                   stratify=self.labels, 
                                   train_size=self.train_size, 
                                   random_state=0)
        
        self.xtr, self.xte, self.ytr, self.yte, self.ptr, self.pte = dataset
        self.weights = {i: self.set_class_weights(i) for i in range(self.start, len(self.losses)-1)}
        
    def combine_data(self):
        x = np.concatenate((self.xtr, self.xte))
        y = np.concatenate((self.ytr, self.yte))
        return x, y
    
    def squeeze(self, x):
        return x.squeeze() if len(x.shape) != 2 else x
    
    def set_class_weights(self, class_id=0):
        if len(self.ytr.shape) == 1:
            neg, true = compute_class_weight(class_weight='balanced', classes=np.unique(self.ytr), y=self.ytr)
        else:
            neg, true = compute_class_weight(class_weight='balanced', classes=np.unique(self.ytr[:, class_id]), y=self.ytr[:, class_id])
        return {0: neg, 1: true}
    
    def feature_selection(self, norm=True, feat_select='no', target=0, percentile=.01):
        print('Using {} feature selection and data normalization = {}'.format(feat_select, norm))
        self.target = target
            
        # --- Set data
        X = self.features_.squeeze()
        y = self.labels_
        
        # --- Remove empty columns
        indices = X.any(axis=0)
        X = X[:, indices]

        # --- Normalize X by column to [0, 1]
        if norm:
            self.min = np.min(X, axis=0, keepdims=True)
            self.max = np.max(X, axis=0, keepdims=True)
            X = X - self.min
            X = X / self.max

        features = X
        feature_names = self.feature_names_[indices]
        labels = y

        if feat_select == 'mi':
            try:
                indices = self.mutual_info(X, y[:, self.target], percentile)
            except: 
                indices = self.mutual_info(X, y, percentile)
            features = X[:, indices]
            feature_names = feature_names[indices]
        elif feat_select == 'chi':
            try:
                indices = self.chi2(X, y[:, self.target], percentile)
            except:
                indices = self.chi2(X, y, percentile)
            features = X[:, indices]
            feature_names = feature_names[indices]
            
        elif feat_select == 'pca':
            pass
            
        # --- Update data
        self.features = features
        self.feature_names = feature_names
        self.labels = labels
    
    def chi2(self, X, y, percentile):
        chi, pval = feature_selection.chi2(X, y)
        
        # --- Find indices of top N features
        indices = np.argsort(chi)[::-1][:int(X.shape[-1] * percentile)]
        
        return indices
    
    def mutual_info(self, X, y, percentile, n=3):
        mi = feature_selection.mutual_info_classif(X, y, n_neighbors=n, random_state=self.rand_state)

        # --- Find indices of top N features
        indices = np.argsort(mi)[::-1][:int(X.shape[-1] * percentile)]

        return indices
    
    def to_dataframe(self, lf, fold, save=False):
        # method only supports single task dataset
        assert self.xte.shape[-1] == len(self.feature_names)
        
        if save:
            # --- Latent features dataframe
            d = {'Participant_ID': self.ptes, 
                 str(self.losses[0]): self.ytes}

            # --- Add latent features
            lf_num = self.lfs.shape[-1]
            for i in range(lf_num):
                d['LF_'+str(i)] = self.lfs[:, i]

            # --- Convert to dataframe
            self.lf_df = pd.DataFrame(data=d)

            # --- Input features dataframe
            d = {'Participant_ID': self.ptes, 
                 str(self.losses[0]): self.ytes}

            # --- Add input features
            ft_num = self.xtes.shape[-1]
            for i in range(ft_num):
                d[self.feature_names[i]] = self.xtes[:, i]

            # --- Convert to dataframe
            self.ft_df = pd.DataFrame(data=d)
            
        else:
            if fold == 1:
                self.ptes = self.pte
                self.xtes = self.xte
                self.ytes = self.yte
                self.lfs = lf
            else:
                self.ptes = np.concatenate((self.ptes, self.pte), axis=0)
                self.xtes = np.concatenate((self.xtes, self.xte), axis=0)
                self.ytes = np.concatenate((self.ytes, self.yte), axis=0)
                self.lfs = np.concatenate((self.lfs, lf), axis=0)