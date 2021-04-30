import numpy as np, pandas as pd
from sklearn import feature_selection, metrics
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, losses, optimizers

from data import Dataset

def load_data(csv=['./data/ctrl_vs_case.csv', './data/bulbar_vs_limb.csv', './data/median_low_vs_high.csv'], norm=True, feat_select=True, **kwargs):
    
    np.random.seed(0)
    dataset = Dataset(csv, train_size=0.7)
    dataset.features = dataset.features_.squeeze()
    
    y = dataset.labels
    X = dataset.features

    # --- Remove empty columns
    X = X[:, X.any(axis=0)]

    # --- Normalize X by column to [0, 1]
    if norm:
        X = X - np.min(X, axis=0, keepdims=True)
        X = X / np.max(X, axis=0, keepdims=True)
    
    dataset.features = X
    
    if feat_select:
        dataset.features = select_features(X, y[:, 0], **kwargs)
        
    dataset.split_data()

    return dataset

def select_features(X, y, top_percent=0.01, random_state=0):
    """
    Method to select top N features

    """
    mi = feature_selection.mutual_info_classif(X, y, random_state=random_state)

    # --- Find indices of top N features
    indices = np.argsort(mi)[::-1][:int(X.shape[-1] * top_percent)]

    return X[:, indices]

def inference(model, x, class_id):
    return model.predict(x)[class_id][:, -1]
    
def roc_auc(pred, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    print('auc score: {}'.format(metrics.auc(fpr, tpr)))

def create_model(dataset, shape, lr):

    x = Input(shape=shape)

    l0 = layers.Dense(64, activation='relu')(layers.Dropout(0.5)(x))
    l1 = layers.Dense(64, activation='relu')(layers.Dropout(0.5)(l0))

    logits = {dataset.losses[i]: layers.Dense(2, name=dataset.losses[i])(layers.Dropout(0.5)(l1)) for i in range(len(dataset.losses)-1)}

    model = Model(inputs=x, outputs=logits)
    
    loss = {dataset.losses[i]: losses.SparseCategoricalCrossentropy(from_logits=True) for i in range(len(dataset.losses)-1)}
    metrics = {dataset.losses[i]: ['sparse_categorical_accuracy'] for i in range(len(dataset.losses)-1)}

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics)

    return model

def train_model(dataset, lr, batch_size, epochs):
    
    # --- Create model
    tf.random.set_seed(0)
    model = create_model(dataset, dataset.xtr.shape[1:], lr)

    # --- Find train/valid cohort split
    X_train = dataset.xtr
    X_valid = dataset.xte
    
    # --- Identify tasks
    y_train = {dataset.losses[i]: dataset.ytr[:, i] for i in range(len(dataset.losses)-1)}
    y_train[dataset.losses[-1]] = X_train
    y_valid = {dataset.losses[i]: dataset.yte[:, i] for i in range(len(dataset.losses)-1)}
    y_valid[dataset.losses[-1]] = X_valid
    
    # --- Add class weights to each task
    class_weights = {dataset.losses[i]: dataset.class_weights(i) for i in range(len(dataset.losses)-1)}

    history = model.fit(
        x=X_train, 
        y=y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        class_weight=class_weights,
        epochs=epochs)

    for i in range(len(dataset.losses)-1):
        print(dataset.losses[i])
        roc_auc(inference(model, X_train, i), y_train[dataset.losses[i]])
        roc_auc(inference(model, X_valid, i), y_valid[dataset.losses[i]])
    
    return history, model

def cross_valid(dataset, lr, batch_size, epochs, n_folds=5):
    X = dataset.features
    y = dataset.labels
    kf = StratifiedKFold(n_splits=n_folds)
    histories = []

    for train_index, test_index in kf.split(X, y):
        xtr, xte = X[train_index], X[test_index]
        ytr, yte = y[train_index], y[test_index]
        data = (xtr, xte, ytr, yte)
        dataset.set_data(data)
        history, model = train_model(dataset, batch_size, epochs)
        histories.append(history)
        
    return histories, model


def learn(dataset, lr, batch_size, epochs, n_folds):
    if n_folds < 2:
        return train_model(dataset, lr, batch_size, epochs)
    else:
        return cross_valid(dataset, lr, batch_size, epochs, n_folds)
    
# ======================================================================
if __name__ == '__main__':
    dataset = load_data()
    learn(dataset, lr=1e-3, batch_size=32, epochs=100, n_folds=1)
# ======================================================================