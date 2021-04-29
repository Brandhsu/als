import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


def train(model, dataset):
    tf.random.set_seed(0)
    outputs = {}
    outputs[dataset.l1] = dataset.ytr
    outputs[dataset.l2] = dataset.xtr
    class_weights = {dataset.l1: dataset.weights}
    
    validation = {}
    validation[dataset.l1] = dataset.yte
    validation[dataset.l2] = dataset.xte
    
    history = model.fit(
        x=dataset.xtr,
        y=outputs, 
        batch_size=5, 
        epochs=120,
        validation_data=(dataset.xte, validation), 
        validation_freq=4,
        class_weight=class_weights,
        verbose=0,
    )
    return history


def cross_valid(model, dataset, n_folds=5):
    x = dataset.features_
    y = dataset.labels
    kf = StratifiedKFold(n_splits=n_folds)
    histories = []

    for train_index, test_index in kf.split(x):
        xtr, xte = x[train_index], x[test_index]
        ytr, yte = y[train_index], y[test_index]
        data = (xtr, xte, ytr, yte)
        dataset.set_data(data)
        histories.append(train(model, dataset))
        
    return histories


def learn(model, dataset, n_folds=1):
    if n_folds < 2:
        return train(model, dataset)
    else:
        return cross_valid(model, dataset, n_folds)