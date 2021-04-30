import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from op import *
from ae import *
from metrics import *
np.set_printoptions(precision=3)


def summarize(model, dataset):
    for i in range(len(dataset.losses)-1):
        print('=========TRAIN=========')
        evaluate(model, dataset.xtr, dataset.ytr[:, i], i)

        print('=========VALID=========')
        evaluate(model, dataset.xte, dataset.yte[:, i], i)


def evaluate(model, x, y, class_id=0):
    logits = model.predict(x)

    if type(logits) == list: logits = logits[class_id];
    if logits.shape[-1] == 1: 
        conf_scores = sigmoid(logits.squeeze())
        preds = 1*(logits.squeeze() > 0.5)
    else: 
        conf_scores = softmax(logits.squeeze()[:, 1])
        preds = np.argmax(logits, axis=-1);
    
    print(preds, conf_scores)
    print('baseline acc: {}'.format(class_one_acc(y)))
    print('model pred acc: {}'.format(acc(y, preds)))
    print('model pred auc: {}'.format(roc_auc(y, conf_scores)))
    
    plot_auc(y, conf_scores, mode='roc', lw=2)
    plot_auc(y, conf_scores, mode='prc', lw=2)

    
def train(dataset, batch_size, epochs):
    model = model_init(dataset)
    tf.random.set_seed(0)
    # must have at least two classification losses
    outputs = {dataset.losses[i]: dataset.ytr[:, i] for i in range(len(dataset.losses)-1)}
    outputs[dataset.losses[-1]] = dataset.xtr
    class_weights = {dataset.losses[i]: dataset.class_weights(i) for i in range(len(dataset.losses)-1)}
    
    validation = {}
    validation[dataset.losses[0]] = dataset.yte
    validation[dataset.losses[-1]] = dataset.xte
    
    history = model.fit(
        x=dataset.xtr,
        y=outputs, 
        batch_size=8, 
        epochs=epochs,
#         validation_data=(dataset.xte, validation), 
#         validation_freq=4,
        class_weight=class_weights,
#         verbose=0,
    )
    
    summarize(model, dataset)
    return history, model


def cross_valid(dataset, batch_size, epochs, n_folds=5):
    x = dataset.features_
    y = dataset.labels
    kf = StratifiedKFold(n_splits=n_folds)
    histories = []

    for train_index, test_index in kf.split(x, y):
        xtr, xte = x[train_index], x[test_index]
        ytr, yte = y[train_index], y[test_index]
        data = (xtr, xte, ytr, yte)
        dataset.set_data(data)
        history, network = train(dataset, batch_size, epochs)
        histories.append(history)
        
    return histories, network


def learn(dataset, batch_size=4, epochs=4, n_folds=1):
    if n_folds < 2:
        return train(dataset, batch_size, epochs)
    else:
        return cross_valid(dataset, batch_size, epochs)