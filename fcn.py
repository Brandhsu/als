import numpy as np, pandas as pd
from sklearn import feature_selection, metrics
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import Input, Model, models, layers, losses, optimizers

from data import Dataset

def inference(model, x, class_id=0):
    pred = model.predict(x)
    return pred if len(pred) == x.shape[0] else model.predict(x)[class_id]

def acc(pred, y):
    pred = np.argmax(pred, axis=-1)
    print('acc     score: {}'.format(sum(pred == y) / len(y)))
          
def roc_auc(pred, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred[:, -1])
    print('roc_auc score: {}'.format(metrics.auc(fpr, tpr)))
    
def prc_auc(pred, y):
    precision, recall, _ = metrics.precision_recall_curve(y, pred[:, -1])
    print('prc_auc score: {}'.format(metrics.auc(recall, precision)))
    
def load_model(path='./model.hdf5'):
    return models.load_model(path, compile=False)

def save_model(model, path='./model.hdf5'):
    model.save(path)
    
def get_embed(model):
    return tf.keras.Sequential(model.layers[:5])
    
def create_model(dataset, shape, lr, drop=0.5):

    x = Input(shape=shape)

    l0 = layers.Dense(64, activation='relu')(layers.Dropout(drop)(x))
    l1 = layers.Dense(64, activation='relu', name='latent_feature')(layers.Dropout(drop)(l0))
    
    logits = {dataset.losses[i]: layers.Dense(2, name=dataset.losses[i])(layers.Dropout(drop)(l1)) for i in range(len(dataset.losses)-1)}       
   
    model = Model(inputs=x, outputs=logits)
    
    loss = {dataset.losses[i]: losses.SparseCategoricalCrossentropy(from_logits=True) for i in range(len(dataset.losses)-1)}
    metrics = {dataset.losses[i]: ['sparse_categorical_accuracy'] for i in range(len(dataset.losses)-1)}

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics)

    return model

def train_model(dataset, lr, batch_size, epochs, drop, verbose):
    
    # --- Create model
    tf.random.set_seed(0)
    model = create_model(dataset, dataset.xtr.shape[1:], lr, drop)

    # --- Find train/valid cohort split
    X_train = dataset.xtr
    X_valid = dataset.xte
    
    # --- Identify tasks
    try:
        y_train = {dataset.losses[i]: dataset.ytr[:, i] for i in range(len(dataset.losses)-1)}
        y_valid = {dataset.losses[i]: dataset.yte[:, i] for i in range(len(dataset.losses)-1)}
    except:
        y_train = {dataset.losses[i]: dataset.ytr for i in range(len(dataset.losses)-1)}
        y_valid = {dataset.losses[i]: dataset.yte for i in range(len(dataset.losses)-1)}
    
    # --- Add class weights to each task
    try:
        class_weights = {dataset.losses[i]: dataset.weights[i] for i in range(dataset.start, len(dataset.losses)-1)}
    except:
        class_weights = dataset.weights[0]
    
    history = model.fit(
        x=X_train, 
        y=y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        class_weight=class_weights,
        epochs=epochs,
        use_multiprocessing=False,
        verbose=verbose)

    for i in range(len(dataset.losses)-1):
        print(dataset.losses[i])
        print('--- Training Results')
        acc(inference(model, X_train, i), y_train[dataset.losses[i]])
        roc_auc(inference(model, X_train, i), y_train[dataset.losses[i]])
        prc_auc(inference(model, X_train, i), y_train[dataset.losses[i]])
        print('--- Validation Results')
        acc(inference(model, X_valid, i), y_valid[dataset.losses[i]])
        roc_auc(inference(model, X_valid, i), y_valid[dataset.losses[i]])
        prc_auc(inference(model, X_valid, i), y_valid[dataset.losses[i]])
    
    return history, model

def cross_valid(dataset, lr, batch_size, epochs, drop, verbose, n_folds=5):
    X = dataset.features
    y = dataset.labels
    kf = StratifiedKFold(n_splits=n_folds)
    histories = []
    fold = 1
    
    try:
        z = y[:, 0]
    except:
        z = y

    for train_index, test_index in kf.split(X, z):
        print(f'--- Fold {fold}')
        dataset.cross_valid_split_data(train_index, test_index)
        history, model = train_model(dataset, lr, batch_size, epochs, drop, verbose)
        histories.append(history)
        
        # --- Get latent features
        network = Model(inputs=model.inputs,
                        outputs=model.get_layer(name='latent_feature').output,)
            
        lf = network.predict(dataset.xte)
        
        # --- Store to dataframe
        dataset.to_dataframe(lf, fold)
        fold += 1
        
    dataset.to_dataframe(lf, fold, save=True)
        
    return histories, model


def learn(dataset, lr, batch_size, epochs, drop, n_folds, verbose):
    if n_folds < 2:
        return train_model(dataset, lr, batch_size, epochs, drop, verbose)
    else:
        return cross_valid(dataset, lr, batch_size, epochs, drop, verbose, n_folds)
    
# ======================================================================
if __name__ == '__main__':
    dataset = load_data()
    learn(dataset, lr=1e-3, batch_size=32, epochs=100, drop=0.5, n_folds=1)
# ======================================================================