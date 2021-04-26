from tensorflow.keras import Model, losses, optimizers
import tensorflow as tf
from op import *

# --- Model Blocks
def encoder(x, features):
    for f in features:
        x = mlp(x, f)
    return x

def decoder(x, features):
    features = features[::-1]
    for f in features:
        x = mlp(x, f)
    return x

def network(dataset):
    tf.random.set_seed(0)
    x = dataset.Input
    input_size = x.shape[-1]
    features = [128, 64]
    embed_size = 32
    outputs = {}
    
    x_ = encoder(x, features)
    embedding = mlp(x_, embed_size)
    x_ = decoder(embedding, features)

    classification = conv(embedding, 2, name=dataset.l1)
    reconstruction = conv(x_, input_size, name=dataset.l2)
    
    outputs[dataset.l1] = classification
    outputs[dataset.l2] = reconstruction
    
    return Model(x, outputs)
    
# --- Model Compile
def compile_(model, dataset, lr=1e-4):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr), 
        loss={dataset.l1: losses.SparseCategoricalCrossentropy(from_logits=True), 
              dataset.l2: losses.MSE}, 
        metrics={dataset.l1: 'accuracy', 
                 dataset.l2: 'mean_absolute_error'},
        experimental_run_tf_function=False)

    return model

# --- Model Trainer
def train(model, dataset):
    outputs = {}
    outputs[dataset.l1] = dataset.train_labels
    outputs[dataset.l2] = dataset.train_data
    class_weights = {dataset.l1: dataset.weights}
    
    validation = {}
    validation[dataset.l1] = dataset.test_labels
    validation[dataset.l2] = dataset.test_data
    
    history = model.fit(
        x=dataset.train_data,
        y=outputs, 
        batch_size=4, 
        epochs=80, 
        validation_data=(dataset.test_data, validation), 
        validation_freq=10,
        class_weight=class_weights,
    )
    return history

# --- Initalize model for training
def model_init(dataset):
    model = network(dataset)
    model = compile_(model, dataset)
    return model