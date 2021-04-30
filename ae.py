from tensorflow.keras import Model, losses, optimizers
import tensorflow as tf
from op import *
from custom import *


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
    x = dataset.Input
    input_size = x.shape[-1]
    features = [64]
    embed_size = 32
    
    x_ = encoder(x, features)
    embedding = mlp(x_, embed_size)
    x_ = decoder(embedding, features)

    name_me = layers.Lambda(lambda x: x, name=dataset.losses[-1])
    reconstruction = layers.ReLU()(conv(x_, input_size))
    
    outputs = {dataset.losses[i]: conv(embedding, 1, name=dataset.losses[i]) for i in range(len(dataset.losses)-1)}
    outputs[dataset.losses[-1]] = name_me(reconstruction)
    
    return Model(x, outputs)
    
    
# --- Model Compile
def compile_(model, dataset, lr=1e-2):
    # --- Define losses and metrics
    loss = {dataset.losses[i]: BCE() for i in range(len(dataset.losses)-1)}
    loss[dataset.losses[-1]] = WMSE()
    metrics = {dataset.losses[i]: ['accuracy', tf.keras.metrics.AUC()] for i in range(len(dataset.losses)-1)}
    metrics[dataset.losses[-1]] = 'mean_absolute_error'
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr), 
        # loss={dataset.losses[0]: losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=loss, 
        metrics=metrics,
        experimental_run_tf_function=False)

    return model


# --- Initalize model for training
def model_init(dataset):
    tf.random.set_seed(0)
    model = network(dataset)
    model = compile_(model, dataset)
    return model