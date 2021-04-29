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
    features = [128, 64]
    embed_size = 32
    features = [1024, 128]
    embed_size = 32
    outputs = {}
    
    x_ = encoder(x, features)
    embedding = mlp(x_, embed_size)
    x_ = decoder(embedding, features)

    classification = conv(embedding, 1, name=dataset.l1)
    # reconstruction = conv(x_, input_size, name=dataset.l2)
    name_me = layers.Lambda(lambda x: x, name=dataset.l2)
    reconstruction = layers.ReLU()(conv(x_, input_size))
    reconstruction = name_me(reconstruction)
    
    outputs[dataset.l1] = classification
    outputs[dataset.l2] = reconstruction
    
    return Model(x, outputs)
    
    
# --- Model Compile
def compile_(model, dataset, lr=1e-2):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr), 
        # loss={dataset.l1: losses.SparseCategoricalCrossentropy(from_logits=True),
        loss={dataset.l1: BCE(),
              dataset.l2: WMSE()}, 
        metrics={dataset.l1: ['accuracy', tf.keras.metrics.AUC()],
                 dataset.l2: 'mean_absolute_error'},
        experimental_run_tf_function=False)

    return model


# --- Initalize model for training
def model_init(dataset):
    tf.random.set_seed(0)
    model = network(dataset)
    model = compile_(model, dataset)
    return model