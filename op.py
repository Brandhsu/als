import numpy as np
from tensorflow.keras import layers

# --- Model Operations
conv = lambda x, features, dr=1, name=None : layers.Conv1D(filters=features, kernel_size=1, strides=1, dilation_rate=dr, padding='same', name=name)(x)
elu  = lambda x: layers.ELU()(x)
norm = lambda x: layers.BatchNormalization()(x)
mlp  = lambda x, features, dr=1: elu(norm((conv(x, features, dr))))

# --- Helper operations
argmax = lambda x: np.argmax(x.squeeze(), axis=-1)
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
sigmoid = lambda x: 1 / (1 + np.exp(-x))