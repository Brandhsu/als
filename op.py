from tensorflow.keras import layers

# --- Model Operations
conv = lambda x, features, dr=1, name=None : layers.Conv1D(filters=features, kernel_size=1, strides=1, dilation_rate=dr, padding='same', name=name)(x)
elu  = lambda x: layers.ELU()(x)
norm = lambda x: layers.BatchNormalization()(x)
mlp  = lambda x, features, dr=1: elu(norm((conv(x, features, dr))))