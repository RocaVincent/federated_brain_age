from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ReLU, BatchNormalization, LayerNormalization, MaxPool3D, Flatten, Dense, Activation, Input
from tensorflow.keras.initializers import Constant
from declearn.model.tensorflow import TensorflowModel
from .constants import IMAGE_SHAPE

def Regressor(declearn: bool, last_bias_value: float = 0):
    def conv_block(n_filters):
        return Sequential([
            Conv3D(filters=n_filters, kernel_size=3, strides=1, padding='same', use_bias=False),
            ReLU(),
            Conv3D(filters=n_filters, kernel_size=3, strides=1, padding="same", use_bias=False),
            LayerNormalization() if declearn else BatchNormalization(),
            ReLU(),
            MaxPool3D(pool_size=(2,2,1), strides=(2,2,1))
        ])
    model = Sequential([
        Input(shape=IMAGE_SHAPE+[1]),
        conv_block(8),
        conv_block(16),
        conv_block(32),
        conv_block(64),
        conv_block(128),
        Flatten(),
        Dense(units=1, use_bias=True, bias_initializer=Constant(last_bias_value)),
        Activation('linear', dtype='float32')
    ])
    if declearn: model = TensorflowModel(model, loss='mean_absolute_error')
    return model
    