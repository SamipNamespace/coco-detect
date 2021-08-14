import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_model():
    img_inputs = keras.Input(shape=(416, 416, 3))
    # x = layers.experimental.preprocessing.Rescaling(scale=1./255.)(img_inputs)

    x = layers.Conv2D(98, (11, 11), strides=1)(img_inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=1)(x)

    x = layers.Conv2D(98, (11, 11), strides=1)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=1)(x)

    x = layers.Conv2D(192, (7, 7))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=1)(x)
    x = layers.Conv2D(192, (7, 7))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=1)(x)

    x = layers.Conv2D(252, (6, 6), strides=2)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(192, (1, 1))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(192, (5, 5), strides=2)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(126, (1, 1))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)


    x = layers.Conv2D(77, (3, 3), strides=1)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(70, (1, 1))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(35, (6, 6), strides=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(7, (5, 5), strides=2)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, kernel_regularizer='l2', activation='relu')(x)
    x = layers.Dense(2048, kernel_regularizer='l2', activation='relu')(x)
    # x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(13*13*(2+1*5), activation='linear', kernel_regularizer='l2')(x)
    x = layers.Reshape((13*13, (2 + 1 * 5)))(x)

    model = keras.Model(img_inputs, x)
    
    return model