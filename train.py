import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
from keras import backend as K
import sys
from utils import * 
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

img_inputs = keras.Input(shape=(416, 416, 3))
x = layers.experimental.preprocessing.Rescaling(scale=1./255.)(img_inputs)

x = layers.Conv2D(98, (11, 11), strides=1)(x)
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
x = layers.Dense(13*13*(2+1*5), activation='sigmoid', kernel_regularizer='l2')(x)
x = layers.Reshape((13*13, (2 + 1 * 5)))(x)

model = keras.Model(img_inputs, x)

# model = tf.keras.models.load_model("model250_s", compile=False)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=custom_loss, optimizer=adam)
model.summary()

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 41 == 0  and epoch !=0:  
            print("saving")
            self.model.save("model250_s")

files = os.listdir("two_objects")
total_files = len(files)

input_ = np.zeros((total_files, 416, 416, 3))
output_ = np.zeros((total_files,169 , 7))

for file in files:
    if file.endswith(".jpg"):
        total_files -= 1
        filename = file.split(".")[0]
        print("*"*45 + f"two_objects/{filename}.jpg")
        img = cv2.imread(f"two_objects/{filename}.jpg")
        
        with open(f'np_label/{filename}.npy', 'rb') as f:
            np_mat = np.load(f)
        np_mat = np_mat.reshape(169,7)  
        input_[total_files, :, :, :] = img 
        output_[total_files, :, :] = np_mat
        print ("\r", f"total left: {total_files}", end="")
        
        
saver = CustomSaver()

callbacks=[saver]

model.fit(input_, output_, batch_size=8, epochs= 1200, validation_split= 0.1, callbacks = callbacks )

model.save("model250_f")