import numpy as np
import os
import cv2
import sys
from utils import * 
from model import * 
import datetime
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

model = get_model()

# model = tf.keras.models.load_model("model250_s", compile=False)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=custom_loss, optimizer=adam)
model.summary()

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0  and epoch !=0:  
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
#         print("*"*45 + f"two_objects/{filename}.jpg")
        img = cv2.imread(f"two_objects/{filename}.jpg")
        
        with open(f'np_label/{filename}.npy', 'rb') as f:
            np_mat = np.load(f)
        np_mat = np_mat.reshape(169,7)  
        input_[total_files, :, :, :] = img / 255.
        output_[total_files, :, :] = np_mat
        print ("\r", f"total left: {total_files}", end="")
        
        
saver = CustomSaver()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks=[saver, tensorboard_callback]

model.fit(input_, output_, batch_size=8, epochs= 1200, validation_split= 0.1, callbacks = callbacks, shuffle=True)

model.save("model250_f")