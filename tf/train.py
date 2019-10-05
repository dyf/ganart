import tensorflow as tf
import numpy as np
from model import GenartAutoencoder
from data import GenartDataSet



class SaveImageCB(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        bi = logs['batch']
        if bi % 100 == 0:

            img = ds[100:101]
            out_img = mod.call(img)
            out_img = tf.image.convert_image_dtype(img[0], dtype=tf.uint8)
            jpg = tf.io.encode_jpeg(out_img, quality=100)
            tf.io.write_file(f'out/train_{bi:04d}.jpg', jpg)

    def on_epoch_end(self, epoch, logs=None):
        pass

ds = GenartDataSet("../circles.h5")


latent_size = 512
img_shape = ds.shape[1:3]
batch_size = 10
n_epochs = 100

train_ds = tf.data.Dataset.from_generator(
    ds, 
    output_types=(tf.float32, tf.float32),
    output_shapes=((256,256,3),(256,256,3))
).batch(batch_size)

mod = GenartAutoencoder(img_shape, latent_size)

mod.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
mod.fit(train_ds, epochs=n_epochs, callbacks=[SaveImageCB()])


