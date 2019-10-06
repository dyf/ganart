import tensorflow as tf
import numpy as np
from model import GenartAutoencoder
from data import GenartDataSet



class SaveCB(tf.keras.callbacks.Callback):
    def __init__(self):
        self.current_epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        bi = logs['batch']
        if bi % 200 == 0:

            img = ds[100:101]
            out_img = mod.call(img)
            out_img = tf.image.convert_image_dtype(out_img[0], dtype=tf.uint8)
            jpg = tf.io.encode_jpeg(out_img, quality=100)
            tf.io.write_file(f'out/train_{self.current_epoch:04d}_{bi:04d}.jpg', jpg)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            manager.save()

ds = GenartDataSet("../circles.h5")


latent_size = 512
img_shape = ds.shape[1:3]
batch_size = 10
n_epochs = 100

train_ds = tf.data.Dataset.from_generator(
    ds, 
    output_types=(tf.float32, tf.float32),
    output_shapes=(ds.shape[1:], ds.shape[1:])
).batch(batch_size)

#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
mod = GenartAutoencoder(img_shape, latent_size)

opt = tf.keras.optimizers.Adam()

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=mod)
manager = tf.train.CheckpointManager(ckpt, './out/tf_ckpts', max_to_keep=5, keep_checkpoint_every_n_hours=1)
ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
  

#opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

mod.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
mod.fit(train_ds, epochs=n_epochs, callbacks=[SaveCB()])


