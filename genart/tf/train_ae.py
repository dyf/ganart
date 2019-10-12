import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from genart.tf.model import GenartAutoencoder
import genart.gen_images as gi

def aeloss(outputs, inputs):
    return tf.keras.losses.mean_squared_error(outputs, inputs)

def generate_and_save_images(model, epoch, batch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).    
    predictions = model(test_input, training=False)
    loss = tf.math.reduce_mean(aeloss(predictions, test_input))
    print(f"epoch {epoch}, batch {batch}, loss {loss}")

    fig = plt.figure(figsize=(8,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 8, (2*i)+1)
        plt.imshow(np.clip(predictions[i],0,1))
        plt.axis('off')

        plt.subplot(4, 8, (2*i)+2)
        plt.imshow(test_input[i])
        plt.axis('off')

    plt.savefig(f'out_ae/image_{epoch:04d}_{batch:04d}.png')
    plt.close(fig)

@tf.function
def train_step(images):    
    with tf.GradientTape() as ae_tape:        
        output = autoencoder(images, training=True)
        loss = aeloss(output, images)

    gradients = ae_tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

def train():
    for epoch in range(n_epochs):
        start = time.time()

        print("generating images")
        imgs = gi.gen_shapes_set(epoch_size, **gi_params)
        print("training", imgs.shape)

        for batch in range(0, epoch_size, batch_size):
            batch_imgs = imgs[batch:batch+batch_size]

            train_step(batch_imgs)

            if batch % 500 == 0:
                generate_and_save_images(autoencoder,
                                         epoch,
                                         batch,
                                         seed)

        # Save the model every 10 epochs
        if epoch % 10 == 0:
            manager.save()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(autoencoder,
                             epoch,
                             batch,
                             seed)
    
    manager.save()

latent_size = 2048
img_shape = (256,256,3)
batch_size = 10
epoch_size = 1000
n_epochs = 1000

gi_params = { 'shape': None, 'img_shape': img_shape, 'n_min': 1, 'n_max': 20 }

seed = gi.gen_shapes_set(16, **gi_params)

#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
autoencoder = GenartAutoencoder(img_shape, latent_size)
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = 'out_ae/tf_ckpts'
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=autoencoder)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3, keep_checkpoint_every_n_hours=1)
ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
  

train()
#opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
#save_cb = SaveCB()

#mod.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
#mod.fit(train_ds, epochs=n_epochs, callbacks=[save_cb])


