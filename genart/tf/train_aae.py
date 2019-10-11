import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from genart.tf.model import GenartAutoencoder, GenartAaeDiscriminator
import genart.gen_images as gi

def ae_loss(outputs, inputs):
    return 

def generate_and_save_images(model, epoch, batch, img_input, latent_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).    
    _,img_output = model(img_input, training=False)
    loss = tf.math.reduce_mean(mse(img_output, img_input))
    print(f"epoch {epoch}, batch {batch}, loss {loss}")

    latent_output = model.decoder(latent_input, training=False)

    fig = plt.figure(figsize=(8,5))

    for i in range(img_output.shape[0]):
        plt.subplot(5, 8, (2*i)+1)
        plt.imshow(np.clip(img_output[i],0,1))
        plt.axis('off')

        plt.subplot(5, 8, (2*i)+2)
        plt.imshow(img_input[i])
        plt.axis('off')

    for i in range(8):
        plt.subplot(5, 8, 2*img_output.shape[0]+i+1)
        plt.imshow(np.clip(latent_output[i],0,1))
        plt.axis('off')

    plt.savefig(f'out_aae/image_{epoch:04d}_{batch:04d}.png')
    plt.close(fig)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):   
    noise = tf.random.normal([batch_size, latent_size])

    with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:        
        input_encoded, output_decoded = autoencoder(images, training=True)        
        
        real_output = discriminator(input_encoded, training=True)
        fake_output = discriminator(noise, training=True)

        ae_loss = mse(output_decoded, images) + generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)


    ae_gradients = ae_tape.gradient(ae_loss, autoencoder.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    autoencoder_optimizer.apply_gradients(zip(ae_gradients, autoencoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def train():
    for epoch in range(n_epochs):
        start = time.time()

        print("generating images")
        imgs = gi.gen_circles(epoch_size, **gi_params)
        print("training", imgs.shape)

        for batch in range(0, epoch_size, batch_size):
            batch_imgs = imgs[batch:batch+batch_size]

            train_step(batch_imgs)

            if batch % 500 == 0:
                generate_and_save_images(autoencoder,
                                         epoch,
                                         batch,
                                         img_seed,
                                         latent_seed)

        # Save the model every 10 epochs
        if epoch % 10 == 0:
            manager.save()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(autoencoder,
                             epoch,
                             batch,
                             img_seed,
                             latent_seed)
    
    manager.save()

latent_size = 2048
img_shape = (256,256,3)
batch_size = 10
epoch_size = 1000
n_epochs = 500
seed_size = 16

gi_params = { 'shape': img_shape, 'n_min': 1, 'n_max': 20, 'dtype': np.float32 }

img_seed = gi.gen_circles(seed_size, **gi_params)
latent_seed = tf.random.normal([seed_size, latent_size])


mse = tf.keras.losses.mean_squared_error
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
autoencoder = GenartAutoencoder(img_shape, latent_size)
discriminator = GenartAaeDiscriminator(latent_size)

autoencoder_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = 'out_aae/tf_ckpts'
ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
                           autoencoder_optimizer=autoencoder_optimizer, 
                           discriminator_optimizer=discriminator_optimizer,
                           autoencoder=autoencoder,
                           discriminator=discriminator)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=2, keep_checkpoint_every_n_hours=1)
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


