import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
import numpy as np

from model import GenartAutoencoder, GenartGenerator, GenartDiscriminator
from data import GenartDataSet


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def load_autoencoder(img_shape, latent_size):
    mod = GenartAutoencoder(img_shape, latent_size)
    opt = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=mod)
    manager = tf.train.CheckpointManager(ckpt, './foo/tf_ckpts', max_to_keep=5, keep_checkpoint_every_n_hours=1)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("compiling from scratch")#raise Exception("No checkpoint found")

    mod.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    return mod

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, latent_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for i in range(0, len(dataset), BATCH_SIZE):
            if i % 1000 == 0:
                print(i)
            image_batch = dataset[i:i+BATCH_SIZE]
            train_step(image_batch)
    
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if epoch % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             seed)
    
    checkpoint.save(file_prefix = checkpoint_prefix)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(np.clip(predictions[i],0,1))
      plt.axis('off')

  plt.savefig('out_gan/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close(fig)

EPOCHS = 50
BATCH_SIZE = 10

img_shape = (256,256,3)
latent_size = 256
num_examples_to_generate = 16

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

ds = GenartDataSet('../circles.h5')
#ae = load_autoencoder(img_shape, latent_size)

generator = GenartGenerator(img_shape, latent_size)
discriminator = GenartDiscriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './out_gan/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

seed = tf.random.normal([num_examples_to_generate, latent_size])

train(ds, EPOCHS)