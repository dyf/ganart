import os
import tensorflow as tf
import numpy as np
import time
from model import GenartAutoencoder, GenartAeGanDiscriminator, GenartAeGanGenerator
from data import GenartDataSet
import matplotlib.pyplot as plt

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, latent_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as ae_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        autoencoder_output = autoencoder(images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        autoencoder_loss = tf.keras.losses.mean_squared_error(autoencoder_output, images)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_autoencoder = ae_tape.gradient(autoencoder_loss, autoencoder.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    autoencoder_optimizer.apply_gradients(zip(gradients_of_autoencoder, autoencoder.trainable_variables))

def generate_and_save_images(model, epoch, batch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).    
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 8, (2*i)+1)
        plt.imshow(np.clip(predictions[i],0,1))
        plt.axis('off')

        plt.subplot(4, 8, (2*i)+2)
        plt.imshow(test_input[i])
        plt.axis('off')

    plt.savefig(f'out_ae_gan/image_{epoch:04d}_{batch:05d}.png')
    plt.close(fig)

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for i in range(0, len(dataset), BATCH_SIZE):
            if i % 1000 == 0:
                print(i)
            image_batch = dataset[i:i+BATCH_SIZE]
            train_step(image_batch)
    
            if i % 2000 == 0:
                generate_and_save_images(autoencoder,        
                                         epoch,
                                         i,
                                         seed)

        # Save the model every 15 epochs
        if epoch % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(autoencoder,
                             epochs,
                             seed)
    
    checkpoint.save(file_prefix = checkpoint_prefix)

ds = GenartDataSet("../circles.h5")

BATCH_SIZE = 10
EPOCHS = 100

latent_size = 256
img_shape = ds.shape[1:3]

ridx = np.sort(np.random.choice(np.arange(len(ds)), 16))
seed = ds[ridx]

train_ds = tf.data.Dataset.from_generator(
    ds, 
    output_types=(tf.float32, tf.float32),
    output_shapes=(ds.shape[1:], ds.shape[1:])
).batch(BATCH_SIZE)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

autoencoder = GenartAutoencoder(img_shape, latent_size)
generator = GenartAeGanGenerator(autoencoder)
discriminator = GenartAeGanDiscriminator(autoencoder)

autoencoder_optimizer = tf.keras.optimizers.Adam(1e-4)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './out_ae_gan/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    autoencoder_optimizer=autoencoder_optimizer,
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    autoencoder=autoencoder,
    generator=generator,
    discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

train(ds, EPOCHS)


