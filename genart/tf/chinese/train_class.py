import os, time
import matplotlib.pyplot as plt

import tensorflow as tf
import genart.tf.chinese.data as data
import genart.tf.chinese.model as model


categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output):
    # if the generator is succeeding, the discriminator thinks the images are real.
    # first one-hot position is fake, so let's make sure that's all zeros.
    return binary_cross_entropy(tf.zeros([fake_output.shape[0]]), fake_output[:,0])

def discriminator_loss(real_output, real_classes, fake_output):
    real_loss = categorical_cross_entropy(real_classes, real_output)

    fake_cats = tf.ones_like(fake_output) * data.CharacterClass.FAKE.value
    fake_oh = tf.one_hot(fake_cats, depth=len(data.CharacterClass))
    fake_loss = categorical_cross_entropy(fake_oh, fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

@tf.function
def train_step(images, image_classes, batch_size, latent_size):
    noise = tf.random.normal([batch_size, latent_size])
    noise_classes_cat = tf.random.uniform([batch_size], minval=1, maxval=len(data.CharacterClass))
    noise_classes_oh = tf.one_hot(noise_classes_cat, depth=len(data.CharacterClass))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, noise_classes_oh], training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, batch, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fname = os.path.join(checkpoint_dir, 'image_batch_{:07d}_epoch_{:02d}.png'.format(batch, epoch))
    plt.savefig(fname)
    plt.close()

def train(dataset, epochs, latent_size):
    for epoch in range(epochs):
        start = time.time()

        bi = 0
        for metadata_batch, image_batch in dataset():                        
            iamge_classes = tf.one_hot(metadata_batch['class'])

            gl, dl = train_step(image_batch, image_classes, image_batch.shape[0], latent_size)

            if bi % 4000 == 0:
                print(f'epoch({epoch}) batch({bi//1000}) genloss({gl}) discloss({dl})')
                generate_and_save_images(generator,
                                         bi + 1,
                                         epoch + 1,
                                         [seed, seed_classes])

            bi+=1

        

        # Save the model every epoch
        manager.save()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                             0,
                             epochs,
                             [seed, seed_classes])


if __name__ == "__main__":
    num_examples_to_generate = 16
    latent_size = 100
    batch_size = 20
    num_epochs = 10

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, latent_size])
    seed_classes = tf.random.uniform([num_examples_to_generate], minval=1, maxval=len(data.CharacterClass))

    generator, discriminator = model.build_gan(latent_size=latent_size)

    generator_optimizer = tf.keras.optimizers.Adam(1e-5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

    checkpoint_dir = './data/chinese_class_output/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)                                    

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train(lambda: data.iterdata(batch_size=batch_size), epochs=num_epochs, latent_size=latent_size)