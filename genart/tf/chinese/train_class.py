import os, time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import genart.tf.chinese.data as data
import genart.tf.chinese.model as model


categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output):
    fake_class_output = fake_output[0]
    # if the generator is succeeding, the discriminator thinks the images are real.
    # first one-hot position is fake, so let's make sure that's all zeros.
    return binary_cross_entropy(tf.zeros([fake_class_output.shape[0]]), fake_class_output[:,0])

def discriminator_loss(real_output, real_classes, real_fonts, fake_output):
    real_class_output, real_font_output = real_output    

    # discriminator wants to correctly identify class and font
    real_class_loss = categorical_cross_entropy(real_classes, real_class_output)
    real_font_loss = categorical_cross_entropy(real_fonts, real_font_output)

    fake_class_output, fake_font_output = fake_output

    # discriminator wants to correctly yell fake    
    fake_class_cats = tf.ones((fake_class_output.shape[0],1), dtype=tf.dtypes.int32) * data.CharacterClass.FAKE.value    
    fake_class_oh = tf.one_hot(fake_class_cats, depth=len(data.CharacterClass))
    fake_class_loss = categorical_cross_entropy(fake_class_oh, fake_class_output)

    fake_font_cats = tf.ones((fake_font_output.shape[0],1), dtype=tf.dtypes.int32) * font_lut['FAKE']
    fake_font_oh = tf.one_hot(fake_font_cats, depth=len(all_fonts))
    fake_font_loss = categorical_cross_entropy(fake_font_oh, fake_font_output)

    return real_class_loss + real_font_loss, fake_class_loss + fake_font_loss

@tf.function
def train_step(images, image_classes, image_fonts, batch_size, latent_size):
    noise = tf.random.normal([batch_size, latent_size])
    noise_classes_cat = tf.random.uniform([batch_size], minval=1, maxval=len(data.CharacterClass), dtype=tf.dtypes.int32)
    noise_classes_oh = tf.one_hot(noise_classes_cat, depth=len(data.CharacterClass))
    noise_fonts_cat = tf.random.uniform([batch_size], minval=1, maxval=len(all_fonts), dtype=tf.dtypes.int32)
    noise_fonts_oh = tf.one_hot(noise_fonts_cat, depth=len(all_fonts))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, noise_classes_oh, noise_fonts_oh], training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        #$gen_loss = generator_loss(fake_output)
        disc_real_loss, disc_fake_loss = discriminator_loss(real_output, image_classes, image_fonts, fake_output)
        
        gen_loss = 1.0 / disc_fake_loss
        disc_loss = disc_real_loss + disc_fake_loss

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

    fname = os.path.join(checkpoint_dir, f'image_epoch_{epoch:04d}.png')
    plt.savefig(fname)
    plt.close()

def train(dataset, epochs, latent_size):
    for epoch in range(epochs):
        start = time.time()

        for metadata_batch, image_batch in dataset():                        
            image_classes = tf.one_hot(metadata_batch['class_code'], depth=len(data.CharacterClass))
            image_fonts = tf.one_hot(metadata_batch['font_code'], depth=len(all_fonts))

            gl, dl = train_step(image_batch, image_classes, image_fonts, image_batch.shape[0], latent_size)            

        # Save the model every epoch
        manager.save()

        print (f'Time for epoch {epoch+start_epoch}: {time.time()-start}s, genloss: {gl}, discloss: {dl}')

        generate_and_save_images(generator,
                                 0,
                                 epoch+start_epoch,
                                 [seed, seed_classes, seed_fonts])


if __name__ == "__main__":
    num_examples_to_generate = 16
    latent_size = 100
    batch_size = 20
    num_epochs = 1000
    start_epoch = 1342

    random_seed = 54321
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    
    df = data.load()       
    font_lut =  data.font_lut(df['font'])
    all_fonts = font_lut.values()
    
    # We will reuse this seed overtime (so it's easier)    
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, latent_size])
    seed_classes = tf.random.uniform([num_examples_to_generate], minval=1, maxval=len(data.CharacterClass), dtype=tf.dtypes.int32)
    seed_classes = tf.one_hot(seed_classes, depth=len(data.CharacterClass))
    seed_fonts = tf.random.uniform([num_examples_to_generate], minval=1, maxval=len(all_fonts), dtype=tf.dtypes.int32)
    seed_fonts = tf.one_hot(seed_fonts, depth=len(all_fonts))

    generator, discriminator = model.build_class_gan(latent_size=latent_size, num_fonts=len(all_fonts))

    generator_optimizer = tf.keras.optimizers.Adam(1e-5)
    discriminator_optimizer = tf.keras.optimizers.Adam(4e-5)

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

    train(lambda: data.iterdata(batch_size=batch_size, random_seed=random_seed), epochs=num_epochs, latent_size=latent_size)