import os, time
import matplotlib.pyplot as plt
import numpy as np
import random

import tensorflow as tf
import genart.tf.charts.data as mdata
import genart.tf.charts.model as mmodel
import genart.gen_charts as gc

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

checkpoint_dir = os.path.normpath(f'./data/charts_output/')
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
level = 0
manager = None

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss_l2(fake_output):
    return tf.reduce_mean(tf.nn.l2_loss(fake_output - tf.ones_like(fake_output))) 

def discriminator_loss_l2(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.l2_loss(real_output - tf.ones_like(real_output))) 
    fake_loss =  tf.reduce_mean(tf.nn.l2_loss(fake_output - tf.zeros_like(fake_output))) 
    total_loss = real_loss + fake_loss
    return total_loss    

def get_train_step_fn():
    @tf.function
    def train_step(images, labels, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, latent_size):
        noise = tf.random.normal([batch_size, latent_size])
        noise_labels = tf.one_hot(tf.random.uniform([batch_size], minval=0, maxval=len(gc.Chart)-1, dtype=tf.dtypes.int32), depth=len(gc.Chart))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator([noise_labels, noise], training=True)

            real_output = discriminator([labels, images], training=True)
            fake_output = discriminator([noise_labels, generated_images], training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))   

        return gen_loss, disc_loss
    return train_step

def generate_and_save_images(model, batch, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(12,12))

    for i in range(predictions.shape[0]):
        plt.subplot(3, 3, i+1)
        plt.imshow(tf.cast(predictions[i, :, :, :] * 127.5 + 127.5, tf.uint8))
        plt.axis('off')

    fname = os.path.join(checkpoint_dir, f'image_level_{level}_epoch_{epoch:03d}_batch_{batch:07d}.png')
    plt.savefig(fname)
    plt.close()

def set_alpha(models, alpha):
    for model in models:
        for layer in model.layers:
            if isinstance(layer, mmodel.WeightedSum):
                tf.keras.backend.set_value(layer.alpha, alpha)

def train(gen, disc, gen_opt, disc_opt, data, epochs, latent_size, fade_batches, test_seed, start_epoch=0):
    train_step = get_train_step_fn()

    tbi = 0
    for epoch in range(epochs):
        start = time.time()

        bi = 0
        for labels_batch, image_batch in data():
            alpha = min(max(tbi / fade_batches, 0), 1.0)

            set_alpha([gen,disc], alpha)
            
            gl, dl = train_step(image_batch, labels_batch, 
                                gen, disc, gen_opt, disc_opt,                                 
                                image_batch.shape[0], latent_size)

            if bi % 1000 == 0:
                print(f'epoch({epoch}) batch({bi}) genloss({gl}) discloss({dl})')


                generate_and_save_images(gen,
                                         bi,
                                         epoch + start_epoch,
                                         test_seed)

            if bi % 3000 == 0:
                manager.save()
               

            bi+=1
            tbi+=1

        

        # Save the model every epoch
        manager.save()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(gen,
                             0,
                             epochs,
                             test_seed)

def datagen(batch_size, random_seed, resolution):
    def f():
        return mdata.iterdata(batch_size=batch_size, random_seed=random_seed, resolution=resolution)
    return f
    

def main():
    global manager, level

    num_examples_to_generate = 8
    latent_size = 256
    batch_size = 8
    num_epochs = 20
    start_epoch = 0
    layer_sizes = [128,128,128,128,128,128,128]
    fade_batches = 6000
    level = 0

    random_seed = 123456
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed_images = tf.random.normal([num_examples_to_generate, latent_size])
    seed_labels =  tf.one_hot(np.arange(8).T, depth=8)
    seed = [seed_labels, seed_images]

    
    builder = mmodel.PCGANBuilder(latent_size=latent_size, num_labels=8)

    generators, discriminators = builder.build_model(layer_sizes=layer_sizes)
    
    datagens = [ datagen(
        batch_size=batch_size,
        random_seed=random_seed, 
        resolution=512/pow(2, f)
    ) for f in range(len(layer_sizes)-1,-1,-1) ]

    generator_optimizers = [ tf.keras.optimizers.Adam(1e-4) for _ in range(len(generators)) ]
    discriminator_optimizers = [ tf.keras.optimizers.Adam(1e-4) for _ in range(len(discriminators)) ]

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_data = {}
    for i,(g,d,go,do) in enumerate(zip(generators, discriminators, generator_optimizers, discriminator_optimizers)):
        checkpoint_data[f'generator_{i}'] = g
        checkpoint_data[f'discriminator_{i}'] = d
        checkpoint_data[f'generator_optimizer_{i}'] = go
        checkpoint_data[f'discriminator_optimizer_{i}'] = do

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(**checkpoint_data)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)                                    
    
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for (g,d,go,do,data) in zip(generators, discriminators, generator_optimizers, discriminator_optimizers, datagens):
        train(g, d, go, do, data, epochs=num_epochs, latent_size=latent_size, fade_batches=fade_batches, start_epoch=start_epoch, test_seed=seed)
        level += 1

if __name__ == "__main__": main()