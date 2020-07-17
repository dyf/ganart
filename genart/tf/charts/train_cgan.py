import os, time
import matplotlib.pyplot as plt
import numpy as np
import random

import tensorflow as tf
import genart.tf.charts.data as data
import genart.tf.charts.model as model
import genart.gen_charts as gc


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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

@tf.function
def train_step(images, labels, batch_size, latent_size):
    noise = tf.random.normal([batch_size, latent_size])
    noise_labels = tf.random.uniform([batch_size,1], minval=0, maxval=len(gc.Chart)-1, dtype=tf.dtypes.int32)    

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

def generate_and_save_images(model, batch, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(12,12))

    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i+1)
        plt.imshow(tf.cast(predictions[i, :, :, :] * 127.5 + 127.5, tf.uint8))
        plt.axis('off')

    fname = os.path.join(checkpoint_dir, f'image_epoch_{epoch:03d}_batch_{batch:07d}.png')
    plt.savefig(fname)
    plt.close()

def train(dataset, epochs, latent_size):
    for epoch in range(epochs):
        start = time.time()

        bi = 0
        for md, image_batch in dataset():            
            labels_batch = md.reshape((len(md),1)).astype(float)
            gl, dl = train_step(image_batch, labels_batch, image_batch.shape[0], latent_size)

            if bi % 1000 == 0:
                print(f'epoch({epoch}) batch({bi}) genloss({gl}) discloss({dl})')


                generate_and_save_images(generator,
                                         bi,
                                         epoch + start_epoch,
                                         seed)

            if bi % 3000 == 0:
                manager.save()
               

            bi+=1

        

        # Save the model every epoch
        manager.save()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                             0,
                             epochs,
                             seed)

def restore_weights(num_layers, checkpoint_prefix, generator, discriminator):    
    old_gen, old_disc = builder.build_model(num_layers=num_layers)
    checkpoint = tf.train.Checkpoint(generator=old_gen, discriminator=old_disc)
    
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    
    if manager.latest_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        status.expect_partial()

        copy_layer_weights(old_gen, generator)
        copy_layer_weights(old_disc, discriminator)
        
        return True    

def copy_layer_weights(source_model, target_model):
    source_layer_dict = dict([(layer.name, layer) for layer in source_model.layers])
    target_layer_dict = dict([(layer.name, layer) for layer in target_model.layers])
    print(source_model.summary())
    print(target_model.summary())

    for name, source_layer in source_layer_dict.items():
        target_layer = target_layer_dict.get(name, None)

        if target_layer is not None:
            target_layer.set_weights(source_layer.get_weights())

    
if __name__ == "__main__":
    

    num_examples_to_generate = 4
    latent_size = 256
    batch_size = 16
    num_epochs = 20
    start_epoch = 0
    num_layers = 1
    
    random_seed = None
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed_images = tf.random.normal([num_examples_to_generate, latent_size])
    seed_labels =  np.array([[0,1,2,0]]).T
    seed = [seed_labels, seed_images]

    builder = model.PCGANBuilder(latent_size=latent_size, num_labels=3)


    generator, discriminator = builder.build_model(num_layers=num_layers)
    
    generator_optimizer = tf.keras.optimizers.Adam(2e-5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

    checkpoint_dir = f'./data/charts_output/layers_{num_layers}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)                                    
    
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        lowres_checkpoint_dir = f'./data/charts_output/layers_{num_layers-1}/ckpt'
        success = restore_weights(num_layers-1, lowres_checkpoint_dir, generator, discriminator)
        if success:
            print("Restored from lower resolution.")
        else:
            print("Initializing from scratch.")

    downsample_factor = pow(2, builder.max_layers-num_layers)

    datagen = lambda: data.iterdata(
        batch_size=batch_size, 
        random_seed=random_seed, 
        resolution=512//downsample_factor
    )

    train(datagen, epochs=num_epochs, latent_size=latent_size)