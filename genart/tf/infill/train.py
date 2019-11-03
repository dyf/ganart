import tensorflow as tf
from genart.tf.infill.model import UNet, Discriminator
from genart.gen_photos import IndexedImageLoader
import time
import matplotlib.pyplot as plt
import os

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

def blanked_im(im):
    im = im.numpy()
    im[:,64:192,64:192,:] = 0.0
    return tf.convert_to_tensor(im)

def fit(train_ds, epochs, batch_size, test_ds):
    example_target = next(test_ds.iter_patch((256,256),3))
    example_input = blanked_im(example_target)

    for epoch in range(epochs):
        start = time.time()

        # Train        
        bi = 0
        for im in test_ds.iter_patch((256,256), batch_size):
            batch_target = im
            batch_input = blanked_im(im)

            train_step(batch_input, batch_target)
            if bi % 100 == 0:
                print(f'trained {bi*batch_size} images')
                generate_images(generator, example_input, example_target, epoch, bi)
            bi += 1

        # Test on the same image so that the progress of the model can be 
        # easily seen.
        
        

        manager.save()

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))

def generate_images(model, test_input, tar, epoch, image_num):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(10,10))
    
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    nrows = test_input.shape[0]
    for ri in range(nrows):
        display_list = [test_input[ri], tar[ri], prediction[ri]]
        for ci in range(3):
            plt.subplot(nrows, 3, ri*3 + ci+1)
            if ri == 0:
                plt.title(title[ci])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[ci] * 0.5 + 0.5)
            plt.axis('off')
    plt.savefig(f'{out_dir}/test_{epoch:04d}_{image_num:05d}.png')
    plt.close()

LAMBDA = 100
EPOCHS = 100
BATCH_SIZE = 10

data = IndexedImageLoader('images/large/img-large-{index}.jpg', expected_shape=(800,800))
generator = UNet()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

out_dir ='./out_infill'
checkpoint_dir = './out_infill/checkpoints'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)
checkpoint.restore(manager.latest_checkpoint)

fit(data, EPOCHS, BATCH_SIZE, data)

sm, lg = data[:3]
out = model(sm)
print(sm.shape)
print(out.shape)

