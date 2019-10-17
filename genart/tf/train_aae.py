import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

import skimage.io as skio
import imageio

from genart.tf.model import GenartAutoencoder, GenartAaeDiscriminator
import genart.gen_images as gi

mse = tf.keras.losses.mean_squared_error
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generate_and_save_images(model, epoch, batch, img_input, img_test, latent_input, out_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).    
    _,img_output = model(img_input, training=False)
    loss = tf.math.reduce_mean(mse(img_output, img_test))
    print(f"epoch {epoch}, batch {batch}, loss {loss}")

    latent_output = model.decoder(latent_input, training=False)
    loss = generator_loss(latent_output)

    fig = plt.figure(figsize=(8,5))

    for i in range(img_output.shape[0]):
        plt.subplot(5, 8, (2*i)+1)
        plt.imshow(np.clip(img_output[i],0,1))
        plt.axis('off')

        plt.subplot(5, 8, (2*i)+2)
        plt.imshow(img_test[i])
        plt.axis('off')

    for i in range(8):
        plt.subplot(5, 8, 2*img_output.shape[0]+i+1)
        plt.imshow(np.clip(latent_output[i],0,1))
        plt.axis('off')

    plt.savefig(f'{out_dir}/image_{epoch:04d}_{batch:04d}.png')
    plt.close(fig)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, images_test, noise, 
               autoencoder, discriminator,
               autoencoder_optimizer, discriminator_optimizer):       

    with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:        
        input_encoded, output_decoded = autoencoder(images, training=True)        
        
        real_output = discriminator(input_encoded, training=True)
        fake_output = discriminator(noise, training=True)

        im_loss = mse(output_decoded, images_test)
        gen_loss = generator_loss(fake_output)
        
        ae_loss =  im_loss + gen_loss
        disc_loss = discriminator_loss(real_output, fake_output)


    ae_gradients = ae_tape.gradient(ae_loss, autoencoder.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    autoencoder_optimizer.apply_gradients(zip(ae_gradients, autoencoder.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def train(autoencoder, discriminator,
          autoencoder_optimizer, discriminator_optimizer,
          manager, train_params, gi_params, out_dir):

    batch_size = train_params['batch_size']
    n_epochs = train_params['n_epochs']
    epoch_size = train_params['epoch_size']    
    seed_size = train_params['seed_size']
    
    latent_size = autoencoder.latent_size

    img_seed = gi.render_shape_sets(seed_size, **gi_params)[0]
    latent_seed = tf.random.normal([seed_size, latent_size])

    for epoch in range(n_epochs):
        start = time.time()

        print("generating images")
        imgs = gi.render_shape_sets(epoch_size, **gi_params)[0]
        print("training", imgs.shape)

        for batch in range(0, epoch_size, batch_size):
            batch_imgs = imgs[batch:batch+batch_size]
            
            noise = tf.random.normal([batch_size, latent_size])

            train_step(batch_imgs, batch_imgs, noise,
                       autoencoder, discriminator,
                       autoencoder_optimizer, discriminator_optimizer)

            if batch % 500 == 0:
                generate_and_save_images(autoencoder,
                                         epoch,
                                         batch,
                                         img_seed,
                                         img_seed,
                                         latent_seed,
                                         out_dir)

        # Save the model every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            manager.save()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(autoencoder,
                             epoch,
                             batch,
                             img_seed,
                             img_seed,
                             latent_seed,
                             out_dir)
    
    manager.save()

def vis(autoencoder, discriminator, gi_params):
    n_imgs = 4
    imgs = gi.render_shape_sets(n_imgs, **gi_params)[0]

    latent = autoencoder.encoder(imgs, training=False)

    alpha = np.array([-np.cos(np.linspace(0,np.pi,30))*0.5+.5]).T
    
    gil = []
    for i in range(n_imgs):
        i0 = i
        i1 = i+1 if i < n_imgs-1 else 0

        latent_interp = latent[i0:i0+1] * (1.0-alpha) + latent[i1:i1+1] * alpha
        gen_imgs = autoencoder.decoder(latent_interp, training=False)
        gil.append(gen_imgs)

    gen_imgs = tf.concat(gil, axis=0).numpy()
    gen_imgs = (gen_imgs * 255).astype(np.uint8)
    imageio.mimsave('test/interp.gif', 
                    gen_imgs, 
                    fps=30)

    img_fnames = [ 'octopus7.png', 'octopus5.png', 'cat.jpg' ]
    n_imgs = len(img_fnames)

    for i,img_fname in enumerate(img_fnames):
        img = skio.imread(img_fname)[:,:,:3]
        img = np.array([img]).astype(np.float32) / 255.0        

        _,out_img = autoencoder(img)

        imageio.imsave(f'test/untrained_{i:04d}.png', np.clip(out_img[0],0,1))

    img_fnames = [ 'octopus7.png', 'octopus5.png', 'cat.jpg' ]
    imgs = np.zeros([n_imgs, 256, 256,3], dtype=float)
    for i,img_fname in enumerate(img_fnames):
        imgs[i] = skio.imread(img_fname)[:,:,:3].astype(np.float32)/255.0

    latent = autoencoder.encoder(imgs, training=False)

    
    gil = []
    for i in range(n_imgs):
        i0 = i
        i1 = i+1 if i < n_imgs-1 else 0

        latent_interp = latent[i0:i0+1] * (1.0-alpha) + latent[i1:i1+1] * alpha
        gen_imgs = autoencoder.decoder(latent_interp, training=False)
        gil.append(gen_imgs)

    gen_imgs = tf.concat(gil, axis=0).numpy()
    gen_imgs = (gen_imgs * 255).astype(np.uint8)
    imageio.mimsave('test/untrained_interp.gif', 
                    gen_imgs, 
                    fps=30)


def main():
    latent_size = 3000
    img_shape = (256,256,3)

    train_params = dict(
        batch_size = 20,
        epoch_size = 1000,
        n_epochs = 500,
        seed_size = 16
    )

    gi_params = dict( 
        shape = None,
        img_sizes = (img_shape,), 
        n_min = 1,
        n_max = 20
    )

    out_dir = 'out_aae'

    autoencoder = GenartAutoencoder(img_shape, latent_size)
    discriminator = GenartAaeDiscriminator(latent_size)

    autoencoder_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    
    checkpoint_dir = os.path.join(out_dir, 'tf_ckpts')

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
    
    cmd = sys.argv[1] 
    if cmd == 'train':
        train(autoencoder, discriminator,
              autoencoder_optimizer, discriminator_optimizer,
              manager, train_params, gi_params, out_dir)
    elif cmd == 'vis':
        vis(autoencoder, discriminator, gi_params)
    

if __name__ == "__main__": main()