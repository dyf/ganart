import tensorflow as tf
import os
import numpy as np
import genart.tf.chinese.data as mdata
import genart.tf.chinese.model as mmodel
import itertools
import matplotlib.pyplot as plt


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, manager, test_data, model, out_dir):
        super().__init__()
        self.manager = manager
        self.test_data = test_data
        self.model = model
        self.out_dir = out_dir

    def on_epoch_end(self, epoch, logs):
        generate_and_save_images(self.model, epoch, self.test_data, self.out_dir)
        self.manager.save()

def generate_and_save_images(model, epoch, test_data, checkpoint_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_data[0], training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(4):
        plt.subplot(4, 4, i+1)
        plt.imshow(test_data[0][i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    for i in range(4):
        plt.subplot(4, 4, i+1+4)
        plt.imshow(test_data[1][i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    for i in range(4):
        plt.subplot(4, 4, i+1+8)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fname = os.path.join(checkpoint_dir, f'image_epoch_{epoch:04d}.png')
    
    plt.savefig(fname)
    plt.close()



def train():   
    latent_size = 100
    batch_size = 10
    random_seed = 12345
    train_test_split = 0.9
    num_epochs = 100

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    
    train_gen = lambda: mdata.iterdata_variants(random_seed=random_seed, split_range=[0,train_test_split])
    test_gen = lambda: mdata.iterdata_variants(random_seed=random_seed, split_range=[train_test_split,1.0])

    train_data = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.float32), output_shapes=((128,128,1), (128,128,1)))
    test_data = tf.data.Dataset.from_generator(test_gen, output_types=(tf.float32, tf.float32), output_shapes=((128,128,1), (128,128,1)))
    vis_data = tf.data.Dataset.from_generator(test_gen, output_types=(tf.float32, tf.float32), output_shapes=((128,128,1), (128,128,1)))

    vis_data = next(vis_data.batch(4).as_numpy_iterator())

    model = mmodel.build_trad2simp(latent_size)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    
    checkpoint_dir = './data/chinese_p2p_output/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)

    cb = CustomCallback(manager, vis_data, model, checkpoint_dir)

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    model.compile(optimizer, loss='mse', metrics=["accuracy"])

    history = model.fit(train_data.batch(batch_size), 
                        validation_data=test_data.batch(batch_size),
                        epochs=num_epochs,
                        callbacks=[cb])

if __name__ == "__main__": train()