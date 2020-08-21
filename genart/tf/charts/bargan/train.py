import tensorflow as tf
import genart.tf.charts.bargan.model as mmodel
import genart.tf.charts.bargan.data as mdata
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_and_save_images(epoch, batch):
    predictions = model.predict([test_x, test_y, test_ori, test_color])

    fig = plt.figure(figsize=(12,6))

    for i in range(min(predictions.shape[0],9)):
        plt.subplot(3,6, 2*i+2)
        plt.imshow(tf.image.convert_image_dtype(predictions[i], tf.uint8, saturate=True)),
        plt.title('predicted')
        plt.axis('off')
        plt.subplot(3,6, 2*i+1)
        plt.imshow(tf.image.convert_image_dtype(test_image[i], tf.uint8, saturate=True))
        plt.title('target')
        plt.axis('off')

    fname = os.path.join(output_dir, f'image_epoch_{epoch:03d}_batch_{batch:07d}.png')
    plt.savefig(fname)
    plt.close()

def train(epochs, data, model):
    for ei in range(epochs):
        for bi,(x,y,img,ori,color) in enumerate(data):            
            loss = model.train_on_batch([x,y,ori,color],img)
            if bi % 100 == 0:
                print(f'{ei} {bi} {loss:.5f}')
                generate_and_save_images(ei, bi)

            if bi % 1000 == 0:
                manager.save()
            
        manager.save()
        

if __name__ == "__main__":   
    batch_size = 10
    num_bins = 10
    image_shape = (256,256,3)
    layer_filters = [ 256, 256, 128, 128, 128 ]
    epochs = 50
    output_dir = 'data/charts_output'

    ds_output_shapes = ( (batch_size, num_bins), (batch_size, num_bins), (batch_size, image_shape[0], image_shape[1], image_shape[2]), (batch_size, 1),  (batch_size, 3) )
    ds_output_types = ( np.float32, np.float32, np.float32, np.uint8, np.float32 )
    train_dataset = tf.data.Dataset.from_generator(
        generator = lambda: mdata.iterdata(file_pattern=mdata.TRAIN_FILE_PATTERN, batch_size=batch_size),
        output_shapes = ds_output_shapes,
        output_types = ds_output_types
    )

    test_dataset = tf.data.Dataset.from_generator(
        generator = lambda: mdata.iterdata(file_pattern=mdata.TEST_FILE_PATTERN, batch_size=batch_size),
        output_shapes = ds_output_shapes,
        output_types = ds_output_types
    )

    test_x, test_y, test_image, test_ori, test_color = list(test_dataset.take(1).as_numpy_iterator())[0]            


    model = mmodel.barchart(num_bins, layer_filters, image_shape)
    
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    checkpoint_prefix = os.path.join(output_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model,optimizer=optimizer)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)                                    

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train(epochs, train_dataset, model)
    