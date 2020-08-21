import tensorflow as tf
import genart.tf.charts.bargan.model as mmodel
import genart.tf.charts.bargan.data as mdata
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_and_save_images(epoch, batch):
    pred_y = len_model.predict(test_image)
    pred_xmin, pred_xmax, pred_yscale = scale_model.predict(test_image)
    
    fig = plt.figure(figsize=(16,8))

    ori_lookup = {
        0: 'horizontal',
        1: 'vertical'
    }
    
    for i in range(min(pred_y.shape[0],9)):
        plt.subplot(3,6, 2*i+1)        
        
        x = np.linspace(pred_xmin[i,0], pred_xmax[i,0], num=pred_y[i].shape[0])
        y = pred_y[i] * pred_yscale[i,0]
        #x = np.linspace(0, 1, num=pred_y[i].shape[0])
        #y = pred_y[i]
        size = (x[1] - x[0]) * 1.0        
        
        ori = ori_lookup[int(test_ori[i]>0.5)]

        if ori == 'horizontal':
            plt.barh(x, y, height=size, color=test_color[i])
        else:
            plt.bar(x, y, width=size, color=test_color[i])

        plt.title('predicted')
        
        plt.subplot(3,6, 2*i+2)
        plt.imshow(tf.image.convert_image_dtype(test_image[i], tf.uint8, saturate=True))
        plt.title('target')
        plt.axis('off')

    fname = os.path.join(output_dir, f'image_epoch_{epoch:03d}_batch_{batch:07d}.png')
    plt.savefig(fname)
    plt.close()

def train(epochs, data, len_model, scale_model):
    for ei in range(epochs):
        for bi,(x,y,img,ori,color) in enumerate(data):
            xmin, xmax = tf.reduce_min(x, axis=1), tf.reduce_max(x, axis=1)
            y_scale = tf.reduce_max(y, axis=1)
                
            len_loss = len_model.train_on_batch(img, y/y_scale[:, tf.newaxis])            
            scale_loss = scale_model.train_on_batch(img, [xmin, xmax, y_scale])
            if bi % 500 == 0:
                print(f'{ei} {bi} {len_loss} {scale_loss}')
                generate_and_save_images(ei, bi)

            if bi % 1000 == 0:
                manager.save()
            
        manager.save()
        

if __name__ == "__main__":   
    batch_size = 10
    num_points = 10
    image_shape = (256,256,3)
    layer_filters = [ 64, 64, 128, 256, 512 ]
    epochs = 50
    output_dir = 'data/charts_bardata_output'

    ds_output_shapes = ( (batch_size, num_points), (batch_size, num_points), (batch_size, image_shape[0], image_shape[1], image_shape[2]), (batch_size, 1),  (batch_size, 3) )
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


    len_model, scale_model = mmodel.chartdata(num_points, layer_filters, image_shape)
        
    len_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.MeanSquaredError())
    scale_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=[
        tf.keras.losses.MeanSquaredError(),
        tf.keras.losses.MeanSquaredError(),
        tf.keras.losses.MeanSquaredError()
    ])

    print(len_model.summary())
    print(scale_model.summary())
    

    checkpoint_prefix = os.path.join(output_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(len_model=len_model, scale_model=scale_model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)                                    

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train(epochs, train_dataset, len_model, scale_model)
    