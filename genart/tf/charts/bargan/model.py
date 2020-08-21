import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import numpy as np

def up_block(nf, name=None):
    return tfk.Sequential([        
        #tfkl.Conv2DTranspose(nf, kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False),
        #tfkl.LeakyReLU(),
        #tfkl.BatchNormalization(),
        tfkl.UpSampling2D(),
        tfkl.Conv2D(nf, kernel_size=(5,5), strides=(1,1), padding='same'),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.BatchNormalization(),
        tfkl.Conv2D(nf, kernel_size=(5,5), strides=(1,1), padding='same'),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.BatchNormalization(),
    ], name=name)

def down_block(nf, k=3, name=None):
    return tfk.Sequential([        
        tfkl.Conv2D(nf, 
                    kernel_size=(k,k),
                    strides=(1,1),
                    padding='same'),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.BatchNormalization(),
        tfkl.Conv2D(nf, 
                    kernel_size=(k,k),
                    strides=(2,2),
                    padding='same'),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.BatchNormalization()
    ], name=name)

def chartdata(num_points, layer_filters, image_shape):
    image_input = tfkl.Input(shape=image_shape)

    x = image_input
    for li,lf in enumerate(layer_filters):
        k = 5 if li == 0 else 3
        x = down_block(lf,k)(x)
    
    bar_lengths = tfk.Sequential([
        tfkl.Flatten(),
        tfkl.Dense(512),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.BatchNormalization(),
        tfkl.Dense(num_points)
    ])(x)

    range_out = tfk.Sequential([
        tfkl.Flatten(),
        tfkl.Dense(512),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.BatchNormalization()
    ])(x)

    x_min = tfkl.Dense(1)(range_out)
    x_max = tfkl.Dense(1)(range_out)
    y_scale = tfkl.Dense(1)(range_out)

    return tfk.Model(image_input, bar_lengths), tfk.Model(image_input, [x_min, x_max, y_scale])


def barchart(num_bins, layer_filters, image_shape):
    x_input = tfkl.Input(shape=num_bins)
    y_input = tfkl.Input(shape=num_bins)
    ori_input = tfkl.Input(shape=1)
    color_input = tfkl.Input(shape=3)

    data_input = tfkl.Concatenate()([x_input, y_input])
    
    param_input = tfkl.Concatenate()([ori_input, color_input])

    f = pow(2, len(layer_filters))
    base_shape = (image_shape[0] // f, image_shape[1] // f, layer_filters[0])
    
    gen = tfk.Sequential([
        tfkl.Dense(np.prod(base_shape)),
        tfkl.LeakyReLU(alpha=0.2),
        tfkl.BatchNormalization(),        
        tfkl.Reshape(base_shape)
    ])(data_input)

    for li,lf in enumerate(layer_filters):
        f = pow(2, len(layer_filters) - li)
        base_shape = (image_shape[0] // f, image_shape[1] // f, 1)

        side_in = tfk.Sequential([
            tfkl.Dense(np.prod(base_shape)),
            tfkl.LeakyReLU(alpha=0.2),
            tfkl.BatchNormalization(),        
            tfkl.Reshape(base_shape)
        ])(param_input)

        layer_in = tfkl.Concatenate()([gen, side_in])
        gen = up_block(lf)(layer_in)
    
    gen = tfkl.Conv2D(3, kernel_size=(5,5), padding='same', activation='sigmoid')(gen)

    return tfk.Model([x_input, y_input, ori_input, color_input], gen)

    

def discriminator():
    pass

if __name__ == "__main__":
    gen = generator(50, [40, 30, 20, 10], [256,256,3])
    print(gen.summary())
