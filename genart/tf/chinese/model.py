import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU, Dropout, Input, Activation, ZeroPadding2D, Concatenate, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


def build_gan(latent_size=100):

    generator = Sequential([
        Dense(8*8*256, use_bias=False, input_shape=(latent_size,)),
        BatchNormalization(),
        LeakyReLU(),

        Reshape((8,8,256)),
        
        Conv2DTranspose(256, (5,5), strides=(1,1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        
        Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
    ])

    discriminator = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1]),
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(64, (5, 5), strides=(2, 2), padding='same'), # input 64x64
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'), # input 32x32
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(256, (5, 5), strides=(2, 2), padding='same'), # input 16x16
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),
        Dense(1)
    ])

    return generator, discriminator

def build_class_gan(latent_size=100):
    
    latent_input = Input(shape=(latent_size,))
    class_input = Input(shape=(1,))

    layer_1_input = Concatenate()([latent_input, class_input])

    layer_1 =  Sequential([
        Dense(7*7*256, use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((7,7,256))
    ])

    layer_1_output = layer_1(layer_1_input)
    layer_2_input = Concatenate()([layer_1_output, class_input])

    layer_2 = Sequential([
        Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
    ])

    layer_2_output = layer_2(layer_2_input)
    layer_2_input = Concatenate()([layer_2_output, class_input])

    layer_3 = Sequential([
        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU()
    ])

    layer_3_output = layer_3(layer_3_input)
    layer_4_input = Concatenate()([layer_3_output, class_input])

    layer_4 = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
    generator_output = layer_4(layer_4_input)

    generator = Model([latent_input, class_input], generator_output)

    discriminator = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),
        Dense(3, activation='sigmoid')
    ])

    return generator, discriminator

if __name__ == "__main__":
    seed = tf.random.normal((1,100))
    gen, disc = build_gan(100)
    gpred = gen.predict(seed)

    dpred = disc.predict(gpred)
    print(gpred.shape, dpred.shape)