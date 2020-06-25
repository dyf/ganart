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

def build_class_gan(latent_size=100, num_fonts=10):
    
    latent_input = Input(shape=(latent_size,))
    ts_input = Input(shape=(3,))
    font_input = Input(shape=(num_fonts,))

    layer_1_input = Concatenate()([latent_input, ts_input, font_input])

    layer_1 =  Sequential([
        Dense(8*8*256, use_bias=False),
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

    generator_output = layer_1(layer_1_input)
    
    generator = Model([latent_input, ts_input, font_input], generator_output)

    discriminator_input = Input(shape=(128,128,1))
    
    discriminator_conv = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(64, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Flatten()
    ])

    ts_classifier = Dense(3, activation='sigmoid')
    font_classifier = Dense(num_fonts, activation='sigmoid')

    dco = discriminator_conv(discriminator_input)

    discriminator = Model(discriminator_input, [ ts_classifier(dco), font_classifier(dco) ])

    return generator, discriminator

if __name__ == "__main__":
    batch_size = 2
    latent_size = 100
    num_fonts = 10
    
    seed = tf.random.normal((batch_size,latent_size))
    classes = tf.random.uniform((batch_size,), minval=0, maxval=3, dtype=tf.dtypes.int32)
    fonts = tf.random.uniform((batch_size,), minval=0, maxval=num_fonts+1, dtype=tf.dtypes.int32)

    gen, disc = build_class_gan(latent_size, num_fonts)
    gpred = gen.predict([seed, tf.one_hot(classes, depth=3), tf.one_hot(fonts, depth=num_fonts+1)])
    [dpred_ts, dpred_font] = disc.predict(gpred)
    print(gpred.shape, dpred_ts.shape, dpred_font.shape)
