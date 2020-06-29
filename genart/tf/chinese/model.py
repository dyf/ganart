import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU, Dropout, Input, Activation, ZeroPadding2D, Concatenate, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

def build_trad2simp(latent_size=100):
    def enc_layer(N, k=(5,5), s=(2,2)):
        return Sequential([
            Conv2D(N, k, strides=s, padding='same'),
            LeakyReLU(),
            Dropout(0.3)
        ])

    def dec_layer(N, k=(5,5), s=(2,2)):
        return Sequential([
            Conv2DTranspose(N, k, strides=s, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU()
        ])

    input = Input(shape=(128,128,1))

    e1 = enc_layer(64) # 64
    e2 = enc_layer(64) # 32
    e3 = enc_layer(128) # 16
    e4 = enc_layer(256) # 8

    latent_layer = Sequential([
        Flatten(),
        Dense(latent_size)
    ])

    dec_input_layer = Sequential([
        Dense(8*8*256),
        Reshape((8,8,256)),
        Conv2DTranspose(256, (5,5), strides=(1,1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),
    ])
    
    d1 = dec_layer(128) # 16
    d2 = dec_layer(64) # 32
    d3 = dec_layer(64) # 64
    out_layer = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh') # 128

    e1o = e1(input)
    e2o = e2(e1o)
    e3o = e3(e2o)
    e4o = e4(e3o)
    
    lo = latent_layer(e4o)
    di = dec_input_layer(lo)

    d1o = d1(Concatenate()([e4o, di]))
    d2o = d2(Concatenate()([e3o, d1o]))
    d3o = d3(Concatenate()([e2o, d2o]))
    output = out_layer(d3o)

    return Model(input, output)






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
    
    seed = tf.random.normal((10,128,128,1))
    model = build_trad2simp()
    print(model.summary())
    output = model.predict(seed)
    print(output.shape)