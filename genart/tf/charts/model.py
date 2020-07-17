import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, ReLU, LeakyReLU, Dropout, Input, Activation, ZeroPadding2D, Concatenate, Flatten, Reshape, Conv2DTranspose, Embedding, UpSampling2D, AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

def gen_layer_bc(nf, k=(3,3), name=None):
    return Sequential([
        UpSampling2D(),
        Conv2D(nf, k, strides=(1,1), padding='same'),
        LeakyReLU(),
        BatchNormalization(),
        Conv2D(nf, k, strides=(1,1), padding='same'),
        LeakyReLU(),
        BatchNormalization(),
    ], name=name)

def gen_layer_tc(nf, k=(5,5), strides=(2,2), name=None):
    return Sequential([
        Conv2DTranspose(nf, k, strides=strides, padding='same'), 
        BatchNormalization(),
        LeakyReLU(),
    ], name=name)

def disc_layer_dpool(nf, name=None):
    return Sequential([
        Conv2D(nf, (3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(),
        Conv2D(nf, (3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(),
        AveragePooling2D()
    ], name=name)

def disc_layer_sc(nf, name=None):
    return Sequential([
        Conv2D(nf, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        Dropout(0.3),
    ], name=name)

class PCGANBuilder:
    def __init__(self, latent_size=256, num_labels=3, max_shape=(512,512), embedding_size=10):
        self.latent_size = latent_size
        self.num_labels = num_labels
        self.max_shape = max_shape
        self.embedding_size = embedding_size

        self.gen_layer_filters = [ 128, 128, 64, 64, 32, 32 ]        
        self.disc_layer_filters = self.gen_layer_filters[::-1]

        self.max_layers = len(self.gen_layer_filters)

    def build_model(self, num_layers=None):
        if num_layers is None:
            num_layers = self.max_layers

        return ( 
            self.build_generator(num_layers), 
            self.build_discriminator(num_layers)
        )
        
    def build_generator(self, num_layers):            
        factor = pow(2, self.max_layers)
        image_shape = ( self.max_shape[0]//factor, self.max_shape[1]//factor, 1 )
        
        labels_input = Input(shape=(1,), name='gen_labels_input')
        
        labels = Sequential([
            Embedding(self.num_labels, self.embedding_size),
            Dense(image_shape[0]*image_shape[1]),
            Reshape(image_shape)
        ], name='gen_labels_reshape')(labels_input)

        latent_input = Input(shape=(self.latent_size,), name='gen_latent_input')

        latent = Sequential([
            Dense(image_shape[0]*image_shape[1]*256),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((image_shape[0],image_shape[1],256))
        ], name='gen_latent_reshape')(latent_input)

        gen_input = Concatenate()([labels, latent])

        gen = gen_layer_tc(256, strides=(1,1), name='gen_conv_0')(gen_input)
        
        for i,nf in enumerate(self.gen_layer_filters[:num_layers]):
            #gen = gen_layer_tc(nf, name=f"gen_upconv_{i}")(gen)
            gen = gen_layer_bc(nf, name=f"gen_upconv_{i}")(gen)
            
        gen = Conv2D(3, (1,1), padding='same', activation='tanh', name='gen_to_rgb')(gen) # 512

        return Model([labels_input, latent_input], gen)

    def build_discriminator(self, num_layers):    
        factor = pow(2, self.max_layers - num_layers)
        image_shape = ( self.max_shape[0]//factor, self.max_shape[1]//factor, 3 )
        
        labels_input = Input(shape=(1,), name=f'disc_labels_input_level_{num_layers}')
        image_input = Input(shape=image_shape, name=f'disc_image_input_level_{num_layers}')

        labels = Sequential([
            Embedding(self.num_labels, self.embedding_size),
            Dense(image_shape[0]*image_shape[1]),
            Reshape((image_shape[0], image_shape[1],1))
        ], name=f'disc_labels_reshape_level_{num_layers}')(labels_input)        

        disc = Concatenate(name=f'disc_incat_level_{num_layers}')([labels, image_input])

        for i in range(num_layers):
            idx = num_layers - i - 1
            
            if i == 0:            
                name = f"disc_indownconv_level_{num_layers}"
            else:
                name = f"disc_downconv_{idx}"
                

            #disc = disc_layer_sc(self.disc_layer_filters[i], name=name)(disc)
            disc = disc_layer_dpool(self.disc_layer_filters[i], name=name)(disc)

        disc = Sequential([
            Conv2D(256, (3, 3), strides=(1, 1), padding='same'),
            LeakyReLU(),
            Flatten(),
            Dense(1)
        ], name='disc_out')(disc)

        return Model([labels_input, image_input], disc)


def build_gan(latent_size=100):

    generator = Sequential([
        Dense(8*8*256, use_bias=False, input_shape=(latent_size,)),
        BatchNormalization(),
        LeakyReLU(),

        Reshape((8,8,256)),
        
        Conv2DTranspose(256, (5,5), strides=(1,1), padding='same', use_bias=False), # 8
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', use_bias=False), # 16
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False), # 32
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False), # 64
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False), # 128
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False), # 256
        BatchNormalization(),
        LeakyReLU(),
        
        Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh') # 512
    ])

    discriminator = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[512, 512, 3]),
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(64, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),
        
        Conv2D(128, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(128, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(256, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(256, (5, 5), strides=(2, 2), padding='same'), 
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),
        Dense(1)
    ])

    return generator, discriminator

if __name__ == "__main__":
    pcgan = PCGAN()

    gen, disc = pcgan.build_model(num_layers=1)
    print(disc.summary())