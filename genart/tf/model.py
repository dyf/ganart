import numpy as np
import tensorflow as tf


from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dropout, UpSampling2D, Reshape, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

class GenartAutoencoder(Model):
    def __init__(self, image_shape, latent_size, rlslope=0.2, dropout=0.25):
        super().__init__()

        self.latent_size = latent_size
    
        self.encoder = Sequential([
           
            Conv2D(32, kernel_size=3, strides=(2,2)),
            LeakyReLU(0.2),

            Conv2D(64, kernel_size=3, strides=(2,2)),
            LeakyReLU(0.2),
            
            Conv2D(64, kernel_size=3, strides=(2,2)),
            LeakyReLU(0.2),
            
            Conv2D(128, kernel_size=3, strides=(2,2)),
            LeakyReLU(0.2),
            
            Flatten(),
            Dense(latent_size, activation='tanh')
        ])

        dec_input_shape = (image_shape[0] // 8, image_shape[1] // 8, 128)

        self.decoder = Sequential([
            InputLayer(input_shape=(latent_size,)),            
            Dense(np.prod(dec_input_shape), activation='relu'),
            Reshape(target_shape=dec_input_shape),

            
            Conv2DTranspose(64, kernel_size=3, strides=(2,2), padding='same'),
            LeakyReLU(0.2),
            
            Conv2DTranspose(64, kernel_size=3, strides=(2,2), padding='same'),
            LeakyReLU(0.2),
            
            Conv2DTranspose(32, kernel_size=3, strides=(2,2), padding='same'),
            LeakyReLU(0.2),
            
            Conv2D(3, kernel_size=7, padding='same', activation='sigmoid')
        ])        

    def call(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return x,y

class GenartAaeDiscriminator(Model):
    def __init__(self, latent_size):
        super().__init__()

        self.discriminator = Sequential([
            InputLayer(input_shape=(latent_size,)),
            Dense(latent_size, activation='relu'),
            Dense(latent_size, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        return self.discriminator(x)

class GenartAeGanDiscriminator(Model):
    def __init__(self, ae):
        super().__init__()

        self.ae = ae
        self.discriminator = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.ae.encoder(x)
        x = self.discriminator(x)
        return x

class GenartAeGanGenerator(Model):
    def __init__(self, ae):
        super().__init__()

        self.ae = ae
    
    def call(self, x):
        x = self.ae.decoder(x)
        return x

class GenartDiscriminator(Model):
    def __init__(self):
        super().__init__()
        
        self.net = Sequential([
            Conv2D(16, kernel_size=3, activation='relu'),            
            MaxPooling2D((2,2), padding='same'),
            Dropout(0.3),

            BatchNormalization(),
            Conv2D(32, kernel_size=3, activation='relu'),            
            MaxPooling2D((2,2), padding='same'),         
            Dropout(0.3),   
            
            BatchNormalization(),
            Conv2D(64, kernel_size=3, activation='relu'),            
            MaxPooling2D((2,2), padding='same'),
            Dropout(0.3),
            
            Flatten(),
            Dense(1)
        ])

    def call(self, x):
        return self.net(x)

class GenartGenerator(Model):
    def __init__(self, image_shape, latent_size):
        super().__init__()
        
        dec_input_shape = (image_shape[0] // 4, image_shape[1] // 4, 128)

        self.net = Sequential([
            InputLayer(input_shape=(latent_size,)),
            Dense(np.prod(dec_input_shape), use_bias=False, activation='relu'),            
            Reshape(target_shape=dec_input_shape),
            BatchNormalization(),
            Conv2D(64, kernel_size=3, padding='same', activation='relu'),            
            UpSampling2D((2,2)),
            BatchNormalization(),
            Conv2D(32, kernel_size=3, padding='same', activation='relu'),            
            UpSampling2D((2,2)),            
            BatchNormalization(),
            Conv2D(3, kernel_size=3, padding='same', activation='sigmoid'),            
        ])
    
    def call(self, x):
        return self.net(x)