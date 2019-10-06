import numpy as np
import tensorflow as tf


from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dropout, UpSampling2D, Reshape, MaxPooling2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

class GenartAutoencoder(Model):
    def __init__(self, image_shape, latent_size, rlslope=0.2, dropout=0.25):
        super().__init__()
    
        self.encoder = Sequential([
            Conv2D(16, kernel_size=3, activation='relu'),
            MaxPooling2D((2,2), padding='same'),
            Conv2D(32, kernel_size=3, activation='relu'),
            MaxPooling2D((2,2), padding='same'),
            Conv2D(64, kernel_size=3, activation='relu'),
            MaxPooling2D((2,2), padding='same'),
            Flatten()            
        ])

        self.latent_layer = Dense(latent_size)

        dec_input_shape = (image_shape[0] // 4, image_shape[1] // 4, 128)

        self.decoder = Sequential([
            InputLayer(input_shape=(latent_size,)),
            Dense(np.prod(dec_input_shape), activation='relu'),
            Reshape(target_shape=dec_input_shape),
            Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            UpSampling2D((2,2)),
            Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            UpSampling2D((2,2)),            
            Conv2D(3, kernel_size=3, padding='same', activation='tanh'),            
        ])

        

    def call(self, x):
        x = self.encoder(x)
        x = self.latent_layer(x)
        x = self.decoder(x)
        return x

class GenartDiscriminator(Model):
    def __init__(self):
        super().__init__()
        
        self.net = Sequential([
            Conv2D(16, kernel_size=3, activation='relu'),
            Dropout(0.3),
            MaxPooling2D((2,2), padding='same'),
            BatchNormalization(),
            Conv2D(32, kernel_size=3, activation='relu'),
            Dropout(0.3),
            MaxPooling2D((2,2), padding='same'),            
            BatchNormalization(),
            Conv2D(64, kernel_size=3, activation='relu'),
            MaxPooling2D((2,2), padding='same'),
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