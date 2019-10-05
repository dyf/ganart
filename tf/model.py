import numpy as np
import tensorflow as tf


from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Flatten, Dropout, UpSampling2D, Reshape
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

class GenartAutoencoder(Model):
    def __init__(self, image_shape, latent_size, rlslope=0.2, dropout=0.25):
        super().__init__()
    
        self.econv1 = Conv2D(16, kernel_size=3, strides=(2,2))
        self.elr1 = LeakyReLU(rlslope)
        self.ed1 = Dropout(dropout)

        self.econv2 = Conv2D(32, kernel_size=3, strides=(2,2))
        self.elr2 = LeakyReLU(rlslope)
        self.ed2 = Dropout(dropout)

        self.econv2 = Conv2D(64, kernel_size=3, strides=(2,2))
        self.elr2 = LeakyReLU(rlslope)
        self.ed2 = Dropout(dropout)
    
        self.econv3 = Conv2D(128, kernel_size=3, strides=(2,2))
        self.elr3 = LeakyReLU(rlslope)
        self.ed3 = Dropout(dropout)

        self.eflat = Flatten()    
        self.latent_layer = Dense(latent_size, activation='sigmoid')

        self.encoder = Sequential([
            self.econv1, self.elr1, self.ed1,
            self.econv2, self.elr2, self.ed2,
            self.econv3, self.elr3, self.ed3,
            self.eflat,            
            self.latent_layer
        ])

        dec_input_shape = (image_shape[0] // 4, image_shape[1] // 4, 128)

        self.dinput = Dense(np.prod(dec_input_shape))
        self.dresh = Reshape(dec_input_shape)

        self.dup1 = UpSampling2D((2,2))
        self.dconv1 = Conv2D(64, 3, padding='same')
        self.dlr1 = LeakyReLU(rlslope)

        self.dup2 = UpSampling2D((2,2))
        self.dconv2 = Conv2D(32, 3, padding='same')
        self.dlr2 = LeakyReLU(rlslope)
        
        self.dconv3 = Conv2D(3, 3, padding='same', activation='tanh')

        self.decoder = Sequential([
            self.dinput, self.dresh,
            self.dup1, self.dconv1, self.dlr1,
            self.dup2, self.dconv2, self.dlr2,
            self.dconv3, 
        ])

        

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x